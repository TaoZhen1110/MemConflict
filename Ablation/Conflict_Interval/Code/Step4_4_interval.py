import argparse
import copy
import json
import jsonlines
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
from llm_request import llm_request, calculate_cumulative_cost
import logging

THIS_FILE = Path(__file__).resolve()
LOCAL_DIR = THIS_FILE.parent


def find_project_dir(start_path: Path) -> Path:
    for candidate in [start_path.parent] + list(start_path.parents):
        if (candidate / "Code").exists() and (candidate / "Prompt").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate project root from {start_path}")


PROJECT_DIR = find_project_dir(THIS_FILE)
CODE_DIR = PROJECT_DIR / "Code"
DATA_DIR = PROJECT_DIR / "Ablation" / "Conflict_Interval" / "Data"

if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

logger = logging.getLogger(__name__)

load_dotenv()


DEFAULT_SYSTEM_PROMPT = """You rewrite template questions for memory-conflict evaluation.
Do not change factual meaning.
Use only provided visible context.
Return valid JSON only."""


def load_jsonl_items(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items.append(item)
    return items


def write_jsonl_items(path: str, items: List[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(out_path), "w") as writer:
        for item in items:
            writer.write(item)


def write_json_items(path: str, items: List[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def resolve_input_path(interval_mode: str, input_file: Optional[str] = None) -> str:
    if input_file:
        return input_file
    return str(DATA_DIR / f"Step4_3_{interval_mode}_interval.jsonl")


def resolve_output_paths(
    interval_mode: str,
    output_file: Optional[str] = None,
    output_json_file: Optional[str] = None,
) -> Tuple[str, str]:
    suffix = f"{interval_mode}_interval"
    resolved_output_file = output_file or str(DATA_DIR / f"Step4_4_{suffix}.jsonl")
    resolved_output_json = output_json_file or str(DATA_DIR / f"Step4_4_{suffix}.json")
    return resolved_output_file, resolved_output_json


def resolve_prompt_path(prompt_file: Optional[str] = None) -> str:
    if prompt_file:
        return prompt_file
    return str(PROJECT_DIR / "Prompt" / "Prompt4_4.txt")


def Load_Step4_4_Prompt(prompt_path: str) -> str:
    """Load Step 4.4 prompt text. Fallback to default prompt if file is missing."""
    try:
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                return content
        return DEFAULT_SYSTEM_PROMPT
    except Exception:
        return DEFAULT_SYSTEM_PROMPT


def role_rank(role: str) -> int:
    """
    Convert role text to rank:
    Point_A -> 1, Point_B -> 2, ...
    Non-point role returns -1.
    """
    if not isinstance(role, str):
        return -1
    role = role.strip()
    if not role.startswith("Point_") or len(role) < 7:
        return -1
    letter = role[-1].upper()
    if "A" <= letter <= "Z":
        return ord(letter) - ord("A") + 1
    return -1


def is_non_empty_after(update_item: Dict[str, Any]) -> bool:
    """Check whether one updated item has a meaningful After value."""
    if not isinstance(update_item, dict):
        return False
    if "After" not in update_item:
        return False
    after = update_item.get("After")
    return after not in [None, "", {}]


def collect_visible_context(full_session_chain: List[Dict[str, Any]], current_idx: int) -> List[Dict[str, Any]]:
    """Collect compact visible session context from 0..current_idx."""
    visible = []
    for i in range(current_idx + 1):
        session = full_session_chain[i]
        visible.append({
            "Session_ID": session.get("Session_ID"),
            "Date": session.get("Date"),
            "Session_Type": session.get("Session_Type"),
            "Updated_Attributes": copy.deepcopy(session.get("Updated_Attributes", [])),
            "Static_Conflict_Information": copy.deepcopy(session.get("Static_Conflict_Information", [])),
            "Conditional_Conflict_Information": copy.deepcopy(session.get("Conditional_Conflict_Information", []))
        })
    return visible


def place_question_fields_before_event_types(session_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reorder one session dict so question fields appear right before Event_Types.
    Keep all original fields and relative order for non-question fields.
    """
    question_keys = ["Question_Trigger_Types", "Session_Questions", "Session_Question_Count"]
    question_payload = {k: copy.deepcopy(session_item.get(k)) for k in question_keys}

    reordered = {}
    inserted = False

    for key, value in session_item.items():
        if key in question_keys:
            continue
        if key == "Event_Types" and not inserted:
            for qk in question_keys:
                reordered[qk] = question_payload[qk]
            inserted = True
        reordered[key] = value

    if not inserted:
        for qk in question_keys:
            reordered[qk] = question_payload[qk]

    return reordered


def prune_session_fields_for_step4_4(session_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove fields not needed in Step 4.4 output.
    """
    pruned = copy.deepcopy(session_item)
    pruned.pop("State_Before", None)
    pruned.pop("State_After", None)
    pruned.pop("Update_Reason", None)
    return pruned


def find_point_a_static_value(visible_context: List[Dict[str, Any]],
                              conflict_id: Optional[str],
                              target_field_path: Optional[str]) -> Optional[Any]:
    """Find static Point_A value in visible context for the same conflict."""
    if not conflict_id:
        return None

    for session in visible_context:
        for item in session.get("Static_Conflict_Information", []):
            if not isinstance(item, dict):
                continue
            if item.get("Conflict_ID") != conflict_id:
                continue
            if item.get("Role") != "Point_A":
                continue
            if target_field_path and item.get("Target_Field_Path") != target_field_path:
                continue
            return copy.deepcopy(item.get("Value"))
    return None


def find_previous_conditional_point(visible_context: List[Dict[str, Any]],
                                    conflict_id: Optional[str],
                                    current_rank: int) -> Optional[Dict[str, Any]]:
    """
    For Point_B/C/D..., find previous point in same conflict chain.
    Example: current Point_C -> find Point_B.
    """
    if not conflict_id or current_rank <= 1:
        return None

    target_rank = current_rank - 1
    candidate = None

    for session in visible_context:
        for item in session.get("Conditional_Conflict_Information", []):
            if not isinstance(item, dict):
                continue
            if item.get("Conflict_ID") != conflict_id:
                continue
            rank = role_rank(item.get("Role", ""))
            if rank == target_rank:
                candidate = copy.deepcopy(item)

    return candidate


def detect_session_triggers(session_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    A-class trigger:
      - Updated_Attributes has valid After.
    B-class trigger:
      - Static/Conditional conflict has non-Point_A non-Distractor points.
    C-class (distractor-only) is intentionally not used as independent trigger.
    """
    updated_trigger_items = []
    updated_attributes = session_item.get("Updated_Attributes", [])
    if isinstance(updated_attributes, list):
        for item in updated_attributes:
            if is_non_empty_after(item):
                updated_trigger_items.append(copy.deepcopy(item))

    static_trigger_points = []
    static_conflicts = session_item.get("Static_Conflict_Information", [])
    if isinstance(static_conflicts, list):
        for item in static_conflicts:
            if not isinstance(item, dict):
                continue
            role = str(item.get("Role", "")).strip()
            if role in ["Point_A", "Distractor", ""]:
                continue
            static_trigger_points.append({
                "Conflict_ID": item.get("Conflict_ID"),
                "Role": role,
                "Target_Field_Path": item.get("Target_Field_Path"),
                "Value": copy.deepcopy(item.get("Value"))
            })

    conditional_trigger_points = []
    conditional_conflicts = session_item.get("Conditional_Conflict_Information", [])
    if isinstance(conditional_conflicts, list):
        for item in conditional_conflicts:
            if not isinstance(item, dict):
                continue
            role = str(item.get("Role", "")).strip()
            if role in ["Point_A", "Distractor", ""]:
                continue
            conditional_trigger_points.append({
                "Conflict_ID": item.get("Conflict_ID"),
                "Rule_ID": item.get("Rule_ID"),
                "Role": role,
                "Preference_Type": item.get("Preference_Type"),
                "Item": item.get("Item"),
                "Condition": item.get("Condition")
            })

    trigger_types = []
    if len(updated_trigger_items) > 0:
        trigger_types.append("dynamic_update")
    if len(static_trigger_points) > 0:
        trigger_types.append("static_conflict")
    if len(conditional_trigger_points) > 0:
        trigger_types.append("conditional_conflict")

    return {
        "should_generate": len(trigger_types) > 0,
        "trigger_types": trigger_types,
        "updated_trigger_items": updated_trigger_items,
        "static_trigger_points": static_trigger_points,
        "conditional_trigger_points": conditional_trigger_points
    }


def add_question(result: List[Dict[str, Any]],
                 question_idx: int,
                 question: str,
                 answer: str,
                 conflict_type: str,
                 ability_target: str,
                 difficulty: str = "easy") -> int:
    """Append one question and return next question index."""
    result.append({
        "question_id": f"Q_{question_idx:03d}",
        "question": question,
        "answer": answer,
        "conflict_type": conflict_type,
        "ability_target": ability_target,
        "difficulty": difficulty
    })
    return question_idx + 1


def to_natural_text(value: Any) -> str:
    """
    Convert structured values into more natural short text.
    Special handling:
    - {"Current_State": "..."} -> "..."
    """
    if isinstance(value, dict):
        if set(value.keys()) == {"Current_State"}:
            current_state = value.get("Current_State")
            return str(current_state)
        parts = []
        for key, sub_value in value.items():
            label = to_natural_field_name(key).lower()
            parts.append(f"{label}: {to_natural_text(sub_value)}")
        return "; ".join(parts)
    return json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value


def build_marital_status_text(state: Any) -> str:
    """Build a natural short text for marital status values."""
    if not isinstance(state, dict):
        return to_natural_text(state)

    status = state.get("Status")
    name = state.get("Name")

    if status is None:
        return to_natural_text(state)

    status_text = str(status).replace("_", " ").lower()
    if not name:
        return status_text

    if status_text in ["dating", "married", "engaged", "partnered"]:
        return f"{status_text} {name}"
    return f"{status_text} ({name})"


def to_natural_attribute_value(attribute: Any, value: Any) -> str:
    """Convert one attribute value to a more natural form."""
    attr_text = str(attribute) if attribute is not None else ""

    if attr_text == "Marital_Status":
        return build_marital_status_text(value)
    if attr_text == "Career_Status":
        return build_career_snapshot_text(value)

    return to_natural_text(value)


def to_natural_field_name(field_path: Any) -> str:
    """Convert field path to readable display text."""
    text = str(field_path) if field_path is not None else "Unknown field"
    text = text.replace(".", " ").replace("_", " ")
    return " ".join(text.split())


CAREER_FIELD_PRIORITY = [
    "Company_Name",
    "Job_Title",
    "Industry",
    "Monthly_Income",
    "Company_Type",
    "Savings_Amount",
    "Employment_Status"
]

CAREER_FIELD_LABELS = {
    "Company_Name": "company",
    "Job_Title": "job title",
    "Industry": "industry",
    "Monthly_Income": "monthly income",
    "Company_Type": "company type",
    "Savings_Amount": "savings range",
    "Employment_Status": "employment status"
}


def build_career_snapshot_text(state: Any) -> str:
    """Build a concise natural snapshot text for one career state."""
    if not isinstance(state, dict):
        return to_natural_text(state)

    parts = []
    employment = state.get("Employment_Status")
    title = state.get("Job_Title")
    company = state.get("Company_Name")
    industry = state.get("Industry")
    income = state.get("Monthly_Income")

    if employment:
        parts.append(f"employment status: {employment}")
    if title:
        parts.append(f"title: {title}")
    if company:
        parts.append(f"company: {company}")
    if industry:
        parts.append(f"industry: {industry}")
    if income:
        parts.append(f"income range: {income}")

    if len(parts) == 0:
        return to_natural_text(state)
    return "; ".join(parts)


def extract_career_changes(before: Dict[str, Any], after: Dict[str, Any]) -> tuple:
    """Return (changed_items, unchanged_items) ordered by field priority."""
    changed = []
    unchanged = []

    ordered_keys = copy.deepcopy(CAREER_FIELD_PRIORITY)
    for key in sorted(set(before.keys()) | set(after.keys())):
        if key not in ordered_keys:
            ordered_keys.append(key)

    for key in ordered_keys:
        if key not in before and key not in after:
            continue
        before_v = before.get(key)
        after_v = after.get(key)
        if before_v != after_v:
            changed.append((key, before_v, after_v))
        else:
            unchanged.append((key, before_v))

    return changed, unchanged


def generate_dynamic_questions(updated_items: List[Dict[str, Any]], current_session_id: int, start_idx: int) -> List[Dict[str, Any]]:
    """Generate direct dynamic-update questions from Updated_Attributes."""
    questions = []
    q_idx = start_idx

    for item in updated_items:
        attr = item.get("Attribute", "Unknown_Attribute")
        attr_name = to_natural_field_name(attr).lower()
        before = item.get("Before")
        after = item.get("After")
        before_text = to_natural_attribute_value(attr, before)
        after_text = to_natural_attribute_value(attr, after)

        # Career_Status is a composite object: generate multiple focused questions.
        if attr == "Career_Status" and isinstance(before, dict) and isinstance(after, dict):
            changed_items, unchanged_items = extract_career_changes(before, after)

            q_idx = add_question(
                result=questions,
                question_idx=q_idx,
                question="How has the user's career situation changed?",
                answer=(
                    "The career profile changed from "
                    f"{build_career_snapshot_text(before)} to {build_career_snapshot_text(after)}."
                ),
                conflict_type="dynamic_conflict",
                ability_target="track_state_over_time",
                difficulty="medium"
            )

            for key, before_v, after_v in changed_items[:3]:
                label = CAREER_FIELD_LABELS.get(key, to_natural_field_name(key))
                q_idx = add_question(
                    result=questions,
                    question_idx=q_idx,
                    question=f"What changed about the user's {label}?",
                    answer=f"The {label} changed from {to_natural_text(before_v)} to {to_natural_text(after_v)}.",
                    conflict_type="dynamic_conflict",
                    ability_target="track_state_over_time",
                    difficulty="medium"
                )

            stable_preference = ["Employment_Status", "Savings_Amount", "Company_Type"]
            stable_item = None
            for key in stable_preference:
                for unchanged_key, unchanged_v in unchanged_items:
                    if unchanged_key == key:
                        stable_item = (unchanged_key, unchanged_v)
                        break
                if stable_item is not None:
                    break

            if stable_item is not None:
                stable_key, stable_value = stable_item
                stable_label = CAREER_FIELD_LABELS.get(stable_key, to_natural_field_name(stable_key))
                q_idx = add_question(
                    result=questions,
                    question_idx=q_idx,
                    question=f"Has the user's {stable_label} stayed the same?",
                    answer=f"Yes. The {stable_label} remained {to_natural_text(stable_value)}.",
                    conflict_type="dynamic_conflict",
                    ability_target="track_state_over_time",
                    difficulty="easy"
                )
            continue

        q_idx = add_question(
            result=questions,
            question_idx=q_idx,
            question=f"Has the user's {attr_name} changed recently?",
            answer="Yes.",
            conflict_type="dynamic_conflict",
            ability_target="track_state_over_time",
            difficulty="easy"
        )

        q_idx = add_question(
            result=questions,
            question_idx=q_idx,
            question=f"How has the user's {attr_name} changed?",
            answer=f"The user's {attr_name} changed from {before_text} to {after_text}.",
            conflict_type="dynamic_conflict",
            ability_target="track_state_over_time",
            difficulty="medium"
        )

    return questions


def generate_static_questions(static_points: List[Dict[str, Any]],
                              visible_context: List[Dict[str, Any]],
                              current_session_id: int,
                              start_idx: int) -> List[Dict[str, Any]]:
    """Generate static-conflict questions focused on inconsistency and truth recovery."""
    questions = []
    q_idx = start_idx

    for point in static_points:
        conflict_id = point.get("Conflict_ID")
        field_path = point.get("Target_Field_Path", "Unknown_Field")
        field_name = to_natural_field_name(field_path)
        point_b_value = point.get("Value")
        point_a_value = find_point_a_static_value(visible_context, conflict_id, field_path)
        point_b_text = to_natural_text(point_b_value)

        if point_a_value is None:
            answer_text = (
                f"The information is inconsistent for {field_name}. "
                f"Another statement says {point_b_text}. "
                "The correct value cannot be confirmed from the available context."
            )
        else:
            point_a_text = to_natural_text(point_a_value)
            answer_text = (
                f"There is conflicting information about the user's {field_name}. "
                f"The correct value is {point_a_text}, while another statement says {point_b_text}."
            )

        q_idx = add_question(
            result=questions,
            question_idx=q_idx,
            question=f"What is the user's {field_name}?",
            answer=answer_text,
            conflict_type="static_conflict",
            ability_target="recover_truth",
            difficulty="hard"
        )

    return questions


def generate_conditional_questions(conditional_points: List[Dict[str, Any]],
                                   visible_context: List[Dict[str, Any]],
                                   current_session_id: int,
                                   start_idx: int) -> List[Dict[str, Any]]:
    """
    Generate conditional-conflict questions:
    - Point_B session asks previous point's rule (usually Point_A).
    - Point_C session asks previous point's rule (Point_B), etc.
    """
    questions = []
    q_idx = start_idx

    for point in conditional_points:
        conflict_id = point.get("Conflict_ID")
        current_role = point.get("Role", "")
        current_rank = role_rank(current_role)

        previous_point = find_previous_conditional_point(
            visible_context=visible_context,
            conflict_id=conflict_id,
            current_rank=current_rank
        )

        if previous_point:
            pref_type = previous_point.get("Preference_Type", point.get("Preference_Type", "Unknown_Preference"))
            item = previous_point.get("Item", "Unknown_Item")
            condition = previous_point.get("Condition", "Unknown_Condition")
            role_text = previous_point.get("Role", "previous point")
            answer_text = (
                f"For the user's {pref_type}, {to_natural_text(item)} is associated with the condition {to_natural_text(condition)}."
            )
        else:
            pref_type = point.get("Preference_Type", "Unknown_Preference")
            item = point.get("Item", "Unknown_Item")
            condition = point.get("Condition", "Unknown_Condition")
            answer_text = (
                f"For the user's {pref_type}, {to_natural_text(item)} is associated with the condition {to_natural_text(condition)}."
            )

        q_idx = add_question(
            result=questions,
            question_idx=q_idx,
            question=f"For the user's {pref_type}, under what condition is {json.dumps(item, ensure_ascii=False)} preferred or selected?",
            answer=answer_text,
            conflict_type="conditional_conflict",
            ability_target="bind_condition",
            difficulty="hard"
            )

    return questions


def sanitize_questions(raw_questions: List[Dict[str, Any]], current_idx: int) -> List[Dict[str, Any]]:
    """Validate and sanitize question schema."""
    valid_conflict_types = {"dynamic_conflict", "static_conflict", "conditional_conflict"}
    valid_ability_targets = {"track_state_over_time", "detect_conflict", "recover_truth", "bind_condition"}
    valid_difficulty = {"easy", "medium", "hard"}

    cleaned = []
    for i, q in enumerate(raw_questions, start=1):
        if not isinstance(q, dict):
            continue
        question = str(q.get("question", "")).strip()
        answer = str(q.get("answer", "")).strip()
        if not question or not answer:
            continue

        conflict_type = str(q.get("conflict_type", "dynamic_conflict")).strip()
        if conflict_type not in valid_conflict_types:
            conflict_type = "dynamic_conflict"

        ability_target = str(q.get("ability_target", "recover_truth")).strip()
        if ability_target not in valid_ability_targets:
            ability_target = "recover_truth"

        difficulty = str(q.get("difficulty", "medium")).strip()
        if difficulty not in valid_difficulty:
            difficulty = "medium"

        question_id = str(q.get("question_id", f"Q_{i:03d}")).strip() or f"Q_{i:03d}"

        cleaned.append({
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "conflict_type": conflict_type,
            "ability_target": ability_target,
            "difficulty": difficulty
        })

    return cleaned


def validate_rewritten_questions(rewritten_questions: List[Dict[str, Any]],
                                 template_questions: List[Dict[str, Any]]) -> bool:
    """
    Strictly validate rewrite output before accepting it.
    We require:
    - same number of questions
    - same question_id sequence
    - non-empty rewritten question and answer
    """
    if not isinstance(rewritten_questions, list):
        return False
    if len(rewritten_questions) != len(template_questions):
        return False

    for template_q, rewritten_q in zip(template_questions, rewritten_questions):
        if not isinstance(rewritten_q, dict):
            return False
        if str(rewritten_q.get("question_id", "")).strip() != str(template_q.get("question_id", "")).strip():
            return False
        if not str(rewritten_q.get("question", "")).strip():
            return False
        if not str(rewritten_q.get("answer", "")).strip():
            return False

    return True


def merge_rewritten_with_template(rewritten_questions: List[Dict[str, Any]],
                                  template_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preserve structural labels from template questions and only accept rewritten
    natural-language surface forms for question/answer.
    """
    merged = []
    for template_q, rewritten_q in zip(template_questions, rewritten_questions):
        merged.append({
            "question_id": template_q.get("question_id"),
            "question": str(rewritten_q.get("question", template_q.get("question", ""))).strip(),
            "answer": str(rewritten_q.get("answer", template_q.get("answer", ""))).strip(),
            "conflict_type": template_q.get("conflict_type"),
            "ability_target": template_q.get("ability_target"),
            "difficulty": template_q.get("difficulty")
        })
    return merged


def rewrite_questions_with_llm(system_prompt: str,
                               visible_context: List[Dict[str, Any]],
                               questions: List[Dict[str, Any]]) -> tuple:
    """
    Optional rewrite-only mode:
    Keep semantics but improve wording.
    """
    zero_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False
    }

    if not os.getenv("OPENAI_API_KEY"):
        return questions, zero_cost

    try:
        payload = {
            "Task": "Rewrite questions for clarity without changing facts or answers.",
            "Visible_Session_Context": visible_context,
            "Template_Questions": questions,
            "Output_JSON_Schema": {"questions": questions}
        }

        user_prompt = "Input data:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        json_markers = ["Final JSON", "JSON Result", "Rewritten Questions", "Generation Result"]

        parsed_result, cost_info = llm_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            return_parsed_json=True,
            json_markers=json_markers
        )

        rewritten = parsed_result.get("questions", [])
        if not validate_rewritten_questions(rewritten, questions):
            print("[DEBUG] Rewritten questions failed structural validation, keep template questions.")
            return questions, cost_info

        merged_questions = merge_rewritten_with_template(rewritten, questions)
        return merged_questions, cost_info

    except Exception as e:
        print(f"[DEBUG] LLM rewrite failed, keep template questions: {e}:{traceback.format_exc()}")
        return questions, zero_cost


def update_stage_cost_aggregate(stage_cost_aggregate: Dict[str, Any], call_cost: Dict[str, Any]):
    """Accumulate one LLM call cost into stage aggregate cost."""
    if not isinstance(call_cost, dict):
        return

    stage_cost_aggregate["input_tokens"] += call_cost.get("input_tokens", 0) or 0
    stage_cost_aggregate["output_tokens"] += call_cost.get("output_tokens", 0) or 0
    stage_cost_aggregate["total_tokens"] += call_cost.get("total_tokens", 0) or 0
    stage_cost_aggregate["total_cost_usd"] += call_cost.get("total_cost_usd", 0.0) or 0.0

    if stage_cost_aggregate["model"] is None:
        stage_cost_aggregate["model"] = call_cost.get("model")
    if call_cost.get("pricing_available") is True:
        stage_cost_aggregate["pricing_available"] = True


def Generate_Single_Session_Questions(persona_item: Dict[str, Any], system_prompt: str, enable_llm_rewrite: bool = True):
    """
    Step 4.4:
    Session-by-session question generation using rule templates.
    """
    try:
        full_session_chain = copy.deepcopy(persona_item["Full_Session_Chain"])
        previous_cost = persona_item.get("token_cost", None)

        current_stage_total_cost = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "model": None,
            "pricing_available": False,
            "note": "Rule-based session-level question generation"
        }

        total_question_count = 0
        triggered_session_count = 0

        for current_idx, session_item in enumerate(full_session_chain):
            print(f"[DEBUG] Processing session {current_idx + 1}/{len(full_session_chain)}")

            trigger_info = detect_session_triggers(session_item)
            session_item["Question_Trigger_Types"] = trigger_info["trigger_types"]
            session_item["Session_Questions"] = []
            session_item["Session_Question_Count"] = 0

            if not trigger_info["should_generate"]:
                session_item = prune_session_fields_for_step4_4(session_item)
                full_session_chain[current_idx] = place_question_fields_before_event_types(session_item)
                continue

            triggered_session_count += 1
            current_session_id = session_item.get("Session_ID", current_idx)
            visible_context = collect_visible_context(full_session_chain, current_idx)

            questions = []
            next_idx = 1

            # Step 1. Dynamic questions
            dynamic_questions = generate_dynamic_questions(
                updated_items=trigger_info["updated_trigger_items"],
                current_session_id=current_session_id,
                start_idx=next_idx
            )
            questions.extend(dynamic_questions)
            next_idx = len(questions) + 1

            # Step 2. Static conflict questions
            static_questions = generate_static_questions(
                static_points=trigger_info["static_trigger_points"],
                visible_context=visible_context,
                current_session_id=current_session_id,
                start_idx=next_idx
            )
            questions.extend(static_questions)
            next_idx = len(questions) + 1

            # Step 3. Conditional conflict questions
            conditional_questions = generate_conditional_questions(
                conditional_points=trigger_info["conditional_trigger_points"],
                visible_context=visible_context,
                current_session_id=current_session_id,
                start_idx=next_idx
            )
            questions.extend(conditional_questions)

            # Optional rewrite with LLM (disabled by default)
            if enable_llm_rewrite:
                rewritten_questions, call_cost = rewrite_questions_with_llm(
                    system_prompt=system_prompt,
                    visible_context=visible_context,
                    questions=questions
                )
                questions = rewritten_questions
                update_stage_cost_aggregate(current_stage_total_cost, call_cost)

            questions = sanitize_questions(questions, current_idx)

            session_item["Session_Questions"] = questions
            session_item["Session_Question_Count"] = len(questions)
            total_question_count += len(questions)
            session_item = prune_session_fields_for_step4_4(session_item)
            full_session_chain[current_idx] = place_question_fields_before_event_types(session_item)

            print(
                f"[DEBUG] Triggered session {current_session_id} with trigger types "
                f"{trigger_info['trigger_types']} and generated {len(questions)} questions"
            )


        final_cost = calculate_cumulative_cost(previous_cost, current_stage_total_cost)
        return full_session_chain, total_question_count, triggered_session_count, final_cost

    except Exception as e:
        print(f"[DEBUG] Generate_Single_Session_Questions failed: {e}:{traceback.format_exc()}")
        raise


def Generate_User_Session_Questions(args):
    print(f"Processing file: {args.input_file}")
    print(f"Output file: {args.output_file}")

    try:
        system_prompt = Load_Step4_4_Prompt(args.prompt_file)
        print(f"[DEBUG] Loaded prompt file: {args.prompt_file}")
        print(f"[DEBUG] Enable LLM rewrite: {args.enable_llm_rewrite}")

        all_personas = load_jsonl_items(args.input_file)
        print(f"[DEBUG] Read {len(all_personas)} personas")

        results: List[Dict[str, Any]] = []
        for idx, persona_item in enumerate(all_personas):
            print(f"[DEBUG] Processing persona {idx + 1}/{len(all_personas)}")

            updated_chain, total_question_count, triggered_session_count, final_cost = (
                Generate_Single_Session_Questions(
                    persona_item=persona_item,
                    system_prompt=system_prompt,
                    enable_llm_rewrite=args.enable_llm_rewrite
                )
            )

            result_item = copy.deepcopy(persona_item)
            result_item["Full_Session_Chain"] = updated_chain
            result_item["Total_Session_Question_Count"] = total_question_count
            result_item["Triggered_Session_Count"] = triggered_session_count
            result_item["token_cost"] = final_cost
            results.append(result_item)

            print(
                f"[DEBUG] Persona {idx + 1} completed - Triggered sessions: "
                f"{triggered_session_count}, Total questions: {total_question_count}"
            )

        write_jsonl_items(args.output_file, results)
        write_json_items(args.output_json_file, results)
        print("[DEBUG] Successfully processed Step 4.4 session-level question generation")
        return True

    except Exception as e:
        print(f"Error processing Step 4.4: {e}:{traceback.format_exc()}")
        return False


def run_single_mode(args: argparse.Namespace, interval_mode: str) -> bool:
    input_file = resolve_input_path(interval_mode, args.input_file)
    output_file, output_json_file = resolve_output_paths(
        interval_mode,
        args.output_file,
        args.output_json_file,
    )
    prompt_file = resolve_prompt_path(args.prompt_file)

    run_args = SimpleNamespace(
        input_file=input_file,
        output_file=output_file,
        output_json_file=output_json_file,
        prompt_file=prompt_file,
        disable_llm_rewrite=args.disable_llm_rewrite,
        enable_llm_rewrite=not args.disable_llm_rewrite,
    )

    print(f"[DEBUG] Interval mode: {interval_mode}")
    print(f"[DEBUG] Output JSON file: {output_json_file}")
    return Generate_User_Session_Questions(run_args)


def main(args: argparse.Namespace) -> bool:
    if args.interval_mode == "both":
        short_ok = run_single_mode(args, "short")
        long_ok = run_single_mode(args, "long")
        return short_ok and long_ok
    return run_single_mode(args, args.interval_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conflict interval ablation: Step4_4 session-level question generation.")
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input Step4_3 interval JSONL file; if interval_mode=both, leave empty to use default per-mode names",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output JSONL file; if interval_mode=both, leave empty to use default per-mode names",
    )
    parser.add_argument(
        "--output_json_file",
        type=str,
        default=None,
        help="Output JSON file; if interval_mode=both, leave empty to use default per-mode names",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Prompt file for optional rewrite stage",
    )
    parser.add_argument(
        "--disable_llm_rewrite",
        action="store_true",
        help="Disable LLM rewrite and use rule-template text directly",
    )
    parser.add_argument(
        "--interval_mode",
        type=str,
        choices=["short", "long", "both"],
        default="both",
        help="Conflict interval mode for the first ablation version",
    )
    args = parser.parse_args()
    main(args)
