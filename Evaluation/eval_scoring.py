import argparse
import copy
import math
import json
import os
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import logging

try:
    import jsonlines
except ImportError:
    jsonlines = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    from llm_request import (
        calculate_cumulative_cost as project_calculate_cumulative_cost,
        llm_request as project_llm_request,
    )
except Exception:
    project_calculate_cumulative_cost = None
    project_llm_request = None

logger = logging.getLogger(__name__)
load_dotenv()

PRIMARY_WHITE_BOX_TOP_K = 3
WHITE_BOX_TOP_K_VALUES = [2, 3, 5]
WHITE_BOX_TOP_K = PRIMARY_WHITE_BOX_TOP_K
MAX_WHITE_BOX_TOP_K = max(WHITE_BOX_TOP_K_VALUES)

METRIC_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "dynamic_conflict": {
        "summary_name": "Dynamic_Conflict",
        "black_box_metrics": [
            "dynamic_answer_accuracy",
            "update_awareness_and_order_consistency_score",
        ],
        "white_box_metrics": [
            "updated_evidence_hit_at_3",
            "updated_evidence_log_rank_score_at_3",
        ],
    },
    "static_conflict": {
        "summary_name": "Static_Conflict",
        "black_box_metrics": [
            "static_answer_accuracy",
            "conflict_recognition_score",
        ],
        "white_box_metrics": [
            "truth_evidence_hit_at_3",
            "truth_evidence_log_rank_score_at_3",
        ],
    },
    "conditional_conflict": {
        "summary_name": "Conditional_Conflict",
        "black_box_metrics": [
            "conditional_answer_accuracy",
        ],
        "white_box_metrics": [
            "correct_condition_evidence_hit_at_3",
            "correct_condition_evidence_log_rank_score_at_3",
        ],
    },
}

WHITE_BOX_METRIC_CONFIGS: Dict[str, Dict[str, str]] = {
    "dynamic_conflict": {
        "hit_metric_base": "updated_evidence_hit_at",
        "log_rank_metric_base": "updated_evidence_log_rank_score_at",
        "support_rank_field": "updated_evidence_first_support_rank",
        "support_description": "updated-state evidence",
    },
    "static_conflict": {
        "hit_metric_base": "truth_evidence_hit_at",
        "log_rank_metric_base": "truth_evidence_log_rank_score_at",
        "support_rank_field": "truth_evidence_first_support_rank",
        "support_description": "truth-supporting evidence",
    },
    "conditional_conflict": {
        "hit_metric_base": "correct_condition_evidence_hit_at",
        "log_rank_metric_base": "correct_condition_evidence_log_rank_score_at",
        "support_rank_field": "correct_condition_evidence_first_support_rank",
        "support_description": "correct-condition evidence",
    },
}

LLM_JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for memory-conflict experiments.
You must score only the requested metrics.
Use the reference answer as the gold standard.
Use the retrieved memories only for white-box scoring.
Return valid JSON only."""

def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    normalized = str(text)
    normalized = normalized.replace("_", " ").replace("-", " ").lower()
    normalized = re.sub(r"[\"'`]", " ", normalized)
    normalized = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_normalized_answer_variants(text: Any) -> List[str]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return []

    variants: List[str] = []
    separators = ["||", "\n", ";"]

    def add_variant(candidate: str):
        normalized = normalize_text(candidate)
        if normalized and normalized not in variants:
            variants.append(normalized)

    add_variant(raw_text)

    for separator in separators:
        if separator not in raw_text:
            continue
        for part in raw_text.split(separator):
            add_variant(part)

    return variants


def answers_match_strict(gold_answer: Any, model_answer: Any) -> bool:
    gold_variants = extract_normalized_answer_variants(gold_answer)
    model_variants = extract_normalized_answer_variants(model_answer)
    if not gold_variants or not model_variants:
        return False
    return any(gold_variant == model_variant for gold_variant in gold_variants for model_variant in model_variants)


FALLBACK_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "from", "of", "in", "on", "at", "for", "with", "by", "as",
    "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "it", "they", "them", "their", "he", "she", "his", "her", "you", "your",
    "user", "users", "now", "current", "currently", "recently", "about",
    "did", "does", "do", "has", "have", "had", "what", "where", "when", "which",
}


def extract_meaningful_tokens(text: Any) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    tokens: List[str] = []
    for token in normalized.split():
        if len(token) <= 1:
            continue
        if token in FALLBACK_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def compute_partial_credit_score(gold_answer: Any, model_answer: Any) -> float:
    if answers_match_strict(gold_answer, model_answer):
        return 1.0

    gold_variants = extract_normalized_answer_variants(gold_answer)
    model_variants = extract_normalized_answer_variants(model_answer)
    if not gold_variants or not model_variants:
        return 0.0

    best_overlap_ratio = 0.0
    best_shared_count = 0
    for gold_variant in gold_variants:
        gold_tokens = set(extract_meaningful_tokens(gold_variant))
        if not gold_tokens:
            continue
        for model_variant in model_variants:
            model_tokens = set(extract_meaningful_tokens(model_variant))
            if not model_tokens:
                continue
            shared_tokens = gold_tokens & model_tokens
            shared_count = len(shared_tokens)
            overlap_ratio = shared_count / max(1, len(gold_tokens))
            if overlap_ratio > best_overlap_ratio:
                best_overlap_ratio = overlap_ratio
            if shared_count > best_shared_count:
                best_shared_count = shared_count

            if gold_variant in model_variant and len(gold_variant) >= 8:
                return 0.5
            if model_variant in gold_variant and len(model_variant) >= 8:
                return 0.5

    if best_shared_count >= 2 or best_overlap_ratio >= 0.5:
        return 0.5
    return 0.0


def has_update_order_signal(model_answer: Any) -> bool:
    normalized = normalize_text(model_answer)
    if not normalized:
        return False

    change_markers = [
        "changed", "change", "updated", "update", "switched",
    ]
    order_markers = [
        ("from", "to"),
        ("previously", "now"),
        ("used to", "now"),
        ("before", "now"),
    ]

    has_change = any(marker in normalized for marker in change_markers)
    has_order = any(left in normalized and right in normalized for left, right in order_markers)
    return has_change and has_order


def has_conflict_recognition_signal(model_answer: Any) -> bool:
    normalized = normalize_text(model_answer)
    if not normalized:
        return False

    keywords = [
        "inconsisten", "conflict", "contradict", "cannot confirm", "uncertain", "mismatch",
    ]
    return any(keyword in normalized for keyword in keywords)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def Build_Observable_Token_Cost_Summary(stage_cost: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    return {
        "Stage_Name": stage_name,
        "Input_Tokens": stage_cost.get("input_tokens", 0) or 0,
        "Output_Tokens": stage_cost.get("output_tokens", 0) or 0,
        "Total_Tokens": stage_cost.get("total_tokens", 0) or 0,
        "Total_Cost_USD": stage_cost.get("total_cost_usd", 0.0) or 0.0,
        "Model": stage_cost.get("model"),
        "Pricing_Available": bool(stage_cost.get("pricing_available")),
    }


def calculate_cumulative_cost(previous_cost: Optional[Dict[str, Any]], current_stage_total_cost: Dict[str, Any]) -> Dict[str, Any]:
    if project_calculate_cumulative_cost is not None:
        return project_calculate_cumulative_cost(previous_cost, current_stage_total_cost)

    if not isinstance(previous_cost, dict):
        return copy.deepcopy(current_stage_total_cost)

    merged = copy.deepcopy(previous_cost)
    for key in ["input_tokens", "output_tokens", "total_tokens"]:
        merged[key] = (merged.get(key, 0) or 0) + (current_stage_total_cost.get(key, 0) or 0)
    merged["total_cost_usd"] = (merged.get("total_cost_usd", 0.0) or 0.0) + (
        current_stage_total_cost.get("total_cost_usd", 0.0) or 0.0
    )
    if merged.get("model") is None:
        merged["model"] = current_stage_total_cost.get("model")
    merged["pricing_available"] = bool(
        merged.get("pricing_available") or current_stage_total_cost.get("pricing_available")
    )
    if current_stage_total_cost.get("note"):
        merged["note"] = current_stage_total_cost.get("note")
    return merged


def load_jsonl_items(input_file: str) -> List[Dict[str, Any]]:
    items = []
    if jsonlines is not None:
        with jsonlines.open(input_file) as reader:
            for item in reader:
                items.append(item)
        return items

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl_items(output_file: str, items: List[Dict[str, Any]]):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if jsonlines is not None:
        with jsonlines.open(output_file, "w") as writer:
            for item in items:
                writer.write(item)
        return

    with open(output_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl_item(output_file: str, item: Dict[str, Any]):
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_model_answer(question_item: Dict[str, Any], prediction_fields: List[str]) -> Tuple[str, Optional[str]]:
    for field_name in prediction_fields:
        if field_name in question_item:
            value = question_item.get(field_name)
            if value not in [None, ""]:
                return str(value), field_name
    return "", None


def extract_top_k_retrieved_memories(question_item: Dict[str, Any], top_k: int = WHITE_BOX_TOP_K) -> List[Dict[str, Any]]:
    retrieved_memories = question_item.get("Retrieved_Memories", [])
    if not isinstance(retrieved_memories, list):
        return []

    normalized_results = []
    for index, item in enumerate(retrieved_memories[:top_k], start=1):
        if not isinstance(item, dict):
            continue
        normalized_results.append({
            "rank": index,
            "memory": str(item.get("memory", "")),
            "created_at": item.get("created_at", "Unknown Time"),
            "score": item.get("score"),
        })
    return normalized_results


def format_retrieved_memories_for_prompt(retrieved_memories: List[Dict[str, Any]]) -> str:
    if len(retrieved_memories) == 0:
        return "No retrieved memories."

    lines = []
    for item in retrieved_memories:
        score_text = "None" if item.get("score") is None else str(item.get("score"))
        lines.append(
            f"{item.get('rank')}. [{item.get('created_at')}] {item.get('memory')} (score={score_text})"
        )
    return "\n".join(lines)


def get_white_box_metric_config(conflict_type: str, top_k: int = WHITE_BOX_TOP_K) -> Dict[str, str]:
    config = WHITE_BOX_METRIC_CONFIGS[conflict_type]
    return {
        "hit_metric": f"{config['hit_metric_base']}_{top_k}",
        "log_rank_metric": f"{config['log_rank_metric_base']}_{top_k}",
        "support_rank_field": config["support_rank_field"],
        "support_description": config["support_description"],
    }


def get_metric_keys(conflict_type: str, top_k: int = WHITE_BOX_TOP_K) -> List[str]:
    schema = METRIC_SCHEMAS[conflict_type]
    white_box_config = get_white_box_metric_config(conflict_type, top_k)
    return schema["black_box_metrics"] + [
        white_box_config["hit_metric"],
        white_box_config["log_rank_metric"],
    ]


def build_default_metric_payload(conflict_type: str, top_k: int = WHITE_BOX_TOP_K) -> Tuple[Dict[str, float], Dict[str, bool]]:
    metrics = {metric_key: 0.0 for metric_key in get_metric_keys(conflict_type, top_k)}
    applicability: Dict[str, bool] = {}
    return metrics, applicability


def build_missing_answer_result(conflict_type: str, used_field: Optional[str], top_k: int = WHITE_BOX_TOP_K) -> Dict[str, Any]:
    metrics, applicability = build_default_metric_payload(conflict_type, top_k)
    return {
        "Conflict_Type": conflict_type,
        "Used_Prediction_Field": used_field,
        "Judge_Method": "missing_answer",
        "Metrics": metrics,
        "Applicable": applicability,
        "Reasoning": "No model answer was found in the configured prediction fields.",
        "Error_Tags": ["missing_model_answer"],
        "White_Box_Top_K": top_k,
    }


def build_rule_based_result(
    question_item: Dict[str, Any],
    model_answer: str,
    used_field: Optional[str],
    top_k: int = WHITE_BOX_TOP_K,
) -> Dict[str, Any]:
    conflict_type = question_item.get("conflict_type", "unknown")
    metrics, applicability = build_default_metric_payload(conflict_type, top_k)

    gold_answer = str(question_item.get("answer", "")).strip()
    answer_score = compute_partial_credit_score(gold_answer, model_answer)

    if conflict_type == "dynamic_conflict":
        metrics["dynamic_answer_accuracy"] = answer_score
        if answer_score >= 0.5 and has_update_order_signal(model_answer):
            metrics["update_awareness_and_order_consistency_score"] = 1
    elif conflict_type == "static_conflict":
        metrics["static_answer_accuracy"] = answer_score
        if has_conflict_recognition_signal(model_answer):
            metrics["conflict_recognition_score"] = 1
    elif conflict_type == "conditional_conflict":
        metrics["conditional_answer_accuracy"] = 1.0 if answer_score >= 0.5 else 0.0

    return {
        "Conflict_Type": conflict_type,
        "Used_Prediction_Field": used_field,
        "Judge_Method": "rule_based",
        "Metrics": metrics,
        "Applicable": applicability,
        "Reasoning": "Rule-based fallback evaluation was used because the LLM judge was unavailable.",
        "Error_Tags": ["rule_based_fallback"],
        "White_Box_Top_K": top_k,
    }


def build_llm_judge_prompt(
    question_item: Dict[str, Any],
    model_answer: str,
    retrieved_memories: List[Dict[str, Any]],
    top_k: int = WHITE_BOX_TOP_K,
) -> str:
    conflict_type = question_item.get("conflict_type", "unknown")
    white_box_config = get_white_box_metric_config(conflict_type, top_k)
    question = question_item.get("question", "")
    gold_answer = question_item.get("answer", "")
    retrieved_text = format_retrieved_memories_for_prompt(retrieved_memories)
    if conflict_type == "dynamic_conflict":
        return f"""You are evaluating one dynamic-conflict question.

Inputs:
1. Question: {question}
2. Reference Answer: {gold_answer}
3. Model Answer: {model_answer}
4. Top-{top_k} Retrieved Memories:
{retrieved_text}

Metric definitions:
- dynamic_answer_accuracy: Score 1.0 if the model answer captures the core updated fact correctly according to the reference answer. Score 0.5 if it is partially correct: it mentions a correct key entity, destination, new state, or update direction, but misses important old/new details or is too incomplete for full credit. Score 0.0 if it is wrong, contradictory, does not contain the key updated fact, or is overly uncertain. Do not require exact wording.
- update_awareness_and_order_consistency_score: 1 if the model answer clearly shows that an update/change happened and preserves the correct old-to-new direction. Do not require every old/new detail to be repeated, but the answer must still indicate the update in the right direction. Otherwise 0.
- {white_box_config['support_rank_field']}: an integer from 0 to {top_k}. Set it to the 1-based rank of the first retrieved memory in Top-{top_k} that contains {white_box_config['support_description']} supporting the reference answer. Set it to 0 if no such evidence appears in Top-{top_k}.

Return JSON exactly in this schema:
{{
  "dynamic_answer_accuracy": 0.0,
  "update_awareness_and_order_consistency_score": 0,
  "{white_box_config['support_rank_field']}": 0,
  "reasoning": "short explanation"
}}"""
    if conflict_type == "static_conflict":
        return f"""You are evaluating one static-conflict question.

Inputs:
1. Question: {question}
2. Reference Answer: {gold_answer}
3. Model Answer: {model_answer}
4. Top-{top_k} Retrieved Memories:
{retrieved_text}

Metric definitions:
- static_answer_accuracy: Score 1.0 if the model answer captures the core true fact or judgment correctly according to the reference answer. Score 0.5 if it contains a correct key fact or partially correct semantic match, but is incomplete, vague, or misses an important detail needed for full correctness. Score 0.0 if it is wrong, contradictory, does not contain the key true fact, or is overly uncertain. Do not require exact wording.
- conflict_recognition_score: 1 if the model answer appropriately recognizes the inconsistency or uncertainty that exists in this static-conflict case; otherwise 0.
- {white_box_config['support_rank_field']}: an integer from 0 to {top_k}. Set it to the 1-based rank of the first retrieved memory in Top-{top_k} that contains {white_box_config['support_description']} supporting the reference answer. Set it to 0 if no such evidence appears in Top-{top_k}.

Return JSON exactly in this schema:
{{
  "static_answer_accuracy": 0.0,
  "conflict_recognition_score": 0,
  "{white_box_config['support_rank_field']}": 0,
  "reasoning": "short explanation"
}}"""
    if conflict_type == "conditional_conflict":
        return f"""You are evaluating one conditional-conflict question.

Inputs:
1. Question: {question}
2. Reference Answer: {gold_answer}
3. Model Answer: {model_answer}
4. Top-{top_k} Retrieved Memories:
{retrieved_text}

Metric definitions:
- conditional_answer_accuracy: Score 1.0 if the model answer gives the correct condition according to the reference answer in a semantically correct way. If the answer captures any correct core condition that satisfies the reference, give 1.0. Score 0.0 if the condition is wrong, contradictory, absent, or overly uncertain. Do not require exact wording.
- {white_box_config['support_rank_field']}: an integer from 0 to {top_k}. Set it to the 1-based rank of the first retrieved memory in Top-{top_k} that contains {white_box_config['support_description']} supporting the reference answer. Set it to 0 if no such evidence appears in Top-{top_k}.

Return JSON exactly in this schema:
{{
  "conditional_answer_accuracy": 0.0,
  "{white_box_config['support_rank_field']}": 0,
  "reasoning": "short explanation"
}}"""
    raise ValueError(f"Unsupported conflict_type: {conflict_type}")


def parse_binary_value(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        numeric = int(value)
        return 1 if numeric != 0 else 0
    except Exception:
        return 0


PARTIAL_CREDIT_BLACK_BOX_METRICS = {
    "dynamic_answer_accuracy",
    "static_answer_accuracy",
}


def parse_trinary_score_value(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    if numeric >= 0.75:
        return 1.0
    if numeric >= 0.25:
        return 0.5
    return 0.0


def parse_support_rank(value: Any, top_k: int) -> int:
    try:
        rank_value = int(value)
    except Exception:
        return 0
    if 1 <= rank_value <= top_k:
        return rank_value
    return 0


def derive_white_box_metrics_from_rank(conflict_type: str, support_rank: int, top_k: int) -> Tuple[Dict[str, float], Dict[str, int]]:
    white_box_config = get_white_box_metric_config(conflict_type, top_k)
    hit_metric = white_box_config["hit_metric"]
    log_rank_metric = white_box_config["log_rank_metric"]
    if 1 <= support_rank <= top_k:
        return {
            hit_metric: 1.0,
            log_rank_metric: 1.0 / math.log2(float(support_rank) + 1.0),
        }, {
            white_box_config["support_rank_field"]: support_rank,
        }
    return {
        hit_metric: 0.0,
        log_rank_metric: 0.0,
    }, {
        white_box_config["support_rank_field"]: 0,
    }


def build_white_box_result_by_k(conflict_type: str, global_support_rank: int, judge_method: Optional[str]) -> Dict[str, Any]:
    white_box_by_k: Dict[str, Any] = {}
    for top_k in WHITE_BOX_TOP_K_VALUES:
        white_box_metrics, white_box_metadata = derive_white_box_metrics_from_rank(
            conflict_type=conflict_type,
            support_rank=global_support_rank,
            top_k=top_k,
        )
        white_box_config = get_white_box_metric_config(conflict_type, top_k)
        white_box_by_k[str(top_k)] = {
            "Metrics": white_box_metrics,
            "Support_Rank": white_box_metadata.get(white_box_config["support_rank_field"], 0),
            "Applicable": {},
            "Judge_Method": judge_method,
            "White_Box_Top_K": top_k,
        }
    return white_box_by_k


def parse_llm_metric_result(
    conflict_type: str,
    parsed_result: Dict[str, Any],
    used_field: Optional[str],
    top_k: int = WHITE_BOX_TOP_K,
) -> Dict[str, Any]:
    metrics, applicability = build_default_metric_payload(conflict_type, top_k)
    white_box_config = get_white_box_metric_config(conflict_type, top_k)
    schema = METRIC_SCHEMAS[conflict_type]

    for metric_key in schema["black_box_metrics"]:
        if metric_key in PARTIAL_CREDIT_BLACK_BOX_METRICS:
            metrics[metric_key] = parse_trinary_score_value(parsed_result.get(metric_key, 0))
        else:
            metrics[metric_key] = float(parse_binary_value(parsed_result.get(metric_key, 0)))

    support_rank = parse_support_rank(parsed_result.get(white_box_config["support_rank_field"], 0), top_k)
    derived_white_box_metrics, white_box_metadata = derive_white_box_metrics_from_rank(
        conflict_type=conflict_type,
        support_rank=support_rank,
        top_k=top_k,
    )
    metrics.update(derived_white_box_metrics)

    return {
        "Conflict_Type": conflict_type,
        "Used_Prediction_Field": used_field,
        "Judge_Method": "llm_judge",
        "Metrics": metrics,
        "Applicable": applicability,
        "White_Box_Metadata": white_box_metadata,
        "Reasoning": str(parsed_result.get("reasoning", "")).strip(),
        "Error_Tags": [],
        "White_Box_Top_K": top_k,
    }


def evaluate_question_with_llm(
    question_item: Dict[str, Any],
    model_answer: str,
    used_field: Optional[str],
    retrieved_memories: List[Dict[str, Any]],
    top_k: int = WHITE_BOX_TOP_K,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    zero_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False,
    }

    if project_llm_request is None:
        return None, zero_cost

    try:
        parsed_result, cost_info = project_llm_request(
            system_prompt=LLM_JUDGE_SYSTEM_PROMPT,
            user_prompt=build_llm_judge_prompt(question_item, model_answer, retrieved_memories, top_k),
            return_parsed_json=True,
            json_markers=["Final JSON", "JSON Result", "Evaluation Result", "Result"],
        )
        evaluation_result = parse_llm_metric_result(
            conflict_type=question_item.get("conflict_type", "unknown"),
            parsed_result=parsed_result,
            used_field=used_field,
            top_k=top_k,
        )
        return evaluation_result, cost_info or zero_cost
    except Exception as e:
        print(f"[DEBUG] LLM judge failed, fallback to rule-based scoring: {e}:{traceback.format_exc()}")
        return None, zero_cost


def Evaluate_Single_Question(
    question_item: Dict[str, Any],
    prediction_fields: List[str],
    enable_llm_judge: bool = True,
    top_k: int = WHITE_BOX_TOP_K,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    conflict_type = question_item.get("conflict_type", "unknown")
    zero_cost = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "pricing_available": False,
    }

    if conflict_type not in METRIC_SCHEMAS:
        raise ValueError(f"Unsupported conflict_type: {conflict_type}")

    model_answer, used_field = extract_model_answer(question_item, prediction_fields)
    retrieved_memories = extract_top_k_retrieved_memories(question_item, top_k)

    if not model_answer:
        return build_missing_answer_result(conflict_type, used_field, top_k), zero_cost

    if enable_llm_judge:
        llm_result, llm_cost = evaluate_question_with_llm(
            question_item=question_item,
            model_answer=model_answer,
            used_field=used_field,
            retrieved_memories=retrieved_memories,
            top_k=top_k,
        )
        if llm_result is not None:
            return llm_result, llm_cost

    return build_rule_based_result(question_item, model_answer, used_field, top_k), zero_cost


def place_session_evaluation_before_event_types(session_item: Dict[str, Any]) -> Dict[str, Any]:
    target_key = "Session_Evaluation"
    session_evaluation = copy.deepcopy(session_item.get(target_key))
    reordered = {}
    inserted = False
    for key, value in session_item.items():
        if key == target_key:
            continue
        if key == "Event_Types" and not inserted:
            reordered[target_key] = session_evaluation
            inserted = True
        reordered[key] = value
    if not inserted:
        reordered[target_key] = session_evaluation
    return reordered


def update_error_tag_statistics(error_tag_statistics: Dict[str, int], error_tags: List[str]):
    for tag in error_tags:
        error_tag_statistics[tag] = error_tag_statistics.get(tag, 0) + 1


def build_runtime_summary_from_sessions(full_session_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
    session_count = len(full_session_chain)
    answered_session_count = 0
    persona_add_time_ms = 0.0
    persona_retrieval_time_ms = 0.0
    persona_response_time_ms = 0.0
    persona_total_runtime_ms = 0.0

    for session_item in full_session_chain:
        metadata = session_item.get("Session_Memory_Metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        add_time_ms = float(metadata.get("Add_Duration_ms", 0.0) or 0.0)
        retrieval_time_ms = float(metadata.get("Session_Retrieval_Time_ms", 0.0) or 0.0)
        response_time_ms = float(metadata.get("Session_Response_Time_ms", 0.0) or 0.0)
        total_runtime_ms = float(metadata.get("Session_Total_Runtime_ms", 0.0) or 0.0)
        answered_question_count = int(metadata.get("Session_Answered_Question_Count", 0) or 0)

        persona_add_time_ms += add_time_ms
        persona_retrieval_time_ms += retrieval_time_ms
        persona_response_time_ms += response_time_ms
        persona_total_runtime_ms += total_runtime_ms
        if answered_question_count > 0:
            answered_session_count += 1

    return {
        "Persona_Add_Time_ms": persona_add_time_ms,
        "Persona_Retrieval_Time_ms": persona_retrieval_time_ms,
        "Persona_Response_Time_ms": persona_response_time_ms,
        "Persona_Total_Runtime_ms": persona_total_runtime_ms,
        "Average_Add_Time_Per_Session_ms": safe_divide(persona_add_time_ms, session_count),
        "Average_Retrieval_Time_Per_Session_ms": safe_divide(persona_retrieval_time_ms, answered_session_count),
        "Average_Response_Time_Per_Session_ms": safe_divide(persona_response_time_ms, answered_session_count),
        "Average_Total_Runtime_Per_Session_ms": safe_divide(persona_total_runtime_ms, session_count),
        "Session_Count": session_count,
        "Answered_Session_Count": answered_session_count,
    }


def Build_Compact_Evaluated_Question(question_item: Dict[str, Any]) -> Dict[str, Any]:
    compact_question = {
        "question_id": question_item.get("question_id"),
        "question": question_item.get("question"),
        "answer": question_item.get("answer"),
        "conflict_type": question_item.get("conflict_type"),
        "ability_target": question_item.get("ability_target"),
        "difficulty": question_item.get("difficulty"),
        "Model_Answer": question_item.get("Model_Answer"),
        "Evaluation_Result": copy.deepcopy(question_item.get("Evaluation_Result", {})),
    }
    return compact_question



def Build_Compact_Evaluated_Session(session_item: Dict[str, Any]) -> Dict[str, Any]:
    compact_session = {
        "Session_ID": session_item.get("Session_ID"),
        "Date": session_item.get("Date"),
        "Session_Question_Count": session_item.get("Session_Question_Count", 0),
        "Session_Memory_Metadata": copy.deepcopy(session_item.get("Session_Memory_Metadata", {})),
        "Session_Evaluation": copy.deepcopy(session_item.get("Session_Evaluation", {})),
        "Session_Questions": [],
    }

    session_questions = session_item.get("Session_Questions", [])
    if isinstance(session_questions, list):
        compact_session["Session_Questions"] = [
            Build_Compact_Evaluated_Question(question_item)
            for question_item in session_questions
            if isinstance(question_item, dict)
        ]

    return compact_session



def extract_user_id_fields(persona_item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in persona_item.items()
        if isinstance(key, str) and key.endswith("_User_ID")
    }



def extract_source_runtime_summary(persona_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key, value in persona_item.items():
        if isinstance(key, str) and key.endswith("_Runtime_Summary") and isinstance(value, dict):
            return value
    return None



def Build_Compact_Evaluation_Result_Item(
    persona_item: Dict[str, Any],
    updated_chain: List[Dict[str, Any]],
    evaluation_summary: Dict[str, Any],
    final_cost: Dict[str, Any],
) -> Dict[str, Any]:
    compact_item = {
        "ID": persona_item.get("ID"),
        "Memory_System": persona_item.get("Memory_System"),
        "Eval_Top_K": persona_item.get("Eval_Top_K"),
        "Answered_Session_Count": persona_item.get("Answered_Session_Count"),
        "Answered_Question_Count": persona_item.get("Answered_Question_Count"),
        "Evaluation_Summary": evaluation_summary,
        "token_cost": final_cost,
        "Full_Session_Chain": [
            Build_Compact_Evaluated_Session(session_item)
            for session_item in updated_chain
        ],
    }
    compact_item.update(extract_user_id_fields(persona_item))
    return compact_item


def initialize_conflict_summary(conflict_type: str, top_k: int = WHITE_BOX_TOP_K) -> Dict[str, Any]:
    metric_keys = get_metric_keys(conflict_type, top_k)
    return {
        "Question_Count": 0,
        "Metric_Sums": {metric_key: 0.0 for metric_key in metric_keys},
    }


def accumulate_question_metrics(conflict_summary: Dict[str, Any], evaluation_result: Dict[str, Any]):
    metrics = evaluation_result.get("Metrics", {})
    conflict_summary["Question_Count"] += 1

    for metric_key in conflict_summary["Metric_Sums"].keys():
        metric_value = float(metrics.get(metric_key, 0) or 0)
        conflict_summary["Metric_Sums"][metric_key] += metric_value


def build_conflict_metric_summary(conflict_type: str, conflict_summary: Dict[str, Any], top_k: int = WHITE_BOX_TOP_K) -> Dict[str, Any]:
    schema = METRIC_SCHEMAS[conflict_type]
    white_box_config = get_white_box_metric_config(conflict_type, top_k)
    metric_averages = {}
    denominator_info = {}

    for metric_key in get_metric_keys(conflict_type, top_k):
        denominator = conflict_summary["Question_Count"]
        metric_averages[metric_key] = safe_divide(conflict_summary["Metric_Sums"].get(metric_key, 0.0), denominator)
        denominator_info[metric_key] = denominator

    return {
        "Question_Count": conflict_summary["Question_Count"],
        "Black_Box_Metrics": {
            metric_key: metric_averages[metric_key] for metric_key in schema["black_box_metrics"]
        },
        "White_Box_Metrics": {
            metric_key: metric_averages[metric_key]
            for metric_key in [
                white_box_config["hit_metric"],
                white_box_config["log_rank_metric"],
            ]
        },
        "Metric_Denominators": denominator_info,
        "Rank_Metric_Applicable_Counts": {},
    }


def build_session_evaluation_summary(session_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    conflict_aggregates = {
        conflict_type: initialize_conflict_summary(conflict_type)
        for conflict_type in METRIC_SCHEMAS.keys()
    }
    judge_method_statistics: Dict[str, int] = {}
    error_tag_statistics: Dict[str, int] = {}

    for question_item in session_questions:
        evaluation_result = question_item.get("Evaluation_Result", {})
        conflict_type = question_item.get("conflict_type", "unknown")
        if conflict_type not in conflict_aggregates:
            continue
        accumulate_question_metrics(conflict_aggregates[conflict_type], evaluation_result)

        judge_method = evaluation_result.get("Judge_Method", "unknown")
        judge_method_statistics[judge_method] = judge_method_statistics.get(judge_method, 0) + 1
        update_error_tag_statistics(error_tag_statistics, evaluation_result.get("Error_Tags", []))

    summary = {
        "Question_Count": len(session_questions),
        "White_Box_Top_K": WHITE_BOX_TOP_K,
        "Judge_Method_Statistics": judge_method_statistics,
        "Error_Tag_Statistics": error_tag_statistics,
    }

    for conflict_type, conflict_summary in conflict_aggregates.items():
        summary[METRIC_SCHEMAS[conflict_type]["summary_name"]] = build_conflict_metric_summary(
            conflict_type,
            conflict_summary,
        )
    return summary


def build_white_box_summary_by_k(question_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary_by_k: Dict[str, Any] = {}
    for top_k in WHITE_BOX_TOP_K_VALUES:
        conflict_aggregates = {
            conflict_type: initialize_conflict_summary(conflict_type, top_k)
            for conflict_type in METRIC_SCHEMAS.keys()
        }

        for question_item in question_items:
            conflict_type = question_item.get("conflict_type", "unknown")
            if conflict_type not in conflict_aggregates:
                continue
            white_box_by_k = (
                question_item.get("Evaluation_Result", {})
                .get("White_Box_By_K", {})
                .get(str(top_k))
            )
            if not isinstance(white_box_by_k, dict):
                continue

            metrics = white_box_by_k.get("Metrics", {})
            aggregate = conflict_aggregates[conflict_type]
            aggregate["Question_Count"] += 1

            for metric_key in aggregate["Metric_Sums"].keys():
                metric_value = float(metrics.get(metric_key, 0) or 0)
                aggregate["Metric_Sums"][metric_key] += metric_value

        summary_by_k[str(top_k)] = {
            "White_Box_Top_K": top_k,
            "Conflict_Summaries": {
                METRIC_SCHEMAS[conflict_type]["summary_name"]: {
                    "Question_Count": conflict_summary["Question_Count"],
                    "White_Box_Metrics": {
                        metric_key: safe_divide(
                            conflict_summary["Metric_Sums"].get(metric_key, 0.0),
                            conflict_summary["Question_Count"],
                        )
                        for metric_key in conflict_summary["Metric_Sums"].keys()
                    },
                    "Metric_Denominators": {
                        metric_key: conflict_summary["Question_Count"]
                        for metric_key in conflict_summary["Metric_Sums"].keys()
                    },
                    "Rank_Metric_Applicable_Counts": {},
                }
                for conflict_type, conflict_summary in conflict_aggregates.items()
            },
        }
    return summary_by_k


def Generate_Single_Persona_Evaluation(
    persona_item: Dict[str, Any],
    prediction_fields: List[str],
    enable_llm_judge: bool = True,
):
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
            "note": "Conflict-type black-box and white-box evaluation",
        }

        persona_aggregates = {
            conflict_type: initialize_conflict_summary(conflict_type)
            for conflict_type in METRIC_SCHEMAS.keys()
        }
        judge_method_statistics: Dict[str, int] = {}
        error_tag_statistics: Dict[str, int] = {}
        evaluated_session_count = 0
        evaluated_question_count = 0

        for current_idx, session_item in enumerate(full_session_chain):
            print(f"[DEBUG] Evaluating session {current_idx + 1}/{len(full_session_chain)}")
            session_questions = session_item.get("Session_Questions", [])

            if not isinstance(session_questions, list) or len(session_questions) == 0:
                session_item["Session_Evaluation"] = {
                    "Question_Count": 0,
                    "White_Box_Top_K": WHITE_BOX_TOP_K,
                    "White_Box_Metrics_By_K": {},
                    "Judge_Method_Statistics": {},
                    "Error_Tag_Statistics": {},
                }
                full_session_chain[current_idx] = place_session_evaluation_before_event_types(session_item)
                continue

            evaluated_session_count += 1

            for q_idx, question_item in enumerate(session_questions):
                conflict_type = question_item.get("conflict_type", "unknown")
                if conflict_type not in METRIC_SCHEMAS:
                    raise ValueError(f"Unsupported conflict_type: {conflict_type}")

                judged_result, judge_cost = Evaluate_Single_Question(
                    question_item=question_item,
                    prediction_fields=prediction_fields,
                    enable_llm_judge=enable_llm_judge,
                    top_k=MAX_WHITE_BOX_TOP_K,
                )
                support_rank_field = WHITE_BOX_METRIC_CONFIGS[conflict_type]["support_rank_field"]
                global_support_rank = int(
                    judged_result.get("White_Box_Metadata", {}).get(support_rank_field, 0) or 0
                )
                white_box_by_k = build_white_box_result_by_k(
                    conflict_type=conflict_type,
                    global_support_rank=global_support_rank,
                    judge_method=judged_result.get("Judge_Method"),
                )
                primary_white_box_config = get_white_box_metric_config(conflict_type, WHITE_BOX_TOP_K)
                primary_white_box_metrics = white_box_by_k[str(WHITE_BOX_TOP_K)]["Metrics"]
                primary_support_rank = white_box_by_k[str(WHITE_BOX_TOP_K)]["Support_Rank"]

                evaluation_result = copy.deepcopy(judged_result)
                evaluation_result["Metrics"] = {
                    **{
                        metric_key: judged_result.get("Metrics", {}).get(metric_key, 0)
                        for metric_key in METRIC_SCHEMAS[conflict_type]["black_box_metrics"]
                    },
                    **primary_white_box_metrics,
                }
                evaluation_result["White_Box_Metadata"] = {
                    primary_white_box_config["support_rank_field"]: primary_support_rank,
                }
                evaluation_result["White_Box_Top_K"] = WHITE_BOX_TOP_K
                evaluation_result["White_Box_By_K"] = white_box_by_k
                session_questions[q_idx]["Evaluation_Result"] = evaluation_result

                current_stage_total_cost["input_tokens"] += judge_cost.get("input_tokens", 0) or 0
                current_stage_total_cost["output_tokens"] += judge_cost.get("output_tokens", 0) or 0
                current_stage_total_cost["total_tokens"] += judge_cost.get("total_tokens", 0) or 0
                current_stage_total_cost["total_cost_usd"] += judge_cost.get("total_cost_usd", 0.0) or 0.0
                if current_stage_total_cost["model"] is None:
                    current_stage_total_cost["model"] = judge_cost.get("model")
                if judge_cost.get("pricing_available") is True:
                    current_stage_total_cost["pricing_available"] = True

                if conflict_type in persona_aggregates:
                    accumulate_question_metrics(persona_aggregates[conflict_type], evaluation_result)

                judged_by = evaluation_result.get("Judge_Method", "unknown")
                judge_method_statistics[judged_by] = judge_method_statistics.get(judged_by, 0) + 1
                update_error_tag_statistics(error_tag_statistics, evaluation_result.get("Error_Tags", []))
                evaluated_question_count += 1

            session_item["Session_Questions"] = session_questions
            session_item["Session_Evaluation"] = build_session_evaluation_summary(session_questions)
            session_item["Session_Evaluation"]["White_Box_Metrics_By_K"] = build_white_box_summary_by_k(session_questions)
            session_metadata = session_item.get("Session_Memory_Metadata", {})
            if isinstance(session_metadata, dict):
                session_item["Session_Evaluation"]["Runtime_Summary"] = {
                    "Add_Duration_ms": float(session_metadata.get("Add_Duration_ms", 0.0) or 0.0),
                    "Session_Retrieval_Time_ms": float(session_metadata.get("Session_Retrieval_Time_ms", 0.0) or 0.0),
                    "Session_Response_Time_ms": float(session_metadata.get("Session_Response_Time_ms", 0.0) or 0.0),
                    "Session_Total_Runtime_ms": float(session_metadata.get("Session_Total_Runtime_ms", 0.0) or 0.0),
                }
            full_session_chain[current_idx] = place_session_evaluation_before_event_types(session_item)

        runtime_summary = build_runtime_summary_from_sessions(full_session_chain)
        existing_runtime_summary = extract_source_runtime_summary(persona_item)
        source_observable_token_cost_summary = persona_item.get("Observable_Token_Cost_Summary")
        observable_token_cost_summary = Build_Observable_Token_Cost_Summary(
            stage_cost=current_stage_total_cost,
            stage_name="evaluation_judge",
        )
        evaluation_summary = {
            "Evaluated_Session_Count": evaluated_session_count,
            "Evaluated_Question_Count": evaluated_question_count,
            "White_Box_Top_K": WHITE_BOX_TOP_K,
            "White_Box_Top_K_Values": WHITE_BOX_TOP_K_VALUES,
            "Judge_Method_Statistics": judge_method_statistics,
            "Error_Tag_Statistics": error_tag_statistics,
            "Runtime_Summary": runtime_summary,
            "Observable_Token_Cost_Summary": observable_token_cost_summary,
            "White_Box_Metrics_By_K": build_white_box_summary_by_k(
                [
                    question_item
                    for session_item in full_session_chain
                    for question_item in session_item.get("Session_Questions", [])
                    if isinstance(question_item, dict)
                ]
            ),
        }
        if isinstance(existing_runtime_summary, dict):
            evaluation_summary["Source_Runtime_Summary"] = existing_runtime_summary
        if isinstance(source_observable_token_cost_summary, dict):
            evaluation_summary["Source_Observable_Token_Cost_Summary"] = source_observable_token_cost_summary
            memory_system_name = str(persona_item.get("Memory_System", "memory_system") or "memory_system").lower()
            evaluation_summary["Combined_Observable_Token_Cost_Summary"] = {
                "Stage_Name": f"{memory_system_name}_answer_generation_plus_evaluation_judge",
                "Input_Tokens": (source_observable_token_cost_summary.get("Input_Tokens", 0) or 0) + (observable_token_cost_summary.get("Input_Tokens", 0) or 0),
                "Output_Tokens": (source_observable_token_cost_summary.get("Output_Tokens", 0) or 0) + (observable_token_cost_summary.get("Output_Tokens", 0) or 0),
                "Total_Tokens": (source_observable_token_cost_summary.get("Total_Tokens", 0) or 0) + (observable_token_cost_summary.get("Total_Tokens", 0) or 0),
                "Total_Cost_USD": (source_observable_token_cost_summary.get("Total_Cost_USD", 0.0) or 0.0) + (observable_token_cost_summary.get("Total_Cost_USD", 0.0) or 0.0),
                "Model": {
                    "answer_generation": source_observable_token_cost_summary.get("Model"),
                    "evaluation_judge": observable_token_cost_summary.get("Model"),
                },
                "Pricing_Available": bool(
                    source_observable_token_cost_summary.get("Pricing_Available")
                    or observable_token_cost_summary.get("Pricing_Available")
                ),
            }

        for conflict_type, conflict_summary in persona_aggregates.items():
            evaluation_summary[METRIC_SCHEMAS[conflict_type]["summary_name"]] = build_conflict_metric_summary(
                conflict_type,
                conflict_summary,
            )

        final_cost = calculate_cumulative_cost(previous_cost, current_stage_total_cost)
        return full_session_chain, evaluation_summary, final_cost

    except Exception as e:
        print(f"[DEBUG] Generate_Single_Persona_Evaluation failed: {e}:{traceback.format_exc()}")
        raise


def Build_Evaluation_Result_For_Persona(
    persona_item: Dict[str, Any],
    prediction_fields: List[str],
    enable_llm_judge: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    persona_copy = copy.deepcopy(persona_item)
    updated_chain, evaluation_summary, final_cost = Generate_Single_Persona_Evaluation(
        persona_item=persona_copy,
        prediction_fields=prediction_fields,
        enable_llm_judge=enable_llm_judge,
    )
    result_item = Build_Compact_Evaluation_Result_Item(
        persona_item=persona_copy,
        updated_chain=updated_chain,
        evaluation_summary=evaluation_summary,
        final_cost=final_cost,
    )
    return result_item, evaluation_summary


def write_full_json_results(output_perfect_file: str, all_results: List[Dict[str, Any]]) -> None:
    with open(output_perfect_file, "w", encoding="utf-8") as f:
        if len(all_results) == 1:
            json.dump(all_results[0], f, ensure_ascii=False, indent=4)
        else:
            json.dump(all_results, f, ensure_ascii=False, indent=4)


def Generate_User_Evaluation(args):
    print(f"Processing file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    try:
        prediction_fields = [item.strip() for item in args.prediction_fields.split(",") if item.strip()]
        parallel_workers = max(1, int(getattr(args, "parallel_workers", 1) or 1))
        print(f"[DEBUG] Prediction fields: {prediction_fields}")
        print(f"[DEBUG] Enable LLM judge: {args.enable_llm_judge}")
        print(f"[DEBUG] Parallel workers: {parallel_workers}")
        print(f"[DEBUG] Primary white-box Top-K: {WHITE_BOX_TOP_K}")
        print(f"[DEBUG] Judge-once white-box Top-K: {MAX_WHITE_BOX_TOP_K}")

        all_personas = load_jsonl_items(args.input_file)
        print(f"[DEBUG] Read {len(all_personas)} personas")

        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write("")

        output_perfect_dir = os.path.dirname(args.output_perfect_file)
        if output_perfect_dir:
            os.makedirs(output_perfect_dir, exist_ok=True)

        if parallel_workers <= 1 or len(all_personas) <= 1:
            all_results = []
            for idx, persona_item in enumerate(all_personas):
                print(f"[DEBUG] Processing persona {idx + 1}/{len(all_personas)}")
                result_item, evaluation_summary = Build_Evaluation_Result_For_Persona(
                    persona_item=persona_item,
                    prediction_fields=prediction_fields,
                    enable_llm_judge=args.enable_llm_judge,
                )
                all_results.append(result_item)
                append_jsonl_item(args.output_file, result_item)
                write_full_json_results(args.output_perfect_file, all_results)

                print(
                    f"[DEBUG] Persona {idx + 1} completed - "
                    f"Evaluated questions: {evaluation_summary['Evaluated_Question_Count']}"
                )
        else:
            worker_count = min(parallel_workers, len(all_personas))
            print(f"[DEBUG] Running persona evaluation with {worker_count} threads")
            all_results_by_index: List[Optional[Dict[str, Any]]] = [None] * len(all_personas)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_index = {
                    executor.submit(
                        Build_Evaluation_Result_For_Persona,
                        persona_item,
                        prediction_fields,
                        args.enable_llm_judge,
                    ): idx
                    for idx, persona_item in enumerate(all_personas)
                }

                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    print(f"[DEBUG] Collecting persona {idx + 1}/{len(all_personas)}")
                    result_item, evaluation_summary = future.result()
                    all_results_by_index[idx] = result_item
                    completed_results = [item for item in all_results_by_index if item is not None]
                    append_jsonl_item(args.output_file, result_item)
                    write_full_json_results(args.output_perfect_file, completed_results)

                    print(
                        f"[DEBUG] Persona {idx + 1} completed - "
                        f"Evaluated questions: {evaluation_summary['Evaluated_Question_Count']}"
                    )

        print("[DEBUG] Successfully processed conflict-type evaluation")
        return True
    except Exception as e:
        print(f"Error processing evaluation: {e}:{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conflict-focused evaluation for MemConflict.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/home/taoz/Mem_Conflict/MemConflict/Evaluation/Results/memzero_results.jsonl",
        help="Input JSONL file containing questions and model answers",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/taoz/Mem_Conflict/MemConflict/Evaluation/Scores/memzero_eval_scores.jsonl",
        help="Output JSONL file for evaluation results",
    )
    parser.add_argument(
        "--output_perfect_file",
        type=str,
        default="/home/taoz/Mem_Conflict/MemConflict/Evaluation/Scores/memzero_eval_scores.json",
        help="Output JSON file for evaluation results",
    )
    parser.add_argument(
        "--prediction_fields",
        type=str,
        default="Model_Answer,Predicted_Answer,Generated_Answer,memory_answer,model_answer,predicted_answer",
        help="Comma-separated candidate fields that may store the model answer",
    )
    parser.add_argument(
        "--disable_llm_judge",
        action="store_true",
        help="Disable LLM judge and use rule-based scoring only",
    )
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=int(os.getenv("EVAL_SCORING_PARALLEL_WORKERS", "1")),
        help="Number of persona-level worker threads for LLM-judge scoring. Default: 1.",
    )
    args = parser.parse_args()
    args.enable_llm_judge = not args.disable_llm_judge
    Generate_User_Evaluation(args)
