import argparse
import copy
import json
import jsonlines
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_FILE = Path(__file__).resolve()
LOCAL_DIR = THIS_FILE.parent
PROJECT_DIR = THIS_FILE.parents[2]
CODE_DIR = PROJECT_DIR / 'Code'
if str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from llm_request import llm_request, calculate_cumulative_cost


SYSTEM_PROMPT = """You rewrite benchmark questions from a direct style into a more indirect but still clear style.

Rules:
1. Rewrite only the question text.
2. Do not change the meaning.
3. Do not add new facts.
4. Do not mention session, point labels, distractor, benchmark, memory system, or evaluation.
5. Keep each question answerable and target-specific.
6. Preserve question_id exactly.
7. Return valid JSON only.

Output format:
{
  \"rewritten_questions\": [
    {\"question_id\": \"Q_001\", \"question\": \"...\"}
  ]
}
"""


def load_jsonl_items(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items.append(item)
    return items


def write_jsonl_items(path: str, items: List[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(out_path), 'w') as writer:
        for item in items:
            writer.write(item)


def write_json_items(path: str, items: List[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def build_rewrite_payload(question_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for item in question_items:
        payload.append({
            'question_id': item.get('question_id'),
            'question': item.get('question'),
            'conflict_type': item.get('conflict_type'),
        })
    return payload


def parse_rewrite_json_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None

    text = raw_text.strip()
    if '```json' in text:
        start = text.find('```json') + len('```json')
        end = text.rfind('```')
        if end > start:
            text = text[start:end].strip()
    elif '```' in text:
        start = text.find('```') + len('```')
        end = text.rfind('```')
        if end > start:
            text = text[start:end].strip()

    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        return None

    candidate = match.group(0).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def validate_rewrite(question_items: List[Dict[str, Any]], llm_output: Any) -> Optional[Dict[str, str]]:
    if not isinstance(llm_output, dict):
        return None
    rewritten_questions = llm_output.get('rewritten_questions')
    if not isinstance(rewritten_questions, list):
        return None

    expected_ids = [str(item.get('question_id', '')).strip() for item in question_items]
    got_ids: List[str] = []
    rewrite_map: Dict[str, str] = {}

    for item in rewritten_questions:
        if not isinstance(item, dict):
            return None
        qid = str(item.get('question_id', '')).strip()
        question = str(item.get('question', '')).strip()
        if not qid or not question:
            return None
        got_ids.append(qid)
        rewrite_map[qid] = question

    if got_ids != expected_ids:
        return None
    return rewrite_map


def rewrite_session_questions(question_items: List[Dict[str, Any]], previous_cost: Optional[Dict[str, Any]], model: str) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    payload = build_rewrite_payload(question_items)
    user_prompt = (
        'Rewrite the following questions into a more indirect style while keeping them equally answerable. '\
        'Return JSON only.\n\n'
        f'{json.dumps({"questions": payload}, ensure_ascii=False, indent=2)}'
    )

    raw_output, cost_info = llm_request(
        SYSTEM_PROMPT,
        user_prompt,
        model=model,
        return_parsed_json=False,
        extract_json=False,
        max_tokens=2048,
        temperature=0.2,
    )
    cumulative_cost = calculate_cumulative_cost(previous_cost, cost_info)
    llm_output = parse_rewrite_json_from_text(raw_output)
    rewrite_map = validate_rewrite(question_items, llm_output)

    rewritten_questions = []
    if rewrite_map is None:
        for item in question_items:
            updated = copy.deepcopy(item)
            original_question = str(updated.get('question', '')).strip()
            updated['Original_Question'] = original_question
            updated['Question_Style'] = 'indirect'
            updated['Rewrite_Method'] = 'fallback_original'
            rewritten_questions.append(updated)
        return rewritten_questions, cumulative_cost, False

    for item in question_items:
        updated = copy.deepcopy(item)
        original_question = str(updated.get('question', '')).strip()
        updated['Original_Question'] = original_question
        updated['question'] = rewrite_map[str(updated.get('question_id'))]
        updated['Question_Style'] = 'indirect'
        updated['Rewrite_Method'] = 'llm'
        rewritten_questions.append(updated)

    return rewritten_questions, cumulative_cost, True


def transform_persona(persona_item: Dict[str, Any], persona_idx: int, total_personas: int, model: str) -> Dict[str, Any]:
    result_item = copy.deepcopy(persona_item)
    persona_label = result_item.get('ID', '<unknown>')
    print(f"[DEBUG] Processing persona {persona_idx}/{total_personas}: {persona_label}")

    previous_cost = result_item.get('token_cost')
    cumulative_cost = previous_cost
    rewritten_count = 0
    unchanged_count = 0
    llm_session_success_count = 0
    llm_session_fallback_count = 0

    for session_item in result_item.get('Full_Session_Chain', []):
        questions = session_item.get('Session_Questions', [])
        if not isinstance(questions, list) or not questions:
            continue

        session_id = session_item.get('Session_ID', '?')
        print(f"[DEBUG]   Session {session_id}: rewriting {len(questions)} questions with LLM ({model})")
        rewritten_questions, cumulative_cost, success = rewrite_session_questions(questions, previous_cost=cumulative_cost, model=model)
        if success:
            llm_session_success_count += 1
            print(f"[DEBUG]   Session {session_id}: LLM rewrite accepted")
        else:
            llm_session_fallback_count += 1
            print(f"[DEBUG]   Session {session_id}: invalid JSON structure, kept original questions")

        session_item['Session_Questions'] = rewritten_questions
        for q in rewritten_questions:
            if q.get('question') != q.get('Original_Question'):
                rewritten_count += 1
            else:
                unchanged_count += 1

    print(
        f"[DEBUG] Persona rewrite summary - rewritten: {rewritten_count}, unchanged: {unchanged_count}, "
        f"llm_ok_sessions: {llm_session_success_count}, llm_fallback_sessions: {llm_session_fallback_count}"
    )

    result_item['Question_Style_Metadata'] = {
        'Source_Style': 'direct',
        'Target_Style': 'indirect',
        'Rewritten_Question_Count': rewritten_count,
        'Unchanged_Question_Count': unchanged_count,
        'LLM_Session_Success_Count': llm_session_success_count,
        'LLM_Session_Fallback_Count': llm_session_fallback_count,
        'Answer_Unchanged': True,
        'Conflict_Type_Unchanged': True,
        'Ability_Target_Unchanged': True,
        'Difficulty_Unchanged': True,
        'Rewrite_Method': 'basic_llm',
    }
    result_item['token_cost'] = cumulative_cost
    return result_item


def main(args: argparse.Namespace) -> bool:
    print(f'Processing file: {args.input_file}')
    print(f'Output file: {args.output_file}')
    print(f'[DEBUG] Rewrite model: {args.model}')

    personas = load_jsonl_items(args.input_file)
    transformed: List[Dict[str, Any]] = []
    total_personas = len(personas)
    for idx, item in enumerate(personas, start=1):
        transformed.append(transform_persona(item, persona_idx=idx, total_personas=total_personas, model=args.model))

    write_jsonl_items(args.output_file, transformed)
    write_json_items(args.output_perfect_file, transformed)

    total_rewritten = sum(item.get('Question_Style_Metadata', {}).get('Rewritten_Question_Count', 0) for item in transformed)
    total_unchanged = sum(item.get('Question_Style_Metadata', {}).get('Unchanged_Question_Count', 0) for item in transformed)
    print(f'[DEBUG] Rewrote {total_rewritten} questions into indirect style across {len(transformed)} personas.')
    print(f'[DEBUG] Left {total_unchanged} questions unchanged due to fallback.')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build indirect-style question variant from Step4_4 dataset.')
    parser.add_argument('--input_file', type=str,
                        default=r'/home/taoz/Mem_Conflict/MemConflict/Data/Step4_4.jsonl',
                        help='Input Step4_4-style JSONL file')
    parser.add_argument('--output_file', type=str,
                        default=r'/home/taoz/Mem_Conflict/MemConflict/Ablation/Question_Style/Data/Step4_4_indirect.jsonl',
                        help='Output JSONL file for indirect question style')
    parser.add_argument('--output_perfect_file', type=str,
                        default=r'/home/taoz/Mem_Conflict/MemConflict/Ablation/Question_Style/Data/Step4_4_indirect.json',
                        help='Output JSON file for indirect question style')
    parser.add_argument('--model', type=str, default=None,
                        help='LLM model used for question rewriting; defaults to OPENAI_MODEL from environment')
    args = parser.parse_args()
    main(args)
