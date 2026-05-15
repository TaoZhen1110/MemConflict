import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jsonlines


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


def build_persona_index(step4_4_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for item in step4_4_items:
        persona_id = str(item.get('ID', '')).strip()
        if persona_id:
            index[persona_id] = item
    return index


def build_session_question_index(full_session_chain: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    index: Dict[int, Dict[str, Any]] = {}
    for session in full_session_chain:
        session_id = session.get('Session_ID')
        if isinstance(session_id, int):
            index[session_id] = session
    return index


def copy_question_fields(target_session: Dict[str, Any], source_session: Dict[str, Any]) -> Tuple[bool, int]:
    question_fields = ['Question_Trigger_Types', 'Session_Questions', 'Session_Question_Count']
    copied_any = False
    copied_count = 0
    for field in question_fields:
        if field in source_session:
            target_session[field] = copy.deepcopy(source_session.get(field))
            copied_any = True
    copied_count = int(target_session.get('Session_Question_Count', 0) or 0)
    return copied_any, copied_count


def transform_persona(no_interference_item: Dict[str, Any], original_step4_4_item: Dict[str, Any]) -> Dict[str, Any]:
    result_item = copy.deepcopy(no_interference_item)
    source_chain = original_step4_4_item.get('Full_Session_Chain', [])
    source_session_index = build_session_question_index(source_chain)

    copied_session_count = 0
    copied_question_count = 0
    missing_session_ids: List[int] = []

    for session in result_item.get('Full_Session_Chain', []):
        session_id = session.get('Session_ID')
        if not isinstance(session_id, int):
            continue
        source_session = source_session_index.get(session_id)
        if source_session is None:
            missing_session_ids.append(session_id)
            session['Question_Trigger_Types'] = []
            session['Session_Questions'] = []
            session['Session_Question_Count'] = 0
            continue

        copied_any, copied_count = copy_question_fields(session, source_session)
        if copied_any:
            copied_session_count += 1
            copied_question_count += copied_count

    result_item['Total_Session_Question_Count'] = copy.deepcopy(original_step4_4_item.get('Total_Session_Question_Count', copied_question_count))
    result_item['Triggered_Session_Count'] = copy.deepcopy(original_step4_4_item.get('Triggered_Session_Count', copied_session_count))
    result_item['No_Interference_Question_Copy_Metadata'] = {
        'Question_Source': 'original_step4_4',
        'Copied_Session_Count': copied_session_count,
        'Copied_Question_Count': copied_question_count,
        'Missing_Session_IDs': missing_session_ids,
        'Questions_Unchanged_From_Original': True,
    }
    return result_item


def main(args: argparse.Namespace) -> bool:
    print(f'No-interference Step4_3 input: {args.input_file}')
    print(f'Original Step4_4 input: {args.original_step4_4_file}')
    print(f'Output file: {args.output_file}')

    no_interference_items = load_jsonl_items(args.input_file)
    original_step4_4_items = load_jsonl_items(args.original_step4_4_file)
    original_index = build_persona_index(original_step4_4_items)

    results: List[Dict[str, Any]] = []
    total_copied_questions = 0

    for idx, item in enumerate(no_interference_items, start=1):
        persona_id = str(item.get('ID', '')).strip()
        print(f'[DEBUG] Processing persona {idx}/{len(no_interference_items)}: {persona_id}')
        original_item = original_index.get(persona_id)
        if original_item is None:
            raise ValueError(f'Could not find original Step4_4 persona for ID={persona_id}')

        transformed = transform_persona(item, original_item)
        total_copied_questions += transformed.get('No_Interference_Question_Copy_Metadata', {}).get('Copied_Question_Count', 0)
        results.append(transformed)

    write_jsonl_items(args.output_file, results)
    write_json_items(args.output_perfect_file, results)

    print('[DEBUG] Successfully copied original Step4_4 questions into no-interference branch.')
    print(f'[DEBUG] Persona count: {len(results)}, total copied questions: {total_copied_questions}')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build Step4_4 no-interference file by copying original Step4_4 questions.')
    parser.add_argument('--input_file', type=str,
                        default=r'/home/taoz/Mem_Conflict/MemConflict/Ablation/No_Interference/Data/Step4_3_no_interference.jsonl',
                        help='Input no-interference Step4_3 JSONL file')
    parser.add_argument('--original_step4_4_file', type=str,
                        default=r'/home/taoz/Mem_Conflict/MemConflict/Data/Step4_4.jsonl',
                        help='Original Step4_4 JSONL file to copy questions from')
    parser.add_argument('--output_file', type=str,
                        default=r'/home/taoz/Mem_Conflict/MemConflict/Ablation/No_Interference/Data/Step4_4_no_interference.jsonl',
                        help='Output JSONL file')
    parser.add_argument('--output_perfect_file', type=str,
                        default=r'/home/taoz/Mem_Conflict/MemConflict/Ablation/No_Interference/Data/Step4_4_no_interference.json',
                        help='Output JSON file')
    args = parser.parse_args()
    main(args)
