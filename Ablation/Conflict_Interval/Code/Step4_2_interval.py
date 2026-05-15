import argparse
import copy
import json
import jsonlines
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

PROMPT_PATH = PROJECT_DIR / 'Prompt' / 'Prompt4_2.txt'
with PROMPT_PATH.open('r', encoding='utf-8') as f:
    STEP4_2_PROMPT = f.read()


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


def resolve_input_path(interval_mode: str, input_file: str | None = None) -> str:
    if input_file:
        return input_file
    base_dir = PROJECT_DIR / 'Ablation' / 'Conflict_Interval' / 'Data'
    return str(base_dir / f'Step4_1_{interval_mode}_interval.jsonl')


def resolve_output_paths(interval_mode: str, output_file: str | None = None, output_json_file: str | None = None) -> Tuple[str, str]:
    suffix = f"{interval_mode}_interval"
    base_dir = PROJECT_DIR / 'Ablation' / 'Conflict_Interval' / 'Data'
    resolved_output_file = output_file or str(base_dir / f'Step4_2_{suffix}.jsonl')
    resolved_output_json = output_json_file or str(base_dir / f'Step4_2_{suffix}.json')
    return resolved_output_file, resolved_output_json


def build_session_dialogue_input(session_item: Dict[str, Any]) -> Dict[str, Any]:
    session_type = session_item.get('Session_Type')

    dialogue_input = {
        'Session_Type': session_type,
        'Event_Types': copy.deepcopy(session_item.get('Event_Types', [])),
        'Session_Outline': session_item.get('Session_Outline', ''),
    }

    if session_type == 'initial_reveal':
        dialogue_input['Main_Session_Information'] = {
            'Revealed_Attributes': copy.deepcopy(session_item.get('Revealed_Attributes', {}))
        }
    elif session_type == 'update':
        raw_updated_attributes = session_item.get('Updated_Attributes', [])
        simplified_updated_attributes = []
        for update_item in raw_updated_attributes:
            if isinstance(update_item, dict):
                simplified_updated_attributes.append({
                    'Attribute': update_item.get('Attribute'),
                    'After': copy.deepcopy(update_item.get('After')),
                })
        dialogue_input['Main_Session_Information'] = {
            'Updated_Attributes': simplified_updated_attributes
        }
    else:
        dialogue_input['Main_Session_Information'] = {}

    additional_information: Dict[str, Any] = {}

    static_conflict_information = session_item.get('Static_Conflict_Information', [])
    simplified_static_conflict = []
    for item in static_conflict_information:
        if item.get('Role') == 'Distractor' or item.get('Source_Person_ID'):
            simplified_static_conflict.append({
                'Source_Person_ID': item.get('Source_Person_ID'),
                'Relationship_To_User': item.get('Relationship_To_User'),
                'Target_Field_Path': item.get('Target_Field_Path'),
                'Value': item.get('Value'),
            })
        else:
            simplified_static_conflict.append({
                'Target_Field_Path': item.get('Target_Field_Path'),
                'Value': item.get('Value'),
            })
    if simplified_static_conflict:
        additional_information['Static_Conflict_Information'] = simplified_static_conflict

    conditional_conflict_information = session_item.get('Conditional_Conflict_Information', [])
    simplified_conditional_conflict = []
    for item in conditional_conflict_information:
        if item.get('Role') == 'Distractor' or item.get('Source_Person_ID'):
            simplified_conditional_conflict.append({
                'Source_Person_ID': item.get('Source_Person_ID'),
                'Relationship_To_User': item.get('Relationship_To_User'),
                'Preference_Key': item.get('Preference_Key'),
                'Preference_Description': item.get('Preference_Description'),
            })
        else:
            simplified_conditional_conflict.append({
                'Preference_Type': item.get('Preference_Type'),
                'Item': item.get('Item'),
                'Condition': item.get('Condition'),
            })
    if simplified_conditional_conflict:
        additional_information['Conditional_Conflict_Information'] = simplified_conditional_conflict

    others_dynamic_information = session_item.get('Others_Dynamic_Information', [])
    simplified_others_dynamic = []
    for item in others_dynamic_information:
        if item.get('Role') == 'Distractor' or item.get('Source_Person_ID'):
            simplified_others_dynamic.append({
                'Source_Person_ID': item.get('Source_Person_ID'),
                'Relationship_To_User': item.get('Relationship_To_User'),
                'Attribute': item.get('Attribute'),
                'Value': item.get('Value'),
            })
    if simplified_others_dynamic:
        additional_information['Others_Dynamic_Information'] = simplified_others_dynamic

    dialogue_input['Additional_Information_To_Mention'] = additional_information
    return dialogue_input


def generate_session_dialogue(dialogue_input: Dict[str, Any],
                              previous_cost: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    try:
        print('[DEBUG] Sending session dialogue generation request to LLM...')

        target_turn_num = random.randint(40, 50)
        print('target_turn_num', target_turn_num)

        user_prompt = (
            f'Generate one complete session dialogue with exactly {target_turn_num} dialogue turns.\n\n'
            'Input data:\n'
            f'{json.dumps(dialogue_input, ensure_ascii=False, indent=2)}'
        )

        json_markers = [
            'Corrected fixed part', 'Corrected persona', 'Corrected JSON',
            'Final JSON', 'Complete JSON', 'Correction result'
        ]

        dialogue_result, cost_info = llm_request(
            STEP4_2_PROMPT,
            user_prompt,
            return_parsed_json=True,
            json_markers=json_markers,
        )
        cost_info = calculate_cumulative_cost(previous_cost, cost_info)

        print('[DEBUG] Successfully generated session dialogue with LLM')
        return dialogue_result, cost_info

    except Exception as e:
        print(f'[DEBUG] Session dialogue generation failed: {e}:{traceback.format_exc()}')
        raise


def generate_single_session_dialogues(persona_item: Dict[str, Any],
                                      previous_cost: Dict[str, Any] | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    try:
        print('[DEBUG] Step 4.2 ---- Generating full session dialogues...')

        full_session_chain = copy.deepcopy(persona_item['Full_Session_Chain'])
        cost_info = previous_cost

        for session_idx, session_item in enumerate(full_session_chain):
            print(f'[DEBUG] Processing session dialogue {session_idx + 1}/{len(full_session_chain)}')
            dialogue_input = build_session_dialogue_input(session_item=session_item)
            session_dialogue, cost_info = generate_session_dialogue(
                dialogue_input=dialogue_input,
                previous_cost=cost_info,
            )
            session_item['Session_Dialogue'] = session_dialogue

        print('[DEBUG] Step 4.2 ---- Session dialogues generated successfully.')
        return full_session_chain, cost_info

    except Exception as e:
        print(f'[DEBUG] generate_single_session_dialogues failed: {e}:{traceback.format_exc()}')
        raise


def run_single_mode(args: argparse.Namespace, interval_mode: str) -> bool:
    input_file = resolve_input_path(interval_mode, args.input_file)
    output_file, output_json_file = resolve_output_paths(interval_mode, args.output_file, args.output_json_file)

    print(f'Processing file: {input_file}')
    print(f'Output file: {output_file}')
    print(f'Output JSON file: {output_json_file}')
    print(f'[DEBUG] Interval mode: {interval_mode}')

    try:
        all_personas = load_jsonl_items(input_file)
        results: List[Dict[str, Any]] = []

        for idx, persona_item in enumerate(all_personas, start=1):
            print(f"[DEBUG] Processing persona {idx}/{len(all_personas)}: {persona_item.get('ID')}")
            try:
                previous_cost = persona_item.get('token_cost')
                updated_full_session_chain, cost_info = generate_single_session_dialogues(
                    persona_item=persona_item,
                    previous_cost=previous_cost,
                )

                result_item = copy.deepcopy(persona_item)
                result_item['Full_Session_Chain'] = updated_full_session_chain
                result_item['token_cost'] = cost_info
                results.append(result_item)
            except Exception as e:
                print(f"[DEBUG] Failed to process persona {idx}: {e}:{traceback.format_exc()}")
                continue

        write_jsonl_items(output_file, results)
        write_json_items(output_json_file, results)
        print(f'[DEBUG] Successfully generated Step4_2 interval ablation ({interval_mode}) with {len(results)} personas.')
        return True
    except Exception as e:
        print(f'[DEBUG] Step4_2 interval ablation failed: {e}:{traceback.format_exc()}')
        return False


def main(args: argparse.Namespace) -> bool:
    if args.interval_mode == 'both':
        short_ok = run_single_mode(args, 'short')
        long_ok = run_single_mode(args, 'long')
        return short_ok and long_ok
    return run_single_mode(args, args.interval_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conflict interval ablation: Step4_2 full session dialogue generation.')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input Step4_1 interval JSONL file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSONL file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--output_json_file', type=str, default=None,
                        help='Output JSON file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--interval_mode', type=str, choices=['short', 'long', 'both'], default='both',
                        help='Conflict interval mode for the first ablation version')
    args = parser.parse_args()
    main(args)
