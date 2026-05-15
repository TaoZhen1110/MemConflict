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

PROMPT_PATH = PROJECT_DIR / 'Prompt' / 'Prompt4_1.txt'
with PROMPT_PATH.open('r', encoding='utf-8') as f:
    STEP4_1_PROMPT = f.read()


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
    return str(base_dir / f'Step3_3_{interval_mode}_interval.jsonl')


def resolve_output_paths(interval_mode: str, output_file: str | None = None, output_json_file: str | None = None) -> Tuple[str, str]:
    suffix = f"{interval_mode}_interval"
    base_dir = PROJECT_DIR / 'Ablation' / 'Conflict_Interval' / 'Data'
    resolved_output_file = output_file or str(base_dir / f'Step4_1_{suffix}.jsonl')
    resolved_output_json = output_json_file or str(base_dir / f'Step4_1_{suffix}.json')
    return resolved_output_file, resolved_output_json


def assign_session_event_types(session_item: Dict[str, Any]) -> List[str]:
    session_type = session_item.get('Session_Type')
    event_types: List[str] = []

    initial_attr_to_event = {
        'Residence': 'Talking_About_Current_Residence',
        'Work_Status': 'Talking_About_Current_Work_Rhythm',
        'Career_Status': 'Talking_About_Current_Career',
        'Health_Status': 'Talking_About_Current_Health',
        'Social_Status': 'Talking_About_Recent_Social_Life',
        'Marital_Status': 'Talking_About_Current_Relationship',
        'Children_Status': 'Talking_About_Current_Family_Life',
    }

    update_attr_to_event = {
        'Residence': 'Relocation_Update',
        'Work_Status': 'Workload_Change_Update',
        'Career_Status': 'Career_Change_Update',
        'Health_Status': 'Health_Condition_Update',
        'Social_Status': 'Social_Life_Change_Update',
        'Marital_Status': 'Relationship_Status_Change',
        'Children_Status': 'Family_Expansion_Update',
    }

    if session_type == 'initial_reveal':
        revealed_attributes = session_item.get('Revealed_Attributes', {})
        if isinstance(revealed_attributes, dict):
            for attr_name in revealed_attributes.keys():
                if attr_name in initial_attr_to_event:
                    event_types.append(initial_attr_to_event[attr_name])
        if not event_types:
            event_types.append('Initial_Life_Update')

    elif session_type == 'update':
        updated_attributes = session_item.get('Updated_Attributes', [])
        if isinstance(updated_attributes, list):
            for update_item in updated_attributes:
                attr_name = update_item.get('Attribute')
                if attr_name in update_attr_to_event:
                    event_types.append(update_attr_to_event[attr_name])
        if not event_types:
            event_types.append('General_Life_Update')

    elif session_type == 'chitchat':
        chitchat_event_pool = [
            'Casual_Catch_Up',
            'Sharing_a_Small_Daily_Story',
            'Talking_About_Food_and_Mood',
            'Weekend_Plan_Chat',
            'Weather_and_Daily_Life_Chat',
            'Remembering_an_Old_Experience',
            'Talking_About_a_Hobby',
            'Light_Complaint_or_Funny_Incident',
            'Reflecting_on_Life_Lately',
            'Talking_About_a_Recent_Show_or_Movie',
        ]
        event_types.append(random.choice(chitchat_event_pool))

    elif session_type == 'future_plan':
        event_types.append('Future_Planning_Conversation')

    else:
        event_types.append('General_Conversation')

    deduped_event_types: List[str] = []
    for event_type in event_types:
        if event_type not in deduped_event_types:
            deduped_event_types.append(event_type)
    return deduped_event_types


def build_session_outline_input(session_item: Dict[str, Any], event_types: List[str], life_goal: Any) -> Dict[str, Any]:
    session_type = session_item.get('Session_Type')

    if session_type == 'initial_reveal':
        llm_input = {
            'Session_Type': session_type,
            'Event_Types': event_types,
            'Revealed_Attributes': session_item.get('Revealed_Attributes', {}),
        }

    elif session_type == 'update':
        raw_updated_attributes = session_item.get('Updated_Attributes', [])
        simplified_updated_attributes = []
        if isinstance(raw_updated_attributes, list):
            for update_item in raw_updated_attributes:
                if not isinstance(update_item, dict):
                    continue
                simplified_updated_attributes.append({
                    'Attribute': update_item.get('Attribute'),
                    'After': copy.deepcopy(update_item.get('After')),
                })

        llm_input = {
            'Session_Type': session_type,
            'Event_Types': event_types,
            'Updated_Attributes': simplified_updated_attributes,
        }

    elif session_type == 'chitchat':
        llm_input = {
            'Session_Type': session_type,
            'Event_Types': event_types,
        }

    elif session_type == 'future_plan':
        llm_input = {
            'Session_Type': session_type,
            'Event_Types': event_types,
            'Life_Goal': life_goal,
        }

    else:
        llm_input = {
            'Session_Type': session_type,
            'Event_Types': event_types,
        }

    return llm_input


def build_session_outline(session_item: Dict[str, Any], event_types: List[str], life_goal: Any,
                          previous_cost: Dict[str, Any] | None = None) -> Tuple[str, Dict[str, Any] | None]:
    try:
        print('[DEBUG] Sending session outline generation request to LLM...')

        llm_input = build_session_outline_input(
            session_item=session_item,
            event_types=event_types,
            life_goal=life_goal,
        )

        user_prompt = (
            'Input data:\n'
            f'{json.dumps(llm_input, ensure_ascii=False, indent=2)}'
        )

        json_markers = [
            'Corrected fixed part', 'Corrected persona', 'Corrected JSON',
            'Final JSON', 'Complete JSON', 'Correction result'
        ]

        outline_result, cost_info = llm_request(
            STEP4_1_PROMPT,
            user_prompt,
            return_parsed_json=True,
            json_markers=json_markers,
        )
        cost_info = calculate_cumulative_cost(previous_cost, cost_info)

        print('[DEBUG] Successfully generated session outline with LLM')
        return outline_result['Session_Outline'], cost_info

    except Exception as e:
        print(f'[DEBUG] Session outline generation failed: {e}:{traceback.format_exc()}')
        raise


def generate_single_session_event_outlines(persona_item: Dict[str, Any],
                                           previous_cost: Dict[str, Any] | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    try:
        print('[DEBUG] Step 4.1 ---- Generating session event types and outlines...')

        full_session_chain = copy.deepcopy(persona_item['Full_Session_Chain'])
        life_goal = persona_item['Life_Goal']
        cost_info = previous_cost

        for session_idx, session_item in enumerate(full_session_chain):
            print(f'[DEBUG] Processing session {session_idx + 1}/{len(full_session_chain)}')
            event_types = assign_session_event_types(session_item=session_item)
            session_outline, cost_info = build_session_outline(
                session_item=session_item,
                event_types=event_types,
                life_goal=life_goal,
                previous_cost=cost_info,
            )
            session_item['Event_Types'] = event_types
            session_item['Session_Outline'] = session_outline

        print('[DEBUG] Step 4.1 ---- Session event types and outlines generated successfully.')
        return full_session_chain, cost_info

    except Exception as e:
        print(f'[DEBUG] generate_single_session_event_outlines failed: {e}:{traceback.format_exc()}')
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
                updated_full_session_chain, cost_info = generate_single_session_event_outlines(
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
        print(f'[DEBUG] Successfully generated Step4_1 interval ablation ({interval_mode}) with {len(results)} personas.')
        return True
    except Exception as e:
        print(f'[DEBUG] Step4_1 interval ablation failed: {e}:{traceback.format_exc()}')
        return False


def main(args: argparse.Namespace) -> bool:
    if args.interval_mode == 'both':
        short_ok = run_single_mode(args, 'short')
        long_ok = run_single_mode(args, 'long')
        return short_ok and long_ok
    return run_single_mode(args, args.interval_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conflict interval ablation: Step4_1 session event type and outline generation.')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input Step3_3 interval JSONL file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSONL file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--output_json_file', type=str, default=None,
                        help='Output JSON file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--interval_mode', type=str, choices=['short', 'long', 'both'], default='both',
                        help='Conflict interval mode for the first ablation version')
    args = parser.parse_args()
    main(args)
