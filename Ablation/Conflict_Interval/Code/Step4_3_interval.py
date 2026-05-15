import argparse
import copy
import json
import jsonlines
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

from llm_request import calculate_cumulative_cost

try:
    import tiktoken
except Exception:
    tiktoken = None


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
    return str(base_dir / f'Step4_2_{interval_mode}_interval.jsonl')


def resolve_output_paths(interval_mode: str, output_file: str | None = None, output_json_file: str | None = None) -> Tuple[str, str]:
    suffix = f"{interval_mode}_interval"
    base_dir = PROJECT_DIR / 'Ablation' / 'Conflict_Interval' / 'Data'
    resolved_output_file = output_file or str(base_dir / f'Step4_3_{suffix}.jsonl')
    resolved_output_json = output_json_file or str(base_dir / f'Step4_3_{suffix}.json')
    return resolved_output_file, resolved_output_json


def load_tokenizer(tokenizer_name: str = 'o200k_base'):
    if tiktoken is None:
        raise ImportError('tiktoken is not installed. Please install it before running Step4_3.')

    try:
        tokenizer = tiktoken.get_encoding(tokenizer_name)
        print(f'[DEBUG] Successfully loaded tiktoken tokenizer: {tokenizer_name}')
        return tokenizer
    except Exception as e:
        print(f'[DEBUG] Failed to load tokenizer: {e}:{traceback.format_exc()}')
        raise


def _extract_turn_index(turn_key: str) -> int:
    import re
    match = re.search(r'(\d+)$', turn_key)
    if match:
        return int(match.group(1))
    return 0


def extract_dialogue_messages(session_dialogue: Dict[str, Any]) -> List[Tuple[str, str]]:
    message_list: List[Tuple[str, str]] = []
    if not isinstance(session_dialogue, dict):
        return message_list

    turn_items = []
    for turn_key, turn_value in session_dialogue.items():
        if isinstance(turn_key, str) and turn_key.startswith('dialogue_turn_'):
            turn_items.append((turn_key, turn_value))
    turn_items.sort(key=lambda x: _extract_turn_index(x[0]))

    for _, turn_value in turn_items:
        if not isinstance(turn_value, list):
            continue
        for msg in turn_value:
            if not isinstance(msg, dict):
                continue
            role = msg.get('role')
            content = msg.get('content')
            if role in ['user', 'assistant'] and isinstance(content, str) and content.strip():
                message_list.append((role, content))

    return message_list


def calculate_single_session_dialogue_tokens(session_item: Dict[str, Any], tokenizer) -> int:
    session_dialogue = session_item.get('Session_Dialogue', {})
    message_list = extract_dialogue_messages(session_dialogue)

    total_tokens = 0
    for _, content in message_list:
        total_tokens += len(tokenizer.encode(content))
    return total_tokens


def build_current_stage_zero_cost() -> Dict[str, Any]:
    return {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'total_cost_usd': 0.0,
        'model': None,
        'input_cost_usd': 0.0,
        'output_cost_usd': 0.0,
        'pricing_available': False,
        'note': 'No LLM calls in this stage - token calculation only',
    }


def generate_single_dialogue_token_calculation(persona_item: Dict[str, Any], tokenizer):
    try:
        full_session_chain = copy.deepcopy(persona_item['Full_Session_Chain'])
        total_dialogue_token_length = 0
        valid_session_dialogue_count = 0

        for session_item in full_session_chain:
            session_token_length = calculate_single_session_dialogue_tokens(session_item, tokenizer)
            session_item['Session_Dialogue_Token_Length'] = session_token_length
            if session_token_length > 0:
                valid_session_dialogue_count += 1
            total_dialogue_token_length += session_token_length

        previous_cost = persona_item.get('token_cost')
        current_stage_cost = build_current_stage_zero_cost()
        cost_info = calculate_cumulative_cost(previous_cost, current_stage_cost)

        return full_session_chain, total_dialogue_token_length, valid_session_dialogue_count, cost_info

    except Exception as e:
        print(f'[DEBUG] generate_single_dialogue_token_calculation failed: {e}:{traceback.format_exc()}')
        raise


def run_single_mode(args: argparse.Namespace, interval_mode: str) -> bool:
    input_file = resolve_input_path(interval_mode, args.input_file)
    output_file, output_json_file = resolve_output_paths(interval_mode, args.output_file, args.output_json_file)

    print(f'Processing file: {input_file}')
    print(f'Output file: {output_file}')
    print(f'Output JSON file: {output_json_file}')
    print(f'[DEBUG] Interval mode: {interval_mode}')

    try:
        tokenizer = load_tokenizer(args.tokenizer_name)
        all_personas = load_jsonl_items(input_file)
        results: List[Dict[str, Any]] = []

        for idx, persona_item in enumerate(all_personas, start=1):
            print(f'[DEBUG] Processing persona {idx}/{len(all_personas)}: {persona_item.get("ID")}')
            updated_full_session_chain, total_dialogue_token_length, valid_session_dialogue_count, cost_info = (
                generate_single_dialogue_token_calculation(
                    persona_item=persona_item,
                    tokenizer=tokenizer,
                )
            )

            result_item = copy.deepcopy(persona_item)
            result_item['Full_Session_Chain'] = updated_full_session_chain
            result_item['Total_Dialogue_Token_Length'] = total_dialogue_token_length
            result_item['Valid_Session_Dialogue_Count'] = valid_session_dialogue_count
            result_item['token_cost'] = cost_info
            results.append(result_item)

            print(
                f'[DEBUG] Persona {idx} completed - '
                f'Valid sessions: {valid_session_dialogue_count}, '
                f'Total dialogue tokens: {total_dialogue_token_length}'
            )

        write_jsonl_items(output_file, results)
        write_json_items(output_json_file, results)
        print(f'[DEBUG] Successfully generated Step4_3 interval ablation ({interval_mode}) with {len(results)} personas.')
        return True
    except Exception as e:
        print(f'[DEBUG] Step4_3 interval ablation failed: {e}:{traceback.format_exc()}')
        return False


def main(args: argparse.Namespace) -> bool:
    if args.interval_mode == 'both':
        short_ok = run_single_mode(args, 'short')
        long_ok = run_single_mode(args, 'long')
        return short_ok and long_ok
    return run_single_mode(args, args.interval_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conflict interval ablation: Step4_3 dialogue token calculation.')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input Step4_2 interval JSONL file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSONL file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--output_json_file', type=str, default=None,
                        help='Output JSON file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--tokenizer_name', type=str, default='o200k_base',
                        help='tiktoken tokenizer name')
    parser.add_argument('--interval_mode', type=str, choices=['short', 'long', 'both'], default='both',
                        help='Conflict interval mode for the first ablation version')
    args = parser.parse_args()
    main(args)
