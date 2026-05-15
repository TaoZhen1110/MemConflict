import argparse
import copy
import json
import jsonlines
import os
import re
import traceback
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from llm_request import calculate_cumulative_cost
import logging

logger = logging.getLogger(__name__)

load_dotenv()

try:
    import tiktoken
except Exception:
    tiktoken = None


def load_tokenizer(tokenizer_name: str = "o200k_base"):
    """Load tiktoken tokenizer."""
    if tiktoken is None:
        raise ImportError("tiktoken is not installed. Please install it before running Step4_3.")

    try:
        tokenizer = tiktoken.get_encoding(tokenizer_name)
        print(f"[DEBUG] Successfully loaded tiktoken tokenizer: {tokenizer_name}")
        return tokenizer
    except Exception as e:
        print(f"[DEBUG] Failed to load tokenizer: {e}:{traceback.format_exc()}")
        raise


def _extract_turn_index(turn_key: str) -> int:
    """Extract numeric index from keys like dialogue_turn_12."""
    match = re.search(r"(\d+)$", turn_key)
    if match:
        return int(match.group(1))
    return 0


def extract_dialogue_messages(session_dialogue: Dict) -> List[Tuple[str, str]]:
    """
    Extract (role, content) pairs from Session_Dialogue.
    Expected format:
    {
        "dialogue_turn_1": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
        ...
    }
    """
    message_list = []

    if not isinstance(session_dialogue, dict):
        return message_list

    turn_items = []
    for turn_key, turn_value in session_dialogue.items():
        if isinstance(turn_key, str) and turn_key.startswith("dialogue_turn_"):
            turn_items.append((turn_key, turn_value))

    turn_items.sort(key=lambda x: _extract_turn_index(x[0]))

    for _, turn_value in turn_items:
        if not isinstance(turn_value, list):
            continue

        for msg in turn_value:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role in ["user", "assistant"] and isinstance(content, str) and content.strip():
                message_list.append((role, content))

    return message_list


def calculate_single_session_dialogue_tokens(session_item: Dict, tokenizer) -> int:
    """Calculate one session dialogue token length."""
    session_dialogue = session_item.get("Session_Dialogue", {})
    message_list = extract_dialogue_messages(session_dialogue)

    total_tokens = 0
    for _, content in message_list:
        total_tokens += len(tokenizer.encode(content))

    return total_tokens


def build_current_stage_zero_cost() -> Dict:
    """Build a standard zero-cost record for non-LLM stage."""
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "model": None,
        "input_cost_usd": 0.0,
        "output_cost_usd": 0.0,
        "pricing_available": False,
        "note": "No LLM calls in this stage - token calculation only"
    }


def Generate_Single_Dialogue_Token_Calculation(persona_item: Dict, tokenizer):
    """
    Step 4.3:
    Calculate token length for each session dialogue and total dialogue length.
    """
    try:
        full_session_chain = copy.deepcopy(persona_item["Full_Session_Chain"])
        total_dialogue_token_length = 0
        valid_session_dialogue_count = 0

        for session_item in full_session_chain:
            session_token_length = calculate_single_session_dialogue_tokens(session_item, tokenizer)
            session_item["Session_Dialogue_Token_Length"] = session_token_length

            if session_token_length > 0:
                valid_session_dialogue_count += 1

            total_dialogue_token_length += session_token_length

        previous_cost = persona_item.get("token_cost", None)
        current_stage_cost = build_current_stage_zero_cost()
        cost_info = calculate_cumulative_cost(previous_cost, current_stage_cost)

        return full_session_chain, total_dialogue_token_length, valid_session_dialogue_count, cost_info

    except Exception as e:
        print(f"[DEBUG] Generate_Single_Dialogue_Token_Calculation failed: {e}:{traceback.format_exc()}")
        raise


def Generate_User_Dialogue_Token_Calculation(args):
    print(f"Processing file: {args.input_file}")
    print(f"Output file: {args.output_file}")

    try:
        tokenizer = load_tokenizer(args.tokenizer_name)

        all_personas = []
        with jsonlines.open(args.input_file) as reader:
            for item in reader:
                all_personas.append(item)

        print(f"[DEBUG] Read {len(all_personas)} personas")

        for idx, persona_item in enumerate(all_personas):
            print(f"[DEBUG] Processing persona {idx + 1}/{len(all_personas)}")

            updated_full_session_chain, total_dialogue_token_length, valid_session_dialogue_count, cost_info = (
                Generate_Single_Dialogue_Token_Calculation(
                    persona_item=persona_item,
                    tokenizer=tokenizer
                )
            )

            result_item = copy.deepcopy(persona_item)
            result_item["Full_Session_Chain"] = updated_full_session_chain
            result_item["Total_Dialogue_Token_Length"] = total_dialogue_token_length
            result_item["Valid_Session_Dialogue_Count"] = valid_session_dialogue_count
            result_item["token_cost"] = cost_info

            with jsonlines.open(args.output_file, "a") as writer:
                writer.write(result_item)

            with open(args.output_perfect_file, "a", encoding="utf-8") as f:
                json.dump(result_item, f, ensure_ascii=False, indent=4)

            print(
                f"[DEBUG] Persona {idx + 1} completed - "
                f"Valid sessions: {valid_session_dialogue_count}, "
                f"Total dialogue tokens: {total_dialogue_token_length}"
            )

        print("[DEBUG] Successfully processed Step 4.3 dialogue token calculation")
        return True

    except Exception as e:
        print(f"Error processing Step 4.3: {e}:{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4.3 ---- Session Dialogue Token Calculation.")
    parser.add_argument("--input_file", type=str,
                        default="MemConflict/Data/Step4_2.jsonl",
                        help="Last Step output file for dialogue token calculation")
    parser.add_argument("--output_file", type=str,
                        default="MemConflict/Data/Step4_3.jsonl",
                        help="Output JSONL file for dialogue token calculation")
    parser.add_argument("--output_perfect_file", type=str,
                        default="MemConflict/Data_perfect/Step4_3.json",
                        help="Output JSON file for dialogue token calculation")
    parser.add_argument("--tokenizer_name", type=str,
                        default="o200k_base",
                        help="tiktoken tokenizer name")
    args = parser.parse_args()

    Generate_User_Dialogue_Token_Calculation(args)
