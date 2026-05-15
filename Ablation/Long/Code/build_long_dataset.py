import argparse
import copy
import json
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import tiktoken
except Exception:
    tiktoken = None


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DEFAULT_INPUT_FILE = os.path.join(PROJECT_ROOT, "Data", "Step4_4.jsonl")
DEFAULT_OUTPUT_FILE = os.path.join(CURRENT_DIR, "Data", "Step4_5_long.jsonl")
DEFAULT_SUPPLEMENT_FILES = [
    os.path.join(CURRENT_DIR, "Data", "native_math_messages_sample.jsonl"),
    os.path.join(CURRENT_DIR, "Data", "eli5_messages.jsonl"),
    os.path.join(CURRENT_DIR, "Data", "self_gene_unrelated_qa.jsonl"),
]
DEFAULT_TOKENIZER = "o200k_base"
DATETIME_FORMAT = "%Y-%m-%d"


def load_jsonl_items(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl_items(path: str, items: List[Dict[str, Any]]) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_tokenizer(tokenizer_name: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding(tokenizer_name)
    except Exception:
        return None


def count_tokens(text: str, tokenizer) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text))
    return len(text.split())


def count_messages_tokens(messages: List[Dict[str, Any]], tokenizer) -> int:
    return sum(count_tokens(str(message.get("content", "")), tokenizer) for message in messages if isinstance(message, dict))


def normalize_supplement_record(record: Dict[str, Any], tokenizer) -> Optional[Dict[str, Any]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    normalized_messages: List[Dict[str, str]] = []
    for message in messages[:2]:
        if not isinstance(message, dict):
            return None
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            return None
        normalized_messages.append({"role": role, "content": content})

    if normalized_messages[0]["role"] != "user" or normalized_messages[1]["role"] != "assistant":
        return None

    token_length = int(record.get("token_length") or 0)
    if token_length <= 0:
        token_length = count_messages_tokens(normalized_messages, tokenizer)

    return {
        "messages": normalized_messages,
        "token_length": max(1, token_length),
    }


def load_supplement_items(paths: List[str], tokenizer) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in paths:
        if not os.path.exists(path):
            continue
        for record in load_jsonl_items(path):
            normalized = normalize_supplement_record(record, tokenizer)
            if normalized is not None:
                items.append(normalized)
    return items


def build_weighted_distribution(items: List[Dict[str, Any]]) -> List[float]:
    weights = [float(max(1, item.get("token_length", 1))) for item in items]
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(items) for _ in items]
    return [weight / total for weight in weights]


def sample_supplement_items(
    supplement_items: List[Dict[str, Any]],
    needed_tokens: int,
    rng: random.Random,
    deduplicate_samples: bool,
) -> List[Dict[str, Any]]:
    if needed_tokens <= 0 or not supplement_items:
        return []

    sampled: List[Dict[str, Any]] = []
    weights = build_weighted_distribution(supplement_items)
    seen = set()
    acc_tokens = 0
    max_rounds = max(needed_tokens * 2, 200)

    for _ in range(max_rounds):
        if acc_tokens >= needed_tokens:
            break
        item = copy.deepcopy(rng.choices(supplement_items, weights=weights, k=1)[0])
        if deduplicate_samples:
            key = tuple((message["role"], message["content"]) for message in item["messages"])
            if key in seen:
                continue
            seen.add(key)
        sampled.append(item)
        acc_tokens += int(item.get("token_length", 0))

    return sampled


def extract_dialogue_turns(session_dialogue: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    if not isinstance(session_dialogue, dict):
        return []

    indexed_turns: List[Tuple[int, List[Dict[str, Any]]]] = []
    for key, value in session_dialogue.items():
        if not (isinstance(key, str) and key.startswith("dialogue_turn_") and isinstance(value, list)):
            continue
        try:
            index = int(key.rsplit("_", 1)[-1])
        except Exception:
            index = 0
        normalized_turn = []
        for message in value:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                normalized_turn.append({"role": role, "content": content})
        if normalized_turn:
            indexed_turns.append((index, normalized_turn))

    indexed_turns.sort(key=lambda item: item[0])
    return [turn for _, turn in indexed_turns]


def turns_to_session_dialogue(turns: List[List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    session_dialogue: Dict[str, List[Dict[str, Any]]] = {}
    for index, turn in enumerate(turns, start=1):
        session_dialogue[f"dialogue_turn_{index}"] = copy.deepcopy(turn)
    return session_dialogue


def get_session_date(session_item: Dict[str, Any]) -> Optional[datetime]:
    raw_date = session_item.get("Date")
    if not isinstance(raw_date, str) or not raw_date.strip():
        return None
    try:
        return datetime.strptime(raw_date.strip(), DATETIME_FORMAT)
    except Exception:
        return None


def format_session_date(dt: datetime) -> str:
    return dt.strftime(DATETIME_FORMAT)


def sample_intermediate_date(left_session: Dict[str, Any], right_session: Dict[str, Any], rng: random.Random) -> str:
    left_date = get_session_date(left_session)
    right_date = get_session_date(right_session)
    if left_date is not None and right_date is not None and right_date > left_date:
        delta_days = max((right_date - left_date).days, 1)
        offset = rng.randint(1, delta_days)
        candidate = left_date + timedelta(days=offset)
        if candidate >= right_date:
            candidate = right_date - timedelta(days=1)
        if candidate <= left_date:
            candidate = left_date + timedelta(days=1)
        return format_session_date(candidate)
    if left_date is not None:
        return format_session_date(left_date + timedelta(days=1))
    if right_date is not None:
        return format_session_date(right_date - timedelta(days=1))
    return "2026-01-01"


def rebuild_session_ids(full_session_chain: List[Dict[str, Any]]) -> None:
    for index, session_item in enumerate(full_session_chain):
        session_item["Session_ID"] = index


def update_session_dialogue_fields(session_item: Dict[str, Any], tokenizer) -> None:
    turns = extract_dialogue_turns(session_item.get("Session_Dialogue", {}))
    session_item["Session_Dialogue"] = turns_to_session_dialogue(turns)
    flat_messages = [message for turn in turns for message in turn]
    session_item["Session_Dialogue_Token_Length"] = count_messages_tokens(flat_messages, tokenizer)


def insert_internal_qa(session_item: Dict[str, Any], qa_messages: List[Dict[str, Any]], tokenizer, rng: random.Random) -> None:
    turns = extract_dialogue_turns(session_item.get("Session_Dialogue", {}))
    insert_pos = rng.randint(0, len(turns))
    turns.insert(insert_pos, copy.deepcopy(qa_messages))
    session_item["Session_Dialogue"] = turns_to_session_dialogue(turns)
    session_item["Session_Dialogue_Token_Length"] = int(session_item.get("Session_Dialogue_Token_Length", 0) or 0) + count_messages_tokens(qa_messages, tokenizer)


def chunk_items_for_new_sessions(items: List[Dict[str, Any]], target_session_tokens: int) -> List[List[Dict[str, Any]]]:
    if not items:
        return []
    target_session_tokens = max(1, target_session_tokens)
    groups: List[List[Dict[str, Any]]] = []
    current_group: List[Dict[str, Any]] = []
    current_tokens = 0

    for item in items:
        item_tokens = int(item.get("token_length", 0) or 0)
        if current_group and current_tokens + item_tokens > target_session_tokens:
            groups.append(current_group)
            current_group = [item]
            current_tokens = item_tokens
        else:
            current_group.append(item)
            current_tokens += item_tokens

    if current_group:
        groups.append(current_group)
    return groups


def create_generated_session(
    qa_group: List[Dict[str, Any]],
    left_session: Optional[Dict[str, Any]],
    right_session: Optional[Dict[str, Any]],
    tokenizer,
    rng: random.Random,
) -> Dict[str, Any]:
    turns = [copy.deepcopy(item["messages"]) for item in qa_group]
    flat_messages = [message for turn in turns for message in turn]
    token_length = count_messages_tokens(flat_messages, tokenizer)

    if left_session is not None and right_session is not None:
        date_text = sample_intermediate_date(left_session, right_session, rng)
    elif left_session is not None:
        left_date = get_session_date(left_session) or datetime(2026, 1, 1)
        date_text = format_session_date(left_date + timedelta(days=1))
    elif right_session is not None:
        right_date = get_session_date(right_session) or datetime(2026, 1, 2)
        date_text = format_session_date(right_date - timedelta(days=1))
    else:
        date_text = "2026-01-01"

    return {
        "Session_ID": -1,
        "Date": date_text,
        "Session_Type": "long_supplement",
        "Session_Outline": "Unrelated long-context supplement session inserted for long-context evaluation.",
        "Event_Types": ["unrelated_qa"],
        "Question_Trigger_Types": [],
        "Session_Dialogue": turns_to_session_dialogue(turns),
        "Session_Dialogue_Token_Length": token_length,
        "Session_Questions": [],
        "Session_Question_Count": 0,
        "Generated_Long_Session": True,
    }


def estimate_average_session_tokens(full_session_chain: List[Dict[str, Any]]) -> int:
    lengths = [
        int(session_item.get("Session_Dialogue_Token_Length", 0) or 0)
        for session_item in full_session_chain
        if isinstance(session_item, dict)
    ]
    lengths = [length for length in lengths if length > 0]
    if not lengths:
        return 800
    return max(1, int(sum(lengths) / len(lengths)))


def expand_persona_to_long(
    persona_item: Dict[str, Any],
    supplement_items: List[Dict[str, Any]],
    tokenizer,
    rng: random.Random,
    target_total_tokens: int,
    new_session_ratio: float,
    deduplicate_samples: bool,
) -> Dict[str, Any]:
    result_item = copy.deepcopy(persona_item)
    full_session_chain = copy.deepcopy(persona_item.get("Full_Session_Chain", []))
    if not isinstance(full_session_chain, list):
        return result_item

    for session_item in full_session_chain:
        update_session_dialogue_fields(session_item, tokenizer)

    current_total_tokens = sum(int(session_item.get("Session_Dialogue_Token_Length", 0) or 0) for session_item in full_session_chain)
    needed_tokens = max(0, int(target_total_tokens) - current_total_tokens)
    picked_items = sample_supplement_items(
        supplement_items=supplement_items,
        needed_tokens=needed_tokens,
        rng=rng,
        deduplicate_samples=deduplicate_samples,
    )

    if not picked_items:
        result_item["Full_Session_Chain"] = full_session_chain
        result_item["Total_Dialogue_Token_Length"] = current_total_tokens
        result_item["Long_Version_Metadata"] = {
            "Applied": False,
            "Reason": "No supplement items selected.",
            "Target_Total_Tokens": target_total_tokens,
            "Original_Total_Tokens": current_total_tokens,
            "Expanded_Total_Tokens": current_total_tokens,
        }
        return result_item

    internal_count = len(picked_items) - int(round(len(picked_items) * new_session_ratio))
    internal_items = picked_items[:internal_count]
    new_session_items = picked_items[internal_count:]

    if full_session_chain and internal_items:
        for item in internal_items:
            target_session = rng.choice(full_session_chain)
            insert_internal_qa(target_session, item["messages"], tokenizer, rng)

    if new_session_items:
        grouped_items = chunk_items_for_new_sessions(
            new_session_items,
            target_session_tokens=estimate_average_session_tokens(full_session_chain),
        )
        for qa_group in grouped_items:
            if len(full_session_chain) >= 2:
                insert_pos = rng.randint(1, len(full_session_chain) - 1)
                left_session = full_session_chain[insert_pos - 1]
                right_session = full_session_chain[insert_pos]
                generated_session = create_generated_session(qa_group, left_session, right_session, tokenizer, rng)
                full_session_chain.insert(insert_pos, generated_session)
            else:
                generated_session = create_generated_session(
                    qa_group,
                    full_session_chain[-1] if full_session_chain else None,
                    None,
                    tokenizer,
                    rng,
                )
                full_session_chain.append(generated_session)

    rebuild_session_ids(full_session_chain)

    expanded_total_tokens = sum(int(session_item.get("Session_Dialogue_Token_Length", 0) or 0) for session_item in full_session_chain)
    valid_session_dialogue_count = sum(
        1
        for session_item in full_session_chain
        if int(session_item.get("Session_Dialogue_Token_Length", 0) or 0) > 0
    )

    result_item["Full_Session_Chain"] = full_session_chain
    result_item["Total_Dialogue_Token_Length"] = expanded_total_tokens
    result_item["Valid_Session_Dialogue_Count"] = valid_session_dialogue_count
    result_item["Long_Version_Metadata"] = {
        "Applied": True,
        "Target_Total_Tokens": target_total_tokens,
        "Original_Total_Tokens": current_total_tokens,
        "Expanded_Total_Tokens": expanded_total_tokens,
        "Inserted_Internal_QA_Count": len(internal_items),
        "Inserted_New_Session_QA_Count": len(new_session_items),
        "New_Session_Ratio": new_session_ratio,
    }
    return result_item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand MemConflict Step4_4 medium dataset into a long-context version.")
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE, help="Input JSONL file. This should normally be Step4_4.jsonl.")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE, help="Output JSONL file for the long dataset.")
    parser.add_argument("--supplement_files", type=str, nargs="*", default=DEFAULT_SUPPLEMENT_FILES, help="Supplement JSONL files in messages format.")
    parser.add_argument("--target_total_tokens", type=int, default=300000, help="Target total dialogue tokens per persona after expansion.")
    parser.add_argument("--new_session_ratio", type=float, default=0.5, help="Fraction of sampled supplements used to create new sessions.")
    parser.add_argument("--seed", type=int, default=20260408, help="Random seed for reproducible long-version generation.")
    parser.add_argument("--disable_deduplicate_samples", action="store_true", help="Disable supplement-level deduplication.")
    parser.add_argument("--tokenizer_name", type=str, default=DEFAULT_TOKENIZER, help="Tokenizer name used for token counting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    tokenizer = load_tokenizer(args.tokenizer_name)

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    supplement_items = load_supplement_items(args.supplement_files, tokenizer)
    if not supplement_items:
        raise RuntimeError(
            "No valid supplement items were loaded. Please prepare at least one supplement JSONL file in messages format."
        )

    all_personas = load_jsonl_items(args.input_file)
    expanded_personas = [
        expand_persona_to_long(
            persona_item=persona_item,
            supplement_items=supplement_items,
            tokenizer=tokenizer,
            rng=rng,
            target_total_tokens=args.target_total_tokens,
            new_session_ratio=args.new_session_ratio,
            deduplicate_samples=not args.disable_deduplicate_samples,
        )
        for persona_item in all_personas
    ]

    write_jsonl_items(args.output_file, expanded_personas)
    print(f"[DEBUG] Long dataset written to: {args.output_file}")
    print(f"[DEBUG] Input personas: {len(all_personas)}")
    print(f"[DEBUG] Supplement items loaded: {len(supplement_items)}")


if __name__ == "__main__":
    main()


