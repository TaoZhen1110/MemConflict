import argparse
import copy
import json
import os
from typing import Any, Dict, List


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))


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


def write_json_items(path: str, items: List[Dict[str, Any]]) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if len(items) == 1:
            json.dump(items[0], f, ensure_ascii=False, indent=4)
        else:
            json.dump(items, f, ensure_ascii=False, indent=4)


def is_interference_item(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    role = str(item.get("Role", "")).strip()
    if role == "Distractor":
        return True
    return False


def clean_session(session_item: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = copy.deepcopy(session_item)

    static_items = cleaned.get("Static_Conflict_Information", [])
    if isinstance(static_items, list):
        cleaned["Static_Conflict_Information"] = [
            item for item in static_items
            if isinstance(item, dict) and not is_interference_item(item)
        ]

    conditional_items = cleaned.get("Conditional_Conflict_Information", [])
    if isinstance(conditional_items, list):
        cleaned["Conditional_Conflict_Information"] = [
            item for item in conditional_items
            if isinstance(item, dict) and not is_interference_item(item)
        ]

    others_dynamic_items = cleaned.get("Others_Dynamic_Information", [])
    if isinstance(others_dynamic_items, list):
        cleaned["Others_Dynamic_Information"] = [
            item for item in others_dynamic_items
            if isinstance(item, dict) and not is_interference_item(item)
        ]
    else:
        cleaned["Others_Dynamic_Information"] = []

    return cleaned


def build_no_interference_item(persona_item: Dict[str, Any]) -> Dict[str, Any]:
    result_item = copy.deepcopy(persona_item)
    full_session_chain = persona_item.get("Full_Session_Chain", [])
    if not isinstance(full_session_chain, list):
        return result_item

    removed_static = 0
    removed_conditional = 0
    removed_dynamic = 0
    cleaned_chain = []

    for session_item in full_session_chain:
        static_before = len(session_item.get("Static_Conflict_Information", [])) if isinstance(session_item.get("Static_Conflict_Information", []), list) else 0
        conditional_before = len(session_item.get("Conditional_Conflict_Information", [])) if isinstance(session_item.get("Conditional_Conflict_Information", []), list) else 0
        dynamic_before = len(session_item.get("Others_Dynamic_Information", [])) if isinstance(session_item.get("Others_Dynamic_Information", []), list) else 0

        cleaned_session = clean_session(session_item)

        static_after = len(cleaned_session.get("Static_Conflict_Information", []))
        conditional_after = len(cleaned_session.get("Conditional_Conflict_Information", []))
        dynamic_after = len(cleaned_session.get("Others_Dynamic_Information", []))

        removed_static += max(0, static_before - static_after)
        removed_conditional += max(0, conditional_before - conditional_after)
        removed_dynamic += max(0, dynamic_before - dynamic_after)
        cleaned_chain.append(cleaned_session)

    result_item["Full_Session_Chain"] = cleaned_chain
    result_item["No_Interference_Metadata"] = {
        "Applied": True,
        "Removed_Static_Distractor_Count": removed_static,
        "Removed_Conditional_Distractor_Count": removed_conditional,
        "Removed_Others_Dynamic_Count": removed_dynamic,
    }
    return result_item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a no-interference ablation dataset by removing distractor / other-person information from Step3_3 output."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "Data", "Step3_3.jsonl"),
        help="Input JSONL file. This should normally be Step3_3.jsonl.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(CURRENT_DIR, "Data", "Step3_3_no_interference.jsonl"),
        help="Output JSONL file for the no-interference ablation branch.",
    )
    parser.add_argument(
        "--output_perfect_file",
        type=str,
        default=os.path.join(CURRENT_DIR, "Data", "Step3_3_no_interference.json"),
        help="Output JSON file for easier manual inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    personas = load_jsonl_items(args.input_file)
    cleaned_personas = [build_no_interference_item(persona_item) for persona_item in personas]
    write_jsonl_items(args.output_file, cleaned_personas)
    write_json_items(args.output_perfect_file, cleaned_personas)
    print(f"[DEBUG] No-interference Step3_3 written to: {args.output_file}")
    print(f"[DEBUG] No-interference JSON written to: {args.output_perfect_file}")
    print(f"[DEBUG] Persona count: {len(cleaned_personas)}")


if __name__ == "__main__":
    main()

