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

INTERVAL_RANGES = {
    'short': (5, 15),
    'long': (20, None),
}


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
    return str(base_dir / f'Step3_2_{interval_mode}_interval.jsonl')


def resolve_output_paths(interval_mode: str, output_file: str | None = None, output_json_file: str | None = None) -> Tuple[str, str]:
    suffix = f"{interval_mode}_interval"
    base_dir = PROJECT_DIR / 'Ablation' / 'Conflict_Interval' / 'Data'
    resolved_output_file = output_file or str(base_dir / f'Step3_3_{suffix}.jsonl')
    resolved_output_json = output_json_file or str(base_dir / f'Step3_3_{suffix}.json')
    return resolved_output_file, resolved_output_json


def collect_dynamic_attribute_session_distribution(full_session_chain: List[Dict[str, Any]]) -> Tuple[Dict[str, List[int]], Dict[str, List[Dict[str, Any]]]]:
    attribute_session_distribution: Dict[str, List[int]] = {}
    attribute_update_details: Dict[str, List[Dict[str, Any]]] = {}

    dynamic_attr_name_map = {
        'Residence': 'Residence',
        'Marital_Status': 'Marital_Status',
        'Children_Status': 'Children_Status',
        'Career_Status': 'Career_Status',
        'Work_Status': 'Work_Status',
        'Health_Status': 'Health_Status',
        'Social_Status': 'Social_Status',
    }

    def add_occurrence(attr_name: str, session_id: int, session_date: Any, source_type: str,
                       before_value: Any, after_value: Any) -> None:
        if attr_name not in attribute_session_distribution:
            attribute_session_distribution[attr_name] = []
        if attr_name not in attribute_update_details:
            attribute_update_details[attr_name] = []

        attribute_session_distribution[attr_name].append(session_id)
        attribute_update_details[attr_name].append({
            'Session_ID': session_id,
            'Date': session_date,
            'Source_Type': source_type,
            'Before': copy.deepcopy(before_value),
            'After': copy.deepcopy(after_value),
        })

    for session in full_session_chain:
        session_id = session.get('Session_ID')
        session_date = session.get('Date')
        session_type = session.get('Session_Type')

        revealed_attributes = session.get('Revealed_Attributes', {})
        if isinstance(revealed_attributes, dict) and revealed_attributes:
            for raw_attr_name, raw_value in revealed_attributes.items():
                if raw_attr_name in dynamic_attr_name_map:
                    attr_name = dynamic_attr_name_map[raw_attr_name]
                    add_occurrence(
                        attr_name=attr_name,
                        session_id=session_id,
                        session_date=session_date,
                        source_type='initial_reveal',
                        before_value=None,
                        after_value=raw_value,
                    )

        updated_attributes = session.get('Updated_Attributes', [])
        if isinstance(updated_attributes, list) and updated_attributes:
            for update_item in updated_attributes:
                if not isinstance(update_item, dict):
                    continue
                attr_name = update_item.get('Attribute')
                if attr_name in [None, '', {}]:
                    continue
                add_occurrence(
                    attr_name=attr_name,
                    session_id=session_id,
                    session_date=session_date,
                    source_type=session_type if session_type else 'update',
                    before_value=update_item.get('Before'),
                    after_value=update_item.get('After'),
                )

    for attr_name in attribute_session_distribution:
        attribute_session_distribution[attr_name] = sorted(attribute_session_distribution[attr_name])
    for attr_name in attribute_update_details:
        attribute_update_details[attr_name] = sorted(attribute_update_details[attr_name], key=lambda x: x['Session_ID'])

    return attribute_session_distribution, attribute_update_details


def collect_others_dynamic_attribute_pool(others_profile: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    others_dynamic_attribute_pool: Dict[str, List[Dict[str, Any]]] = {}

    def add_fact(attribute_name: str, source_person_id: str, relationship_to_user: str, value: Any) -> None:
        if value in [None, '', {}]:
            return
        if attribute_name not in others_dynamic_attribute_pool:
            others_dynamic_attribute_pool[attribute_name] = []
        others_dynamic_attribute_pool[attribute_name].append({
            'Source_Person_ID': source_person_id,
            'Relationship_To_User': relationship_to_user,
            'Value': copy.deepcopy(value),
        })

    if not isinstance(others_profile, dict):
        return others_dynamic_attribute_pool

    for person_id, person_info in others_profile.items():
        if not isinstance(person_info, dict):
            continue
        relationship_to_user = person_info.get('Relationship_To_User', person_id)

        add_fact('Residence', person_id, relationship_to_user, person_info.get('Residence'))
        add_fact('Career_Status', person_id, relationship_to_user, person_info.get('Career_Status'))
        add_fact('Work_Status', person_id, relationship_to_user, person_info.get('Work_Status'))
        add_fact('Health_Status', person_id, relationship_to_user, person_info.get('Health_Status'))

        if 'Marital_Status' in person_info:
            add_fact('Marital_Status', person_id, relationship_to_user, person_info.get('Marital_Status'))
        if 'Children_Status' in person_info:
            add_fact('Children_Status', person_id, relationship_to_user, person_info.get('Children_Status'))

    return others_dynamic_attribute_pool


def gap_in_range(gap: int, gap_low: int, gap_high: int | None) -> bool:
    if gap < gap_low:
        return False
    if gap_high is not None and gap > gap_high:
        return False
    return True


def choose_best_pair(pair_candidates: List[Tuple[Dict[str, Any], Dict[str, Any]]], gap_low: int,
                     gap_high: int | None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if gap_high is None:
        pair_candidates.sort(key=lambda pair: pair[1]['Session_ID'] - pair[0]['Session_ID'], reverse=True)
        return pair_candidates[0]

    target_center = (gap_low + gap_high) / 2.0
    pair_candidates.sort(key=lambda pair: abs((pair[1]['Session_ID'] - pair[0]['Session_ID']) - target_center))
    return pair_candidates[0]


def choose_dynamic_pair(detail_list: List[Dict[str, Any]], gap_low: int,
                        gap_high: int | None) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any]]], int]:
    strict_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    relaxed_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for pair_idx in range(len(detail_list) - 1):
        left_detail = detail_list[pair_idx]
        right_detail = detail_list[pair_idx + 1]
        gap = right_detail['Session_ID'] - left_detail['Session_ID']
        if gap <= 1:
            continue
        pair = (left_detail, right_detail)
        relaxed_pairs.append(pair)
        if gap_in_range(gap, gap_low, gap_high):
            strict_pairs.append(pair)

    if strict_pairs:
        return strict_pairs, 0
    if relaxed_pairs:
        return [choose_best_pair(relaxed_pairs, gap_low, gap_high)], 1
    return [], 0


def choose_distractor_session_id(updated_full_session_chain: List[Dict[str, Any]], left_sid: int, right_sid: int,
                                 used_distractor_session_ids: set[int], rng: random.Random) -> int | None:
    total_sessions = len(updated_full_session_chain)
    last_session_id = total_sessions - 1

    candidate_session_ids = [
        sid for sid in range(left_sid + 1, right_sid)
        if sid != last_session_id
        and sid not in used_distractor_session_ids
        and updated_full_session_chain[sid].get('Session_Type') == 'chitchat'
    ]
    if not candidate_session_ids:
        candidate_session_ids = [
            sid for sid in range(left_sid + 1, right_sid)
            if sid != last_session_id and sid not in used_distractor_session_ids
        ]
    if not candidate_session_ids:
        candidate_session_ids = [
            sid for sid in range(left_sid + 1, right_sid)
            if sid != last_session_id
        ]
    if not candidate_session_ids:
        return None
    return rng.choice(candidate_session_ids)


def assign_and_inject_dynamic_distractors_interval(full_session_chain: List[Dict[str, Any]],
                                                   attribute_update_details: Dict[str, List[Dict[str, Any]]],
                                                   others_dynamic_attribute_pool: Dict[str, List[Dict[str, Any]]],
                                                   gap_low: int,
                                                   gap_high: int | None,
                                                   seed: int = 42) -> Tuple[List[Dict[str, Any]], List[int], int, int]:
    rng = random.Random(seed)
    updated_full_session_chain = copy.deepcopy(full_session_chain)

    if not updated_full_session_chain:
        raise ValueError('Full_Session_Chain is empty.')

    for session in updated_full_session_chain:
        if 'Others_Dynamic_Information' not in session:
            session['Others_Dynamic_Information'] = []

    used_distractor_session_ids: set[int] = set()
    realized_gaps: List[int] = []
    applied_distractor_count = 0
    skipped_attribute_count = 0
    fallback_pair_count = 0

    for attr_name, detail_list in attribute_update_details.items():
        if attr_name not in others_dynamic_attribute_pool:
            continue
        if len(detail_list) < 2:
            continue

        other_candidates = others_dynamic_attribute_pool.get(attr_name, [])
        if not other_candidates:
            continue

        selected_pairs, pair_mode = choose_dynamic_pair(detail_list, gap_low, gap_high)
        if not selected_pairs:
            skipped_attribute_count += 1
            continue
        if pair_mode == 1:
            fallback_pair_count += 1

        for left_detail, right_detail in selected_pairs:
            left_sid = left_detail['Session_ID']
            right_sid = right_detail['Session_ID']
            left_after = left_detail.get('After')
            right_after = right_detail.get('After')

            filtered_candidates = []
            for candidate in other_candidates:
                candidate_value = candidate.get('Value')
                if candidate_value == left_after or candidate_value == right_after:
                    continue
                filtered_candidates.append(candidate)
            if not filtered_candidates:
                filtered_candidates = other_candidates

            distractor_info = rng.choice(filtered_candidates)
            distractor_session_id = choose_distractor_session_id(
                updated_full_session_chain=updated_full_session_chain,
                left_sid=left_sid,
                right_sid=right_sid,
                used_distractor_session_ids=used_distractor_session_ids,
                rng=rng,
            )
            if distractor_session_id is None:
                continue

            used_distractor_session_ids.add(distractor_session_id)
            realized_gaps.append(right_sid - left_sid)
            applied_distractor_count += 1
            updated_full_session_chain[distractor_session_id]['Others_Dynamic_Information'].append({
                'Attribute': attr_name,
                'Role': 'Distractor',
                'Source_Person_ID': distractor_info.get('Source_Person_ID'),
                'Relationship_To_User': distractor_info.get('Relationship_To_User'),
                'Value': copy.deepcopy(distractor_info.get('Value')),
                'Linked_Left_Session_ID': left_sid,
                'Linked_Right_Session_ID': right_sid,
            })

    return updated_full_session_chain, realized_gaps, applied_distractor_count, skipped_attribute_count, fallback_pair_count


def build_interval_metadata(interval_mode: str, gap_low: int, gap_high: int | None,
                            realized_gaps: List[int], applied_distractor_count: int,
                            skipped_attribute_count: int, fallback_pair_count: int) -> Dict[str, Any]:
    metadata = {
        'Experiment': 'Conflict_Interval',
        'Step': 'Step3_3',
        'Interval_Mode': interval_mode,
        'Target_Gap_Range': [gap_low, gap_high if gap_high is not None else 'last'],
        'Applied_Distractor_Count': applied_distractor_count,
        'Skipped_Attribute_Count': skipped_attribute_count,
        'Fallback_Pair_Count': fallback_pair_count,
        'Realized_Gaps': realized_gaps,
    }
    if realized_gaps:
        metadata['Realized_Gap_Min'] = min(realized_gaps)
        metadata['Realized_Gap_Max'] = max(realized_gaps)
        metadata['Realized_Gap_Avg'] = round(sum(realized_gaps) / len(realized_gaps), 4)
    else:
        metadata['Realized_Gap_Min'] = None
        metadata['Realized_Gap_Max'] = None
        metadata['Realized_Gap_Avg'] = None
    return metadata


def merge_interval_metadata(previous_metadata: Any, current_metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(previous_metadata, dict) or not previous_metadata:
        return {'Step3_3': current_metadata}

    if 'Step3_1' in previous_metadata or 'Step3_2' in previous_metadata or 'Step3_3' in previous_metadata:
        merged = copy.deepcopy(previous_metadata)
        merged['Step3_3'] = current_metadata
        return merged

    previous_step = previous_metadata.get('Step')
    if previous_step in {'Step3_1', 'Step3_2'}:
        merged = {previous_step: copy.deepcopy(previous_metadata)}
        merged['Step3_3'] = current_metadata
        return merged

    return {
        'Previous': copy.deepcopy(previous_metadata),
        'Step3_3': current_metadata,
    }


def generate_single_dynamic_distractors_interval(persona_item: Dict[str, Any], gap_low: int,
                                                 gap_high: int | None, seed: int,
                                                 interval_mode: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    full_session_chain = copy.deepcopy(persona_item['Full_Session_Chain'])
    others_profile = persona_item.get('Others_Profile', {})

    _, attribute_update_details = collect_dynamic_attribute_session_distribution(full_session_chain)
    others_dynamic_attribute_pool = collect_others_dynamic_attribute_pool(others_profile)

    updated_full_session_chain, realized_gaps, applied_distractor_count, skipped_attribute_count, fallback_pair_count = assign_and_inject_dynamic_distractors_interval(
        full_session_chain=full_session_chain,
        attribute_update_details=attribute_update_details,
        others_dynamic_attribute_pool=others_dynamic_attribute_pool,
        gap_low=gap_low,
        gap_high=gap_high,
        seed=seed,
    )

    interval_metadata = build_interval_metadata(
        interval_mode=interval_mode,
        gap_low=gap_low,
        gap_high=gap_high,
        realized_gaps=realized_gaps,
        applied_distractor_count=applied_distractor_count,
        skipped_attribute_count=skipped_attribute_count,
        fallback_pair_count=fallback_pair_count,
    )
    return updated_full_session_chain, interval_metadata


def run_single_mode(args: argparse.Namespace, interval_mode: str) -> bool:
    input_file = resolve_input_path(interval_mode, args.input_file)
    output_file, output_json_file = resolve_output_paths(interval_mode, args.output_file, args.output_json_file)
    gap_low, gap_high = INTERVAL_RANGES[interval_mode]
    display_high = gap_high if gap_high is not None else 'last'

    print(f"Processing file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Output JSON file: {output_json_file}")
    print(f"[DEBUG] Interval mode: {interval_mode}, target gap range: [{gap_low}, {display_high}]")

    try:
        all_personas = load_jsonl_items(input_file)
        results: List[Dict[str, Any]] = []

        for idx, persona_item in enumerate(all_personas, start=1):
            print(f"[DEBUG] Processing persona {idx}/{len(all_personas)}: {persona_item.get('ID')}")
            try:
                full_session_chain, interval_metadata = generate_single_dynamic_distractors_interval(
                    persona_item=persona_item,
                    gap_low=gap_low,
                    gap_high=gap_high,
                    seed=args.seed,
                    interval_mode=interval_mode,
                )

                result_item = {
                    'ID': persona_item['ID'],
                    'Fixed_Profile': persona_item['Fixed_Profile'],
                    'Dynamic_Profile': persona_item['Dynamic_Profile'],
                    'Preference_Profile': persona_item['Preference_Profile'],
                    'Personality': persona_item['Personality'],
                    'Life_Goal': persona_item['Life_Goal'],
                    'Others_Profile': persona_item['Others_Profile'],
                    'Full_Session_Chain': full_session_chain,
                    'metadata': persona_item['metadata'],
                    'token_cost': persona_item.get('token_cost'),
                    'Conflict_Interval_Metadata': merge_interval_metadata(
                        persona_item.get('Conflict_Interval_Metadata'),
                        interval_metadata,
                    ),
                }
                results.append(result_item)
                print(
                    f"[DEBUG] Persona interval summary - applied distractors: {interval_metadata['Applied_Distractor_Count']}, "
                    f"fallback pairs: {interval_metadata['Fallback_Pair_Count']}, skipped attributes: {interval_metadata['Skipped_Attribute_Count']}, "
                    f"avg gap: {interval_metadata['Realized_Gap_Avg']}"
                )
            except Exception as e:
                print(f"[DEBUG] Failed to process persona {idx}: {e}:{traceback.format_exc()}")
                continue

        write_jsonl_items(output_file, results)
        write_json_items(output_json_file, results)
        print(f"[DEBUG] Successfully generated Step3_3 interval ablation ({interval_mode}) with {len(results)} personas.")
        return True
    except Exception as e:
        print(f"[DEBUG] Step3_3 interval ablation failed: {e}:{traceback.format_exc()}")
        return False


def main(args: argparse.Namespace) -> bool:
    if args.interval_mode == 'both':
        short_ok = run_single_mode(args, 'short')
        long_ok = run_single_mode(args, 'long')
        return short_ok and long_ok
    return run_single_mode(args, args.interval_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conflict interval ablation: Step3_3 dynamic distractor construction.')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input Step3_2 interval JSONL file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSONL file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--output_json_file', type=str, default=None,
                        help='Output JSON file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--interval_mode', type=str, choices=['short', 'long', 'both'], default='both',
                        help='Conflict interval mode for the first ablation version')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for interval assignment')
    args = parser.parse_args()
    main(args)
