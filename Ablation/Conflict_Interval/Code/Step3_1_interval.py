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
PROJECT_DIR = THIS_FILE.parents[2]
CODE_DIR = PROJECT_DIR / 'Code'
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from llm_request import llm_request, calculate_cumulative_cost

PROMPT_PATH = PROJECT_DIR / 'Prompt' / 'Prompt3_1.txt'
with PROMPT_PATH.open('r', encoding='utf-8') as f:
    STEP3_1_PROMPT = f.read()

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


def resolve_output_paths(interval_mode: str, output_file: str | None = None, output_json_file: str | None = None) -> Tuple[str, str]:
    suffix = f"{interval_mode}_interval"
    base_dir = PROJECT_DIR / 'Ablation' / 'Conflict_Interval' / 'Data'
    resolved_output_file = output_file or str(base_dir / f'Step3_1_{suffix}.jsonl')
    resolved_output_json = output_json_file or str(base_dir / f'Step3_1_{suffix}.json')
    return resolved_output_file, resolved_output_json


def build_full_session_chain(timeline_initial_state: Dict[str, Any], timeline_sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    full_session_chain = []

    initial_items = []
    for _, node in timeline_initial_state.items():
        initial_items.append({
            'Date': node['Date'],
            'Session_Type': 'initial_reveal',
            'Revealed_Attributes': copy.deepcopy(node['Revealed_Attributes'])
        })
    initial_items.sort(key=lambda x: x['Date'])

    regular_items = copy.deepcopy(timeline_sessions)
    regular_items.sort(key=lambda x: x['Date'])

    global_session_id = 0
    for item in initial_items:
        full_session_chain.append({
            'Session_ID': global_session_id,
            'Date': item['Date'],
            'Session_Type': item['Session_Type'],
            'Revealed_Attributes': item['Revealed_Attributes']
        })
        global_session_id += 1

    for item in regular_items:
        new_item = copy.deepcopy(item)
        new_item['Session_ID'] = global_session_id
        full_session_chain.append(new_item)
        global_session_id += 1

    return full_session_chain


def collect_user_static_fact_candidates(fixed_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    static_fact_candidates = []

    for field in ['Name', 'Gender', 'Birthdate', 'Birthplace']:
        value = fixed_profile.get(field)
        if value not in [None, '', {}]:
            static_fact_candidates.append({'Field_Path': field, 'Value': copy.deepcopy(value)})

    education_background = fixed_profile.get('Education_Background', {})
    for field in ['Highest_Degree', 'Major', 'University']:
        value = education_background.get(field)
        if value not in [None, '', {}]:
            static_fact_candidates.append({'Field_Path': f'Education_Background.{field}', 'Value': copy.deepcopy(value)})

    family_information = fixed_profile.get('Family_Information', {})
    has_mother = 'Yes' if 'Mother' in family_information else 'No'
    has_father = 'Yes' if 'Father' in family_information else 'No'
    sibling_keys = sorted([key for key in family_information.keys() if isinstance(key, str) and key.startswith('Sibling_')])
    sibling_count = len(sibling_keys)
    has_siblings = 'Yes' if sibling_count > 0 else 'No'

    static_fact_candidates.append({'Field_Path': 'Family_Information.Has_Mother', 'Value': has_mother})
    static_fact_candidates.append({'Field_Path': 'Family_Information.Has_Father', 'Value': has_father})
    static_fact_candidates.append({'Field_Path': 'Family_Information.Has_Siblings', 'Value': has_siblings})
    static_fact_candidates.append({'Field_Path': 'Family_Information.Sibling_Count', 'Value': sibling_count})

    for parent_key in ['Mother', 'Father']:
        if parent_key in family_information:
            parent_info = family_information[parent_key]
            if 'Name' in parent_info and parent_info['Name'] not in [None, '', {}]:
                static_fact_candidates.append({'Field_Path': f'Family_Information.{parent_key}.Name', 'Value': copy.deepcopy(parent_info['Name'])})
            if 'Birth_Date' in parent_info and parent_info['Birth_Date'] not in [None, '', {}]:
                static_fact_candidates.append({'Field_Path': f'Family_Information.{parent_key}.Birth_Date', 'Value': copy.deepcopy(parent_info['Birth_Date'])})

    for sibling_key in sibling_keys:
        sibling_info = family_information[sibling_key]
        if 'Type' in sibling_info and sibling_info['Type'] not in [None, '', {}]:
            static_fact_candidates.append({'Field_Path': f'Family_Information.{sibling_key}.Type', 'Value': copy.deepcopy(sibling_info['Type'])})
        if 'Name' in sibling_info and sibling_info['Name'] not in [None, '', {}]:
            static_fact_candidates.append({'Field_Path': f'Family_Information.{sibling_key}.Name', 'Value': copy.deepcopy(sibling_info['Name'])})
        if 'Birth_Date' in sibling_info and sibling_info['Birth_Date'] not in [None, '', {}]:
            static_fact_candidates.append({'Field_Path': f'Family_Information.{sibling_key}.Birth_Date', 'Value': copy.deepcopy(sibling_info['Birth_Date'])})

    return static_fact_candidates


def collect_others_static_fact_pool(others_profile: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    others_static_fact_pool: Dict[str, List[Dict[str, Any]]] = {}

    def add_fact(field_path: str, value: Any, source_person_id: str, relationship_to_user: str) -> None:
        if value in [None, '', {}]:
            return
        if field_path not in others_static_fact_pool:
            others_static_fact_pool[field_path] = []
        others_static_fact_pool[field_path].append({
            'Source_Person_ID': source_person_id,
            'Relationship_To_User': relationship_to_user,
            'Value': copy.deepcopy(value)
        })

    if not isinstance(others_profile, dict):
        return others_static_fact_pool

    for person_id, person_info in others_profile.items():
        if not isinstance(person_info, dict):
            continue

        relationship_to_user = person_info.get('Relationship_To_User', person_id)

        for field in ['Name', 'Gender', 'Birthdate', 'Birthplace']:
            add_fact(field, person_info.get(field), person_id, relationship_to_user)

        education_background = person_info.get('Education_Background', {})
        if isinstance(education_background, dict):
            for field in ['Highest_Degree', 'Major', 'University']:
                add_fact(f'Education_Background.{field}', education_background.get(field), person_id, relationship_to_user)

        family_information = person_info.get('Family_Information', {})
        if isinstance(family_information, dict):
            for field in ['Father_Alive', 'Mother_Alive', 'Has_Siblings', 'Sibling_Count']:
                add_fact(f'Family_Information.{field}', family_information.get(field), person_id, relationship_to_user)

    return others_static_fact_pool


def generate_static_conflict_triples(user_static_fact_candidates: List[Dict[str, Any]],
                                     others_static_fact_pool: Dict[str, List[Dict[str, Any]]],
                                     num_conflicts: int,
                                     previous_cost: Dict[str, Any] | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    llm_input = {
        'User_Static_Fact_Candidates': user_static_fact_candidates,
        'Others_Static_Fact_Pool_By_Field_Path': others_static_fact_pool,
    }
    user_prompt = (
        f"Number of static conflict triples to generate: {num_conflicts}\n\n"
        "Input data:\n"
        f"{json.dumps(llm_input, ensure_ascii=False, indent=2)}\n\n"
        "Generate static conflict triples."
    )

    json_markers = [
        'Corrected fixed part', 'Corrected persona', 'Corrected JSON',
        'Final JSON', 'Complete JSON', 'Correction result'
    ]

    static_conflict_result, cost_info = llm_request(
        STEP3_1_PROMPT,
        user_prompt,
        return_parsed_json=True,
        json_markers=json_markers,
    )
    cost_info = calculate_cumulative_cost(previous_cost, cost_info)
    return static_conflict_result['Static_Conflict_Triples'], cost_info


def gap_in_range(gap: int, gap_low: int, gap_high: int | None) -> bool:
    if gap < gap_low:
        return False
    if gap_high is not None and gap > gap_high:
        return False
    return True


def build_point_b_candidates(updated_full_session_chain: List[Dict[str, Any]], point_a_session_id: int,
                             gap_low: int, gap_high: int | None, strict: bool) -> List[int]:
    total_sessions = len(updated_full_session_chain)
    last_session_id = total_sessions - 1

    candidate_ids = []
    for sid, session in enumerate(updated_full_session_chain):
        if sid == last_session_id:
            continue
        if sid <= point_a_session_id:
            continue
        gap = sid - point_a_session_id
        if strict and not gap_in_range(gap, gap_low, gap_high):
            continue
        candidate_ids.append(sid)

    if not candidate_ids:
        return []

    chitchat_ids = [sid for sid in candidate_ids if updated_full_session_chain[sid]['Session_Type'] == 'chitchat']
    return chitchat_ids if chitchat_ids else candidate_ids


def choose_point_b_session_id(updated_full_session_chain: List[Dict[str, Any]], point_a_session_id: int,
                              gap_low: int, gap_high: int | None, rng: random.Random) -> Tuple[int | None, bool]:
    strict_candidates = build_point_b_candidates(
        updated_full_session_chain=updated_full_session_chain,
        point_a_session_id=point_a_session_id,
        gap_low=gap_low,
        gap_high=gap_high,
        strict=True,
    )
    if strict_candidates:
        return rng.choice(strict_candidates), True

    relaxed_candidates = build_point_b_candidates(
        updated_full_session_chain=updated_full_session_chain,
        point_a_session_id=point_a_session_id,
        gap_low=gap_low,
        gap_high=gap_high,
        strict=False,
    )
    if relaxed_candidates:
        if gap_high is None:
            relaxed_candidates.sort(key=lambda sid: sid - point_a_session_id, reverse=True)
            return relaxed_candidates[0], False
        relaxed_candidates.sort(key=lambda sid: abs((sid - point_a_session_id) - gap_low))
        return relaxed_candidates[0], False

    return None, False


def choose_distractor_session_id(point_a_session_id: int, point_b_session_id: int,
                                 updated_full_session_chain: List[Dict[str, Any]], rng: random.Random) -> int | None:
    if point_b_session_id <= point_a_session_id + 1:
        return None

    candidate_d_ids = [
        sid for sid in range(point_a_session_id + 1, point_b_session_id)
        if updated_full_session_chain[sid].get('Session_Type') == 'chitchat'
    ]
    if not candidate_d_ids:
        candidate_d_ids = list(range(point_a_session_id + 1, point_b_session_id))
    if not candidate_d_ids:
        return None
    return rng.choice(candidate_d_ids)


def assign_and_inject_static_conflicts_interval(full_session_chain: List[Dict[str, Any]],
                                                static_conflict_triples: List[Dict[str, Any]],
                                                point_a_range: Tuple[int, int],
                                                gap_low: int,
                                                gap_high: int | None,
                                                seed: int = 42) -> Tuple[List[Dict[str, Any]], List[int], int, int]:
    rng = random.Random(seed)
    updated_full_session_chain = copy.deepcopy(full_session_chain)
    total_sessions = len(updated_full_session_chain)
    min_a, max_a = point_a_range
    last_session_id = total_sessions - 1

    for session in updated_full_session_chain:
        if 'Static_Conflict_Information' not in session:
            session['Static_Conflict_Information'] = []

    realized_gaps: List[int] = []
    skipped_conflict_count = 0
    fallback_conflict_count = 0

    point_a_candidates = [
        sid for sid in range(max(0, min_a), min(max_a + 1, total_sessions))
        if sid != last_session_id
    ]
    rng.shuffle(point_a_candidates)

    shuffled_triples = copy.deepcopy(static_conflict_triples)
    rng.shuffle(shuffled_triples)

    for idx, triple in enumerate(shuffled_triples):
        if not point_a_candidates:
            break

        conflict_id = f"SC_{idx + 1:03d}"
        point_a_session_id = point_a_candidates[idx % len(point_a_candidates)]
        point_b_session_id, matched_target_range = choose_point_b_session_id(
            updated_full_session_chain=updated_full_session_chain,
            point_a_session_id=point_a_session_id,
            gap_low=gap_low,
            gap_high=gap_high,
            rng=rng,
        )

        if point_b_session_id is None:
            skipped_conflict_count += 1
            continue

        if not matched_target_range:
            fallback_conflict_count += 1

        realized_gaps.append(point_b_session_id - point_a_session_id)

        updated_full_session_chain[point_a_session_id]['Static_Conflict_Information'].append({
            'Conflict_ID': conflict_id,
            'Role': 'Point_A',
            'Target_Field_Path': triple.get('Target_Field_Path'),
            'Value': copy.deepcopy(triple.get('Point_A_Truth_Value'))
        })
        updated_full_session_chain[point_b_session_id]['Static_Conflict_Information'].append({
            'Conflict_ID': conflict_id,
            'Role': 'Point_B',
            'Target_Field_Path': triple.get('Target_Field_Path'),
            'Value': copy.deepcopy(triple.get('Point_B_Conflict_Value'))
        })

        distractor_info = triple.get('Distractor')
        if distractor_info:
            distractor_sid = choose_distractor_session_id(
                point_a_session_id=point_a_session_id,
                point_b_session_id=point_b_session_id,
                updated_full_session_chain=updated_full_session_chain,
                rng=rng,
            )
            if distractor_sid is not None:
                updated_full_session_chain[distractor_sid]['Static_Conflict_Information'].append({
                    'Conflict_ID': conflict_id,
                    'Role': 'Distractor',
                    'Source_Person_ID': distractor_info.get('Source_Person_ID'),
                    'Relationship_To_User': distractor_info.get('Relationship_To_User'),
                    'Target_Field_Path': distractor_info.get('Field'),
                    'Value': copy.deepcopy(distractor_info.get('Value'))
                })

    return updated_full_session_chain, realized_gaps, skipped_conflict_count, fallback_conflict_count


def build_interval_metadata(interval_mode: str, gap_low: int, gap_high: int | None,
                            realized_gaps: List[int], skipped_conflict_count: int,
                            fallback_conflict_count: int) -> Dict[str, Any]:
    metadata = {
        'Experiment': 'Conflict_Interval',
        'Step': 'Step3_1',
        'Interval_Mode': interval_mode,
        'Target_Gap_Range': [gap_low, gap_high if gap_high is not None else 'last'],
        'Applied_Conflict_Count': len(realized_gaps),
        'Skipped_Conflict_Count': skipped_conflict_count,
        'Fallback_Conflict_Count': fallback_conflict_count,
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


def generate_single_static_conflict_interval(persona_item: Dict[str, Any], num_conflicts: int,
                                             gap_low: int, gap_high: int | None, seed: int,
                                             point_a_range: Tuple[int, int], interval_mode: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
    fixed_profile = persona_item['Fixed_Profile']
    others_profile = persona_item['Others_Profile']
    timeline_initial_state = persona_item['Timeline_Initial_State']
    timeline_sessions = persona_item['Timeline_Sessions']
    previous_cost = persona_item.get('token_cost')

    full_session_chain = build_full_session_chain(
        timeline_initial_state=timeline_initial_state,
        timeline_sessions=timeline_sessions,
    )
    user_static_fact_candidates = collect_user_static_fact_candidates(fixed_profile=fixed_profile)
    others_static_fact_pool = collect_others_static_fact_pool(others_profile=others_profile)

    static_conflict_triples, cost_info = generate_static_conflict_triples(
        user_static_fact_candidates=user_static_fact_candidates,
        others_static_fact_pool=others_static_fact_pool,
        num_conflicts=num_conflicts,
        previous_cost=previous_cost,
    )

    updated_full_session_chain, realized_gaps, skipped_conflict_count, fallback_conflict_count = assign_and_inject_static_conflicts_interval(
        full_session_chain=full_session_chain,
        static_conflict_triples=static_conflict_triples,
        point_a_range=point_a_range,
        gap_low=gap_low,
        gap_high=gap_high,
        seed=seed,
    )

    interval_metadata = build_interval_metadata(
        interval_mode=interval_mode,
        gap_low=gap_low,
        gap_high=gap_high,
        realized_gaps=realized_gaps,
        skipped_conflict_count=skipped_conflict_count,
        fallback_conflict_count=fallback_conflict_count,
    )
    return updated_full_session_chain, cost_info, interval_metadata


def run_single_mode(args: argparse.Namespace, interval_mode: str) -> bool:
    output_file, output_json_file = resolve_output_paths(interval_mode, args.output_file, args.output_json_file)
    gap_low, gap_high = INTERVAL_RANGES[interval_mode]
    display_high = gap_high if gap_high is not None else 'last'

    print(f"Processing file: {args.input_file}")
    print(f"Output file: {output_file}")
    print(f"Output JSON file: {output_json_file}")
    print(f"[DEBUG] Interval mode: {interval_mode}, target gap range: [{gap_low}, {display_high}]")

    try:
        all_personas = load_jsonl_items(args.input_file)
        results: List[Dict[str, Any]] = []

        for idx, persona_item in enumerate(all_personas, start=1):
            print(f"[DEBUG] Processing persona {idx}/{len(all_personas)}: {persona_item.get('ID')}")
            try:
                full_session_chain, cost_info, interval_metadata = generate_single_static_conflict_interval(
                    persona_item=persona_item,
                    num_conflicts=args.num_conflicts,
                    gap_low=gap_low,
                    gap_high=gap_high,
                    seed=args.seed,
                    point_a_range=(args.point_a_min, args.point_a_max),
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
                    'token_cost': cost_info,
                    'Conflict_Interval_Metadata': interval_metadata,
                }
                results.append(result_item)
                print(
                    f"[DEBUG] Persona interval summary - applied conflicts: {interval_metadata['Applied_Conflict_Count']}, "
                    f"fallback: {interval_metadata['Fallback_Conflict_Count']}, skipped: {interval_metadata['Skipped_Conflict_Count']}, "
                    f"avg gap: {interval_metadata['Realized_Gap_Avg']}"
                )
            except Exception as e:
                print(f"[DEBUG] Failed to process persona {idx}: {e}:{traceback.format_exc()}")
                continue

        write_jsonl_items(output_file, results)
        write_json_items(output_json_file, results)
        print(f"[DEBUG] Successfully generated Step3_1 interval ablation ({interval_mode}) with {len(results)} personas.")
        return True
    except Exception as e:
        print(f"[DEBUG] Step3_1 interval ablation failed: {e}:{traceback.format_exc()}")
        return False


def main(args: argparse.Namespace) -> bool:
    if args.interval_mode == 'both':
        short_ok = run_single_mode(args, 'short')
        long_ok = run_single_mode(args, 'long')
        return short_ok and long_ok
    return run_single_mode(args, args.interval_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conflict interval ablation: Step3_1 static conflict construction.')
    parser.add_argument('--input_file', type=str, default='/home/taoz/Mem_Conflict/MemConflict/Data/Step2_2.jsonl',
                        help='Input Step2_2 JSONL file')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSONL file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--output_json_file', type=str, default=None,
                        help='Output JSON file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--interval_mode', type=str, choices=['short', 'long', 'both'], default='both',
                        help='Conflict interval mode for the first ablation version')
    parser.add_argument('--num_conflicts', type=int, default=12,
                        help='Number of static conflicts to generate')
    parser.add_argument('--point_a_min', type=int, default=0,
                        help='Lower bound of Point A session range')
    parser.add_argument('--point_a_max', type=int, default=9,
                        help='Upper bound of Point A session range')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for interval assignment')
    args = parser.parse_args()
    main(args)
