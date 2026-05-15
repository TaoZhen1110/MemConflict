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

PROMPT_PATH = PROJECT_DIR / 'Prompt' / 'Prompt3_2.txt'
with PROMPT_PATH.open('r', encoding='utf-8') as f:
    STEP3_2_PROMPT = f.read()

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
    return str(base_dir / f'Step3_1_{interval_mode}_interval.jsonl')


def resolve_output_paths(interval_mode: str, output_file: str | None = None, output_json_file: str | None = None) -> Tuple[str, str]:
    suffix = f"{interval_mode}_interval"
    base_dir = PROJECT_DIR / 'Ablation' / 'Conflict_Interval' / 'Data'
    resolved_output_file = output_file or str(base_dir / f'Step3_2_{suffix}.jsonl')
    resolved_output_json = output_json_file or str(base_dir / f'Step3_2_{suffix}.json')
    return resolved_output_file, resolved_output_json


def collect_user_conditional_preference_candidates(preference_profile: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    user_candidates: Dict[str, List[Dict[str, Any]]] = {}

    for pref_type, rules in preference_profile.items():
        rule_list = []
        for item, condition in rules.items():
            if item in [None, '', {}] or condition in [None, '', {}]:
                continue
            rule_list.append({'Item': copy.deepcopy(item), 'Condition': copy.deepcopy(condition)})
        if rule_list:
            user_candidates[pref_type] = rule_list

    return user_candidates


def collect_others_conditional_preference_candidates(others_profile: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    others_candidates: Dict[str, List[Dict[str, Any]]] = {}

    if not isinstance(others_profile, dict):
        return others_candidates

    for person_id, person_info in others_profile.items():
        relationship_to_user = person_info.get('Relationship_To_User', person_id)
        pref_profile = person_info.get('Preference_Profile', {})
        if not isinstance(pref_profile, dict):
            continue

        for pref_key, pref_description in pref_profile.items():
            if pref_description in [None, '', {}]:
                continue
            if pref_key not in others_candidates:
                others_candidates[pref_key] = []
            others_candidates[pref_key].append({
                'Source_Person_ID': person_id,
                'Relationship_To_User': relationship_to_user,
                'Preference_Key': pref_key,
                'Preference_Description': copy.deepcopy(pref_description)
            })

    return others_candidates


def generate_conditional_conflict_groups_with_llm(user_groups: List[Dict[str, Any]],
                                                  others_candidates: Dict[str, List[Dict[str, Any]]],
                                                  previous_cost: Dict[str, Any] | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    llm_input = {
        'User_Conditional_Groups': user_groups,
        'Others_Conditional_Candidates': others_candidates,
    }

    user_prompt = (
        'Input data:\n'
        f'{json.dumps(llm_input, ensure_ascii=False, indent=2)}\n\n'
        'Generate conditional conflict groups.'
    )

    json_markers = [
        'Corrected fixed part', 'Corrected persona', 'Corrected JSON',
        'Final JSON', 'Complete JSON', 'Correction result'
    ]

    conflict_groups_with_distractors, cost_info = llm_request(
        STEP3_2_PROMPT,
        user_prompt,
        return_parsed_json=True,
        json_markers=json_markers,
    )
    cost_info = calculate_cumulative_cost(previous_cost, cost_info)
    return conflict_groups_with_distractors['Conditional_Conflict_Groups'], cost_info


def gap_in_range(gap: int, gap_low: int, gap_high: int | None) -> bool:
    if gap < gap_low:
        return False
    if gap_high is not None and gap > gap_high:
        return False
    return True


def build_rule_candidates(updated_full_session_chain: List[Dict[str, Any]], previous_sid: int,
                          gap_low: int, gap_high: int | None, strict: bool) -> List[int]:
    total_sessions = len(updated_full_session_chain)
    last_session_id = total_sessions - 1

    candidate_ids = []
    for sid, session in enumerate(updated_full_session_chain):
        if sid == last_session_id:
            continue
        if sid <= previous_sid:
            continue
        gap = sid - previous_sid
        if strict and not gap_in_range(gap, gap_low, gap_high):
            continue
        candidate_ids.append(sid)

    if not candidate_ids:
        return []

    chitchat_ids = [sid for sid in candidate_ids if updated_full_session_chain[sid].get('Session_Type') == 'chitchat']
    return chitchat_ids if chitchat_ids else candidate_ids


def build_any_later_candidates(updated_full_session_chain: List[Dict[str, Any]], previous_sid: int,
                               local_used_sids: set[int], include_last: bool) -> List[int]:
    total_sessions = len(updated_full_session_chain)
    last_session_id = total_sessions - 1
    candidate_ids = []

    for sid in range(previous_sid + 1, total_sessions):
        if not include_last and sid == last_session_id:
            continue
        if sid in local_used_sids:
            continue
        candidate_ids.append(sid)

    return candidate_ids


def choose_best_relaxed_candidate(candidate_ids: List[int], previous_sid: int,
                                  gap_low: int, gap_high: int | None) -> int:
    if gap_high is None:
        candidate_ids.sort(key=lambda sid: sid - previous_sid, reverse=True)
        return candidate_ids[0]

    target_center = (gap_low + gap_high) / 2.0
    candidate_ids.sort(key=lambda sid: abs((sid - previous_sid) - target_center))
    return candidate_ids[0]


def choose_rule_session_id(updated_full_session_chain: List[Dict[str, Any]], previous_sid: int,
                           gap_low: int, gap_high: int | None, local_used_sids: set[int],
                           rng: random.Random) -> Tuple[int, bool]:
    strict_candidates = [
        sid for sid in build_rule_candidates(updated_full_session_chain, previous_sid, gap_low, gap_high, True)
        if sid not in local_used_sids
    ]
    if strict_candidates:
        return rng.choice(strict_candidates), True

    relaxed_candidates = [
        sid for sid in build_rule_candidates(updated_full_session_chain, previous_sid, gap_low, gap_high, False)
        if sid not in local_used_sids
    ]
    if relaxed_candidates:
        return choose_best_relaxed_candidate(relaxed_candidates, previous_sid, gap_low, gap_high), False

    later_unused_with_last = build_any_later_candidates(
        updated_full_session_chain=updated_full_session_chain,
        previous_sid=previous_sid,
        local_used_sids=local_used_sids,
        include_last=True,
    )
    if later_unused_with_last:
        return choose_best_relaxed_candidate(later_unused_with_last, previous_sid, gap_low, gap_high), False

    later_or_same_used = sorted([sid for sid in local_used_sids if sid >= previous_sid])
    if later_or_same_used:
        return later_or_same_used[0], False

    return previous_sid, False


def choose_distractor_session_id(updated_full_session_chain: List[Dict[str, Any]], left_bound: int, right_bound: int,
                                 local_used_sids: set[int], rng: random.Random) -> int | None:
    if right_bound <= left_bound + 1:
        return None

    candidate_sids = [
        sid for sid in range(left_bound + 1, right_bound)
        if sid not in local_used_sids and updated_full_session_chain[sid].get('Session_Type') == 'chitchat'
    ]
    if not candidate_sids:
        candidate_sids = [
            sid for sid in range(left_bound + 1, right_bound)
            if sid not in local_used_sids
        ]
    if not candidate_sids:
        candidate_sids = list(range(left_bound + 1, right_bound))
    if not candidate_sids:
        return None
    return rng.choice(candidate_sids)


def assign_and_inject_conditional_conflict_groups_interval(full_session_chain: List[Dict[str, Any]],
                                                           conditional_conflict_groups: List[Dict[str, Any]],
                                                           point_a_range: Tuple[int, int],
                                                           gap_low: int,
                                                           gap_high: int | None,
                                                           seed: int = 42) -> Tuple[List[Dict[str, Any]], List[int], int, int, int]:
    rng = random.Random(seed)
    updated_full_session_chain = copy.deepcopy(full_session_chain)
    total_sessions = len(updated_full_session_chain)
    min_a, max_a = point_a_range
    last_session_id = total_sessions - 1

    for session in updated_full_session_chain:
        if 'Conditional_Conflict_Information' not in session:
            session['Conditional_Conflict_Information'] = []

    point_a_candidates = [sid for sid in range(max(0, min_a), min(max_a + 1, total_sessions)) if sid != last_session_id]
    rng.shuffle(point_a_candidates)

    realized_rule_gaps: List[int] = []
    skipped_group_count = 0
    fallback_rule_count = 0
    applied_group_count = 0

    shuffled_groups = copy.deepcopy(conditional_conflict_groups)
    rng.shuffle(shuffled_groups)

    for group_idx, group in enumerate(shuffled_groups):
        conflict_id = group.get('Conflict_ID', f'CC_{group_idx + 1:03d}')
        pref_type = group.get('Preference_Type')
        rules = copy.deepcopy(group.get('Preference_Rules', []))
        distractors = copy.deepcopy(group.get('Distractors', []))

        if not rules:
            skipped_group_count += 1
            continue

        for idx, rule in enumerate(rules):
            if 'Rule_ID' not in rule:
                rule['Rule_ID'] = f'{conflict_id}_R{idx + 1}'

        point_a_session_id = point_a_candidates[group_idx % len(point_a_candidates)] if point_a_candidates else 0
        local_used_sids = {point_a_session_id}
        rule_session_ids = [point_a_session_id]
        injected_rules = [
            {
                'Conflict_ID': conflict_id,
                'Rule_ID': rules[0]['Rule_ID'],
                'Role': 'Point_A',
                'Preference_Type': pref_type,
                'Item': rules[0]['Item'],
                'Condition': rules[0]['Condition'],
            }
        ]

        for idx, rule in enumerate(rules[1:], start=1):
            previous_sid = rule_session_ids[-1]
            selected_sid, matched_target_range = choose_rule_session_id(
                updated_full_session_chain=updated_full_session_chain,
                previous_sid=previous_sid,
                gap_low=gap_low,
                gap_high=gap_high,
                local_used_sids=local_used_sids,
                rng=rng,
            )

            if not matched_target_range:
                fallback_rule_count += 1

            local_used_sids.add(selected_sid)
            rule_session_ids.append(selected_sid)
            realized_rule_gaps.append(selected_sid - previous_sid)
            role = f"Point_{chr(ord('B') + idx - 1)}"
            injected_rules.append({
                'Conflict_ID': conflict_id,
                'Rule_ID': rule['Rule_ID'],
                'Role': role,
                'Preference_Type': pref_type,
                'Item': rule['Item'],
                'Condition': rule['Condition'],
            })

        for injected_rule, sid in zip(injected_rules, rule_session_ids):
            updated_full_session_chain[sid]['Conditional_Conflict_Information'].append(injected_rule)

        left_bound = point_a_session_id
        right_bound = max(rule_session_ids)
        local_distractor_used = set(rule_session_ids)
        for distractor in distractors:
            distractor_sid = choose_distractor_session_id(
                updated_full_session_chain=updated_full_session_chain,
                left_bound=left_bound,
                right_bound=right_bound,
                local_used_sids=local_distractor_used,
                rng=rng,
            )
            if distractor_sid is None:
                continue
            local_distractor_used.add(distractor_sid)
            updated_full_session_chain[distractor_sid]['Conditional_Conflict_Information'].append({
                'Conflict_ID': conflict_id,
                'Role': 'Distractor',
                'Source_Person_ID': distractor.get('Source_Person_ID'),
                'Relationship_To_User': distractor.get('Relationship_To_User'),
                'Preference_Key': distractor.get('Preference_Key'),
                'Preference_Description': distractor.get('Preference_Description'),
            })

        applied_group_count += 1

    return updated_full_session_chain, realized_rule_gaps, skipped_group_count, fallback_rule_count, applied_group_count


def build_interval_metadata(interval_mode: str, gap_low: int, gap_high: int | None,
                            realized_rule_gaps: List[int], skipped_group_count: int,
                            fallback_rule_count: int, applied_group_count: int) -> Dict[str, Any]:
    metadata = {
        'Experiment': 'Conflict_Interval',
        'Step': 'Step3_2',
        'Interval_Mode': interval_mode,
        'Target_Gap_Range': [gap_low, gap_high if gap_high is not None else 'last'],
        'Applied_Group_Count': applied_group_count,
        'Applied_Rule_Gap_Count': len(realized_rule_gaps),
        'Skipped_Group_Count': skipped_group_count,
        'Fallback_Rule_Count': fallback_rule_count,
        'Realized_Rule_Gaps': realized_rule_gaps,
    }
    if realized_rule_gaps:
        metadata['Realized_Gap_Min'] = min(realized_rule_gaps)
        metadata['Realized_Gap_Max'] = max(realized_rule_gaps)
        metadata['Realized_Gap_Avg'] = round(sum(realized_rule_gaps) / len(realized_rule_gaps), 4)
    else:
        metadata['Realized_Gap_Min'] = None
        metadata['Realized_Gap_Max'] = None
        metadata['Realized_Gap_Avg'] = None
    return metadata


def merge_interval_metadata(previous_metadata: Any, current_metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(previous_metadata, dict) or not previous_metadata:
        return {'Step3_2': current_metadata}

    if 'Step3_1' in previous_metadata or 'Step3_2' in previous_metadata or 'Step3_3' in previous_metadata:
        merged = copy.deepcopy(previous_metadata)
        merged['Step3_2'] = current_metadata
        return merged

    previous_step = previous_metadata.get('Step')
    if previous_step == 'Step3_1':
        return {
            'Step3_1': copy.deepcopy(previous_metadata),
            'Step3_2': current_metadata,
        }

    return {
        'Previous': copy.deepcopy(previous_metadata),
        'Step3_2': current_metadata,
    }


def generate_single_conditional_conflict_interval(persona_item: Dict[str, Any], gap_low: int,
                                                  gap_high: int | None, seed: int,
                                                  point_a_range: Tuple[int, int],
                                                  interval_mode: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
    full_session_chain = copy.deepcopy(persona_item['Full_Session_Chain'])
    preference_profile = persona_item.get('Preference_Profile', {})
    others_profile = persona_item.get('Others_Profile', {})
    previous_cost = persona_item.get('token_cost')

    user_candidates = collect_user_conditional_preference_candidates(preference_profile)
    others_candidates = collect_others_conditional_preference_candidates(others_profile)

    user_groups = [
        {
            'Conflict_ID': f'CC_{i + 1:03d}',
            'Preference_Type': pref_type,
            'Preference_Rules': rules,
        }
        for i, (pref_type, rules) in enumerate(user_candidates.items())
    ]

    conflict_groups, cost_info = generate_conditional_conflict_groups_with_llm(
        user_groups=user_groups,
        others_candidates=others_candidates,
        previous_cost=previous_cost,
    )

    updated_full_session_chain, realized_rule_gaps, skipped_group_count, fallback_rule_count, applied_group_count = assign_and_inject_conditional_conflict_groups_interval(
        full_session_chain=full_session_chain,
        conditional_conflict_groups=conflict_groups,
        point_a_range=point_a_range,
        gap_low=gap_low,
        gap_high=gap_high,
        seed=seed,
    )

    interval_metadata = build_interval_metadata(
        interval_mode=interval_mode,
        gap_low=gap_low,
        gap_high=gap_high,
        realized_rule_gaps=realized_rule_gaps,
        skipped_group_count=skipped_group_count,
        fallback_rule_count=fallback_rule_count,
        applied_group_count=applied_group_count,
    )
    return updated_full_session_chain, cost_info, interval_metadata


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
                full_session_chain, cost_info, interval_metadata = generate_single_conditional_conflict_interval(
                    persona_item=persona_item,
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
                    'Conflict_Interval_Metadata': merge_interval_metadata(
                        persona_item.get('Conflict_Interval_Metadata'),
                        interval_metadata,
                    ),
                }
                results.append(result_item)
                print(
                    f"[DEBUG] Persona interval summary - applied groups: {interval_metadata['Applied_Group_Count']}, "
                    f"applied rule gaps: {interval_metadata['Applied_Rule_Gap_Count']}, "
                    f"fallback: {interval_metadata['Fallback_Rule_Count']}, skipped groups: {interval_metadata['Skipped_Group_Count']}, "
                    f"avg gap: {interval_metadata['Realized_Gap_Avg']}"
                )
            except Exception as e:
                print(f"[DEBUG] Failed to process persona {idx}: {e}:{traceback.format_exc()}")
                continue

        write_jsonl_items(output_file, results)
        write_json_items(output_json_file, results)
        print(f"[DEBUG] Successfully generated Step3_2 interval ablation ({interval_mode}) with {len(results)} personas.")
        return True
    except Exception as e:
        print(f"[DEBUG] Step3_2 interval ablation failed: {e}:{traceback.format_exc()}")
        return False


def main(args: argparse.Namespace) -> bool:
    if args.interval_mode == 'both':
        short_ok = run_single_mode(args, 'short')
        long_ok = run_single_mode(args, 'long')
        return short_ok and long_ok
    return run_single_mode(args, args.interval_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conflict interval ablation: Step3_2 conditional conflict construction.')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input Step3_1 interval JSONL file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSONL file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--output_json_file', type=str, default=None,
                        help='Output JSON file; if interval_mode=both, leave empty to use default per-mode names')
    parser.add_argument('--interval_mode', type=str, choices=['short', 'long', 'both'], default='both',
                        help='Conflict interval mode for the first ablation version')
    parser.add_argument('--point_a_min', type=int, default=0,
                        help='Lower bound of Point A session range')
    parser.add_argument('--point_a_max', type=int, default=9,
                        help='Upper bound of Point A session range')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for interval assignment')
    args = parser.parse_args()
    main(args)

