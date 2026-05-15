import copy
import argparse
import jsonlines
import json
import os
import random
import traceback
from datetime import datetime
from typing import Dict, List
from llm_request import llm_request, calculate_cumulative_cost
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

############### Load prompt
with open("MemConflict/Prompt/Prompt3_2.txt", 'r', encoding='utf-8') as f:
    Step3_2_Prompt = f.read()




def collect_user_conditional_preference_candidates(preference_profile: dict) -> dict:
    """
    Collect user conditional preference candidates from Preference_Profile.
    Input:
        preference_profile: dict, e.g.,
        {
            "Music_Preference": {
                "Lo-fi Focus Beats": "Morning Wake-up",
                "Smooth Jazz": "Evening Relaxation"
            },
            "Food_Preference": {
                "Kimchi Stew": "Feeling Nostalgic"
            }
        }
    Output:
        dict mapping Preference_Type to list of rules
        {
            "Music_Preference": [
                {"Item": "Lo-fi Focus Beats", "Condition": "Morning Wake-up"},
                {"Item": "Smooth Jazz", "Condition": "Evening Relaxation"}
            ],
            "Food_Preference": [
                {"Item": "Kimchi Stew", "Condition": "Feeling Nostalgic"}
            ]
        }
    """
    user_candidates = {}

    for pref_type, rules in preference_profile.items():
        rule_list = []
        for item, condition in rules.items():
            if item in [None, "", {}] or condition in [None, "", {}]:
                continue
            rule_list.append({"Item": copy.deepcopy(item), "Condition": copy.deepcopy(condition)})
        if rule_list:
            user_candidates[pref_type] = rule_list

    return user_candidates


def collect_others_conditional_preference_candidates(Others_Profile: dict) -> dict:
    """
    Collect other people's raw conditional preferences for Step 3.2.
    Input:
        Others_Profile: dict
        Example:
        {
            "Mother": {
                "Preference_Profile": {
                    "Reading": "Loves reading historical novels during the afternoons",
                    "Gardening": "Cultivates a small vegetable garden in her backyard"
                },
                "Relationship_To_User": "Mother",
                "Source_Key": "Mother"
            },
            ...
        }
    Output:
        dict mapping Preference_Type (raw keys) to list of candidate dicts:
        {
            "Reading": [
                {
                    "Source_Person_ID": "Mother",
                    "Relationship_To_User": "Mother",
                    "Preference_Key": "Reading",
                    "Preference_Description": "Loves reading historical novels during the afternoons"
                }
            ]
        }
    """
    others_candidates = {}

    if not isinstance(Others_Profile, dict):
        return others_candidates

    for person_id, person_info in Others_Profile.items():
        relationship_to_user = person_info.get("Relationship_To_User", person_id)
        pref_profile = person_info.get("Preference_Profile", {})

        if not isinstance(pref_profile, dict):
            continue

        for pref_key, pref_description in pref_profile.items():
            if pref_description in [None, "", {}]:
                continue

            if pref_key not in others_candidates:
                others_candidates[pref_key] = []

            others_candidates[pref_key].append({
                "Source_Person_ID": person_id,
                "Relationship_To_User": relationship_to_user,
                "Preference_Key": pref_key,
                "Preference_Description": copy.deepcopy(pref_description)
            })

    return others_candidates


def generate_conditional_conflict_groups_with_llm(user_groups: List[Dict],
                                                  others_candidates: List[Dict],
                                                  previous_cost: Dict = None) -> tuple:
    """
    Use LLM to select semantically relevant distractors for each user conditional conflict group.

    Args:
        user_groups: List of user conditional conflict groups, each with:
            - Conflict_ID
            - Preference_Type
            - Preference_Rules (list of dicts with Item + Condition)
        others_candidates: List of all other people's raw preference candidates
        previous_cost: optional, for cumulative cost tracking

    Returns:
        conflict_groups_with_distractors: List of groups with selected distractors
        cost_info: dict
    """
    try:
        print("[DEBUG] Sending conditional conflict groups request to LLM...")

        # Prepare input for LLM
        llm_input = {
            "User_Conditional_Groups": user_groups,
            "Others_Conditional_Candidates": others_candidates
        }

        User_Prompt = (
            "Input data:\n"
            f"{json.dumps(llm_input, ensure_ascii=False, indent=2)}\n\n"
            "Generate conditional conflict groups."
        )

        # JSON markers to locate response
        json_markers = [
            "Corrected fixed part", "Corrected persona", "Corrected JSON",
            "Final JSON", "Complete JSON", "Correction result"
        ]

        # Call LLM
        conflict_groups_with_distractors, cost_info = llm_request(
            Step3_2_Prompt,
            User_Prompt,
            return_parsed_json=True,
            json_markers=json_markers
        )

        cost_info = calculate_cumulative_cost(previous_cost, cost_info)

        print("[DEBUG] Successfully processed conditional conflict groups with LLM")

        if cost_info:
            current_cost = cost_info.get('current_stage', {})
            cumulative_cost = cost_info.get('cumulative', {})
            print(f"[DEBUG] Current stage - Input: {current_cost.get('input_tokens', 'N/A')}, "
                  f"Output: {current_cost.get('output_tokens', 'N/A')}, "
                  f"Cost: ${current_cost.get('total_cost_usd', 'N/A')}")
            print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
                  f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")
        

        return conflict_groups_with_distractors["Conditional_Conflict_Groups"], cost_info

    except Exception as e:
        print(f"[DEBUG] LLM generation of conditional conflict groups failed: {e}:{traceback.format_exc()}")
        raise



def assign_and_inject_conditional_conflict_groups(Full_Session_Chain: List[Dict], Conditional_Conflict_Groups: List[Dict], 
                                                  point_a_range=(0, 9), min_gap=5, seed=42) -> List[Dict]:
    """
    Assign positions for conditional conflict groups and inject all information directly into sessions.
    - Point A/B/C include Preference_Type
    - Distractors only contain Source_Person_ID, Preference_Key, Preference_Description
    """
    random.seed(seed)
    updated_full_session_chain = copy.deepcopy(Full_Session_Chain)
    total_sessions = len(updated_full_session_chain)
    min_a, max_a = point_a_range
    last_session_id = total_sessions - 1  # 最后一个 session 不插入任何信息

    # Initialize session markers
    for session in updated_full_session_chain:
        if "Conditional_Conflict_Information" not in session:
            session["Conditional_Conflict_Information"] = []

    # Shuffle conflict groups to avoid集中
    random.shuffle(Conditional_Conflict_Groups)

    # Precompute Point A session candidates
    point_a_candidates = [sid for sid in range(min_a, min(max_a + 1, total_sessions)) if sid != last_session_id]
    random.shuffle(point_a_candidates)  

    # Track used session per Preference_Type for min_gap
    last_used_sid_per_type = {}

    for group_idx, group in enumerate(Conditional_Conflict_Groups):
        conflict_id = group.get("Conflict_ID", f"CC_{group_idx+1:03d}")
        pref_type = group.get("Preference_Type")
        rules = group.get("Preference_Rules", [])
        distractors = group.get("Distractors", [])

        # Assign Rule_ID if missing
        for idx, rule in enumerate(rules):
            if "Rule_ID" not in rule:
                rule["Rule_ID"] = f"{conflict_id}_R{idx+1}"

        if pref_type not in last_used_sid_per_type:
            last_used_sid_per_type[pref_type] = []

        # ---- Point A ----
        point_a_sid = point_a_candidates[group_idx % len(point_a_candidates)]
        last_used_sid_per_type[pref_type].append(point_a_sid)
        updated_full_session_chain[point_a_sid]["Conditional_Conflict_Information"].append({
            "Conflict_ID": conflict_id,
            "Rule_ID": rules[0]["Rule_ID"],
            "Role": "Point_A",
            "Preference_Type": pref_type,
            "Item": rules[0]["Item"],
            "Condition": rules[0]["Condition"]
        })

        # ---- Other user rules (Point B/C) ----
        rule_session_ids = [point_a_sid]
        for idx, rule in enumerate(rules[1:], start=1):
            candidate_sids = [
                sid for sid, s in enumerate(updated_full_session_chain)
                if sid != last_session_id
                and sid > max(rule_session_ids)
                and all(abs(sid - prev) >= min_gap for prev in rule_session_ids)
            ]
            if not candidate_sids:
                candidate_sids = [
                    sid for sid in range(max(rule_session_ids) + 1, total_sessions - 1)
                ]  # exclude last session and preserve temporal order

            if not candidate_sids:
                continue

            random.shuffle(candidate_sids)
            selected_sid = candidate_sids[0]
            rule_session_ids.append(selected_sid)
            last_used_sid_per_type[pref_type].append(selected_sid)

            role = f"Point_{chr(ord('B') + idx - 1)}"
            updated_full_session_chain[selected_sid]["Conditional_Conflict_Information"].append({
                "Conflict_ID": conflict_id,
                "Rule_ID": rule["Rule_ID"],
                "Role": role,
                "Preference_Type": pref_type,
                "Item": rule["Item"],
                "Condition": rule["Condition"]
            })


        # ---- Distractors ----
        distractor_session_ids = []

        left_bound = point_a_sid
        right_bound = max(rule_session_ids)

        for distractor in distractors:
            # 必须放在 Point_A 和最后一个 point 之间
            candidate_sids = [
                sid for sid in range(left_bound + 1, right_bound)
                if sid != last_session_id
                and sid not in distractor_session_ids
                and updated_full_session_chain[sid]["Session_Type"] == "chitchat"
            ]

            if not candidate_sids:
                candidate_sids = [
                    sid for sid in range(left_bound + 1, right_bound)
                    if sid != last_session_id
                    and sid not in distractor_session_ids
                ]

            if not candidate_sids:
                continue

            random.shuffle(candidate_sids)
            distractor_sid = candidate_sids[0]
            distractor_session_ids.append(distractor_sid)

            updated_full_session_chain[distractor_sid]["Conditional_Conflict_Information"].append({
                "Conflict_ID": conflict_id,
                "Role": "Distractor",
                "Source_Person_ID": distractor.get("Source_Person_ID"),
                "Relationship_To_User": distractor.get("Relationship_To_User"),
                "Preference_Key": distractor.get("Preference_Key"),
                "Preference_Description": distractor.get("Preference_Description")
            })


    return updated_full_session_chain




def Generate_Single_Conditional_Conflict(persona_item: Dict, previous_cost: Dict = None) -> tuple:
    """
    Complete Step 3.2 pipeline for one persona:
    - collect user candidates
    - collect others candidates
    - generate conflict groups with LLM
    - assign and inject into session chain
    """
    try:
        print("[DEBUG] Step 3.2 ---- Generating conditional conflicts for persona")

        # Full session chain from previous steps
        full_session_chain = persona_item["Full_Session_Chain"]

        # Module 1: user candidates
        user_candidates = collect_user_conditional_preference_candidates(
            persona_item.get("Preference_Profile", {})
        )

        print(f"[DEBUG] User conditional candidates count per type: "
              f"{ {k: len(v) for k,v in user_candidates.items()} }")

        # Module 2: others candidates
        others_candidates = collect_others_conditional_preference_candidates(
            persona_item.get("Others_Profile", {})
        )

        print(f"[DEBUG] Others conditional candidates count per type: "
              f"{ {k: len(v) for k,v in others_candidates.items()} }")

        # Module 3: generate conflict groups via LLM
        conflict_groups, cost_info = generate_conditional_conflict_groups_with_llm(
            user_groups=[{
                "Conflict_ID": f"CC_{i+1:03d}",
                "Preference_Type": k,
                "Preference_Rules": v
            } for i,(k,v) in enumerate(user_candidates.items())],
            others_candidates=others_candidates,
            previous_cost=previous_cost
        )

        print(f"[DEBUG] Conditional conflict groups generated: {len(conflict_groups)}")

        # Module 4: assign and inject into session chain
        updated_full_session_chain = assign_and_inject_conditional_conflict_groups(
            Full_Session_Chain=full_session_chain,
            Conditional_Conflict_Groups=conflict_groups,
            point_a_range=(0,9),
            min_gap=5
        )

        print("[DEBUG] Conditional conflicts injected into full session chain successfully")

        return updated_full_session_chain, cost_info

    except Exception as e:
        print(f"[DEBUG] Generate_User_Conditional_Conflict failed: {e}")
        raise



def Generate_User_Conditional_Conflict(args):
    print(f"Processing file: {args.input_file}")
    print(f"Output file: {args.output_file}")

    try:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

        all_personas = []
        with jsonlines.open(args.input_file) as reader:
            for item in reader:
                all_personas.append(item)

        print(f"[DEBUG] Total personas loaded: {len(all_personas)}")


        for idx, persona_item in enumerate(all_personas):
            try:
                print(f"[DEBUG] Processing persona {idx + 1}/{len(all_personas)} ...")

                previous_cost = persona_item.get("token_cost", None)

                updated_full_session_chain, cost_info = Generate_Single_Conditional_Conflict(
                    persona_item=persona_item,
                    previous_cost=previous_cost
                )
            
                result_item = {
                    "ID": persona_item["ID"],
                    "Fixed_Profile": persona_item["Fixed_Profile"],
                    "Dynamic_Profile": persona_item["Dynamic_Profile"],
                    "Preference_Profile": persona_item["Preference_Profile"],
                    "Personality": persona_item["Personality"],
                    "Life_Goal": persona_item["Life_Goal"],
                    "Others_Profile": persona_item["Others_Profile"],
                    "Full_Session_Chain": updated_full_session_chain,
                    # "Static_Conflict_Profile": persona_item["Static_Conflict_Profile"],
                    # "Conditional_Conflict_Profile": conditional_conflict_profile,
                    "metadata": persona_item["metadata"],
                    "token_cost": cost_info
                }

                with jsonlines.open(args.output_file, 'a') as writer:
                    writer.write(result_item)

                with open(args.output_perfect_file, "a", encoding="utf-8") as f:
                    json.dump(result_item, f, ensure_ascii=False, indent=4)

            except Exception as e:
                print(f"[DEBUG] Failed to process persona {idx + 1}: {e}:{traceback.format_exc()}")
                continue

        print(f"[DEBUG] Conditional conflict generation finished successfully.")

    except Exception as e:
        print(f"[DEBUG] Generate_User_Conditional_Conflict failed: {e}:{traceback.format_exc()}")
        raise



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3.2 ---- Conditional Conflict Construction and Insertion.")
    parser.add_argument("--config_file", type=str,
                        default="MemConflict/Data/Config.json",
                        help="Configuration file for persona processing")
    parser.add_argument("--input_file", type=str,
                        default="MemConflict/Data/Step3_1.jsonl",
                        help="Last Step output file for persona processing")
    parser.add_argument("--output_file", type=str,
                        default="MemConflict/Data/Step3_2.jsonl",
                        help="Output JSONL file for conditional conflict insertion")
    parser.add_argument("--output_perfect_file", type=str,
                        default="MemConflict/Data_perfect/Step3_2.json",
                        help="Output JSON file for conditional conflict insertion")
    args = parser.parse_args()

    Generate_User_Conditional_Conflict(args)
