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
with open("MemConflict/Prompt/Prompt3_1.txt", 'r', encoding='utf-8') as f:
    Step3_1_Prompt = f.read()




def build_full_session_chain(Timeline_Initial_State, Timeline_Sessions):
    """
    Merge Timeline_Initial_State and Timeline_Sessions into one full session chain.
    """

    full_session_chain = []

    initial_items = []
    for _, node in Timeline_Initial_State.items():
        initial_items.append({
            "Date": node["Date"],
            "Session_Type": "initial_reveal",
            "Revealed_Attributes": copy.deepcopy(node["Revealed_Attributes"])
        })

    # sort initial reveals by date
    initial_items.sort(key=lambda x: x["Date"])

    # ---- Part 2: sort Timeline_Sessions by date ----
    regular_items = copy.deepcopy(Timeline_Sessions)
    regular_items.sort(key=lambda x: x["Date"])

    # ---- Part 3: merge and reassign Session_ID globally ----
    global_session_id = 0

    for item in initial_items:
        full_session_chain.append({
            "Session_ID": global_session_id,
            "Date": item["Date"],
            "Session_Type": item["Session_Type"],
            "Revealed_Attributes": item["Revealed_Attributes"]
        })
        global_session_id += 1

    for item in regular_items:
        new_item = copy.deepcopy(item)
        new_item["Session_ID"] = global_session_id
        full_session_chain.append(new_item)
        global_session_id += 1

    return full_session_chain



def collect_user_static_fact_candidates(Fixed_Profile):
    """
    Collect user static fact candidates from Fixed_Profile for Step 3.1 static conflict construction.
    """
    static_fact_candidates = []

    # ---- Basic Profile ----
    for field in ["Name", "Gender", "Birthdate", "Birthplace"]:
        value = Fixed_Profile.get(field)
        if value not in [None, "", {}]:
            static_fact_candidates.append({
                "Field_Path": field,
                "Value": copy.deepcopy(value)
            })

    # ---- Education_Background ----
    education_background = Fixed_Profile.get("Education_Background", {})
    for field in ["Highest_Degree", "Major", "University"]:
        value = education_background.get(field)
        if value not in [None, "", {}]:
            static_fact_candidates.append({
                "Field_Path": f"Education_Background.{field}",
                "Value": copy.deepcopy(value)
            })

    # ---- Family_Information ----
    family_information = Fixed_Profile.get("Family_Information", {})

    has_mother = "Yes" if "Mother" in family_information else "No"
    has_father = "Yes" if "Father" in family_information else "No"

    sibling_keys = sorted([
        key for key in family_information.keys()
        if isinstance(key, str) and key.startswith("Sibling_")
    ])
    sibling_count = len(sibling_keys)
    has_siblings = "Yes" if sibling_count > 0 else "No"

    static_fact_candidates.append({
        "Field_Path": "Family_Information.Has_Mother",
        "Value": has_mother
    })
    static_fact_candidates.append({
        "Field_Path": "Family_Information.Has_Father",
        "Value": has_father
    })
    static_fact_candidates.append({
        "Field_Path": "Family_Information.Has_Siblings",
        "Value": has_siblings
    })
    static_fact_candidates.append({
        "Field_Path": "Family_Information.Sibling_Count",
        "Value": sibling_count
    })

    # Mother / Father details
    for parent_key in ["Mother", "Father"]:
        if parent_key in family_information:
            parent_info = family_information[parent_key]

            if "Name" in parent_info and parent_info["Name"] not in [None, "", {}]:
                static_fact_candidates.append({
                    "Field_Path": f"Family_Information.{parent_key}.Name",
                    "Value": copy.deepcopy(parent_info["Name"])
                })

            if "Birth_Date" in parent_info and parent_info["Birth_Date"] not in [None, "", {}]:
                static_fact_candidates.append({
                    "Field_Path": f"Family_Information.{parent_key}.Birth_Date",
                    "Value": copy.deepcopy(parent_info["Birth_Date"])
                })

    # Sibling details
    for sibling_key in sibling_keys:
        sibling_info = family_information[sibling_key]

        if "Type" in sibling_info and sibling_info["Type"] not in [None, "", {}]:
            static_fact_candidates.append({
                "Field_Path": f"Family_Information.{sibling_key}.Type",
                "Value": copy.deepcopy(sibling_info["Type"])
            })

        if "Name" in sibling_info and sibling_info["Name"] not in [None, "", {}]:
            static_fact_candidates.append({
                "Field_Path": f"Family_Information.{sibling_key}.Name",
                "Value": copy.deepcopy(sibling_info["Name"])
            })

        if "Birth_Date" in sibling_info and sibling_info["Birth_Date"] not in [None, "", {}]:
            static_fact_candidates.append({
                "Field_Path": f"Family_Information.{sibling_key}.Birth_Date",
                "Value": copy.deepcopy(sibling_info["Birth_Date"])
            })

    return static_fact_candidates





def collect_others_static_fact_pool(Others_Profile):
    """
    Collect static fact pool from Others_Profile and group them by Field_Path.
    """

    others_static_fact_pool = {}

    def add_fact(field, field_path, value, source_person_id, relationship_to_user):
        if value in [None, "", {}]:
            return

        if field_path not in others_static_fact_pool:
            others_static_fact_pool[field_path] = []

        others_static_fact_pool[field_path].append({
            "Source_Person_ID": source_person_id,
            "Relationship_To_User": relationship_to_user,
            "Value": copy.deepcopy(value)
        })

    if not isinstance(Others_Profile, dict):
        return others_static_fact_pool

    for person_id, person_info in Others_Profile.items():
        if not isinstance(person_info, dict):
            continue

        relationship_to_user = person_info.get("Relationship_To_User", person_id)

        # ---- Basic Profile ----
        for field in ["Name", "Gender", "Birthdate", "Birthplace"]:
            value = person_info.get(field)
            add_fact(
                field=field,
                field_path=field,
                value=value,
                source_person_id=person_id,
                relationship_to_user=relationship_to_user
            )

        # ---- Education_Background ----
        education_background = person_info.get("Education_Background", {})
        if isinstance(education_background, dict):
            for field in ["Highest_Degree", "Major", "University"]:
                value = education_background.get(field)
                add_fact(
                    field=field,
                    field_path=f"Education_Background.{field}",
                    value=value,
                    source_person_id=person_id,
                    relationship_to_user=relationship_to_user
                )

        # ---- Family_Information (for contacts or others who have it) ----
        family_information = person_info.get("Family_Information", {})
        if isinstance(family_information, dict):
            for field in ["Father_Alive", "Mother_Alive", "Has_Siblings", "Sibling_Count"]:
                value = family_information.get(field)
                add_fact(
                    field=field,
                    field_path=f"Family_Information.{field}",
                    value=value,
                    source_person_id=person_id,
                    relationship_to_user=relationship_to_user
                )

    return others_static_fact_pool



def generate_static_conflict_triples(user_static_fact_candidates: List[Dict], others_static_fact_pool: Dict, 
                                     previous_cost: Dict = None) -> tuple:
    """Use large language model to generate multiple static conflict triples"""

    try:
        print("[DEBUG] Sending static conflict generation request to LLM...")

        llm_input = {
            "User_Static_Fact_Candidates": user_static_fact_candidates,
            "Others_Static_Fact_Pool_By_Field_Path": others_static_fact_pool
        }

        User_Prompt = (
            f"Number of static conflict triples to generate: {args.num_conflicts}\n\n"
            "Input data:\n"
            f"{json.dumps(llm_input, ensure_ascii=False, indent=2)}\n\n"
            "Generate static conflict triples."
        )

        json_markers = [
            "Corrected fixed part", "Corrected persona", "Corrected JSON",
            "Final JSON", "Complete JSON", "Correction result"
        ]

        static_conflict_result, cost_info = llm_request(
            Step3_1_Prompt,
            User_Prompt,
            return_parsed_json=True,
            json_markers=json_markers
        )

        cost_info = calculate_cumulative_cost(previous_cost, cost_info)

        print("[DEBUG] Successfully processed static conflict triples with LLM caller")

        if cost_info:
            current_cost = cost_info.get('current_stage', {})
            cumulative_cost = cost_info.get('cumulative', {})
            print(f"[DEBUG] Current stage - Input: {current_cost.get('input_tokens', 'N/A')}, "
                  f"Output: {current_cost.get('output_tokens', 'N/A')}, "
                  f"Cost: ${current_cost.get('total_cost_usd', 'N/A')}")
            print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
                  f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")

        return static_conflict_result["Static_Conflict_Triples"], cost_info


    except Exception as e:
        print(f"[DEBUG] Large language model static conflict generation failed: {e}:{traceback.format_exc()}")
        raise




def assign_and_inject_static_conflicts(Full_Session_Chain: List[Dict],
                                       Static_Conflict_Triples: List[Dict],
                                       point_a_range=(0, 9),
                                       seed=42) -> List[Dict]:

    random.seed(seed)
    updated_full_session_chain = copy.deepcopy(Full_Session_Chain)
    total_sessions = len(updated_full_session_chain)
    min_a, max_a = point_a_range
    last_session_id = total_sessions - 1  # 最后一个 session 不放任何信息

    for session in updated_full_session_chain:
        if "Static_Conflict_Information" not in session:
            session["Static_Conflict_Information"] = []

    # 这里只记录“已占用”，不再拿它做全局 min_gap 硬约束
    used_session_ids = set()

    random.shuffle(Static_Conflict_Triples)

    point_a_candidates = [
        sid for sid in range(max(0, min_a), min(max_a + 1, total_sessions))
        if sid != last_session_id
    ]
    random.shuffle(point_a_candidates)

    for idx, triple in enumerate(Static_Conflict_Triples):
        conflict_id = f"SC_{idx + 1:03d}"

        # ---- Step 1: choose Point A ----
        point_a_session_id = point_a_candidates[idx % len(point_a_candidates)]
        used_session_ids.add(point_a_session_id)

        updated_full_session_chain[point_a_session_id]["Static_Conflict_Information"].append({
            "Conflict_ID": conflict_id,
            "Role": "Point_A",
            "Target_Field_Path": triple.get("Target_Field_Path"),
            "Value": copy.deepcopy(triple.get("Point_A_Truth_Value"))
        })

        # ---- Step 2: choose Point B ----
        # 关键：必须保证 B 在 A 后面，并且至少相隔 min_gap
        candidate_b_ids = [
            sid for sid, s in enumerate(updated_full_session_chain)
            if sid != last_session_id
            and sid > point_a_session_id
            and (sid - point_a_session_id) >= args.min_gap
            and s["Session_Type"] == "chitchat"
            and sid not in used_session_ids
        ]

        if not candidate_b_ids:
            # fallback 1: 去掉 chitchat 限制，但仍要求在 A 后面且满足 min_gap
            candidate_b_ids = [
                sid for sid in range(total_sessions - 1)
                if sid > point_a_session_id
                and (sid - point_a_session_id) >= args.min_gap
                and sid not in used_session_ids
            ]

        if not candidate_b_ids:
            # fallback 2: 放松到只要在 A 后面
            candidate_b_ids = [
                sid for sid in range(total_sessions - 1)
                if sid > point_a_session_id
                and sid not in used_session_ids
            ]

        if not candidate_b_ids:
            # fallback 3: 允许复用，但仍要求在 A 后面
            candidate_b_ids = [
                sid for sid in range(total_sessions - 1)
                if sid > point_a_session_id
            ]

        if candidate_b_ids:
            point_b_session_id = random.choice(candidate_b_ids)
            used_session_ids.add(point_b_session_id)

            updated_full_session_chain[point_b_session_id]["Static_Conflict_Information"].append({
                "Conflict_ID": conflict_id,
                "Role": "Point_B",
                "Target_Field_Path": triple.get("Target_Field_Path"),
                "Value": copy.deepcopy(triple.get("Point_B_Conflict_Value"))
            })
        else:
            point_b_session_id = None

        # ---- Step 3: choose Distractor session ----
        distractor_info = triple.get("Distractor", None)
        if distractor_info and point_b_session_id is not None:
            # A 和 B 之间优先选未被占用的位置
            candidate_d_ids = [
                sid for sid in range(point_a_session_id + 1, point_b_session_id)
                if sid not in used_session_ids
            ]

            if not candidate_d_ids:
                # fallback: A 和 B 之间允许复用
                candidate_d_ids = list(range(point_a_session_id + 1, point_b_session_id))

            if candidate_d_ids:
                distractor_sid = random.choice(candidate_d_ids)

                updated_full_session_chain[distractor_sid]["Static_Conflict_Information"].append({
                    "Conflict_ID": conflict_id,
                    "Role": "Distractor",
                    "Source_Person_ID": distractor_info.get("Source_Person_ID"),
                    "Relationship_To_User": distractor_info.get("Relationship_To_User"),
                    "Target_Field_Path": distractor_info.get("Field"),
                    "Value": copy.deepcopy(distractor_info.get("Value"))
                })

                used_session_ids.add(distractor_sid)

    return updated_full_session_chain


def Generate_Single_Static_Conflict(persona_item: Dict, previous_cost: Dict = None) -> tuple:
    """
    Generate static conflict information for one persona.
    """

    try:
        print("[DEBUG] Building full session chain for static conflict generation...")

        fixed_profile = persona_item["Fixed_Profile"]
        others_profile = persona_item["Others_Profile"]
        timeline_initial_state = persona_item["Timeline_Initial_State"]
        timeline_sessions = persona_item["Timeline_Sessions"]

        # Step 1: merge full session chain
        full_session_chain = build_full_session_chain(
            Timeline_Initial_State=timeline_initial_state,
            Timeline_Sessions=timeline_sessions
        )

        print(f"[DEBUG] Full session chain size: {len(full_session_chain)}")

        # Step 2: collect user static fact candidates
        user_static_fact_candidates = collect_user_static_fact_candidates(
            Fixed_Profile=fixed_profile
        )

        print(f"[DEBUG] User static fact candidate count: {len(user_static_fact_candidates)}")

        # Step 3: collect others static fact pool
        others_static_fact_pool = collect_others_static_fact_pool(
            Others_Profile=others_profile
        )

        print(f"[DEBUG] Others static fact pool field-path count: {len(others_static_fact_pool)}")

        # Step 4: generate one static conflict triple with LLM
        static_conflict_triple, cost_info = generate_static_conflict_triples(
            user_static_fact_candidates=user_static_fact_candidates,
            others_static_fact_pool=others_static_fact_pool,
            previous_cost=previous_cost
        )

        print("[DEBUG] Static conflict triple generated successfully")

        # Step 5: assign positions and inject into session chain
        updated_full_session_chain = assign_and_inject_static_conflicts(
            Full_Session_Chain=full_session_chain,
            Static_Conflict_Triples=static_conflict_triple,
            point_a_range=(0, 9)
        )

        print("[DEBUG] Static conflict injected into full session chain successfully")

        return updated_full_session_chain, cost_info

    except Exception as e:
        print(f"[DEBUG] Generate_Single_Static_Conflict failed: {e}:{traceback.format_exc()}")
        raise



def Generate_User_Static_Conflict(args):
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

                full_session_chain, cost_info = Generate_Single_Static_Conflict(
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
                    "Full_Session_Chain": full_session_chain,
                    # "Static_Conflict_Profile": static_conflict_profile,
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


        print(f"[DEBUG] Static conflict generation finished successfully.")

    except Exception as e:
        print(f"[DEBUG] Generate_User_Static_Conflict failed: {e}:{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3.1 ---- Static Conflict Construction and Insertion.")
    parser.add_argument("--config_file", type=str,
                        default="MemConflict/Data/Config.json",
                        help="Configuration file for persona processing")
    parser.add_argument("--input_file", type=str,
                        default="MemConflict/Data/Step2_2.jsonl",
                        help="Last Step output file for persona processing")
    parser.add_argument("--output_file", type=str,
                        default="MemConflict/Data/Step3_1.jsonl",
                        help="Output JSONL file for static conflict insertion")
    parser.add_argument("--output_perfect_file", type=str,
                        default="MemConflict/Data_perfect/Step3_1.json",
                        help="Output JSON file for static conflict insertion")
    parser.add_argument("--num_conflicts", type=int,
                        default=12,
                        help="numbers of conflicts")
    parser.add_argument("--min_gap", type=int,
                        default=10,
                        help="Minimum distance between Point A and Point B")
    args = parser.parse_args()

    Generate_User_Static_Conflict(args)
