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







def collect_dynamic_attribute_session_distribution(Full_Session_Chain):
    """
    Collect dynamic attribute occurrences from Full_Session_Chain.

    Two sources are considered:
    1. initial reveal sessions: session["Revealed_Attributes"]
       -> treated as Point A
    2. update sessions: session["Updated_Attributes"]
       -> treated as later points (Point B / C / D ...)
    """

    attribute_session_distribution = {}
    attribute_update_details = {}

    dynamic_attr_name_map = {
        "Residence": "Residence",
        "Marital_Status": "Marital_Status",
        "Children_Status": "Children_Status",
        "Career_Status": "Career_Status",
        "Work_Status": "Work_Status",
        "Health_Status": "Health_Status",
        "Social_Status": "Social_Status"
    }

    def add_occurrence(attr_name, session_id, session_date, source_type, before_value, after_value):
        if attr_name not in attribute_session_distribution:
            attribute_session_distribution[attr_name] = []

        if attr_name not in attribute_update_details:
            attribute_update_details[attr_name] = []

        attribute_session_distribution[attr_name].append(session_id)
        attribute_update_details[attr_name].append({
            "Session_ID": session_id,
            "Date": session_date,
            "Source_Type": source_type,
            "Before": copy.deepcopy(before_value),
            "After": copy.deepcopy(after_value)
        })

    for session in Full_Session_Chain:
        session_id = session.get("Session_ID")
        session_date = session.get("Date")
        session_type = session.get("Session_Type")

        # ---- Part 1: initial reveals ----
        revealed_attributes = session.get("Revealed_Attributes", {})
        if isinstance(revealed_attributes, dict) and len(revealed_attributes) > 0:
            for raw_attr_name, raw_value in revealed_attributes.items():
                if raw_attr_name in dynamic_attr_name_map:
                    attr_name = dynamic_attr_name_map[raw_attr_name]
                    add_occurrence(
                        attr_name=attr_name,
                        session_id=session_id,
                        session_date=session_date,
                        source_type="initial_reveal",
                        before_value=None,
                        after_value=raw_value
                    )

        # ---- Part 2: later updates ----
        updated_attributes = session.get("Updated_Attributes", [])
        if isinstance(updated_attributes, list) and len(updated_attributes) > 0:
            for update_item in updated_attributes:
                if not isinstance(update_item, dict):
                    continue

                attr_name = update_item.get("Attribute")
                if attr_name in [None, "", {}]:
                    continue

                add_occurrence(
                    attr_name=attr_name,
                    session_id=session_id,
                    session_date=session_date,
                    source_type=session_type if session_type else "update",
                    before_value=update_item.get("Before"),
                    after_value=update_item.get("After")
                )

    # sort by session id
    for attr_name in attribute_session_distribution:
        attribute_session_distribution[attr_name] = sorted(attribute_session_distribution[attr_name])

    for attr_name in attribute_update_details:
        attribute_update_details[attr_name] = sorted(
            attribute_update_details[attr_name],
            key=lambda x: x["Session_ID"]
        )

    return attribute_session_distribution, attribute_update_details




def collect_others_dynamic_attribute_pool(Others_Profile):
    """
    Collect other people's dynamic-like attribute facts for Step 3.3 distraction insertion.
    """

    others_dynamic_attribute_pool = {}

    def add_fact(attribute_name, source_person_id, relationship_to_user, value):
        if value in [None, "", {}]:
            return

        if attribute_name not in others_dynamic_attribute_pool:
            others_dynamic_attribute_pool[attribute_name] = []

        others_dynamic_attribute_pool[attribute_name].append({
            "Source_Person_ID": source_person_id,
            "Relationship_To_User": relationship_to_user,
            "Value": copy.deepcopy(value)
        })

    if not isinstance(Others_Profile, dict):
        return others_dynamic_attribute_pool

    for person_id, person_info in Others_Profile.items():
        if not isinstance(person_info, dict):
            continue

        relationship_to_user = person_info.get("Relationship_To_User", person_id)

        # ---- Residence ----
        add_fact(
            attribute_name="Residence",
            source_person_id=person_id,
            relationship_to_user=relationship_to_user,
            value=person_info.get("Residence")
        )

        # ---- Career_Status ----
        add_fact(
            attribute_name="Career_Status",
            source_person_id=person_id,
            relationship_to_user=relationship_to_user,
            value=person_info.get("Career_Status")
        )

        # ---- Work_Status ----
        add_fact(
            attribute_name="Work_Status",
            source_person_id=person_id,
            relationship_to_user=relationship_to_user,
            value=person_info.get("Work_Status")
        )

        # ---- Health_Status ----
        add_fact(
            attribute_name="Health_Status",
            source_person_id=person_id,
            relationship_to_user=relationship_to_user,
            value=person_info.get("Health_Status")
        )

        # ---- Marital_Status ----
        if "Marital_Status" in person_info:
            add_fact(
                attribute_name="Marital_Status",
                source_person_id=person_id,
                relationship_to_user=relationship_to_user,
                value=person_info.get("Marital_Status")
            )

        # ---- Children_Status ----
        if "Children_Status" in person_info:
            add_fact(
                attribute_name="Children_Status",
                source_person_id=person_id,
                relationship_to_user=relationship_to_user,
                value=person_info.get("Children_Status")
            )

    return others_dynamic_attribute_pool



def assign_and_inject_dynamic_distractors(Full_Session_Chain,
                                          attribute_session_distribution,
                                          attribute_update_details,
                                          others_dynamic_attribute_pool,
                                          seed=42):
    """
    Insert other people's same-attribute distractors into the gaps
    between adjacent true points of the same attribute.

    True points include:
    - initial reveal sessions (Point A)
    - later update sessions (Point B / C / D ...)
    """

    random.seed(seed)
    updated_full_session_chain = copy.deepcopy(Full_Session_Chain)

    if not updated_full_session_chain:
        raise ValueError("Full_Session_Chain is empty.")

    total_sessions = len(updated_full_session_chain)
    last_session_id = total_sessions - 1

    for session in updated_full_session_chain:
        if "Others_Dynamic_Information" not in session:
            session["Others_Dynamic_Information"] = []

    used_distractor_session_ids = set()

    for attr_name, detail_list in attribute_update_details.items():
        if attr_name not in others_dynamic_attribute_pool:
            continue

        if len(detail_list) < 2:
            continue

        other_candidates = others_dynamic_attribute_pool.get(attr_name, [])
        if len(other_candidates) == 0:
            continue

        # detail_list 已经按 Session_ID 排序
        for pair_idx in range(len(detail_list) - 1):
            left_detail = detail_list[pair_idx]
            right_detail = detail_list[pair_idx + 1]

            left_sid = left_detail["Session_ID"]
            right_sid = right_detail["Session_ID"]

            # 必须有中间区间才能插空
            if right_sid <= left_sid + 1:
                continue

            if left_sid == last_session_id or right_sid == last_session_id:
                continue

            left_after = left_detail.get("After")
            right_after = right_detail.get("After")

            # ---- Step 1: choose distractor candidate ----
            filtered_candidates = []
            for candidate in other_candidates:
                candidate_value = candidate.get("Value")

                # 避免和左右真实值完全一致
                if candidate_value == left_after or candidate_value == right_after:
                    continue

                filtered_candidates.append(candidate)

            if len(filtered_candidates) == 0:
                filtered_candidates = other_candidates

            distractor_info = random.choice(filtered_candidates)

            # ---- Step 2: choose insertion session between adjacent true points ----
            candidate_session_ids = [
                sid for sid in range(left_sid + 1, right_sid)
                if sid != last_session_id
                and sid not in used_distractor_session_ids
                and updated_full_session_chain[sid]["Session_Type"] == "chitchat"
            ]

            if len(candidate_session_ids) == 0:
                candidate_session_ids = [
                    sid for sid in range(left_sid + 1, right_sid)
                    if sid != last_session_id
                    and sid not in used_distractor_session_ids
                ]

            if len(candidate_session_ids) == 0:
                candidate_session_ids = [
                    sid for sid in range(left_sid + 1, right_sid)
                    if sid != last_session_id
                ]

            if len(candidate_session_ids) == 0:
                continue

            distractor_session_id = random.choice(candidate_session_ids)
            used_distractor_session_ids.add(distractor_session_id)

            # ---- Step 3: inject distractor ----
            updated_full_session_chain[distractor_session_id]["Others_Dynamic_Information"].append({
                "Attribute": attr_name,
                "Role": "Distractor",
                "Source_Person_ID": distractor_info.get("Source_Person_ID"),
                "Relationship_To_User": distractor_info.get("Relationship_To_User"),
                "Value": copy.deepcopy(distractor_info.get("Value")),
                "Linked_Left_Session_ID": left_sid,
                "Linked_Right_Session_ID": right_sid
            })

    return updated_full_session_chain



def Generate_Single_Dynamic_Distractors(persona_item):
    """
    Step 3.3:
    Insert other people's same-attribute dynamic distractors into the user's timeline
    by using the gap-insertion strategy.

    Input:
        persona_item:
            must contain:
            - Full_Session_Chain
            - Others_Profile

    Output:
        updated_full_session_chain
    """

    try:
        print("[DEBUG] Step 3.3 ---- Collecting dynamic attribute session distribution...")

        full_session_chain = copy.deepcopy(persona_item["Full_Session_Chain"])
        others_profile = persona_item["Others_Profile"]

        # ---- Module 1 ----
        attribute_session_distribution, attribute_update_details = collect_dynamic_attribute_session_distribution(
            Full_Session_Chain=full_session_chain
        )

        print(f"[DEBUG] Dynamic attribute types found: {list(attribute_session_distribution.keys())}")

        # ---- Module 2 ----
        print("[DEBUG] Step 3.3 ---- Collecting others' dynamic attribute pool...")

        others_dynamic_attribute_pool = collect_others_dynamic_attribute_pool(
            Others_Profile=others_profile
        )

        print(f"[DEBUG] Others dynamic attribute types found: {list(others_dynamic_attribute_pool.keys())}")

        # ---- Module 3 ----
        print("[DEBUG] Step 3.3 ---- Assigning and injecting dynamic distractors...")

        updated_full_session_chain = assign_and_inject_dynamic_distractors(
            Full_Session_Chain=full_session_chain,
            attribute_session_distribution=attribute_session_distribution,
            attribute_update_details=attribute_update_details,
            others_dynamic_attribute_pool=others_dynamic_attribute_pool
        )

        print("[DEBUG] Step 3.3 ---- Dynamic distractors inserted successfully.")

        return updated_full_session_chain

    except Exception as e:
        print(f"[DEBUG] Generate_Single_Dynamic_Distractors failed: {e}:{traceback.format_exc()}")
        raise


def Generate_User_Dynamic_Distractors(args):
    print(f"Processing file: {args.input_file}")
    print(f"Output file: {args.output_file}")

    try:
        all_personas = []
        with jsonlines.open(args.input_file) as reader:
            for item in reader:
                all_personas.append(item)

        print(f"[DEBUG] Read {len(all_personas)} personas")

        for persona_item in all_personas:
            result_item = copy.deepcopy(persona_item)

            updated_full_session_chain = Generate_Single_Dynamic_Distractors(
                persona_item=persona_item
            )

            result_item["Full_Session_Chain"] = updated_full_session_chain

            with jsonlines.open(args.output_file, 'a') as writer:
                writer.write(result_item)

            with open(args.output_perfect_file, "a", encoding="utf-8") as f:
                json.dump(result_item, f, ensure_ascii=False, indent=4)

        print("[DEBUG] Successfully processed Step 3.3 dynamic distractor insertion")
        return True

    except Exception as e:
        print(f"Error processing Step 3.3: {e}:{traceback.format_exc()}")
        return False



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3.3 ---- Dynamic Distractor Insertion.")
    parser.add_argument("--input_file", type=str,
                        default="MemConflict/Data/Step3_2.jsonl",
                        help="Last Step output file for dynamic distractor insertion")
    parser.add_argument("--output_file", type=str,
                        default="MemConflict/Data/Step3_3.jsonl",
                        help="Output JSONL file for dynamic distractor insertion")
    parser.add_argument("--output_perfect_file", type=str,
                        default="MemConflict/Data_perfect/Step3_3.json",
                        help="Output JSON file for dynamic distractor insertion")
    args = parser.parse_args()

    Generate_User_Dynamic_Distractors(args)
