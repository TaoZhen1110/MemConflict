import copy
import argparse
import jsonlines
import json
import os
import random
import traceback
from llm_request import llm_request, calculate_cumulative_cost
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()


############### Load prompt
with open("MemConflict/Prompt/Prompt4_2.txt", 'r', encoding='utf-8') as f:
    Step4_2_Prompt = f.read()


def build_session_dialogue_input(session_item):
    """
    Build one session dialogue input for Step 4.2.

    Main framework:
    - Session_Type
    - Event_Types
    - Session_Outline
    - Revealed_Attributes / Updated_Attributes

    Additional information to mention naturally:
    - Static_Conflict_Information
    - Conditional_Conflict_Information
    - Others_Dynamic_Information
    """

    session_type = session_item.get("Session_Type")

    dialogue_input = {
        "Session_Type": session_type,
        "Event_Types": copy.deepcopy(session_item.get("Event_Types", [])),
        "Session_Outline": session_item.get("Session_Outline", "")
    }

    # ---- main session content ----
    if session_type == "initial_reveal":
        dialogue_input["Main_Session_Information"] = {
            "Revealed_Attributes": copy.deepcopy(session_item.get("Revealed_Attributes", {}))
        }
    elif session_type == "update":
        raw_updated_attributes = session_item.get("Updated_Attributes", [])
        simplified_updated_attributes = []

        for update_item in raw_updated_attributes:
            if isinstance(update_item, dict):
                simplified_updated_attributes.append({
                    "Attribute": update_item.get("Attribute"),
                    "After": copy.deepcopy(update_item.get("After"))
                })

        dialogue_input["Main_Session_Information"] = {
            "Updated_Attributes": simplified_updated_attributes
        }
    else:
        dialogue_input["Main_Session_Information"] = {}

    # ---- additional information that should be naturally mentioned ----
    additional_information = {}

    # ---- Static Conflict Information ----
    static_conflict_information = session_item.get("Static_Conflict_Information", [])
    simplified_static_conflict = []
    for item in static_conflict_information:
        if item.get("Role") == "Distractor" or item.get("Source_Person_ID"):
            simplified_static_conflict.append({
                "Source_Person_ID": item.get("Source_Person_ID"),
                "Relationship_To_User": item.get("Relationship_To_User"),
                "Target_Field_Path": item.get("Target_Field_Path"),
                "Value": item.get("Value")
            })
        else:  # 本人
            simplified_static_conflict.append({
                "Target_Field_Path": item.get("Target_Field_Path"),
                "Value": item.get("Value")
            })
    if simplified_static_conflict:
        additional_information["Static_Conflict_Information"] = simplified_static_conflict

    # ---- Conditional Conflict Information ----
    conditional_conflict_information = session_item.get("Conditional_Conflict_Information", [])
    simplified_conditional_conflict = []
    for item in conditional_conflict_information:
        if item.get("Role") == "Distractor" or item.get("Source_Person_ID"):
            simplified_conditional_conflict.append({
                "Source_Person_ID": item.get("Source_Person_ID"),
                "Relationship_To_User": item.get("Relationship_To_User"),
                "Preference_Key": item.get("Preference_Key"),
                "Preference_Description": item.get("Preference_Description")
            })
        else:  # 本人
            simplified_conditional_conflict.append({
                "Preference_Type": item.get("Preference_Type"),
                "Item": item.get("Item"),
                "Condition": item.get("Condition")
            })
    if simplified_conditional_conflict:
        additional_information["Conditional_Conflict_Information"] = simplified_conditional_conflict

    # ---- Others Dynamic Information ----
    others_dynamic_information = session_item.get("Others_Dynamic_Information", [])
    simplified_others_dynamic = []
    for item in others_dynamic_information:
        if item.get("Role") == "Distractor" or item.get("Source_Person_ID"):
            simplified_others_dynamic.append({
                "Source_Person_ID": item.get("Source_Person_ID"),
                "Relationship_To_User": item.get("Relationship_To_User"),
                "Attribute": item.get("Attribute"),
                "Value": item.get("Value")
            })
    if simplified_others_dynamic:
        additional_information["Others_Dynamic_Information"] = simplified_others_dynamic

    dialogue_input["Additional_Information_To_Mention"] = additional_information

    return dialogue_input






def generate_session_dialogue(dialogue_input, previous_cost=None):
    """
    Use large language model to generate one full session dialogue.
    """

    try:
        print("[DEBUG] Sending session dialogue generation request to LLM...")

        target_turn_num = random.randint(40, 50)

        print("target_turn_num", target_turn_num)

        User_Prompt = (
            f"Generate one complete session dialogue with exactly {target_turn_num} dialogue turns.\n\n"
            "Input data:\n"
            f"{json.dumps(dialogue_input, ensure_ascii=False, indent=2)}"
        )

        json_markers = [
            "Corrected fixed part", "Corrected persona", "Corrected JSON",
            "Final JSON", "Complete JSON", "Correction result"
        ]

        dialogue_result, cost_info = llm_request(
            Step4_2_Prompt,
            User_Prompt,
            return_parsed_json=True,
            json_markers=json_markers
        )

        cost_info = calculate_cumulative_cost(previous_cost, cost_info)

        print("[DEBUG] Successfully generated session dialogue with LLM")


        if cost_info:
            current_cost = cost_info.get("current_stage", {})
            cumulative_cost = cost_info.get("cumulative", {})
            print(f"[DEBUG] Current stage - Input: {current_cost.get('input_tokens', 'N/A')}, "
                  f"Output: {current_cost.get('output_tokens', 'N/A')}, "
                  f"Cost: ${current_cost.get('total_cost_usd', 'N/A')}")
            print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
                  f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")

        return dialogue_result, cost_info

    except Exception as e:
        print(f"[DEBUG] Session dialogue generation failed: {e}:{traceback.format_exc()}")
        raise


def Generate_Single_Session_Dialogues(persona_item, previous_cost=None):
    """
    Step 4.2:
    For each session in Full_Session_Chain:
    1. build dialogue input
    2. generate session dialogue
    3. write back to session

    Returns:
        updated_full_session_chain
        cost_info
    """

    try:
        print("[DEBUG] Step 4.2 ---- Generating full session dialogues...")

        full_session_chain = copy.deepcopy(persona_item["Full_Session_Chain"])
        cost_info = previous_cost

        for session_idx, session_item in enumerate(full_session_chain):
            print(f"[DEBUG] Processing session dialogue {session_idx + 1}/{len(full_session_chain)}")

            # ---- Module 1: build input ----
            dialogue_input = build_session_dialogue_input(
                session_item=session_item
            )

            # ---- Module 2: generate full dialogue ----
            session_dialogue, cost_info = generate_session_dialogue(
                dialogue_input=dialogue_input,
                previous_cost=cost_info
            )

            # ---- write back ----
            session_item["Session_Dialogue"] = session_dialogue

        print("[DEBUG] Step 4.2 ---- Session dialogues generated successfully.")

        return full_session_chain, cost_info

    except Exception as e:
        print(f"[DEBUG] Generate_Single_Session_Dialogues failed: {e}:{traceback.format_exc()}")
        raise


def Generate_User_Session_Dialogues(args):
    print(f"Processing file: {args.input_file}")
    print(f"Output file: {args.output_file}")

    try:
        all_personas = []
        with jsonlines.open(args.input_file) as reader:
            for item in reader:
                all_personas.append(item)

        print(f"[DEBUG] Read {len(all_personas)} personas")

        for persona_item in all_personas:
            previous_cost = persona_item.get("token_cost", None)

            updated_full_session_chain, cost_info = Generate_Single_Session_Dialogues(
                persona_item=persona_item,
                previous_cost=previous_cost
            )

            result_item = copy.deepcopy(persona_item)
            result_item["Full_Session_Chain"] = updated_full_session_chain
            result_item["token_cost"] = cost_info

            with jsonlines.open(args.output_file, "a") as writer:
                writer.write(result_item)

            with open(args.output_perfect_file, "a", encoding="utf-8") as f:
                json.dump(result_item, f, ensure_ascii=False, indent=4)

        print("[DEBUG] Successfully processed Step 4.2 session dialogue generation")
        return True

    except Exception as e:
        print(f"Error processing Step 4.2: {e}:{traceback.format_exc()}")
        return False



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4.2 ---- Full Session Dialogue Generation.")
    parser.add_argument("--input_file", type=str,
                        default="MemConflict/Data/Step4_1.jsonl",
                        help="Last Step output file for full session dialogue generation")
    parser.add_argument("--output_file", type=str,
                        default="MemConflict/Data/Step4_2.jsonl",
                        help="Output JSONL file for full session dialogue generation")
    parser.add_argument("--output_perfect_file", type=str,
                        default="MemConflict/Data_perfect/Step4_2.json",
                        help="Output JSON file for full session dialogue generation")
    args = parser.parse_args()

    Generate_User_Session_Dialogues(args)
