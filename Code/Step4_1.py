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
with open("MemConflict/Prompt/Prompt4_1.txt", 'r', encoding='utf-8') as f:
    Step4_1_Prompt = f.read()


def assign_session_event_types(session_item):
    """
    Assign one or more event types for one session based on session content.

    Returns:
        List[str]
    """

    session_type = session_item.get("Session_Type")
    event_types = []

    initial_attr_to_event = {
        "Residence": "Talking_About_Current_Residence",
        "Work_Status": "Talking_About_Current_Work_Rhythm",
        "Career_Status": "Talking_About_Current_Career",
        "Health_Status": "Talking_About_Current_Health",
        "Social_Status": "Talking_About_Recent_Social_Life",
        "Marital_Status": "Talking_About_Current_Relationship",
        "Children_Status": "Talking_About_Current_Family_Life"
    }

    update_attr_to_event = {
        "Residence": "Relocation_Update",
        "Work_Status": "Workload_Change_Update",
        "Career_Status": "Career_Change_Update",
        "Health_Status": "Health_Condition_Update",
        "Social_Status": "Social_Life_Change_Update",
        "Marital_Status": "Relationship_Status_Change",
        "Children_Status": "Family_Expansion_Update"
    }

    if session_type == "initial_reveal":
        revealed_attributes = session_item.get("Revealed_Attributes", {})
        if isinstance(revealed_attributes, dict):
            for attr_name in revealed_attributes.keys():
                if attr_name in initial_attr_to_event:
                    event_types.append(initial_attr_to_event[attr_name])

        if len(event_types) == 0:
            event_types.append("Initial_Life_Update")

    elif session_type == "update":
        updated_attributes = session_item.get("Updated_Attributes", [])
        if isinstance(updated_attributes, list):
            for update_item in updated_attributes:
                attr_name = update_item.get("Attribute")
                if attr_name in update_attr_to_event:
                    event_types.append(update_attr_to_event[attr_name])

        if len(event_types) == 0:
            event_types.append("General_Life_Update")

    elif session_type == "chitchat":
        chitchat_event_pool = [
            "Casual_Catch_Up",
            "Sharing_a_Small_Daily_Story",
            "Talking_About_Food_and_Mood",
            "Weekend_Plan_Chat",
            "Weather_and_Daily_Life_Chat",
            "Remembering_an_Old_Experience",
            "Talking_About_a_Hobby",
            "Light_Complaint_or_Funny_Incident",
            "Reflecting_on_Life_Lately",
            "Talking_About_a_Recent_Show_or_Movie"
        ]
        event_types.append(random.choice(chitchat_event_pool))

    elif session_type == "future_plan":
        event_types.append("Future_Planning_Conversation")

    else:
        event_types.append("General_Conversation")

    # 去重，保持顺序
    deduped_event_types = []
    for event_type in event_types:
        if event_type not in deduped_event_types:
            deduped_event_types.append(event_type)

    return deduped_event_types



def build_session_outline_input(session_item, event_types, life_goal):
    """
    Build minimal LLM input for one session outline based on Session_Type.
    Avoid passing full current state to reduce repeated memory leakage.
    """

    session_type = session_item.get("Session_Type")

    if session_type == "initial_reveal":
        llm_input = {
            "Session_Type": session_type,
            "Event_Types": event_types,
            "Revealed_Attributes": session_item.get("Revealed_Attributes", {})
        }

    elif session_type == "update":
        raw_updated_attributes = session_item.get("Updated_Attributes", [])
        simplified_updated_attributes = []

        if isinstance(raw_updated_attributes, list):
            for update_item in raw_updated_attributes:
                if not isinstance(update_item, dict):
                    continue

                simplified_updated_attributes.append({
                    "Attribute": update_item.get("Attribute"),
                    "After": copy.deepcopy(update_item.get("After"))
                })

        llm_input = {
            "Session_Type": session_type,
            "Event_Types": event_types,
            "Updated_Attributes": simplified_updated_attributes
        }

    elif session_type == "chitchat":
        llm_input = {
            "Session_Type": session_type,
            "Event_Types": event_types
        }

    elif session_type == "future_plan":
        llm_input = {
            "Session_Type": session_type,
            "Event_Types": event_types,
            "Life_Goal": life_goal
        }

    else:
        llm_input = {
            "Session_Type": session_type,
            "Event_Types": event_types
        }

    return llm_input


def build_session_outline(session_item, event_types, life_goal, previous_cost=None):
    """
    Use large language model to generate one session outline.

    Returns:
        session_outline: str
        cost_info: dict
    """

    try:
        print("[DEBUG] Sending session outline generation request to LLM...")

        llm_input = build_session_outline_input(
            session_item=session_item,
            event_types=event_types,
            life_goal=life_goal
        )

        User_Prompt = (
            "Input data:\n"
            f"{json.dumps(llm_input, ensure_ascii=False, indent=2)}"
        )

        json_markers = [
            "Corrected fixed part", "Corrected persona", "Corrected JSON",
            "Final JSON", "Complete JSON", "Correction result"
        ]

        outline_result, cost_info = llm_request(
            Step4_1_Prompt,
            User_Prompt,
            return_parsed_json=True,
            json_markers=json_markers
        )

        cost_info = calculate_cumulative_cost(previous_cost, cost_info)

        print("[DEBUG] Successfully generated session outline with LLM")

        if cost_info:
            current_cost = cost_info.get("current_stage", {})
            cumulative_cost = cost_info.get("cumulative", {})
            print(f"[DEBUG] Current stage - Input: {current_cost.get('input_tokens', 'N/A')}, "
                  f"Output: {current_cost.get('output_tokens', 'N/A')}, "
                  f"Cost: ${current_cost.get('total_cost_usd', 'N/A')}")
            print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
                  f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")

        return outline_result["Session_Outline"], cost_info

    except Exception as e:
        print(f"[DEBUG] Session outline generation failed: {e}:{traceback.format_exc()}")
        raise



def Generate_Single_Session_Event_Outlines(persona_item, previous_cost=None):
    """
    Step 4.1:
    For each session in Full_Session_Chain:
    1. assign event types
    2. generate session outline
    3. write back to session

    Returns:
        updated_full_session_chain
        cost_info
    """

    try:
        print("[DEBUG] Step 4.1 ---- Generating session event types and outlines...")


        full_session_chain = copy.deepcopy(persona_item["Full_Session_Chain"])
        life_goal = persona_item["Life_Goal"]

        cost_info = previous_cost

        for session_idx, session_item in enumerate(full_session_chain):
            print(f"[DEBUG] Processing session {session_idx + 1}/{len(full_session_chain)}")

            # ---- Module 1: assign event types ----
            event_types = assign_session_event_types(
                session_item=session_item
            )

            # ---- Module 2: build outline ----
            session_outline, cost_info = build_session_outline(
                session_item=session_item,
                event_types=event_types,
                life_goal=life_goal,
                previous_cost=cost_info
            )

            # ---- write back ----
            session_item["Event_Types"] = event_types
            session_item["Session_Outline"] = session_outline


        print("[DEBUG] Step 4.1 ---- Session event types and outlines generated successfully.")

        return full_session_chain, cost_info

    except Exception as e:
        print(f"[DEBUG] Generate_Single_Session_Event_Outlines failed: {e}:{traceback.format_exc()}")
        raise


def Generate_User_Session_Event_Outlines(args):
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

            updated_full_session_chain, cost_info = Generate_Single_Session_Event_Outlines(
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

        print("[DEBUG] Successfully processed Step 4.1 session event outlines")
        return True

    except Exception as e:
        print(f"Error processing Step 4.1: {e}:{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4.1 ---- Session Event Type and Outline Generation.")
    parser.add_argument("--input_file", type=str,
                        default="MemConflict/Data/Step3_3.jsonl",
                        help="Last Step output file for session outline generation")
    parser.add_argument("--output_file", type=str,
                        default="MemConflict/Data/Step4_1.jsonl",
                        help="Output JSONL file for session outline generation")
    parser.add_argument("--output_perfect_file", type=str,
                        default="MemConflict/Data_perfect/Step4_1.json",
                        help="Output JSON file for session outline generation")
    args = parser.parse_args()

    Generate_User_Session_Event_Outlines(args)
