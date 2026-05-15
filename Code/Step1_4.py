import argparse
import json
import jsonlines
import traceback
import random
from llm_request import llm_request, calculate_cumulative_cost
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

############### Load prompt
with open("MemConflict/Prompt/Prompt1_4.txt", 'r', encoding='utf-8') as f:
    Step1_4_Prompt = f.read()

def Generate_Single_Goal_Persona(args):
    print("[DEBUG] Step 1.4: Generating goal part...")
    with open(args.config_file, 'r', encoding="utf-8") as f:
        config = json.load(f)

    # 1. Randomly select MBTI
    Personality_Information = config["Personality"]
    MBTI_Type = random.choice(Personality_Information["MBTI_Types"])
    MBTI_Tags = Personality_Information["MBTI_Tags"].get(MBTI_Type, [])
    Personality = {
        "MBTI": MBTI_Type,
        "MBTI_Tags": MBTI_Tags
    }

    # 2. Randomly select goal
    life_goal_type = random.choice(config["life_goal"]["life_goal_types"])
    Life_Goal = {
        "Type": life_goal_type,
        "Description": ""
    }
    return Personality, Life_Goal


def validate_and_correct_goal_persona(persona_seed, Fixed_Profile, Dynamic_Profile,
                                      Personality, Life_Goal, previous_cost):
    """Use large language model to validate and correct goal part persona information"""

    Career_Status = Dynamic_Profile["Career_Status"]
    try:
        print("[DEBUG] Sending goal part correction request to LLM...")
        User_Prompt = f"Original persona seed: {persona_seed}\n" + \
                    f"Fixed Profile: {json.dumps(Fixed_Profile, ensure_ascii=False)}\n" + \
                    f"Career Status: {json.dumps(Career_Status, ensure_ascii=False)}\n" + \
                    f"Personality: {json.dumps(Personality, ensure_ascii=False)}\n" + \
                    "Currently generated goal information:\n" + \
                    f"{json.dumps(Life_Goal, ensure_ascii=False, indent=2)}\n\n" + \
                    "Please analyze and correct the above persona information, " + \
                    "and present the final result only as valid JSON. The JSON must be wrapped " + \
                    "inside a Markdown code block: ```json```."
        
        json_markers = [
            "Corrected fixed part", "Corrected persona", "Corrected JSON", 
            "Final JSON", "Complete JSON", "Correction result"
        ]

        corrected_goal_persona, cost_info = llm_request(
            Step1_4_Prompt, 
            User_Prompt, 
            return_parsed_json=True,
            json_markers=json_markers
        )        

        cost_info = calculate_cumulative_cost(previous_cost, cost_info)
        
        print(f"[DEBUG] Successfully processed preference part with LLM caller")

        if cost_info:
            current_cost = cost_info.get('current_stage', {})
            cumulative_cost = cost_info.get('cumulative', {})
            print(f"[DEBUG] Current stage - Input: {current_cost.get('input_tokens', 'N/A')}, "
                  f"Output: {current_cost.get('output_tokens', 'N/A')}, "
                  f"Cost: ${current_cost.get('total_cost_usd', 'N/A')}")
            print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
                  f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")

        return corrected_goal_persona['Corrected_Life_Goal'], cost_info        

    except Exception as e:
        print(f"[DEBUG] Large language model validation failed: {e}:{traceback.format_exc()}")
        raise 


def Generate_User_Goal_Profile(args):
    print(f"Processing file: {args.input_file}")
    print(f"Output file: {args.output_file}")

    try:
        all_personas = []
        with jsonlines.open(args.input_file) as reader:
            for item in reader:
                all_personas.append(item)

        print(f"[DEBUG] Read {len(all_personas)} personas")

        for persona_item in all_personas:
            ID = persona_item["ID"]
            Fixed_Profile = persona_item["Fixed_Profile"]
            Dynamic_Profile = persona_item["Dynamic_Profile"]
            Preference_Profile = persona_item["Preference_Profile"]
            persona_seed = persona_item["metadata"]["persona_seed"]
            previous_cost = persona_item["token_cost"]

            Personality, Life_Goal = Generate_Single_Goal_Persona(args)

            corrected_goal_persona, cost_info = validate_and_correct_goal_persona(persona_seed, Fixed_Profile, Dynamic_Profile,
                                                                                    Personality, Life_Goal, previous_cost)
            print(f"[DEBUG] Goal part generation completed")

            result_item = {
                "ID": ID,
                "Fixed_Profile": Fixed_Profile,
                "Dynamic_Profile": Dynamic_Profile,
                "Preference_Profile": Preference_Profile,
                "Personality": Personality,
                "Life_Goal": corrected_goal_persona,
                "metadata": {
                    "persona_seed": persona_seed
                },
                "token_cost": cost_info
            }
            
            with jsonlines.open(args.output_file, 'a') as writer:
                writer.write(result_item)
            with open(args.output_perfect_file, "a", encoding="utf-8") as f:
                json.dump(result_item, f, ensure_ascii=False, indent=4)

        print("[DEBUG] Successfully processed and saved persona")
        return True
        
    except Exception as e:
        print(f"Error processing persona: {e}:{traceback.format_exc()}")
        return False       



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 1.4 ---- User Goal Profile Construction.')
    parser.add_argument("--config_file", type=str, 
                default="MemConflict/Data/Config.json", 
                help="Configuration file for persona processing")
    parser.add_argument("--input_file", type=str, 
                    default="MemConflict/Data/Step1_3.jsonl", 
                    help="Last Step output file for persona processing")
    parser.add_argument("--output_file", type=str, 
                    default="MemConflict/Data/Step1_4.jsonl", 
                    help="Output JSON file for User Goal")
    parser.add_argument("--output_perfect_file", type=str, 
                    default="MemConflict/Data_perfect/Step1_4.json", 
                    help="Output JSON file for User Goal")
    args = parser.parse_args()

    Generate_User_Goal_Profile(args)     