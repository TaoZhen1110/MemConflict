import json
import jsonlines
import random
import argparse
import traceback
from typing import Dict
from llm_request import llm_request, calculate_cumulative_cost
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

############### Load prompt
with open("MemConflict/Prompt/Prompt1_3.txt", 'r', encoding='utf-8') as f:
    Step1_3_Prompt = f.read()


def Generate_Single_Preference_Persona(args):
    print("[DEBUG] Step 3: Generating preference part...")
    with open(args.config_file, 'r', encoding="utf-8") as f:
        config = json.load(f)
    
    Preferences_Information = config["Preferences"]
    Preferences_Information_Result = {}
    Preferences_Count = random.randint(Preferences_Information["Preference_Types_Count"][0], 
                                        Preferences_Information["Preference_Types_Count"][1])

    Preference_Types = random.sample(list(Preferences_Information["Preference_Types"].keys()), k=Preferences_Count)

    for Preference_type in Preference_Types:
        # 1.1 select food preference
        if Preference_type == "Food_Preference":
            Food_Preference_Information = Preferences_Information["Preference_Types"]["Food_Preference"]
            Food_Items = random.sample(Food_Preference_Information["Food_Items"], k=random.randint(2,4))
            pairs = random.sample(Food_Preference_Information["Condition_Pairs"], k=len(Food_Items))
            Conditions = [random.choice(pair) for pair in pairs]

            food_to_condition = dict(zip(Food_Items, Conditions))
            Preferences_Information_Result["Food_Preference"] = food_to_condition
        
        # 1.2 select beverage preference
        elif Preference_type == "Beverage_Preference":
            Beverage_Preference_Information = Preferences_Information["Preference_Types"]["Beverage_Preference"]
            Beverage_Items = random.sample(Beverage_Preference_Information["Beverage_Items"], k=random.randint(2,4))
            pairs = random.sample(Beverage_Preference_Information["Condition_Pairs"], k=len(Beverage_Items))
            Conditions = [random.choice(pair) for pair in pairs]

            beverage_to_condition = dict(zip(Beverage_Items, Conditions))
            Preferences_Information_Result["Beverage_Preference"] = beverage_to_condition
        
        # 1.3 select reading preference
        elif Preference_type == "Reading_Preference":
            Reading_Preference_Information = Preferences_Information["Preference_Types"]["Reading_Preference"]
            Reading_Items = random.sample(Reading_Preference_Information["Reading_Items"], k=random.randint(2,4))
            pairs = random.sample(Reading_Preference_Information["Condition_Pairs"], k=len(Reading_Items))
            Conditions = [random.choice(pair) for pair in pairs]

            reading_to_condition = dict(zip(Reading_Items, Conditions))
            Preferences_Information_Result["Reading_Preference"] = reading_to_condition
        
        # 1.4 select music preference
        elif Preference_type == "Music_Preference":
            Music_Preference_Information = Preferences_Information["Preference_Types"]["Music_Preference"]
            Music_Items = random.sample(Music_Preference_Information["Music_Items"], k=random.randint(2,4))
            pairs = random.sample(Music_Preference_Information["Condition_Pairs"], k=len(Music_Items))
            Conditions = [random.choice(pair) for pair in pairs]

            music_to_condition = dict(zip(Music_Items, Conditions))
            Preferences_Information_Result["Music_Preference"] = music_to_condition
        
        # 1.5 select movie preference
        elif Preference_type == "Movie_Preference":
            Movie_Preference_Information = Preferences_Information["Preference_Types"]["Movie_Preference"]
            Movie_Items = random.sample(Movie_Preference_Information["Movie_Items"], k=random.randint(2,4))
            pairs = random.sample(Movie_Preference_Information["Condition_Pairs"], k=len(Movie_Items))
            Conditions = [random.choice(pair) for pair in pairs]

            movie_to_condition = dict(zip(Movie_Items, Conditions))
            Preferences_Information_Result["Movie_Preference"] = movie_to_condition
        
        # 1.6 select game preference
        elif Preference_type == "Game_Preference":
            Game_Preference_Information = Preferences_Information["Preference_Types"]["Game_Preference"]
            Game_Items = random.sample(Game_Preference_Information["Game_Items"], k=random.randint(2,4))
            pairs = random.sample(Game_Preference_Information["Condition_Pairs"], k=len(Game_Items))
            Conditions = [random.choice(pair) for pair in pairs]

            game_to_condition = dict(zip(Game_Items, Conditions))
            Preferences_Information_Result["Game_Preference"] = game_to_condition
        
        # 1.7 select sport preference
        elif Preference_type == "Sport_Preference":
            Sport_Preference_Information = Preferences_Information["Preference_Types"]["Sport_Preference"]
            Sport_Items = random.sample(Sport_Preference_Information["Sport_Items"], k=random.randint(2,4))
            pairs = random.sample(Sport_Preference_Information["Condition_Pairs"], k=len(Sport_Items))
            Conditions = [random.choice(pair) for pair in pairs]

            sport_to_condition = dict(zip(Sport_Items, Conditions))
            Preferences_Information_Result["Sport_Preference"] = sport_to_condition
        
        # 1.8 select clothing preference
        elif Preference_type == "Clothing_Preference":
            Clothing_Preference_Information = Preferences_Information["Preference_Types"]["Clothing_Preference"]
            Clothing_Items = random.sample(Clothing_Preference_Information["Clothing_Items"], k=random.randint(2,4))
            pairs = random.sample(Clothing_Preference_Information["Condition_Pairs"], k=len(Clothing_Items))
            Conditions = [random.choice(pair) for pair in pairs]

            clothing_to_condition = dict(zip(Clothing_Items, Conditions))
            Preferences_Information_Result["Clothing_Preference"] = clothing_to_condition
        
        # 1.9 select Pet preference
        elif Preference_type == "Pet_Preference":
            Pet_Preference_Information = Preferences_Information["Preference_Types"]["Pet_Preference"]
            Pet_Items = random.sample(Pet_Preference_Information["Pet_Items"], k=random.randint(2,4))
            pairs = random.sample(Pet_Preference_Information["Condition_Pairs"], k=len(Pet_Items))
            Conditions = [random.choice(pair) for pair in pairs]

            pet_to_condition = dict(zip(Pet_Items, Conditions))
            Preferences_Information_Result["Pet_Preference"] = pet_to_condition
        
        # 1.10 select travel preference
        elif Preference_type == "Travel_Preference":
            Travel_Preference_Information = Preferences_Information["Preference_Types"]["Travel_Preference"]
            Travel_Items = random.sample(Travel_Preference_Information["Travel_Items"], k=random.randint(2,4))
            pairs = random.sample(Travel_Preference_Information["Condition_Pairs"], k=len(Travel_Items))
            Conditions = [random.choice(pair) for pair in pairs]

            travel_to_condition = dict(zip(Travel_Items, Conditions))
            Preferences_Information_Result["Travel_Preference"] = travel_to_condition

    return Preferences_Information_Result


def validate_and_correct_preference_persona(persona_seed: str, preference_persona: Dict, previous_cost: Dict = None) -> tuple:
    """Use large language model to validate and correct preference part persona information"""
    try:
        print("[DEBUG] Sending preference part correction request to LLM...")

        User_Prompt = f"\nOriginal persona seed: {persona_seed}\n\n" + \
                       "Currently generated persona information:\n" + \
                       f"{json.dumps(preference_persona, ensure_ascii=False, indent=2)}\n\n" + \
                       "Please analyze and correct the above persona information, " + \
                       "and present the final result only as valid JSON. The JSON must be wrapped " + \
                       "inside a Markdown code block: ```json```."  

        json_markers = [
            "Corrected fixed part", "Corrected persona", "Corrected JSON", 
            "Final JSON", "Complete JSON", "Correction result"
        ]

        corrected_preference_persona, cost_info = llm_request(
            Step1_3_Prompt, 
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
        
        return corrected_preference_persona['Corrected_Preference_Persona'], cost_info                   

    except Exception as e:
        print(f"[DEBUG] Large language model validation failed: {e}:{traceback.format_exc()}")
        raise 


def Generate_User_Preference_Profile(args):
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
            persona_seed = persona_item["metadata"]["persona_seed"]
            previous_cost = persona_item["token_cost"]

            All_Preference_Profile = Generate_Single_Preference_Persona(args)

            print(All_Preference_Profile)
            corrected_preference_persona, cost_info = validate_and_correct_preference_persona(persona_seed, All_Preference_Profile, previous_cost)
            print(f"[DEBUG] Preference part generation completed")

            result_item = {
                "ID": ID,
                "Fixed_Profile": Fixed_Profile,
                "Dynamic_Profile": Dynamic_Profile,
                "Preference_Profile": corrected_preference_persona,
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
    parser = argparse.ArgumentParser(description='Step 1.3 ---- User Preference Profile Construction.')
    parser.add_argument("--config_file", type=str, 
                default="MemConflict/Data/Config.json", 
                help="Configuration file for persona processing")
    parser.add_argument("--input_file", type=str, 
                    default="MemConflict/Data/Step1_2.jsonl", 
                    help="Last Step output file for persona processing")
    parser.add_argument("--output_file", type=str, 
                    default="MemConflict/Data/Step1_3.jsonl", 
                    help="Output JSON file for User Preference Persona")
    parser.add_argument("--output_perfect_file", type=str, 
                    default="MemConflict/Data_perfect/Step1_3.json", 
                    help="Output JSON file for User Preference Persona")
    args = parser.parse_args()

    Generate_User_Preference_Profile(args) 