import argparse
import jsonlines
import json
import os
import random
import traceback
from datetime import datetime
from typing import Dict
from llm_request import llm_request, calculate_cumulative_cost
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

############### Load prompt
with open("MemConflict/Prompt/Prompt1_2.txt", 'r', encoding='utf-8') as f:
    Step1_2_Prompt = f.read()




def Generate_Single_Dynamic_Persona(Fixed_Profile) -> dict:
    print("[DEBUG] Step 2: Generating dynamic part...")
    with open(args.config_file, 'r', encoding="utf-8") as f:
        config = json.load(f)

    dynamic_config = config["Dynamic_Information"]

    # 1. select initial residence
    Birthplace = Fixed_Profile.get("Birthplace", "Unknown")
    Birth_City = Birthplace.split(",")[0].strip()
    Birth_Country = Birthplace.split(",")[-1].strip()

    all_countries = config["Fixed_Information"]["Birthplace"]["Countries"]
    all_cities = config["Fixed_Information"]["Birthplace"]["Cities"]

    # mostly stay in the birth country, but allow cross-country residence
    same_country_prob = 0.80

    if random.random() < same_country_prob:
        Current_Country = Birth_Country
        candidate_cities = all_cities.get(Current_Country, [])
        if candidate_cities:
            Current_City = random.choice(candidate_cities)
        else:
            Current_City = Birth_City
    else:
        other_countries = [c for c in all_countries if c != Birth_Country]
        if other_countries:
            Current_Country = random.choice(other_countries)
            candidate_cities = all_cities.get(Current_Country, [])
            if candidate_cities:
                Current_City = random.choice(candidate_cities)
            else:
                Current_City = Birth_City
        else:
            Current_Country = Birth_Country
            candidate_cities = all_cities.get(Current_Country, [])
            if candidate_cities:
                Current_City = random.choice(candidate_cities)
            else:
                Current_City = Birth_City

    Initial_Residence = Current_City + ", " + Current_Country


    # 2. select initial marital status
    Birthdate = Fixed_Profile["Birthdate"]
    Birth_Datetime = datetime.strptime(Birthdate, "%Y-%m-%d")

    marital_config = dynamic_config["Marital_Status"]
    Status = random.choice(marital_config["Status"])

    if Status in ["Dating", "Married"]:
        Partner_age_diff = random.randint(
            marital_config["Marital_Age_Diff_Range"][0],
            marital_config["Marital_Age_Diff_Range"][1]
        )

        direction = random.choice([-1, 1])
        Partner_Birth_Year = Birth_Datetime.year + direction * Partner_age_diff
        Partner_Birth_Month = random.randint(1, 12)
        Partner_Birth_Day = random.randint(1, 28)
        Partner_Birth = f"{Partner_Birth_Year:04d}-{Partner_Birth_Month:02d}-{Partner_Birth_Day:02d}"

        Marital_Status = {
            "Status": Status,
            "Name": "",
            "Birthdate": Partner_Birth
        }
    else:
        Marital_Status = {
            "Status": Status
        }

    # 3. select initial children status
    children_config = dynamic_config["Children_Status"]
    Has_Children = random.choice(children_config["Has_Children"])
    allowed_parent_status = children_config["Allowed_Parent_Status"]

    if Has_Children == "Yes" and Marital_Status["Status"] in allowed_parent_status:
        Num_Children = random.choices(
            list(children_config["Children_Count_Range"].keys()),
            weights=children_config["Children_Count_Range"].values()
        )[0]

        Children_Status = {
            "Status": "Yes"
        }

        for i in range(int(Num_Children)):
            Children_age_diff = random.randint(
                children_config["Parent_Child_Age_Gap_Range"][0],
                children_config["Parent_Child_Age_Gap_Range"][1]
            )

            Children_Birth_Year = Birth_Datetime.year + Children_age_diff
            Children_Birth_Month = random.randint(1, 12)
            Children_Birth_Day = random.randint(1, 28)
            Children_Birth = f"{Children_Birth_Year:04d}-{Children_Birth_Month:02d}-{Children_Birth_Day:02d}"

            Children_Status[f"Child_{i+1}"] = {
                "Name": "",
                "Birthdate": Children_Birth
            }
    else:
        Children_Status = {
            "Status": "No"
        }

    # 4. select initial career status
    career_config = dynamic_config["Career_Status"]

    Employment_Status = random.choices(
        list(career_config["Employment_Status"].keys()),
        weights=career_config["Employment_Status"].values(),
        k=1
    )[0]

    Career_Status_Result = {}

    if Employment_Status == "Employed":
        Company_Type = random.choices(
            list(career_config["Company_Types"].keys()),
            weights=career_config["Company_Types"].values(),
            k=1
        )[0]

        Job_Title = random.choices(
            list(career_config["Job_Titles"].keys()),
            weights=career_config["Job_Titles"].values(),
            k=1
        )[0]

        Industry = random.choices(
            list(career_config["Industries"].keys()),
            weights=career_config["Industries"].values(),
            k=1
        )[0]

        Company_Name = random.choice(career_config["Company_Names"].get(Company_Type, []))

        Monthly_Income = random.choices(
            list(career_config["Monthly_Income"].keys()),
            weights=career_config["Monthly_Income"].values(),
            k=1
        )[0]

        Savings_Amount = random.choices(
            list(career_config["Savings_Amount"].keys()),
            weights=career_config["Savings_Amount"].values(),
            k=1
        )[0]

        Career_Status_Result.update({
            "Employment_Status": Employment_Status,
            "Company_Name": Company_Name,
            "Job_Title": Job_Title,
            "Industry": Industry,
            "Monthly_Income": Monthly_Income,
            "Savings_Amount": Savings_Amount
        })

    elif Employment_Status == "Entrepreneur":
        Industry = random.choices(
            list(career_config["Industries"].keys()),
            weights=career_config["Industries"].values(),
            k=1
        )[0]

        Career_Status_Result.update({
            "Employment_Status": Employment_Status,
            "Company_Name": "Self-founded Company",
            "Job_Title": "Founder/CEO",
            "Industry": Industry,
            "Monthly_Income": "80000-120000",
            "Savings_Amount": "500000-1000000"
        })

    elif Employment_Status == "Unemployed":
        Savings_Amount = random.choices(
            list(career_config["Savings_Amount"].keys()),
            weights=career_config["Savings_Amount"].values(),
            k=1
        )[0]

        Career_Status_Result.update({
            "Employment_Status": Employment_Status,
            "Savings_Amount": Savings_Amount
        })

    # add career direction
    # Career_Direction = random.choices(
    #     list(career_config["Career_Direction"]["Direction"].keys()),
    #     weights=career_config["Career_Direction"]["Direction"].values(),
    #     k=1
    # )[0]
    # Career_Status_Result["Career_Direction"] = Career_Direction

    # 5. select initial work status
    work_config = dynamic_config["Work_Status"]
    Work_Status_Result = {
        "Current_State": random.choice(work_config["Current_State"])
    }

    # 6. select initial health status
    health_config = dynamic_config["Health_Status"]

    Physical_Health = random.choice(health_config["Physical_Health"])
    Mental_Health = random.choice(health_config["Mental_Health"])
    Health_Status_Result = {
        "Physical_Health": Physical_Health,
        "Mental_Health": Mental_Health
    }

    # 7. select initial Social_Relationships and Social_Status
    social_config = dynamic_config["Social_Relationships"]
    Relationship_Count = random.randint(
        social_config["Relationship_Count_Range"][0],
        social_config["Relationship_Count_Range"][1]
    )

    Contacts_Result = {}
    for i in range(Relationship_Count):
        Relationship_Type = random.choice(social_config["Relationship_Types"])
        Contacts_Result[f"Contacts_{i+1}"] = {
            "Name": "",
            "Type": Relationship_Type
        }

    Social_Status_Result = {
        "Current_State": random.choice(social_config["Social_Status"]["Current_State"])
    }

    Social_Relationships_Result = {
        "Contacts": Contacts_Result,
        "Social_Status": Social_Status_Result
    }

    All_Dynamic_Profile = {
        "Residence": Initial_Residence,
        "Marital_Status": Marital_Status,
        "Children_Status": Children_Status,
        "Career_Status": Career_Status_Result,
        "Work_Status": Work_Status_Result,
        "Health_Status": Health_Status_Result,
        "Social_Relationships": Social_Relationships_Result
    }

    print(All_Dynamic_Profile)

    return All_Dynamic_Profile




def validate_and_correct_dynamic_persona(persona_seed: str, dynamic_persona: Dict, previous_cost: Dict = None) -> tuple:
    """Use large language model to validate and correct dynamic part persona information"""

    try:
        print("[DEBUG] Sending dynamic part correction request to LLM...")

        User_Prompt = f"\nOriginal persona seed: {persona_seed}\n\n" + \
                       "Currently generated persona information:\n" + \
                       f"{json.dumps(dynamic_persona, ensure_ascii=False, indent=2)}\n\n" + \
                       "Please analyze and correct the above persona information, " + \
                       "and present the final result only as valid JSON. The JSON must be wrapped " + \
                       "inside a Markdown code block: ```json```."
        
        json_markers = [
            "Corrected fixed part", "Corrected persona", "Corrected JSON", 
            "Final JSON", "Complete JSON", "Correction result"
        ]

        corrected_dynamic_persona, cost_info = llm_request(
            Step1_2_Prompt, 
            User_Prompt, 
            return_parsed_json=True,
            json_markers=json_markers
        )

        cost_info = calculate_cumulative_cost(previous_cost, cost_info)
        
        print(f"[DEBUG] Successfully processed dynamic part with LLM caller")

        if cost_info:
            current_cost = cost_info.get('current_stage', {})
            cumulative_cost = cost_info.get('cumulative', {})
            print(f"[DEBUG] Current stage - Input: {current_cost.get('input_tokens', 'N/A')}, "
                  f"Output: {current_cost.get('output_tokens', 'N/A')}, "
                  f"Cost: ${current_cost.get('total_cost_usd', 'N/A')}")
            print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
                  f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")
        
        return corrected_dynamic_persona['Corrected_Dynamic_Persona'], cost_info
        
    except Exception as e:
        print(f"[DEBUG] Large language model validation failed: {e}:{traceback.format_exc()}")
        raise 




def Generate_User_Dynamic_Profile(args):
    print(f"Processing file: {args.input_file}")
    print(f"Output file: {args.output_file}")

    try:
        all_fixed_personas = []
        with jsonlines.open(args.input_file) as reader:
            for item in reader:
                all_fixed_personas.append(item)

        print(f"[DEBUG] Read {len(all_fixed_personas)} fixed personas")
        
        for fixed_item in all_fixed_personas:
            ID = fixed_item["ID"]
            Fixed_Profile = fixed_item["Fixed_Profile"]
            persona_seed = fixed_item["metadata"]["persona_seed"]
            previous_cost = fixed_item["token_cost"]
            All_Dynamic_Profile = Generate_Single_Dynamic_Persona(Fixed_Profile)
            corrected_dynamic_persona, cost_info = validate_and_correct_dynamic_persona(persona_seed, All_Dynamic_Profile, previous_cost)
            print(f"[DEBUG] Dynamic part generation completed")

            result_item = {
                "ID": ID,
                "Fixed_Profile": Fixed_Profile,
                "Dynamic_Profile": corrected_dynamic_persona,
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
    parser = argparse.ArgumentParser(description='Step 1.2 ---- User Dynamic Profile Construction.')
    parser.add_argument("--config_file", type=str, 
                default="MemConflict/Data/Config.json", 
                help="Configuration file for persona processing")
    parser.add_argument("--input_file", type=str, 
                    default="MemConflict/Data/Step1_1.jsonl", 
                    help="Last Step output file for persona processing")
    parser.add_argument("--output_file", type=str, 
                    default="MemConflict/Data/Step1_2.jsonl", 
                    help="Output JSON file for User Dynamic Persona")
    parser.add_argument("--output_perfect_file", type=str, 
                    default="MemConflict/Data_perfect/Step1_2.json", 
                    help="Output JSON file for User Dynamic Persona")
    args = parser.parse_args()

    Generate_User_Dynamic_Profile(args) 