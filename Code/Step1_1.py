import argparse
import json
import jsonlines
import os
import random
import traceback
import hashlib
from typing import Dict
from datetime import datetime
from llm_request import llm_request, calculate_cumulative_cost
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
REPO_ROOT = os.path.dirname(PROJECT_DIR)

############### Load prompt
with open(os.path.join(PROJECT_DIR, "Prompt", "Prompt1_1.txt"), 'r', encoding='utf-8') as f:
    Step1_1_Prompt = f.read()


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def is_seed_exists(output_file: str, persona_seed: str) -> bool:
    if not os.path.exists(output_file):
        return False
        
    try:
        with jsonlines.open(output_file) as reader:
            for item in reader:
                if not isinstance(item, dict):
                    continue
                existing_seed = item.get("persona_seed")
                if existing_seed is None:
                    existing_seed = item.get("metadata", {}).get("persona_seed")
                if existing_seed == persona_seed:
                    return True
    except Exception as e:
        print(f"[DEBUG] Error checking if seed exists: {e}:{traceback.format_exc()}")
    
    return False


def get_persona_seed_at_index(input_file: str, index: int) -> str:
    """Get persona seed at specified index"""
    try:
        with jsonlines.open(input_file) as reader:
            for i, obj in enumerate(reader):
                if i == index:
                    if isinstance(obj, dict) and 'persona' in obj:
                        return obj['persona']
                    break
        return None
    except Exception as e:
        print(f"[DEBUG] Error reading persona seed: {e}:{traceback.format_exc()}")
        return None


def get_random_persona_seed(args) -> tuple:

    """Randomly get an unprocessed persona seed"""
    line_count = 0
    with jsonlines.open(args.input_file) as reader:
        for _ in reader:
            line_count += 1

        if line_count == 0:
            print("[DEBUG] Input file is empty")
            return None, -1
    
    existing_seeds = set()
    if os.path.exists(args.output_file):
        try:
            with jsonlines.open(args.output_file) as reader:
                for item in reader:
                    if isinstance(item, dict) and 'metadata' in item and 'persona_seed' in item['metadata']:
                        existing_seeds.add(item['metadata']['persona_seed'])
        except Exception as e:
            print(f"[DEBUG] Error loading existing seeds: {e}:{traceback.format_exc()}")
    
    # Try random sampling at most 100 times
    max_attempts = 100
    attempts = 0

    while attempts < max_attempts:
        random_index = random.randint(0, line_count - 1)
        persona_seed = get_persona_seed_at_index(args.input_file, random_index)

        if persona_seed and persona_seed not in existing_seeds:
            print(f"[DEBUG] Successfully randomly sampled unprocessed persona seed, index: {random_index}")
            return persona_seed, random_index
    
        attempts += 1

    print(f"[DEBUG] Tried {max_attempts} times, no unprocessed persona seed found")
    return None, -1


def sync_output_perfect_file(output_file: str, output_perfect_file: str) -> None:
    items = []
    if os.path.exists(output_file):
        with jsonlines.open(output_file) as reader:
            for item in reader:
                items.append(item)

    os.makedirs(os.path.dirname(output_perfect_file), exist_ok=True)
    with open(output_perfect_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)



def Generate_Single_Fixed_Persona(persona_seed: str) -> Dict:
    with open(args.config_file, 'r', encoding="utf-8") as f:
        config = json.load(f)
    Fixed_Information = config["Fixed_Information"]

    # 0 Generate UUID
    hash_object = hashlib.sha256(persona_seed.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    persona_uuid = f"{hash_hex[:8]}-{hash_hex[8:12]}-{hash_hex[12:16]}-{hash_hex[16:20]}-{hash_hex[20:32]}"
    print(f"[DEBUG] Generated UUID: {persona_uuid}")

    # 1 Randomly select gender
    Gender = random.choice(Fixed_Information["Genders"])

    # 2 Randomly select Name
    Family = random.choice(Fixed_Information["Names"]["Family_Names"])
    if Gender == "Male":
        Name = Family + " " + random.choice(Fixed_Information["Names"]["Male_Names"])
    else:
        Name = Family + " " + random.choice(Fixed_Information["Names"]["Female_Names"])

    # 3 Randomly select Birthdate
    Years_Range = Fixed_Information["Birthdate"]["Years_Range"]
    Months_Range = Fixed_Information["Birthdate"]["Months_Range"]
    Days_Range = Fixed_Information["Birthdate"]["Days_Range"]

    Birth_Year = random.randint(Years_Range[0], Years_Range[1])
    Birth_Month = random.randint(Months_Range[0], Months_Range[1])
    Birth_Day = random.randint(Days_Range[0], Days_Range[1])

    Birth_Date = f"{Birth_Year:04d}-{Birth_Month:02d}-{Birth_Day:02d}"

    Birth_Datetime = datetime.strptime(Birth_Date, "%Y-%m-%d")
    Current_Datetime = datetime.now()
    Age = Current_Datetime.year - Birth_Datetime.year
    if Current_Datetime.month < Birth_Datetime.month or (Current_Datetime.month == Birth_Datetime.month and Current_Datetime.day < Birth_Datetime.day):
        Age -= 1

    # 4 Randomly select Birthplace
    Country = random.choice(Fixed_Information["Birthplace"]["Countries"])
    City = random.choice(Fixed_Information["Birthplace"]["Cities"].get(Country, []))
    Location = City + ", " + Country

    # 5 Randomly select Education
    Highest_Degree = random.choices(
        list(Fixed_Information["Education"]["Highest_Degree"].keys()),  
        weights=Fixed_Information["Education"]["Highest_Degree"].values(),  
        k=1  # 只选择一个元素
    )[0]
    Major = random.choice(Fixed_Information["Education"]["Major"])
    if City == "Hong Kong":
        University = random.choice(Fixed_Information["Education"]["University"].get("Hong Kong", []))
    else:
        University = random.choice(Fixed_Information["Education"]["University"].get(Country, []))

    Education_Information = {
        "Highest_Degree": Highest_Degree,
        "Major": Major,
        "University": University
    }

    # 6 Randomly select family information
    Family_Life = Fixed_Information["Family_Life"]

    ## 6.1 Randomly select family status
    Typical_Ages = sorted(Family_Life["Typical_Ages"])
    Age_based_Family_States = Family_Life["Age_based_Family_States"]

    Lower_age = 0
    Upper_age = 0

    if Age <= Typical_Ages[0]:
        Lower_age = Typical_Ages[0]
        Upper_age = Typical_Ages[0]
    elif Age >= Typical_Ages[-1]:
        Lower_age = Typical_Ages[-1]
        Upper_age = Typical_Ages[-1]
    else:
        for i in range(len(Typical_Ages) - 1):
            if Typical_Ages[i] <= Age <= Typical_Ages[i + 1]:
                Lower_age = Typical_Ages[i]
                Upper_age = Typical_Ages[i + 1]
                break
    
    if Lower_age == Upper_age:
        Family_States = Age_based_Family_States[str(Lower_age)]
    else:
        weight = (Age - Lower_age) / (Upper_age - Lower_age)
        Lower_states = Age_based_Family_States[str(Lower_age)]
        Upper_states = Age_based_Family_States[str(Upper_age)]

        def Interpolate_probabilities(lower_probs, upper_probs, weight):
            """Interpolate probabilities"""
            interpolated = {}
            for key in lower_probs.keys():
                interpolated[key] = lower_probs[key] * (1 - weight) + upper_probs[key] * weight
            return interpolated
        
        Parent_status_probs = Interpolate_probabilities(
            Lower_states['parent_status'], 
            Upper_states['parent_status'], 
            weight
        )

        Family_States = {
            'parent_status': Parent_status_probs
        }

    Parent_Status = random.choices(
        list(Family_States['parent_status'].keys()),
        weights=list(Family_States['parent_status'].values())
    )[0]
    

    ## 6.2 Add parent information
    Parent_Information = {}
    if Parent_Status != "both_deceased":
        if Parent_Status == "both_alive":
            Father_age_diff = random.randint(Family_Life["Parent_Age_Range"][0], Family_Life["Parent_Age_Range"][1])
            Father_Birth_Year = Birth_Datetime.replace(year=Birth_Datetime.year - Father_age_diff).year
            Father_Birth_Month = random.randint(Months_Range[0], Months_Range[1])
            Father_Birth_Day = random.randint(Days_Range[0], Days_Range[1])
            Father_Birth = f"{Father_Birth_Year:04d}-{Father_Birth_Month:02d}-{Father_Birth_Day:02d}"
            Parent_Information["Father"] = {
                "Name": "",
                "Birth_Date": Father_Birth
            }
            
            Mother_age_diff = random.randint(Family_Life["Parent_Age_Range"][0], Family_Life["Parent_Age_Range"][1])
            Mother_Birth_Year = Birth_Datetime.replace(year=Birth_Datetime.year - Mother_age_diff).year
            Mother_Birth_Month = random.randint(Months_Range[0], Months_Range[1])
            Mother_Birth_Day = random.randint(Days_Range[0], Days_Range[1])
            Mother_Birth = f"{Mother_Birth_Year:04d}-{Mother_Birth_Month:02d}-{Mother_Birth_Day:02d}"
            Parent_Information["Mother"] = {
                "Name": "",
                "Birth_Date": Mother_Birth
            }


        else:
            if random.choice([True, False]):
                Father_age_diff = random.randint(Family_Life["Parent_Age_Range"][0], Family_Life["Parent_Age_Range"][1])
                Father_Birth_Year = Birth_Datetime.replace(year=Birth_Datetime.year - Father_age_diff).year
                Father_Birth_Month = random.randint(Months_Range[0], Months_Range[1])
                Father_Birth_Day = random.randint(Days_Range[0], Days_Range[1])
                Father_Birth = f"{Father_Birth_Year:04d}-{Father_Birth_Month:02d}-{Father_Birth_Day:02d}"
                Parent_Information["Father"] = {
                    "Name": "",
                    "Birth_Date": Father_Birth
                }

            
            else:
                Mother_age_diff = random.randint(Family_Life["Parent_Age_Range"][0], Family_Life["Parent_Age_Range"][1])
                Mother_Birth_Year = Birth_Datetime.replace(year=Birth_Datetime.year - Mother_age_diff).year
                Mother_Birth_Month = random.randint(Months_Range[0], Months_Range[1])
                Mother_Birth_Day = random.randint(Days_Range[0], Days_Range[1])
                Mother_Birth = f"{Mother_Birth_Year:04d}-{Mother_Birth_Month:02d}-{Mother_Birth_Day:02d}"
                Parent_Information["Mother"] = {
                    "Name": "",
                    "Birth_Date": Mother_Birth
                }

    ## 6.3 Add sibling information
    Sibling_Count = random.choices(
        list(Fixed_Information["Family_Life"]["Sibling_Count_Distribution"].keys()),  
        weights=Fixed_Information["Family_Life"]["Sibling_Count_Distribution"].values(),  
        k=1
    )[0]

    Sibling_Information = {}
    if int(Sibling_Count) > 0:
        for i in range(int(Sibling_Count)):
            Sibling_age_diff = random.randint(Fixed_Information["Family_Life"]["Sibling_Age_Diff_Range"][0], Fixed_Information["Family_Life"]["Sibling_Age_Diff_Range"][1])
        
            if Sibling_age_diff == 0:
                Sibling_Birthdate = Birth_Datetime
                Sibling_Type = random.choice(Fixed_Information["Family_Life"]["Sibling_Type"])
                key = f"Sibling_{i+1}"
                Sibling_Information[key] = {
                    "Type": Sibling_Type,
                    "Name": "",
                    "Birth_Date": Sibling_Birthdate
                }
            else:
                direction = random.choice([-1, 1])
                Sibling_Birth_Year = Birth_Datetime.replace(year=Birth_Datetime.year + direction * Sibling_age_diff).year
                Sibling_Birth_Month = random.randint(Months_Range[0], Months_Range[1])
                Sibling_Birth_Day = random.randint(Days_Range[0], Days_Range[1])
                Sibling_Birthdate = f"{Sibling_Birth_Year:04d}-{Sibling_Birth_Month:02d}-{Sibling_Birth_Day:02d}"
                Sibling_Type = random.choice(Fixed_Information["Family_Life"]["Sibling_Type"])
                key = f"Sibling_{i+1}"
                Sibling_Information[key] = {
                    "Type": Sibling_Type,
                    "Name": "",
                    "Birth_Date": Sibling_Birthdate
                }


    Family_Information = {}
    if Parent_Information:
        Family_Information.update(Parent_Information)
    if Sibling_Information:
        Family_Information.update(Sibling_Information)


    All_Fixed_Profile = {
        "Name": Name,
        "Gender": Gender,
        "Birthdate": Birth_Date,
        "Birthplace": Location,
        "Education_Background": Education_Information
    }

    if Family_Information:
        All_Fixed_Profile["Family_Information"] = Family_Information
    
    print(All_Fixed_Profile)

    return persona_uuid, All_Fixed_Profile



def Validate_Correct_Single_Persona(persona_seed: str, generated_persona: Dict) -> tuple:
    try:
        print("[DEBUG] Sending correction request to LLM...")

        User_Prompt = f"\nOriginal persona seed: {persona_seed}\n\n" + \
                "Currently generated persona information:\n" + \
                f"```json\n{json.dumps(generated_persona, ensure_ascii=False, indent=2)}\n```\n\n" + \
                "Please analyze and correct the above persona information, " + \
                "and present the final result only as valid JSON. The JSON must be wrapped " + \
                "inside a Markdown code block: ```json```."
        
        json_markers = [ 
            "Corrected Profile", "Corrected persona", "Corrected JSON", 
            "Final JSON", "Complete JSON", "Correction result"
        ]

        corrected_persona_information, cost_info = llm_request(
            Step1_1_Prompt, 
            User_Prompt, 
            return_parsed_json=True,
            json_markers=json_markers
        )
        cost_info = calculate_cumulative_cost(None, cost_info)
        print(f"[DEBUG] Successfully processed persona with LLM caller")

        if cost_info:
            print(
                f"[DEBUG] Token usage - Input: {cost_info.get('cumulative', {}).get('input_tokens', 'N/A')}, "
                f"Output: {cost_info.get('cumulative', {}).get('output_tokens', 'N/A')}, "
                f"Cost: ${cost_info.get('cumulative', {}).get('total_cost_usd', 'N/A')}"
            )
        return corrected_persona_information['Corrected_Fixed_Persona'], cost_info

    except Exception as e:
        print(f"[DEBUG] Large language model validation failed: {e}")
        print("[DEBUG] Full traceback:")
        traceback.print_exc()
        raise  




def Generate_User_Fixed_Profile(seed_index: int = 0):
    print(f"Processing file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Processing index: {seed_index}")

    try:
        # 1. Load persona seed
        persona_seed = get_persona_seed_at_index(args.input_file, seed_index)
        if not persona_seed:
            print(f"[DEBUG] No persona seed found at index {seed_index}")
            return False
        print(f"[DEBUG] Processing persona seed: {persona_seed[:100]}...")

        # 2. Check if persona seed has already been processed
        if is_seed_exists(args.output_file, persona_seed):
            print(f"[DEBUG] This persona seed already exists, trying to randomly select another")
            # Randomly select an unprocessed seed
            new_seed, new_index = get_random_persona_seed(args)
            if new_seed:
                print(f"[DEBUG] Re-selected index: {new_index}")
                return Generate_User_Fixed_Profile(new_index)
            else:
                print("[DEBUG] No unprocessed persona seed found")
                return False
        
        # 3. generate initial user fixed single persona
        persona_uuid, detailed_persona = Generate_Single_Fixed_Persona(persona_seed)

        # 4. validate and correct the generated persona
        corrected_persona, cost_info = Validate_Correct_Single_Persona(persona_seed, detailed_persona)
        token_cost = cost_info
        result = {
            "ID": persona_uuid,
            "Fixed_Profile": corrected_persona,
            "metadata":{
                "persona_seed": persona_seed
            }
        }
        if token_cost:
            result["token_cost"] = token_cost

        print("[DEBUG] Processing completed")

        with jsonlines.open(args.output_file, 'a') as writer:
            writer.write(result)
        sync_output_perfect_file(args.output_file, args.output_perfect_file)
        
        print("[DEBUG] Successfully processed and saved persona")
        return True
    
    except Exception as e:
        print(f"Error processing persona: {e}:{traceback.format_exc()}")
        return False



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 1.1 ---- User Fixed Profile Construction.')
    parser.add_argument('--index', type=int, help='Persona index to process (optional)')
    parser.add_argument('--count', type=int, default=1, help='Number of unprocessed persona seeds to generate when index is not specified')
    parser.add_argument("--config_file", type=str, 
                    default="MemConflict/Data/Config.json", 
                    help="Configuration file for persona processing")
    parser.add_argument("--input_file", type=str, 
                    default="MemConflict/Data/Step0.jsonl", 
                    help="Input JSON file with persona seeds")
    parser.add_argument("--output_file", type=str, 
                    default="MemConflict/Data/Step1_1.jsonl", 
                    help="Output JSONL file for User Fixed Persona")
    parser.add_argument("--output_perfect_file", type=str, 
                    default="MemConflict/Data_perfect/Step1_1.json", 
                    help="Output JSON file for User Fixed Persona")
    args = parser.parse_args()
    args.config_file = resolve_repo_path(args.config_file)
    args.input_file = resolve_repo_path(args.input_file)
    args.output_file = resolve_repo_path(args.output_file)
    args.output_perfect_file = resolve_repo_path(args.output_perfect_file)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_perfect_file), exist_ok=True)

    if args.index is not None:
        print(f"Using specified index: {args.index}")
        Generate_User_Fixed_Profile(args.index)
    else:
        success_count = 0
        for _ in range(max(args.count, 1)):
            seed, index = get_random_persona_seed(args)
            if seed is None:
                print("No available persona seed found")
                break
            print(f"Randomly selected index: {index}")
            if Generate_User_Fixed_Profile(index):
                success_count += 1
        print(f"[DEBUG] Successfully generated {success_count} persona(s)")
