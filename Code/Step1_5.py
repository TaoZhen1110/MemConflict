import argparse
import json
import jsonlines
import traceback
from llm_request import llm_request, calculate_cumulative_cost
from dotenv import load_dotenv
import logging
import copy

logger = logging.getLogger(__name__)

load_dotenv()


############### Load prompt
with open("MemConflict/Prompt/Prompt1_5.txt", 'r', encoding='utf-8') as f:
    Step1_5_Prompt = f.read()


def infer_gender_from_relation(relation_key="", relation_type=""):
    text = f"{relation_key} {relation_type}".lower()

    if "mother" in text or "sister" in text or "wife" in text or "daughter" in text:
        return "Female"
    if "father" in text or "brother" in text or "husband" in text or "son" in text:
        return "Male"
    return ""


def build_family_person_template(name="", gender="", birthdate="", relationship_to_user="", source_key=""):
    return {
        "Name": name,
        "Gender": gender,
        "Birthdate": birthdate,
        "Birthplace": "",
        "Education_Background": {
            "Highest_Degree": "",
            "Major": "",
            "University": ""
        },
        "Residence": "",
        "Career_Status": "",
        "Work_Status": "",
        "Health_Status": "",
        "Preference_Profile": {},
        "Relationship_To_User": relationship_to_user,
        "Source_Key": source_key
    }


def build_social_person_template(name="", gender="", birthdate="", relationship_to_user="", source_key=""):
    return {
        "Name": name,
        "Gender": gender,
        "Birthdate": birthdate,
        "Birthplace": "",
        "Education_Background": {
            "Highest_Degree": "",
            "Major": "",
            "University": ""
        },
        "Family_Information": {
            "Father_Alive": "",
            "Mother_Alive": "",
            "Has_Siblings": "",
            "Sibling_Count": ""
        },
        "Residence": "",
        "Marital_Status": {
            "Status": ""
        },
        "Children_Status": {
            "Has_Children": "",
            "Children_Count": ""
        },
        "Career_Status": "",
        "Work_Status": "",
        "Health_Status": "",
        "Preference_Profile": {},
        "Relationship_To_User": relationship_to_user,
        "Source_Key": source_key
    }


def Generate_Single_Other_Persona(Fixed_Profile, Dynamic_Profile):
    print("[DEBUG] Step 1.5: Generating others part...")

    Other_Persona = {}

    # 1. family members: parents / siblings
    Family_Information = copy.deepcopy(Fixed_Profile.get("Family_Information", {}))
    if not isinstance(Family_Information, dict):
        Family_Information = {}
    for key, value in Family_Information.items():
        if not isinstance(value, dict):
            continue
        relation_type = value.get("Type", key)

        Other_Persona[key] = build_family_person_template(
            name=value.get("Name", ""),
            gender=value.get("Gender", infer_gender_from_relation(key, relation_type)),
            birthdate=value.get("Birth_Date", value.get("Birthdate", "")),
            relationship_to_user=relation_type,
            source_key=key
        )

    # 2. partner / spouse -> family-related person
    Marital_Status = copy.deepcopy(Dynamic_Profile["Marital_Status"])
    if Marital_Status.get("Status") in ["Dating", "Married"]:
        Other_Persona["Partner"] = build_family_person_template(
            name=Marital_Status.get("Name", ""),
            gender=Marital_Status.get("Gender", ""),
            birthdate=Marital_Status.get("Birthdate", ""),
            relationship_to_user="Partner",
            source_key="Partner"
        )

    # 3. children -> family-related person
    Children_Status = copy.deepcopy(Dynamic_Profile["Children_Status"])
    if Children_Status.get("Status") == "Yes":
        for key, value in Children_Status.items():
            if key == "Status":
                continue

            Other_Persona[key] = build_family_person_template(
                name=value.get("Name", ""),
                gender=value.get("Gender", infer_gender_from_relation(key, "Child")),
                birthdate=value.get("Birthdate", value.get("Birth_Date", "")),
                relationship_to_user="Child",
                source_key=key
            )

    # 4. social contacts -> more complete social-person schema
    Social_Relationships = copy.deepcopy(Dynamic_Profile["Social_Relationships"])
    if "Contacts" in Social_Relationships:
        Contacts = Social_Relationships["Contacts"]
    else:
        Contacts = Social_Relationships

    for key, value in Contacts.items():
        relation_type = value.get("Type", "Social_Contact")

        Other_Persona[key] = build_social_person_template(
            name=value.get("Name", ""),
            gender=value.get("Gender", infer_gender_from_relation(key, relation_type)),
            birthdate=value.get("Birthdate", value.get("Birth_Date", "")),
            relationship_to_user=relation_type,
            source_key=key
        )

    return Other_Persona


def validate_and_correct_others_persona(Others_Information, previous_cost):
    """Use large language model to validate and correct others part persona information"""

    try:
        print("[DEBUG] Sending others part correction request to LLM...")

        User_Prompt = "Currently generated others information:\n" + \
                    f"{json.dumps(Others_Information, ensure_ascii=False, indent=2)}\n\n" + \
                    "Please analyze and correct the above persona information, " + \
                    "and present the final result only as valid JSON. The JSON must be wrapped " + \
                    "inside a Markdown code block: ```json```."

        json_markers = [
            "Corrected fixed part", "Corrected persona", "Corrected JSON",
            "Final JSON", "Complete JSON", "Correction result"
        ]

        corrected_others_persona, cost_info = llm_request(
            Step1_5_Prompt,
            User_Prompt,
            return_parsed_json=True,
            json_markers=json_markers
        )

        cost_info = calculate_cumulative_cost(previous_cost, cost_info)

        print(f"[DEBUG] Successfully processed others part with LLM caller")

        if cost_info:
            current_cost = cost_info.get('current_stage', {})
            cumulative_cost = cost_info.get('cumulative', {})
            print(f"[DEBUG] Current stage - Input: {current_cost.get('input_tokens', 'N/A')}, "
                  f"Output: {current_cost.get('output_tokens', 'N/A')}, "
                  f"Cost: ${current_cost.get('total_cost_usd', 'N/A')}")
            print(f"[DEBUG] Cumulative - Total tokens: {cumulative_cost.get('total_tokens', 'N/A')}, "
                  f"Total cost: ${cumulative_cost.get('total_cost_usd', 'N/A')}")

        return corrected_others_persona["Corrected_Others_Profile"], cost_info

    except Exception as e:
        print(f"Error during LLM request: {e}:{traceback.format_exc()}")
        return None, None


def Generate_Others_Profile(args):
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
            Personality = persona_item["Personality"]
            Life_Goal = persona_item["Life_Goal"]
            previous_cost = persona_item["token_cost"]

            Others_Information = Generate_Single_Other_Persona(Fixed_Profile, Dynamic_Profile)
            corrected_others_persona, cost_info = validate_and_correct_others_persona(
                Others_Information, previous_cost
            )

            print(f"[DEBUG] Others part generation completed")

            result_item = {
                "ID": ID,
                "Fixed_Profile": Fixed_Profile,
                "Dynamic_Profile": Dynamic_Profile,
                "Preference_Profile": Preference_Profile,
                "Personality": Personality,
                "Life_Goal": Life_Goal,
                "Others_Profile": corrected_others_persona,
                "metadata": persona_item["metadata"],
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
    parser = argparse.ArgumentParser(description='Step 1.5 ---- Others Information Construction.')
    parser.add_argument("--config_file", type=str,
                default="MemConflict/Data/Config.json",
                help="Configuration file for persona processing")
    parser.add_argument("--input_file", type=str,
                    default="MemConflict/Data/Step1_4.jsonl",
                    help="Last Step output file for persona processing")
    parser.add_argument("--output_file", type=str,
                    default="MemConflict/Data/Step1_5.jsonl",
                    help="Output JSON file for Others Information")
    parser.add_argument("--output_perfect_file", type=str,
                    default="MemConflict/Data_perfect/Step1_5.json",
                    help="Output JSON file for Others Information")
    args = parser.parse_args()

    Generate_Others_Profile(args)
