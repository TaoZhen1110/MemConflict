import argparse
import json
import jsonlines
import random
import traceback
import hashlib
import copy
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)



def build_deterministic_rng(persona_seed: str) -> random.Random:
    """Build a deterministic random generator based on persona_seed"""
    hash_hex = hashlib.sha256(persona_seed.encode("utf-8")).hexdigest()
    seed_int = int(hash_hex[:16], 16)
    return random.Random(seed_int)


def get_january_2022_dates():
    """Return all dates in January 2022"""
    start_date = datetime(2022, 1, 1)
    dates = []
    for i in range(31):
        current_date = start_date + timedelta(days=i)
        dates.append(current_date.strftime("%Y-%m-%d"))
    return dates


def split_attributes_into_groups(attributes, group_count, rng):
    """Randomly split attributes into several groups, each group has at least one attribute"""
    shuffled_attributes = copy.deepcopy(attributes)
    rng.shuffle(shuffled_attributes)

    groups = [[] for _ in range(group_count)]

    # Ensure each group has at least one attribute
    for i in range(group_count):
        groups[i].append(shuffled_attributes[i])

    # Distribute the remaining attributes randomly
    for attr in shuffled_attributes[group_count:]:
        random_group_index = rng.randint(0, group_count - 1)
        groups[random_group_index].append(attr)

    return groups


def Generate_Single_Initial_Timeline(Dynamic_Profile, persona_seed: str):
    print("[DEBUG] Step 2.1: Generating initial timeline state...")

    rng = build_deterministic_rng(persona_seed)

    dynamic_attributes = [
        "Residence",
        "Marital_Status",
        "Children_Status",
        "Career_Status",
        "Work_Status",
        "Health_Status",
        "Social_Relationships"
    ]


    # 1. choose several specific dates in January 2022
    january_dates = get_january_2022_dates()
    reveal_date_count = rng.randint(3, 6)
    selected_dates = sorted(rng.sample(january_dates, k=reveal_date_count))

    # 2. randomly distribute the 6 attributes onto these dates
    attribute_groups = split_attributes_into_groups(dynamic_attributes, reveal_date_count, rng)

    # 3. build initial reveal timeline
    Timeline_State = {}
    num = 0
    for date, attr_group in zip(selected_dates, attribute_groups):
        revealed_attributes = {}
        for attr in attr_group:
            revealed_attributes[attr] = copy.deepcopy(Dynamic_Profile[attr])

        Timeline_State[f"{num}"] = {
            "Date": date,
            "Revealed_Attributes": revealed_attributes
        }
        num += 1


    return Timeline_State





def Generate_User_Initial_Timeline(args):
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
            Others_Profile = persona_item["Others_Profile"]
            metadata = persona_item["metadata"]
            persona_seed = metadata["persona_seed"]
            previous_cost = persona_item.get("token_cost", None)

            Timeline_Initial_State = Generate_Single_Initial_Timeline(Dynamic_Profile, persona_seed)

            print(f"[DEBUG] Initial timeline state generation completed")

            result_item = {
                "ID": ID,
                "Fixed_Profile": Fixed_Profile,
                "Dynamic_Profile": Dynamic_Profile,
                "Preference_Profile": Preference_Profile,
                "Personality": Personality,
                "Life_Goal": Life_Goal,
                "Others_Profile": Others_Profile,
                "Timeline_Initial_State": Timeline_Initial_State,
                "metadata": metadata,
                "token_cost": previous_cost
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
    parser = argparse.ArgumentParser(description='Step 2.1 ---- Initial Timeline State Construction.')
    parser.add_argument("--config_file", type=str,
                        default="MemConflict/Data/Config.json",
                        help="Configuration file for persona processing")
    parser.add_argument("--input_file", type=str,
                        default="MemConflict/Data/Step1_5.jsonl",
                        help="Last Step output file for persona processing")
    parser.add_argument("--output_file", type=str,
                        default="MemConflict/Data/Step2_1.jsonl",
                        help="Output JSON file for Initial Timeline State")
    parser.add_argument("--output_perfect_file", type=str,
                        default="MemConflict/Data_perfect/Step2_1.json",
                        help="Output JSON file for Initial Timeline State")
    args = parser.parse_args()

    Generate_User_Initial_Timeline(args)