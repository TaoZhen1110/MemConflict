import argparse
import copy
import hashlib
import json
import jsonlines
import random
import traceback
from datetime import datetime

import logging

logger = logging.getLogger(__name__)


def build_deterministic_rng(persona_seed: str) -> random.Random:
    hash_hex = hashlib.sha256(persona_seed.encode("utf-8")).hexdigest()
    seed_int = int(hash_hex[:16], 16)
    return random.Random(seed_int)


def weighted_choice(rng, weight_dict):
    items = list(weight_dict.keys())
    weights = list(weight_dict.values())
    return rng.choices(items, weights=weights, k=1)[0]


def collect_excluded_names(current_state, fixed_profile):
    excluded_names = set()

    protagonist_name = fixed_profile.get("Name", "")
    if protagonist_name:
        excluded_names.add(protagonist_name)

    marital_info = current_state.get("Marital_Status", {})
    partner_name = marital_info.get("Name", "")
    if partner_name:
        excluded_names.add(partner_name)

    children_info = current_state.get("Children_Status", {})
    for key, value in children_info.items():
        if key.startswith("Child_") and isinstance(value, dict):
            child_name = value.get("Name", "")
            if child_name:
                excluded_names.add(child_name)

    family_info = fixed_profile.get("Family_Information", {})
    if isinstance(family_info, dict):
        for _, person_info in family_info.items():
            if isinstance(person_info, dict):
                name = person_info.get("Name", "")
                if name:
                    excluded_names.add(name)

    return excluded_names


def sample_full_name(rng, config, gender=None, excluded_names=None):
    if excluded_names is None:
        excluded_names = set()

    names_config = config["Fixed_Information"]["Names"]
    family_names = names_config["Family_Names"]
    male_names = names_config["Male_Names"]
    female_names = names_config["Female_Names"]

    if gender == "Male":
        given_name_pool = male_names
    elif gender == "Female":
        given_name_pool = female_names
    else:
        given_name_pool = male_names + female_names

    candidates = []
    for family_name in family_names:
        for given_name in given_name_pool:
            full_name = f"{given_name} {family_name}"
            if full_name not in excluded_names:
                candidates.append(full_name)

    if len(candidates) == 0:
        # fallback: allow reuse if all names are excluded
        family_name = rng.choice(family_names)
        given_name = rng.choice(given_name_pool)
        return f"{given_name} {family_name}"

    return rng.choice(candidates)


def random_date_in_month(year: int, month: int, rng: random.Random) -> str:
    day = rng.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def build_monthly_session_dates(rng: random.Random):
    dates = []
    for year in range(2022, 2026):
        start_month = 2 if year == 2022 else 1
        end_month = 12
        for month in range(start_month, end_month + 1):
            dates.append(random_date_in_month(year, month, rng))
    return dates


def build_session_type_sequence(rng: random.Random, total_sessions: int):
    update_count = int(total_sessions * 0.65)
    chitchat_count = total_sessions - update_count

    session_types = ["update"] * update_count + ["chitchat"] * chitchat_count
    rng.shuffle(session_types)
    return session_types


def extract_current_state(persona_item):
    return copy.deepcopy(persona_item["Dynamic_Profile"])


def get_social_status(dynamic_state):
    return dynamic_state["Social_Relationships"]["Social_Status"]["Current_State"]


def set_social_status(dynamic_state, new_status):
    dynamic_state["Social_Relationships"]["Social_Status"]["Current_State"] = new_status





def build_driver_profile(Personality, Life_Goal):
    attr_multipliers = {
        "Health_Status": 1.0,
        "Social_Status": 1.0,
        "Work_Status": 1.0,
        "Residence": 1.3,
        "Marital_Status": 1.6,
        "Children_Status": 1.8,
        "Career_Status": 1.3
    }

    mbti = str(Personality.get("MBTI", "")).lower()
    tags = " ".join(Personality.get("MBTI_Tags", [])).lower()
    goal_type = str(Life_Goal.get("Type", "")).lower()
    goal_desc = str(Life_Goal.get("Description", "")).lower()
    goal_text = f"{goal_type} {goal_desc}"

    if "career" in goal_text or "achievement" in goal_text or "success" in goal_text:
        attr_multipliers["Career_Status"] *= 1.6
        attr_multipliers["Work_Status"] *= 1.2

    if "family" in goal_text or "relationship" in goal_text or "marriage" in goal_text:
        attr_multipliers["Marital_Status"] *= 2.0
        attr_multipliers["Children_Status"] *= 1.8

    if "health" in goal_text or "wellness" in goal_text or "fitness" in goal_text:
        attr_multipliers["Health_Status"] *= 1.8

    if "travel" in goal_text or "explore" in goal_text or "move" in goal_text or "settle" in goal_text:
        attr_multipliers["Residence"] *= 1.6

    if "social" in goal_text or "friend" in goal_text or "community" in goal_text:
        attr_multipliers["Social_Status"] *= 1.6

    if "e" in mbti[:1] or "social" in tags or "outgoing" in tags:
        attr_multipliers["Social_Status"] *= 1.2

    if "j" in mbti[-1:] or "ambitious" in tags or "organized" in tags:
        attr_multipliers["Career_Status"] *= 1.1
        attr_multipliers["Work_Status"] *= 1.1

    if "sensitive" in tags or "stress" in tags:
        attr_multipliers["Health_Status"] *= 1.1

    return attr_multipliers



def get_cooldown_months(attr_name, config):
    dynamic_config = config["Dynamic_Information"]

    if attr_name == "Health_Status":
        return dynamic_config["Health_Status"]["Cooldown_Months"]
    elif attr_name == "Social_Status":
        return dynamic_config["Social_Relationships"]["Social_Status"]["Cooldown_Months"]
    elif attr_name == "Work_Status":
        return dynamic_config["Work_Status"]["Cooldown_Months"]
    elif attr_name == "Marital_Status":
        return dynamic_config["Marital_Status"]["Cooldown_Months"]
    elif attr_name == "Children_Status":
        return dynamic_config["Children_Status"]["Cooldown_Months"]
    elif attr_name == "Career_Direction":
        return dynamic_config["Career_Status"]["Cooldown_Months"]
    elif attr_name == "Residence":
        return 3
    else:
        return 1


def is_in_cooldown(attr_name, session_idx, cooldown_records, config):
    if attr_name not in cooldown_records:
        return False
    last_idx = cooldown_records[attr_name]
    cooldown = get_cooldown_months(attr_name, config)
    return (session_idx - last_idx) <= cooldown


def is_attribute_available(attr_name, current_state, session_idx, cooldown_records, config, session_date=None):
    if is_in_cooldown(attr_name, session_idx, cooldown_records, config):
        return False

    if attr_name == "Children_Status":
        marital_status = current_state["Marital_Status"]["Status"]

        # 强约束：必须结婚后才能生孩子
        if marital_status != "Married":
            return False

        # 最多 4 个孩子
        if current_state["Children_Status"]["Status"] == "Yes":
            existing_children = [k for k in current_state["Children_Status"].keys() if k.startswith("Child_")]
            if len(existing_children) >= 4:
                return False

        # age legality check
        if session_date is not None:
            user_birthdate = current_state.get("_User_Birthdate_For_Internal_Use", "1990-01-01")
            user_birth_dt = datetime.strptime(user_birthdate, "%Y-%m-%d")
            session_dt = datetime.strptime(session_date, "%Y-%m-%d")

            min_gap = config["Dynamic_Information"]["Children_Status"]["Parent_Child_Age_Gap_Range"][0]

            user_age_at_session = session_dt.year - user_birth_dt.year
            if (session_dt.month, session_dt.day) < (user_birth_dt.month, user_birth_dt.day):
                user_age_at_session -= 1

            if user_age_at_session < min_gap:
                return False

    if attr_name == "Marital_Status":
        current_status = current_state["Marital_Status"]["Status"]
        allowed = config["Dynamic_Information"]["Marital_Status"]["Allowed_Transitions"].get(current_status, [])
        if len(allowed) == 0:
            return False

    return True



def get_available_attributes(current_state, session_idx, cooldown_records, config, session_date):
    high_freq = ["Health_Status", "Social_Status", "Work_Status"]
    low_freq = ["Residence", "Marital_Status", "Children_Status", "Career_Status"]

    available_high = [
        attr for attr in high_freq
        if is_attribute_available(attr, current_state, session_idx, cooldown_records, config, session_date=session_date)
    ]
    available_low = [
        attr for attr in low_freq
        if is_attribute_available(attr, current_state, session_idx, cooldown_records, config, session_date=session_date)
    ]

    return available_high, available_low



def sample_update_attributes(rng, current_state, session_idx, cooldown_records, config,
                             driver_profile, session_date):
    available_high, available_low = get_available_attributes(
        current_state=current_state,
        session_idx=session_idx,
        cooldown_records=cooldown_records,
        config=config,
        session_date=session_date
    )

    if len(available_high) == 0 and len(available_low) == 0:
        return []

    update_count = 1 if rng.random() < 0.65 else 2
    selected = []

    target_min_update_counts = {
        "Residence": 3,
        "Career_Status": 3,
        "Marital_Status": 3,
        "Children_Status": 2
    }

    for _ in range(update_count):
        remaining_high = [a for a in available_high if a not in selected]
        remaining_low = [a for a in available_low if a not in selected]

        if len(remaining_high) == 0 and len(remaining_low) == 0:
            break

        group_weights = {}
        if len(remaining_high) > 0:
            high_weight = 0.60 * sum(driver_profile[a] for a in remaining_high) / len(remaining_high)
            group_weights["high"] = high_weight

        if len(remaining_low) > 0:
            low_weight = 0.40 * sum(driver_profile[a] for a in remaining_low) / len(remaining_low)
            group_weights["low"] = low_weight

        selected_group = weighted_choice(rng, group_weights)

        if selected_group == "high":
            attr_weights = {a: driver_profile[a] for a in remaining_high}
        else:
            attr_weights = {}
            for a in remaining_low:
                last_idx = cooldown_records.get(a, -3)
                months_since_last_change = session_idx - last_idx

                base_weight = driver_profile[a]
                boost = 1.0 + min(months_since_last_change * 0.08, 0.8)

                if a in target_min_update_counts:
                    required_min = target_min_update_counts[a]
                    soft_bonus = 1.0 + (required_min - 1) * 0.20
                else:
                    soft_bonus = 1.0

                # 对婚姻和孩子再额外加强
                if a == "Marital_Status":
                    soft_bonus *= 1.35
                elif a == "Children_Status":
                    soft_bonus *= 1.80

                    # 如果已经结婚，进一步提高孩子状态被采样概率
                    current_marital_status = current_state["Marital_Status"]["Status"]
                    if current_marital_status == "Married":
                        soft_bonus *= 1.50

                attr_weights[a] = base_weight * boost * soft_bonus

        selected_attr = weighted_choice(rng, attr_weights)
        if selected_attr not in selected:
            selected.append(selected_attr)

    return selected







def update_work_status(current_state, rng, config):
    before = copy.deepcopy(current_state["Work_Status"])
    states = config["Dynamic_Information"]["Work_Status"]["Current_State"]
    current_value = before["Current_State"]

    candidates = [s for s in states if s != current_value]
    new_value = rng.choice(candidates)

    current_state["Work_Status"]["Current_State"] = new_value
    after = copy.deepcopy(current_state["Work_Status"])

    return before, after, {
        "Driver": "routine_workload_change",
        "Trigger_Type": "short_term_state_change"
    }


def update_health_status(current_state, rng, config):
    before = copy.deepcopy(current_state["Health_Status"])
    physical_options = config["Dynamic_Information"]["Health_Status"]["Physical_Health"]
    mental_options = config["Dynamic_Information"]["Health_Status"]["Mental_Health"]

    change_target = rng.choice(["Physical_Health", "Mental_Health"])

    if change_target == "Physical_Health":
        current_value = current_state["Health_Status"]["Physical_Health"]
        candidates = [x for x in physical_options if x != current_value]
        current_state["Health_Status"]["Physical_Health"] = rng.choice(candidates)
    else:
        current_value = current_state["Health_Status"]["Mental_Health"]
        candidates = [x for x in mental_options if x != current_value]
        current_state["Health_Status"]["Mental_Health"] = rng.choice(candidates)

    after = copy.deepcopy(current_state["Health_Status"])

    return before, after, {
        "Driver": "health_fluctuation",
        "Trigger_Type": "short_term_state_change"
    }


def update_social_status(current_state, rng, config):
    before = {"Current_State": get_social_status(current_state)}

    status_options = config["Dynamic_Information"]["Social_Relationships"]["Social_Status"]["Current_State"]
    current_value = before["Current_State"]

    candidates = [x for x in status_options if x != current_value]
    new_value = rng.choice(candidates)

    set_social_status(current_state, new_value)
    after = {"Current_State": get_social_status(current_state)}

    return before, after, {
        "Driver": "social_activity_change",
        "Trigger_Type": "short_term_state_change"
    }



def update_residence(current_state, rng, config):
    before = current_state["Residence"]

    fixed_birthplace = current_state.get("_Fixed_Birthplace_For_Internal_Use", None)
    if fixed_birthplace is not None:
        current_residence = current_state["Residence"]
    else:
        current_residence = current_state["Residence"]

    all_countries = config["Fixed_Information"]["Birthplace"]["Countries"]
    all_cities = config["Fixed_Information"]["Birthplace"]["Cities"]

    current_city = current_residence.split(",")[0].strip()
    current_country = current_residence.split(",")[-1].strip()

    move_type = rng.choices(
        ["same_country", "cross_country"],
        weights=[0.5, 0.5],
        k=1
    )[0]

    if move_type == "same_country":
        candidate_cities = [c for c in all_cities.get(current_country, []) if c != current_city]
        if len(candidate_cities) == 0:
            other_countries = [c for c in all_countries if c != current_country]
            if len(other_countries) > 0:
                new_country = rng.choice(other_countries)
                new_city = rng.choice(all_cities.get(new_country, [current_city]))
            else:
                new_country = current_country
                new_city = current_city
        else:
            new_country = current_country
            new_city = rng.choice(candidate_cities)
    else:
        other_countries = [c for c in all_countries if c != current_country]
        if len(other_countries) > 0:
            new_country = rng.choice(other_countries)
            new_city = rng.choice(all_cities.get(new_country, [current_city]))
        else:
            new_country = current_country
            candidate_cities = [c for c in all_cities.get(current_country, []) if c != current_city]
            new_city = rng.choice(candidate_cities) if len(candidate_cities) > 0 else current_city

    current_state["Residence"] = f"{new_city}, {new_country}"
    after = current_state["Residence"]

    return before, after, {
        "Driver": "life_relocation",
        "Trigger_Type": "structural_change"
    }


def update_marital_status(current_state, rng, config, fixed_profile):
    before = copy.deepcopy(current_state["Marital_Status"])

    marital_config = config["Dynamic_Information"]["Marital_Status"]
    current_status = before["Status"]
    allowed_next = marital_config["Allowed_Transitions"].get(current_status, [])

    if len(allowed_next) == 0:
        return before, before, {
            "Driver": "no_valid_transition",
            "Trigger_Type": "blocked"
        }

    next_status = rng.choice(allowed_next)

    def generate_new_partner_birthdate():
        base_birthdate = current_state.get("_User_Birthdate_For_Internal_Use", "1990-01-01")
        birth_dt = datetime.strptime(base_birthdate, "%Y-%m-%d")
        age_gap = rng.randint(
            marital_config["Marital_Age_Diff_Range"][0],
            marital_config["Marital_Age_Diff_Range"][1]
        )
        direction = rng.choice([-1, 1])
        partner_year = birth_dt.year + direction * age_gap
        partner_month = rng.randint(1, 12)
        partner_day = rng.randint(1, 28)
        return f"{partner_year:04d}-{partner_month:02d}-{partner_day:02d}"

    def generate_new_partner_name():
        excluded_names = collect_excluded_names(current_state, fixed_profile)

        user_gender = fixed_profile.get("Gender", "")
        if user_gender == "Male":
            partner_gender = "Female"
        elif user_gender == "Female":
            partner_gender = "Male"
        else:
            partner_gender = rng.choice(["Male", "Female"])

        return sample_full_name(
            rng=rng,
            config=config,
            gender=partner_gender,
            excluded_names=excluded_names
        )


    if current_status == "Single" and next_status == "Dating":
        current_state["Marital_Status"] = {
            "Status": "Dating",
            "Name": generate_new_partner_name(),
            "Birthdate": generate_new_partner_birthdate()
        }

    elif current_status == "Dating" and next_status == "Married":
        # 必须与正在约会的是同一个人
        current_state["Marital_Status"] = {
            "Status": "Married",
            "Name": before.get("Name", generate_new_partner_name()),
            "Birthdate": before.get("Birthdate", generate_new_partner_birthdate())
        }

    elif current_status == "Dating" and next_status == "Single":
        current_state["Marital_Status"] = {
            "Status": "Single"
        }

    elif current_status == "Married" and next_status in ["Divorced", "Single"]:
        current_state["Marital_Status"] = {
            "Status": next_status
        }

    elif current_status == "Divorced" and next_status == "Dating":
        current_state["Marital_Status"] = {
            "Status": "Dating",
            "Name": generate_new_partner_name(),
            "Birthdate": generate_new_partner_birthdate()
        }

    elif next_status in ["Dating", "Married"]:
        if current_status == "Dating" and next_status == "Married":
            current_state["Marital_Status"] = {
                "Status": "Married",
                "Name": before.get("Name", generate_new_partner_name()),
                "Birthdate": before.get("Birthdate", generate_new_partner_birthdate())
            }
        else:
            current_state["Marital_Status"] = {
                "Status": next_status,
                "Name": generate_new_partner_name(),
                "Birthdate": generate_new_partner_birthdate()
            }
    else:
        current_state["Marital_Status"] = {
            "Status": next_status
        }

    after = copy.deepcopy(current_state["Marital_Status"])

    return before, after, {
        "Driver": "relationship_progression",
        "Trigger_Type": "structural_change"
    }




def update_children_status(current_state, rng, config, session_date, fixed_profile):
    before = copy.deepcopy(current_state["Children_Status"])

    children_config = config["Dynamic_Information"]["Children_Status"]
    user_birthdate = current_state.get("_User_Birthdate_For_Internal_Use", "1990-01-01")
    user_birth_dt = datetime.strptime(user_birthdate, "%Y-%m-%d")
    session_dt = datetime.strptime(session_date, "%Y-%m-%d")

    min_gap = children_config["Parent_Child_Age_Gap_Range"][0]

    # child birthdate must be within current session month and not later than session date
    child_year = session_dt.year
    child_month = session_dt.month
    child_day = rng.randint(1, session_dt.day)
    child_birth_dt = datetime(child_year, child_month, child_day)
    child_birthdate = child_birth_dt.strftime("%Y-%m-%d")

    # age legality check
    parent_age_at_child_birth = child_birth_dt.year - user_birth_dt.year
    if (child_birth_dt.month, child_birth_dt.day) < (user_birth_dt.month, user_birth_dt.day):
        parent_age_at_child_birth -= 1

    if parent_age_at_child_birth < min_gap:
        after = copy.deepcopy(current_state["Children_Status"])
        return before, after, {
            "Driver": "family_expansion_blocked",
            "Trigger_Type": "age_constraint_failed"
        }

    def generate_child_name():
        excluded_names = collect_excluded_names(current_state, fixed_profile)
        child_gender = rng.choice(["Male", "Female"])

        return sample_full_name(
            rng=rng,
            config=config,
            gender=child_gender,
            excluded_names=excluded_names
        )


    if current_state["Children_Status"]["Status"] == "No":
        current_state["Children_Status"] = {
            "Status": "Yes",
            "Child_1": {
                "Name": generate_child_name(),
                "Birthdate": child_birthdate
            }
        }
    else:
        existing_children = [k for k in current_state["Children_Status"].keys() if k.startswith("Child_")]
        current_count = len(existing_children)

        if current_count < 4:
            next_child_idx = current_count + 1
            current_state["Children_Status"][f"Child_{next_child_idx}"] = {
                "Name": generate_child_name(),
                "Birthdate": child_birthdate
            }
        else:
            after = copy.deepcopy(current_state["Children_Status"])
            return before, after, {
                "Driver": "family_expansion_blocked",
                "Trigger_Type": "max_children_reached"
            }

    after = copy.deepcopy(current_state["Children_Status"])

    return before, after, {
        "Driver": "family_expansion",
        "Trigger_Type": "structural_change"
    }





def update_career_status(current_state, rng, config):
    before = copy.deepcopy(current_state["Career_Status"])

    career_config = config["Dynamic_Information"]["Career_Status"]

    employment_weights = copy.deepcopy(career_config["Employment_Status"])
    company_type_weights = copy.deepcopy(career_config["Company_Types"])
    job_title_weights = copy.deepcopy(career_config["Job_Titles"])
    industry_weights = copy.deepcopy(career_config["Industries"])
    income_weights = copy.deepcopy(career_config["Monthly_Income"])
    savings_weights = copy.deepcopy(career_config["Savings_Amount"])
    company_name_pool = career_config["Company_Names"]

    old_status = current_state["Career_Status"]

    for _ in range(10):
        new_employment = weighted_choice(rng, employment_weights)
        new_company_type = weighted_choice(rng, company_type_weights)
        new_job_title = weighted_choice(rng, job_title_weights)
        new_industry = weighted_choice(rng, industry_weights)
        new_income = weighted_choice(rng, income_weights)
        new_savings = weighted_choice(rng, savings_weights)

        if new_employment == "Unemployed":
            new_company_type = ""
            new_company_name = ""
            new_job_title = ""
        elif new_employment == "Entrepreneur":
            new_company_type = "Startup"
            new_company_name = rng.choice(company_name_pool["Startup"])
        else:
            name_candidates = company_name_pool.get(new_company_type, ["Generic Company"])
            new_company_name = rng.choice(name_candidates)

        new_career_status = {
            "Employment_Status": new_employment,
            "Company_Type": new_company_type,
            "Company_Name": new_company_name,
            "Job_Title": new_job_title,
            "Industry": new_industry,
            "Monthly_Income": new_income,
            "Savings_Amount": new_savings
        }

        if new_career_status != old_status:
            current_state["Career_Status"] = new_career_status
            break

    after = copy.deepcopy(current_state["Career_Status"])

    return before, after, {
        "Driver": "career_status_change",
        "Trigger_Type": "structural_change"
    }



def apply_single_update(attr_name, current_state, rng, config, fixed_profile, session_date=None):
    if attr_name == "Work_Status":
        return update_work_status(current_state, rng, config)
    elif attr_name == "Health_Status":
        return update_health_status(current_state, rng, config)
    elif attr_name == "Social_Status":
        return update_social_status(current_state, rng, config)
    elif attr_name == "Residence":
        return update_residence(current_state, rng, config)
    elif attr_name == "Marital_Status":
        return update_marital_status(current_state, rng, config, fixed_profile)
    elif attr_name == "Children_Status":
        return update_children_status(current_state, rng, config, session_date, fixed_profile)
    elif attr_name == "Career_Status":
        return update_career_status(current_state, rng, config)
    else:
        return None, None, {"Driver": "unknown", "Trigger_Type": "unknown"}





def sanitize_internal_fields(state_dict):
    cleaned = copy.deepcopy(state_dict)
    cleaned.pop("_User_Birthdate_For_Internal_Use", None)
    cleaned.pop("_Fixed_Birthplace_For_Internal_Use", None)
    return cleaned


def Generate_Single_Timeline_Sessions(persona_item, config):
    metadata = persona_item.get("metadata", {})
    persona_seed = metadata["persona_seed"]
    rng = build_deterministic_rng(persona_seed)

    Personality = persona_item["Personality"]
    Life_Goal = persona_item["Life_Goal"]
    Fixed_Profile = persona_item["Fixed_Profile"]

    current_state = extract_current_state(persona_item)
    current_state["_User_Birthdate_For_Internal_Use"] = Fixed_Profile.get("Birthdate", "1990-01-01")
    current_state["_Fixed_Birthplace_For_Internal_Use"] = Fixed_Profile.get("Birthplace", "")

    driver_profile = build_driver_profile(Personality, Life_Goal)

    monthly_dates = build_monthly_session_dates(rng)
    session_types = build_session_type_sequence(rng, len(monthly_dates))

    cooldown_records = {}
    timeline_sessions = []

    actual_update_counts = {
        "Residence": 0,
        "Career_Status": 0,
        "Marital_Status": 0,
        "Children_Status": 0
    }

    for session_idx, (session_date, session_type) in enumerate(zip(monthly_dates, session_types), start=1):
        state_before = sanitize_internal_fields(current_state)

        if session_type == "chitchat":
            session_item = {
                "Session_ID": session_idx,
                "Date": session_date,
                "Session_Type": "chitchat",
                "State_Before": state_before,
                "State_After": state_before,
                "Updated_Attributes": [],
                "Update_Reason": None
            }
            timeline_sessions.append(session_item)
            continue

        selected_attributes = sample_update_attributes(
            rng=rng,
            current_state=current_state,
            session_idx=session_idx,
            cooldown_records=cooldown_records,
            config=config,
            driver_profile=driver_profile,
            session_date=session_date
        )

        if len(selected_attributes) == 0:
            session_item = {
                "Session_ID": session_idx,
                "Date": session_date,
                "Session_Type": "chitchat",
                "State_Before": state_before,
                "State_After": state_before,
                "Updated_Attributes": [],
                "Update_Reason": None
            }
            timeline_sessions.append(session_item)
            continue

        updated_attributes = []
        reason_list = []

        for attr_name in selected_attributes:
            before_value, after_value, reason_info = apply_single_update(
                attr_name=attr_name,
                current_state=current_state,
                rng=rng,
                config=config,
                fixed_profile=Fixed_Profile,
                session_date=session_date
            )

            if before_value != after_value:
                updated_attributes.append({
                    "Attribute": attr_name,
                    "Before": before_value,
                    "After": after_value
                })
                reason_list.append({
                    "Attribute": attr_name,
                    "Reason": reason_info
                })
                cooldown_records[attr_name] = session_idx

                if attr_name in actual_update_counts:
                    actual_update_counts[attr_name] += 1

        state_after = sanitize_internal_fields(current_state)

        if len(updated_attributes) == 0:
            session_item = {
                "Session_ID": session_idx,
                "Date": session_date,
                "Session_Type": "chitchat",
                "State_Before": state_before,
                "State_After": state_before,
                "Updated_Attributes": [],
                "Update_Reason": None
            }
        else:
            session_item = {
                "Session_ID": session_idx,
                "Date": session_date,
                "Session_Type": "update",
                "State_Before": state_before,
                "State_After": state_after,
                "Updated_Attributes": updated_attributes,
                "Update_Reason": reason_list
            }

        timeline_sessions.append(session_item)

    future_plan_session = {
        "Session_ID": len(timeline_sessions) + 1,
        "Date": "2026-01-15",
        "Session_Type": "future_plan",
        "State_Before": sanitize_internal_fields(current_state),
        "State_After": sanitize_internal_fields(current_state),
        "Updated_Attributes": [],
        "Update_Reason": {
            "Driver": "future_planning",
            "Trigger_Type": "planning_only"
        }
    }
    timeline_sessions.append(future_plan_session)

    return timeline_sessions




def Generate_User_Timeline_Sessions(args):
    print(f"Processing file: {args.input_file}")
    print(f"Output file: {args.output_file}")

    try:
        with open(args.config_file, 'r', encoding="utf-8") as f:
            config = json.load(f)

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
            Timeline_Initial_State = persona_item["Timeline_Initial_State"]
            metadata = persona_item["metadata"]
            previous_cost = persona_item.get("token_cost", None)

            Timeline_Sessions = Generate_Single_Timeline_Sessions(
                persona_item=persona_item,
                config=config
            )

            result_item = {
                "ID": ID,
                "Fixed_Profile": Fixed_Profile,
                "Dynamic_Profile": Dynamic_Profile,
                "Preference_Profile": Preference_Profile,
                "Personality": Personality,
                "Life_Goal": Life_Goal,
                "Others_Profile": Others_Profile,
                "Timeline_Initial_State": Timeline_Initial_State,
                "Timeline_Sessions": Timeline_Sessions,
                "metadata": metadata,
                "token_cost": previous_cost
            }

            with jsonlines.open(args.output_file, 'a') as writer:
                writer.write(result_item)

            with open(args.output_perfect_file, "a", encoding="utf-8") as f:
                json.dump(result_item, f, ensure_ascii=False, indent=4)

        print("[DEBUG] Successfully processed Step 2.2 timeline sessions")
        return True

    except Exception as e:
        print(f"Error processing persona: {e}:{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step 2.2 ---- Monthly Timeline Session Construction.')
    parser.add_argument("--config_file", type=str,
                        default="MemConflict/Data/Config.json",
                        help="Configuration file for persona processing")
    parser.add_argument("--input_file", type=str,
                        default="MemConflict/Data/Step2_1.jsonl",
                        help="Last Step output file for persona processing")
    parser.add_argument("--output_file", type=str,
                        default="MemConflict/Data/Step2_2.jsonl",
                        help="Output JSONL file for timeline sessions")
    parser.add_argument("--output_perfect_file", type=str,
                        default="MemConflict/Data_perfect/Step2_2.json",
                        help="Output JSON file for timeline sessions")
    args = parser.parse_args()

    Generate_User_Timeline_Sessions(args)
