import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MEMORY_SYSTEMS = [
    ("A-Mem", "a_mem"),
    ("LangMem", "langmem"),
    ("Letta", "letta"),
    ("MemOS", "memos"),
    ("Mem0", "memzero"),
    ("MemoBase", "memobase"),
]

CONFLICT_CONFIG = {
    "dynamic_conflict": {
        "summary_key": "Dynamic_Conflict",
        "aa_key": "dynamic_answer_accuracy",
        "hit_key": "updated_evidence_hit_at_3",
        "eug_label": "Dynamic_EUG",
        "rf_label": "Dynamic_RF",
        "uf_label": "Dynamic_UF",
    },
    "static_conflict": {
        "summary_key": "Static_Conflict",
        "aa_key": "static_answer_accuracy",
        "hit_key": "truth_evidence_hit_at_3",
        "eug_label": "Static_EUG",
        "rf_label": "Static_RF",
        "uf_label": "Static_UF",
    },
    "conditional_conflict": {
        "summary_key": "Conditional_Conflict",
        "aa_key": "conditional_answer_accuracy",
        "hit_key": "correct_condition_evidence_hit_at_3",
        "eug_label": "Conditional_EUG",
        "rf_label": "Conditional_RF",
        "uf_label": "Conditional_UF",
    },
}


def Load_Jsonl_Items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def Weighted_Summary_Metric(
    items: List[Dict[str, Any]],
    summary_key: str,
    metric_group: str,
    metric_key: str,
) -> Tuple[float, int]:
    numerator = 0.0
    denominator = 0
    for item in items:
        conflict_summary = item.get("Evaluation_Summary", {}).get(summary_key, {})
        question_count = int(conflict_summary.get("Question_Count", 0) or 0)
        if question_count <= 0:
            continue
        metric_value = float(conflict_summary.get(metric_group, {}).get(metric_key, 0.0) or 0.0)
        numerator += metric_value * question_count
        denominator += question_count
    if denominator == 0:
        return 0.0, 0
    return numerator / denominator, denominator


def Compute_Evidence_Utilization_Gap(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    gap_values = []
    for config in CONFLICT_CONFIG.values():
        aa, question_count = Weighted_Summary_Metric(
            items,
            config["summary_key"],
            "Black_Box_Metrics",
            config["aa_key"],
        )
        hit, _ = Weighted_Summary_Metric(
            items,
            config["summary_key"],
            "White_Box_Metrics",
            config["hit_key"],
        )
        gap = hit - aa
        result[config["eug_label"]] = gap
        result[config["eug_label"].replace("_EUG", "_AA")] = aa
        result[config["eug_label"].replace("_EUG", "_Hit@3")] = hit
        result[config["eug_label"].replace("_EUG", "_Question_Count")] = question_count
        gap_values.append(gap)
    result["Average_EUG"] = sum(gap_values) / len(gap_values) if gap_values else 0.0
    return result


def Iterate_Questions(items: List[Dict[str, Any]]):
    for item in items:
        for session in item.get("Full_Session_Chain", []):
            for question in session.get("Session_Questions", []):
                yield question


def Compute_Failure_Decomposition(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Dict[str, int]] = {
        conflict_type: {
            "Total": 0,
            "Answered_Correct": 0,
            "Failed": 0,
            "Retrieval_Failure": 0,
            "Utilization_Failure": 0,
            "Other_Failure": 0,
            "Missing_Metrics": 0,
        }
        for conflict_type in CONFLICT_CONFIG
    }

    for question in Iterate_Questions(items):
        conflict_type = question.get("conflict_type")
        if conflict_type not in CONFLICT_CONFIG:
            continue
        config = CONFLICT_CONFIG[conflict_type]
        stats[conflict_type]["Total"] += 1

        metrics = question.get("Evaluation_Result", {}).get("Metrics", {})
        if config["aa_key"] not in metrics or config["hit_key"] not in metrics:
            stats[conflict_type]["Missing_Metrics"] += 1
            continue

        aa = float(metrics.get(config["aa_key"], 0.0) or 0.0)
        hit = float(metrics.get(config["hit_key"], 0.0) or 0.0)

        # The diagnosis focuses on complete failures. Partial-credit answers are treated as usable answers.
        if aa > 0.0:
            stats[conflict_type]["Answered_Correct"] += 1
            continue

        stats[conflict_type]["Failed"] += 1
        if hit <= 0.0:
            stats[conflict_type]["Retrieval_Failure"] += 1
        elif hit > 0.0:
            stats[conflict_type]["Utilization_Failure"] += 1
        else:
            stats[conflict_type]["Other_Failure"] += 1

    result: Dict[str, Any] = {}
    for conflict_type, config in CONFLICT_CONFIG.items():
        conflict_stats = stats[conflict_type]
        failed = conflict_stats["Failed"]
        total = conflict_stats["Total"]
        rf = conflict_stats["Retrieval_Failure"]
        uf = conflict_stats["Utilization_Failure"]

        rf_rate_over_failures = rf / failed if failed else 0.0
        uf_rate_over_failures = uf / failed if failed else 0.0
        rf_rate_over_all = rf / total if total else 0.0
        uf_rate_over_all = uf / total if total else 0.0

        result[f"{config['rf_label']}_Count"] = rf
        result[f"{config['uf_label']}_Count"] = uf
        result[f"{config['rf_label']}_Over_Failures"] = rf_rate_over_failures
        result[f"{config['uf_label']}_Over_Failures"] = uf_rate_over_failures
        result[f"{config['rf_label']}_Over_All"] = rf_rate_over_all
        result[f"{config['uf_label']}_Over_All"] = uf_rate_over_all
        result[f"{config['summary_key']}_Total"] = total
        result[f"{config['summary_key']}_Failed"] = failed
        result[f"{config['summary_key']}_Missing_Metrics"] = conflict_stats["Missing_Metrics"]

    return result


def Build_System_Diagnosis(display_name: str, system_key: str, score_path: str) -> Dict[str, Any]:
    items = Load_Jsonl_Items(score_path)
    eug = Compute_Evidence_Utilization_Gap(items)
    failures = Compute_Failure_Decomposition(items)
    return {
        "Memory_System": display_name,
        "Score_File": score_path,
        "Persona_Count": len(items),
        "Answered_Session_Count": sum(int(item.get("Answered_Session_Count", 0) or 0) for item in items),
        "Answered_Question_Count": sum(int(item.get("Answered_Question_Count", 0) or 0) for item in items),
        **eug,
        **failures,
    }


def Write_Json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


def Write_Csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as outfile:
            outfile.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def Parse_Memory_Systems(raw_value: Optional[str]) -> List[Tuple[str, str]]:
    if not raw_value:
        return DEFAULT_MEMORY_SYSTEMS
    selected_keys = [part.strip() for part in raw_value.split(",") if part.strip()]
    lookup = {key: (display, key) for display, key in DEFAULT_MEMORY_SYSTEMS}
    result = []
    for key in selected_keys:
        if key not in lookup:
            raise ValueError(f"Unknown memory system key: {key}")
        result.append(lookup[key])
    return result


def Generate_Diagnosis(args: argparse.Namespace) -> List[Dict[str, Any]]:
    systems = Parse_Memory_Systems(args.memory_systems)
    rows = []
    for display_name, system_key in systems:
        score_path = os.path.join(args.scores_dir, f"{system_key}_eval_scores.jsonl")
        if not os.path.exists(score_path):
            raise FileNotFoundError(f"Score file not found: {score_path}")
        rows.append(Build_System_Diagnosis(display_name, system_key, score_path))

    Write_Json(args.output_json, rows)
    Write_Csv(args.output_csv, rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute diagnostic failure analysis from scored MemConflict outputs.")
    parser.add_argument(
        "--scores_dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "Scores"),
        help="Directory containing *_eval_scores.jsonl files.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.path.join(CURRENT_DIR, "Diagnostics", "failure_diagnosis_summary.json"),
        help="Output JSON path for diagnostic summary.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=os.path.join(CURRENT_DIR, "Diagnostics", "failure_diagnosis_summary.csv"),
        help="Output CSV path for diagnostic summary.",
    )
    parser.add_argument(
        "--memory_systems",
        type=str,
        default=None,
        help="Comma-separated memory system keys. Default: all systems.",
    )
    parsed_args = parser.parse_args()
    Generate_Diagnosis(parsed_args)
