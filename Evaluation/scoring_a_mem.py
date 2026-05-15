import argparse
import os

from eval_scoring import Generate_User_Evaluation


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score A-MEM evaluation results.")
    parser.add_argument(
        "--input_file",
        type=str,
        default=os.path.join(CURRENT_DIR, "Results", "a_mem_results.jsonl"),
        help="Input JSONL file containing A-MEM results.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(CURRENT_DIR, "Scores", "a_mem_eval_scores.jsonl"),
        help="Output JSONL file for A-MEM evaluation scores.",
    )
    parser.add_argument(
        "--output_perfect_file",
        type=str,
        default=os.path.join(CURRENT_DIR, "Scores", "a_mem_eval_scores.json"),
        help="Output JSON file for A-MEM evaluation scores.",
    )
    parser.add_argument(
        "--prediction_fields",
        type=str,
        default="Model_Answer,Predicted_Answer,Generated_Answer,memory_answer,model_answer,predicted_answer",
        help="Comma-separated candidate fields that may store the model answer.",
    )
    parser.add_argument(
        "--disable_llm_judge",
        action="store_true",
        help="Disable LLM judge and use rule-based scoring only.",
    )
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=int(os.getenv("EVAL_SCORING_PARALLEL_WORKERS", "1")),
        help="Number of persona-level worker threads for LLM-judge scoring. Default: 1.",
    )
    args = parser.parse_args()
    args.enable_llm_judge = not args.disable_llm_judge
    return args


if __name__ == "__main__":
    Generate_User_Evaluation(build_args())
