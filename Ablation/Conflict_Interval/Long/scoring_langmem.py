import argparse
import os

from eval_scoring import Generate_User_Evaluation


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score LangMem evaluation results.")
    parser.add_argument(
        "--input_file",
        type=str,
        default=os.path.join(CURRENT_DIR, "Results", "langmem_results.jsonl"),
        help="Input JSONL file containing LangMem results.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=os.path.join(CURRENT_DIR, "Scores", "langmem_eval_scores.jsonl"),
        help="Output JSONL file for LangMem evaluation scores.",
    )
    parser.add_argument(
        "--output_perfect_file",
        type=str,
        default=os.path.join(CURRENT_DIR, "Scores", "langmem_eval_scores.json"),
        help="Output JSON file for LangMem evaluation scores.",
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
    args = parser.parse_args()
    args.enable_llm_judge = not args.disable_llm_judge
    return args


if __name__ == "__main__":
    Generate_User_Evaluation(build_args())
