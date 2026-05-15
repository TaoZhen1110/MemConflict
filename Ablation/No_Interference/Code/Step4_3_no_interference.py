import argparse
import importlib.util
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parents[2]
CODE_DIR = PROJECT_DIR / "Code"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def load_module(module_filename: str, module_name: str):
    module_path = CODE_DIR / module_filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


module = load_module("Step4_3.py", "step4_3_main_module")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4.3 for no-interference ablation branch.")
    parser.add_argument("--input_file", type=str, default=r"/home/taoz/Mem_Conflict/MemConflict/Ablation/No_Interference/Data/Step4_2_no_interference.jsonl", help="Input JSONL file")
    parser.add_argument("--output_file", type=str, default=r"/home/taoz/Mem_Conflict/MemConflict/Ablation/No_Interference/Data/Step4_3_no_interference.jsonl", help="Output JSONL file")
    parser.add_argument("--output_perfect_file", type=str, default=r"/home/taoz/Mem_Conflict/MemConflict/Ablation/No_Interference/Data/Step4_3_no_interference.json", help="Output JSON file")
    parser.add_argument("--tokenizer_name", type=str, default="o200k_base", help="tiktoken tokenizer name")
    args = parser.parse_args()

    module.Generate_User_Dialogue_Token_Calculation(args)
