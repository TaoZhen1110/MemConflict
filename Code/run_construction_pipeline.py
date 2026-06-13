import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_DIR / "Code"
DATA_DIR = PROJECT_DIR / "Data"
PERFECT_DIR = PROJECT_DIR / "Data_perfect"

STEP_ORDER = [
    "Step1_1",
    "Step1_2",
    "Step1_3",
    "Step1_4",
    "Step1_5",
    "Step2_1",
    "Step2_2",
    "Step3_1",
    "Step3_2",
    "Step3_3",
    "Step4_1",
    "Step4_2",
    "Step4_3",
    "Step4_4",
]


def build_step_commands(args):
    output_dir = args.output_dir.resolve()
    perfect_dir = args.perfect_dir.resolve()

    def data_file(step_name):
        return output_dir / f"{step_name}.jsonl"

    def perfect_file(step_name):
        return perfect_dir / f"{step_name}.json"

    common_config = ["--config_file", str(args.config_file.resolve())]
    commands = [
        [
            "Step1_1",
            [sys.executable, str(CODE_DIR / "Step1_1.py"), "--count", str(args.count)]
            + common_config
            + [
                "--input_file",
                str(args.seed_file.resolve()),
                "--output_file",
                str(data_file("Step1_1")),
                "--output_perfect_file",
                str(perfect_file("Step1_1")),
            ],
        ],
        [
            "Step1_2",
            [sys.executable, str(CODE_DIR / "Step1_2.py")]
            + common_config
            + [
                "--input_file",
                str(data_file("Step1_1")),
                "--output_file",
                str(data_file("Step1_2")),
                "--output_perfect_file",
                str(perfect_file("Step1_2")),
            ],
        ],
        [
            "Step1_3",
            [sys.executable, str(CODE_DIR / "Step1_3.py")]
            + common_config
            + [
                "--input_file",
                str(data_file("Step1_2")),
                "--output_file",
                str(data_file("Step1_3")),
                "--output_perfect_file",
                str(perfect_file("Step1_3")),
            ],
        ],
        [
            "Step1_4",
            [sys.executable, str(CODE_DIR / "Step1_4.py")]
            + common_config
            + [
                "--input_file",
                str(data_file("Step1_3")),
                "--output_file",
                str(data_file("Step1_4")),
                "--output_perfect_file",
                str(perfect_file("Step1_4")),
            ],
        ],
        [
            "Step1_5",
            [sys.executable, str(CODE_DIR / "Step1_5.py")]
            + common_config
            + [
                "--input_file",
                str(data_file("Step1_4")),
                "--output_file",
                str(data_file("Step1_5")),
                "--output_perfect_file",
                str(perfect_file("Step1_5")),
            ],
        ],
        [
            "Step2_1",
            [sys.executable, str(CODE_DIR / "Step2_1.py")]
            + common_config
            + [
                "--input_file",
                str(data_file("Step1_5")),
                "--output_file",
                str(data_file("Step2_1")),
                "--output_perfect_file",
                str(perfect_file("Step2_1")),
            ],
        ],
        [
            "Step2_2",
            [sys.executable, str(CODE_DIR / "Step2_2.py")]
            + common_config
            + [
                "--input_file",
                str(data_file("Step2_1")),
                "--output_file",
                str(data_file("Step2_2")),
                "--output_perfect_file",
                str(perfect_file("Step2_2")),
            ],
        ],
        [
            "Step3_1",
            [sys.executable, str(CODE_DIR / "Step3_1.py")]
            + common_config
            + [
                "--input_file",
                str(data_file("Step2_2")),
                "--output_file",
                str(data_file("Step3_1")),
                "--output_perfect_file",
                str(perfect_file("Step3_1")),
            ],
        ],
        [
            "Step3_2",
            [sys.executable, str(CODE_DIR / "Step3_2.py")]
            + common_config
            + [
                "--input_file",
                str(data_file("Step3_1")),
                "--output_file",
                str(data_file("Step3_2")),
                "--output_perfect_file",
                str(perfect_file("Step3_2")),
            ],
        ],
        [
            "Step3_3",
            [
                sys.executable,
                str(CODE_DIR / "Step3_3.py"),
                "--input_file",
                str(data_file("Step3_2")),
                "--output_file",
                str(data_file("Step3_3")),
                "--output_perfect_file",
                str(perfect_file("Step3_3")),
            ],
        ],
        [
            "Step4_1",
            [
                sys.executable,
                str(CODE_DIR / "Step4_1.py"),
                "--input_file",
                str(data_file("Step3_3")),
                "--output_file",
                str(data_file("Step4_1")),
                "--output_perfect_file",
                str(perfect_file("Step4_1")),
            ],
        ],
        [
            "Step4_2",
            [
                sys.executable,
                str(CODE_DIR / "Step4_2.py"),
                "--input_file",
                str(data_file("Step4_1")),
                "--output_file",
                str(data_file("Step4_2")),
                "--output_perfect_file",
                str(perfect_file("Step4_2")),
            ],
        ],
        [
            "Step4_3",
            [
                sys.executable,
                str(CODE_DIR / "Step4_3.py"),
                "--input_file",
                str(data_file("Step4_2")),
                "--output_file",
                str(data_file("Step4_3")),
                "--output_perfect_file",
                str(perfect_file("Step4_3")),
                "--tokenizer_name",
                args.tokenizer_name,
            ],
        ],
        [
            "Step4_4",
            [
                sys.executable,
                str(CODE_DIR / "Step4_4.py"),
                "--input_file",
                str(data_file("Step4_3")),
                "--output_file",
                str(data_file("Step4_4")),
                "--output_perfect_file",
                str(perfect_file("Step4_4")),
                "--prompt_file",
                str((PROJECT_DIR / "Prompt" / "Prompt4_4.txt").resolve()),
            ],
        ],
    ]

    if args.disable_llm_rewrite:
        commands[-1][1].append("--disable_llm_rewrite")

    return commands


def validate_args(args):
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if not args.seed_file.exists():
        raise SystemExit(f"Seed file not found: {args.seed_file}")
    if not args.config_file.exists():
        raise SystemExit(f"Config file not found: {args.config_file}")
    if args.output_dir.resolve() == DATA_DIR.resolve() and args.overwrite:
        raise SystemExit("Refusing to overwrite the released Data/ directory. Use a separate --output_dir.")


def prepare_outputs(args):
    output_dir = args.output_dir.resolve()
    perfect_dir = args.perfect_dir.resolve()
    existing = [
        output_dir / f"{step_name}.jsonl"
        for step_name in STEP_ORDER
        if (output_dir / f"{step_name}.jsonl").exists()
    ]
    existing += [
        perfect_dir / f"{step_name}.json"
        for step_name in STEP_ORDER
        if (perfect_dir / f"{step_name}.json").exists()
    ]

    if existing and not args.overwrite:
        paths = "\n".join(str(path) for path in existing[:5])
        more = "" if len(existing) <= 5 else f"\n... and {len(existing) - 5} more"
        raise SystemExit(
            "Output files already exist. Use --overwrite or choose a new --output_dir.\n"
            f"{paths}{more}"
        )

    if args.overwrite:
        for path in existing:
            path.unlink()
        if perfect_dir.exists() and not any(perfect_dir.iterdir()):
            shutil.rmtree(perfect_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    perfect_dir.mkdir(parents=True, exist_ok=True)


def run_pipeline(args):
    validate_args(args)
    prepare_outputs(args)

    commands = build_step_commands(args)
    for step_name, command in commands:
        print(f"\n[PIPELINE] Running {step_name}")
        subprocess.run(command, cwd=PROJECT_DIR, check=True)

    final_file = args.output_dir.resolve() / "Step4_4.jsonl"
    print(f"\n[PIPELINE] Done. Final benchmark file: {final_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the MemConflict construction pipeline from Step1_1 to Step4_4."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of persona seeds to sample in Step1_1.",
    )
    parser.add_argument(
        "--seed_file",
        type=Path,
        default=DATA_DIR / "Step0.jsonl",
        help="Persona seed JSONL file.",
    )
    parser.add_argument(
        "--config_file",
        type=Path,
        default=DATA_DIR / "Config.json",
        help="Construction configuration JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DATA_DIR / "Generated",
        help="Directory for generated Step*.jsonl files.",
    )
    parser.add_argument(
        "--perfect_dir",
        type=Path,
        default=PERFECT_DIR / "Generated",
        help="Directory for pretty-printed JSON mirrors.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="o200k_base",
        help="Tokenizer name used by Step4_3.",
    )
    parser.add_argument(
        "--disable_llm_rewrite",
        action="store_true",
        help="Disable the optional LLM rewrite stage in Step4_4.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing generated files in output_dir/perfect_dir before running.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
