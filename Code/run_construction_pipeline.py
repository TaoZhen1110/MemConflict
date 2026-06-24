import argparse
import json
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
    if not args.seed_file.exists():
        raise SystemExit(f"Seed file not found: {args.seed_file}")
    if not args.config_file.exists():
        raise SystemExit(f"Config file not found: {args.config_file}")
    if args.output_dir.resolve() == DATA_DIR.resolve() and args.overwrite:
        raise SystemExit("Refusing to overwrite the released Data/ directory. Use a separate --output_dir.")
    if args.append_final_to and args.output_dir.resolve() == args.append_final_to.resolve().parent:
        raise SystemExit("--output_dir must be separate from the directory containing --append_final_to.")
    if args.target_total is not None:
        if args.target_total < 1:
            raise SystemExit("--target_total must be at least 1")
        existing_count = count_jsonl_lines(args.append_final_to) if args.append_final_to else 0
        args.count = args.target_total - existing_count
        if args.count <= 0:
            raise SystemExit(
                f"Target already has {existing_count} rows, which is >= --target_total {args.target_total}."
            )
        print(
            f"[PIPELINE] Target total: {args.target_total}; existing rows: {existing_count}; "
            f"generating {args.count} new rows."
        )
    if args.count < 1:
        raise SystemExit("--count must be at least 1")


def count_jsonl_lines(path):
    if not path or not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


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
    prepare_append_seed_file(args)

    commands = build_step_commands(args)
    for step_name, command in commands:
        print(f"\n[PIPELINE] Running {step_name}")
        subprocess.run(command, cwd=PROJECT_DIR, check=True)

    final_file = args.output_dir.resolve() / "Step4_4.jsonl"
    print(f"\n[PIPELINE] Done. Final benchmark file: {final_file}")
    if args.append_final_to:
        append_jsonl(final_file, args.append_final_to.resolve())


def append_jsonl(source_file, target_file):
    if not source_file.exists():
        raise SystemExit(f"Final generated file not found: {source_file}")

    source_data = source_file.read_bytes()
    if not source_data.strip():
        raise SystemExit(f"Final generated file is empty: {source_file}")

    target_file.parent.mkdir(parents=True, exist_ok=True)
    existing_count = count_jsonl_lines(target_file)
    with target_file.open("ab") as target:
        if target_file.exists() and target_file.stat().st_size > 0:
            with target_file.open("rb") as reader:
                reader.seek(-1, 2)
                if reader.read(1) != b"\n":
                    target.write(b"\n")
        target.write(source_data)
        if not source_data.endswith(b"\n"):
            target.write(b"\n")

    appended_count = count_jsonl_lines(source_file)
    final_count = count_jsonl_lines(target_file)
    print(
        f"[PIPELINE] Appended {appended_count} rows to {target_file}. "
        f"Rows before: {existing_count}; rows after: {final_count}."
    )


def prepare_append_seed_file(args):
    if not args.append_final_to or not args.append_final_to.exists():
        return

    existing_seeds = collect_persona_seeds(args.append_final_to)
    if not existing_seeds:
        return

    filtered_seed_file = args.output_dir.resolve() / "_available_seeds.jsonl"
    available_count = 0
    with args.seed_file.open("r", encoding="utf-8") as source, filtered_seed_file.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            seed = item.get("persona")
            if seed in existing_seeds:
                continue
            target.write(json.dumps(item, ensure_ascii=False) + "\n")
            available_count += 1

    if available_count < args.count:
        raise SystemExit(
            f"Only {available_count} unused persona seeds are available, but {args.count} are requested."
        )

    args.seed_file = filtered_seed_file
    print(
        f"[PIPELINE] Excluding {len(existing_seeds)} existing persona seeds from "
        f"{args.append_final_to.resolve()}."
    )


def collect_persona_seeds(jsonl_file):
    seeds = set()
    with jsonl_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            seed = item.get("metadata", {}).get("persona_seed")
            if seed:
                seeds.add(seed)
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the MemConflict construction pipeline from Step1_1 to Step4_4."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of new persona seeds to sample in Step1_1.",
    )
    parser.add_argument(
        "--target_total",
        type=int,
        default=None,
        help="Generate only enough new rows for append_final_to to reach this total.",
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
    parser.add_argument(
        "--append_final_to",
        type=Path,
        default=None,
        help="Append the generated Step4_4.jsonl rows to this existing JSONL file after the run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
