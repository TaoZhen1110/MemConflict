# MemConflict

MemConflict is a benchmark and evaluation toolkit for studying long-term memory systems under memory conflicts. It constructs long-horizon multi-session histories, introduces dynamic, static, and conditional conflicts, injects semantically similar distractors, and evaluates memory systems with both black-box answer accuracy and white-box memory retrieval/ranking metrics.

This repository contains the construction scripts, prompt templates, evaluation scripts, ablation scripts, and plotting scripts used in the paper:

**MemConflict: Evaluating Long-Term Memory Systems under Memory Conflicts**

## Repository Structure

| Path | Purpose |
| --- | --- |
| `Code/` | Main benchmark construction pipeline, aligned with Section 3 of the paper. |
| `Data/` | Released benchmark data in JSONL format. |
| `Data_perfect/` | Intermediate structured data used by the construction pipeline. |
| `Prompt/` | Prompt templates used by LLM-assisted construction and query refinement. |
| `Evaluation/` | Memory-system evaluation, LLM-assisted scoring, and failure diagnosis scripts. |
| `Ablation/` | Scripts for dialogue-length, distractor, query-formulation, and conflict-distance variants. |
| `Figure/` | Plotting scripts for experimental figures. |
| `PIPELINE.md` | Detailed mapping between paper sections and implementation files. |

Experiment result files, logs, local model files, and private environment files are excluded from the release copy and ignored by default. The benchmark data are retained in `Data/` and `Data_perfect/`.

## Setup

Create an environment and install the core dependencies:

```bash
pip install -r requirements.txt
```

Create a local `.env` file from `.env.example` and set the LLM credentials:

```bash
cp .env.example .env
```

The evaluation scripts for external memory systems may require additional packages and services. Install A-Mem, LangMem, Letta, MemOS, Mem0, and Memobase following their official documentation before running the corresponding `Evaluation/eval_*.py` scripts.

## Running the Pipeline

The main construction scripts are organized in numbered stages:

```text
Step1_*  User profile initialization
Step2_*  Timeline simulation and dynamic state transitions
Step3_*  Conflict and distractor construction
Step4_*  Dialogue generation, query construction, and dataset statistics
```

See `PIPELINE.md` for the file-level mapping. The released benchmark files are provided under `Data/`, while `Data_perfect/` contains intermediate structured construction outputs. The scripts assume a repository layout named `MemConflict`; if the folder is renamed locally, update hardcoded paths or run from a parent directory that contains this repository as `MemConflict`.

## Evaluation

System-specific evaluation scripts are located in `Evaluation/`:

```text
eval_a_mem.py
eval_langmem.py
eval_letta.py
eval_memos.py
eval_memzero.py
eval_memobase.py
```

The scoring scripts compute the metrics described in the paper, including AA, SEH@K, SRS, UOCS, CRS, and Evidence Utilization Gap diagnostics. LLM-assisted answer and memory-item judgments should be followed by human verification, matching the evaluation protocol in the paper.

## Notes for Reproducibility

- Keep private keys in `.env`; do not commit `.env`.
- New experiment outputs should remain under `Results/` or `Scores/`.
- Prompt templates in `Prompt/` are the implementation prompts; the appendix presents condensed, publication-friendly versions.
- Numerical results depend on the selected LLM backend, memory-system versions, and system-specific configuration.
