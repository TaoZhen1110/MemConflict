# MemConflict

**MemConflict: Evaluating Long-Term Memory Systems under Memory Conflicts**

MemConflict is a benchmark and evaluation toolkit for diagnosing how long-term memory systems handle conflicting user memories in multi-session interactions. It evaluates whether a system can retrieve and use the memory item that is temporally valid, factually correct, and contextually applicable for the current query.

## Overview

Long-term memory systems for LLM agents must do more than store previous interactions. In realistic multi-session settings, a memory store may contain outdated states, later updates, contradictory mentions, condition-dependent preferences, and semantically similar memories about related entities. MemConflict turns these situations into controlled benchmark instances and evaluates memory systems from both answer-level and memory-level perspectives.

MemConflict supports three conflict types:

| Conflict type | Validity dimension | Core challenge |
| --- | --- | --- |
| Dynamic conflict | Temporal validity | Identify the current valid state after true user updates. |
| Static conflict | Factual correctness | Preserve stable user facts despite later false contradictions. |
| Conditional conflict | Contextual applicability | Recover the correct condition-value association for a query. |

The benchmark also injects semantically similar distractors from related entities, creating realistic retrieval competition without changing the target user's gold memory.

## Repository Structure

| Path | Description |
| --- | --- |
| `Code/` | Main construction pipeline for user profiles, timelines, conflicts, dialogues, and queries. |
| `Data/` | Released benchmark data in JSONL format. |
| `Prompt/` | Full implementation prompts used by LLM-assisted construction and query refinement. |
| `Evaluation/` | Evaluation scripts for memory systems, scoring, and failure diagnosis. |
| `Ablation/` | Scripts for sensitivity analyses, including dialogue length, distractors, query style, and conflict distance. |
| `PIPELINE.md` | File-level mapping between the paper sections and implementation scripts. |
| `requirements.txt` | Core Python dependencies. |

Experiment outputs, logs, local model files, and private environment files are intentionally excluded.

## Dataset

The released data are stored in `Data/`. The final benchmark file is:

```text
Data/Step4_4.jsonl
```

Each instance contains a multi-session dialogue history, conflict metadata, query information, gold labels, and fields used for black-box and white-box evaluation. Earlier `Step*.jsonl` files are retained to make the construction process inspectable.

## Installation

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file if you need to run LLM-assisted construction or scoring:

```bash
cp .env.example .env
```

Then set:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-5.0-mini
```

External memory-system backends may require additional installation steps. Please install A-Mem, LangMem, Letta, MemOS, Mem0, and Memobase according to their official documentation before running the corresponding evaluation scripts.

## Construction Pipeline

The main construction pipeline follows the paper structure:

```text
Step1_*  User profile initialization
Step2_*  Timeline simulation and dynamic state transitions
Step3_*  Conflict and distractor construction
Step4_*  Dialogue generation, query construction, and dataset statistics
```

The numbered filenames are kept to preserve the original construction order and their alignment with the prompt files. See `PIPELINE.md` for a detailed mapping between scripts, prompts, and paper sections.

## Evaluation

System-specific evaluation scripts are provided in `Evaluation/`:

```text
eval_a_mem.py
eval_langmem.py
eval_letta.py
eval_memos.py
eval_memzero.py
eval_memobase.py
```

MemConflict reports:

| Metric | Level | Meaning |
| --- | --- | --- |
| AA | Black-box | Whether the final answer matches the gold answer. |
| SEH@K | White-box | Whether the top-K retrieved memories contain the gold memory item. |
| SRS | White-box | How highly the gold memory item is ranked. |
| UOCS | Dynamic diagnostic | Whether the system recognizes update order. |
| CRS | Static diagnostic | Whether the system recognizes contradictory candidates. |
| EUG | Reliability diagnostic | Whether retrieved gold memories are converted into correct answers. |

LLM-assisted answer and memory-item matching should be followed by human verification, consistent with the evaluation protocol in the paper.

## Ablation and Diagnostics

The `Ablation/` directory contains scripts for the controlled factors studied in the paper:

| Analysis | Directory |
| --- | --- |
| Longer dialogue histories | `Ablation/Long/` |
| Removing distractors | `Ablation/No_Interference/` |
| Implicit query formulation | `Ablation/Question_Style/` |
| Near/far conflict distance | `Ablation/Conflict_Interval/` |

Failure diagnosis scripts decompose incorrect answers into retrieval failures and utilization failures, helping identify whether a system misses the gold memory item or fails to use it after retrieval.

## Notes

- Keep private API keys in `.env`; do not commit `.env`.
- New experiment outputs should be written under `Results/` or `Scores/`.
- Prompt templates in `Prompt/` are the full implementation prompts. The paper appendix presents condensed, publication-friendly versions.
- Numerical results may vary with LLM backend, memory-system versions, and local system configuration.

## Citation

If you use MemConflict, please cite our paper. The BibTeX entry will be updated after publication:

```bibtex
@article{memconflict2026,
  title   = {MemConflict: Evaluating Long-Term Memory Systems under Memory Conflicts},
  author  = {To be updated},
  journal = {To be updated},
  year    = {2026}
}
```
