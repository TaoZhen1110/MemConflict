# MemConflict Pipeline Map

This file maps the implementation files to the construction and evaluation stages described in the paper. The original numbered filenames are kept because scripts and prompt files are coupled by stage number.

## Section 3: Benchmark Construction

| Paper stage | Main files | Prompt files | Description |
| --- | --- | --- | --- |
| 3.2 User profile initialization | `Code/Step1_1.py` | `Prompt/Prompt1_1.txt` | Builds invariant user profile information from persona seeds. |
| 3.2 User profile initialization | `Code/Step1_2.py` | `Prompt/Prompt1_2.txt` | Generates initial dynamic status attributes. |
| 3.2 User profile initialization | `Code/Step1_3.py` | `Prompt/Prompt1_3.txt` | Generates conditional preference attributes. |
| 3.2 User profile initialization | `Code/Step1_4.py` | `Prompt/Prompt1_4.txt` | Generates personality traits and life goals. |
| 3.2 User profile initialization | `Code/Step1_5.py` | `Prompt/Prompt1_5.txt` | Generates related-entity information for distractor construction. |
| 3.3.1 Timeline simulation | `Code/Step2_1.py` | - | Places initial profile mentions into early session slots. |
| 3.3.2 Dynamic conflict construction | `Code/Step2_2.py` | - | Simulates update sessions and accepted dynamic state transitions. |
| 3.3.3 Static conflict construction | `Code/Step3_1.py` | `Prompt/Prompt3_1.txt` | Constructs invariant-fact contradictions and static conflict metadata. |
| 3.3.4 Conditional conflict construction | `Code/Step3_2.py` | `Prompt/Prompt3_2.txt` | Constructs condition-value bindings and conditional conflict metadata. |
| 3.3.2 Dynamic query construction | `Code/Step3_3.py` | - | Converts dynamic state changes into evaluable conflict/query records. |
| 3.4 Dialogue generation | `Code/Step4_1.py` | `Prompt/Prompt4_1.txt` | Generates session synopses or outlines from session-level information. |
| 3.4 Dialogue generation | `Code/Step4_2.py` | `Prompt/Prompt4_2.txt` | Realizes each session synopsis as a multi-turn dialogue. |
| 3.7 Dataset statistics | `Code/Step4_3.py` | - | Computes token, turn, and context statistics. |
| 3.5 Query and label construction | `Code/Step4_4.py` | `Prompt/Prompt4_4.txt` | Refines questions and labels into the final benchmark format. |

## Section 3.6 and Section 4: Evaluation

| Paper component | Files | Description |
| --- | --- | --- |
| Main system evaluation | `Evaluation/eval_a_mem.py`, `Evaluation/eval_langmem.py`, `Evaluation/eval_letta.py`, `Evaluation/eval_memos.py`, `Evaluation/eval_memzero.py`, `Evaluation/eval_memobase.py` | Runs each memory system on the same MemConflict query set and exports answers plus retrieved memory items. |
| LLM-assisted scoring | `Evaluation/eval_scoring.py` and `Evaluation/scoring_*.py` | Computes black-box and white-box metrics using answer matching and gold-memory matching. |
| Failure diagnosis | `Evaluation/diagnose_failures.py` | Decomposes incorrect cases into retrieval failures and utilization failures. |
| LLM helper | `Evaluation/llm_request.py` | Shared OpenAI-compatible request wrapper for evaluation and scoring. |

## Section 4.4: Ablation and Sensitivity Analysis

| Paper analysis | Files | Description |
| --- | --- | --- |
| Dialogue length | `Ablation/Long/` | Builds and evaluates longer-history variants. |
| Distractor injection | `Ablation/No_Interference/` | Builds and evaluates variants without related-entity distractors. |
| Query formulation | `Ablation/Question_Style/` | Builds and evaluates implicit-query variants. |
| Conflict distance | `Ablation/Conflict_Interval/` | Builds and evaluates near-distance and far-distance variants. |

## Section 4 Figures

| Figure purpose | File |
| --- | --- |
| Retrieval depth sensitivity | `Figure/EXP4_4_1/plot_retrieval_depth.py` |
| Dialogue length sensitivity | `Figure/EXP4_4_2/plot_dialogue_length.py` |
| Distractor injection sensitivity | `Figure/EXP4_4_3/plot_distractor_injection.py` |
| Conflict distance sensitivity | `Figure/EXP4_4_5/plot_conflict_distance.py` |
| Failure decomposition | `Figure/EXP4_5_2/plot_failure_decomposition.py` |

## Naming Notes

- The `StepX_Y.py` files are intentionally kept in their original order to preserve reproducibility and alignment with the prompt files.
- The paper-facing stage names are documented in this file and in `README.md`, so the repository remains readable without breaking existing script dependencies.
- If a future refactor renames scripts to descriptive filenames, update all hardcoded prompt and data paths at the same time.
