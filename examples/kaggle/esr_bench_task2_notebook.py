# %% [markdown]
# # ESR-Bench Task 2
# ## Belief Tracking From The Same Negotiation Worlds
#
# Final Kaggle Benchmarks notebook for the Social Cognition Task 2 submission.

# %%
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import kaggle_benchmarks as kbench


os.environ["RENDER_SUBRUNS"] = "False"

CHOICE_LABELS = ("A", "B", "C", "D")
N_JOBS = 2
TIMEOUT_SECONDS = 120
RETRY_DELAY_SECONDS = 15
MAX_ATTEMPTS = 1
DEFAULT_TASK_PATHS = [
    Path("/kaggle/input/esr-bench-gold/task2_gold.csv"),
    Path("/kaggle/input/esr-bench-gold/benchmark_data/task2_gold.csv"),
    Path("/kaggle/input/datasets/annajsims/esr-bench-gold/task2_gold.csv"),
    Path("benchmark_data/task2_gold.csv"),
]


def resolve_task2_csv() -> Path:
    env_override = os.environ.get("ESR_BENCH_TASK2_CSV")
    if env_override:
        path = Path(env_override)
        if path.exists():
            return path

    for path in DEFAULT_TASK_PATHS:
        if path.exists():
            return path

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = sorted(kaggle_input.rglob("task2_gold.csv"))
        if matches:
            print(f"Found task2_gold.csv at: {matches[0]}")
            return matches[0]

    raise FileNotFoundError(
        "Could not find task2_gold.csv. Attach the gold dataset or set ESR_BENCH_TASK2_CSV."
    )


def normalize_choice(response: str) -> str | None:
    patterns = [
        re.compile(r"^\s*([ABCD])\s*$", re.IGNORECASE),
        re.compile(r"\boption\s*([ABCD])\b", re.IGNORECASE),
        re.compile(r"\banswer\s*[:=-]?\s*([ABCD])\b", re.IGNORECASE),
        re.compile(r"\bchoice\s*[:=-]?\s*([ABCD])\b", re.IGNORECASE),
        re.compile(r"\b([ABCD])\b", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(response or "")
        if match:
            return match.group(1).upper()
    return None


def build_prompt(row: pd.Series) -> str:
    options = "\n".join(
        [
            f"A. {row['option_a']}",
            f"B. {row['option_b']}",
            f"C. {row['option_c']}",
            f"D. {row['option_d']}",
        ]
    )
    return (
        "You are answering a multiple-choice social cognition benchmark.\n\n"
        f"Scenario:\n{row['scenario_context']}\n\n"
        f"Transcript Prefix:\n{row['transcript_prefix']}\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Options:\n{options}\n\n"
        "Reply with only one letter: A, B, C, or D."
    )


def load_task2_dataframe() -> pd.DataFrame:
    path = resolve_task2_csv()
    df = pd.read_csv(path)
    required = {
        "item_id",
        "scenario",
        "belief_tier",
        "difficulty_level",
        "incentive_condition",
        "scenario_context",
        "transcript_prefix",
        "question",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "correct_option",
        "correct_belief_target",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Task 2 CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["prompt"] = df.apply(build_prompt, axis=1)
    return df


TASK2_DF = load_task2_dataframe()
TASK2_EVAL_DF = TASK2_DF[
    [
        "item_id",
        "prompt",
        "correct_option",
        "scenario",
        "belief_tier",
        "difficulty_level",
        "correct_belief_target",
    ]
].copy()

print("Loaded Task 2 rows:", len(TASK2_DF))
print("First item:", TASK2_DF.iloc[0]["item_id"])


@kbench.task(store_task=False)
def esr_task2_item(
    llm,
    item_id: str,
    prompt: str,
    correct_option: str,
    scenario: str,
    belief_tier: str,
    difficulty_level: str,
    correct_belief_target: str,
) -> dict:
    response = str(llm.prompt(prompt))
    parsed_choice = normalize_choice(response)
    is_valid = parsed_choice in CHOICE_LABELS
    is_correct = parsed_choice == correct_option
    return {
        "item_id": item_id,
        "scenario": scenario,
        "belief_tier": belief_tier,
        "difficulty_level": difficulty_level,
        "correct_belief_target": correct_belief_target,
        "parsed_choice": parsed_choice,
        "raw_response": response,
        "is_valid_choice": is_valid,
        "is_correct": is_correct,
    }


@kbench.task(
    name="esr_task2_belief_tracking",
    description="Track what another agent most likely believes in a hidden-information negotiation.",
)
def esr_task2_benchmark(llm) -> float:
    with kbench.client.enable_cache():
        runs = esr_task2_item.evaluate(
            stop_condition=lambda runs: len(runs) == TASK2_EVAL_DF.shape[0],
            max_attempts=MAX_ATTEMPTS,
            retry_delay=RETRY_DELAY_SECONDS,
            llm=[llm],
            evaluation_data=TASK2_EVAL_DF,
            n_jobs=N_JOBS,
            timeout=TIMEOUT_SECONDS,
            remove_run_files=True,
        )

    eval_df = runs.as_dataframe()
    accuracy = float(eval_df.result.str.get("is_correct").mean())
    valid_rate = float(eval_df.result.str.get("is_valid_choice").mean())

    kbench.assertions.assert_equal(
        TASK2_EVAL_DF.shape[0],
        int(eval_df.shape[0]),
        expectation="Task 2 should evaluate every gold item exactly once.",
    )
    kbench.assertions.assert_true(
        0.0 <= accuracy <= 1.0,
        expectation="Task 2 accuracy must be a valid probability.",
    )
    kbench.assertions.assert_true(
        valid_rate >= 0.5,
        expectation="Task 2 models should usually answer with a parseable A/B/C/D choice.",
    )
    return accuracy


task2_run = esr_task2_benchmark.run(kbench.llm)
task2_run

# In the final Kaggle notebook cell, run:
# %choose esr_task2_benchmark
