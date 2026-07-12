#!/usr/bin/env python3
"""Seed a candidate item bank from QC-scored local transcripts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCENARIO_TO_DECEPTION_FAMILY = {
    "alliance_betrayal": "commission",
    "ultimatum_bluff": "commission",
    "info_withholding": "omission",
}
DISALLOWED_FLAGS = {
    "prompt_overlap_high",
    "repetition_high",
    "role_confusion",
    "prompt_leak_marker",
    "template_artifact",
    "scenario_restatement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qc",
        default="benchmark_data/qc_merged_round1_probe.csv",
        help="QC CSV produced by qc_hf_transcripts.py",
    )
    parser.add_argument(
        "--output",
        default="benchmark_data/candidate_item_bank.csv",
        help="Output candidate bank CSV",
    )
    parser.add_argument(
        "--source-summary",
        default="benchmark_data/source_quality_summary.csv",
        help="Optional source quality summary used to filter sources",
    )
    parser.add_argument(
        "--allowed-tiers",
        nargs="*",
        default=["trusted"],
        help="Source tiers allowed into the candidate bank. Start with trusted only.",
    )
    parser.add_argument(
        "--per-scenario-task1",
        type=int,
        default=12,
        help="Number of Task 1 candidates to seed per scenario",
    )
    parser.add_argument(
        "--per-scenario-task2",
        type=int,
        default=12,
        help="Number of Task 2 candidates to seed per scenario",
    )
    parser.add_argument(
        "--max-per-file",
        type=int,
        default=4,
        help="Maximum number of seeded candidates per source file and task",
    )
    return parser.parse_args()


def load_jsonl_cache() -> dict[str, list[dict[str, Any]]]:
    return {}


def get_file_rows(cache: dict[str, list[dict[str, Any]]], source_file: str) -> list[dict[str, Any]]:
    if source_file not in cache:
        with open(source_file, "r", encoding="utf-8") as handle:
            cache[source_file] = [json.loads(line) for line in handle if line.strip()]
    return cache[source_file]


def format_history(history: Any, limit: int = 4) -> str:
    if not history or not isinstance(history, list):
        return ""
    items = [str(item).strip() for item in history[-limit:] if str(item).strip()]
    return "\n".join(items)


def scenario_context(file_rows: list[dict[str, Any]], trial_id: str) -> str:
    target = None
    for row in file_rows:
        if str(row.get("trial_id")) == str(trial_id) and row.get("round_num") == -1:
            target = row
            break
    if target is None and file_rows:
        target = file_rows[0]
    if target is None:
        return ""
    prompt = str(target.get("prompt", "") or "").strip()
    response = str(target.get("response", "") or "").strip()
    text = f"{prompt}\n{response}".strip()
    return text[:1600]


def transcript_prefix(row: dict[str, Any]) -> str:
    history = format_history(row.get("dialogue_history"))
    prompt = str(row.get("prompt", "") or "").strip()
    if history and prompt:
        return f"{history}\nPROMPT: {prompt}"[:2200]
    return (history or prompt)[:2200]


def task1_score(row: dict[str, str]) -> float:
    score = 100.0 if row["qc_status"] == "keep" else 85.0
    flags = set(filter(None, (row.get("qc_flags") or "").split("|")))
    score -= 10.0 * len(flags & DISALLOWED_FLAGS)
    if "truncation" in flags:
        score -= 8.0
    if "history_corruption" in flags:
        score -= 5.0
    response_length = int(row["response_length"])
    if 50 <= response_length <= 500:
        score += 8.0
    elif response_length < 30:
        score -= 15.0
    elif response_length > 900:
        score -= 10.0
    round_num = int(row["round_num"])
    if 0 <= round_num <= 4:
        score += 5.0
    actual_deception = str(row.get("actual_deception", ""))
    if actual_deception not in {"", "0.0", "0"}:
        score += 3.0
    return score


def task2_score(row: dict[str, str]) -> float:
    score = 100.0 if row["qc_status"] == "keep" else 85.0
    flags = set(filter(None, (row.get("qc_flags") or "").split("|")))
    score -= 10.0 * len(flags & DISALLOWED_FLAGS)
    if "history_corruption" in flags:
        score -= 4.0
    round_num = int(row["round_num"])
    if round_num >= 1:
        score += 10.0
    if round_num >= 2:
        score += 5.0
    response_length = int(row["response_length"])
    if 40 <= response_length <= 600:
        score += 5.0
    elif response_length > 1000:
        score -= 10.0
    return score


def eligible(row: dict[str, str], task_key: str) -> bool:
    if row["qc_status"] == "drop":
        return False
    flags = set(filter(None, (row.get("qc_flags") or "").split("|")))
    if flags & DISALLOWED_FLAGS:
        return False
    return row[task_key] == "true"


def seed_task_candidates(
    rows: list[dict[str, str]],
    scenarios: list[str],
    per_scenario: int,
    max_per_file: int,
    task_name: str,
    task_key: str,
    score_fn,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    per_file_counts: dict[tuple[str, str], int] = defaultdict(int)

    for scenario in scenarios:
        eligible_rows = [row for row in rows if row["scenario"] == scenario and eligible(row, task_key)]
        eligible_rows.sort(key=score_fn, reverse=True)

        picked = 0
        for row in eligible_rows:
            file_key = (task_name, row["source_file"])
            if per_file_counts[file_key] >= max_per_file:
                continue
            per_file_counts[file_key] += 1
            candidates.append(dict(row, task_name=task_name))
            picked += 1
            if picked >= per_scenario:
                break

    return candidates


def infer_difficulty(task_name: str, round_num: int) -> str:
    if task_name == "task1_strategic_action_selection":
        if round_num <= 1:
            return "easy"
        if round_num <= 3:
            return "medium"
        return "hard"
    if round_num <= 1:
        return "tier1_candidate"
    if round_num == 2:
        return "tier2_candidate"
    return "tier3_candidate"


def infer_belief_tier(round_num: int) -> str:
    if round_num <= 1:
        return "tier1_candidate"
    if round_num == 2:
        return "tier2_candidate"
    return "tier3_candidate"


def main() -> int:
    args = parse_args()

    with open(args.qc, "r", encoding="utf-8", newline="") as handle:
        qc_rows = list(csv.DictReader(handle))
    with open(args.source_summary, "r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    allowed_sources = {
        row["source_file"]
        for row in source_rows
        if row.get("source_tier") in set(args.allowed_tiers)
    }
    if allowed_sources:
        qc_rows = [row for row in qc_rows if row["source_file"] in allowed_sources]

    scenarios = sorted({row["scenario"] for row in qc_rows if row["scenario"]})

    task1_candidates = seed_task_candidates(
        qc_rows,
        scenarios,
        args.per_scenario_task1,
        args.max_per_file,
        "task1_strategic_action_selection",
        "usable_for_task1",
        task1_score,
    )
    task2_candidates = seed_task_candidates(
        qc_rows,
        scenarios,
        args.per_scenario_task2,
        args.max_per_file,
        "task2_belief_tracking",
        "usable_for_task2",
        task2_score,
    )

    all_candidates = task1_candidates + task2_candidates
    jsonl_cache = load_jsonl_cache()

    fieldnames = [
        "candidate_id",
        "task_name",
        "source_file",
        "row_index",
        "model_name",
        "scenario",
        "trial_id",
        "round_num",
        "incentive_condition",
        "deception_family",
        "belief_tier",
        "difficulty_level",
        "source_prompt",
        "source_response",
        "source_dialogue_history",
        "transcript_prefix",
        "scenario_context",
        "question",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "correct_option",
        "rationale",
        "qc_status",
        "qc_flags",
        "candidate_status",
        "reviewer_notes",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for idx, candidate in enumerate(all_candidates, start=1):
            source_file = candidate["source_file"]
            row_index = int(candidate["row_index"])
            file_rows = get_file_rows(jsonl_cache, source_file)
            source_row = file_rows[row_index]
            round_num = int(candidate["round_num"])
            task_name = candidate["task_name"]
            scenario = candidate["scenario"]

            writer.writerow(
                {
                    "candidate_id": f"{task_name}_{idx:04d}",
                    "task_name": task_name,
                    "source_file": source_file,
                    "row_index": row_index,
                    "model_name": candidate["model_name"],
                    "scenario": scenario,
                    "trial_id": candidate["trial_id"],
                    "round_num": round_num,
                    "incentive_condition": candidate["incentive_condition"],
                    "deception_family": SCENARIO_TO_DECEPTION_FAMILY.get(scenario, ""),
                    "belief_tier": infer_belief_tier(round_num) if task_name == "task2_belief_tracking" else "",
                    "difficulty_level": infer_difficulty(task_name, round_num),
                    "source_prompt": str(source_row.get("prompt", "") or "")[:1200],
                    "source_response": str(source_row.get("response", "") or "")[:1600],
                    "source_dialogue_history": format_history(source_row.get("dialogue_history"), limit=4)[:1600],
                    "transcript_prefix": transcript_prefix(source_row),
                    "scenario_context": scenario_context(file_rows, candidate["trial_id"]),
                    "question": "",
                    "option_a": "",
                    "option_b": "",
                    "option_c": "",
                    "option_d": "",
                    "correct_option": "",
                    "rationale": "",
                    "qc_status": candidate["qc_status"],
                    "qc_flags": candidate["qc_flags"],
                    "candidate_status": "seed",
                    "reviewer_notes": "",
                }
            )

    print(f"Wrote {len(all_candidates)} candidates to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
