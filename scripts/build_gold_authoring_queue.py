#!/usr/bin/env python3
"""Build a manual authoring queue from QC-ranked local transcripts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


META_RESPONSE_MARKERS = (
    "exercise:",
    "answer:",
    "current negotiation situation",
    "negotiation summary",
    "negotiation status",
    "negotiation decision",
    "prior negotiation",
    "this approach aims",
    "target value:",
    "reservation value",
    "bonus:",
    "competitive tactics",
    "empathy mode",
    "self evaluation",
    "motivation analysis",
    "recursive reasoning",
    "emotional state",
    "theory of mind",
    "artificial intelligence analysis",
    "level 1",
    "level 2",
    "level 3",
    "[your move]",
    "please provide",
)

ACTION_MARKERS = {
    "alliance_betrayal": ("alliance", "finalize", "deal", "offer", "betray", "player b", "player c"),
    "info_withholding": ("disclose", "issue", "repair", "buyer", "car", "price", "offer"),
    "ultimatum_bluff": ("accept", "reject", "counter", "offer", "points", "minimum", "walk away"),
}

SUGGESTED_FOCUS = {
    "alliance_betrayal": ("strategic_commitment", "belief_about_ally", "betrayal_signal"),
    "info_withholding": ("defect_disclosure", "buyer_belief", "price_anchoring"),
    "ultimatum_bluff": ("reservation_bluff", "offer_selection", "walkaway_belief"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qc",
        default="benchmark_data/qc_merged_round1_probe.csv",
        help="Merged QC CSV to mine from",
    )
    parser.add_argument(
        "--source-summary",
        default="benchmark_data/source_quality_summary.csv",
        help="Source quality summary created by summarize_qc_sources.py",
    )
    parser.add_argument(
        "--output",
        default="benchmark_data/gold_authoring_queue.csv",
        help="Output CSV path for the manual authoring queue",
    )
    parser.add_argument(
        "--per-scenario-task1",
        type=int,
        default=12,
        help="Maximum Task 1 queue rows per scenario",
    )
    parser.add_argument(
        "--per-scenario-task2",
        type=int,
        default=12,
        help="Maximum Task 2 queue rows per scenario",
    )
    parser.add_argument(
        "--max-per-source-task",
        type=int,
        default=6,
        help="Maximum queue rows per source file and task",
    )
    parser.add_argument(
        "--allowed-tiers",
        nargs="*",
        default=["trusted"],
        help="Source tiers allowed into the queue. Start with trusted only and widen later if needed.",
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


def transcript_prefix(row: dict[str, Any]) -> str:
    history = format_history(row.get("dialogue_history"), limit=4)
    prompt = str(row.get("prompt", "") or "").strip()
    if history and prompt:
        return f"{history}\nPROMPT: {prompt}"[:2400]
    return (history or prompt)[:2400]


def scenario_context(file_rows: list[dict[str, Any]], trial_id: str) -> str:
    context_rows = [row for row in file_rows if str(row.get("trial_id")) == str(trial_id)]
    if not context_rows:
        return ""

    warmup = next((row for row in context_rows if row.get("round_num") == -1), None)
    target = warmup or context_rows[0]
    prompt = str(target.get("prompt", "") or "").strip()
    response = str(target.get("response", "") or "").strip()
    return f"{prompt}\n{response}".strip()[:1800]


def parse_flags(value: str) -> set[str]:
    return {flag for flag in (value or "").split("|") if flag}


def meta_marker_hits(text: str) -> int:
    lowered = text.lower()
    return sum(marker in lowered for marker in META_RESPONSE_MARKERS)


def action_marker_hits(text: str, scenario: str) -> int:
    lowered = text.lower()
    return sum(marker in lowered for marker in ACTION_MARKERS.get(scenario, ()))


def history_turn_count(history: Any) -> int:
    return len(history) if isinstance(history, list) else 0


def task1_authoring_score(row: dict[str, str], source_quality_score: float, source_row: dict[str, Any]) -> float:
    response = str(source_row.get("response", "") or "")
    prompt = str(source_row.get("prompt", "") or "")
    history = source_row.get("dialogue_history")
    history_text = format_history(history, limit=6)
    flags = parse_flags(row["qc_flags"])
    scenario = row["scenario"]

    score = 0.0
    score += 30.0 if row["qc_status"] == "keep" else 18.0
    score += min(source_quality_score, 100.0) * 0.25
    score += min(action_marker_hits(response + "\n" + prompt + "\n" + history_text, scenario), 4) * 4.0
    score += min(history_turn_count(history), 4) * 1.5
    score += 4.0 if 0 <= int(row["round_num"]) <= 3 else 1.0
    score += 3.0 if str(row.get("actual_deception", "")) not in {"", "0", "0.0"} else 0.0

    if "truncation" in flags:
        score -= 8.0
    if "very_short_response" in flags:
        score -= 10.0
    if "history_corruption" in flags:
        score -= 5.0
    score -= meta_marker_hits(response) * 6.0
    score -= meta_marker_hits(history_text) * 3.0
    score -= meta_marker_hits(prompt) * 2.0

    response_length = len(response.strip())
    if 40 <= response_length <= 500:
        score += 5.0
    elif response_length < 25:
        score -= 8.0
    elif response_length > 1000:
        score -= 6.0

    return score


def task2_authoring_score(row: dict[str, str], source_quality_score: float, source_row: dict[str, Any]) -> float:
    response = str(source_row.get("response", "") or "")
    prompt = str(source_row.get("prompt", "") or "")
    history = source_row.get("dialogue_history")
    history_text = format_history(history, limit=6)
    flags = parse_flags(row["qc_flags"])

    score = 0.0
    score += 28.0 if row["qc_status"] == "keep" else 16.0
    score += min(source_quality_score, 100.0) * 0.22
    score += min(history_turn_count(history), 6) * 3.0
    score += 8.0 if int(row["round_num"]) >= 1 else 0.0
    score += 6.0 if "Counterpart:" in history_text and "Negotiator:" in history_text else 0.0
    score += min(action_marker_hits(history_text + "\n" + prompt + "\n" + response, row["scenario"]), 3) * 2.0

    if "truncation" in flags:
        score -= 5.0
    if "very_short_response" in flags:
        score -= 6.0
    if "history_corruption" in flags:
        score -= 8.0
    score -= meta_marker_hits(response) * 4.0
    score -= meta_marker_hits(history_text) * 2.0
    score -= meta_marker_hits(prompt) * 1.5

    return score


def proposed_difficulty(task_name: str, round_num: int) -> str:
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


def build_rows(
    qc_rows: list[dict[str, str]],
    source_lookup: dict[str, dict[str, str]],
    per_scenario: int,
    max_per_source_task: int,
    task_name: str,
    task_key: str,
    score_fn,
) -> list[dict[str, str]]:
    scenarios = sorted({row["scenario"] for row in qc_rows if row["scenario"]})
    picked_rows: list[dict[str, str]] = []
    cache = load_jsonl_cache()
    per_source_counts: defaultdict[tuple[str, str], int] = defaultdict(int)

    for scenario in scenarios:
        eligible = []
        for row in qc_rows:
            if row["scenario"] != scenario or row[task_key] != "true" or row["qc_status"] == "drop":
                continue
            source_summary = source_lookup.get(row["source_file"])
            if not source_summary:
                continue
            file_rows = get_file_rows(cache, row["source_file"])
            source_row = file_rows[int(row["row_index"])]
            score = score_fn(row, float(source_summary["quality_score"]), source_row)
            eligible.append((score, row, source_summary, source_row, file_rows))

        eligible.sort(key=lambda item: item[0], reverse=True)

        picked = 0
        for score, row, source_summary, source_row, file_rows in eligible:
            source_key = (task_name, row["source_file"])
            if per_source_counts[source_key] >= max_per_source_task:
                continue
            per_source_counts[source_key] += 1
            picked += 1

            picked_rows.append(
                {
                    "queue_id": f"{task_name}_{len(picked_rows)+1:04d}",
                    "proposed_task": task_name,
                    "source_tier": source_summary["source_tier"],
                    "source_quality_score": source_summary["quality_score"],
                    "authoring_score": f"{score:.2f}",
                    "source_file": row["source_file"],
                    "row_index": row["row_index"],
                    "model_name": row["model_name"],
                    "scenario": row["scenario"],
                    "trial_id": row["trial_id"],
                    "round_num": row["round_num"],
                    "incentive_condition": row["incentive_condition"],
                    "actual_deception": row["actual_deception"],
                    "qc_status": row["qc_status"],
                    "qc_flags": row["qc_flags"],
                    "difficulty_level": proposed_difficulty(task_name, int(row["round_num"])),
                    "suggested_focus": SUGGESTED_FOCUS.get(row["scenario"], ("",))[0 if task_name == "task1_strategic_action_selection" else 1],
                    "source_prompt": str(source_row.get("prompt", "") or "")[:1200],
                    "source_response": str(source_row.get("response", "") or "")[:1600],
                    "source_dialogue_history": format_history(source_row.get("dialogue_history"), limit=4)[:1800],
                    "transcript_prefix": transcript_prefix(source_row),
                    "scenario_context": scenario_context(file_rows, row["trial_id"]),
                    "queue_status": "queued",
                    "author_notes": "",
                }
            )
            if picked >= per_scenario:
                break

    return picked_rows


def main() -> int:
    args = parse_args()

    with open(args.qc, "r", encoding="utf-8", newline="") as handle:
        qc_rows = list(csv.DictReader(handle))
    with open(args.source_summary, "r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    allowed_tiers = set(args.allowed_tiers)
    source_lookup = {
        row["source_file"]: row
        for row in source_rows
        if row["source_tier"] in allowed_tiers
    }

    task1_rows = build_rows(
        qc_rows,
        source_lookup,
        args.per_scenario_task1,
        args.max_per_source_task,
        "task1_strategic_action_selection",
        "usable_for_task1",
        task1_authoring_score,
    )
    task2_rows = build_rows(
        qc_rows,
        source_lookup,
        args.per_scenario_task2,
        args.max_per_source_task,
        "task2_belief_tracking",
        "usable_for_task2",
        task2_authoring_score,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "queue_id",
        "proposed_task",
        "source_tier",
        "source_quality_score",
        "authoring_score",
        "source_file",
        "row_index",
        "model_name",
        "scenario",
        "trial_id",
        "round_num",
        "incentive_condition",
        "actual_deception",
        "qc_status",
        "qc_flags",
        "difficulty_level",
        "suggested_focus",
        "source_prompt",
        "source_response",
        "source_dialogue_history",
        "transcript_prefix",
        "scenario_context",
        "queue_status",
        "author_notes",
    ]

    rows = task1_rows + task2_rows
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} authoring queue rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
