#!/usr/bin/env python3
"""Summarize transcript QC at the source-file level."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


FLAG_COLUMNS = (
    "prompt_overlap_high",
    "prompt_overlap_medium",
    "repetition_high",
    "repetition_medium",
    "truncation",
    "role_confusion",
    "prompt_leak_marker",
    "template_artifact",
    "scenario_restatement",
    "history_corruption",
    "label_alignment_unclear",
    "very_short_response",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qc",
        default="benchmark_data/qc_merged_round1_probe.csv",
        help="Merged QC CSV produced by qc_hf_transcripts.py and merge_qc_csv.py",
    )
    parser.add_argument(
        "--output",
        default="benchmark_data/source_quality_summary.csv",
        help="Output CSV path for the source quality summary",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=10,
        help="Minimum number of rows required to include a source file in the summary",
    )
    return parser.parse_args()


def parse_flags(value: str) -> set[str]:
    return {flag for flag in (value or "").split("|") if flag}


def safe_rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def compute_quality_score(metrics: dict[str, float]) -> float:
    score = 5.0
    score += metrics["keep_rate"] * 45.0
    score += metrics["task1_keep_share"] * 32.0
    score += metrics["task2_keep_share"] * 14.0
    score += metrics["task1_review_share"] * 6.0
    score += metrics["task2_review_share"] * 3.0
    score -= metrics["drop_rate"] * 14.0
    score -= metrics["prompt_leak_marker_rate"] * 12.0
    score -= metrics["template_artifact_rate"] * 12.0
    score -= metrics["scenario_restatement_rate"] * 10.0
    score -= metrics["role_confusion_rate"] * 8.0
    score -= metrics["repetition_high_rate"] * 5.0
    score -= metrics["prompt_overlap_high_rate"] * 5.0
    score -= metrics["truncation_rate"] * 5.0
    score -= metrics["very_short_response_rate"] * 4.0
    score -= max(0.0, metrics["avg_prompt_overlap"] - 0.12) * 40.0
    score -= max(0.0, metrics["avg_repetition"] - 0.08) * 60.0
    score += min(metrics["avg_response_length"], 500.0) / 50.0
    return max(0.0, min(100.0, score))


def main() -> int:
    args = parse_args()

    with Path(args.qc).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_file"]].append(row)

    fieldnames = [
        "source_file",
        "scenario",
        "model_name",
        "total_rows",
        "keep_rows",
        "review_rows",
        "drop_rows",
        "keep_rate",
        "review_rate",
        "drop_rate",
        "task1_rows",
        "task1_keep_rows",
        "task1_review_rows",
        "task1_keep_rate",
        "task1_review_rate",
        "task1_keep_share",
        "task1_review_share",
        "task2_rows",
        "task2_keep_rows",
        "task2_review_rows",
        "task2_keep_rate",
        "task2_review_rate",
        "task2_keep_share",
        "task2_review_share",
        "avg_prompt_overlap",
        "avg_repetition",
        "avg_response_length",
        "prompt_overlap_high_rate",
        "prompt_overlap_medium_rate",
        "repetition_high_rate",
        "repetition_medium_rate",
        "truncation_rate",
        "role_confusion_rate",
        "prompt_leak_marker_rate",
        "template_artifact_rate",
        "scenario_restatement_rate",
        "history_corruption_rate",
        "label_alignment_unclear_rate",
        "very_short_response_rate",
        "quality_score",
        "source_tier",
    ]

    output_rows: list[dict[str, str]] = []
    for source_file, source_rows in grouped.items():
        total_rows = len(source_rows)
        if total_rows < args.min_rows:
            continue

        status_counts = Counter(row["qc_status"] for row in source_rows)
        flag_counts = Counter()
        task1_rows = [row for row in source_rows if row["usable_for_task1"] == "true"]
        task2_rows = [row for row in source_rows if row["usable_for_task2"] == "true"]
        prompt_overlaps = [float(row["prompt_overlap_score"]) for row in source_rows]
        repetitions = [float(row["repetition_score"]) for row in source_rows]
        response_lengths = [int(row["response_length"]) for row in source_rows]

        for row in source_rows:
            flag_counts.update(parse_flags(row["qc_flags"]))

        metrics: dict[str, float] = {
            "keep_rows": status_counts["keep"],
            "review_rows": status_counts["review"],
            "drop_rows": status_counts["drop"],
            "keep_rate": safe_rate(status_counts["keep"], total_rows),
            "review_rate": safe_rate(status_counts["review"], total_rows),
            "drop_rate": safe_rate(status_counts["drop"], total_rows),
            "task1_rows": len(task1_rows),
            "task1_keep_rows": sum(row["qc_status"] == "keep" for row in task1_rows),
            "task1_review_rows": sum(row["qc_status"] == "review" for row in task1_rows),
            "task1_keep_rate": safe_rate(sum(row["qc_status"] == "keep" for row in task1_rows), total_rows),
            "task1_review_rate": safe_rate(sum(row["qc_status"] == "review" for row in task1_rows), total_rows),
            "task1_keep_share": safe_rate(sum(row["qc_status"] == "keep" for row in task1_rows), len(task1_rows)),
            "task1_review_share": safe_rate(sum(row["qc_status"] == "review" for row in task1_rows), len(task1_rows)),
            "task2_rows": len(task2_rows),
            "task2_keep_rows": sum(row["qc_status"] == "keep" for row in task2_rows),
            "task2_review_rows": sum(row["qc_status"] == "review" for row in task2_rows),
            "task2_keep_rate": safe_rate(sum(row["qc_status"] == "keep" for row in task2_rows), total_rows),
            "task2_review_rate": safe_rate(sum(row["qc_status"] == "review" for row in task2_rows), total_rows),
            "task2_keep_share": safe_rate(sum(row["qc_status"] == "keep" for row in task2_rows), len(task2_rows)),
            "task2_review_share": safe_rate(sum(row["qc_status"] == "review" for row in task2_rows), len(task2_rows)),
            "avg_prompt_overlap": sum(prompt_overlaps) / total_rows,
            "avg_repetition": sum(repetitions) / total_rows,
            "avg_response_length": sum(response_lengths) / total_rows,
        }

        for flag_name in FLAG_COLUMNS:
            metrics[f"{flag_name}_rate"] = safe_rate(flag_counts[flag_name], total_rows)

        metrics["quality_score"] = compute_quality_score(metrics)

        scenario = ""
        model_name = ""
        for row in source_rows:
            if row["scenario"] and not scenario:
                scenario = row["scenario"]
            if row["model_name"] and not model_name:
                model_name = row["model_name"]

        output_rows.append(
            {
                "source_file": source_file,
                "scenario": scenario,
                "model_name": model_name,
                "total_rows": str(total_rows),
                "keep_rows": str(int(metrics["keep_rows"])),
                "review_rows": str(int(metrics["review_rows"])),
                "drop_rows": str(int(metrics["drop_rows"])),
                "keep_rate": f"{metrics['keep_rate']:.4f}",
                "review_rate": f"{metrics['review_rate']:.4f}",
                "drop_rate": f"{metrics['drop_rate']:.4f}",
                "task1_rows": str(int(metrics["task1_rows"])),
                "task1_keep_rows": str(int(metrics["task1_keep_rows"])),
                "task1_review_rows": str(int(metrics["task1_review_rows"])),
                "task1_keep_rate": f"{metrics['task1_keep_rate']:.4f}",
                "task1_review_rate": f"{metrics['task1_review_rate']:.4f}",
                "task1_keep_share": f"{metrics['task1_keep_share']:.4f}",
                "task1_review_share": f"{metrics['task1_review_share']:.4f}",
                "task2_rows": str(int(metrics["task2_rows"])),
                "task2_keep_rows": str(int(metrics["task2_keep_rows"])),
                "task2_review_rows": str(int(metrics["task2_review_rows"])),
                "task2_keep_rate": f"{metrics['task2_keep_rate']:.4f}",
                "task2_review_rate": f"{metrics['task2_review_rate']:.4f}",
                "task2_keep_share": f"{metrics['task2_keep_share']:.4f}",
                "task2_review_share": f"{metrics['task2_review_share']:.4f}",
                "avg_prompt_overlap": f"{metrics['avg_prompt_overlap']:.4f}",
                "avg_repetition": f"{metrics['avg_repetition']:.4f}",
                "avg_response_length": f"{metrics['avg_response_length']:.1f}",
                "prompt_overlap_high_rate": f"{metrics['prompt_overlap_high_rate']:.4f}",
                "prompt_overlap_medium_rate": f"{metrics['prompt_overlap_medium_rate']:.4f}",
                "repetition_high_rate": f"{metrics['repetition_high_rate']:.4f}",
                "repetition_medium_rate": f"{metrics['repetition_medium_rate']:.4f}",
                "truncation_rate": f"{metrics['truncation_rate']:.4f}",
                "role_confusion_rate": f"{metrics['role_confusion_rate']:.4f}",
                "prompt_leak_marker_rate": f"{metrics['prompt_leak_marker_rate']:.4f}",
                "template_artifact_rate": f"{metrics['template_artifact_rate']:.4f}",
                "scenario_restatement_rate": f"{metrics['scenario_restatement_rate']:.4f}",
                "history_corruption_rate": f"{metrics['history_corruption_rate']:.4f}",
                "label_alignment_unclear_rate": f"{metrics['label_alignment_unclear_rate']:.4f}",
                "very_short_response_rate": f"{metrics['very_short_response_rate']:.4f}",
                "quality_score": f"{metrics['quality_score']:.2f}",
                "source_tier": "",
            }
        )

    by_scenario: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in output_rows:
        by_scenario[row["scenario"]].append(row)

    for scenario_rows in by_scenario.values():
        scenario_rows.sort(key=lambda row: (-float(row["quality_score"]), row["source_file"]))
        for rank, row in enumerate(scenario_rows):
            task1_keep = int(row["task1_keep_rows"])
            task2_keep = int(row["task2_keep_rows"])
            task1_review = int(row["task1_review_rows"])
            task2_review = int(row["task2_review_rows"])

            if rank == 0 and (task1_keep + task2_keep) >= 10:
                row["source_tier"] = "trusted"
            elif rank <= 2 and (task1_keep + task2_keep + task1_review + task2_review) >= 20:
                row["source_tier"] = "review_priority"
            else:
                row["source_tier"] = "avoid"

    output_rows.sort(
        key=lambda row: (
            {"trusted": 0, "review_priority": 1, "avoid": 2}.get(row["source_tier"], 3),
            -float(row["quality_score"]),
            row["source_file"],
        )
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} source summaries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
