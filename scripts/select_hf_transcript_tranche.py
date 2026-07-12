#!/usr/bin/env python3
"""Select a balanced transcript tranche from the HF manifest."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


SCENARIOS = ("alliance_betrayal", "info_withholding", "ultimatum_bluff")
MODEL_BUCKETS = ("top_level", "gemma7b", "llama31_8b", "mistral7b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="benchmark_data/hf_manifest.csv",
        help="Input HF manifest CSV.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_data/tranche_round1.csv",
        help="Output CSV for the selected tranche.",
    )
    parser.add_argument(
        "--base-url",
        default="https://huggingface.co/datasets/sycorpia/multiagent-lab-data/resolve/main",
        help="Base URL used to build direct file URLs.",
    )
    parser.add_argument(
        "--downloads-root",
        default="benchmark_data/downloads",
        help="Local root where selected files should be downloaded.",
    )
    parser.add_argument(
        "--top-level-per-scenario",
        type=int,
        default=4,
        help="Number of top-level transcript files to select per scenario.",
    )
    parser.add_argument(
        "--per-model-per-scenario",
        type=int,
        default=2,
        help="Number of model-specific transcript files to select per scenario.",
    )
    parser.add_argument(
        "--include-medium",
        action="store_true",
        help="Include medium-priority transcripts if there are not enough high-priority files.",
    )
    return parser.parse_args()


def sort_key(row: dict[str, str]) -> tuple[int, str, str]:
    timestamp = row.get("timestamp_hint", "")
    is_checkpoint = row.get("is_checkpoint") == "true"
    # Prefer later timestamps, then checkpoint files, then path as a stable tie-breaker.
    return (0 if is_checkpoint else 1, timestamp, row["path"])


def load_manifest(path: str, include_medium: bool) -> list[dict[str, str]]:
    allowed_priorities = {"high"}
    if include_medium:
        allowed_priorities.add("medium")

    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    filtered: list[dict[str, str]] = []
    for row in rows:
        if row.get("is_transcript") != "true":
            continue
        if row.get("scenario") not in SCENARIOS:
            continue
        if row.get("candidate_priority") not in allowed_priorities:
            continue
        filtered.append(row)

    return filtered


def bucket_name(row: dict[str, str]) -> str:
    model_name = row.get("model_name", "")
    return model_name if model_name else "top_level"


def choose_rows(
    rows: list[dict[str, str]],
    top_level_per_scenario: int,
    per_model_per_scenario: int,
) -> list[dict[str, str]]:
    by_scenario_bucket: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_scenario_bucket[row["scenario"]][bucket_name(row)].append(row)

    for scenario in SCENARIOS:
        for bucket in list(by_scenario_bucket[scenario].keys()):
            by_scenario_bucket[scenario][bucket].sort(key=sort_key, reverse=True)

    selected: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    target_per_scenario = top_level_per_scenario + per_model_per_scenario * 3

    for scenario in SCENARIOS:
        top_rows = by_scenario_bucket[scenario].get("top_level", [])[:top_level_per_scenario]
        for row in top_rows:
            if row["path"] in seen_paths:
                continue
            selected.append(row)
            seen_paths.add(row["path"])

        for model_name in ("gemma7b", "llama31_8b", "mistral7b"):
            model_rows = by_scenario_bucket[scenario].get(model_name, [])[:per_model_per_scenario]
            for row in model_rows:
                if row["path"] in seen_paths:
                    continue
                selected.append(row)
                seen_paths.add(row["path"])

        # Backfill scenarios that do not have all desired buckets represented.
        if len([r for r in selected if r["scenario"] == scenario]) < target_per_scenario:
            remaining = []
            for bucket_rows in by_scenario_bucket[scenario].values():
                for row in bucket_rows:
                    if row["path"] not in seen_paths:
                        remaining.append(row)
            remaining.sort(key=sort_key, reverse=True)

            needed = target_per_scenario - len([r for r in selected if r["scenario"] == scenario])
            for row in remaining[:needed]:
                selected.append(row)
                seen_paths.add(row["path"])

    return selected


def main() -> int:
    args = parse_args()
    rows = load_manifest(args.manifest, args.include_medium)
    selected = choose_rows(rows, args.top_level_per_scenario, args.per_model_per_scenario)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path",
        "type",
        "size_bytes",
        "model_name",
        "scenario",
        "source_group",
        "timestamp_hint",
        "is_transcript",
        "is_checkpoint",
        "is_activation",
        "candidate_priority",
        "bucket",
        "source_url",
        "local_path",
        "selection_reason",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in selected:
            bucket = bucket_name(row)
            local_path = str(Path(args.downloads_root) / row["path"])
            selection_reason = f"{row['scenario']}:{bucket}:{row.get('timestamp_hint', '')}"
            writer.writerow(
                {
                    **row,
                    "bucket": bucket,
                    "source_url": f"{args.base_url.rstrip('/')}/{row['path']}",
                    "local_path": local_path,
                    "selection_reason": selection_reason,
                }
            )

    print(f"Wrote {len(selected)} selected transcript files to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
