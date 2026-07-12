#!/usr/bin/env python3
"""Build a file-level manifest for the HF benchmark source corpus."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any


SCENARIOS = {"alliance_betrayal", "info_withholding", "ultimatum_bluff", "all_scenarios"}
MODELS = {"gemma7b", "llama31_8b", "mistral7b"}
TIMESTAMP_RE = re.compile(r"(20\d{6}(?:_\d{6})?)")


def fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def infer_scenario(parts: list[str]) -> str:
    for part in parts:
        if part in SCENARIOS:
            return part
    return ""


def infer_model(parts: list[str]) -> str:
    for part in parts:
        if part in MODELS:
            return part
    return ""


def infer_source_group(parts: list[str], model_name: str) -> str:
    if not parts:
        return "unknown"
    if model_name:
        return "model_specific"
    if parts[0] in {"alliance_betrayal", "info_withholding", "ultimatum_bluff"}:
        return "top_level"
    return "other"


def infer_timestamp(path: str) -> str:
    match = TIMESTAMP_RE.search(path)
    return match.group(1) if match else ""


def infer_priority(path: str, is_transcript: bool, is_checkpoint: bool, model_name: str) -> str:
    if is_transcript and "all_scenarios" not in path and model_name:
        return "high"
    if is_transcript and "checkpoints" in path:
        return "high"
    if is_transcript:
        return "medium"
    if is_checkpoint:
        return "low"
    if path.endswith(".pt"):
        return "reference_only"
    return "ignore"


def build_rows(tree_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in tree_entries:
        if entry.get("type") != "file":
            continue

        path = entry["path"]
        parts = path.split("/")
        model_name = infer_model(parts)
        scenario = infer_scenario(parts)
        is_transcript = path.endswith("_transcripts.jsonl")
        is_checkpoint = "checkpoints" in path
        is_activation = path.endswith(".pt")

        rows.append(
            {
                "path": path,
                "type": Path(path).suffix.lstrip("."),
                "size_bytes": entry.get("size", ""),
                "model_name": model_name,
                "scenario": scenario,
                "source_group": infer_source_group(parts, model_name),
                "timestamp_hint": infer_timestamp(path),
                "is_transcript": str(is_transcript).lower(),
                "is_checkpoint": str(is_checkpoint).lower(),
                "is_activation": str(is_activation).lower(),
                "candidate_priority": infer_priority(path, is_transcript, is_checkpoint, model_name),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        default="sycorpia/multiagent-lab-data",
        help="HF dataset id in owner/name format",
    )
    parser.add_argument(
        "--tree-json",
        help="Optional local JSON file with the dataset tree. If omitted, fetch from the HF API.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_data/hf_manifest.csv",
        help="Path to the output CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.tree_json:
        with open(args.tree_json, "r", encoding="utf-8") as handle:
            tree_entries = json.load(handle)
    else:
        tree_url = f"https://huggingface.co/api/datasets/{args.dataset_id}/tree/main?recursive=true"
        tree_entries = fetch_json(tree_url)

    rows = build_rows(tree_entries)
    rows.sort(key=lambda row: (row["candidate_priority"], row["scenario"], row["model_name"], row["path"]))

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
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
