#!/usr/bin/env python3
"""Run automatic QC on transcript JSONL files from the HF benchmark corpus."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
PROMPT_LEAK_MARKERS = (
    "you are negotiating",
    "bonus structure",
    "the other party will make offers",
    "begin the negotiation",
)
ROLE_MARKERS = ("Negotiator:", "Counterpart:", "You:", "Other party:")
TEMPLATE_ARTIFACT_MARKERS = (
    "please provide the answer to the exercise",
    "please provide your answer",
    "please provide your response",
    "the final answer is",
    "**answer:**",
    "**your answer:**",
    "**your response:**",
    "[your response here]",
    "select an option below",
    "please select",
    "please choose your next action",
    "choose your next action",
    "**make your move:**",
    "make your move:",
    "this is an exercise in negotiation simulation",
    "simulated negotiation exercise",
    "this text describes a negotiation scenario",
    "competitive strategy:",
    "negotiation memory:",
)
SCENARIO_RESTATEMENT_MARKERS = {
    "alliance_betrayal": (
        "secret offer from c",
        "player b messages you",
        "honor alliance",
        "accept c's offer",
    ),
    "ultimatum_bluff": (
        "true minimum acceptable amount",
        "the other party will make offers",
        "declare offers as",
        "maximum possible bonus",
    ),
    "info_withholding": (
        "known issue",
        "buyer asks",
        "disclose the issue",
        "withhold the issue",
    ),
}


def normalize_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def token_overlap_score(prompt: str, response: str) -> float:
    prompt_tokens = set(normalize_tokens(prompt))
    response_tokens = normalize_tokens(response)
    if not response_tokens:
        return 0.0
    overlap = sum(1 for tok in response_tokens if tok in prompt_tokens)
    return overlap / max(1, len(response_tokens))


def repetition_score(text: str, ngram_size: int = 5) -> float:
    tokens = normalize_tokens(text)
    if len(tokens) < ngram_size:
        return 0.0
    ngrams = [tuple(tokens[i : i + ngram_size]) for i in range(len(tokens) - ngram_size + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(ngrams))


def truncation_flag(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith((".", "?", "!", "\"", "'", "</s>")):
        return False
    if len(stripped) < 40:
        return False
    return True


def role_confusion_flag(text: str) -> bool:
    hits = sum(text.count(marker) for marker in ROLE_MARKERS)
    leak_hits = sum(text.lower().count(marker) for marker in PROMPT_LEAK_MARKERS)
    return hits >= 3 or leak_hits >= 2


def prompt_leak_marker_flag(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in PROMPT_LEAK_MARKERS)


def template_artifact_flag(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in TEMPLATE_ARTIFACT_MARKERS)


def scenario_restatement_flag(text: str, scenario: str) -> bool:
    lowered = text.lower()
    markers = SCENARIO_RESTATEMENT_MARKERS.get(scenario, ())
    hits = sum(1 for marker in markers if marker in lowered)
    return hits >= 2


def history_corruption_flag(history: Any) -> bool:
    if not history or not isinstance(history, list):
        return False
    combined = " ".join(str(item) for item in history)
    return repetition_score(combined, ngram_size=8) > 0.20


def parse_jsonl_lines(source_file: str, base_url: str) -> Iterable[dict[str, Any]]:
    local_path = Path(source_file)
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    source = source_file
    if not source.startswith(("http://", "https://")):
        source = f"{base_url.rstrip('/')}/{source_file}"

    with urllib.request.urlopen(source) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            yield json.loads(line)


def qc_status_for_row(
    prompt: str,
    response: str,
    prompt_overlap: float,
    repetition: float,
    truncation: bool,
    role_confusion: bool,
    prompt_leak_marker: bool,
    template_artifact: bool,
    scenario_restatement: bool,
    history_corruption: bool,
    label_unclear: bool,
) -> tuple[str, list[str]]:
    flags: list[str] = []

    if not response:
        flags.append("empty_response")
    if prompt_overlap >= 0.75:
        flags.append("prompt_overlap_high")
    elif prompt_overlap >= 0.50:
        flags.append("prompt_overlap_medium")

    if repetition >= 0.35:
        flags.append("repetition_high")
    elif repetition >= 0.18:
        flags.append("repetition_medium")

    if truncation:
        flags.append("truncation")
    if role_confusion:
        flags.append("role_confusion")
    if prompt_leak_marker:
        flags.append("prompt_leak_marker")
    if template_artifact:
        flags.append("template_artifact")
    if scenario_restatement:
        flags.append("scenario_restatement")
    if history_corruption:
        flags.append("history_corruption")
    if label_unclear:
        flags.append("label_alignment_unclear")
    if len(response) < 20:
        flags.append("very_short_response")

    drop_flags = {
        "empty_response",
        "prompt_overlap_high",
        "repetition_high",
        "role_confusion",
        "prompt_leak_marker",
        "template_artifact",
        "scenario_restatement",
    }
    review_flags = {
        "prompt_overlap_medium",
        "repetition_medium",
        "truncation",
        "history_corruption",
        "label_alignment_unclear",
        "very_short_response",
    }

    if any(flag in drop_flags for flag in flags):
        return "drop", flags
    if any(flag in review_flags for flag in flags):
        return "review", flags
    return "keep", flags


def infer_usable_flags(row: dict[str, Any], qc_status: str) -> tuple[bool, bool]:
    if qc_status == "drop":
        return False, False

    round_num = row.get("round_num")
    agent_name = row.get("agent_name")
    usable_task1 = agent_name == "Negotiator" and isinstance(round_num, int) and round_num >= 0
    usable_task2 = usable_task1 and bool(row.get("dialogue_history") or row.get("prompt"))
    return usable_task1, usable_task2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="benchmark_data/hf_manifest.csv",
        help="Manifest CSV created by build_hf_manifest.py",
    )
    parser.add_argument(
        "--output",
        default="benchmark_data/transcript_qc.csv",
        help="Path to the QC output CSV",
    )
    parser.add_argument(
        "--base-url",
        default="https://huggingface.co/datasets/sycorpia/multiagent-lab-data/resolve/main",
        help="Base URL used to resolve manifest paths",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on transcript files processed. 0 means no cap.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=0,
        help="Optional cap on rows processed per file. 0 means no cap.",
    )
    parser.add_argument(
        "--priority",
        nargs="*",
        default=["high", "medium"],
        help="Manifest candidate_priority values to include",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))

    transcript_rows = [
        row
        for row in manifest_rows
        if row.get("is_transcript") == "true" and row.get("candidate_priority") in set(args.priority)
    ]
    if args.max_files:
        transcript_rows = transcript_rows[: args.max_files]

    fieldnames = [
        "source_file",
        "row_index",
        "model_name",
        "scenario",
        "trial_id",
        "round_num",
        "agent_name",
        "incentive_condition",
        "actual_deception",
        "experiment_mode",
        "prompt_length",
        "response_length",
        "prompt_overlap_score",
        "repetition_score",
        "truncation_flag",
        "role_confusion_flag",
        "history_corruption_flag",
        "label_alignment_unclear",
        "qc_status",
        "qc_flags",
        "usable_for_task1",
        "usable_for_task2",
        "qc_notes",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for manifest_row in transcript_rows:
            source_file = manifest_row["path"]
            row_count = 0
            for idx, row in enumerate(parse_jsonl_lines(source_file, args.base_url)):
                if args.max_rows_per_file and row_count >= args.max_rows_per_file:
                    break

                prompt = str(row.get("prompt", "") or "")
                response = str(row.get("response", "") or "")
                overlap = token_overlap_score(prompt, response)
                repeat = repetition_score(response)
                trunc = truncation_flag(response)
                role_conf = role_confusion_flag(response)
                prompt_leak = prompt_leak_marker_flag(response)
                template_art = template_artifact_flag(response)
                scenario_restate = scenario_restatement_flag(response, str(row.get("scenario", "")))
                history_conf = history_corruption_flag(row.get("dialogue_history"))
                label_unclear = row.get("trial_id") is None or row.get("round_num") is None
                qc_status, flags = qc_status_for_row(
                    prompt,
                    response,
                    overlap,
                    repeat,
                    trunc,
                    role_conf,
                    prompt_leak,
                    template_art,
                    scenario_restate,
                    history_conf,
                    label_unclear,
                )
                usable_task1, usable_task2 = infer_usable_flags(row, qc_status)

                writer.writerow(
                    {
                        "source_file": source_file,
                        "row_index": idx,
                        "model_name": manifest_row.get("model_name", ""),
                        "scenario": row.get("scenario", manifest_row.get("scenario", "")),
                        "trial_id": row.get("trial_id", ""),
                        "round_num": row.get("round_num", ""),
                        "agent_name": row.get("agent_name", ""),
                        "incentive_condition": row.get("incentive_condition", ""),
                        "actual_deception": row.get("actual_deception", ""),
                        "experiment_mode": row.get("experiment_mode", ""),
                        "prompt_length": len(prompt),
                        "response_length": len(response),
                        "prompt_overlap_score": f"{overlap:.4f}",
                        "repetition_score": f"{repeat:.4f}",
                        "truncation_flag": str(trunc).lower(),
                        "role_confusion_flag": str(role_conf).lower(),
                        "history_corruption_flag": str(history_conf).lower(),
                        "label_alignment_unclear": str(label_unclear).lower(),
                        "qc_status": qc_status,
                        "qc_flags": "|".join(flags),
                        "usable_for_task1": str(usable_task1).lower(),
                        "usable_for_task2": str(usable_task2).lower(),
                        "qc_notes": "",
                    }
                )
                row_count += 1

    print(f"Wrote transcript QC rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
