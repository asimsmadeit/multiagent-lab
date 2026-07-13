#!/usr/bin/env python3
"""Discover, merge, summarize, and analyze safe activation datasets."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np

from interpretability.data import load_activation_dataset, save_activation_dataset


def _is_activation_manifest(path: str | Path) -> bool:
    candidate = Path(path)
    if candidate.suffix != ".json" or not candidate.is_file():
        return False
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    manifest = payload.get("manifest")
    return isinstance(manifest, dict) and isinstance(
        manifest.get("activation_dataset_schema_version"), str
    )


def find_data_files(paths: Sequence[str]) -> list[str]:
    """Resolve safe manifests and explicitly named legacy .pt artifacts."""
    candidates: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            if path.suffix == ".pt" or _is_activation_manifest(path):
                candidates.append(str(path))
        elif path.is_dir():
            candidates.extend(
                str(item)
                for item in path.rglob("*.json")
                if _is_activation_manifest(item)
            )
            candidates.extend(str(item) for item in path.rglob("*.pt"))
        elif glob.has_magic(raw_path):
            for match in glob.glob(raw_path, recursive=True):
                item = Path(match)
                if item.suffix == ".pt" or _is_activation_manifest(item):
                    candidates.append(str(item))
    data_files = sorted(set(candidates))
    final_files = [path for path in data_files if "checkpoint" not in path.lower()]
    if final_files:
        data_files = final_files
    return data_files


def load_and_merge(
    data_files: Sequence[str],
    verbose: bool = True,
    *,
    trusted_legacy: bool = False,
) -> dict[str, Any]:
    """Load one dataset or safely merge several datasets."""
    if not data_files:
        raise ValueError("at least one activation dataset is required")
    if len(data_files) == 1:
        if verbose:
            print(f"Loading: {data_files[0]}")
        return load_activation_dataset(
            data_files[0],
            trusted_legacy=trusted_legacy,
        )
    if verbose:
        print(f"Merging {len(data_files)} data files")
    from interpretability.merge_parallel_results import merge_parallel_activations

    with tempfile.TemporaryDirectory() as temporary_directory:
        merged_path = merge_parallel_activations(
            list(data_files),
            output_dir=temporary_directory,
            timestamp="analysis",
            verbose=verbose,
            trusted_legacy=trusted_legacy,
        )
        return load_activation_dataset(merged_path)


def run_analysis(
    data: Mapping[str, Any],
    output_dir: str,
    timestamp: str,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the primary grouped probe analysis through a safe temporary bundle."""
    from interpretability.probes.train_probes import run_full_analysis

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base = destination / f"_temp_merged_{timestamp}"
    saved = save_activation_dataset(base.with_suffix(".json"), data)
    manifest_path = saved[1]
    try:
        if verbose:
            print("RUNNING PROBE ANALYSIS")
        results = run_full_analysis(str(manifest_path), trusted_legacy=False)
        results_path = destination / f"probe_results_{timestamp}.json"
        results_path.write_text(
            json.dumps(results, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        if verbose:
            print(f"Results saved to: {results_path}")
        return results
    finally:
        saved[0].unlink(missing_ok=True)
        saved[1].unlink(missing_ok=True)


def print_summary(
    data: Mapping[str, Any],
    results: Mapping[str, Any] | None = None,
) -> None:
    """Print concise dataset and available-analysis summaries."""
    config = data.get("config", {})
    labels = data.get("labels", {})
    print("DATA SUMMARY")
    print(f"Total samples: {config.get('n_samples', len(labels.get('gm_labels', [])))}")
    print(f"Layers: {config.get('layers', list(data.get('activations', {})))}")
    gm_labels = [
        float(value)
        for value in labels.get("gm_labels", [])
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
    ]
    if gm_labels:
        print(f"Deception rate: {np.mean(np.asarray(gm_labels) > 0.5):.1%}")
    for label_name, heading in (
        ("mode_labels", "Mode distribution"),
        ("scenario", "Scenarios"),
    ):
        values = labels.get(label_name, [])
        if values:
            counts: dict[Any, int] = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            print(f"{heading}: {counts}")
    if results and results.get("best_probe"):
        best_probe = results["best_probe"]
        print(
            f"Best probe - Layer {best_probe.get('layer')}: "
            f"R²={best_probe.get('r2', 0):.3f}, "
            f"AUC={best_probe.get('auc', 0):.3f}"
        )
    if results and results.get("gm_vs_agent"):
        comparison = results["gm_vs_agent"]
        if comparison.get("available", True):
            print(
                "GM vs Agent - "
                f"GM R²={comparison['gm_ridge_r2']:.3f}, "
                f"Agent R²={comparison['agent_ridge_r2']:.3f}"
            )
        else:
            print(f"GM vs Agent unavailable: {comparison.get('reason', 'unknown')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze safe activation datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "data_paths",
        nargs="+",
        help="Safe .json manifests, reviewed legacy .pt files, directories, or globs",
    )
    parser.add_argument("--output", "-o", default="./analysis_output")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument(
        "--trust-legacy-pt",
        action="store_true",
        help="Allow pickle-capable .pt loading only for reviewed inputs",
    )
    args = parser.parse_args()
    data_files = find_data_files(args.data_paths)
    if not data_files:
        print("Error: no activation datasets found")
        sys.exit(1)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data = load_and_merge(
        data_files,
        verbose=not args.quiet,
        trusted_legacy=args.trust_legacy_pt,
    )
    if args.merge_only:
        saved = save_activation_dataset(
            output_dir / f"merged_data_{timestamp}.json",
            data,
        )
        print(f"Merged data saved to: {saved[1]}")
        print_summary(data)
        return
    if args.summary_only:
        print_summary(data)
        return
    results = run_analysis(data, str(output_dir), timestamp, verbose=not args.quiet)
    print_summary(data, results)
    print(f"Analysis complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
