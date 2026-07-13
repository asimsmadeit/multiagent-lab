#!/usr/bin/env python3
"""Aggregate parallel activation pods into one validated safe dataset.

The output is a checksummed JSON manifest plus non-executable NPZ arrays.
Legacy ``.pt`` inputs are accepted only with ``--trust-legacy-pt``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from interpretability.data import load_activation_dataset
from interpretability.merge_parallel_results import merge_parallel_activations
from interpretability.script_artifacts import (
    add_legacy_trust_argument,
    prefer_safe_activation_path,
)


def find_activation_files(input_dir: Path) -> list[Path]:
    """Find pod artifacts, preferring safe manifests over sibling ``.pt`` files."""
    files: set[Path] = set()
    for subdir in input_dir.iterdir():
        if not subdir.is_dir():
            continue
        files.update(subdir.rglob("*activation*.json"))
        files.update(
            prefer_safe_activation_path(path)
            for path in subdir.rglob("*.pt")
        )
    return sorted(files)


def _relocate_safe_bundle(manifest_path: Path, output_path: Path) -> Path:
    """Move a freshly generated safe bundle to the requested base name."""
    destination = (
        output_path.with_suffix(".json")
        if output_path.suffix == ""
        else output_path
    )
    if destination.suffix != ".json":
        raise ValueError("merged activation output must use a .json manifest")
    if manifest_path.resolve() == destination.resolve():
        return manifest_path

    outer_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    array_name = outer_manifest.get("array_file")
    if not isinstance(array_name, str) or not array_name:
        raise ValueError("generated activation manifest has no array_file")
    source_array = manifest_path.with_name(array_name)
    destination_array = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_array.replace(destination_array)
    outer_manifest["array_file"] = destination_array.name
    destination.write_text(
        json.dumps(outer_manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path.unlink()
    return destination


def aggregate_activation_files(
    files: list[Path],
    output_path: Path,
    *,
    trust_legacy_pt: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Validate, merge, save, and reload activation pods."""
    generated = Path(merge_parallel_activations(
        [str(path) for path in files],
        output_dir=str(output_path.parent),
        verbose=True,
        trusted_legacy=trust_legacy_pt,
    ))
    saved = _relocate_safe_bundle(generated, output_path)
    return load_activation_dataset(saved), saved


def print_summary(merged: dict[str, Any]) -> None:
    """Print aligned sample, scenario, label, and activation dimensions."""
    labels = merged["labels"]
    gm_labels = np.asarray(labels["gm_labels"], dtype=float)
    scenarios = labels.get("scenario", ["unknown"] * len(gm_labels))

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(gm_labels)}")
    print("\nSamples per scenario:")
    for scenario, count in sorted(Counter(scenarios).items()):
        print(f"  {scenario}: {count}")
    print("\nGM labels:")
    print(f"  Mean: {gm_labels.mean():.3f}")
    deceptive = gm_labels > 0.5
    print(f"  Deceptive (>0.5): {deceptive.sum()} ({100 * deceptive.mean():.1f}%)")
    print(f"  Honest (<=0.5): {(~deceptive).sum()} ({100 * (~deceptive).mean():.1f}%)")

    layers = merged["activations"]
    print(f"\nActivation layers: {sorted(layers)}")
    first_layer = next(iter(layers.values()))
    print(f"Model dimension: {first_layer.shape[1]}")


def train_probes_on_merged(data_path: Path, output_dir: Path) -> None:
    """Run the public probe pipeline against the saved safe dataset."""
    from interpretability.probes.train_probes import run_full_analysis

    results = run_full_analysis(str(data_path))
    results_path = output_dir / "probe_results.json"

    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"cannot serialize {type(value).__name__}")

    results_path.write_text(
        json.dumps(results, indent=2, default=convert) + "\n",
        encoding="utf-8",
    )
    print(f"\nProbe results saved to {results_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate results from parallel cluster runs",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing scenario subdirectories with activation pods",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Safe .json output (default: INPUT_DIR/merged_activations.json)",
    )
    parser.add_argument(
        "--train-probes",
        action="store_true",
        help="Train probes on the merged safe dataset",
    )
    add_legacy_trust_argument(parser)
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        parser.error(f"input directory not found: {input_dir}")
    output_path = (
        Path(args.output)
        if args.output
        else input_dir / "merged_activations.json"
    )
    files = find_activation_files(input_dir)
    if not files:
        parser.error(f"no activation pods found in {input_dir}")

    print(f"Found {len(files)} activation files")
    merged, saved = aggregate_activation_files(
        files,
        output_path,
        trust_legacy_pt=args.trust_legacy_pt,
    )
    print_summary(merged)
    print(f"\nSaved merged activations to {saved}")
    if args.train_probes:
        train_probes_on_merged(saved, input_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
