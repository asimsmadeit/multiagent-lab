#!/usr/bin/env python3
"""
Aggregate Results from Parallel Cluster Runs

Combines activation files and results from multiple scenario runs
into a single unified dataset for probe training.

Usage:
    python scripts/aggregate_results.py --input-dir ./cluster_results/20250128_123456
    python scripts/aggregate_results.py --input-dir ./cluster_results/20250128_123456 --train-probes
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

import torch
import numpy as np


def find_activation_files(input_dir: Path) -> List[Path]:
    """Find all activation .pt files in subdirectories."""
    files = []
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            for pt_file in subdir.glob("*.pt"):
                files.append(pt_file)
            for pt_file in subdir.glob("**/*.pt"):
                files.append(pt_file)
    return list(set(files))


def load_activation_file(path: Path) -> Dict[str, Any]:
    """Load a single activation file."""
    data = torch.load(path, map_location="cpu")
    return data


def merge_activations(files: List[Path]) -> Dict[str, Any]:
    """Merge multiple activation files into one."""
    print(f"\nMerging {len(files)} activation files...")

    merged = {
        "activations": {},
        "labels": {
            "gm_labels": [],
            "agent_labels": [],
        },
        "metadata": [],
        "scenarios": [],
    }

    total_samples = 0

    for path in files:
        print(f"  Loading {path.name}...")
        try:
            data = load_activation_file(path)
        except Exception as e:
            print(f"    Error loading {path}: {e}")
            continue

        # Get number of samples in this file
        n_samples = len(data.get("metadata", []))
        if n_samples == 0:
            # Try to infer from activations
            acts = data.get("activations", {})
            if acts:
                first_layer = list(acts.keys())[0]
                n_samples = len(acts[first_layer])

        print(f"    Found {n_samples} samples")

        # Merge activations
        for layer, acts in data.get("activations", {}).items():
            if layer not in merged["activations"]:
                merged["activations"][layer] = []

            if isinstance(acts, torch.Tensor):
                acts = acts.numpy()
            merged["activations"][layer].append(acts)

        # Merge labels
        labels = data.get("labels", {})
        if "gm_labels" in labels:
            gm = labels["gm_labels"]
            if isinstance(gm, torch.Tensor):
                gm = gm.numpy()
            merged["labels"]["gm_labels"].extend(gm.tolist() if hasattr(gm, 'tolist') else list(gm))

        if "agent_labels" in labels:
            agent = labels["agent_labels"]
            if isinstance(agent, torch.Tensor):
                agent = agent.numpy()
            merged["labels"]["agent_labels"].extend(agent.tolist() if hasattr(agent, 'tolist') else list(agent))

        # Merge metadata
        merged["metadata"].extend(data.get("metadata", []))

        # Track scenarios
        scenario = path.parent.name.split("_")[0]  # Extract scenario from dir name
        merged["scenarios"].extend([scenario] * n_samples)

        total_samples += n_samples

    # Convert lists to arrays
    for layer in merged["activations"]:
        merged["activations"][layer] = np.concatenate(merged["activations"][layer], axis=0)

    merged["labels"]["gm_labels"] = np.array(merged["labels"]["gm_labels"])
    merged["labels"]["agent_labels"] = np.array(merged["labels"]["agent_labels"])

    print(f"\nMerged total: {total_samples} samples")
    print(f"Layers: {list(merged['activations'].keys())}")
    print(f"Scenarios: {list(set(merged['scenarios']))}")

    return merged


def save_merged(merged: Dict[str, Any], output_path: Path) -> None:
    """Save merged activations."""
    # Convert numpy arrays to torch tensors for saving
    save_data = {
        "activations": {
            layer: torch.from_numpy(acts)
            for layer, acts in merged["activations"].items()
        },
        "labels": {
            "gm_labels": torch.from_numpy(merged["labels"]["gm_labels"]),
            "agent_labels": torch.from_numpy(merged["labels"]["agent_labels"]),
        },
        "metadata": merged["metadata"],
        "scenarios": merged["scenarios"],
    }

    torch.save(save_data, output_path)
    print(f"\nSaved merged activations to {output_path}")


def print_summary(merged: Dict[str, Any]) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    n_samples = len(merged["labels"]["gm_labels"])
    print(f"Total samples: {n_samples}")

    # Scenario breakdown
    from collections import Counter
    scenario_counts = Counter(merged["scenarios"])
    print("\nSamples per scenario:")
    for scenario, count in sorted(scenario_counts.items()):
        print(f"  {scenario}: {count}")

    # Label distribution
    gm = merged["labels"]["gm_labels"]
    print(f"\nGM labels:")
    print(f"  Mean: {gm.mean():.3f}")
    print(f"  Deceptive (>0.5): {(gm > 0.5).sum()} ({100*(gm > 0.5).mean():.1f}%)")
    print(f"  Honest (<=0.5): {(gm <= 0.5).sum()} ({100*(gm <= 0.5).mean():.1f}%)")

    # Layer info
    print(f"\nActivation layers: {sorted(merged['activations'].keys())}")
    first_layer = list(merged["activations"].keys())[0]
    d_model = merged["activations"][first_layer].shape[1]
    print(f"Model dimension: {d_model}")


def train_probes_on_merged(merged: Dict[str, Any], output_dir: Path) -> None:
    """Train probes on merged data."""
    print("\n" + "=" * 60)
    print("TRAINING PROBES")
    print("=" * 60)

    try:
        from interpretability.probes.train_probes import run_full_analysis
    except ImportError:
        print("Error: Could not import probe training functions")
        print("Make sure you're in the project directory and it's installed")
        return

    results = run_full_analysis(
        activations_by_layer=merged["activations"],
        labels={
            "gm_labels": merged["labels"]["gm_labels"],
            "agent_labels": merged["labels"]["agent_labels"],
        },
        scenarios=merged["scenarios"],
        output_dir=str(output_dir),
    )

    # Save results
    results_path = output_dir / "probe_results.json"
    with open(results_path, "w") as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        json.dump(results, f, indent=2, default=convert)

    print(f"\nProbe results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results from parallel cluster runs",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing scenario subdirectories with .pt files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for merged .pt file (default: input_dir/merged_activations.pt)",
    )
    parser.add_argument(
        "--train-probes",
        action="store_true",
        help="Train probes on merged data",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_dir / "merged_activations.pt"

    # Find activation files
    files = find_activation_files(input_dir)
    if not files:
        print(f"No .pt files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} activation files")

    # Merge
    merged = merge_activations(files)

    # Summary
    print_summary(merged)

    # Save
    save_merged(merged, output_path)

    # Optionally train probes
    if args.train_probes:
        train_probes_on_merged(merged, input_dir)


if __name__ == "__main__":
    main()
