#!/usr/bin/env python3
"""
Flexible Data Analysis Script

Analyze activation data from any combination of experiment runs.
Supports merging multiple data files and running all probe analyses.

Usage:
    # Analyze single file
    python analyze_data.py experiment_output/activations_*.pt

    # Analyze multiple files (auto-merges)
    python analyze_data.py gpu1/data.pt gpu2/data.pt gpu3/data.pt

    # Analyze all .pt files in a directory
    python analyze_data.py experiment_output/

    # Specify output directory
    python analyze_data.py data/*.pt --output results/

    # Just merge, don't analyze
    python analyze_data.py data/*.pt --merge-only --output merged/
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np


def find_data_files(paths: List[str]) -> List[str]:
    """Find all .pt data files from given paths (files or directories)."""
    data_files = []

    for path in paths:
        if os.path.isfile(path) and path.endswith('.pt'):
            data_files.append(path)
        elif os.path.isdir(path):
            # Find all .pt files in directory
            pt_files = glob.glob(os.path.join(path, '*.pt'))
            pt_files += glob.glob(os.path.join(path, '**/*.pt'), recursive=True)
            data_files.extend(pt_files)
        elif '*' in path:
            # Glob pattern
            data_files.extend(glob.glob(path))

    # Remove duplicates and sort
    data_files = sorted(set(data_files))

    # Filter out checkpoint files if there are final activation files
    final_files = [f for f in data_files if 'checkpoint' not in f.lower()]
    if final_files:
        # Prefer final files over checkpoints
        checkpoint_files = [f for f in data_files if 'checkpoint' in f.lower()]
        if checkpoint_files:
            print(f"Note: Ignoring {len(checkpoint_files)} checkpoint files (using final activation files)")
        data_files = final_files

    return data_files


def load_and_merge(data_files: List[str], verbose: bool = True) -> Dict[str, Any]:
    """Load and merge multiple data files."""

    if len(data_files) == 1:
        if verbose:
            print(f"Loading: {data_files[0]}")
        return torch.load(data_files[0], weights_only=False)

    # Multiple files - use merge utility
    if verbose:
        print(f"\nMerging {len(data_files)} data files:")
        for f in data_files:
            print(f"  - {f}")

    from interpretability.merge_parallel_results import merge_parallel_activations
    import tempfile

    # Merge to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        merged_path = merge_parallel_activations(
            data_files,
            output_dir=tmpdir,
            timestamp="merged",
            verbose=verbose,
        )
        merged_data = torch.load(merged_path, weights_only=False)

    return merged_data


def run_analysis(data: Dict[str, Any], output_dir: str, timestamp: str, verbose: bool = True) -> Dict[str, Any]:
    """Run full probe analysis on data."""

    from interpretability.probes.train_probes import run_full_analysis

    # Save merged data temporarily for analysis
    temp_path = os.path.join(output_dir, f"_temp_merged_{timestamp}.pt")
    torch.save(data, temp_path)

    try:
        if verbose:
            print(f"\n{'='*60}")
            print("RUNNING PROBE ANALYSIS")
            print(f"{'='*60}")

        results = run_full_analysis(temp_path)

        # Save results
        results_path = os.path.join(output_dir, f"probe_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        if verbose:
            print(f"\nResults saved to: {results_path}")

        return results

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def print_summary(data: Dict[str, Any], results: Dict[str, Any] = None):
    """Print summary of data and results."""

    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")

    config = data.get('config', {})
    labels = data.get('labels', {})

    print(f"Total samples: {config.get('n_samples', len(labels.get('gm_labels', [])))}")
    print(f"Layers: {config.get('layers', list(data.get('activations', {}).keys()))}")

    # Deception rate
    gm_labels = labels.get('gm_labels', [])
    if gm_labels:
        deception_rate = np.mean([l > 0.5 for l in gm_labels])
        print(f"Deception rate: {deception_rate:.1%}")

    # Mode distribution
    mode_labels = labels.get('mode_labels', [])
    if mode_labels:
        modes = {}
        for m in mode_labels:
            modes[m] = modes.get(m, 0) + 1
        print(f"Mode distribution: {modes}")

    # Scenario distribution
    scenarios = labels.get('scenario', [])
    if scenarios:
        scenario_counts = {}
        for s in scenarios:
            scenario_counts[s] = scenario_counts.get(s, 0) + 1
        print(f"Scenarios: {scenario_counts}")

    # Pod info
    merge_info = data.get('merge_info', {})
    if merge_info:
        print(f"Merged from {merge_info.get('n_pods', 1)} sources")

    pod_info = data.get('pod_info', {})
    if pod_info and not merge_info:
        print(f"Pod ID: {pod_info.get('pod_id', 0)}")
        print(f"Trial ID range: {pod_info.get('trial_id_range', 'N/A')}")

    if results:
        print(f"\n{'='*60}")
        print("ANALYSIS RESULTS")
        print(f"{'='*60}")

        if results.get('best_probe'):
            bp = results['best_probe']
            print(f"Best probe - Layer {bp.get('layer')}: R²={bp.get('r2', 0):.3f}, AUC={bp.get('auc', 0):.3f}")

        if results.get('gm_vs_agent'):
            gva = results['gm_vs_agent']
            print(f"GM vs Agent - GM R²={gva.get('gm_ridge_r2', 0):.3f}, Agent R²={gva.get('agent_ridge_r2', 0):.3f}")
            if gva.get('gm_wins'):
                print("  >> GM labels more predictable (implicit encoding detected)")

        if results.get('cross_mode_transfer'):
            cmt = results['cross_mode_transfer']
            print(f"Cross-mode transfer - Forward AUC={cmt.get('forward_transfer_auc', 0):.3f}, Reverse AUC={cmt.get('reverse_transfer_auc', 0):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze activation data from deception detection experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single file
    python analyze_data.py experiment_output/activations_*.pt

    # Analyze multiple files (auto-merges)
    python analyze_data.py gpu1/data.pt gpu2/data.pt gpu3/data.pt

    # Analyze all .pt files in a directory
    python analyze_data.py experiment_output/

    # Just merge, don't analyze
    python analyze_data.py data/*.pt --merge-only --output merged/
        """
    )

    parser.add_argument(
        'data_paths',
        nargs='+',
        help="Data files (.pt) or directories containing them"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./analysis_output',
        help="Output directory for results (default: ./analysis_output)"
    )
    parser.add_argument(
        '--merge-only',
        action='store_true',
        help="Only merge data files, don't run analysis"
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help="Only print data summary, don't run full analysis"
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help="Suppress verbose output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Find all data files
    data_files = find_data_files(args.data_paths)

    if not data_files:
        print("Error: No .pt data files found")
        sys.exit(1)

    if verbose:
        print(f"Found {len(data_files)} data file(s)")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load and optionally merge data
    data = load_and_merge(data_files, verbose=verbose)

    if args.merge_only:
        # Save merged data
        merged_path = output_dir / f"merged_data_{timestamp}.pt"
        torch.save(data, merged_path)
        print(f"\nMerged data saved to: {merged_path}")
        print_summary(data)
        return

    if args.summary_only:
        print_summary(data)
        return

    # Run full analysis
    results = run_analysis(data, str(output_dir), timestamp, verbose=verbose)

    # Print summary
    print_summary(data, results)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Results in: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
