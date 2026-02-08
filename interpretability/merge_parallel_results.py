#!/usr/bin/env python3
"""
Merge activation files from parallel GPU execution.

This utility combines results from multiple pods that ran the same experiment
in parallel, producing a single merged dataset for probe training.

Usage:
    python merge_parallel_results.py pod1.pt pod2.pt pod3.pt -o merged.pt

    # Or from run_deception_experiment.py:
    python run_deception_experiment.py --merge-pods pod1.pt pod2.pt pod3.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict


def merge_parallel_activations(
    pod_files: List[str],
    output_dir: str = None,
    timestamp: str = None,
    verbose: bool = True,
) -> str:
    """
    Merge activation files from parallel pods.

    Handles:
    1. Concatenate activation tensors per layer
    2. Merge all label arrays maintaining alignment
    3. Remap counterpart_idxs to global indices
    4. Preserve and merge metadata
    5. Validate alignment across all arrays

    Args:
        pod_files: List of paths to pod activation files (.pt)
        output_dir: Directory for output file (default: same as first pod file)
        timestamp: Session timestamp for output filename
        verbose: Print progress information

    Returns:
        Path to merged output file
    """
    if not pod_files:
        raise ValueError("No pod files provided")

    # Sort files to ensure consistent ordering
    pod_files = sorted(pod_files)

    if verbose:
        print(f"\n{'='*60}")
        print("MERGING PARALLEL POD RESULTS")
        print(f"{'='*60}")
        print(f"Input files: {len(pod_files)}")
        for f in pod_files:
            print(f"  - {f}")

    # Initialize containers
    all_activations: Dict[Any, List[torch.Tensor]] = defaultdict(list)
    all_labels: Dict[str, List] = defaultdict(list)
    all_metadata: List[Dict] = []
    all_sae_features: List[torch.Tensor] = []
    all_sae_top_features: List[List] = []

    # Track sample offsets for counterpart index remapping
    sample_offset = 0
    counterpart_remap: Dict[tuple, int] = {}  # (pod_id, local_idx) -> global_idx

    # Collect pod info for validation
    pod_infos = []
    total_samples = 0

    for pod_file in pod_files:
        if verbose:
            print(f"\nLoading: {pod_file}")

        # Load pod data
        data = torch.load(pod_file, weights_only=False)

        # Extract pod info
        pod_info = data.get('pod_info', {
            'pod_id': 0,
            'trial_id_offset': 0,
            'n_samples': len(data['labels']['gm_labels']),
        })
        pod_infos.append(pod_info)

        n_samples = len(data['labels']['gm_labels'])
        pod_id = pod_info.get('pod_id', 0)

        if verbose:
            print(f"  Pod ID: {pod_id}")
            print(f"  Samples: {n_samples}")
            print(f"  Trial offset: {pod_info.get('trial_id_offset', 0)}")

        # Build counterpart remap table for this pod
        # Maps (pod_id, local_sample_idx) -> global_sample_idx
        for local_idx in range(n_samples):
            global_idx = sample_offset + local_idx
            counterpart_remap[(pod_id, local_idx)] = global_idx

        # Merge activations (concatenate along sample dimension)
        for layer, tensor in data['activations'].items():
            all_activations[layer].append(tensor)

        # Merge labels (simple concatenation, order preserved)
        for key, values in data['labels'].items():
            if isinstance(values, list):
                all_labels[key].extend(values)
            elif isinstance(values, np.ndarray):
                all_labels[key].extend(values.tolist())

        # Merge metadata
        if 'metadata' in data:
            all_metadata.extend(data['metadata'])

        # Merge SAE features if present
        if 'sae_features' in data and data['sae_features'] is not None:
            all_sae_features.append(data['sae_features'])
        if 'sae_top_features' in data and data['sae_top_features'] is not None:
            all_sae_top_features.extend(data['sae_top_features'])

        # Update offset for next pod
        sample_offset += n_samples
        total_samples += n_samples

    if verbose:
        print(f"\nTotal samples after merge: {total_samples}")

    # Stack activations per layer
    merged_activations = {}
    for layer, tensors in all_activations.items():
        merged_activations[layer] = torch.cat(tensors, dim=0)
        if verbose:
            print(f"  Layer {layer}: {merged_activations[layer].shape}")

    # Remap counterpart_idxs to global indices
    if 'counterpart_idxs' in all_labels and 'pod_ids' in all_labels:
        pod_ids = all_labels['pod_ids']
        original_idxs = all_labels['counterpart_idxs']

        remapped_idxs = []
        for i, (pod_id, local_idx) in enumerate(zip(pod_ids, original_idxs)):
            if local_idx is None:
                remapped_idxs.append(None)
            else:
                # Look up global index for this (pod_id, local_idx) pair
                global_idx = counterpart_remap.get((pod_id, local_idx))
                if global_idx is not None:
                    remapped_idxs.append(global_idx)
                else:
                    # Counterpart from different pod - mark as cross-pod
                    # This happens when pods run different trials
                    remapped_idxs.append(None)  # Can't link across pods

        all_labels['counterpart_idxs'] = remapped_idxs

        # Count how many cross-pod references were lost
        n_lost = sum(1 for orig, new in zip(original_idxs, remapped_idxs)
                     if orig is not None and new is None)
        if n_lost > 0 and verbose:
            print(f"  Warning: {n_lost} cross-pod counterpart references could not be remapped")

    # Validate alignment
    n_labels = len(all_labels.get('gm_labels', []))
    for key, values in all_labels.items():
        if len(values) != n_labels:
            raise ValueError(f"Label array '{key}' has {len(values)} items, expected {n_labels}")

    for layer, tensor in merged_activations.items():
        if tensor.shape[0] != n_labels:
            raise ValueError(f"Activation layer {layer} has {tensor.shape[0]} samples, expected {n_labels}")

    if verbose:
        print(f"\nValidation passed: all arrays aligned with {n_labels} samples")

    # Build merged dataset
    merged = {
        'activations': merged_activations,
        'labels': dict(all_labels),
        'config': {
            'model': 'merged',
            'layers': list(merged_activations.keys()),
            'n_samples': total_samples,
            'has_sae': len(all_sae_features) > 0,
        },
        'metadata': all_metadata,
        'merge_info': {
            'source_files': [str(f) for f in pod_files],
            'n_pods': len(pod_files),
            'pod_infos': pod_infos,
            'total_samples': total_samples,
            'merged_at': timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"),
        },
    }

    # Add merged SAE features if available
    if all_sae_features:
        try:
            merged['sae_features'] = torch.cat(all_sae_features, dim=0)
            merged['sae_top_features'] = all_sae_top_features
            if verbose:
                print(f"  SAE features merged: {merged['sae_features'].shape}")
        except Exception as e:
            print(f"  Warning: Could not merge SAE features: {e}")

    # Determine output path
    if output_dir is None:
        output_dir = Path(pod_files[0]).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"activations_merged_{ts}.pt"

    # Save merged dataset
    torch.save(merged, output_path)

    if verbose:
        print(f"\nMerged dataset saved to: {output_path}")
        print(f"{'='*60}\n")

    return str(output_path)


def validate_merge(merged_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Validate a merged dataset for consistency.

    Args:
        merged_path: Path to merged .pt file
        verbose: Print validation results

    Returns:
        Dict with validation results
    """
    data = torch.load(merged_path, weights_only=False)

    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {},
    }

    n_samples = data['config']['n_samples']
    results['stats']['n_samples'] = n_samples
    results['stats']['n_pods'] = data.get('merge_info', {}).get('n_pods', 1)
    results['stats']['layers'] = list(data['activations'].keys())

    # Check activation shapes
    for layer, tensor in data['activations'].items():
        if tensor.shape[0] != n_samples:
            results['errors'].append(f"Layer {layer} has {tensor.shape[0]} samples, expected {n_samples}")
            results['valid'] = False

    # Check label array lengths
    for key, values in data['labels'].items():
        if len(values) != n_samples:
            results['errors'].append(f"Label '{key}' has {len(values)} items, expected {n_samples}")
            results['valid'] = False

    # Check for label balance
    gm_labels = data['labels'].get('gm_labels', [])
    if gm_labels:
        deception_rate = np.mean([l > 0.5 for l in gm_labels])
        results['stats']['deception_rate'] = deception_rate
        if deception_rate < 0.05 or deception_rate > 0.95:
            results['warnings'].append(f"Extreme deception rate: {deception_rate:.1%}")

    # Check mode distribution
    mode_labels = data['labels'].get('mode_labels', [])
    if mode_labels:
        mode_counts = {}
        for m in mode_labels:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        results['stats']['mode_distribution'] = mode_counts

    if verbose:
        print(f"\nValidation results for: {merged_path}")
        print(f"  Valid: {results['valid']}")
        print(f"  Samples: {results['stats']['n_samples']}")
        print(f"  Pods merged: {results['stats']['n_pods']}")
        print(f"  Layers: {results['stats']['layers']}")
        if 'deception_rate' in results['stats']:
            print(f"  Deception rate: {results['stats']['deception_rate']:.1%}")
        if results['errors']:
            print(f"  Errors: {results['errors']}")
        if results['warnings']:
            print(f"  Warnings: {results['warnings']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Merge activation files from parallel GPU execution"
    )
    parser.add_argument(
        'pod_files',
        nargs='+',
        help="Paths to pod activation files (.pt)"
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help="Output directory (default: same as first input file)"
    )
    parser.add_argument(
        '--timestamp',
        type=str,
        default=None,
        help="Timestamp for output filename"
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help="Only validate an existing merged file (first arg)"
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help="Suppress progress output"
    )

    args = parser.parse_args()

    if args.validate_only:
        validate_merge(args.pod_files[0], verbose=not args.quiet)
    else:
        output_path = merge_parallel_activations(
            args.pod_files,
            output_dir=args.output_dir,
            timestamp=args.timestamp,
            verbose=not args.quiet,
        )
        print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
