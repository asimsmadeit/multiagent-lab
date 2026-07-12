#!/usr/bin/env python3
"""Re-run probe analysis on existing activations with the post-audit pipeline.

Created 2026-04-21. Produces a before/after delta table so we can see
exactly which numbers moved as a result of the 2026-04-20 audit fixes
(dyadic dedup, GroupShuffleSplit for outcome prediction, residualizer split
fix, best-layer by AUC) plus the 2026-04-21 sample_type filter.

The "before" column reads the existing probe_results.json already on
sycorpia/multiagent-lab-data for each run. The "after" column re-runs
run_full_analysis locally on the raw activations.pt with today's pipeline.

By default the script downloads files from Hugging Face into the local HF
cache and does not upload anything. CPU-only.

Usage:
    # Process every run in the default list:
    python scripts/rerun_probes_after_audit.py --out audit_qc_delta.csv

    # Process a specific set of runs from a manifest CSV:
    python scripts/rerun_probes_after_audit.py \\
        --manifest runs.csv --out delta.csv

    # Dry-run to see what would be processed:
    python scripts/rerun_probes_after_audit.py --dry-run

Manifest CSV columns (for custom runs):
    model,scenario,run_id,activations_path,probe_results_path
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Default set of runs known to exist on sycorpia/multiagent-lab-data as of
# 2026-04-21. Each tuple is (model, scenario, run_id, activations_path,
# probe_results_path). run_id is a short label for the output CSV.
DEFAULT_RUNS: List[Tuple[str, str, str, str, Optional[str]]] = [
    # Gemma-7B no-ToM (organized layout, cleanest data)
    ('gemma7b', 'ultimatum_bluff', 'no_tom_20260412',
     'gemma7b/ultimatum_bluff/no_tom_20260412_042958/activations.pt',
     None),
    ('gemma7b', 'alliance_betrayal', 'no_tom_20260412',
     'gemma7b/alliance_betrayal/no_tom_20260412_043012/activations.pt',
     None),
    ('gemma7b', 'info_withholding', 'no_tom_20260412',
     'gemma7b/info_withholding/no_tom_20260412_043021/activations.pt',
     None),

    # Gemma-7B with-ToM legacy (transcripts not on HF; filter is structural only)
    ('gemma7b', 'ultimatum_bluff', 'tom_20260211',
     'gemma7b/ultimatum_bluff/tom_20260211/activations.pt',
     None),
    ('gemma7b', 'alliance_betrayal', 'tom_20260210',
     'gemma7b/alliance_betrayal/tom_20260210/activations.pt',
     None),
    ('gemma7b', 'info_withholding', 'tom_20260211',
     'gemma7b/info_withholding/tom_20260211/activations.pt',
     None),

    # Llama-3.1-8B (organized layout, marginal quality)
    ('llama31_8b', 'ultimatum_bluff', 'tom_20260414',
     'llama31_8b/ultimatum_bluff/tom_20260414_192608/activations.pt',
     'llama31_8b/ultimatum_bluff/tom_20260414_192608/probe_results.json'),
    ('llama31_8b', 'alliance_betrayal', 'tom_20260412',
     'llama31_8b/alliance_betrayal/tom_20260412_043955/activations.pt',
     None),
    ('llama31_8b', 'info_withholding', 'tom_20260412',
     'llama31_8b/info_withholding/tom_20260412_051758/activations.pt',
     'llama31_8b/info_withholding/tom_20260412_051758/probe_results.json'),

    # Mistral-7B (flat layout for AB and IW, organized for UB)
    ('mistral7b', 'alliance_betrayal', 'both_20260413',
     'alliance_betrayal/activations_mistral7b_alliance_betrayal_both_20260413_152248.pt',
     'mistral7b/all_scenarios/tom_20260413_161313/probe_results.json'),
    ('mistral7b', 'info_withholding', 'both_20260413',
     'info_withholding/activations_mistral7b_info_withholding_both_20260413_103456.pt',
     'mistral7b/all_scenarios/tom_20260413_161555/probe_results.json'),
    ('mistral7b', 'ultimatum_bluff', 'both_20260412',
     'ultimatum_bluff/activations_mistral7b_ultimatum_bluff_both_20260412_194428.pt',
     'mistral7b/all_scenarios/tom_20260413_161122/probe_results.json'),
]

HF_REPO = 'sycorpia/multiagent-lab-data'


def _download(path: str) -> str:
    """Download a file from the HF dataset repo; return local path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, path, repo_type='dataset')


def _extract_before(probe_results_path: Optional[str]) -> Dict[str, Any]:
    """Load the existing probe_results.json and return the comparison numbers."""
    if probe_results_path is None:
        return {'source': None}
    try:
        local = _download(probe_results_path)
        d = json.load(open(local))
    except Exception as e:
        return {'source': probe_results_path, 'error': f'{type(e).__name__}: {e}'}

    bp = d.get('best_probe') or {}
    dp = d.get('dyadic_pairs') or {}
    op = d.get('outcome_prediction') or {}
    cmt = d.get('cross_mode_transfer') or {}
    return {
        'source': probe_results_path,
        'best_layer': bp.get('layer'),
        'best_r2': bp.get('r2'),
        'best_auc': bp.get('auc'),
        'dyadic_auc': dp.get('pair_probe_auc'),
        'dyadic_r2': dp.get('pair_probe_r2'),
        'dyadic_n_pairs': dp.get('n_pairs'),
        'outcome_early_auc': op.get('early_rounds_auc'),
        'outcome_early_r2': op.get('early_rounds_r2'),
        'forward_transfer_auc': cmt.get('forward_transfer_auc'),
        'reverse_transfer_auc': cmt.get('reverse_transfer_auc'),
    }


def _extract_after(results: Dict[str, Any]) -> Dict[str, Any]:
    """Pull comparison numbers out of a freshly-computed run_full_analysis dict."""
    bp = results.get('best_probe') or {}
    dp = results.get('dyadic_pairs') or {}
    op = results.get('outcome_prediction') or {}
    cmt = results.get('cross_mode_transfer') or {}
    return {
        'best_layer': bp.get('layer'),
        'best_r2': bp.get('r2'),
        'best_auc': bp.get('auc'),
        'dyadic_auc': dp.get('pair_probe_auc'),
        'dyadic_r2': dp.get('pair_probe_r2'),
        'dyadic_n_pairs': dp.get('n_pairs'),
        'outcome_early_auc': op.get('early_rounds_auc'),
        'outcome_early_r2': op.get('early_rounds_r2'),
        'forward_transfer_auc': cmt.get('forward_transfer_auc'),
        'reverse_transfer_auc': cmt.get('reverse_transfer_auc'),
    }


def process_run(
    model: str,
    scenario: str,
    run_id: str,
    activations_path: str,
    probe_results_path: Optional[str],
) -> Dict[str, Any]:
    """Process one run: download, re-run analysis, return one CSV row."""
    print(f"\n=== {model} / {scenario} / {run_id} ===")
    row = {
        'model': model,
        'scenario': scenario,
        'run_id': run_id,
        'activations_path': activations_path,
        'probe_results_path': probe_results_path or '',
    }

    # Before: existing probe_results.json
    before = _extract_before(probe_results_path)
    if 'error' in before:
        print(f"  WARNING: before numbers unavailable ({before['error']})")
    for k, v in before.items():
        row[f'before_{k}'] = v

    # After: download activations, run run_full_analysis locally
    try:
        print(f"  Downloading activations.pt ({activations_path})...")
        local_acts = _download(activations_path)
        print(f"  Running run_full_analysis with post-audit pipeline...")
        from interpretability.probes.train_probes import run_full_analysis
        results = run_full_analysis(local_acts)
        after = _extract_after(results)
        for k, v in after.items():
            row[f'after_{k}'] = v
        # Best-layer delta and AUC delta for at-a-glance comparison
        if before.get('best_auc') is not None and after.get('best_auc') is not None:
            row['auc_delta'] = after['best_auc'] - before['best_auc']
        if before.get('dyadic_auc') is not None and after.get('dyadic_auc') is not None:
            row['dyadic_auc_delta'] = after['dyadic_auc'] - before['dyadic_auc']
    except Exception as e:
        traceback.print_exc()
        row['error'] = f'{type(e).__name__}: {e}'
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='audit_qc_delta.csv',
                        help='Output CSV (default: audit_qc_delta.csv)')
    parser.add_argument('--manifest', help='Optional manifest CSV of runs to process')
    parser.add_argument('--dry-run', action='store_true',
                        help='List what would be processed and exit')
    parser.add_argument('--only-model', action='append',
                        help='Restrict to given model short-name (can be repeated)')
    parser.add_argument('--only-scenario', action='append',
                        help='Restrict to given scenario (can be repeated)')
    args = parser.parse_args()

    # Assemble the run list
    if args.manifest:
        runs = []
        with open(args.manifest) as f:
            for r in csv.DictReader(f):
                runs.append((r['model'], r['scenario'], r['run_id'],
                             r['activations_path'], r.get('probe_results_path') or None))
    else:
        runs = list(DEFAULT_RUNS)

    if args.only_model:
        runs = [r for r in runs if r[0] in args.only_model]
    if args.only_scenario:
        runs = [r for r in runs if r[1] in args.only_scenario]

    if args.dry_run:
        print(f"Would process {len(runs)} runs:")
        for r in runs:
            print(f"  {r[0]} / {r[1]} / {r[2]}  <-  {r[3]}")
        return 0

    if not runs:
        print("No runs match the filter.")
        return 1

    rows: List[Dict[str, Any]] = []
    for run in runs:
        rows.append(process_run(*run))

    # Union of all keys so the header is stable
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    # Print a terse summary
    print("\n=== Summary (model/scenario/run_id: before_auc -> after_auc, delta) ===")
    for r in rows:
        before = r.get('before_best_auc')
        after = r.get('after_best_auc')
        delta = r.get('auc_delta')
        before_s = f"{before:.3f}" if isinstance(before, (int, float)) else '?'
        after_s = f"{after:.3f}" if isinstance(after, (int, float)) else '?'
        delta_s = f"{delta:+.3f}" if isinstance(delta, (int, float)) else '?'
        err = f"  [ERROR: {r['error']}]" if r.get('error') else ''
        print(f"  {r['model']}/{r['scenario']}/{r['run_id']}: "
              f"AUC {before_s} -> {after_s} ({delta_s}){err}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
