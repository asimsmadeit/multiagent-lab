#!/usr/bin/env python3
"""Backfill sample_type field on legacy activations.pt files.

Created 2026-04-21 as part of the data-quality remediation work. The
`sample_type` field was added to `ActivationSample` on 2026-04-21 to
distinguish real negotiation turns from verification and plausibility probes.
Existing .pt files on Hugging Face and on disk pre-date the field and need
it backfilled from the `round_num` convention:

    round_num == -1  -> sample_type = "pre_verification"
    round_num == -2  -> sample_type = "post_plausibility"
    round_num >= 0   -> sample_type = "negotiation"

Usage:
    python scripts/backfill_sample_type.py path/to/activations.pt
    python scripts/backfill_sample_type.py --in-place path/to/activations.pt
    python scripts/backfill_sample_type.py --dir experiment_results/

By default writes to <input>.backfilled.pt alongside the original so the
source file is preserved. Pass --in-place to overwrite.

This script does NOT upload to Hugging Face. To publish:
    huggingface-cli upload sycorpia/multiagent-lab-data \\
        <local_path> <remote_path> --repo-type dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def _infer_sample_type(round_num: Any) -> str:
    """Map a round_num to a sample_type per the documented convention."""
    if round_num is None:
        return 'negotiation'
    try:
        r = int(round_num)
    except (TypeError, ValueError):
        return 'negotiation'
    if r == -1:
        return 'pre_verification'
    if r == -2:
        return 'post_plausibility'
    return 'negotiation'


def backfill(path: Path, in_place: bool = False, dry_run: bool = False) -> Dict[str, int]:
    """Backfill sample_type on one .pt file. Returns counts by sample_type."""
    print(f"Loading: {path}")
    data = torch.load(path, weights_only=False)

    metadata = data.get('metadata')
    if metadata is None:
        # Some legacy formats embedded per-sample info under 'labels' instead.
        # If there is no per-sample metadata at all, we cannot backfill.
        print(f"  SKIP: no 'metadata' key in this file")
        return {'skipped': 1}

    round_nums = None
    labels_dict = data.get('labels', {})
    if isinstance(labels_dict, dict) and 'round_num' in labels_dict:
        round_nums = labels_dict['round_num']

    counts: Dict[str, int] = {'negotiation': 0, 'pre_verification': 0,
                              'post_plausibility': 0, 'already_set': 0}

    for i, entry in enumerate(metadata):
        if entry.get('sample_type') is not None:
            counts['already_set'] += 1
            counts[entry['sample_type']] = counts.get(entry['sample_type'], 0) + 1
            continue
        # Prefer round_num from metadata; fall back to labels['round_num']
        rn = entry.get('round_num')
        if rn is None and round_nums is not None and i < len(round_nums):
            rn = round_nums[i]
        sample_type = _infer_sample_type(rn)
        entry['sample_type'] = sample_type
        counts[sample_type] += 1

    print(f"  Sample types after backfill:")
    for k, v in sorted(counts.items()):
        print(f"    {k}: {v}")

    if dry_run:
        print(f"  DRY RUN: not writing output")
        return counts

    # Save: either in place or alongside with .backfilled.pt suffix
    out_path = path if in_place else path.with_name(path.stem + '.backfilled.pt')
    print(f"  Writing: {out_path}")
    torch.save(data, out_path)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='*', help='.pt file(s) to backfill')
    parser.add_argument('--dir', help='Process every .pt file under this directory')
    parser.add_argument('--in-place', action='store_true',
                        help='Overwrite the input file instead of writing .backfilled.pt')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print counts but do not write output')
    args = parser.parse_args()

    targets = [Path(p) for p in args.path]
    if args.dir:
        for p in sorted(Path(args.dir).rglob('*.pt')):
            # Skip files we would treat as output of a previous backfill run
            if p.suffixes[-2:] == ['.backfilled', '.pt']:
                continue
            targets.append(p)

    if not targets:
        parser.print_help()
        return 1

    totals: Dict[str, int] = {}
    for path in targets:
        if not path.exists():
            print(f"SKIP: {path} does not exist")
            continue
        try:
            counts = backfill(path, in_place=args.in_place, dry_run=args.dry_run)
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
        except Exception as e:
            print(f"ERROR on {path}: {type(e).__name__}: {e}")

    print("\n=== Totals across all files ===")
    for k, v in sorted(totals.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
