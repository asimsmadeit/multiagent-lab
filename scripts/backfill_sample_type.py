#!/usr/bin/env python3
"""Backfill ``sample_type`` in safe datasets or reviewed legacy ``.pt`` files.

Created 2026-04-21 as part of the data-quality remediation work. The
`sample_type` field was added to `ActivationSample` on 2026-04-21 to
distinguish real negotiation turns from verification and plausibility probes.
Existing .pt files on Hugging Face and on disk pre-date the field and need
it backfilled from the `round_num` convention:

    round_num == -1  -> sample_type = "pre_verification"
    round_num == -2  -> sample_type = "post_plausibility"
    round_num >= 0   -> sample_type = "negotiation"

Usage:
    python scripts/backfill_sample_type.py path/to/activations.json
    python scripts/backfill_sample_type.py --in-place path/to/activations.json
    python scripts/backfill_sample_type.py --dir experiment_results/

By default writes a ``.backfilled.json``/``.npz`` bundle alongside a safe
input. Loading or writing pickle-capable ``.pt`` files requires the explicit
``--trust-legacy-pt`` acknowledgement.

This script does NOT upload to Hugging Face. To publish:
    huggingface-cli upload sycorpia/multiagent-lab-data \\
        <local_path> <remote_path> --repo-type dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

from interpretability.data import save_activation_dataset
from interpretability.script_artifacts import (
    add_legacy_trust_argument,
    load_activation_input,
    prefer_safe_activation_path,
)


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


def backfill(
    path: Path,
    in_place: bool = False,
    dry_run: bool = False,
    *,
    trust_legacy_pt: bool = False,
) -> Dict[str, int]:
    """Backfill one activation dataset and return counts by sample type."""
    source = prefer_safe_activation_path(path)
    print(f"Loading: {source}")
    data = load_activation_input(
        source,
        trust_legacy_pt=trust_legacy_pt,
    )

    metadata = data.get('metadata')
    if metadata is None:
        # Some legacy formats embedded per-sample info under 'labels' instead.
        # If there is no per-sample metadata at all, we cannot backfill.
        print(f"  SKIP: no 'metadata' key in this file")
        return {'skipped': 1}

    round_nums = None
    labels_dict = data.get('labels', {})
    if isinstance(labels_dict, dict):
        round_nums = labels_dict.get('round_nums', labels_dict.get('round_num'))

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

    # Preserve the source format, while keeping legacy writes behind the same
    # explicit trust acknowledgement as legacy reads.
    out_path = source if in_place else source.with_name(
        source.stem + f'.backfilled{source.suffix}'
    )
    print(f"  Writing: {out_path}")
    save_activation_dataset(
        out_path,
        data,
        trusted_legacy=trust_legacy_pt,
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='*', help='activation dataset(s) to backfill')
    parser.add_argument('--dir', help='Process activation datasets under this directory')
    parser.add_argument('--in-place', action='store_true',
                        help='Overwrite the input file instead of writing .backfilled.pt')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print counts but do not write output')
    add_legacy_trust_argument(parser)
    args = parser.parse_args(argv)

    targets = [Path(p) for p in args.path]
    if args.dir:
        directory = Path(args.dir)
        discovered = set(directory.rglob('*activation*.json'))
        discovered.update(
            prefer_safe_activation_path(path)
            for path in directory.rglob('*.pt')
        )
        targets.extend(
            path for path in sorted(discovered)
            if '.backfilled.' not in path.name
        )

    targets = list(dict.fromkeys(
        prefer_safe_activation_path(path) for path in targets
    ))

    if not targets:
        parser.print_help()
        return 1

    totals: Dict[str, int] = {}
    failures = 0
    for path in targets:
        if not path.exists():
            print(f"SKIP: {path} does not exist")
            continue
        try:
            counts = backfill(
                path,
                in_place=args.in_place,
                dry_run=args.dry_run,
                trust_legacy_pt=args.trust_legacy_pt,
            )
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
        except Exception as e:
            print(f"ERROR on {path}: {type(e).__name__}: {e}")
            failures += 1

    print("\n=== Totals across all files ===")
    for k, v in sorted(totals.items()):
        print(f"  {k}: {v}")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
