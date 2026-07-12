#!/usr/bin/env python3
"""Download a selected HF transcript tranche and emit a local manifest."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        default="benchmark_data/tranche_round1.csv",
        help="CSV created by select_hf_transcript_tranche.py",
    )
    parser.add_argument(
        "--local-manifest",
        default="benchmark_data/tranche_round1_local.csv",
        help="Output CSV with local file paths in the path column",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of files to download. 0 means no cap.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files already present at the destination path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with open(args.selection, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if args.limit:
        rows = rows[: args.limit]

    local_rows: list[dict[str, str]] = []
    for row in rows:
        source_url = row["source_url"]
        local_path = Path(row["local_path"])
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if not (args.skip_existing and local_path.exists()):
            subprocess.run(
                ["curl", "-L", "-s", source_url, "-o", str(local_path)],
                check=True,
            )

        local_row = dict(row)
        local_row["path"] = str(local_path)
        local_rows.append(local_row)

    output_path = Path(args.local_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(local_rows[0].keys()) if local_rows else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(local_rows)

    print(f"Wrote local manifest to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
