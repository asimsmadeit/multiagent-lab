#!/usr/bin/env python3
"""Merge multiple QC CSV files and deduplicate by source row."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="QC CSV files to merge",
    )
    parser.add_argument(
        "--output",
        default="benchmark_data/qc_merged.csv",
        help="Output merged QC CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merged: dict[tuple[str, str], dict[str, str]] = {}
    fieldnames: list[str] | None = None

    for path in args.inputs:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                key = (row["source_file"], row["row_index"])
                merged[key] = row

    if fieldnames is None:
        raise SystemExit("No input rows found")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for _, row in sorted(merged.items()):
            writer.writerow(row)

    print(f"Wrote {len(merged)} merged QC rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
