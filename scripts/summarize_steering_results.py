#!/usr/bin/env python3
"""Summarize behavioral steering results across (model, scenario) cells.

Reads JSON files written by `run_behavioral_steering.py` and prints a
table with the metrics that matter for paper integration:

  - passed              strict monotone+control criterion
  - effect_size         max - min deception rate across magnitudes
  - control_ratio       deception_effect / random_direction_effect
  - monotone_violations how many magnitude transitions go the wrong way
  - per-magnitude rates deception sweep, magnitude -> rate

Usage:
    python scripts/summarize_steering_results.py
    python scripts/summarize_steering_results.py --dir experiment_results/behavioral_steering
    python scripts/summarize_steering_results.py --csv summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List


def load_results(directory: str) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as f:
            payload = json.load(f)
        if "result" not in payload:
            continue
        r = payload["result"]
        details = r.get("details", {}) or {}
        rows.append({
            "cell_id": payload.get("cell_id", os.path.basename(path)),
            "model": payload.get("model_name", ""),
            "scenario": payload.get("scenario", ""),
            "layer": payload.get("layer", ""),
            "n_samples": payload.get("n_samples", ""),
            "wall_time_sec": payload.get("wall_time_sec", 0),
            "passed": r.get("passed"),
            "effect_size": r.get("effect_size"),
            "spearman_rho": details.get("spearman_rho"),
            "spearman_p": details.get("spearman_p", r.get("p_value")),
            "control_effect": details.get("control_effect"),
            "control_ratio": details.get("control_ratio"),
            "monotone_violations": details.get("monotone_violations"),
            "scoring_method": details.get("scoring_method", ""),
            "deception_sweep": details.get("deception_sweep", {}),
            "random_sweep": details.get("random_sweep", {}),
            "magnitudes": details.get("magnitudes", []),
            "message": r.get("message", ""),
        })
    return rows


def fmt(v, width=8, fmt_str=".3f"):
    if v is None:
        return "-".ljust(width)
    if isinstance(v, (int, float)):
        return format(v, fmt_str).rjust(width)[:width]
    return str(v)[:width].ljust(width)


def print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No result files found.")
        return

    print(f"\n{'cell':<14} {'pass':<6} {'effect':<8} {'rho':<8} {'p':<7} "
          f"{'ctrl_x':<8} {'n':<5} L  scoring scenario/model")
    print("-" * 110)

    for r in rows:
        cell = r["cell_id"][:14]
        passed = "YES" if r["passed"] else "no"
        eff = fmt(r["effect_size"], 8, "+.3f") if r["effect_size"] is not None else "?"
        rho = fmt(r["spearman_rho"], 8, "+.3f") if r["spearman_rho"] is not None else "?"
        pval = fmt(r["spearman_p"], 7, ".3f") if r["spearman_p"] is not None else "?"
        ctrl = fmt(r["control_ratio"], 8, ".2f") if r["control_ratio"] is not None else "?"
        n = str(r["n_samples"])
        L = str(r["layer"])
        scen = r["scenario"][:18]
        model = r["model"].split("/")[-1][:25]
        scoring = (r.get("scoring_method") or "?")[:8]
        print(f"{cell:<14} {passed:<6} {eff:<8} {rho:<8} {pval:<7} "
              f"{ctrl:<8} {n:<5} {L:<3} {scoring:<8} {scen}/{model}")

    print("\nPer-magnitude deception rates (deception direction):")
    if rows[0].get("magnitudes"):
        mags = rows[0]["magnitudes"]
        header = f"{'cell':<14} " + " ".join(f"{m:+.0f}".rjust(7) for m in mags)
        print(header)
        for r in rows:
            sweep = r.get("deception_sweep", {}) or {}
            cells = []
            for m in mags:
                rate = sweep.get(str(m), sweep.get(m))
                if isinstance(rate, dict):
                    rate = rate.get("rate", rate.get("deception_rate"))
                cells.append(f"{rate:7.2f}" if isinstance(rate, (int, float)) else "    ?  ")
            print(f"{r['cell_id'][:14]:<14} " + " ".join(cells))

    n_total = len(rows)
    n_passed = sum(1 for r in rows if r["passed"])
    avg_effect = sum((r["effect_size"] or 0) for r in rows) / max(n_total, 1)
    avg_ctrl = sum((r["control_ratio"] or 0) for r in rows) / max(n_total, 1)
    total_wall = sum(r.get("wall_time_sec", 0) for r in rows)

    print("\n" + "=" * 60)
    print(f"Cells: {n_total}  passed: {n_passed}  fail: {n_total - n_passed}")
    print(f"Mean effect_size: {avg_effect:+.3f}")
    print(f"Mean control_ratio: {avg_ctrl:.2f}x")
    print(f"Total wall time across cells: {total_wall:.0f}s "
          f"({total_wall/3600:.1f}h)")


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    flat_keys = ["cell_id", "model", "scenario", "layer", "n_samples",
                 "wall_time_sec", "passed", "effect_size",
                 "spearman_rho", "spearman_p",
                 "control_effect", "control_ratio", "monotone_violations",
                 "scoring_method", "message"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in flat_keys})
    print(f"\nWrote CSV: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="experiment_results/behavioral_steering",
                        help="directory of cell JSON files")
    parser.add_argument("--csv", default=None, help="optional CSV output path")
    args = parser.parse_args()

    rows = load_results(args.dir)
    print_table(rows)
    if args.csv:
        write_csv(rows, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
