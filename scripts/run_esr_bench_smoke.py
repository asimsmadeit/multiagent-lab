#!/usr/bin/env python3
"""Local smoke test for ESR-Bench task prompts and grading."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> int:
    try:
        benchmark_data = importlib.import_module("esr_bench.data")
    except ModuleNotFoundError:
        print(
            "ESR-Bench is optional and is not installed. Install the reviewed "
            "benchmark package and gold data before running this smoke test.",
            file=sys.stderr,
        )
        return 2
    task1_items = benchmark_data.load_task1_items()
    task2_items = benchmark_data.load_task2_items()

    print(f"Loaded {len(task1_items)} Task 1 items and {len(task2_items)} Task 2 items.")
    print("Prompt sample:")
    print(task1_items[0].item_id)
    print(task1_items[0].question)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
