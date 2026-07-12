#!/usr/bin/env python3
"""Local smoke test for ESR-Bench task prompts and grading."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from esr_bench.data import load_task1_items, load_task2_items


def main() -> int:
    task1_items = load_task1_items()
    task2_items = load_task2_items()

    print(f"Loaded {len(task1_items)} Task 1 items and {len(task2_items)} Task 2 items.")
    print("Prompt sample:")
    print(task1_items[0].item_id)
    print(task1_items[0].question)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
