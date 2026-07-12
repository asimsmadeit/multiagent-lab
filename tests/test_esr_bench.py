from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys
import types

import pytest

pytest.importorskip(
    "esr_bench",
    reason="ESR-Bench is an optional benchmark package with separate gold data.",
)

from esr_bench.data import build_task1_dataframe, build_task2_dataframe, load_task1_items, load_task2_items
from esr_bench.kaggle_tasks import build_task1_records, build_task2_records
from esr_bench.tasks import build_task1_prompt, build_task2_prompt, grade_task1_response, grade_task2_response, normalize_choice


def test_load_task1_items():
    items = load_task1_items()
    assert len(items) == 18
    assert Counter(item.scenario for item in items) == {
        "alliance_betrayal": 6,
        "info_withholding": 6,
        "ultimatum_bluff": 6,
    }


def test_load_task2_items():
    items = load_task2_items()
    assert len(items) == 18
    assert Counter(item.scenario for item in items) == {
        "alliance_betrayal": 6,
        "info_withholding": 6,
        "ultimatum_bluff": 6,
    }


def test_build_task1_prompt_contains_options():
    item = load_task1_items()[0]
    prompt = build_task1_prompt(item)
    assert item.question in prompt
    assert "Reply with only one letter" in prompt
    assert "A." in prompt
    assert "B." in prompt
    assert item.item_id not in prompt


def test_build_task2_prompt_contains_options():
    item = load_task2_items()[0]
    prompt = build_task2_prompt(item)
    assert item.question in prompt
    assert "Reply with only one letter" in prompt
    assert "C." in prompt
    assert "D." in prompt


def test_normalize_choice_variants():
    assert normalize_choice("B") == "B"
    assert normalize_choice("option c") == "C"
    assert normalize_choice("Answer: d") == "D"
    assert normalize_choice("I choose A because it is best.") == "A"


def test_normalize_choice_matches_option_text():
    item = load_task1_items()[0]
    assert normalize_choice(item.option_b, item.options) == "B"


def test_grade_task1_response():
    item = load_task1_items()[0]
    graded = grade_task1_response(item.correct_option, item)
    assert graded.parsed_choice == item.correct_option
    assert graded.is_correct is True


def test_grade_task2_response():
    item = load_task2_items()[0]
    graded = grade_task2_response(item.correct_option, item)
    assert graded.parsed_choice == item.correct_option
    assert graded.is_correct is True


def test_build_task1_dataframe():
    df = build_task1_dataframe()
    assert len(df) == 18
    assert {"item_id", "correct_option", "option_a", "option_b", "option_c", "option_d"}.issubset(df.columns)


def test_build_task2_dataframe():
    df = build_task2_dataframe()
    assert len(df) == 18
    assert {"item_id", "belief_tier", "correct_option", "option_a", "option_b", "option_c", "option_d"}.issubset(df.columns)


def test_build_task1_records():
    records = build_task1_records()
    assert len(records) == 18
    assert {"item_id", "prompt", "correct_option", "deception_family", "difficulty_level"}.issubset(records[0].keys())


def test_build_task2_records():
    records = build_task2_records()
    assert len(records) == 18
    assert {"item_id", "prompt", "correct_option", "belief_tier", "correct_belief_target"}.issubset(records[0].keys())


def _load_module(module_name: str, path: str):
    fake_kbench = types.SimpleNamespace(
        task=lambda **kwargs: (lambda fn: fn),
        llms={},
        llm=None,
    )
    sys.modules["kaggle_benchmarks"] = fake_kbench
    spec = importlib.util.spec_from_file_location(module_name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_kaggle_task1_template_loads_locally():
    module = _load_module("esr_task1_template", "examples/kaggle/esr_bench_task1_notebook.py")
    df = module.load_task1_dataframe()
    assert len(df) == 18
    assert "prompt" in df.columns
    assert df.iloc[0]["prompt"].startswith("You are answering a multiple-choice social cognition benchmark.")


def test_kaggle_task2_template_loads_locally():
    module = _load_module("esr_task2_template", "examples/kaggle/esr_bench_task2_notebook.py")
    df = module.load_task2_dataframe()
    assert len(df) == 18
    assert "prompt" in df.columns
    assert df.iloc[0]["prompt"].startswith("You are answering a multiple-choice social cognition benchmark.")
