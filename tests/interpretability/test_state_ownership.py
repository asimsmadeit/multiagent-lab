"""Regressions for process-global state and mutable registry ownership."""

from __future__ import annotations

import os
import random
import subprocess
import sys

import interpretability.llm_evaluation as llm_evaluation_module
from interpretability.core.deepeval_detector import load_deepeval_environment
from interpretability.llm_evaluation import (
    LLMAgentConfig,
    LLMAgentRunner,
    create_mock_model,
)
from interpretability.scenarios.deception_scenarios import (
    Condition,
    get_scenario_config,
)


def test_scenario_config_nested_mutation_cannot_change_registry():
    first = get_scenario_config("hidden_value")
    first["conditions"][Condition.DECEPTIVE]["expected_deception"] = False
    first["value_ranges"]["true_value"] = (-1, -1)

    second = get_scenario_config("hidden_value")

    assert second["conditions"][Condition.DECEPTIVE]["expected_deception"] is True
    assert second["value_ranges"]["true_value"] == (20, 60)
    assert first is not second
    assert first["conditions"] is not second["conditions"]


def test_deepeval_module_import_does_not_load_dotenv(tmp_path):
    variable = "DEEPEVAL_IMPORT_SIDE_EFFECT_SENTINEL"
    (tmp_path / ".env").write_text(f"{variable}=mutated\n", encoding="utf-8")
    environment = os.environ.copy()
    environment.pop(variable, None)
    command = (
        "import os; "
        "import interpretability.core.deepeval_detector; "
        f"assert {variable!r} not in os.environ"
    )

    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_deepeval_dotenv_loading_is_explicitly_available(tmp_path, monkeypatch):
    variable = "DEEPEVAL_EXPLICIT_LOAD_SENTINEL"
    dotenv_path = tmp_path / "detector.env"
    dotenv_path.write_text(f"{variable}=loaded\n", encoding="utf-8")
    monkeypatch.delenv(variable, raising=False)

    assert load_deepeval_environment(str(dotenv_path)) is True
    assert os.environ[variable] == "loaded"


def test_mock_model_never_changes_module_global_random_state():
    random.seed(9182)
    initial_state = random.getstate()
    model = create_mock_model()

    model.sample_text("How many boats will you deploy?")
    model.sample_text("What strategy will you use?")

    assert random.getstate() == initial_state


def test_runner_config_seed_replays_mock_responses_without_global_mutation(tmp_path):
    config = LLMAgentConfig(
        name="seed-contract",
        scenario_type="fishery",
        num_trials=0,
        random_seed=42,
    )
    random.seed(2026)
    initial_state = random.getstate()
    first = LLMAgentRunner(output_dir=str(tmp_path / "first"))
    second = LLMAgentRunner(output_dir=str(tmp_path / "second"))

    first.run_experiment(config, verbose=False)
    second.run_experiment(config, verbose=False)
    first_outputs = [
        first.model.sample_text("How many boats will you deploy?")
        for _ in range(5)
    ]
    second_outputs = [
        second.model.sample_text("How many boats will you deploy?")
        for _ in range(5)
    ]

    assert first_outputs == second_outputs
    assert random.getstate() == initial_state


def test_llm_runner_passes_explicit_trial_seed_to_advanced_builder(
    tmp_path,
    monkeypatch,
):
    captured = {}
    monkeypatch.setattr(
        llm_evaluation_module.advanced_negotiator,
        'build_agent',
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    runner = LLMAgentRunner(output_dir=str(tmp_path / 'seed-provenance'))

    runner._create_agent(
        name='Alice',
        goal='Reach agreement',
        modules=['uncertainty_aware'],
        module_configs={},
        trial_seed=47,
    )

    assert captured['trial_seed'] == 47
    assert captured['name'] == 'Alice'
