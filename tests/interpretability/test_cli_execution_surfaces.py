"""Public CLI/config routing tests for tracks, protocols, and publication."""

# Pydantic v2 fields are concrete nested models at runtime; Pylint's static
# inference sees their class-level FieldInfo descriptors instead.
# pylint: disable=no-member

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from interpretability.data import load_activation_recovery_checkpoint
from interpretability.evaluation import InterpretabilityRunner
from interpretability.launch import (
    resolve_config_execution_surface,
    resolve_execution_surface,
)
from interpretability.scenarios.compiled import ExecutionProtocol
from interpretability.scenarios.emergent_prompts import IncentiveCondition
from interpretability.tracks import ExperimentTrack


@pytest.mark.parametrize(
    ("track", "fast", "modules", "capture_policy", "capture_count"),
    [
        ("text_only", True, (), "none", 0),
        ("single_agent_white_box", True, (), "one_logical_actor_per_trial", 1),
        ("bilateral_white_box", True, (), "all_participants_per_trial", 2),
        ("theory_of_mind", False, ("theory_of_mind",), "one_logical_actor_per_trial", 1),
        ("adaptive", False, ("strategy_evolution",), "one_logical_actor_per_trial", 1),
    ],
)
def test_execution_surface_resolves_every_track(
    track, fast, modules, capture_policy, capture_count
) -> None:
    plan = resolve_execution_surface(
        experiment_track=track,
        fast=fast,
    )

    assert plan.experiment_track.value == track
    assert plan.agent_modules == modules
    assert plan.per_trial_capture_policy == capture_policy
    assert plan.per_trial_captured_actor_count == capture_count
    assert plan.protocol is ExecutionProtocol.ALTERNATING


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "experiment_track": "bilateral_white_box",
                "protocol": "solo_no_response",
                "counterpart_types": ["absent"],
                "counterbalance": False,
            },
            "bilateral_white_box",
        ),
        (
            {
                "experiment_track": "single_agent_white_box",
                "protocol": "alternating",
                "counterpart_types": ["absent"],
            },
            "requires protocol='solo_no_response'",
        ),
        (
            {
                "experiment_track": "single_agent_white_box",
                "protocol": "solo_no_response",
                "counterpart_types": ["default"],
                "counterbalance": False,
            },
            "requires exactly",
        ),
        (
            {
                "experiment_track": "single_agent_white_box",
                "counterpart_types": ["default", "skeptical"],
                "counterbalance": False,
            },
            "requires exactly one",
        ),
        (
            {"experiment_track": "adaptive", "ultrafast": True},
            "incompatible",
        ),
        (
            {"experiment_track": "text_only", "mode": "both"},
            "emergent mode only",
        ),
        (
            {
                "experiment_track": "single_agent_white_box",
                "protocol": "simultaneous",
                "probe_mode": "on",
            },
            "require protocol=alternating",
        ),
        (
            {
                "experiment_track": "single_agent_white_box",
                "protocol": "simultaneous",
                "mode": "both",
            },
            "instructed execution supports protocol=alternating only",
        ),
        (
            {
                "experiment_track": "single_agent_white_box",
                "mode": "instructed",
                "counterpart_types": ["skeptical"],
            },
            "do not support custom counterpart policies",
        ),
    ],
)
def test_execution_surface_rejects_invalid_designs(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_execution_surface(**kwargs)


@pytest.mark.parametrize(
    ("protocol", "probe_mode", "run_probes"),
    [
        ("alternating", "auto", None),
        ("simultaneous", "auto", None),
        ("alternating", "on", True),
        ("alternating", "off", False),
        ("simultaneous", "off", False),
    ],
)
def test_execution_surface_preserves_typed_probe_intent(
    protocol, probe_mode, run_probes
) -> None:
    plan = resolve_execution_surface(
        experiment_track="single_agent_white_box",
        protocol=protocol,
        probe_mode=probe_mode,
    )

    assert plan.probe_mode == probe_mode
    assert plan.run_probes is run_probes


def test_default_counterbalance_crosses_every_public_assignment() -> None:
    plan = resolve_execution_surface(
        experiment_track="single_agent_white_box",
    )

    assert plan.counterpart_types == (
        "default", "skeptical", "credulous", "informed"
    )
    assert plan.surface_variants == (
        "default",
        "formal-brief",
        "compact-brief",
        "formal-metadata-only",
    )
    assert plan.executions_per_family == 64


def test_instructed_only_reports_its_effective_unparameterized_budget() -> None:
    plan = resolve_execution_surface(
        experiment_track="single_agent_white_box",
        mode="instructed",
    )

    assert plan.execution_design_scope == "legacy_instructed_not_parameterized"
    assert plan.counterpart_types == ("default",)
    assert plan.surface_variants == ("default",)
    assert plan.counterbalance is False
    assert plan.executions_per_family == 1
    assert plan.run_probes is False


def test_reference_config_resolves_to_the_exact_manual_surface() -> None:
    from config.experiment import ExperimentConfig

    config = ExperimentConfig.load_json("config/reference_offline.json")
    from_config = resolve_config_execution_surface(config)
    manual = resolve_execution_surface(
        experiment_track=config.experiment_track,
        protocol=config.scenarios.protocol,
        counterpart_types=config.scenarios.counterpart_policies,
        counterbalance=config.scenarios.counterbalance,
        counterbalance_seed=config.scenarios.counterbalance_seed,
        surface_variants=config.scenarios.surface_variants,
        mode=config.scenarios.mode,
        probe_mode=config.scenarios.probes,
        agent_modules=config.scenarios.agent_modules,
    )

    assert from_config == manual


def test_config_adapter_rejects_unimplemented_multi_seed_launching() -> None:
    from config.experiment import ExperimentConfig

    config = ExperimentConfig(use_multi_seed=True)
    with pytest.raises(ValueError, match="does not yet support use_multi_seed"):
        resolve_config_execution_surface(config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda data: data["model"].update(use_sae=True),
            "use_sae=True",
        ),
        (
            lambda data: data["model"].update(cache_activations=False),
            "cache_activations=True",
        ),
        (
            lambda data: data["model"].update(auto_configure=False),
            "auto_configure=True",
        ),
        (
            lambda data: data.update(save_activations=True),
            "save_activations=True",
        ),
        (
            lambda data: data["causal"].update(enabled=True),
            "causal.enabled=True",
        ),
        (
            lambda data: data.setdefault("probes", {}).update(
                regularization=2.0
            ),
            "regularization",
        ),
        (
            lambda data: data["causal"].update(min_effect_size=0.2),
            "min_effect_size",
        ),
        (
            lambda data: data.update(verbose=False),
            "verbose=True",
        ),
        (
            lambda data: data.update(log_to_file=True),
            "log_to_file=False",
        ),
    ],
)
def test_config_adapter_rejects_every_unwired_public_knob(
    mutate, message
) -> None:
    from config.experiment import ExperimentConfig

    data = json.loads(Path("config/reference_offline.json").read_text())
    mutate(data)
    config = ExperimentConfig.model_validate(data)

    with pytest.raises(ValueError, match=message):
        resolve_config_execution_surface(config)


def test_default_experiment_config_is_a_valid_public_launch_contract() -> None:
    from config.experiment import ExperimentConfig

    config = ExperimentConfig()
    plan = resolve_config_execution_surface(config)

    assert plan.experiment_track is ExperimentTrack.SINGLE_AGENT_WHITE_BOX
    assert config.model.use_sae is False
    assert config.causal.enabled is False
    assert config.save_activations is False
    assert config.log_to_file is False


def test_local_evaluator_uses_configured_checkpoint_and_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    model_calls = []
    tokenizer_calls = []
    fake_model = SimpleNamespace()
    fake_tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda name, **kwargs: (
            model_calls.append((name, kwargs)) or fake_model
        ),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda name: tokenizer_calls.append(name) or fake_tokenizer,
    )
    runner = object.__new__(InterpretabilityRunner)
    runner._device = "cpu"

    evaluator = runner._setup_evaluator(
        "local",
        model_name="offline/evaluator",
        max_tokens=17,
    )

    assert evaluator is not None
    assert evaluator.model_name == "offline/evaluator"
    assert evaluator.default_max_tokens == 17
    assert model_calls[0][0] == "offline/evaluator"
    assert tokenizer_calls == ["offline/evaluator"]

    generation_calls = []
    runner.evaluator_model = SimpleNamespace(
        sample_text=lambda **kwargs: generation_calls.append(kwargs) or "yes"
    )
    runner.evaluator_max_tokens = 17
    runner._extract_structured_data(
        "hidden_value",
        "I can offer $20.",
        {},
    )
    assert generation_calls[0]["max_tokens"] == 17


def test_runner_text_only_selects_hf_adapter_without_white_box_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import interpretability.evaluation as evaluation

    text_model = SimpleNamespace(call_count=0)
    monkeypatch.setattr(
        evaluation,
        "HuggingFaceTextModel",
        lambda **_kwargs: text_model,
    )
    for wrapper_name in ("TransformerLensWrapper", "HybridLanguageModel"):
        monkeypatch.setattr(
            evaluation,
            wrapper_name,
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("text-only must not construct a white-box wrapper")
            ),
        )

    runner = InterpretabilityRunner(
        model_name="offline-model",
        device="cpu",
        experiment_track="text_only",
        captured_actor_ids=(),
        use_hybrid=True,
        use_sae=True,
        evaluator_type="rule",
    )

    assert runner.model is text_model
    assert runner.use_hybrid is False
    assert runner.use_sae is False


def test_text_only_study_does_not_treat_empty_activation_qc_as_failure() -> None:
    runner = object.__new__(InterpretabilityRunner)
    runner.activation_samples = []
    runner.experiment_track = ExperimentTrack.TEXT_ONLY
    runner.captured_actor_ids = ()
    runner._trial_id = 0
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        return {
            "deception_detected": False,
            "samples_collected": 0,
            "trial_family_id": f"family-{len(calls)}",
            "trial_id": f"trial-{len(calls)}",
            "scenario_instance_id": f"instance-{len(calls)}",
            "counterbalance_id": f"assignment-{len(calls)}",
        }

    runner.run_transactional_emergent_trial = execute
    results = runner.run_emergent_study(
        scenario="hidden_value",
        num_trials=3,
        agent_modules=[],
        max_rounds=1,
        conditions=[IncentiveCondition.HIGH_INCENTIVE],
        counterbalance=False,
        run_probes=False,
    )

    assert len(calls) == 3
    assert results["total_executions"] == 3
    assert runner.activation_samples == []


def test_both_cli_helpers_forward_the_complete_execution_design(monkeypatch) -> None:
    import interpretability.cli as click_cli
    import interpretability.run_deception_experiment as argparse_cli

    calls = []

    class Runner:
        def run_all_emergent_scenarios(self, **kwargs):
            calls.append(kwargs)
            return {"ok": True}

    monkeypatch.setattr(click_cli, "IncentiveCondition", IncentiveCondition, raising=False)
    click_cli._run_emergent_experiment(
        Runner(),
        ["hidden_value"],
        1,
        2,
        ["strategy_evolution"],
        False,
        None,
        counterpart_types=("skeptical", "informed"),
        protocol=ExecutionProtocol.SIMULTANEOUS,
        counterbalance=True,
        counterbalance_seed=17,
        surface_variants=("default", "compact-brief"),
        run_probes=False,
    )
    argparse_cli.run_emergent_experiment(
        Runner(),
        ["hidden_value"],
        trials_per_scenario=1,
        conditions=[IncentiveCondition.HIGH_INCENTIVE],
        max_rounds=2,
        agent_modules=["strategy_evolution"],
        counterpart_types=["skeptical", "informed"],
        protocol=ExecutionProtocol.SIMULTANEOUS,
        counterbalance=True,
        counterbalance_seed=17,
        surface_variants=["default", "compact-brief"],
        run_probes=False,
    )

    assert len(calls) == 2
    for call in calls:
        assert tuple(call["counterpart_types"]) == ("skeptical", "informed")
        assert call["protocol"] is ExecutionProtocol.SIMULTANEOUS
        assert call["counterbalance"] is True
        assert call["counterbalance_seed"] == 17
        assert tuple(call["surface_variants"]) == (
            "default", "compact-brief"
        )
        assert call["run_probes"] is False


def test_run_all_forwards_complete_design_to_each_scenario() -> None:
    runner = object.__new__(InterpretabilityRunner)
    calls = []

    def study(**kwargs):
        calls.append(kwargs)
        return {
            "conditions": {
                "high_incentive": {"deception_rate": None},
                "low_incentive": {"deception_rate": None},
            }
        }

    runner.run_emergent_study = study
    runner.run_all_emergent_scenarios(
        scenarios=["hidden_value", "info_withholding"],
        trials_per_scenario=1,
        conditions=[
            IncentiveCondition.HIGH_INCENTIVE,
            IncentiveCondition.LOW_INCENTIVE,
        ],
        agent_modules=["theory_of_mind"],
        counterpart_types=["skeptical", "informed"],
        protocol=ExecutionProtocol.SIMULTANEOUS,
        counterbalance=True,
        counterbalance_seed=23,
        surface_variants=["default", "formal-brief"],
        run_probes=False,
    )

    assert [call["scenario"] for call in calls] == [
        "hidden_value", "info_withholding"
    ]
    for call in calls:
        assert call["counterpart_type"] is None
        assert call["counterpart_types"] == ["skeptical", "informed"]
        assert call["protocol"] is ExecutionProtocol.SIMULTANEOUS
        assert call["counterbalance_seed"] == 23
        assert call["surface_variants"] == ["default", "formal-brief"]
        assert call["run_probes"] is False


@pytest.mark.parametrize("use_config", [False, True])
def test_click_text_only_writes_recovery_and_results_without_activation_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_config: bool,
) -> None:
    import interpretability.cli as cli_module

    created = []
    trained = []

    def runner_factory(**kwargs):
        runner = object.__new__(InterpretabilityRunner)
        runner.model = SimpleNamespace(
            model_name="offline-model",
            tokenizer=SimpleNamespace(name_or_path="offline-tokenizer"),
        )
        runner.activation_samples = []
        runner.generation_records = []
        runner.label_records = []
        runner.interaction_events = []
        runner.experiment_track = kwargs["experiment_track"]
        runner.captured_actor_ids = tuple(kwargs["captured_actor_ids"])
        runner._pod_id = 0
        runner._trial_id_offset = kwargs.get("trial_id_offset", 0)
        runner._trial_id = runner._trial_id_offset
        runner.evaluator_type = kwargs["evaluator_type"]
        runner._deepeval_detector = (
            object() if runner.evaluator_type == "deepeval" else None
        )
        runner.init_kwargs = kwargs
        runner.save_dataset = lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("text-only must not publish an activation dataset")
        )
        created.append(runner)
        return runner

    monkeypatch.setattr(cli_module, "_lazy_import", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli_module, "get_emergent_scenarios", lambda: ["hidden_value"],
        raising=False,
    )
    monkeypatch.setattr(
        cli_module, "get_instructed_scenarios", lambda: ["hidden_value"],
        raising=False,
    )
    monkeypatch.setattr(
        cli_module, "InterpretabilityRunner", runner_factory, raising=False
    )
    monkeypatch.setattr(
        cli_module,
        "_run_emergent_experiment",
        lambda **kwargs: {
            "protocol": kwargs["protocol"].value,
            "total_samples": 0,
        },
    )
    monkeypatch.setattr(
        cli_module,
        "_train_probes_on_data",
        lambda *_a, **_k: trained.append(True),
    )
    monkeypatch.setattr(cli_module, "_print_summary", lambda *_a, **_k: None)

    command = [
        "run",
        "--experiment-track",
        "text_only",
        "--fast",
        "--scenario-name",
        "hidden_value",
        "--trials",
        "1",
        "--output",
        str(tmp_path),
    ]
    if use_config:
        from config.experiment import (
            CausalConfig,
            EvaluatorConfig,
            ExperimentConfig,
            ModelConfig,
            ScenarioConfig,
        )

        config = ExperimentConfig(
            experiment_name="surface-test",
            experiment_track="text_only",
            model=ModelConfig(
                name="google/gemma-2b-it",
                device="cpu",
                dtype="float32",
                use_sae=False,
            ),
            evaluator=EvaluatorConfig(
                enabled=False,
                model="offline-evaluator",
                max_tokens=17,
            ),
            causal=CausalConfig(enabled=False),
            scenarios=ScenarioConfig(
                scenarios=["hidden_value"],
                num_trials=1,
                agent_modules=[],
            ),
            output_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        config_path = tmp_path / "experiment.json"
        config.save_json(str(config_path))
        command = ["run", "--config", str(config_path)]

    result = CliRunner().invoke(cli_module.cli, command)

    assert result.exit_code == 0, result.output
    assert trained == []
    assert created[0].captured_actor_ids == ()
    assert created[0].init_kwargs["trial_id_offset"] == (42 if use_config else 0)
    if use_config:
        assert created[0].init_kwargs["evaluator_model_name"] == "offline-evaluator"
        assert created[0].init_kwargs["evaluator_max_tokens"] == 17
    evidence_path = next(tmp_path.glob("text_only_evidence_*.json"))
    recovery = load_activation_recovery_checkpoint(evidence_path)
    assert recovery["runner_state"]["experiment_track"] == "text_only"
    assert recovery["runner_state"]["captured_actor_ids"] == []
    assert recovery["runner_state"]["trial_id_offset"] == (
        42 if use_config else 0
    )
    assert recovery["activation_samples"] == []
    results = json.loads(next(tmp_path.glob("experiment_results_*.json")).read_text())
    assert results["emergent"]["protocol"] == "alternating"
    manifest = json.loads(
        (tmp_path / "experiment_track_manifest.json").read_text()
    )
    assert manifest["headline_capture_policy"] == "none"
    assert manifest["headline_captured_actor_count_per_trial"] == 0
    assert (manifest["config_source"] is not None) is use_config
    assert manifest["seed_design"]["family_seed_start"] == (
        42 if use_config else 0
    )
    assert manifest["seed_design"]["configured_random_seeds"] == (
        [42, 123, 456, 789, 1337] if use_config else None
    )
    if use_config:
        extractor = manifest["evaluator_design"]["local_structured_extractor"]
        assert extractor["enabled"] is False
        assert extractor["model"] == "offline-evaluator"
        assert extractor["max_tokens"] == 17
        assert manifest["experiment_name"] == "surface-test"
    detector = manifest["evaluator_design"]["ground_truth_detector"]
    assert detector["requested"] == ("rule" if use_config else "deepeval")
    assert detector["effective"] == detector["requested"]
    assert detector["deepeval_available"] is (not use_config)
    assert manifest["activation_publication"]["enabled"] is (not use_config)
    assert manifest["recovery_contract"]["schedule_resume_supported"] is False
    assert manifest["probes"] == {"enabled": True, "mode": "auto"}
    assert manifest["executions_per_family"] == 64
    progress = recovery["experiment_progress"]
    assert progress["executions_per_family"] == 64
    assert progress["estimated_total_physical_executions"] == 128
    assert progress["surface_variants"] == [
        "default",
        "formal-brief",
        "compact-brief",
        "formal-metadata-only",
    ]


def test_click_text_only_rejects_white_box_options_before_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import interpretability.cli as cli_module

    imported = []
    monkeypatch.setattr(cli_module, "_lazy_import", lambda: imported.append(True))
    result = CliRunner().invoke(
        cli_module.cli,
        ["run", "--experiment-track", "text_only", "--fast", "--causal"],
    )

    assert result.exit_code == 2
    assert "text_only cannot use white-box options" in result.output
    assert imported == []


def test_direct_public_clis_reject_sae_without_hybrid_before_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import interpretability.cli as click_cli
    import interpretability.run_deception_experiment as argparse_cli

    imported = []
    monkeypatch.setattr(click_cli, "_lazy_import", lambda: imported.append(True))
    click_result = CliRunner().invoke(click_cli.cli, ["run", "--sae"])
    assert click_result.exit_code == 2
    assert "--sae requires --hybrid" in click_result.output
    assert imported == []

    monkeypatch.setattr(
        argparse_cli,
        "InterpretabilityRunner",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid SAE routing must fail before model construction")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run-deception-experiment", "--sae", "--output", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as error:
        argparse_cli.main()
    assert error.value.code == 2
    assert "--sae requires --hybrid" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_click_verbose_flag_controls_interpretability_diagnostics() -> None:
    import logging
    import interpretability.cli as cli_module

    package_logger = logging.getLogger("interpretability")
    original_level = package_logger.level
    try:
        cli_module._configure_verbosity(False)
        assert package_logger.level == logging.WARNING
        cli_module._configure_verbosity(True)
        assert package_logger.level == logging.DEBUG
    finally:
        package_logger.setLevel(original_level)


def test_public_clis_reject_ignored_legacy_protocol_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import interpretability.cli as click_cli
    import interpretability.run_deception_experiment as argparse_cli

    imported = []
    monkeypatch.setattr(click_cli, "_lazy_import", lambda: imported.append(True))
    click_result = CliRunner().invoke(
        click_cli.cli,
        ["run", "--mode", "both", "--protocol", "simultaneous"],
    )
    assert click_result.exit_code == 2
    assert "supports protocol=alternating only" in click_result.output
    assert imported == []

    monkeypatch.setattr(
        argparse_cli,
        "InterpretabilityRunner",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid protocol must fail before model construction")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-deception-experiment",
            "--mode",
            "both",
            "--protocol",
            "simultaneous",
            "--output",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as error:
        argparse_cli.main()
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("use_config", [False, True])
def test_argparse_text_only_writes_evidence_and_skips_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_config: bool,
) -> None:
    import interpretability.run_deception_experiment as cli_module

    trained = []
    created = []

    def runner_factory(**kwargs):
        runner = object.__new__(InterpretabilityRunner)
        runner.model = SimpleNamespace(
            model_name="offline-model",
            tokenizer=SimpleNamespace(name_or_path="offline-tokenizer"),
        )
        runner.activation_samples = []
        runner.generation_records = []
        runner.label_records = []
        runner.interaction_events = []
        runner.experiment_track = kwargs["experiment_track"]
        runner.captured_actor_ids = tuple(kwargs["captured_actor_ids"])
        runner._pod_id = 0
        runner._trial_id_offset = kwargs.get("trial_id_offset", 0)
        runner._trial_id = runner._trial_id_offset
        runner.evaluator_type = kwargs["evaluator_type"]
        runner._deepeval_detector = (
            object() if runner.evaluator_type == "deepeval" else None
        )
        runner.init_kwargs = kwargs
        runner.save_dataset = lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("text-only must not publish an activation dataset")
        )
        created.append(runner)
        return runner

    monkeypatch.setattr(cli_module, "InterpretabilityRunner", runner_factory)
    monkeypatch.setattr(
        cli_module,
        "run_emergent_experiment",
        lambda **kwargs: {
            "protocol": kwargs["protocol"].value,
            "total_samples": 0,
        },
    )
    monkeypatch.setattr(
        cli_module,
        "train_probes_on_data",
        lambda *_a, **_k: trained.append(True),
    )
    monkeypatch.setattr(cli_module, "print_limitations", lambda **_kwargs: None)
    argv = [
        "run-deception-experiment",
        "--experiment-track",
        "text_only",
        "--fast",
        "--scenario-name",
        "hidden_value",
        "--trials",
        "1",
        "--skip-api-eval",
        "--output",
        str(tmp_path),
    ]
    if use_config:
        from config.experiment import (
            CausalConfig,
            EvaluatorConfig,
            ExperimentConfig,
            ModelConfig,
            ScenarioConfig,
        )

        config = ExperimentConfig(
            experiment_name="surface-test",
            experiment_track="text_only",
            model=ModelConfig(
                name="google/gemma-2b-it",
                device="cpu",
                dtype="float32",
                use_sae=False,
            ),
            evaluator=EvaluatorConfig(
                enabled=False,
                model="offline-evaluator",
                max_tokens=17,
            ),
            causal=CausalConfig(enabled=False),
            scenarios=ScenarioConfig(
                scenarios=["hidden_value"],
                num_trials=1,
                agent_modules=[],
            ),
            output_dir=str(tmp_path),
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        config_path = tmp_path / "experiment.json"
        config.save_json(str(config_path))
        argv = ["run-deception-experiment", "--config", str(config_path)]
    monkeypatch.setattr(sys, "argv", argv)

    cli_module.main()

    assert trained == []
    assert created[0].init_kwargs["trial_id_offset"] == (
        42 if use_config else 0
    )
    if use_config:
        assert created[0].init_kwargs["evaluator_model_name"] == "offline-evaluator"
        assert created[0].init_kwargs["evaluator_max_tokens"] == 17
    evidence_path = next(tmp_path.rglob("text_only_evidence.json"))
    recovery = load_activation_recovery_checkpoint(evidence_path)
    assert recovery["runner_state"]["experiment_track"] == "text_only"
    assert recovery["runner_state"]["trial_id_offset"] == (
        42 if use_config else 0
    )
    assert recovery["activation_samples"] == []
    assert recovery["experiment_progress"]["executions_per_family"] == 64
    assert (
        recovery["experiment_progress"][
            "estimated_total_physical_executions"
        ]
        == 128
    )
    manifest = json.loads(
        next(tmp_path.rglob("experiment_track_manifest.json")).read_text()
    )
    assert manifest["headline_capture_policy"] == "none"
    assert (manifest["config_source"] is not None) is use_config
    assert manifest["seed_design"]["family_seed_start"] == (
        42 if use_config else 0
    )
    assert manifest["seed_design"]["configured_random_seeds"] == (
        [42, 123, 456, 789, 1337] if use_config else None
    )
    if use_config:
        extractor = manifest["evaluator_design"]["local_structured_extractor"]
        assert extractor["enabled"] is False
        assert extractor["model"] == "offline-evaluator"
        assert extractor["max_tokens"] == 17
        assert manifest["experiment_name"] == "surface-test"
    detector = manifest["evaluator_design"]["ground_truth_detector"]
    assert detector["requested"] == "rule"
    assert detector["effective"] == "rule"
    assert detector["deepeval_available"] is False
    assert manifest["activation_publication"]["enabled"] is (not use_config)
    assert manifest["recovery_contract"]["schedule_resume_supported"] is False
    assert manifest["counterpart_policies"] == [
        "default", "skeptical", "credulous", "informed"
    ]
    assert manifest["probes"] == {"enabled": True, "mode": "auto"}
    assert manifest["executions_per_family"] == 64
    assert manifest["estimated_total_physical_executions"] == 128
    results = json.loads(next(tmp_path.rglob("experiment_results.json")).read_text())
    assert results["emergent"]["protocol"] == "alternating"


def test_argparse_rejects_bilateral_solo_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import interpretability.run_deception_experiment as cli_module

    monkeypatch.setattr(
        cli_module,
        "InterpretabilityRunner",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid designs must fail before model construction")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-deception-experiment",
            "--experiment-track",
            "bilateral_white_box",
            "--protocol",
            "solo_no_response",
            "--counterpart-type",
            "absent",
            "--no-counterbalance",
            "--output",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as error:
        cli_module.main()
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_completed_trial_checkpoint_omits_mismatched_runtime_envelope(
    tmp_path: Path,
) -> None:
    runner = object.__new__(InterpretabilityRunner)
    runner.activation_samples = []
    runner.experiment_track = ExperimentTrack.SINGLE_AGENT_WHITE_BOX
    runner.captured_actor_ids = ("Negotiator", "Counterpart")
    runner._pod_id = 0
    writes = []
    runner.run_transactional_emergent_trial = lambda **_kwargs: {
        "deception_detected": None,
        "samples_collected": 0,
        "runtime_checkpoint": {"captured_actor_ids": ["Negotiator"]},
    }
    runner._write_activation_checkpoint = lambda path, **kwargs: writes.append(
        (Path(path), kwargs)
    )

    runner.run_emergent_study(
        scenario="hidden_value",
        num_trials=1,
        agent_modules=[],
        max_rounds=1,
        conditions=[IncentiveCondition.HIGH_INCENTIVE],
        counterbalance=False,
        checkpoint_dir=str(tmp_path),
    )

    assert len(writes) == 1
    assert writes[0][1]["runtime_checkpoint"] is None
    progress = writes[0][1]["experiment_progress"]
    assert progress["completed_family_number"] == 1
    assert progress["completed_execution_number"] == 1


def test_public_cli_help_states_recovery_is_not_schedule_resume(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import interpretability.cli as click_cli
    import interpretability.run_deception_experiment as argparse_cli

    click_result = CliRunner().invoke(click_cli.cli, ["run", "--help"])
    assert click_result.exit_code == 0
    assert "cannot resume" in click_result.output
    assert "audit/manual-salvage" in click_result.output

    monkeypatch.setattr(sys, "argv", ["run-deception-experiment", "--help"])
    with pytest.raises(SystemExit) as error:
        argparse_cli.main()
    assert error.value.code == 0
    argparse_help = capsys.readouterr().out
    assert "cannot resume" in argparse_help
    assert "audit/manual-salvage" in argparse_help


def test_public_clis_reject_resume_flags_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import interpretability.cli as click_cli
    import interpretability.run_deception_experiment as argparse_cli

    imported = []
    monkeypatch.setattr(click_cli, "_lazy_import", lambda: imported.append(True))
    click_result = CliRunner().invoke(
        click_cli.cli,
        ["run", "--resume-from", str(tmp_path / "snapshot.json")],
    )
    assert click_result.exit_code == 2
    assert "No such option '--resume-from'" in click_result.output
    assert imported == []

    monkeypatch.setattr(
        argparse_cli,
        "InterpretabilityRunner",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("resume flags must fail before model construction")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run-deception-experiment", "--resume", str(tmp_path / "snapshot.json")],
    )
    with pytest.raises(SystemExit) as error:
        argparse_cli.main()
    assert error.value.code == 2
    assert "unrecognized arguments: --resume" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--conditions", "nonsense"], "unsupported or empty"),
        (["--conditions", "high_incentive,"], "unsupported or empty"),
        (["--parallel-pod", "1/2"], "is unsupported"),
    ],
)
def test_argparse_rejects_lossy_condition_or_pod_surfaces_before_outputs(
    arguments,
    message,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import interpretability.run_deception_experiment as cli_module

    monkeypatch.setattr(
        cli_module,
        "InterpretabilityRunner",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid public surfaces must fail before model construction")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run-deception-experiment", *arguments, "--output", str(tmp_path)],
    )

    with pytest.raises(SystemExit) as error:
        cli_module.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_click_config_save_false_writes_recovery_but_no_dataset_or_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import interpretability.cli as cli_module
    from config.experiment import (
        CausalConfig,
        EvaluatorConfig,
        ExperimentConfig,
        ModelConfig,
        ScenarioConfig,
    )

    created = []
    trained = []

    def runner_factory(**kwargs):
        runner = object.__new__(InterpretabilityRunner)
        runner.model = SimpleNamespace(
            model_name="offline-model",
            tokenizer=SimpleNamespace(name_or_path="offline-tokenizer"),
        )
        runner.activation_samples = []
        runner.generation_records = []
        runner.label_records = []
        runner.interaction_events = []
        runner.intervention_designs = []
        runner.intervention_schedules = []
        runner.intervention_application_logs = []
        runner.experiment_track = kwargs["experiment_track"]
        runner.captured_actor_ids = tuple(kwargs["captured_actor_ids"])
        runner._pod_id = 0
        runner._trial_id_offset = kwargs["trial_id_offset"]
        runner._trial_id = kwargs["trial_id_offset"]
        runner.evaluator_type = kwargs["evaluator_type"]
        runner._deepeval_detector = None
        runner.save_dataset = lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("save_activations=False must not publish a dataset")
        )
        created.append(runner)
        return runner

    config = ExperimentConfig(
        experiment_name="recovery-only-contract",
        model=ModelConfig(
            name="google/gemma-2b-it",
            device="cpu",
            dtype="float32",
            use_sae=False,
        ),
        evaluator=EvaluatorConfig(enabled=False, ground_truth_method="rule"),
        causal=CausalConfig(enabled=False),
        scenarios=ScenarioConfig(
            scenarios=["hidden_value"],
            num_trials=1,
            max_rounds=1,
        ),
        output_dir=str(tmp_path),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        save_activations=False,
        log_to_file=False,
    )
    config_path = tmp_path / "experiment.json"
    config.save_json(str(config_path))

    monkeypatch.setattr(cli_module, "_lazy_import", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli_module, "get_emergent_scenarios", lambda: ["hidden_value"],
        raising=False,
    )
    monkeypatch.setattr(
        cli_module, "get_instructed_scenarios", lambda: ["hidden_value"],
        raising=False,
    )
    monkeypatch.setattr(
        cli_module, "InterpretabilityRunner", runner_factory, raising=False
    )
    monkeypatch.setattr(
        cli_module,
        "_run_emergent_experiment",
        lambda **_kwargs: {"total_samples": 0},
    )
    monkeypatch.setattr(
        cli_module,
        "_train_probes_on_data",
        lambda *_a, **_k: trained.append(True),
    )
    monkeypatch.setattr(cli_module, "_print_summary", lambda *_a, **_k: None)

    result = CliRunner().invoke(
        cli_module.cli,
        ["run", "--config", str(config_path)],
    )

    assert result.exit_code == 0, result.output
    assert len(created) == 1
    assert trained == []
    recovery_path = next(tmp_path.glob("emergent_recovery_only_*.json"))
    recovery = load_activation_recovery_checkpoint(recovery_path)
    assert recovery["experiment_progress"]["scientific_status"] == (
        "recovery_evidence_only_no_dataset"
    )
    assert not list(tmp_path.glob("activations_*.json"))
    manifest = json.loads(
        (tmp_path / "experiment_track_manifest.json").read_text()
    )
    assert manifest["experiment_name"] == "recovery-only-contract"
    assert manifest["activation_publication"] == {
        "enabled": False,
        "scientific_status": "recovery_evidence_only_no_dataset",
    }
