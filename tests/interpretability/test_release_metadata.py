"""Release metadata, track separation, and reference-config contracts."""

# Pydantic v2 fields are concrete nested models at runtime; Pylint's static
# inference sees their class-level FieldInfo descriptors instead.
# pylint: disable=no-member

from __future__ import annotations

import hashlib
from importlib import resources
import json
from pathlib import Path
import os
import subprocess
import sys
import tomllib

import pytest

from config.experiment import ExperimentConfig
from interpretability.schema_registry import (
    schema_registry,
    schema_registry_checksum,
)
from interpretability.tracks import validate_track_assignment


def test_schema_registry_is_stable_complete_and_json_safe() -> None:
    registry = schema_registry()
    expected = {
        "activation_dataset", "activation_recovery_checkpoint", "adjudication",
        "array_bundle",
        "causal_application_receipt", "causal_design",
        "counterpart_knowledge_grant", "emergent_scenario_spec",
        "generation_record",
        "headline_probe", "label_record", "negotiation_domain",
        "intervention_application", "intervention_application_log",
        "intervention_design", "intervention_schedule",
        "probe_intervention", "response_qc", "runtime_executor",
        "schema_registry", "scripted_observation", "split_manifest",
        "simultaneous_batch", "solo_no_response_protocol", "trial_runtime",
    }
    assert set(registry) == expected
    assert all(isinstance(value, str) and value for value in registry.values())
    assert len(schema_registry_checksum()) == 64
    assert schema_registry_checksum() == schema_registry_checksum()
    json.dumps(registry, sort_keys=True)


def test_reference_config_validates_and_declares_access_track() -> None:
    config = ExperimentConfig.load_json("config/reference_offline.json")

    assert config.experiment_track == "single_agent_white_box"
    assert config.scenarios.num_trials == 1
    assert config.scenarios.probes == "auto"
    assert config.scenarios.executions_per_family == 64
    assert config.scenarios.agent_modules == ["theory_of_mind"]
    assert config.causal.enabled is False
    assert config.evaluator.ground_truth_method == "rule"
    assert config.model.use_sae is False
    assert config.save_activations is False
    assert config.verbose is True
    assert config.log_to_file is False


def test_published_reference_files_match_release_checksums() -> None:
    checksums = json.loads(
        Path("config/release_checksums.json").read_text(encoding="utf-8")
    )
    assert set(checksums) == {
        "config/reference_offline.json",
        "interpretability/data/DATA_CARD.txt",
    }
    for path, expected in checksums.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected


def test_package_data_names_every_required_resource_exactly() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]

    assert package_data == {
        "config": ["reference_offline.json", "release_checksums.json"],
        "interpretability.data": ["DATA_CARD.txt"],
    }
    assert "Negotiation Activation Dataset Card" in (
        resources.files("interpretability.data")
        .joinpath("DATA_CARD.txt")
        .read_text(encoding="utf-8")
    )


def test_installed_wheel_exposes_resources_outside_source_tree(tmp_path) -> None:
    repository = Path.cwd().resolve()
    wheel_directory = tmp_path / "wheel"
    target = tmp_path / "installed"
    wheel_directory.mkdir()
    environment = {
        **os.environ,
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_directory),
        ],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(wheel_directory.glob("*.whl"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(target),
            str(wheel),
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    script = """
from importlib.resources import files
from pathlib import Path
import json
import sys

repository = Path(sys.argv[1]).resolve()
assert all(repository != Path(entry or '.').resolve() for entry in sys.path)
card = files('interpretability.data').joinpath('DATA_CARD.txt').read_text('utf-8')
checksums = json.loads(
    files('config').joinpath('release_checksums.json').read_text('utf-8')
)
reference = files('config').joinpath('reference_offline.json').read_text('utf-8')
assert 'Negotiation Activation Dataset Card' in card
assert set(checksums) == {
    'config/reference_offline.json',
    'interpretability/data/DATA_CARD.txt',
}
assert json.loads(reference)['experiment_track'] == 'single_agent_white_box'
"""
    isolated_environment = {
        **environment,
        "PYTHONPATH": str(target),
        "PYTHONNOUSERSITE": "1",
    }
    subprocess.run(
        [sys.executable, "-c", script, str(repository)],
        cwd=tmp_path,
        env=isolated_environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_text_only_track_rejects_activation_persistence() -> None:
    with pytest.raises(ValueError, match="cannot save activations"):
        ExperimentConfig(experiment_track="text_only", save_activations=True)
    with pytest.raises(ValueError, match="cannot enable SAE"):
        ExperimentConfig(
            experiment_track="text_only",
            model={"use_sae": True},
        )
    with pytest.raises(ValueError, match="cannot enable causal"):
        ExperimentConfig(
            experiment_track="text_only",
            causal={"enabled": True},
        )


def test_text_only_track_disables_every_white_box_operation() -> None:
    config = ExperimentConfig(experiment_track="text_only")

    assert config.save_activations is False
    assert config.model.use_transformerlens is False
    assert config.model.cache_activations is False
    assert config.model.use_sae is False
    assert config.causal.enabled is False
    assert config.causal.run_activation_patching is False
    assert config.causal.run_ablation is False
    assert config.causal.run_steering is False


def test_track_assignments_enforce_access_and_modules() -> None:
    participants = ("seller", "buyer")
    validate_track_assignment(
        "single_agent_white_box",
        participant_ids=participants,
        captured_actor_ids=("seller",),
    )
    validate_track_assignment(
        "bilateral_white_box",
        participant_ids=participants,
        captured_actor_ids=participants,
    )
    validate_track_assignment(
        "theory_of_mind",
        participant_ids=participants,
        captured_actor_ids=("seller",),
        enabled_modules=("theory_of_mind",),
    )
    with pytest.raises(ValueError, match="every actor"):
        validate_track_assignment(
            "bilateral_white_box",
            participant_ids=participants,
            captured_actor_ids=("seller",),
        )
    with pytest.raises(ValueError, match="requires modules"):
        validate_track_assignment(
            "adaptive",
            participant_ids=participants,
            captured_actor_ids=("seller",),
        )
    with pytest.raises(ValueError, match="does not allow online adaptation"):
        validate_track_assignment(
            "single_agent_white_box",
            participant_ids=participants,
            captured_actor_ids=("seller",),
            enabled_modules=("strategy_evolution",),
        )
    with pytest.raises(ValueError, match="at least one captured actor"):
        validate_track_assignment(
            "adaptive",
            participant_ids=participants,
            captured_actor_ids=(),
            enabled_modules=("strategy_evolution",),
        )
