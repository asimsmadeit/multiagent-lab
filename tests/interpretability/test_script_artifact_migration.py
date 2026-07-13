"""Standalone activation consumers share the safe artifact boundary."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from interpretability.data.io import load_array_bundle, save_array_bundle
from interpretability.script_artifacts import (
    download_activation_input,
    load_activation_input,
    prefer_safe_activation_path,
)
from scripts.aggregate_results import _relocate_safe_bundle
from scripts.backfill_sample_type import backfill


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ACTIVATION_CONSUMERS = (
    "run_confound_controls.py",
    "run_causal.py",
    "run_behavioral_steering.py",
    "label_noise_sensitivity.py",
    "scripts/tier_a_finish.py",
    "scripts/pca_on_directions.py",
    "scripts/backfill_sample_type.py",
    "scripts/recompute_paper_numbers.py",
    "scripts/aggregate_results.py",
    "scripts/ttpd_decomposition.py",
    "scripts/paper_camera_ready_runs.py",
    "scripts/investigate_dyadic_leakage.py",
)


@pytest.mark.parametrize("relative_path", ACTIVATION_CONSUMERS)
def test_consumer_help_exposes_explicit_legacy_trust(relative_path: str) -> None:
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    result = subprocess.run(
        [sys.executable, relative_path, "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--trust-legacy-pt" in result.stdout


@pytest.mark.parametrize("relative_path", ACTIVATION_CONSUMERS)
def test_consumer_has_no_direct_torch_load(relative_path: str) -> None:
    source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_path)
    direct_loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
    ]

    assert direct_loads == []
    assert "weights_only=False" not in source


def test_legacy_input_is_rejected_by_default_and_safe_sibling_wins(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy.pt"
    payload = {"metadata": [{"round_num": 0}], "labels": {}}
    torch.save(payload, legacy_path)

    with pytest.raises(PermissionError, match="trusted=True"):
        load_activation_input(legacy_path)
    assert load_activation_input(
        legacy_path,
        trust_legacy_pt=True,
    ) == payload

    safe_manifest = legacy_path.with_suffix(".json")
    safe_manifest.write_text("{}\n", encoding="utf-8")
    assert prefer_safe_activation_path(legacy_path) == safe_manifest


def test_hugging_face_download_prefers_complete_safe_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "activations.json"
    arrays = tmp_path / "activations.npz"
    manifest.write_text(
        json.dumps({"array_file": arrays.name}) + "\n",
        encoding="utf-8",
    )
    arrays.write_bytes(b"npz-placeholder")
    requested: list[str] = []

    def fake_download(*, repo_id: str, filename: str, repo_type: str) -> str:
        assert repo_id == "research/dataset"
        assert repo_type == "dataset"
        requested.append(filename)
        return str(manifest if filename.endswith(".json") else arrays)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    result = download_activation_input(
        "research/dataset",
        "scenario/activations.pt",
    )

    assert result == manifest
    assert requested == [
        "scenario/activations.json",
        "scenario/activations.npz",
    ]


def test_hugging_face_download_falls_back_to_configured_legacy_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from huggingface_hub.errors import EntryNotFoundError

    legacy_path = tmp_path / "activations.pt"
    legacy_path.write_bytes(b"review-before-loading")
    requested: list[str] = []

    def fake_download(*, repo_id: str, filename: str, repo_type: str) -> str:
        del repo_id, repo_type
        requested.append(filename)
        if filename.endswith(".json"):
            raise EntryNotFoundError("safe manifest is not published")
        return str(legacy_path)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    result = download_activation_input(
        "research/dataset",
        "scenario/activations.pt",
    )

    assert result == legacy_path
    assert requested == [
        "scenario/activations.json",
        "scenario/activations.pt",
    ]


def test_backfill_legacy_read_and_write_require_same_trust(tmp_path: Path) -> None:
    legacy_path = tmp_path / "activations.pt"
    torch.save(
        {
            "metadata": [
                {"round_num": -1},
                {"round_num": -2},
                {"round_num": 0},
            ],
            "labels": {},
        },
        legacy_path,
    )

    with pytest.raises(PermissionError, match="trusted=True"):
        backfill(legacy_path)
    counts = backfill(
        legacy_path,
        trust_legacy_pt=True,
    )

    assert counts["pre_verification"] == 1
    assert counts["post_plausibility"] == 1
    assert counts["negotiation"] == 1
    output_path = tmp_path / "activations.backfilled.pt"
    output = load_activation_input(output_path, trust_legacy_pt=True)
    assert [row["sample_type"] for row in output["metadata"]] == [
        "pre_verification",
        "post_plausibility",
        "negotiation",
    ]


def test_aggregate_output_relocation_preserves_safe_bundle(tmp_path: Path) -> None:
    array_path, manifest_path = save_array_bundle(
        tmp_path / "generated",
        {"activation": np.arange(6).reshape(2, 3)},
        {"kind": "activation-smoke"},
    )
    destination = tmp_path / "published.json"

    saved = _relocate_safe_bundle(manifest_path, destination)
    arrays, manifest = load_array_bundle(saved)

    assert saved == destination
    assert np.array_equal(arrays["activation"], np.arange(6).reshape(2, 3))
    assert manifest == {"kind": "activation-smoke"}
    assert not array_path.exists()
    assert destination.with_suffix(".npz").exists()
