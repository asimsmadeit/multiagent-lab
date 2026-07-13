"""External legacy .pt consumers must require explicit trust."""

import importlib

from click.testing import CliRunner
import pytest
import torch

from interpretability.analyze_data import load_and_merge
from interpretability.probes.train_probes import run_full_analysis


def test_public_probe_loader_rejects_legacy_pt_by_default(tmp_path):
    path = tmp_path / "probe.pt"
    torch.save({"activations": {}, "labels": {}}, path)

    with pytest.raises(PermissionError, match="trusted=True"):
        run_full_analysis(str(path))


def test_analysis_loader_requires_trust_and_accepts_reviewed_artifact(tmp_path):
    path = tmp_path / "analysis.pt"
    payload = {"labels": {"gm_labels": [0.0, 1.0]}}
    torch.save(payload, path)

    with pytest.raises(PermissionError, match="trusted=True"):
        load_and_merge([str(path)], verbose=False)

    assert load_and_merge(
        [str(path)],
        verbose=False,
        trusted_legacy=True,
    ) == payload


def test_click_train_flag_defaults_to_rejection_and_passes_explicit_trust(
    tmp_path,
    monkeypatch,
):
    cli_module = importlib.import_module("interpretability.cli")
    path = tmp_path / "cli.pt"
    path.write_bytes(b"placeholder")
    calls = []

    monkeypatch.setattr(cli_module, "_lazy_import", lambda: None)

    def fake_train(data_path, output_dir, *, trusted_legacy=False):
        del data_path, output_dir
        calls.append(trusted_legacy)
        if not trusted_legacy:
            raise PermissionError(
                "Legacy .pt files can execute pickle payloads. Pass trusted=True"
            )
        return {}

    monkeypatch.setattr(cli_module, "_train_probes_on_data", fake_train)
    runner = CliRunner()

    rejected = runner.invoke(
        cli_module.cli,
        ["train", "--data", str(path), "--output", str(tmp_path / "rejected")],
    )
    accepted = runner.invoke(
        cli_module.cli,
        [
            "train",
            "--data",
            str(path),
            "--output",
            str(tmp_path / "accepted"),
            "--trust-legacy-pt",
        ],
    )

    assert rejected.exit_code != 0
    assert isinstance(rejected.exception, PermissionError)
    assert accepted.exit_code == 0, accepted.output
    assert calls == [False, True]


def test_click_train_help_names_legacy_trust_option():
    cli_module = importlib.import_module("interpretability.cli")
    result = CliRunner().invoke(cli_module.cli, ["train", "--help"])

    assert result.exit_code == 0
    assert "--trust-legacy-pt" in result.output
