"""Offline CLI tests for the `deception events` command group (Plan 2, P9)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from interpretability.cli import cli
from interpretability.events.writer import EventWriter
from tests.events import test_projectors as fixture_lib


def _write_run(tmp_path: Path):
    trial = fixture_lib.build_full_trial()
    path = tmp_path / "run.jsonl"
    writer = EventWriter(
        path,
        run_id=fixture_lib.RUN_ID,
        pod_id=fixture_lib.POD_ID,
        fsync_mode="never",
    )
    for event in trial.events:
        writer.append(event)
    writer.close()
    return path, trial


def test_events_validate_reports_trials(tmp_path):
    path, _ = _write_run(tmp_path)
    result = CliRunner().invoke(cli, ["events", "validate", str(path)])
    assert result.exit_code == 0, result.output
    assert "VALID" in result.output
    assert fixture_lib.TRIAL_ID in result.output


def test_events_validate_fails_on_missing_file(tmp_path):
    result = CliRunner().invoke(
        cli, ["events", "validate", str(tmp_path / "absent.jsonl")]
    )
    assert result.exit_code != 0


def test_events_replay_projects_a_trial(tmp_path):
    path, _ = _write_run(tmp_path)
    result = CliRunner().invoke(
        cli,
        ["events", "replay", str(path),
         "--projection", "transcript", "--trial-id", fixture_lib.TRIAL_ID],
    )
    assert result.exit_code == 0, result.output
    assert "transcript" in result.output
    assert "semantic hash" in result.output


def test_events_trace_reports_lineage(tmp_path):
    path, trial = _write_run(tmp_path)
    target = trial.named["committed_a"].event_id
    result = CliRunner().invoke(
        cli, ["events", "trace", str(path), "--event-id", target]
    )
    assert result.exit_code == 0, result.output
    assert "lineage of" in result.output
    assert "ancestors:" in result.output
