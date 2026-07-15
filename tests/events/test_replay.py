"""Permanent deterministic replay, resume, ownership, and path-safety tests."""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, cast

import pytest

import interpretability.events.replay as replay_module
from interpretability.events.payloads import (
    ActionCommittedPayload,
    RunStartedPayload,
    TrialFailedPayload,
    TrialStartedPayload,
)
from interpretability.events.projectors import (
    ActivationSampleProjector,
    AgentViewProjector,
    DyadProjector,
    MetricInputProjection,
    MetricInputProjector,
    TranscriptProjector,
    TranscriptProjection,
    TrialStateProjection,
    TrialStateProjector,
)
from interpretability.events.replay import (
    MAX_STREAM_FILENAME_BYTES,
    PROJECTION_VERSION,
    SAFE_RESUME_BOUNDARY_EVENT_TYPES,
    ProjectionRequest,
    ReplayOwnershipError,
    ReplayPathError,
    ReplayProjectionError,
    ReplayRequestError,
    ReplaySourceError,
    create_projector,
    inspect_stream,
    inspect_stream_directory,
    inspect_streams,
    projector_registry,
    replay_projection,
    replay_validated_events,
)
from interpretability.events.schema import EventEnvelope, OpaqueEventPayload
from interpretability.events.writer import EventWriter
from tests.events import test_projectors as fixture_lib  # pylint: disable=import-error


@pytest.fixture(scope="module")
def full_trial() -> fixture_lib.FullTrial:
    return fixture_lib.build_full_trial()


def write_stream(
    path: Path,
    events: Iterable[EventEnvelope],
    *,
    run_id: str = fixture_lib.RUN_ID,
    pod_id: str = fixture_lib.POD_ID,
    incomplete: bool = False,
) -> tuple[EventEnvelope, ...]:
    materialized = tuple(events)
    writer = EventWriter(
        path,
        run_id=run_id,
        pod_id=pod_id,
        fsync_mode="never",
        allow_incomplete=incomplete,
    )
    for event in materialized:
        writer.append(event)
    writer.close(allow_incomplete=incomplete)
    return materialized


@pytest.fixture
def complete_stream(tmp_path: Path, full_trial: fixture_lib.FullTrial) -> Path:
    path = tmp_path / "complete.jsonl"
    write_stream(path, full_trial.events)
    return path


def prefix_through(
    trial: fixture_lib.FullTrial, name: str
) -> tuple[EventEnvelope, ...]:
    position = trial.events.index(trial.named[name])
    return trial.events[: position + 1]


def duplicate_event(
    event: EventEnvelope,
    *,
    index: int,
) -> EventEnvelope:
    return fixture_lib.clone_event(
        event,
        event_id=fixture_lib.event_id(index),
        parent_event_ids=(event.event_id,),
    )


def rewrite_stream_identity(
    events: Iterable[EventEnvelope],
    *,
    run_id: str = fixture_lib.RUN_ID,
    pod_id: str = fixture_lib.POD_ID,
) -> tuple[EventEnvelope, ...]:
    rewritten: list[EventEnvelope] = []
    for event in events:
        payload = event.payload
        if isinstance(payload, RunStartedPayload):
            payload = payload.model_copy(update={"run_id": run_id})
        rewritten.append(
            fixture_lib.clone_event(
                event,
                run_id=run_id,
                pod_id=pod_id,
                payload=payload,
            )
        )
    return fixture_lib.normalize(rewritten)


def add_failed_trial(
    trial: fixture_lib.FullTrial,
) -> tuple[EventEnvelope, ...]:
    scenario = trial.named["scenario"]
    started = EventEnvelope(
        event_id=fixture_lib.event_id(920),
        event_type="TrialStarted",
        run_id=fixture_lib.RUN_ID,
        pod_id=fixture_lib.POD_ID,
        trial_id="trial-second",
        dyad_id="dyad-second",
        sequence_num=0,
        recorded_at=trial.events[-1].recorded_at + timedelta(seconds=1),
        actor_id=None,
        actor_role=None,
        model_call_id=None,
        parent_event_ids=(scenario.event_id,),
        previous_event_hash=None,
        payload=TrialStartedPayload(
            trial_id="trial-second",
            scenario_instance_id="scenario-instance",
            dyad_id="dyad-second",
            attempt=1,
            actor_ids=(fixture_lib.ALICE, fixture_lib.BOB),
            source_scenario_event_id=scenario.event_id,
        ),
        payload_schema_version="1.0.0",
    )
    failed = EventEnvelope(
        event_id=fixture_lib.event_id(921),
        event_type="TrialFailed",
        run_id=fixture_lib.RUN_ID,
        pod_id=fixture_lib.POD_ID,
        trial_id="trial-second",
        dyad_id="dyad-second",
        sequence_num=1,
        recorded_at=started.recorded_at + timedelta(seconds=1),
        actor_id=None,
        actor_role=None,
        model_call_id=None,
        parent_event_ids=(started.event_id,),
        previous_event_hash=started.content_hash,
        payload=TrialFailedPayload(
            trial_id="trial-second",
            error_type="ExpectedFailure",
            error_message_hash=fixture_lib.digest("failed trial"),
            resumable=False,
            last_event_id=started.event_id,
        ),
        payload_schema_version="1.0.0",
    )
    return (*trial.events, started, failed)


def opaque_event(
    trial: fixture_lib.FullTrial,
    event_type: str,
    *,
    index: int,
) -> EventEnvelope:
    outcome = trial.named["outcome"]
    return EventEnvelope(
        event_id=fixture_lib.event_id(index),
        event_type=event_type,
        run_id=fixture_lib.RUN_ID,
        pod_id=fixture_lib.POD_ID,
        trial_id=fixture_lib.TRIAL_ID,
        dyad_id=fixture_lib.DYAD_ID,
        sequence_num=outcome.sequence_num + 1,
        recorded_at=outcome.recorded_at + timedelta(microseconds=1),
        actor_id=None,
        actor_role=None,
        model_call_id=None,
        parent_event_ids=(outcome.event_id,),
        previous_event_hash=outcome.content_hash,
        payload=OpaqueEventPayload.from_payload_dict(event_type, "9.0.0", {}),
        payload_schema_version="9.0.0",
    )


def write_raw_events(path: Path, events: Iterable[EventEnvelope]) -> None:
    """Write canonical records without invoking writer-side validation."""

    path.write_bytes(
        b"".join(event.to_json().encode("utf-8") + b"\n" for event in events)
    )


def classification_counts(summary: Any) -> dict[str, int]:
    return {
        item.classification: item.count for item in summary.classification_counts
    }


PROJECTION_CASES: tuple[
    tuple[str, dict[str, Any] | None, type[Any]], ...
] = (
    ("transcript", None, TranscriptProjector),
    (
        "agent_view",
        {"actor_id": fixture_lib.ALICE, "through_sequence_num": 20},
        AgentViewProjector,
    ),
    ("trial_state", None, TrialStateProjector),
    ("activation_samples", None, ActivationSampleProjector),
    ("dyad", None, DyadProjector),
    ("metric_input", None, MetricInputProjector),
)


def test_registry_is_explicit_complete_immutable_and_factory_is_fresh() -> None:
    first = projector_registry()
    second = projector_registry()

    assert first == second
    assert first is not second
    assert [definition.name for definition in first] == [
        "transcript",
        "agent_view",
        "trial_state",
        "activation_samples",
        "dyad",
        "metric_input",
    ]
    assert all(definition.version == PROJECTION_VERSION for definition in first)
    assert first[1].required_config_keys == ("actor_id",)
    assert first[1].optional_config_keys == ("through_sequence_num",)

    for name, config, expected_type in PROJECTION_CASES:
        request = ProjectionRequest(
            name,
            fixture_lib.TRIAL_ID,
            projector_config=config,
        )
        one = create_projector(request)
        two = create_projector(request)
        assert isinstance(one, expected_type)
        assert isinstance(two, expected_type)
        assert one is not two


@pytest.mark.parametrize(
    ("name", "config", "message"),
    (
        ("unknown", None, "unknown projection"),
        ("agent_view", None, "missing"),
        ("agent_view", {}, "missing"),
        ("agent_view", {"actor_id": ""}, "actor_id"),
        ("agent_view", {"actor_id": 3}, "actor_id"),
        ("agent_view", {"actor_id": "alice", "unexpected": 1}, "extra"),
        ("agent_view", {"actor_id": "alice", "external_parent_ids": ()}, "extra"),
        ("transcript", {"actor_id": "alice"}, "extra"),
        ("transcript", {"through_sequence_num": 1}, "extra"),
        ("transcript", {"external_parent_ids": ()}, "extra"),
    ),
)
def test_projection_request_rejects_unknown_missing_and_extra_config(
    name: str, config: dict[str, Any] | None, message: str
) -> None:
    with pytest.raises(ReplayRequestError, match=message):
        ProjectionRequest(name, fixture_lib.TRIAL_ID, projector_config=config)


@pytest.mark.parametrize("cutoff", (-1, True, 1.5, "3", None))
def test_agent_view_config_rejects_invalid_cutoff(cutoff: Any) -> None:
    with pytest.raises(ReplayRequestError, match="through_sequence_num"):
        ProjectionRequest(
            "agent_view",
            fixture_lib.TRIAL_ID,
            projector_config={
                "actor_id": fixture_lib.ALICE,
                "through_sequence_num": cutoff,
            },
        )


def test_projection_request_rejects_invalid_types_versions_and_identities() -> None:
    with pytest.raises(ReplayRequestError, match="mapping"):
        ProjectionRequest(
            "transcript",
            fixture_lib.TRIAL_ID,
            projector_config=("bad",),  # type: ignore[arg-type]
        )
    with pytest.raises(ReplayRequestError, match="keys"):
        ProjectionRequest(
            "transcript",
            fixture_lib.TRIAL_ID,
            projector_config={1: "bad"},  # type: ignore[dict-item]
        )
    with pytest.raises(ReplayRequestError, match="keys"):
        ProjectionRequest(
            "transcript",
            fixture_lib.TRIAL_ID,
            projector_config={"": "bad"},
        )
    with pytest.raises(ReplayRequestError, match="unsupported"):
        ProjectionRequest(
            "transcript",
            fixture_lib.TRIAL_ID,
            projection_version="2.0.0",
        )
    with pytest.raises(ReplayRequestError, match="trial_id"):
        ProjectionRequest("transcript", "")
    with pytest.raises(ReplayRequestError, match="canonical UUID"):
        ProjectionRequest(
            "transcript",
            fixture_lib.TRIAL_ID,
            expected_run_id="not-a-uuid",
        )
    with pytest.raises(ReplayRequestError, match="expected_pod_id"):
        ProjectionRequest(
            "transcript",
            fixture_lib.TRIAL_ID,
            expected_pod_id="",
        )
    with pytest.raises(TypeError, match="ProjectionRequest"):
        create_projector(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("name,config,_", PROJECTION_CASES)
def test_file_and_in_memory_replay_are_byte_equivalent_for_all_projectors(
    name: str,
    config: dict[str, Any] | None,
    _: type[Any],
    complete_stream: Path,
    full_trial: fixture_lib.FullTrial,
) -> None:
    request = ProjectionRequest(
        name,
        fixture_lib.TRIAL_ID,
        expected_run_id=fixture_lib.RUN_ID,
        expected_pod_id=fixture_lib.POD_ID,
        projector_config=config,
    )
    file_result = replay_projection(complete_stream, request)
    memory_result = replay_validated_events(full_trial.events, request)
    generator_result = replay_validated_events(
        (event for event in full_trial.events), request
    )

    assert file_result == memory_result == generator_result
    assert (
        file_result.canonical_semantic_json()
        == memory_result.canonical_semantic_json()
        == generator_result.canonical_semantic_json()
    )
    assert file_result.semantic_hash == memory_result.semantic_hash
    assert file_result.manifest.projection_name == name
    assert file_result.manifest.projector_config == request.projector_config


def test_replay_manifest_records_exact_identities_terminals_hashes_and_warnings(
    complete_stream: Path, full_trial: fixture_lib.FullTrial
) -> None:
    request = ProjectionRequest("activation_samples", fixture_lib.TRIAL_ID)
    result = replay_projection(complete_stream, request)
    manifest = result.manifest
    source = manifest.source

    assert (manifest.run_id, manifest.pod_id, manifest.trial_id, manifest.dyad_id) == (
        fixture_lib.RUN_ID,
        fixture_lib.POD_ID,
        fixture_lib.TRIAL_ID,
        fixture_lib.DYAD_ID,
    )
    assert source.event_count == len(full_trial.events)
    assert sum(lane.event_count for lane in source.lanes) == len(full_trial.events)
    assert [lane.trial_id for lane in source.lanes] == [None, fixture_lib.TRIAL_ID]
    assert source.lanes[-1].last_event_id == full_trial.named["trial_completed"].event_id
    assert source.lanes[-1].last_event_type == "TrialCompleted"
    assert source.lanes[-1].last_content_hash == full_trial.events[-1].content_hash
    assert len(source.source_semantic_hash) == 64
    assert manifest.projection_semantic_hash == result.projection.semantic_hash
    assert len(manifest.warnings) == 8
    assert {warning.field_name for warning in manifest.warnings} == {
        "round_num",
        "prompt",
        "response",
        "activation_values",
    }
    assert all(warning.source_event_id is not None for warning in manifest.warnings)


def test_public_factory_projectors_remain_resettable_and_idempotent(
    full_trial: fixture_lib.FullTrial,
) -> None:
    for name, config, _ in PROJECTION_CASES:
        request = ProjectionRequest(name, fixture_lib.TRIAL_ID, projector_config=config)
        projector = create_projector(request)
        first = projector.project(event for event in full_trial.events)
        second = projector.project(full_trial.events)
        assert first.canonical_semantic_json() == second.canonical_semantic_json()
        assert first.semantic_hash == second.semantic_hash
        assert projector.last_result == second
        projector.reset()
        assert projector.last_result is None


def test_recorded_at_retiming_is_invariant_but_semantic_mutation_changes_result(
    tmp_path: Path, full_trial: fixture_lib.FullTrial
) -> None:
    original = tmp_path / "original.jsonl"
    retimed = tmp_path / "retimed.jsonl"
    write_stream(original, full_trial.events)
    shifted = tuple(
        event.model_copy(update={"recorded_at": event.recorded_at + timedelta(days=9)})
        for event in full_trial.events
    )
    write_stream(retimed, shifted)
    request = ProjectionRequest("transcript", fixture_lib.TRIAL_ID)
    baseline = replay_projection(original, request)
    replayed = replay_projection(retimed, request)
    changed_trial = fixture_lib.build_full_trial(
        first_action_hash=fixture_lib.digest("semantically changed action")
    )
    changed = replay_validated_events(changed_trial.events, request)

    assert original.read_bytes() != retimed.read_bytes()
    assert baseline.canonical_semantic_json() == replayed.canonical_semantic_json()
    assert baseline.manifest.source.source_semantic_hash == replayed.manifest.source.source_semantic_hash
    assert baseline.semantic_hash == replayed.semantic_hash
    assert changed.canonical_semantic_json() != baseline.canonical_semantic_json()
    assert changed.semantic_hash != baseline.semantic_hash


SAFE_BOUNDARY_CASES = (
    "trial_started",
    "turn_a",
    "label_rules",
    "monitor",
    "qc",
    "outcome",
)


@pytest.mark.parametrize("last_name", SAFE_BOUNDARY_CASES)
def test_each_allowlisted_clean_boundary_is_structurally_resumable(
    tmp_path: Path,
    full_trial: fixture_lib.FullTrial,
    last_name: str,
) -> None:
    path = tmp_path / f"safe-{last_name}.jsonl"
    events = prefix_through(full_trial, last_name)
    write_stream(path, events, incomplete=True)

    result = inspect_stream(path)
    trial = result.trials[0]
    terminal = events[-1]

    assert result.valid
    assert trial.classification == "resumable_safe_boundary"
    assert trial.safe_boundary_event_id == terminal.event_id
    assert trial.safe_boundary_event_type == terminal.event_type
    assert trial.pending_linkage.empty
    assert trial.terminal_event_id is None
    assert "not proof of serializable Concordia state" in trial.reason
    assert not hasattr(trial, "serializable_state")


def test_safe_boundary_fixture_exercises_every_allowlisted_event_type(
    full_trial: fixture_lib.FullTrial,
) -> None:
    assert {
        full_trial.named[name].event_type for name in SAFE_BOUNDARY_CASES
    } == set(SAFE_RESUME_BOUNDARY_EVENT_TYPES)


@pytest.mark.parametrize(
    ("last_name", "pending_field"),
    (
        ("call_a_started", "open_model_call_ids"),
        ("call_a_completed", "awaiting_artifact_call_ids"),
        ("capture_a2", "awaiting_artifact_call_ids"),
        ("capture_a5", "awaiting_proposal_call_ids"),
        ("proposed_a", "open_proposal_event_ids"),
        ("committed_a", "open_committed_action_event_ids"),
    ),
)
def test_unfinished_call_artifact_and_action_stages_require_restart(
    tmp_path: Path,
    full_trial: fixture_lib.FullTrial,
    last_name: str,
    pending_field: str,
) -> None:
    path = tmp_path / f"busy-{last_name}.jsonl"
    write_stream(path, prefix_through(full_trial, last_name), incomplete=True)

    trial = inspect_stream(path).trials[0]

    assert trial.classification == "restart_required"
    assert getattr(trial.pending_linkage, pending_field)
    assert "unfinished call/artifact/action linkage" in trial.reason
    assert trial.safe_boundary_event_id is None
    assert trial.safe_boundary_event_type is None


def test_clean_nonallowlisted_boundary_requires_restart(
    tmp_path: Path, full_trial: fixture_lib.FullTrial
) -> None:
    path = tmp_path / "nonallowlisted.jsonl"
    write_stream(
        path,
        prefix_through(full_trial, "initial_b"),
        incomplete=True,
    )

    trial = inspect_stream(path).trials[0]

    assert trial.classification == "restart_required"
    assert trial.pending_linkage.empty
    assert "not an explicitly allowlisted resume boundary" in trial.reason


def test_completed_and_failed_trials_are_terminal_and_not_resume_claims(
    tmp_path: Path, full_trial: fixture_lib.FullTrial
) -> None:
    path = tmp_path / "terminal-trials.jsonl"
    events = add_failed_trial(full_trial)
    write_stream(path, events)

    result = inspect_stream(path)
    trials = {trial.trial_id: trial for trial in result.trials}
    completed = trials[fixture_lib.TRIAL_ID]
    failed = trials["trial-second"]

    assert result.valid
    assert completed.classification == "completed"
    assert completed.terminal_event_id == full_trial.named["trial_completed"].event_id
    assert failed.classification == "failed"
    assert failed.terminal_event_id == events[-1].event_id
    assert completed.safe_boundary_event_id is None
    assert failed.safe_boundary_event_id is None
    assert completed.pending_linkage.empty
    assert failed.pending_linkage.empty


@pytest.mark.parametrize(
    ("attack", "expected"),
    (
        ("commit_hash", "action hash"),
        ("capture_tokenizer", "activation identity/metadata"),
        ("duplicate_terminal", "duplicate terminals"),
    ),
)
def test_resume_inspection_marks_invalid_linkage_without_raising(
    tmp_path: Path,
    full_trial: fixture_lib.FullTrial,
    attack: str,
    expected: str,
) -> None:
    if attack == "commit_hash":
        original = full_trial.named["committed_a"]
        replacement = fixture_lib.clone_event(
            original,
            payload=original.payload.model_copy(
                update={"action_hash": fixture_lib.digest("wrong action")}
            ),
        )
        events = fixture_lib.replace_named(full_trial, "committed_a", replacement)
    elif attack == "capture_tokenizer":
        original = full_trial.named["capture_a2"]
        artifact = original.payload.artifact.model_copy(
            update={"tokenizer_id": "wrong-tokenizer"}
        )
        replacement = fixture_lib.clone_event(
            original,
            payload=original.payload.model_copy(update={"artifact": artifact}),
        )
        events = fixture_lib.replace_named(full_trial, "capture_a2", replacement)
    else:
        events = fixture_lib.lineage_attack(full_trial, "duplicate_terminal")
    path = tmp_path / f"invalid-{attack}.jsonl"
    write_stream(path, events)

    result = inspect_stream(path)

    assert not result.valid
    assert result.error_type is None
    assert result.source is not None
    assert result.trials[0].classification == "invalid"
    assert expected in result.trials[0].reason


def test_exact_trial_replay_uses_only_selected_trial_and_required_run_prefix(
    tmp_path: Path, full_trial: fixture_lib.FullTrial
) -> None:
    path = tmp_path / "two-trials.jsonl"
    events = add_failed_trial(full_trial)
    write_stream(path, events)

    first = replay_projection(
        path,
        ProjectionRequest("transcript", fixture_lib.TRIAL_ID),
    )
    second = replay_projection(
        path,
        ProjectionRequest("trial_state", "trial-second"),
    )

    first_projection = cast(TranscriptProjection, first.projection)
    second_projection = cast(TrialStateProjection, second.projection)
    assert isinstance(first_projection, TranscriptProjection)
    assert isinstance(second_projection, TrialStateProjection)
    assert first_projection.trial_id == fixture_lib.TRIAL_ID  # pylint: disable=no-member
    assert len(first_projection.actions) == 2  # pylint: disable=no-member
    assert second_projection.trial_id == "trial-second"  # pylint: disable=no-member
    assert second_projection.lifecycle_status == "failed"  # pylint: disable=no-member
    assert second_projection.commitments == ()  # pylint: disable=no-member
    assert [lane.trial_id for lane in second.manifest.source.lanes] == [
        None,
        fixture_lib.TRIAL_ID,
        "trial-second",
    ]
    assert second.manifest.source.event_count == len(events)


def test_exact_trial_replay_rejects_absent_trial(
    complete_stream: Path,
) -> None:
    with pytest.raises(ReplayOwnershipError, match="absent"):
        replay_projection(
            complete_stream,
            ProjectionRequest("transcript", "absent-trial"),
        )


def test_expected_file_and_memory_ownership_is_enforced(
    complete_stream: Path, full_trial: fixture_lib.FullTrial
) -> None:
    wrong_run = ProjectionRequest(
        "transcript",
        fixture_lib.TRIAL_ID,
        expected_run_id=fixture_lib.OTHER_RUN_ID,
    )
    wrong_pod = ProjectionRequest(
        "transcript",
        fixture_lib.TRIAL_ID,
        expected_pod_id="wrong-pod",
    )

    with pytest.raises(ReplaySourceError, match="strict validation"):
        replay_projection(complete_stream, wrong_run)
    with pytest.raises(ReplayOwnershipError, match="does not match expected"):
        replay_validated_events(full_trial.events, wrong_run)
    with pytest.raises(ReplaySourceError, match="strict validation"):
        replay_projection(complete_stream, wrong_pod)
    with pytest.raises(ReplayOwnershipError, match="does not match expected"):
        replay_validated_events(full_trial.events, wrong_pod)


@pytest.mark.parametrize(
    ("attack", "expected"),
    (
        ("content_hash", "hash"),
        ("sequence_gap", "sequence"),
        ("previous_hash", "hash"),
        ("missing_parent", "parent"),
    ),
)
def test_replay_rejects_hash_sequence_and_parent_corruption(
    tmp_path: Path,
    full_trial: fixture_lib.FullTrial,
    attack: str,
    expected: str,
) -> None:
    events = list(full_trial.events)
    target_position = events.index(full_trial.named["private_a"])
    target = events[target_position]
    if attack == "content_hash":
        events[target_position] = target.model_copy(
            update={"content_hash": fixture_lib.digest("forged content hash")}
        )
    elif attack == "sequence_gap":
        events[target_position] = fixture_lib.clone_event(
            target,
            sequence_num=target.sequence_num + 1,
        )
    elif attack == "previous_hash":
        events[target_position] = fixture_lib.clone_event(
            target,
            previous_event_hash=fixture_lib.digest("wrong prior hash"),
        )
    else:
        events[target_position] = fixture_lib.clone_event(
            target,
            parent_event_ids=(fixture_lib.event_id(999),),
        )
    path = tmp_path / f"corrupt-{attack}.jsonl"
    write_raw_events(path, events)

    with pytest.raises(ReplaySourceError, match="strict validation"):
        replay_projection(
            path,
            ProjectionRequest("transcript", fixture_lib.TRIAL_ID),
        )
    inspection = inspect_stream(path)
    assert not inspection.valid
    assert inspection.source is None
    assert expected in (inspection.error_reason or "").lower()
    assert inspection.trials[0].classification == "invalid"


def test_truncated_and_noncanonical_records_fail_closed(
    tmp_path: Path, complete_stream: Path
) -> None:
    raw = complete_stream.read_bytes()
    truncated = tmp_path / "truncated.jsonl"
    truncated.write_bytes(raw[:-1])
    values = raw.splitlines(keepends=True)
    noncanonical = tmp_path / "noncanonical.jsonl"
    first = json.dumps(json.loads(values[0]), sort_keys=True).encode("utf-8") + b"\n"
    noncanonical.write_bytes(first + b"".join(values[1:]))

    for path in (truncated, noncanonical):
        with pytest.raises(ReplaySourceError, match="strict validation"):
            replay_projection(
                path,
                ProjectionRequest("transcript", fixture_lib.TRIAL_ID),
            )
        inspection = inspect_stream(path)
        assert not inspection.valid
        assert inspection.source is None
        assert inspection.error_type is not None
        assert inspection.trials[0].classification == "invalid"


def test_opaque_unrelated_event_is_preserved_but_required_type_fails_projection(
    tmp_path: Path, full_trial: fixture_lib.FullTrial
) -> None:
    unrelated = fixture_lib.insert_after(
        full_trial,
        "outcome",
        opaque_event(full_trial, "FutureAnnotation", index=960),
    )
    required = fixture_lib.insert_after(
        full_trial,
        "outcome",
        opaque_event(full_trial, "BehaviorLabeled", index=961),
    )
    unrelated_path = tmp_path / "opaque-unrelated.jsonl"
    required_path = tmp_path / "opaque-required.jsonl"
    write_stream(unrelated_path, unrelated)
    write_stream(required_path, required)

    result = replay_projection(
        unrelated_path,
        ProjectionRequest("metric_input", fixture_lib.TRIAL_ID),
    )
    projection = cast(MetricInputProjection, result.projection)
    assert isinstance(projection, MetricInputProjection)
    assert projection.trial_id == fixture_lib.TRIAL_ID  # pylint: disable=no-member
    with pytest.raises(ReplayProjectionError, match="opaque"):
        replay_projection(
            required_path,
            ProjectionRequest("metric_input", fixture_lib.TRIAL_ID),
        )


def test_concurrent_source_mutation_is_wrapped_as_replay_source_error(
    tmp_path: Path,
    full_trial: fixture_lib.FullTrial,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "mutating.jsonl"
    write_stream(path, full_trial.events)
    original_iter_events = replay_module.EventReader.iter_events
    mutated = False

    def mutating_iter_events(
        reader: Any, *args: Any, **kwargs: Any
    ) -> Iterable[Any]:
        nonlocal mutated
        iterator = original_iter_events(reader, *args, **kwargs)
        yield next(iterator)
        if not mutated:
            mutated = True
            with path.open("ab") as stream:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
        yield from iterator

    monkeypatch.setattr(
        replay_module.EventReader,
        "iter_events",
        mutating_iter_events,
    )

    with pytest.raises(ReplaySourceError, match="changed during iteration"):
        replay_projection(
            path,
            ProjectionRequest("transcript", fixture_lib.TRIAL_ID),
        )


def test_replay_and_inspection_are_read_only_without_sidecars(
    complete_stream: Path,
) -> None:
    request = ProjectionRequest("transcript", fixture_lib.TRIAL_ID)
    parent = complete_stream.parent
    before_bytes = complete_stream.read_bytes()
    before_stat = complete_stream.stat()
    before_entries = tuple(sorted(path.name for path in parent.iterdir()))

    replay_projection(complete_stream, request)
    inspect_stream(complete_stream)
    inspect_streams((complete_stream,))
    inspect_stream_directory(parent, filenames=(complete_stream.name,))

    after_stat = complete_stream.stat()
    assert complete_stream.read_bytes() == before_bytes
    assert (
        after_stat.st_ino,
        after_stat.st_size,
        after_stat.st_mtime_ns,
    ) == (
        before_stat.st_ino,
        before_stat.st_size,
        before_stat.st_mtime_ns,
    )
    assert tuple(sorted(path.name for path in parent.iterdir())) == before_entries


def test_empty_regular_stream_cannot_be_projected_but_is_safely_inspectable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.jsonl"
    path.touch()

    with pytest.raises(ReplaySourceError, match="empty"):
        replay_projection(
            path,
            ProjectionRequest("transcript", fixture_lib.TRIAL_ID),
        )
    inspection = inspect_stream(path)
    assert inspection.valid
    assert inspection.source is not None
    assert inspection.source.event_count == 0
    assert inspection.trials == ()


@pytest.mark.parametrize("bad_limit", (0, -1, True, 1.5, "10"))
def test_record_size_limit_requires_a_positive_integer(
    complete_stream: Path, bad_limit: Any
) -> None:
    expected = TypeError if bad_limit is True or not isinstance(bad_limit, int) else ValueError
    with pytest.raises(expected, match="max_record_bytes"):
        replay_projection(
            complete_stream,
            ProjectionRequest("transcript", fixture_lib.TRIAL_ID),
            max_record_bytes=bad_limit,
        )
    with pytest.raises(expected, match="max_record_bytes"):
        inspect_stream(complete_stream, max_record_bytes=bad_limit)


def test_oversized_record_is_reported_without_partial_projection(
    complete_stream: Path,
) -> None:
    request = ProjectionRequest("transcript", fixture_lib.TRIAL_ID)
    with pytest.raises(ReplaySourceError, match="strict validation"):
        replay_projection(complete_stream, request, max_record_bytes=32)

    inspection = inspect_stream(complete_stream, max_record_bytes=32)
    assert not inspection.valid
    assert inspection.source is None
    assert inspection.error_type == "EventRecordTooLargeError"
    assert inspection.trials[0].classification == "invalid"


def test_source_name_must_be_explicit_and_safe(
    complete_stream: Path,
) -> None:
    for source_name in ("", "bad\x00name", 3):
        with pytest.raises(ReplayPathError, match="source_name"):
            inspect_stream(complete_stream, source_name=source_name)  # type: ignore[arg-type]


def test_missing_directory_fifo_and_symlink_stream_paths_are_rejected(
    tmp_path: Path,
    complete_stream: Path,
) -> None:
    directory = tmp_path / "directory.jsonl"
    directory.mkdir()
    fifo = tmp_path / "events.fifo"
    os.mkfifo(fifo)
    symlink = tmp_path / "events-link.jsonl"
    symlink.symlink_to(complete_stream)
    missing = tmp_path / "missing.jsonl"
    request = ProjectionRequest("transcript", fixture_lib.TRIAL_ID)

    for path in (directory, fifo, symlink, missing):
        with pytest.raises(ReplayPathError):
            inspect_stream(path)
        with pytest.raises(ReplayPathError):
            replay_projection(path, request)
    for path in (None, ""):
        with pytest.raises(ReplayPathError, match="explicit"):
            inspect_stream(path)  # type: ignore[arg-type]


def test_stream_beneath_symlinked_parent_is_rejected(
    tmp_path: Path, full_trial: fixture_lib.FullTrial
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    target = real / "events.jsonl"
    write_stream(target, full_trial.events)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real, target_is_directory=True)

    with pytest.raises(ReplayPathError, match="symlink"):
        inspect_stream(linked_parent / target.name)


def test_multi_stream_order_is_deterministic_and_same_trial_across_pods_is_valid(
    tmp_path: Path, full_trial: fixture_lib.FullTrial
) -> None:
    alpha = tmp_path / "alpha.jsonl"
    zulu = tmp_path / "zulu.jsonl"
    write_stream(
        alpha,
        rewrite_stream_identity(full_trial.events, pod_id="pod-alpha"),
        pod_id="pod-alpha",
    )
    write_stream(
        zulu,
        rewrite_stream_identity(full_trial.events, pod_id="pod-zulu"),
        pod_id="pod-zulu",
    )

    forward = inspect_streams((alpha, zulu))
    reverse = inspect_streams(path for path in (zulu, alpha))

    assert forward.canonical_semantic_json() == reverse.canonical_semantic_json()
    assert [stream.source_name for stream in forward.streams] == [
        "alpha.jsonl",
        "zulu.jsonl",
    ]
    assert forward.run_id == fixture_lib.RUN_ID
    assert forward.stream_count == 2
    assert forward.trial_count == 2
    assert forward.valid
    assert classification_counts(forward) == {
        "completed": 2,
        "failed": 0,
        "resumable_safe_boundary": 0,
        "restart_required": 0,
        "invalid": 0,
    }
    assert {
        stream.trials[0].pod_id for stream in forward.streams
    } == {"pod-alpha", "pod-zulu"}


def test_multi_stream_rejects_mixed_runs_duplicate_paths_names_and_ownership(
    tmp_path: Path, full_trial: fixture_lib.FullTrial
) -> None:
    primary = tmp_path / "primary.jsonl"
    copied = tmp_path / "copied.jsonl"
    other_run = tmp_path / "other-run.jsonl"
    write_stream(primary, full_trial.events)
    write_stream(copied, full_trial.events)
    changed_run_events = rewrite_stream_identity(
        full_trial.events,
        run_id=fixture_lib.OTHER_RUN_ID,
    )
    write_stream(
        other_run,
        changed_run_events,
        run_id=fixture_lib.OTHER_RUN_ID,
    )

    with pytest.raises(ReplayPathError, match="duplicates"):
        inspect_streams((primary, primary))
    with pytest.raises(ReplayOwnershipError, match="duplicate .* ownership"):
        inspect_streams((primary, copied))
    with pytest.raises(ReplayOwnershipError, match="mixes run ownership"):
        inspect_streams((primary, other_run))

    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    same_name_one = one / "events.jsonl"
    same_name_two = two / "events.jsonl"
    write_stream(same_name_one, full_trial.events)
    pod_two_events = rewrite_stream_identity(full_trial.events, pod_id="pod-two")
    write_stream(same_name_two, pod_two_events, pod_id="pod-two")
    with pytest.raises(ReplayOwnershipError, match="duplicate source names"):
        inspect_streams((same_name_one, same_name_two))


@pytest.mark.parametrize("bad_limit", (0, -1, True, 1.5, "2"))
def test_multi_stream_count_limit_requires_a_positive_integer(
    bad_limit: Any,
) -> None:
    expected = TypeError if bad_limit is True or not isinstance(bad_limit, int) else ValueError
    with pytest.raises(expected, match="max_streams"):
        inspect_streams((), max_streams=bad_limit)


def test_multi_stream_input_is_explicit_bounded_and_not_a_single_path(
    complete_stream: Path,
) -> None:
    with pytest.raises(TypeError, match="iterable of paths"):
        inspect_streams(complete_stream)
    with pytest.raises(ReplayPathError, match="maximum"):
        inspect_streams((complete_stream, complete_stream), max_streams=1)


def test_multi_stream_preserves_invalid_records_and_all_classification_counts(
    tmp_path: Path, full_trial: fixture_lib.FullTrial
) -> None:
    complete = tmp_path / "complete.jsonl"
    failed = tmp_path / "failed.jsonl"
    safe = tmp_path / "safe.jsonl"
    restart = tmp_path / "restart.jsonl"
    invalid = tmp_path / "invalid.jsonl"

    complete_events = rewrite_stream_identity(full_trial.events, pod_id="pod-complete")
    write_stream(complete, complete_events, pod_id="pod-complete")
    failed_events = rewrite_stream_identity(
        add_failed_trial(full_trial), pod_id="pod-failed"
    )
    write_stream(failed, failed_events, pod_id="pod-failed")
    safe_events = rewrite_stream_identity(
        prefix_through(full_trial, "outcome"), pod_id="pod-safe"
    )
    write_stream(safe, safe_events, pod_id="pod-safe", incomplete=True)
    restart_events = rewrite_stream_identity(
        prefix_through(full_trial, "call_a_started"), pod_id="pod-restart"
    )
    write_stream(restart, restart_events, pod_id="pod-restart", incomplete=True)
    invalid_events = rewrite_stream_identity(full_trial.events, pod_id="pod-invalid")
    write_stream(invalid, invalid_events, pod_id="pod-invalid")
    invalid.write_bytes(invalid.read_bytes()[:-1])

    summary = inspect_streams((restart, invalid, safe, failed, complete))

    assert not summary.valid
    assert summary.stream_count == 5
    assert summary.trial_count == 6
    assert classification_counts(summary) == {
        "completed": 2,
        "failed": 1,
        "resumable_safe_boundary": 1,
        "restart_required": 1,
        "invalid": 1,
    }
    invalid_inspection = next(
        stream for stream in summary.streams if stream.source_name == invalid.name
    )
    assert not invalid_inspection.valid
    assert invalid_inspection.source is None
    assert invalid_inspection.error_type is not None


def test_directory_inspection_is_explicit_unicode_safe_and_deterministic(
    tmp_path: Path, full_trial: fixture_lib.FullTrial
) -> None:
    directory = tmp_path / "streams"
    directory.mkdir()
    ascii_name = "alpha.jsonl"
    unicode_name = "réplay-数据.jsonl"
    ignored_name = "not-selected.jsonl"
    first = directory / ascii_name
    second = directory / unicode_name
    ignored = directory / ignored_name
    write_stream(
        first,
        rewrite_stream_identity(full_trial.events, pod_id="pod-directory-a"),
        pod_id="pod-directory-a",
    )
    write_stream(
        second,
        rewrite_stream_identity(full_trial.events, pod_id="pod-directory-b"),
        pod_id="pod-directory-b",
    )
    ignored.write_bytes(b"not an event stream\n")

    forward = inspect_stream_directory(
        directory,
        filenames=(ascii_name, unicode_name),
    )
    reverse = inspect_stream_directory(
        directory,
        filenames=(name for name in (unicode_name, ascii_name)),
    )

    assert forward.canonical_semantic_json() == reverse.canonical_semantic_json()
    assert forward.stream_count == 2
    assert forward.trial_count == 2
    assert forward.valid
    assert [stream.source_name for stream in forward.streams] == sorted(
        (ascii_name, unicode_name)
    )
    assert ignored_name not in {stream.source_name for stream in forward.streams}


def test_empty_explicit_directory_selection_is_well_formed(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "streams"
    directory.mkdir()

    summary = inspect_stream_directory(directory, filenames=())

    assert summary.run_id is None
    assert summary.stream_count == 0
    assert summary.trial_count == 0
    assert summary.valid
    assert summary.streams == ()
    assert all(count == 0 for count in classification_counts(summary).values())


@pytest.mark.parametrize(
    "filename",
    (
        None,
        "",
        ".",
        "..",
        "nested/events.jsonl",
        "../events.jsonl",
        "/tmp/events.jsonl",
        "é" * ((MAX_STREAM_FILENAME_BYTES // 2) + 1),
    ),
)
def test_directory_rejects_unsafe_explicit_filenames(
    tmp_path: Path, filename: Any
) -> None:
    directory = tmp_path / "streams"
    directory.mkdir()
    with pytest.raises(ReplayPathError, match="unsafe"):
        inspect_stream_directory(directory, filenames=(filename,))


def test_directory_rejects_duplicate_names_bounds_and_string_iterable(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "streams"
    directory.mkdir()
    with pytest.raises(ReplayPathError, match="duplicates"):
        inspect_stream_directory(directory, filenames=("x", "x"))
    with pytest.raises(ReplayPathError, match="maximum"):
        inspect_stream_directory(
            directory,
            filenames=("one", "two"),
            max_streams=1,
        )
    with pytest.raises(TypeError, match="iterable of explicit names"):
        inspect_stream_directory(directory, filenames="events.jsonl")


def test_directory_and_child_symlinks_are_rejected(
    tmp_path: Path, full_trial: fixture_lib.FullTrial
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    target = real / "target.jsonl"
    write_stream(target, full_trial.events)
    child_link = real / "child.jsonl"
    child_link.symlink_to(target)
    directory_link = tmp_path / "directory-link"
    directory_link.symlink_to(real, target_is_directory=True)

    with pytest.raises(ReplayPathError, match="symlink"):
        inspect_stream_directory(real, filenames=(child_link.name,))
    with pytest.raises(ReplayPathError, match="symlink"):
        inspect_stream_directory(directory_link, filenames=(target.name,))


def test_directory_argument_must_exist_and_be_a_directory(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    regular = tmp_path / "regular"
    regular.write_text("not a directory")

    with pytest.raises(ReplayPathError, match="unavailable"):
        inspect_stream_directory(missing, filenames=())
    with pytest.raises(ReplayPathError, match="not a directory"):
        inspect_stream_directory(regular, filenames=())
