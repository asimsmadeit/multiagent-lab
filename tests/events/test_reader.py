"""Permanent streaming, integrity, filtering, and diagnostics tests."""

from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pytest
from pydantic import ValidationError

import interpretability.events.reader as reader_module
from interpretability.events.reader import (
    BlankEventRecordError,
    EventIntegrityError,
    EventParentValidationError,
    EventParseError,
    EventReader,
    EventRecordTooLargeError,
    EventSequenceValidationError,
    IncompleteTrialsError,
    InvalidEventEncodingError,
    NonCanonicalEventRecordError,
    ReaderFilter,
    StreamMutationError,
    TrialLifecycleValidationError,
    TruncatedEventRecordError,
    UnsafeEventStreamPathError,
)
from interpretability.events.schema import (
    EventEnvelope,
    EventPayload,
    OpaqueEventPayload,
    register_payload,
)
from interpretability.events.writer import EventWriter

RUN_ID = "30000000-0000-4000-8000-000000000001"
OTHER_RUN_ID = "40000000-0000-4000-8000-000000000001"
POD_ID = "pod-reader-tests"
PAYLOAD_VERSION = "reader-tests/1.0.0"


@register_payload("RunStarted", PAYLOAD_VERSION)
class RunStartedPayload(EventPayload):
    value: str


@register_payload("TrialStarted", PAYLOAD_VERSION)
class TrialStartedPayload(EventPayload):
    value: str


@register_payload("ObservationDelivered", PAYLOAD_VERSION)
class ObservationDeliveredPayload(EventPayload):
    value: str


@register_payload("ActionCommitted", PAYLOAD_VERSION)
class ActionCommittedPayload(EventPayload):
    value: str


@register_payload("TrialCompleted", PAYLOAD_VERSION)
class TrialCompletedPayload(EventPayload):
    value: str


@register_payload("TrialFailed", PAYLOAD_VERSION)
class TrialFailedPayload(EventPayload):
    value: str


PAYLOADS: dict[str, type[EventPayload]] = {
    "RunStarted": RunStartedPayload,
    "TrialStarted": TrialStartedPayload,
    "ObservationDelivered": ObservationDeliveredPayload,
    "ActionCommitted": ActionCommittedPayload,
    "TrialCompleted": TrialCompletedPayload,
    "TrialFailed": TrialFailedPayload,
}


def make_event(
    event_type: str,
    *,
    trial_id: str | None,
    sequence_num: int,
    previous_event_hash: str | None = None,
    parent_event_ids: tuple[str, ...] = (),
    event_id: str | None = None,
    run_id: str = RUN_ID,
    pod_id: str = POD_ID,
    actor: bool = False,
    payload: EventPayload | None = None,
    payload_schema_version: str = PAYLOAD_VERSION,
) -> EventEnvelope:
    return EventEnvelope(
        event_id=event_id or str(uuid4()),
        event_type=event_type,
        run_id=run_id,
        pod_id=pod_id,
        trial_id=trial_id,
        dyad_id=None if trial_id is None else f"dyad-{trial_id}",
        sequence_num=sequence_num,
        recorded_at=datetime(2026, 7, 13, 12, sequence_num, tzinfo=timezone.utc),
        actor_id="agent-a" if actor else None,
        actor_role="seller" if actor else None,
        model_call_id="call-a" if actor else None,
        parent_event_ids=parent_event_ids,
        previous_event_hash=previous_event_hash,
        payload=payload or PAYLOADS[event_type](value=event_type),
        payload_schema_version=payload_schema_version,
    )


def trial_events(
    trial_id: str,
    *,
    terminal_type: str = "TrialCompleted",
) -> tuple[EventEnvelope, EventEnvelope, EventEnvelope]:
    started = make_event("TrialStarted", trial_id=trial_id, sequence_num=0)
    action = make_event(
        "ActionCommitted",
        trial_id=trial_id,
        sequence_num=1,
        previous_event_hash=started.content_hash,
        parent_event_ids=(started.event_id,),
        actor=True,
    )
    terminal = make_event(
        terminal_type,
        trial_id=trial_id,
        sequence_num=2,
        previous_event_hash=action.content_hash,
        parent_event_ids=(action.event_id,),
    )
    return started, action, terminal


def write_envelopes(path: Path, events: tuple[EventEnvelope, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(event.to_json().encode("utf-8") + b"\n" for event in events)
    )


def writer_stream(
    path: Path,
    events: tuple[EventEnvelope, ...],
    *,
    allow_incomplete: bool = False,
) -> None:
    writer = EventWriter(
        path,
        run_id=RUN_ID,
        pod_id=POD_ID,
        fsync_mode="never",
        allow_incomplete=allow_incomplete,
    )
    for event in events:
        writer.append(event)
    writer.close()


def test_reader_is_reentrant_and_reports_exact_line_and_byte_locations(
    tmp_path: Path,
) -> None:
    events = trial_events("locations")
    path = tmp_path / "events.jsonl"
    writer_stream(path, events)
    reader = EventReader(path)

    first = list(reader.iter_events())
    second = list(reader.iter_events())
    expected_offsets: list[int] = []
    offset = 0
    for event in events:
        expected_offsets.append(offset)
        offset += len(event.to_json().encode("utf-8")) + 1

    assert [item.event for item in first] == list(events)
    assert [item.event for item in second] == list(events)
    assert first is not second
    assert [item.line_number for item in first] == [1, 2, 3]
    assert [item.byte_offset for item in first] == expected_offsets
    assert [item.byte_length for item in first] == [
        len(event.to_json().encode("utf-8")) + 1 for event in events
    ]
    assert all(item.projectable for item in first)


def test_every_filter_dimension_and_combination(tmp_path: Path) -> None:
    root = make_event("RunStarted", trial_id=None, sequence_num=0)
    a = trial_events("trial-a")
    b = trial_events("trial-b")
    events = (root, a[0], b[0], a[1], b[1], a[2], b[2])
    path = tmp_path / "events.jsonl"
    writer_stream(path, events)
    reader = EventReader(path)

    cases: list[tuple[ReaderFilter, list[str]]] = [
        (ReaderFilter(run_id=RUN_ID), [event.event_id for event in events]),
        (ReaderFilter(pod_id=POD_ID), [event.event_id for event in events]),
        (ReaderFilter(trial_id="trial-a"), [event.event_id for event in a]),
        (
            ReaderFilter(event_type="ActionCommitted"),
            [a[1].event_id, b[1].event_id],
        ),
        (ReaderFilter(actor_id="agent-a"), [a[1].event_id, b[1].event_id]),
        (ReaderFilter(model_call_id="call-a"), [a[1].event_id, b[1].event_id]),
        (
            ReaderFilter(payload_type=ActionCommittedPayload),
            [a[1].event_id, b[1].event_id],
        ),
        (
            ReaderFilter(
                run_id=RUN_ID,
                pod_id=POD_ID,
                trial_id="trial-a",
                event_type="ActionCommitted",
                actor_id="agent-a",
                model_call_id="call-a",
                payload_type=(ActionCommittedPayload,),
            ),
            [a[1].event_id],
        ),
        (ReaderFilter(run_id=OTHER_RUN_ID), []),
    ]
    for reader_filter, expected_ids in cases:
        assert [
            item.event.event_id for item in reader.iter_events(reader_filter)
        ] == expected_ids


def test_filtering_does_not_hide_later_corruption(tmp_path: Path) -> None:
    events = trial_events("filtered-corruption")
    path = tmp_path / "events.jsonl"
    writer_stream(path, events)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[-1]["payload"]["value"] = "tampered"
    path.write_text(
        "\n".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            for record in records
        )
        + "\n"
    )

    filtered = ReaderFilter(event_type="TrialStarted")
    with pytest.raises(EventParseError) as exc_info:
        list(EventReader(path).iter_events(filtered))
    assert exc_info.value.line_number == 3


def test_validation_report_lanes_terminal_states_and_unprojectable_events(
    tmp_path: Path,
) -> None:
    root = make_event("RunStarted", trial_id=None, sequence_num=0)
    completed = trial_events("completed")
    failed = trial_events("failed", terminal_type="TrialFailed")
    opened = make_event("TrialStarted", trial_id="open", sequence_num=0)
    opaque = OpaqueEventPayload.from_payload_dict(
        "FutureAnnotation", "2.0.0", {"new": [1, 2]}
    )
    future = make_event(
        "FutureAnnotation",
        trial_id="open",
        sequence_num=1,
        previous_event_hash=opened.content_hash,
        parent_event_ids=(opened.event_id,),
        payload=opaque,
        payload_schema_version="2.0.0",
    )
    events = (root, *completed, *failed, opened, future)
    path = tmp_path / "events.jsonl"
    writer_stream(path, events, allow_incomplete=True)

    reader = EventReader(path, preserve_unknown_payloads=True)
    report = reader.validate()
    lanes = {lane.trial_id: lane for lane in report.lanes}

    assert report.event_count == 9
    assert report.byte_size == path.stat().st_size
    assert report.run_id == RUN_ID and report.pod_id == POD_ID
    assert report.completed_trials == ("completed",)
    assert report.failed_trials == ("failed",)
    assert report.open_trials == ("open",)
    assert report.unprojectable_event_ids == (future.event_id,)
    assert not report.projectable
    assert lanes[None].event_count == 1
    assert lanes["completed"].event_count == 3
    assert lanes["failed"].event_count == 3
    assert lanes["open"].event_count == 2
    with pytest.raises(IncompleteTrialsError):
        reader.validate(require_sealed_trials=True)


def test_unknown_payload_is_strict_by_default_and_opaque_only_by_opt_in(
    tmp_path: Path,
) -> None:
    started = make_event("TrialStarted", trial_id="future", sequence_num=0)
    opaque = OpaqueEventPayload.from_payload_dict(
        "FutureEvent", "9.0.0", {"field": "preserved"}
    )
    future = make_event(
        "FutureEvent",
        trial_id="future",
        sequence_num=1,
        previous_event_hash=started.content_hash,
        parent_event_ids=(started.event_id,),
        payload=opaque,
        payload_schema_version="9.0.0",
    )
    path = tmp_path / "events.jsonl"
    writer_stream(path, (started, future), allow_incomplete=True)

    with pytest.raises(EventParseError):
        list(EventReader(path).iter_events())
    located = list(
        EventReader(path, preserve_unknown_payloads=True).iter_events()
    )
    assert isinstance(located[1].event.payload, OpaqueEventPayload)
    assert not located[1].projectable
    assert located[1].event.to_json() == future.to_json()


def test_run_pod_identity_and_expected_identity_are_enforced(tmp_path: Path) -> None:
    first = make_event("TrialStarted", trial_id="one", sequence_num=0)
    changed_run = make_event(
        "TrialStarted",
        trial_id="two",
        sequence_num=0,
        run_id=OTHER_RUN_ID,
    )
    changed_pod = make_event(
        "TrialStarted",
        trial_id="two",
        sequence_num=0,
        pod_id="other-pod",
    )
    for name, event in (("run", changed_run), ("pod", changed_pod)):
        path = tmp_path / f"{name}.jsonl"
        write_envelopes(path, (first, event))
        with pytest.raises(EventIntegrityError, match="run/pod"):
            EventReader(path).validate()

    path = tmp_path / "expected.jsonl"
    write_envelopes(path, (first,))
    with pytest.raises(EventIntegrityError, match="expected"):
        EventReader(path, expected_run_id=OTHER_RUN_ID).validate()
    with pytest.raises(EventIntegrityError, match="expected"):
        EventReader(path, expected_pod_id="other-pod").validate()


def test_duplicate_event_ids_are_rejected_globally(tmp_path: Path) -> None:
    event_id = str(uuid4())
    first = make_event(
        "TrialStarted", trial_id="one", sequence_num=0, event_id=event_id
    )
    second = make_event(
        "TrialStarted", trial_id="two", sequence_num=0, event_id=event_id
    )
    path = tmp_path / "events.jsonl"
    write_envelopes(path, (first, second))

    with pytest.raises(EventIntegrityError, match="duplicate event_id"):
        EventReader(path).validate()


def test_sequence_gap_and_previous_hash_mismatch_are_rejected(tmp_path: Path) -> None:
    started = make_event("TrialStarted", trial_id="trial", sequence_num=0)
    gap = make_event(
        "ActionCommitted",
        trial_id="trial",
        sequence_num=2,
        previous_event_hash=started.content_hash,
    )
    wrong_hash = make_event(
        "ActionCommitted",
        trial_id="trial",
        sequence_num=1,
        previous_event_hash="f" * 64,
    )
    for name, event, message in (
        ("gap", gap, "expected sequence 1"),
        ("hash", wrong_hash, "previous hash mismatch"),
    ):
        path = tmp_path / f"{name}.jsonl"
        write_envelopes(path, (started, event))
        with pytest.raises(EventSequenceValidationError, match=message):
            EventReader(path).validate()


def test_tampered_content_hash_and_payload_fail_during_parse(tmp_path: Path) -> None:
    events = trial_events("tamper")
    for field in ("content_hash", "payload"):
        path = tmp_path / f"{field}.jsonl"
        write_envelopes(path, events)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if field == "content_hash":
            records[1]["content_hash"] = "f" * 64
        else:
            records[1]["payload"]["value"] = "changed"
        path.write_text(
            "\n".join(
                json.dumps(record, sort_keys=True, separators=(",", ":"))
                for record in records
            )
            + "\n"
        )
        with pytest.raises(EventParseError):
            EventReader(path).validate()


def test_parent_before_child_missing_duplicate_and_self_are_rejected(
    tmp_path: Path,
) -> None:
    started = make_event("TrialStarted", trial_id="trial", sequence_num=0)
    missing = make_event(
        "ActionCommitted",
        trial_id="trial",
        sequence_num=1,
        previous_event_hash=started.content_hash,
        parent_event_ids=(str(uuid4()),),
    )
    path = tmp_path / "missing.jsonl"
    write_envelopes(path, (started, missing))
    with pytest.raises(EventParentValidationError, match="missing or later"):
        EventReader(path).validate()

    duplicate = make_event(
        "ActionCommitted",
        trial_id="trial",
        sequence_num=1,
        previous_event_hash=started.content_hash,
        parent_event_ids=(started.event_id,),
    )
    object.__setattr__(
        duplicate, "parent_event_ids", (started.event_id, started.event_id)
    )
    object.__setattr__(duplicate, "content_hash", duplicate.compute_content_hash())
    path = tmp_path / "duplicate.jsonl"
    write_envelopes(path, (started, duplicate))
    # The envelope schema rejects duplicate parents before reader ordering state.
    with pytest.raises(EventParseError):
        EventReader(path).validate()

    self_parent = make_event(
        "ActionCommitted",
        trial_id="trial",
        sequence_num=1,
        previous_event_hash=started.content_hash,
    )
    object.__setattr__(self_parent, "parent_event_ids", (self_parent.event_id,))
    object.__setattr__(self_parent, "content_hash", self_parent.compute_content_hash())
    path = tmp_path / "self.jsonl"
    write_envelopes(path, (started, self_parent))
    # Self-parenting likewise fails the stricter envelope parse boundary first.
    with pytest.raises(EventParseError):
        EventReader(path).validate()


def test_annotation_external_parent_allowlist_is_explicit_and_narrow(
    tmp_path: Path,
) -> None:
    external = str(uuid4())
    annotated = make_event(
        "TrialStarted",
        trial_id="annotation",
        sequence_num=0,
        parent_event_ids=(external,),
    )
    path = tmp_path / "events.jsonl"
    write_envelopes(path, (annotated,))

    with pytest.raises(EventParentValidationError):
        EventReader(path).validate()
    report = EventReader(
        path,
        annotation_stream=True,
        external_parent_ids=(external,),
    ).validate()
    assert report.event_count == 1

    reserved = make_event(
        "TrialStarted", trial_id="reserved", sequence_num=0, event_id=external
    )
    write_envelopes(path, (reserved,))
    with pytest.raises(EventIntegrityError, match="reserved"):
        EventReader(
            path,
            annotation_stream=True,
            external_parent_ids=(external,),
        ).validate()


def test_trial_lifecycle_illegal_transitions_are_rejected(tmp_path: Path) -> None:
    before = make_event("ActionCommitted", trial_id="before", sequence_num=0)
    path = tmp_path / "before.jsonl"
    write_envelopes(path, (before,))
    with pytest.raises(TrialLifecycleValidationError, match="before TrialStarted"):
        EventReader(path).validate()

    started = make_event("TrialStarted", trial_id="trial", sequence_num=0)
    duplicate = make_event(
        "TrialStarted",
        trial_id="trial",
        sequence_num=1,
        previous_event_hash=started.content_hash,
    )
    path = tmp_path / "duplicate.jsonl"
    write_envelopes(path, (started, duplicate))
    with pytest.raises(TrialLifecycleValidationError, match="already open"):
        EventReader(path).validate()

    started, action, completed = trial_events("sealed")
    after = make_event(
        "ObservationDelivered",
        trial_id="sealed",
        sequence_num=3,
        previous_event_hash=completed.content_hash,
    )
    path = tmp_path / "after.jsonl"
    write_envelopes(path, (started, action, completed, after))
    with pytest.raises(TrialLifecycleValidationError, match="already sealed"):
        EventReader(path).validate()

    no_trial = make_event("TrialStarted", trial_id=None, sequence_num=0)
    path = tmp_path / "no-trial.jsonl"
    write_envelopes(path, (no_trial,))
    with pytest.raises(TrialLifecycleValidationError, match="requires a trial_id"):
        EventReader(path).validate()


def test_final_truncation_and_merged_middle_records_have_locations(
    tmp_path: Path,
) -> None:
    events = trial_events("truncated")
    valid = tmp_path / "valid.jsonl"
    write_envelopes(valid, events)
    raw = valid.read_bytes()

    final = tmp_path / "final.jsonl"
    final.write_bytes(raw[:-1])
    with pytest.raises(TruncatedEventRecordError) as exc_info:
        EventReader(final).validate()
    assert exc_info.value.line_number == 3
    assert exc_info.value.byte_offset is not None

    first_newline = raw.index(b"\n")
    middle = tmp_path / "middle.jsonl"
    middle.write_bytes(raw[:first_newline] + raw[first_newline + 1 :])
    with pytest.raises(EventParseError) as exc_info:
        EventReader(middle).validate()
    assert exc_info.value.line_number == 1
    assert exc_info.value.byte_offset == 0


@pytest.mark.parametrize(
    ("kind", "expected_error"),
    [
        ("blank", BlankEventRecordError),
        ("extra", EventParseError),
        ("crlf", NonCanonicalEventRecordError),
        ("whitespace", NonCanonicalEventRecordError),
        ("key-order", NonCanonicalEventRecordError),
        ("utf8", InvalidEventEncodingError),
    ],
)
def test_malformed_and_noncanonical_records_are_rejected(
    tmp_path: Path,
    kind: str,
    expected_error: type[Exception],
) -> None:
    event = make_event("TrialStarted", trial_id="record", sequence_num=0)
    canonical = event.to_json().encode("utf-8") + b"\n"
    path = tmp_path / f"{kind}.jsonl"
    if kind == "blank":
        path.write_bytes(b"\n")
    elif kind == "extra":
        path.write_bytes(canonical + b"{}\n")
    elif kind == "crlf":
        path.write_bytes(canonical.replace(b"\n", b"\r\n"))
    elif kind == "whitespace":
        path.write_text(json.dumps(json.loads(event.to_json()), sort_keys=True) + "\n")
    elif kind == "key-order":
        value = json.loads(event.to_json())
        path.write_text(
            json.dumps(dict(reversed(tuple(value.items()))), separators=(",", ":"))
            + "\n"
        )
    else:
        path.write_bytes(b"\xff\n")

    with pytest.raises(expected_error):
        EventReader(path).validate()


@pytest.mark.parametrize("kind", ["crlf", "whitespace", "key-order"])
def test_noncanonical_text_can_be_read_only_with_explicit_opt_out(
    tmp_path: Path, kind: str
) -> None:
    event = make_event("TrialStarted", trial_id="record", sequence_num=0)
    value = json.loads(event.to_json())
    path = tmp_path / f"{kind}.jsonl"
    if kind == "crlf":
        path.write_bytes(event.to_json().encode() + b"\r\n")
    elif kind == "whitespace":
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
    else:
        path.write_text(
            json.dumps(dict(reversed(tuple(value.items()))), separators=(",", ":"))
            + "\n"
        )
    assert EventReader(path, strict_canonical=False).validate().event_count == 1


def test_oversized_record_is_rejected_before_parse(tmp_path: Path) -> None:
    event = make_event("TrialStarted", trial_id="large", sequence_num=0)
    path = tmp_path / "events.jsonl"
    write_envelopes(path, (event,))

    with pytest.raises(EventRecordTooLargeError) as exc_info:
        EventReader(path, max_record_bytes=32).validate()
    assert exc_info.value.line_number == 1
    assert exc_info.value.byte_offset == 0


def test_symlink_nonregular_and_replaced_paths_are_rejected(tmp_path: Path) -> None:
    event = make_event("TrialStarted", trial_id="path", sequence_num=0)
    target = tmp_path / "target.jsonl"
    write_envelopes(target, (event,))
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)
    with pytest.raises(UnsafeEventStreamPathError, match="symlink"):
        EventReader(link)

    directory = tmp_path / "directory.jsonl"
    directory.mkdir()
    with pytest.raises(UnsafeEventStreamPathError, match="regular file"):
        EventReader(directory)

    reader = EventReader(target)
    target.rename(tmp_path / "old.jsonl")
    write_envelopes(target, (event,))
    with pytest.raises(UnsafeEventStreamPathError, match="replaced"):
        reader.validate()


@pytest.mark.parametrize("mutation", ["append", "truncate", "overwrite"])
def test_live_file_mutation_during_iteration_is_detected(
    tmp_path: Path, mutation: str
) -> None:
    events = trial_events("mutation")
    path = tmp_path / f"{mutation}.jsonl"
    write_envelopes(path, events)
    iterator = EventReader(path).iter_events()
    next(iterator)

    if mutation == "append":
        with path.open("ab") as stream:
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    elif mutation == "truncate":
        with path.open("r+b") as stream:
            stream.truncate(path.stat().st_size - 4)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        with path.open("r+b") as stream:
            stream.seek(0)
            first = stream.read(1)
            stream.seek(0)
            stream.write(b"{" if first != b"{" else b"[")
            stream.flush()
            os.fsync(stream.fileno())

    with pytest.raises(StreamMutationError, match="changed during iteration"):
        next(iterator)


def test_located_events_reports_and_filters_are_immutable(tmp_path: Path) -> None:
    events = trial_events("immutable")
    path = tmp_path / "events.jsonl"
    writer_stream(path, events)
    reader_filter = ReaderFilter(event_type="ActionCommitted")
    located = list(EventReader(path).iter_events(reader_filter))[0]
    report = EventReader(path).validate()

    with pytest.raises(FrozenInstanceError):
        located.line_number = 9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.event_count = 9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.lanes[0].event_count = 9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.trials[0].classification = "open"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        reader_filter.event_type = "TrialStarted"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        located.event.sequence_num = 9  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_run_id": "not-a-uuid"},
        {"expected_pod_id": ""},
        {"max_record_bytes": 0},
        {"max_record_bytes": True},
        {"external_parent_ids": ("50000000-0000-4000-8000-000000000001",)},
        {
            "annotation_stream": True,
            "external_parent_ids": ("not-a-uuid",),
        },
    ],
)
def test_reader_configuration_is_strict(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"")
    with pytest.raises((TypeError, ValueError)):
        EventReader(path, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_id": ""},
        {"pod_id": 3},
        {"trial_id": ""},
        {"event_type": ""},
        {"actor_id": ""},
        {"model_call_id": ""},
        {"payload_type": str},
        {"payload_type": ()},
    ],
)
def test_reader_filter_configuration_is_strict(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ReaderFilter(**kwargs)


def test_reader_uses_bounded_readline_and_never_whole_file_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = trial_events("streaming")
    path = tmp_path / "events.jsonl"
    writer_stream(path, events)
    original_fdopen = reader_module.os.fdopen
    readline_limits: list[int] = []

    class BoundedStream:
        def __init__(self, stream: Any) -> None:
            self._stream = stream

        def __enter__(self) -> BoundedStream:
            return self

        def __exit__(self, *args: Any) -> Any:
            return self._stream.__exit__(*args)

        def fileno(self) -> int:
            return self._stream.fileno()

        def readline(self, size: int = -1) -> bytes:
            assert size > 0
            readline_limits.append(size)
            return self._stream.readline(size)

    def bounded_fdopen(*args: Any, **kwargs: Any) -> BoundedStream:
        return BoundedStream(original_fdopen(*args, **kwargs))

    monkeypatch.setattr(reader_module.os, "fdopen", bounded_fdopen)
    reader = EventReader(path, max_record_bytes=4096)
    assert len(list(reader.iter_events())) == 3
    assert readline_limits == [4098, 4098, 4098, 4098]


def test_writer_and_reader_agree_on_counts_lanes_hashes_and_terminal_states(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    root = make_event("RunStarted", trial_id=None, sequence_num=0)
    a = trial_events("trial-a")
    b = trial_events("trial-b", terminal_type="TrialFailed")
    events = (root, a[0], b[0], a[1], b[1], a[2], b[2])
    writer = EventWriter(path, run_id=RUN_ID, pod_id=POD_ID, fsync_mode="never")
    for event in events:
        writer.append(event)
    writer_status = writer.status
    writer.close()

    report = EventReader(
        path, expected_run_id=RUN_ID, expected_pod_id=POD_ID
    ).validate(require_sealed_trials=True)
    writer_lanes = {lane.trial_id: lane for lane in writer_status.lanes}
    reader_lanes = {lane.trial_id: lane for lane in report.lanes}

    assert report.event_count == writer_status.event_count == len(events)
    assert report.byte_size == writer_status.byte_size == path.stat().st_size
    assert set(reader_lanes) == set(writer_lanes)
    for trial_id in reader_lanes:
        assert reader_lanes[trial_id].event_count == writer_lanes[trial_id].event_count
        assert (
            reader_lanes[trial_id].last_sequence_num
            == writer_lanes[trial_id].last_sequence_num
        )
        assert (
            reader_lanes[trial_id].last_event_hash
            == writer_lanes[trial_id].last_event_hash
        )
    assert report.completed_trials == ("trial-a",)
    assert report.failed_trials == ("trial-b",)
