"""Permanent append, recovery, lifecycle, and path-safety tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

import interpretability.events.writer as writer_module
from interpretability.events.schema import EventEnvelope, EventPayload, register_payload
from interpretability.events.writer import (
    AppendReceipt,
    DuplicateEventError,
    EventParentError,
    EventSequenceError,
    EventWriter,
    EventWriterError,
    FsyncMode,
    NonCanonicalStreamError,
    OpenTrialsError,
    StreamCorruptionError,
    StreamIdentityError,
    StreamLockedError,
    TrialLifecycleError,
    TruncatedStreamError,
    UnsafeStreamPathError,
    WriterClosedError,
)

RUN_ID = "10000000-0000-4000-8000-000000000001"
OTHER_RUN_ID = "20000000-0000-4000-8000-000000000001"
POD_ID = "pod-writer-tests"
PAYLOAD_VERSION = "writer-tests/1.0.0"


@register_payload("RunStarted", PAYLOAD_VERSION)
class RunStartedPayload(EventPayload):
    value: str


@register_payload("TrialStarted", PAYLOAD_VERSION)
class TrialStartedPayload(EventPayload):
    value: str


@register_payload("ActionCommitted", PAYLOAD_VERSION)
class ActionCommittedPayload(EventPayload):
    value: str


@register_payload("ObservationDelivered", PAYLOAD_VERSION)
class ObservationDeliveredPayload(EventPayload):
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
    "ActionCommitted": ActionCommittedPayload,
    "ObservationDelivered": ObservationDeliveredPayload,
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
    value: str | None = None,
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
        actor_id=None,
        actor_role=None,
        model_call_id=None,
        parent_event_ids=parent_event_ids,
        previous_event_hash=previous_event_hash,
        payload=PAYLOADS[event_type](value=value or event_type),
        payload_schema_version=PAYLOAD_VERSION,
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
    )
    terminal = make_event(
        terminal_type,
        trial_id=trial_id,
        sequence_num=2,
        previous_event_hash=action.content_hash,
        parent_event_ids=(action.event_id,),
    )
    return started, action, terminal


def write_complete_stream(path: Path) -> tuple[EventEnvelope, ...]:
    events = trial_events("trial-complete")
    writer = EventWriter(path, run_id=RUN_ID, pod_id=POD_ID, fsync_mode="never")
    for event in events:
        writer.append(event)
    writer.close()
    return events


def test_canonical_bytes_receipt_offsets_and_immutable_status(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, run_id=RUN_ID, pod_id=POD_ID, fsync_mode="never")
    started, action, completed = trial_events("trial-canonical")

    receipts = [writer.append(event) for event in (started, action, completed)]
    status = writer.status

    expected = b"".join(event.to_json().encode("utf-8") + b"\n" for event in (started, action, completed))
    assert path.read_bytes() == expected
    assert [receipt.line_number for receipt in receipts] == [1, 2, 3]
    assert [receipt.byte_offset for receipt in receipts] == [
        0,
        receipts[0].byte_length,
        receipts[0].byte_length + receipts[1].byte_length,
    ]
    assert sum(receipt.byte_length for receipt in receipts) == len(expected)
    assert status.event_count == 3
    assert status.byte_size == len(expected)
    assert status.open_trials == ()
    assert status.sealed_trials == (("trial-canonical", "TrialCompleted"),)
    assert status.lanes[0].last_sequence_num == 2

    with pytest.raises(FrozenInstanceError):
        receipts[0].line_number = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        status.event_count = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        status.lanes[0].event_count = 99  # type: ignore[misc]
    writer.close()


def test_exact_run_pod_and_resolved_path_identity(tmp_path: Path) -> None:
    requested = tmp_path / "nested" / "events.jsonl"
    writer = EventWriter(requested, run_id=RUN_ID, pod_id=POD_ID)
    assert writer.path == requested.resolve()
    assert writer.status.run_id == RUN_ID
    assert writer.status.pod_id == POD_ID

    with pytest.raises(StreamIdentityError):
        writer.append(
            make_event(
                "TrialStarted",
                trial_id="wrong-run",
                sequence_num=0,
                run_id=OTHER_RUN_ID,
            )
        )
    with pytest.raises(StreamIdentityError):
        writer.append(
            make_event(
                "TrialStarted",
                trial_id="wrong-pod",
                sequence_num=0,
                pod_id="another-pod",
            )
        )
    assert writer.status.event_count == 0
    writer.close()


def test_root_and_interleaved_trial_lanes_have_independent_hash_chains(
    tmp_path: Path,
) -> None:
    writer = EventWriter(
        tmp_path / "events.jsonl", run_id=RUN_ID, pod_id=POD_ID
    )
    run_root = make_event("RunStarted", trial_id=None, sequence_num=0)
    a_started, a_action, a_done = trial_events("trial-a")
    b_started, b_action, b_done = trial_events("trial-b")

    for event in (
        run_root,
        a_started,
        b_started,
        a_action,
        b_action,
        b_done,
        a_done,
    ):
        writer.append(event)

    lanes = {lane.trial_id: lane for lane in writer.status.lanes}
    assert lanes[None].last_event_hash == run_root.content_hash
    assert lanes["trial-a"].last_event_hash == a_done.content_hash
    assert lanes["trial-b"].last_event_hash == b_done.content_hash
    assert lanes[None].event_count == 1
    assert lanes["trial-a"].event_count == 3
    assert lanes["trial-b"].event_count == 3
    writer.close()


def test_sequence_gap_regression_and_wrong_previous_hash_fail(tmp_path: Path) -> None:
    writer = EventWriter(
        tmp_path / "events.jsonl",
        run_id=RUN_ID,
        pod_id=POD_ID,
        allow_incomplete=True,
    )
    started = make_event("TrialStarted", trial_id="trial", sequence_num=0)
    writer.append(started)

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
    with pytest.raises(EventSequenceError, match="expected sequence 1"):
        writer.append(gap)
    with pytest.raises(EventSequenceError, match="previous hash"):
        writer.append(wrong_hash)
    assert writer.status.event_count == 1
    writer.close()


def test_event_ids_are_unique_across_all_lanes(tmp_path: Path) -> None:
    writer = EventWriter(
        tmp_path / "events.jsonl",
        run_id=RUN_ID,
        pod_id=POD_ID,
        allow_incomplete=True,
    )
    shared_id = str(uuid4())
    writer.append(
        make_event(
            "TrialStarted",
            trial_id="trial-a",
            sequence_num=0,
            event_id=shared_id,
        )
    )
    duplicate = make_event(
        "TrialStarted",
        trial_id="trial-b",
        sequence_num=0,
        event_id=shared_id,
    )

    with pytest.raises(DuplicateEventError, match="duplicate event_id"):
        writer.append(duplicate)
    writer.close()


def test_parent_before_child_missing_duplicate_and_self_rejected(
    tmp_path: Path,
) -> None:
    writer = EventWriter(
        tmp_path / "events.jsonl",
        run_id=RUN_ID,
        pod_id=POD_ID,
        allow_incomplete=True,
    )
    missing_id = str(uuid4())
    missing = make_event(
        "TrialStarted",
        trial_id="missing-parent",
        sequence_num=0,
        parent_event_ids=(missing_id,),
    )
    with pytest.raises(EventParentError, match="missing or later"):
        writer.append(missing)

    started = make_event("TrialStarted", trial_id="trial", sequence_num=0)
    writer.append(started)
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
    with pytest.raises(EventParentError, match="duplicate parent"):
        writer.append(duplicate)

    self_parent = make_event(
        "ActionCommitted",
        trial_id="trial",
        sequence_num=1,
        previous_event_hash=started.content_hash,
    )
    object.__setattr__(self_parent, "parent_event_ids", (self_parent.event_id,))
    object.__setattr__(self_parent, "content_hash", self_parent.compute_content_hash())
    with pytest.raises(EventParentError, match="missing or later"):
        writer.append(self_parent)
    writer.close()


def test_external_parents_require_explicit_annotation_allowlist(tmp_path: Path) -> None:
    external_parent = str(uuid4())
    with pytest.raises(ValueError, match="annotation_stream"):
        EventWriter(
            tmp_path / "ordinary.jsonl",
            run_id=RUN_ID,
            pod_id=POD_ID,
            external_parent_ids=(external_parent,),
        )

    writer = EventWriter(
        tmp_path / "annotation.jsonl",
        run_id=RUN_ID,
        pod_id=POD_ID,
        annotation_stream=True,
        external_parent_ids=(external_parent,),
        allow_incomplete=True,
    )
    accepted = make_event(
        "TrialStarted",
        trial_id="annotation",
        sequence_num=0,
        parent_event_ids=(external_parent,),
    )
    writer.append(accepted)

    unlisted = make_event(
        "TrialStarted",
        trial_id="unlisted",
        sequence_num=0,
        parent_event_ids=(str(uuid4()),),
    )
    with pytest.raises(EventParentError):
        writer.append(unlisted)
    reserved_id = make_event(
        "TrialStarted",
        trial_id="reserved",
        sequence_num=0,
        event_id=external_parent,
    )
    with pytest.raises(DuplicateEventError, match="reserved"):
        writer.append(reserved_id)
    writer.close()


def test_trial_lifecycle_start_complete_fail_and_illegal_transitions(
    tmp_path: Path,
) -> None:
    writer = EventWriter(
        tmp_path / "events.jsonl",
        run_id=RUN_ID,
        pod_id=POD_ID,
        allow_incomplete=True,
    )
    before_start = make_event(
        "ActionCommitted", trial_id="before", sequence_num=0
    )
    with pytest.raises(TrialLifecycleError, match="before TrialStarted"):
        writer.append(before_start)

    started, action, completed = trial_events("completed")
    writer.append(started)
    duplicate_start = make_event(
        "TrialStarted",
        trial_id="completed",
        sequence_num=1,
        previous_event_hash=started.content_hash,
    )
    with pytest.raises(TrialLifecycleError, match="already open"):
        writer.append(duplicate_start)
    writer.append(action)
    writer.append(completed)
    after_seal = make_event(
        "ObservationDelivered",
        trial_id="completed",
        sequence_num=3,
        previous_event_hash=completed.content_hash,
    )
    with pytest.raises(TrialLifecycleError, match="already sealed"):
        writer.append(after_seal)

    failed_events = trial_events("failed", terminal_type="TrialFailed")
    for event in failed_events:
        writer.append(event)
    assert dict(writer.status.sealed_trials) == {
        "completed": "TrialCompleted",
        "failed": "TrialFailed",
    }

    boundary_without_trial = make_event(
        "TrialStarted", trial_id=None, sequence_num=0
    )
    with pytest.raises(TrialLifecycleError, match="requires a trial_id"):
        writer.append(boundary_without_trial)
    writer.close()


def test_close_rejects_open_trials_unless_explicitly_allowed(tmp_path: Path) -> None:
    writer = EventWriter(
        tmp_path / "events.jsonl", run_id=RUN_ID, pod_id=POD_ID
    )
    writer.append(make_event("TrialStarted", trial_id="open", sequence_num=0))

    with pytest.raises(OpenTrialsError, match="open trials"):
        writer.close()
    assert not writer.status.closed
    closed = writer.close(allow_incomplete=True)
    assert closed.closed
    assert closed.open_trials == ("open",)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (FsyncMode.ALWAYS, [True, True, True]),
        (FsyncMode.NEVER, [False, False, False]),
        (FsyncMode.TRIAL_BOUNDARY, [True, False, True]),
        ("trial-boundary", [True, False, True]),
    ],
)
def test_fsync_modes_call_fsync_at_exact_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: FsyncMode | str,
    expected: list[bool],
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(writer_module.os, "fsync", lambda descriptor: calls.append(descriptor))
    writer = EventWriter(
        tmp_path / f"{str(mode)}.jsonl",
        run_id=RUN_ID,
        pod_id=POD_ID,
        fsync_mode=mode,
    )
    receipts = [writer.append(event) for event in trial_events("trial")]
    writer.close()

    assert [receipt.fsynced for receipt in receipts] == expected
    assert len(calls) == sum(expected)


def test_trial_failed_is_an_fsync_boundary(tmp_path: Path) -> None:
    writer = EventWriter(
        tmp_path / "events.jsonl",
        run_id=RUN_ID,
        pod_id=POD_ID,
        fsync_mode="trial-boundary",
    )
    receipts = [
        writer.append(event)
        for event in trial_events("failed", terminal_type="TrialFailed")
    ]
    writer.close()
    assert [receipt.fsynced for receipt in receipts] == [True, False, True]


def test_reopen_reconstructs_and_resumes_without_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    started, action, completed = trial_events("resume")
    first = EventWriter(path, run_id=RUN_ID, pod_id=POD_ID)
    first.append(started)
    first.append(action)
    first.close(allow_incomplete=True)

    reopened = EventWriter(path, run_id=RUN_ID, pod_id=POD_ID)
    assert reopened.status.event_count == 2
    assert reopened.status.open_trials == ("resume",)
    with pytest.raises(DuplicateEventError):
        reopened.append(action)
    receipt = reopened.append(completed)
    assert receipt.line_number == 3
    assert reopened.status.open_trials == ()
    reopened.close()

    final = EventWriter(path, run_id=RUN_ID, pod_id=POD_ID)
    assert final.status.event_count == 3
    assert final.status.sealed_trials == (("resume", "TrialCompleted"),)
    final.close()


def test_reopen_rejects_wrong_stream_identity(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    write_complete_stream(path)
    with pytest.raises(StreamCorruptionError, match="illegal event"):
        EventWriter(path, run_id=OTHER_RUN_ID, pod_id=POD_ID)
    with pytest.raises(StreamCorruptionError, match="illegal event"):
        EventWriter(path, run_id=RUN_ID, pod_id="wrong-pod")


def test_truncated_final_and_middle_records_fail_without_repair(tmp_path: Path) -> None:
    valid = tmp_path / "valid.jsonl"
    write_complete_stream(valid)
    raw = valid.read_bytes()

    final = tmp_path / "truncated-final.jsonl"
    final.write_bytes(raw[:-1])
    before = final.read_bytes()
    with pytest.raises(TruncatedStreamError, match="truncated final line"):
        EventWriter(final, run_id=RUN_ID, pod_id=POD_ID)
    assert final.read_bytes() == before

    middle = tmp_path / "truncated-middle.jsonl"
    first_newline = raw.index(b"\n")
    middle.write_bytes(raw[:first_newline] + raw[first_newline + 1 :])
    before = middle.read_bytes()
    with pytest.raises(StreamCorruptionError, match="invalid event"):
        EventWriter(middle, run_id=RUN_ID, pod_id=POD_ID)
    assert middle.read_bytes() == before


@pytest.mark.parametrize("tamper", ["payload", "hash"])
def test_tampered_payload_and_hash_are_rejected(
    tmp_path: Path, tamper: str
) -> None:
    path = tmp_path / f"{tamper}.jsonl"
    write_complete_stream(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if tamper == "payload":
        records[1]["payload"]["value"] = "tampered"
    else:
        records[1]["content_hash"] = "f" * 64
    path.write_text(
        "\n".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StreamCorruptionError, match="invalid event"):
        EventWriter(path, run_id=RUN_ID, pod_id=POD_ID)


@pytest.mark.parametrize("form", ["whitespace", "key-order"])
def test_noncanonical_whitespace_and_key_order_are_rejected(
    tmp_path: Path, form: str
) -> None:
    path = tmp_path / f"{form}.jsonl"
    write_complete_stream(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if form == "whitespace":
        encoded = [json.dumps(record, sort_keys=True) for record in records]
    else:
        encoded = [
            json.dumps(
                dict(reversed(tuple(record.items()))),
                separators=(",", ":"),
            )
            for record in records
        ]
    path.write_text("\n".join(encoded) + "\n", encoding="utf-8")

    with pytest.raises(NonCanonicalStreamError, match="non-canonical"):
        EventWriter(path, run_id=RUN_ID, pod_id=POD_ID)


@pytest.mark.parametrize("extra", [b"\n", b"{}\n"])
def test_blank_and_extra_invalid_lines_are_rejected(
    tmp_path: Path, extra: bytes
) -> None:
    path = tmp_path / "extra.jsonl"
    write_complete_stream(path)
    path.write_bytes(path.read_bytes() + extra)

    with pytest.raises(StreamCorruptionError):
        EventWriter(path, run_id=RUN_ID, pod_id=POD_ID)


def test_symlink_nonregular_parent_symlink_and_path_swap_are_rejected(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.jsonl"
    target.write_bytes(b"")
    stream_link = tmp_path / "stream-link.jsonl"
    stream_link.symlink_to(target)
    with pytest.raises(UnsafeStreamPathError, match="symlink"):
        EventWriter(stream_link, run_id=RUN_ID, pod_id=POD_ID)

    directory_path = tmp_path / "directory.jsonl"
    directory_path.mkdir()
    with pytest.raises(UnsafeStreamPathError, match="not a regular file"):
        EventWriter(directory_path, run_id=RUN_ID, pod_id=POD_ID)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(UnsafeStreamPathError, match="parent"):
        EventWriter(
            linked_parent / "events.jsonl", run_id=RUN_ID, pod_id=POD_ID
        )

    swap_path = tmp_path / "swap.jsonl"
    writer = EventWriter(
        swap_path,
        run_id=RUN_ID,
        pod_id=POD_ID,
        allow_incomplete=True,
    )
    swap_path.rename(tmp_path / "moved.jsonl")
    swap_path.write_bytes(b"")
    with pytest.raises(UnsafeStreamPathError, match="replaced"):
        writer.append(
            make_event("TrialStarted", trial_id="swap", sequence_num=0)
        )
    writer.close()


def test_active_writer_excludes_another_writer_in_process(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = EventWriter(path, run_id=RUN_ID, pod_id=POD_ID)
    with pytest.raises(StreamLockedError, match="another EventWriter"):
        EventWriter(path, run_id=RUN_ID, pod_id=POD_ID)
    first.close()
    EventWriter(path, run_id=RUN_ID, pod_id=POD_ID).close()


@pytest.mark.skipif(writer_module.fcntl is None, reason="POSIX flock unavailable")
def test_file_lock_excludes_writer_in_child_process(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = EventWriter(path, run_id=RUN_ID, pod_id=POD_ID)
    code = "\n".join(
        [
            "from interpretability.events.writer import EventWriter, StreamLockedError",
            f"path = {str(path)!r}",
            "try:",
            f"    EventWriter(path, run_id={RUN_ID!r}, pod_id={POD_ID!r})",
            "except StreamLockedError:",
            "    raise SystemExit(0)",
            "raise SystemExit(3)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    first.close()
    assert result.returncode == 0, result.stderr


def test_concurrent_threads_append_independent_lanes(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, run_id=RUN_ID, pod_id=POD_ID, fsync_mode="never")

    def append_trial(index: int) -> tuple[AppendReceipt, ...]:
        return tuple(
            writer.append(event) for event in trial_events(f"thread-{index:02d}")
        )

    with ThreadPoolExecutor(max_workers=12) as executor:
        receipts = list(executor.map(append_trial, range(30)))

    assert writer.status.event_count == 90
    assert writer.status.open_trials == ()
    assert len(writer.status.sealed_trials) == 30
    assert len({item.event_id for group in receipts for item in group}) == 90
    assert len(path.read_bytes().splitlines()) == 90
    writer.close()
    EventWriter(path, run_id=RUN_ID, pod_id=POD_ID).close()


def test_concurrent_threads_preserve_declared_same_lane_order(tmp_path: Path) -> None:
    writer = EventWriter(
        tmp_path / "same-lane.jsonl", run_id=RUN_ID, pod_id=POD_ID
    )
    events: list[EventEnvelope] = [
        make_event("TrialStarted", trial_id="shared", sequence_num=0)
    ]
    for sequence in range(1, 20):
        previous = events[-1]
        event_type = "TrialCompleted" if sequence == 19 else "ActionCommitted"
        events.append(
            make_event(
                event_type,
                trial_id="shared",
                sequence_num=sequence,
                previous_event_hash=previous.content_hash,
                parent_event_ids=(previous.event_id,),
            )
        )

    gates = [threading.Event() for _ in range(len(events) + 1)]
    gates[0].set()

    def append_at(index: int) -> AppendReceipt:
        assert gates[index].wait(timeout=10)
        receipt = writer.append(events[index])
        gates[index + 1].set()
        return receipt

    with ThreadPoolExecutor(max_workers=20) as executor:
        receipts = list(executor.map(append_at, range(len(events))))

    assert [receipt.sequence_num for receipt in receipts] == list(range(20))
    assert [receipt.line_number for receipt in receipts] == list(range(1, 21))
    assert writer.status.open_trials == ()
    writer.close()


def test_append_write_failure_does_not_commit_memory_or_event_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, run_id=RUN_ID, pod_id=POD_ID)
    before = writer.status

    def fail_write(descriptor: int, value: Any) -> int:
        del descriptor, value
        raise OSError("simulated append failure")

    monkeypatch.setattr(writer_module.os, "write", fail_write)
    with pytest.raises(EventWriterError, match="append failed"):
        writer.append(
            make_event("TrialStarted", trial_id="failed-write", sequence_num=0)
        )

    assert writer.status.event_count == before.event_count == 0
    assert writer.status.byte_size == before.byte_size == 0
    assert path.read_bytes() == b""
    with pytest.raises(WriterClosedError, match="poisoned"):
        writer.append(
            make_event("TrialStarted", trial_id="retry", sequence_num=0)
        )
    writer.close()


def test_context_manager_closes_completed_stream(tmp_path: Path) -> None:
    with EventWriter(
        tmp_path / "events.jsonl", run_id=RUN_ID, pod_id=POD_ID
    ) as writer:
        for event in trial_events("context"):
            writer.append(event)
    assert writer.status.closed
    with pytest.raises(WriterClosedError):
        writer.append(
            make_event("TrialStarted", trial_id="after-context", sequence_num=0)
        )


def test_context_manager_releases_writer_when_body_raises(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    holder: EventWriter | None = None
    with pytest.raises(RuntimeError, match="body failure"):
        with EventWriter(path, run_id=RUN_ID, pod_id=POD_ID) as writer:
            holder = writer
            writer.append(
                make_event("TrialStarted", trial_id="interrupted", sequence_num=0)
            )
            raise RuntimeError("body failure")
    assert holder is not None and holder.status.closed
    reopened = EventWriter(path, run_id=RUN_ID, pod_id=POD_ID)
    assert reopened.status.open_trials == ("interrupted",)
    reopened.close(allow_incomplete=True)


def test_invalid_fsync_mode_and_non_envelope_append_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fsync_mode"):
        EventWriter(
            tmp_path / "bad-mode.jsonl",
            run_id=RUN_ID,
            pod_id=POD_ID,
            fsync_mode="sometimes",
        )
    writer = EventWriter(tmp_path / "events.jsonl", run_id=RUN_ID, pod_id=POD_ID)
    with pytest.raises(TypeError, match="EventEnvelope"):
        writer.append({"event": "not typed"})  # type: ignore[arg-type]
    writer.close()

