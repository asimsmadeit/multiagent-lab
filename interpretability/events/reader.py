"""Streaming validation and filtering for canonical event JSONL streams."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal
from uuid import UUID

from interpretability.events.schema import (
    EventEnvelope,
    EventPayload,
    OpaqueEventPayload,
)

_TRIAL_START = "TrialStarted"
_TRIAL_SEALS = {
    "TrialCompleted": "completed",
    "TrialFailed": "failed",
}


class EventReaderError(RuntimeError):
    """Base class for event reader failures with byte-level diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        path: Path,
        line_number: int | None = None,
        byte_offset: int | None = None,
    ) -> None:
        self.path = path
        self.line_number = line_number
        self.byte_offset = byte_offset
        location = str(path)
        if line_number is not None:
            location += f":{line_number}"
        if byte_offset is not None:
            location += f" (byte {byte_offset})"
        super().__init__(f"{location}: {message}")


class UnsafeEventStreamPathError(EventReaderError):
    """Raised for missing, symlinked, replaced, or non-regular stream paths."""


class StreamMutationError(EventReaderError):
    """Raised when the stream changes while an iterator is active."""


class EventRecordTooLargeError(EventReaderError):
    """Raised before parsing a record beyond the configured byte bound."""


class InvalidEventEncodingError(EventReaderError):
    """Raised for a record that is not strict UTF-8."""


class TruncatedEventRecordError(EventReaderError):
    """Raised when the final or a merged middle record is incomplete."""


class BlankEventRecordError(EventReaderError):
    """Raised for an empty JSONL record."""


class NonCanonicalEventRecordError(EventReaderError):
    """Raised when valid event JSON does not match canonical wire bytes."""


class EventParseError(EventReaderError):
    """Raised when a line cannot be parsed as a strict EventEnvelope."""


class EventIntegrityError(EventReaderError):
    """Raised for identity, hash, duplicate-ID, or ordering violations."""


class EventSequenceValidationError(EventIntegrityError):
    """Raised for sequence gaps, regressions, or incorrect previous hashes."""


class EventParentValidationError(EventIntegrityError):
    """Raised when a parent is duplicate, missing, or after its child."""


class TrialLifecycleValidationError(EventIntegrityError):
    """Raised for events before start, duplicate starts, or events after seal."""


class IncompleteTrialsError(EventIntegrityError):
    """Raised when validation explicitly requires every trial to be sealed."""


@dataclass(frozen=True, slots=True)
class LocatedEvent:
    """An immutable event plus its exact source location."""

    event: EventEnvelope
    line_number: int
    byte_offset: int
    byte_length: int
    projectable: bool


@dataclass(frozen=True, slots=True)
class ReaderFilter:
    """Typed exact-match filters applied only after each event is validated."""

    run_id: str | None = None
    pod_id: str | None = None
    trial_id: str | None = None
    event_type: str | None = None
    actor_id: str | None = None
    model_call_id: str | None = None
    payload_type: type[EventPayload] | tuple[type[EventPayload], ...] | None = None

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "pod_id",
            "trial_id",
            "event_type",
            "actor_id",
            "model_call_id",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"ReaderFilter.{name} must be a non-empty string")
        payload_types = self.payload_type
        if payload_types is None:
            return
        if isinstance(payload_types, type):
            payload_types = (payload_types,)
        if not payload_types or any(
            not isinstance(item, type) or not issubclass(item, EventPayload)
            for item in payload_types
        ):
            raise TypeError(
                "ReaderFilter.payload_type must contain EventPayload subclasses"
            )

    def matches(self, event: EventEnvelope) -> bool:
        if self.run_id is not None and event.run_id != self.run_id:
            return False
        if self.pod_id is not None and event.pod_id != self.pod_id:
            return False
        if self.trial_id is not None and event.trial_id != self.trial_id:
            return False
        if self.event_type is not None and event.event_type != self.event_type:
            return False
        if self.actor_id is not None and event.actor_id != self.actor_id:
            return False
        if self.model_call_id is not None and event.model_call_id != self.model_call_id:
            return False
        payload_types = self.payload_type
        if payload_types is not None:
            if isinstance(payload_types, type):
                payload_types = (payload_types,)
            if not isinstance(event.payload, payload_types):
                return False
        return True


@dataclass(frozen=True, slots=True)
class StreamLaneReport:
    """Validated terminal position for one run or trial lane."""

    trial_id: str | None
    event_count: int
    first_sequence_num: int
    last_sequence_num: int
    last_event_hash: str


@dataclass(frozen=True, slots=True)
class TrialTerminalReport:
    """Lifecycle classification for one trial at end of stream."""

    trial_id: str
    classification: Literal["completed", "failed", "open"]
    started_event_id: str
    terminal_event_id: str | None


@dataclass(frozen=True, slots=True)
class StreamValidationReport:
    """Immutable summary produced after full streaming validation."""

    path: Path
    event_count: int
    byte_size: int
    run_id: str | None
    pod_id: str | None
    lanes: tuple[StreamLaneReport, ...]
    trials: tuple[TrialTerminalReport, ...]
    unprojectable_event_ids: tuple[str, ...]
    strict_canonical: bool

    @property
    def completed_trials(self) -> tuple[str, ...]:
        return tuple(
            trial.trial_id
            for trial in self.trials
            if trial.classification == "completed"
        )

    @property
    def failed_trials(self) -> tuple[str, ...]:
        return tuple(
            trial.trial_id
            for trial in self.trials
            if trial.classification == "failed"
        )

    @property
    def open_trials(self) -> tuple[str, ...]:
        return tuple(
            trial.trial_id
            for trial in self.trials
            if trial.classification == "open"
        )

    @property
    def projectable(self) -> bool:
        return not self.unprojectable_event_ids


@dataclass(slots=True)
class _LaneState:
    first_sequence_num: int
    last_sequence_num: int
    last_event_hash: str
    event_count: int


@dataclass(slots=True)
class _ValidationState:
    event_ids: set[str] = field(default_factory=set)
    lanes: dict[str | None, _LaneState] = field(default_factory=dict)
    open_trials: set[str] = field(default_factory=set)
    started_event_ids: dict[str, str] = field(default_factory=dict)
    sealed_trials: dict[str, tuple[Literal["completed", "failed"], str]] = field(
        default_factory=dict
    )
    unprojectable_event_ids: list[str] = field(default_factory=list)
    run_id: str | None = None
    pod_id: str | None = None
    event_count: int = 0
    byte_size: int = 0


def _canonical_uuid(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a canonical UUID string") from exc
    if str(parsed) != value:
        raise ValueError(f"{field_name} must use lowercase canonical UUID form")
    return value


class EventReader:
    """Re-entrant streaming validator for one immutable event stream path."""

    def __init__(
        self,
        stream_path: str | os.PathLike[str],
        *,
        expected_run_id: str | None = None,
        expected_pod_id: str | None = None,
        strict_canonical: bool = True,
        preserve_unknown_payloads: bool = False,
        annotation_stream: bool = False,
        external_parent_ids: Iterable[str] = (),
        max_record_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if isinstance(max_record_bytes, bool) or not isinstance(max_record_bytes, int):
            raise TypeError("max_record_bytes must be an integer")
        if max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be positive")
        self._expected_run_id = (
            None
            if expected_run_id is None
            else _canonical_uuid(expected_run_id, "expected_run_id")
        )
        if expected_pod_id is not None and (
            not isinstance(expected_pod_id, str) or not expected_pod_id
        ):
            raise ValueError("expected_pod_id must be a non-empty string")
        self._expected_pod_id = expected_pod_id
        self._strict_canonical = bool(strict_canonical)
        self._preserve_unknown_payloads = bool(preserve_unknown_payloads)
        self._annotation_stream = bool(annotation_stream)
        external = frozenset(
            _canonical_uuid(parent, "external_parent_ids")
            for parent in external_parent_ids
        )
        if external and not self._annotation_stream:
            raise ValueError(
                "external_parent_ids require an explicit annotation_stream"
            )
        self._external_parent_ids = external
        self._max_record_bytes = max_record_bytes
        self._path, self._identity = self._prepare_path(stream_path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def max_record_bytes(self) -> int:
        return self._max_record_bytes

    def iter_events(
        self, reader_filter: ReaderFilter | None = None
    ) -> Iterator[LocatedEvent]:
        """Yield matching events while validating every record in the stream."""

        if reader_filter is not None and not isinstance(reader_filter, ReaderFilter):
            raise TypeError("reader_filter must be a ReaderFilter")
        state = _ValidationState()
        for located in self._stream(state):
            if reader_filter is None or reader_filter.matches(located.event):
                yield located

    def validate(
        self, *, require_sealed_trials: bool = False
    ) -> StreamValidationReport:
        """Consume the stream fully and return immutable validation metadata."""

        state = _ValidationState()
        for _ in self._stream(state):
            pass
        if require_sealed_trials and state.open_trials:
            trials = ", ".join(sorted(state.open_trials))
            raise IncompleteTrialsError(
                f"stream ends with open trials: {trials}",
                path=self._path,
                line_number=state.event_count or None,
                byte_offset=state.byte_size,
            )
        return self._build_report(state)

    def _stream(self, state: _ValidationState) -> Iterator[LocatedEvent]:
        descriptor, initial_stat = self._open_checked()
        line_number = 0
        byte_offset = 0
        try:
            with os.fdopen(descriptor, "rb", buffering=0) as stream:
                descriptor = -1
                while True:
                    self._assert_unchanged(stream.fileno(), initial_stat)
                    record = stream.readline(self._max_record_bytes + 2)
                    self._assert_unchanged(stream.fileno(), initial_stat)
                    if not record:
                        break
                    line_number += 1
                    record_offset = byte_offset
                    byte_offset += len(record)
                    if not record.endswith(b"\n"):
                        if len(record) > self._max_record_bytes:
                            raise EventRecordTooLargeError(
                                f"event record exceeds {self._max_record_bytes} bytes",
                                path=self._path,
                                line_number=line_number,
                                byte_offset=record_offset,
                            )
                        raise TruncatedEventRecordError(
                            "final event record is not newline terminated",
                            path=self._path,
                            line_number=line_number,
                            byte_offset=record_offset,
                        )
                    content = record[:-1]
                    if len(content) > self._max_record_bytes:
                        raise EventRecordTooLargeError(
                            f"event record exceeds {self._max_record_bytes} bytes",
                            path=self._path,
                            line_number=line_number,
                            byte_offset=record_offset,
                        )
                    if not content:
                        raise BlankEventRecordError(
                            "blank event record",
                            path=self._path,
                            line_number=line_number,
                            byte_offset=record_offset,
                        )
                    if self._strict_canonical and content.endswith(b"\r"):
                        raise NonCanonicalEventRecordError(
                            "CRLF line endings are not canonical",
                            path=self._path,
                            line_number=line_number,
                            byte_offset=record_offset,
                        )
                    try:
                        text = content.decode("utf-8", errors="strict")
                    except UnicodeDecodeError as exc:
                        raise InvalidEventEncodingError(
                            "event record is not valid UTF-8",
                            path=self._path,
                            line_number=line_number,
                            byte_offset=record_offset,
                        ) from exc
                    try:
                        event = EventEnvelope.from_json(
                            text,
                            preserve_unknown_payloads=self._preserve_unknown_payloads,
                        )
                    except Exception as exc:
                        raise EventParseError(
                            "record is not a valid EventEnvelope",
                            path=self._path,
                            line_number=line_number,
                            byte_offset=record_offset,
                        ) from exc
                    if (
                        self._strict_canonical
                        and event.to_json().encode("utf-8") != content
                    ):
                        raise NonCanonicalEventRecordError(
                            "event record is not canonical compact JSON",
                            path=self._path,
                            line_number=line_number,
                            byte_offset=record_offset,
                        )
                    self._validate_event(
                        event,
                        state,
                        line_number=line_number,
                        byte_offset=record_offset,
                    )
                    self._commit_event(event, state)
                    state.byte_size = byte_offset
                    yield LocatedEvent(
                        event=event,
                        line_number=line_number,
                        byte_offset=record_offset,
                        byte_length=len(record),
                        projectable=not isinstance(event.payload, OpaqueEventPayload),
                    )
                self._assert_unchanged(stream.fileno(), initial_stat)
                state.byte_size = byte_offset
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _validate_event(
        self,
        event: EventEnvelope,
        state: _ValidationState,
        *,
        line_number: int,
        byte_offset: int,
    ) -> None:
        def error(error_type: type[EventReaderError], message: str) -> None:
            raise error_type(
                message,
                path=self._path,
                line_number=line_number,
                byte_offset=byte_offset,
            )

        try:
            event.verify_content_hash()
        except Exception as exc:
            raise EventIntegrityError(
                f"event {event.event_id} content hash mismatch",
                path=self._path,
                line_number=line_number,
                byte_offset=byte_offset,
            ) from exc

        if state.run_id is None:
            state.run_id = event.run_id
            state.pod_id = event.pod_id
        elif event.run_id != state.run_id or event.pod_id != state.pod_id:
            error(
                EventIntegrityError,
                f"event {event.event_id} changes stream run/pod identity",
            )
        if self._expected_run_id is not None and event.run_id != self._expected_run_id:
            error(
                EventIntegrityError,
                f"event run_id {event.run_id} does not match expected "
                f"{self._expected_run_id}",
            )
        if self._expected_pod_id is not None and event.pod_id != self._expected_pod_id:
            error(
                EventIntegrityError,
                f"event pod_id {event.pod_id} does not match expected "
                f"{self._expected_pod_id}",
            )
        if event.event_id in state.event_ids:
            error(EventIntegrityError, f"duplicate event_id {event.event_id}")
        if event.event_id in self._external_parent_ids:
            error(
                EventIntegrityError,
                f"event_id {event.event_id} is reserved as an external parent",
            )

        if len(event.parent_event_ids) != len(set(event.parent_event_ids)):
            error(
                EventParentValidationError,
                f"event {event.event_id} has duplicate parent IDs",
            )
        for parent_id in event.parent_event_ids:
            if parent_id in state.event_ids:
                continue
            if self._annotation_stream and parent_id in self._external_parent_ids:
                continue
            error(
                EventParentValidationError,
                f"event {event.event_id} references missing or later parent "
                f"{parent_id}",
            )

        lane = state.lanes.get(event.trial_id)
        if lane is None:
            if event.sequence_num != 0 or event.previous_event_hash is not None:
                error(
                    EventSequenceValidationError,
                    f"first event in lane {event.trial_id!r} must have sequence 0 "
                    "and no previous hash",
                )
        else:
            expected_sequence = lane.last_sequence_num + 1
            if event.sequence_num != expected_sequence:
                error(
                    EventSequenceValidationError,
                    f"lane {event.trial_id!r} expected sequence "
                    f"{expected_sequence}, got {event.sequence_num}",
                )
            if event.previous_event_hash != lane.last_event_hash:
                error(
                    EventSequenceValidationError,
                    f"lane {event.trial_id!r} previous hash mismatch",
                )

        trial_id = event.trial_id
        if trial_id is None:
            if event.event_type == _TRIAL_START or event.event_type in _TRIAL_SEALS:
                error(
                    TrialLifecycleValidationError,
                    f"{event.event_type} requires a trial_id",
                )
            return
        if trial_id in state.sealed_trials:
            error(
                TrialLifecycleValidationError,
                f"trial {trial_id} is already sealed",
            )
        if event.event_type == _TRIAL_START:
            if trial_id in state.open_trials:
                error(
                    TrialLifecycleValidationError,
                    f"trial {trial_id} is already open",
                )
            return
        if trial_id not in state.open_trials:
            error(
                TrialLifecycleValidationError,
                f"event {event.event_id} occurs before TrialStarted for {trial_id}",
            )

    @staticmethod
    def _commit_event(event: EventEnvelope, state: _ValidationState) -> None:
        state.event_ids.add(event.event_id)
        lane = state.lanes.get(event.trial_id)
        state.lanes[event.trial_id] = _LaneState(
            first_sequence_num=(
                event.sequence_num if lane is None else lane.first_sequence_num
            ),
            last_sequence_num=event.sequence_num,
            last_event_hash=event.content_hash or "",
            event_count=1 if lane is None else lane.event_count + 1,
        )
        state.event_count += 1
        if isinstance(event.payload, OpaqueEventPayload):
            state.unprojectable_event_ids.append(event.event_id)
        if event.trial_id is None:
            return
        if event.event_type == _TRIAL_START:
            state.open_trials.add(event.trial_id)
            state.started_event_ids[event.trial_id] = event.event_id
        elif event.event_type in _TRIAL_SEALS:
            state.open_trials.remove(event.trial_id)
            state.sealed_trials[event.trial_id] = (
                _TRIAL_SEALS[event.event_type],
                event.event_id,
            )

    def _build_report(self, state: _ValidationState) -> StreamValidationReport:
        lanes = tuple(
            StreamLaneReport(
                trial_id=trial_id,
                event_count=lane.event_count,
                first_sequence_num=lane.first_sequence_num,
                last_sequence_num=lane.last_sequence_num,
                last_event_hash=lane.last_event_hash,
            )
            for trial_id, lane in sorted(
                state.lanes.items(),
                key=lambda item: (item[0] is not None, item[0] or ""),
            )
        )
        trials: list[TrialTerminalReport] = []
        for trial_id, started_event_id in sorted(state.started_event_ids.items()):
            terminal = state.sealed_trials.get(trial_id)
            trials.append(
                TrialTerminalReport(
                    trial_id=trial_id,
                    classification="open" if terminal is None else terminal[0],
                    started_event_id=started_event_id,
                    terminal_event_id=None if terminal is None else terminal[1],
                )
            )
        return StreamValidationReport(
            path=self._path,
            event_count=state.event_count,
            byte_size=state.byte_size,
            run_id=state.run_id,
            pod_id=state.pod_id,
            lanes=lanes,
            trials=tuple(trials),
            unprojectable_event_ids=tuple(state.unprojectable_event_ids),
            strict_canonical=self._strict_canonical,
        )

    def _prepare_path(
        self, stream_path: str | os.PathLike[str]
    ) -> tuple[Path, tuple[int, int]]:
        if stream_path is None or str(stream_path) == "":
            raise ValueError("stream_path must be explicit")
        requested = Path(stream_path).expanduser()
        if requested.is_symlink():
            raise UnsafeEventStreamPathError(
                "event stream path must not be a symlink", path=requested
            )
        if requested.parent.is_symlink():
            raise UnsafeEventStreamPathError(
                "event stream parent must not be a symlink", path=requested
            )
        try:
            path = requested.resolve(strict=True)
            path_stat = os.stat(path, follow_symlinks=False)
        except (FileNotFoundError, OSError) as exc:
            raise UnsafeEventStreamPathError(
                "event stream does not exist", path=requested
            ) from exc
        if not stat.S_ISREG(path_stat.st_mode):
            raise UnsafeEventStreamPathError(
                "event stream is not a regular file", path=path
            )
        return path, (path_stat.st_dev, path_stat.st_ino)

    def _open_checked(self) -> tuple[int, os.stat_result]:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._path, flags)
        except OSError as exc:
            raise UnsafeEventStreamPathError(
                "cannot safely open event stream", path=self._path
            ) from exc
        descriptor_stat = os.fstat(descriptor)
        try:
            path_stat = os.stat(self._path, follow_symlinks=False)
        except OSError as exc:
            os.close(descriptor)
            raise UnsafeEventStreamPathError(
                "event stream path disappeared", path=self._path
            ) from exc
        descriptor_identity = (descriptor_stat.st_dev, descriptor_stat.st_ino)
        if (
            not stat.S_ISREG(descriptor_stat.st_mode)
            or descriptor_identity != self._identity
            or descriptor_identity != (path_stat.st_dev, path_stat.st_ino)
        ):
            os.close(descriptor)
            raise UnsafeEventStreamPathError(
                "event stream path was replaced", path=self._path
            )
        return descriptor, descriptor_stat

    def _assert_unchanged(
        self, descriptor: int, initial_stat: os.stat_result
    ) -> None:
        current = os.fstat(descriptor)
        try:
            path_stat = os.stat(self._path, follow_symlinks=False)
        except OSError as exc:
            raise StreamMutationError(
                "event stream path disappeared during iteration", path=self._path
            ) from exc
        if (
            (current.st_dev, current.st_ino) != self._identity
            or (path_stat.st_dev, path_stat.st_ino) != self._identity
            or current.st_size != initial_stat.st_size
            or current.st_mtime_ns != initial_stat.st_mtime_ns
            or current.st_ctime_ns != initial_stat.st_ctime_ns
        ):
            raise StreamMutationError(
                "event stream changed during iteration", path=self._path
            )


__all__ = [
    "BlankEventRecordError",
    "EventIntegrityError",
    "EventParentValidationError",
    "EventParseError",
    "EventReader",
    "EventReaderError",
    "EventRecordTooLargeError",
    "EventSequenceValidationError",
    "IncompleteTrialsError",
    "InvalidEventEncodingError",
    "LocatedEvent",
    "NonCanonicalEventRecordError",
    "ReaderFilter",
    "StreamLaneReport",
    "StreamMutationError",
    "StreamValidationReport",
    "TrialLifecycleValidationError",
    "TrialTerminalReport",
    "TruncatedEventRecordError",
    "UnsafeEventStreamPathError",
]
