"""Append-only, integrity-checking JSONL event stream writer.

The writer treats sequence numbers and hash links as the scientific ordering
authority.  Wall-clock timestamps are serialized as envelope provenance only.
Existing streams are replayed through the same validation rules before append;
they are never repaired or truncated implicitly.
"""

from __future__ import annotations

import errno
import os
import stat
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable
from uuid import UUID

from interpretability.events.schema import EventEnvelope

try:  # POSIX provides the cross-process exclusivity used in production.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback remains process-local.
    fcntl = None  # type: ignore[assignment]


class EventWriterError(RuntimeError):
    """Base class for event-stream writer failures."""


class StreamLockedError(EventWriterError):
    """Raised when another writer owns the same stream."""


class UnsafeStreamPathError(EventWriterError):
    """Raised for symlinks, non-regular files, or path replacement."""


class StreamCorruptionError(EventWriterError):
    """Raised when an existing JSONL stream cannot be validated exactly."""


class TruncatedStreamError(StreamCorruptionError):
    """Raised when the final JSONL record lacks its terminating newline."""


class NonCanonicalStreamError(StreamCorruptionError):
    """Raised when an existing event line is valid but not canonical JSON."""


class StreamIdentityError(EventWriterError):
    """Raised when an event belongs to another run or pod."""


class DuplicateEventError(EventWriterError):
    """Raised when an event ID already exists in the stream."""


class EventSequenceError(EventWriterError):
    """Raised for a sequence gap, regression, or incorrect hash link."""


class EventParentError(EventWriterError):
    """Raised for duplicate, missing, or out-of-order event parents."""


class TrialLifecycleError(EventWriterError):
    """Raised when an event violates trial open/seal transitions."""


class OpenTrialsError(EventWriterError):
    """Raised when strict close is requested while trials remain open."""


class WriterClosedError(EventWriterError):
    """Raised when append is attempted after close or an I/O failure."""


class FsyncMode(str, Enum):
    """Durability policy applied after each complete event line."""

    ALWAYS = "always"
    NEVER = "never"
    TRIAL_BOUNDARY = "trial-boundary"


@dataclass(frozen=True, slots=True)
class AppendReceipt:
    """Immutable location and durability receipt for one appended event."""

    event_id: str
    content_hash: str
    line_number: int
    byte_offset: int
    byte_length: int
    fsynced: bool
    trial_id: str | None
    sequence_num: int


@dataclass(frozen=True, slots=True)
class LaneStatus:
    """Last committed position in one run or trial lane."""

    trial_id: str | None
    last_sequence_num: int
    last_event_hash: str
    event_count: int


@dataclass(frozen=True, slots=True)
class WriterStatus:
    """Immutable snapshot of reconstructed or newly appended stream state."""

    path: Path
    run_id: str
    pod_id: str
    event_count: int
    byte_size: int
    open_trials: tuple[str, ...]
    sealed_trials: tuple[tuple[str, str], ...]
    lanes: tuple[LaneStatus, ...]
    closed: bool


@dataclass(frozen=True, slots=True)
class _LaneState:
    last_sequence_num: int
    last_event_hash: str
    event_count: int


_ACTIVE_PATHS: set[Path] = set()
_ACTIVE_PATHS_LOCK = threading.RLock()

_TRIAL_START = "TrialStarted"
_TRIAL_SEALS = frozenset({"TrialCompleted", "TrialFailed"})
_TRIAL_BOUNDARIES = frozenset({_TRIAL_START, *_TRIAL_SEALS})


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


class EventWriter:
    """Validate and append canonical :class:`EventEnvelope` JSONL records."""

    def __init__(
        self,
        stream_path: str | os.PathLike[str],
        *,
        run_id: str,
        pod_id: str,
        fsync_mode: FsyncMode | str = FsyncMode.TRIAL_BOUNDARY,
        annotation_stream: bool = False,
        external_parent_ids: Iterable[str] = (),
        allow_incomplete: bool = False,
    ) -> None:
        self._run_id = _canonical_uuid(run_id, "run_id")
        if not isinstance(pod_id, str) or not pod_id:
            raise ValueError("pod_id must be a non-empty string")
        self._pod_id = pod_id
        try:
            self._fsync_mode = FsyncMode(fsync_mode)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in FsyncMode)
            raise ValueError(f"fsync_mode must be one of: {choices}") from exc
        self._allow_incomplete = bool(allow_incomplete)
        self._annotation_stream = bool(annotation_stream)
        external = frozenset(
            _canonical_uuid(parent_id, "external_parent_ids")
            for parent_id in external_parent_ids
        )
        if external and not self._annotation_stream:
            raise ValueError(
                "external_parent_ids are permitted only for an explicit "
                "annotation_stream"
            )
        self._external_parent_ids = external

        self._lock = threading.RLock()
        self._fd: int | None = None
        self._closed = False
        self._poisoned = False
        self._event_ids: set[str] = set()
        self._lanes: dict[str | None, _LaneState] = {}
        self._open_trials: set[str] = set()
        self._sealed_trials: dict[str, str] = {}
        self._event_count = 0
        self._byte_size = 0
        self._path_registered = False

        self._path = self._prepare_path(stream_path)
        try:
            self._register_active_path()
            self._fd = self._open_stream()
            self._acquire_file_lock()
            self._assert_path_identity()
            self._scan_existing_stream()
        except BaseException:
            self._force_close()
            raise

    @property
    def path(self) -> Path:
        return self._path

    @property
    def status(self) -> WriterStatus:
        with self._lock:
            lanes = tuple(
                LaneStatus(
                    trial_id=trial_id,
                    last_sequence_num=state.last_sequence_num,
                    last_event_hash=state.last_event_hash,
                    event_count=state.event_count,
                )
                for trial_id, state in sorted(
                    self._lanes.items(),
                    key=lambda item: (item[0] is not None, item[0] or ""),
                )
            )
            return WriterStatus(
                path=self._path,
                run_id=self._run_id,
                pod_id=self._pod_id,
                event_count=self._event_count,
                byte_size=self._byte_size,
                open_trials=tuple(sorted(self._open_trials)),
                sealed_trials=tuple(sorted(self._sealed_trials.items())),
                lanes=lanes,
                closed=self._closed,
            )

    def append(self, event: EventEnvelope) -> AppendReceipt:
        """Append one validated event as one canonical newline-terminated record."""

        if not isinstance(event, EventEnvelope):
            raise TypeError("append requires an EventEnvelope")
        with self._lock:
            self._require_writable()
            self._assert_path_identity()
            try:
                event.verify_content_hash()
            except Exception as exc:
                raise StreamCorruptionError(
                    f"event {event.event_id} failed content-hash verification"
                ) from exc
            self._validate_event(event)

            canonical = event.to_json().encode("utf-8")
            if b"\n" in canonical or b"\r" in canonical:
                raise NonCanonicalStreamError(
                    f"event {event.event_id} serialized with a literal newline"
                )
            record = canonical + b"\n"
            offset = self._byte_size
            line_number = self._event_count + 1
            try:
                self._write_all(record)
                should_fsync = self._should_fsync(event)
                if should_fsync:
                    os.fsync(self._require_fd())
            except OSError as exc:
                self._poisoned = True
                raise EventWriterError(
                    f"append failed for event {event.event_id}; reopen and validate "
                    "the stream before retrying"
                ) from exc

            self._commit_event(event)
            self._byte_size += len(record)
            return AppendReceipt(
                event_id=event.event_id,
                content_hash=event.content_hash or "",
                line_number=line_number,
                byte_offset=offset,
                byte_length=len(record),
                fsynced=should_fsync,
                trial_id=event.trial_id,
                sequence_num=event.sequence_num,
            )

    def close(self, *, allow_incomplete: bool | None = None) -> WriterStatus:
        """Close the stream, rejecting unsealed trials unless explicitly allowed."""

        with self._lock:
            if self._closed:
                return self.status
            permitted = (
                self._allow_incomplete
                if allow_incomplete is None
                else bool(allow_incomplete)
            )
            if self._open_trials and not permitted:
                trials = ", ".join(sorted(self._open_trials))
                raise OpenTrialsError(
                    f"cannot close stream with open trials: {trials}; pass "
                    "allow_incomplete=True only for an intentional checkpoint"
                )
            self._force_close()
            return self.status

    def __enter__(self) -> EventWriter:
        self._require_writable()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc, traceback
        if exc_type is None:
            self.close()
        else:
            # Preserve the originating exception while always releasing the lock.
            self._force_close()
        return False

    def _prepare_path(self, stream_path: str | os.PathLike[str]) -> Path:
        if stream_path is None or str(stream_path) == "":
            raise ValueError("stream_path must be explicit")
        requested = Path(stream_path).expanduser()
        if requested.is_symlink():
            raise UnsafeStreamPathError("event stream path must not be a symlink")
        parent = requested.parent
        if parent.is_symlink():
            raise UnsafeStreamPathError("event stream parent must not be a symlink")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise UnsafeStreamPathError(
                f"cannot create event stream directory {parent}"
            ) from exc
        if not parent.is_dir() or parent.is_symlink():
            raise UnsafeStreamPathError(
                f"event stream parent is not a safe directory: {parent}"
            )
        path = parent.resolve(strict=True) / requested.name
        if path.exists() and not path.is_file():
            raise UnsafeStreamPathError(
                f"event stream is not a regular file: {path}"
            )
        if path.is_symlink():
            raise UnsafeStreamPathError("event stream path must not be a symlink")
        return path

    def _register_active_path(self) -> None:
        with _ACTIVE_PATHS_LOCK:
            if self._path in _ACTIVE_PATHS:
                raise StreamLockedError(
                    f"another EventWriter in this process owns {self._path}"
                )
            _ACTIVE_PATHS.add(self._path)
            self._path_registered = True

    def _open_stream(self) -> int:
        flags = os.O_APPEND | os.O_CREAT | os.O_RDWR
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self._path, flags, 0o600)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise UnsafeStreamPathError(
                    f"unsafe event stream path {self._path}"
                ) from exc
            raise
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            os.close(descriptor)
            raise UnsafeStreamPathError(
                f"event stream is not a regular file: {self._path}"
            )
        return descriptor

    def _acquire_file_lock(self) -> None:
        if fcntl is None:
            return
        try:
            fcntl.flock(self._require_fd(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise StreamLockedError(
                    f"another process owns event stream {self._path}"
                ) from exc
            raise

    def _assert_path_identity(self) -> None:
        descriptor_stat = os.fstat(self._require_fd())
        try:
            path_stat = os.stat(self._path, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise UnsafeStreamPathError(
                f"event stream path disappeared: {self._path}"
            ) from exc
        if not stat.S_ISREG(path_stat.st_mode):
            raise UnsafeStreamPathError(
                f"event stream path is no longer a regular file: {self._path}"
            )
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            path_stat.st_dev,
            path_stat.st_ino,
        ):
            raise UnsafeStreamPathError(
                f"event stream path was replaced while open: {self._path}"
            )

    def _scan_existing_stream(self) -> None:
        descriptor = self._require_fd()
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1_048_576)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        self._byte_size = len(raw)
        if not raw:
            return
        if not raw.endswith(b"\n"):
            raise TruncatedStreamError(
                f"event stream has a truncated final line at byte {len(raw)}"
            )

        offset = 0
        for line_number, record in enumerate(raw.splitlines(keepends=True), start=1):
            if record == b"\n":
                raise StreamCorruptionError(
                    f"empty event record at line {line_number}, byte {offset}"
                )
            line = record[:-1]
            try:
                event = EventEnvelope.from_json(
                    line, preserve_unknown_payloads=True
                )
            except Exception as exc:
                raise StreamCorruptionError(
                    f"invalid event at line {line_number}, byte {offset}"
                ) from exc
            if event.to_json().encode("utf-8") != line:
                raise NonCanonicalStreamError(
                    f"non-canonical event JSON at line {line_number}, byte {offset}"
                )
            try:
                event.verify_content_hash()
                self._validate_event(event)
            except EventWriterError as exc:
                raise StreamCorruptionError(
                    f"illegal event {event.event_id} at line {line_number}, "
                    f"byte {offset}: {exc}"
                ) from exc
            except Exception as exc:
                raise StreamCorruptionError(
                    f"event hash failure at line {line_number}, byte {offset}"
                ) from exc
            self._commit_event(event)
            offset += len(record)

    def _validate_event(self, event: EventEnvelope) -> None:
        if event.run_id != self._run_id or event.pod_id != self._pod_id:
            raise StreamIdentityError(
                f"event {event.event_id} belongs to run/pod "
                f"{event.run_id}/{event.pod_id}, expected "
                f"{self._run_id}/{self._pod_id}"
            )
        if event.event_id in self._event_ids:
            raise DuplicateEventError(f"duplicate event_id {event.event_id}")
        if event.event_id in self._external_parent_ids:
            raise DuplicateEventError(
                f"event_id {event.event_id} is reserved as an external parent"
            )

        if len(event.parent_event_ids) != len(set(event.parent_event_ids)):
            raise EventParentError(
                f"event {event.event_id} contains duplicate parent IDs"
            )
        for parent_id in event.parent_event_ids:
            if parent_id in self._event_ids:
                continue
            if self._annotation_stream and parent_id in self._external_parent_ids:
                continue
            raise EventParentError(
                f"event {event.event_id} references missing or later parent "
                f"{parent_id}"
            )

        lane = self._lanes.get(event.trial_id)
        if lane is None:
            if event.sequence_num != 0 or event.previous_event_hash is not None:
                raise EventSequenceError(
                    f"first event in lane {event.trial_id!r} must have sequence 0 "
                    "and no previous hash"
                )
        else:
            expected_sequence = lane.last_sequence_num + 1
            if event.sequence_num != expected_sequence:
                raise EventSequenceError(
                    f"lane {event.trial_id!r} expected sequence "
                    f"{expected_sequence}, got {event.sequence_num}"
                )
            if event.previous_event_hash != lane.last_event_hash:
                raise EventSequenceError(
                    f"lane {event.trial_id!r} previous hash does not match "
                    f"sequence {lane.last_sequence_num}"
                )

        self._validate_trial_lifecycle(event)

    def _validate_trial_lifecycle(self, event: EventEnvelope) -> None:
        trial_id = event.trial_id
        if trial_id is None:
            if event.event_type in _TRIAL_BOUNDARIES:
                raise TrialLifecycleError(
                    f"{event.event_type} requires a trial_id"
                )
            return
        if trial_id in self._sealed_trials:
            raise TrialLifecycleError(
                f"trial {trial_id} is already sealed by "
                f"{self._sealed_trials[trial_id]}"
            )
        if event.event_type == _TRIAL_START:
            if trial_id in self._open_trials:
                raise TrialLifecycleError(f"trial {trial_id} is already open")
            return
        if trial_id not in self._open_trials:
            raise TrialLifecycleError(
                f"event {event.event_id} occurs before TrialStarted for {trial_id}"
            )

    def _commit_event(self, event: EventEnvelope) -> None:
        self._event_ids.add(event.event_id)
        lane = self._lanes.get(event.trial_id)
        self._lanes[event.trial_id] = _LaneState(
            last_sequence_num=event.sequence_num,
            last_event_hash=event.content_hash or "",
            event_count=1 if lane is None else lane.event_count + 1,
        )
        self._event_count += 1
        if event.trial_id is None:
            return
        if event.event_type == _TRIAL_START:
            self._open_trials.add(event.trial_id)
        elif event.event_type in _TRIAL_SEALS:
            self._open_trials.remove(event.trial_id)
            self._sealed_trials[event.trial_id] = event.event_type

    def _should_fsync(self, event: EventEnvelope) -> bool:
        if self._fsync_mode is FsyncMode.ALWAYS:
            return True
        if self._fsync_mode is FsyncMode.NEVER:
            return False
        return event.event_type in _TRIAL_BOUNDARIES

    def _write_all(self, value: bytes) -> None:
        view = memoryview(value)
        while view:
            written = os.write(self._require_fd(), view)
            if written <= 0:
                raise OSError("zero-byte append to event stream")
            view = view[written:]

    def _require_fd(self) -> int:
        if self._fd is None:
            raise WriterClosedError("event writer is closed")
        return self._fd

    def _require_writable(self) -> None:
        if self._closed or self._fd is None:
            raise WriterClosedError("event writer is closed")
        if self._poisoned:
            raise WriterClosedError(
                "event writer is poisoned by a previous I/O failure; reopen it"
            )

    def _force_close(self) -> None:
        with self._lock:
            descriptor = self._fd
            self._fd = None
            if descriptor is not None:
                if fcntl is not None:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    except OSError:
                        pass
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if self._path_registered:
                with _ACTIVE_PATHS_LOCK:
                    _ACTIVE_PATHS.discard(self._path)
                self._path_registered = False
            self._closed = True


__all__ = [
    "AppendReceipt",
    "DuplicateEventError",
    "EventParentError",
    "EventSequenceError",
    "EventWriter",
    "EventWriterError",
    "FsyncMode",
    "LaneStatus",
    "NonCanonicalStreamError",
    "OpenTrialsError",
    "StreamCorruptionError",
    "StreamIdentityError",
    "StreamLockedError",
    "TrialLifecycleError",
    "TruncatedStreamError",
    "UnsafeStreamPathError",
    "WriterClosedError",
    "WriterStatus",
]
