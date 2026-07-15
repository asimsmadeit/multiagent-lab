"""Deterministic projection replay and read-only trial resume inspection.

Replay means rebuilding immutable views from recorded events.  It never calls a
model and makes no claim that stochastic inference would regenerate the same
tokens.  Resume inspection reports structural event boundaries only; it does
not infer that arbitrary in-memory Concordia state is serializable.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel

from interpretability.events.payloads import (
    ActionCommittedPayload,
    ActionProposedPayload,
    ModelCallCompletedPayload,
    ModelCallFailedPayload,
    ModelCallStartedPayload,
    TrialCompletedPayload,
    TrialFailedPayload,
    TrialStartedPayload,
    TurnAdvancedPayload,
)
from interpretability.events.projectors import (
    ActivationSampleProjector,
    AgentViewProjector,
    DyadProjector,
    MetricInputProjector,
    ProjectionError,
    ProjectionWarning,
    SemanticProjection,
    TranscriptProjector,
    TrialStateProjector,
)
from interpretability.events.reader import EventReader, EventReaderError
from interpretability.events.schema import (
    ActivationCapturedPayload,
    EventEnvelope,
    OpaqueEventPayload,
)

REPLAY_SCHEMA_VERSION = "1.0.0"
PROJECTION_VERSION = "1.0.0"
DEFAULT_MAX_RECORD_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_STREAMS = 256
MAX_STREAM_FILENAME_BYTES = 255

ProjectionName = Literal[
    "transcript",
    "agent_view",
    "trial_state",
    "activation_samples",
    "dyad",
    "metric_input",
]
ResumeClassification = Literal[
    "completed",
    "failed",
    "restart_required",
    "resumable_safe_boundary",
    "invalid",
]

SAFE_RESUME_BOUNDARY_EVENT_TYPES = frozenset(
    {
        "TrialStarted",
        "TurnAdvanced",
        "BehaviorLabeled",
        "MonitorScored",
        "QualityControlApplied",
        "OutcomeResolved",
    }
)
_LINK_EVENT_TYPES = frozenset(
    {
        "ModelCallStarted",
        "ModelCallCompleted",
        "ModelCallFailed",
        "ActivationCaptured",
        "ActionProposed",
        "ActionCommitted",
        "TurnAdvanced",
    }
)


class ReplayError(RuntimeError):
    """Base class for replay and resume-inspection failures."""


class ReplayRequestError(ReplayError, ValueError):
    """Raised for an unknown projection or invalid projector configuration."""


class ReplaySourceError(ReplayError):
    """Raised when a source stream is invalid, corrupt, or changes during read."""


class ReplayOwnershipError(ReplayError):
    """Raised for mixed run identity or duplicate trial ownership."""


class ReplayProjectionError(ReplayError):
    """Raised when validated events cannot satisfy a requested projection."""


class ReplayPathError(ReplayError, ValueError):
    """Raised for unsafe, ambiguous, or unbounded stream path input."""


def _semantic_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _semantic_value(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, BaseModel):
        return _semantic_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_semantic_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported replay semantic value {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _semantic_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _semantic_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _canonical_uuid(value: str, field_name: str) -> str:
    _nonempty(value, field_name)
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ReplayRequestError(
            f"{field_name} must be a canonical UUID string"
        ) from exc
    if str(parsed) != value:
        raise ReplayRequestError(
            f"{field_name} must use lowercase canonical UUID form"
        )
    return value


def _sha256(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ProjectorConfigEntry(SemanticProjection):
    """One sorted immutable projector configuration item."""

    key: str
    value: str | int | None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ReplayRequestError("projector config keys must be non-empty strings")
        if isinstance(self.value, bool) or not isinstance(
            self.value, (str, int, type(None))
        ):
            raise ReplayRequestError(
                f"projector config {self.key!r} has unsupported value type"
            )


@dataclass(frozen=True, slots=True)
class ProjectorDefinition(SemanticProjection):
    """Immutable registry metadata; factories are selected explicitly by name."""

    name: str
    version: str
    required_config_keys: tuple[str, ...]
    optional_config_keys: tuple[str, ...]


def projector_registry() -> tuple[ProjectorDefinition, ...]:
    """Return a fresh immutable registry; no mutable global registration exists."""

    return (
        ProjectorDefinition("transcript", PROJECTION_VERSION, (), ()),
        ProjectorDefinition(
            "agent_view",
            PROJECTION_VERSION,
            ("actor_id",),
            ("through_sequence_num",),
        ),
        ProjectorDefinition("trial_state", PROJECTION_VERSION, (), ()),
        ProjectorDefinition("activation_samples", PROJECTION_VERSION, (), ()),
        ProjectorDefinition("dyad", PROJECTION_VERSION, (), ()),
        ProjectorDefinition("metric_input", PROJECTION_VERSION, (), ()),
    )


def _definition(name: str) -> ProjectorDefinition:
    for definition in projector_registry():
        if definition.name == name:
            return definition
    choices = ", ".join(item.name for item in projector_registry())
    raise ReplayRequestError(
        f"unknown projection {name!r}; expected one of: {choices}"
    )


def _normalize_projector_config(
    definition: ProjectorDefinition,
    config: Mapping[str, Any] | None,
) -> tuple[ProjectorConfigEntry, ...]:
    if config is None:
        supplied: dict[str, Any] = {}
    elif isinstance(config, Mapping):
        supplied = dict(config)
    else:
        raise ReplayRequestError("projector_config must be a mapping")
    if any(not isinstance(key, str) or not key for key in supplied):
        raise ReplayRequestError("projector_config keys must be non-empty strings")
    allowed = set(definition.required_config_keys + definition.optional_config_keys)
    extra = sorted(set(supplied).difference(allowed))
    missing = sorted(set(definition.required_config_keys).difference(supplied))
    if extra or missing:
        raise ReplayRequestError(
            f"invalid config for {definition.name}: missing={missing}, extra={extra}"
        )

    actor_id = supplied.get("actor_id")
    if "actor_id" in supplied and (
        not isinstance(actor_id, str) or not actor_id
    ):
        raise ReplayRequestError("agent_view actor_id must be a non-empty string")
    cutoff = supplied.get("through_sequence_num")
    if "through_sequence_num" in supplied and (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, int)
        or cutoff < 0
    ):
        raise ReplayRequestError(
            "agent_view through_sequence_num must be a non-negative integer"
        )
    return tuple(
        ProjectorConfigEntry(key=key, value=supplied[key])
        for key in sorted(supplied)
    )


@dataclass(frozen=True, slots=True, init=False)
class ProjectionRequest(SemanticProjection):
    """Strict immutable request for one named projection of one exact trial."""

    schema_version: str
    projection_name: str
    projection_version: str
    trial_id: str
    expected_run_id: str | None
    expected_pod_id: str | None
    projector_config: tuple[ProjectorConfigEntry, ...]

    def __init__(
        self,
        projection_name: str,
        trial_id: str,
        *,
        projection_version: str = PROJECTION_VERSION,
        expected_run_id: str | None = None,
        expected_pod_id: str | None = None,
        projector_config: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(projection_name, str) or not projection_name:
            raise ReplayRequestError("projection_name must be a non-empty string")
        definition = _definition(projection_name)
        if projection_version != definition.version:
            raise ReplayRequestError(
                f"unsupported {projection_name} version {projection_version!r}; "
                f"expected {definition.version!r}"
            )
        if not isinstance(trial_id, str) or not trial_id:
            raise ReplayRequestError("trial_id must be a non-empty string")
        for field_name, value in (
            ("expected_run_id", expected_run_id),
            ("expected_pod_id", expected_pod_id),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ReplayRequestError(
                    f"{field_name} must be a non-empty string when provided"
                )
        if expected_run_id is not None:
            _canonical_uuid(expected_run_id, "expected_run_id")
        normalized = _normalize_projector_config(definition, projector_config)
        object.__setattr__(self, "schema_version", REPLAY_SCHEMA_VERSION)
        object.__setattr__(self, "projection_name", projection_name)
        object.__setattr__(self, "projection_version", projection_version)
        object.__setattr__(self, "trial_id", trial_id)
        object.__setattr__(self, "expected_run_id", expected_run_id)
        object.__setattr__(self, "expected_pod_id", expected_pod_id)
        object.__setattr__(self, "projector_config", normalized)

    @property
    def config(self) -> Mapping[str, str | int | None]:
        return {item.key: item.value for item in self.projector_config}


@dataclass(frozen=True, slots=True)
class SourceLaneTerminal(SemanticProjection):
    """Content-hash terminal identity for one run or trial lane."""

    trial_id: str | None
    event_count: int
    first_sequence_num: int
    last_sequence_num: int
    last_event_id: str
    last_event_type: str
    last_content_hash: str

    def __post_init__(self) -> None:
        if self.trial_id is not None:
            _nonempty(self.trial_id, "trial_id")
        if self.event_count <= 0:
            raise ValueError("source lane event_count must be positive")
        if self.first_sequence_num < 0 or self.last_sequence_num < self.first_sequence_num:
            raise ValueError("source lane sequence bounds are invalid")
        if self.event_count != self.last_sequence_num - self.first_sequence_num + 1:
            raise ValueError("source lane count does not match contiguous sequences")
        _nonempty(self.last_event_id, "last_event_id")
        _nonempty(self.last_event_type, "last_event_type")
        _sha256(self.last_content_hash, "last_content_hash")


@dataclass(frozen=True, slots=True)
class SourceStreamSummary(SemanticProjection):
    """Semantic source identity using counts and hash-chain terminals, not mtime."""

    schema_version: str
    run_id: str | None
    pod_id: str | None
    event_count: int
    lanes: tuple[SourceLaneTerminal, ...]
    source_semantic_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported source summary schema version")
        if (self.run_id is None) != (self.pod_id is None):
            raise ValueError("source run_id and pod_id must be present together")
        if self.run_id is not None:
            _canonical_uuid(self.run_id, "run_id")
            _nonempty(self.pod_id or "", "pod_id")
        if self.event_count < 0:
            raise ValueError("source event_count must be non-negative")
        if sum(lane.event_count for lane in self.lanes) != self.event_count:
            raise ValueError("source lane counts do not equal event_count")
        lane_ids = tuple(lane.trial_id for lane in self.lanes)
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("source summary contains duplicate lanes")
        _sha256(self.source_semantic_hash, "source_semantic_hash")
        expected = _semantic_hash(
            {
                "run_id": self.run_id,
                "pod_id": self.pod_id,
                "event_count": self.event_count,
                "lanes": self.lanes,
            }
        )
        if self.source_semantic_hash != expected:
            raise ValueError("source_semantic_hash does not match lane terminals")


@dataclass(frozen=True, slots=True)
class ReplayWarning(SemanticProjection):
    """Warning retained from a deterministic projector result."""

    code: str
    field_name: str
    detail: str
    source_event_id: str | None

    def __post_init__(self) -> None:
        _nonempty(self.code, "warning code")
        _nonempty(self.field_name, "warning field_name")
        _nonempty(self.detail, "warning detail")


@dataclass(frozen=True, slots=True)
class ProjectionResultManifest(SemanticProjection):
    """Canonical replay manifest tying a result to its exact semantic source."""

    schema_version: str
    projection_name: str
    projection_version: str
    run_id: str
    pod_id: str
    trial_id: str
    dyad_id: str | None
    source: SourceStreamSummary
    projector_config: tuple[ProjectorConfigEntry, ...]
    projection_semantic_hash: str
    warnings: tuple[ReplayWarning, ...]

    def __post_init__(self) -> None:
        definition = _definition(self.projection_name)
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported projection manifest schema version")
        if self.projection_version != definition.version:
            raise ValueError("projection manifest version mismatch")
        _canonical_uuid(self.run_id, "run_id")
        for field_name, value in (
            ("pod_id", self.pod_id),
            ("trial_id", self.trial_id),
        ):
            _nonempty(value, field_name)
        if self.source.run_id != self.run_id or self.source.pod_id != self.pod_id:
            raise ValueError("manifest/source run or pod identity mismatch")
        _sha256(self.projection_semantic_hash, "projection_semantic_hash")
        keys = tuple(entry.key for entry in self.projector_config)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("projector_config must be sorted and unique")


@dataclass(frozen=True, slots=True)
class ProjectionReplayResult(SemanticProjection):
    """A canonical manifest plus the immutable projected value."""

    manifest: ProjectionResultManifest
    projection: SemanticProjection

    def __post_init__(self) -> None:
        if self.projection.semantic_hash != self.manifest.projection_semantic_hash:
            raise ReplayProjectionError(
                "projection value does not match manifest semantic hash"
            )
        for field_name in ("run_id", "pod_id", "trial_id", "dyad_id"):
            if getattr(self.projection, field_name, None) != getattr(
                self.manifest, field_name
            ):
                raise ReplayProjectionError(
                    f"projection/manifest {field_name} identity mismatch"
                )
        if _collect_warnings(self.projection) != self.manifest.warnings:
            raise ReplayProjectionError(
                "projection warnings do not match manifest warnings"
            )


def _validate_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _safe_regular_stream_path(stream_path: str | os.PathLike[str]) -> Path:
    if stream_path is None or str(stream_path) == "":
        raise ReplayPathError("stream path must be explicit")
    requested = Path(stream_path).expanduser()
    if requested.is_symlink() or requested.parent.is_symlink():
        raise ReplayPathError(f"stream path must not be a symlink: {requested}")
    try:
        path = requested.resolve(strict=True)
        path_stat = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ReplayPathError(f"stream path is unavailable: {requested}") from exc
    if not stat.S_ISREG(path_stat.st_mode):
        raise ReplayPathError(f"stream path is not a regular file: {path}")
    return path


def _read_stream(
    stream_path: str | os.PathLike[str],
    *,
    expected_run_id: str | None,
    expected_pod_id: str | None,
    max_record_bytes: int,
) -> tuple[Path, tuple[EventEnvelope, ...]]:
    path = _safe_regular_stream_path(stream_path)
    _validate_positive_int(max_record_bytes, "max_record_bytes")
    try:
        reader = EventReader(
            path,
            expected_run_id=expected_run_id,
            expected_pod_id=expected_pod_id,
            strict_canonical=True,
            preserve_unknown_payloads=True,
            max_record_bytes=max_record_bytes,
        )
        # A single iterator performs validation and mutation checks against the
        # same descriptor snapshot.  Do not split this into validate/read passes.
        events = tuple(item.event for item in reader.iter_events())
    except EventReaderError as exc:
        raise ReplaySourceError(
            f"source stream failed strict validation: {type(exc).__name__}: {exc}"
        ) from exc
    return path, events


def _validate_in_memory_events(
    events: Iterable[EventEnvelope],
    *,
    expected_run_id: str | None,
    expected_pod_id: str | None,
) -> tuple[EventEnvelope, ...]:
    if isinstance(events, (str, bytes)):
        raise TypeError("events must be an iterable of EventEnvelope objects")
    collected: list[EventEnvelope] = []
    seen: set[str] = set()
    lanes: dict[str | None, tuple[int, str]] = {}
    open_trials: set[str] = set()
    sealed_trials: set[str] = set()
    run_id: str | None = None
    pod_id: str | None = None

    for position, event in enumerate(events):
        if not isinstance(event, EventEnvelope):
            raise TypeError(
                f"events[{position}] must be EventEnvelope, got "
                f"{type(event).__name__}"
            )
        try:
            event.verify_content_hash()
        except Exception as exc:
            raise ReplaySourceError(
                f"event {event.event_id} failed content-hash verification"
            ) from exc
        if run_id is None:
            run_id, pod_id = event.run_id, event.pod_id
        elif event.run_id != run_id or event.pod_id != pod_id:
            raise ReplayOwnershipError(
                f"event {event.event_id} changes in-memory run/pod ownership"
            )
        if expected_run_id is not None and event.run_id != expected_run_id:
            raise ReplayOwnershipError(
                f"event run {event.run_id} does not match expected {expected_run_id}"
            )
        if expected_pod_id is not None and event.pod_id != expected_pod_id:
            raise ReplayOwnershipError(
                f"event pod {event.pod_id} does not match expected {expected_pod_id}"
            )
        if event.event_id in seen:
            raise ReplaySourceError(f"duplicate event_id {event.event_id}")
        if len(event.parent_event_ids) != len(set(event.parent_event_ids)):
            raise ReplaySourceError(
                f"event {event.event_id} contains duplicate parent IDs"
            )
        for parent_id in event.parent_event_ids:
            if parent_id not in seen:
                raise ReplaySourceError(
                    f"event {event.event_id} references missing or later parent "
                    f"{parent_id}"
                )
        lane = lanes.get(event.trial_id)
        if lane is None:
            if event.sequence_num != 0 or event.previous_event_hash is not None:
                raise ReplaySourceError(
                    f"first event in lane {event.trial_id!r} is not sequence 0"
                )
        else:
            if event.sequence_num != lane[0] + 1:
                raise ReplaySourceError(
                    f"lane {event.trial_id!r} has a sequence gap"
                )
            if event.previous_event_hash != lane[1]:
                raise ReplaySourceError(
                    f"lane {event.trial_id!r} has a previous-hash mismatch"
                )

        trial_id = event.trial_id
        if trial_id is None:
            if event.event_type in {"TrialStarted", "TrialCompleted", "TrialFailed"}:
                raise ReplaySourceError(f"{event.event_type} requires trial_id")
        else:
            if trial_id in sealed_trials:
                raise ReplaySourceError(
                    f"event {event.event_id} mutates sealed trial {trial_id}"
                )
            if event.event_type == "TrialStarted":
                if trial_id in open_trials:
                    raise ReplaySourceError(f"duplicate TrialStarted for {trial_id}")
                open_trials.add(trial_id)
            elif trial_id not in open_trials:
                raise ReplaySourceError(
                    f"event {event.event_id} occurs before TrialStarted"
                )
            if event.event_type in {"TrialCompleted", "TrialFailed"}:
                open_trials.remove(trial_id)
                sealed_trials.add(trial_id)

        seen.add(event.event_id)
        lanes[event.trial_id] = (event.sequence_num, event.content_hash or "")
        collected.append(event)
    return tuple(collected)


def _source_summary(events: Sequence[EventEnvelope]) -> SourceStreamSummary:
    lane_events: dict[str | None, list[EventEnvelope]] = {}
    for event in events:
        lane_events.setdefault(event.trial_id, []).append(event)
    lanes = tuple(
        SourceLaneTerminal(
            trial_id=trial_id,
            event_count=len(items),
            first_sequence_num=items[0].sequence_num,
            last_sequence_num=items[-1].sequence_num,
            last_event_id=items[-1].event_id,
            last_event_type=items[-1].event_type,
            last_content_hash=items[-1].content_hash or "",
        )
        for trial_id, items in sorted(
            lane_events.items(), key=lambda pair: (pair[0] is not None, pair[0] or "")
        )
    )
    material = {
        "run_id": None if not events else events[0].run_id,
        "pod_id": None if not events else events[0].pod_id,
        "event_count": len(events),
        "lanes": lanes,
    }
    return SourceStreamSummary(
        schema_version=REPLAY_SCHEMA_VERSION,
        run_id=material["run_id"],
        pod_id=material["pod_id"],
        event_count=len(events),
        lanes=lanes,
        source_semantic_hash=_semantic_hash(material),
    )


def _select_exact_trial(
    events: Sequence[EventEnvelope], trial_id: str
) -> tuple[EventEnvelope, ...]:
    trial_events = tuple(event for event in events if event.trial_id == trial_id)
    if not trial_events:
        raise ReplayOwnershipError(f"trial {trial_id!r} is absent from source stream")
    if trial_events[0].event_type != "TrialStarted":
        raise ReplaySourceError(
            f"trial {trial_id!r} does not begin with TrialStarted"
        )

    by_id = {event.event_id: event for event in events}
    required_run_ids: set[str] = set()
    pending = [
        parent_id
        for event in trial_events
        for parent_id in event.parent_event_ids
    ]
    visited: set[str] = set()
    while pending:
        parent_id = pending.pop()
        if parent_id in visited:
            continue
        visited.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            raise ReplaySourceError(
                f"trial {trial_id!r} references missing parent {parent_id}"
            )
        if parent.trial_id is None:
            required_run_ids.add(parent.event_id)
            pending.extend(parent.parent_event_ids)
        elif parent.trial_id != trial_id:
            raise ReplayOwnershipError(
                f"trial {trial_id!r} depends on event owned by trial "
                f"{parent.trial_id!r}"
            )

    run_events = [event for event in events if event.trial_id is None]
    if required_run_ids:
        maximum_sequence = max(
            by_id[parent_id].sequence_num for parent_id in required_run_ids
        )
        run_prefix = tuple(
            event for event in run_events if event.sequence_num <= maximum_sequence
        )
    else:
        run_prefix = ()
    selected_ids = {event.event_id for event in (*run_prefix, *trial_events)}
    for event in (*run_prefix, *trial_events):
        missing = set(event.parent_event_ids).difference(selected_ids)
        if missing:
            raise ReplayOwnershipError(
                f"exact trial filter would omit parents of {event.event_id}: "
                f"{', '.join(sorted(missing))}"
            )
    return tuple(
        event
        for event in events
        if event.event_id in selected_ids
    )


def create_projector(request: ProjectionRequest) -> Any:
    """Create a fresh configured projector from the immutable registry request."""

    if not isinstance(request, ProjectionRequest):
        raise TypeError("request must be ProjectionRequest")
    config = dict(request.config)
    common = {
        "expected_run_id": request.expected_run_id,
        "expected_trial_id": request.trial_id,
    }
    name = request.projection_name
    if name == "transcript":
        return TranscriptProjector(**common)
    if name == "agent_view":
        return AgentViewProjector(
            str(config["actor_id"]),
            through_sequence_num=config.get("through_sequence_num"),
            **common,
        )
    if name == "trial_state":
        return TrialStateProjector(**common)
    if name == "activation_samples":
        return ActivationSampleProjector(**common)
    if name == "dyad":
        return DyadProjector(**common)
    if name == "metric_input":
        return MetricInputProjector(**common)
    raise ReplayRequestError(f"unknown projection {name!r}")


def _collect_warnings(value: Any) -> tuple[ReplayWarning, ...]:
    collected: list[ReplayWarning] = []

    def visit(item: Any) -> None:
        if isinstance(item, ProjectionWarning):
            collected.append(
                ReplayWarning(
                    code=item.code,
                    field_name=item.field_name,
                    detail=item.detail,
                    source_event_id=item.source_event_id,
                )
            )
            return
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                if not field.name.startswith("_"):
                    visit(getattr(item, field.name))
            return
        if isinstance(item, Mapping):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (tuple, list)):
            for nested in item:
                visit(nested)

    visit(value)
    unique: list[ReplayWarning] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    for warning in collected:
        identity = (
            warning.code,
            warning.field_name,
            warning.detail,
            warning.source_event_id,
        )
        if identity not in seen:
            seen.add(identity)
            unique.append(warning)
    return tuple(unique)


def _replay_loaded_events(
    events: Sequence[EventEnvelope], request: ProjectionRequest
) -> ProjectionReplayResult:
    if not events:
        raise ReplaySourceError("cannot replay an empty source stream")
    selected = _select_exact_trial(events, request.trial_id)
    try:
        projection = create_projector(request).project(selected)
    except ProjectionError as exc:
        raise ReplayProjectionError(
            f"{request.projection_name} projection failed: {exc}"
        ) from exc
    if not isinstance(projection, SemanticProjection):
        raise ReplayProjectionError("projector returned a non-semantic result")
    run_id = getattr(projection, "run_id", None)
    pod_id = getattr(projection, "pod_id", None)
    trial_id = getattr(projection, "trial_id", None)
    dyad_id = getattr(projection, "dyad_id", None)
    if not all(isinstance(value, str) and value for value in (run_id, pod_id, trial_id)):
        raise ReplayProjectionError("projection omitted run/pod/trial identity")
    manifest = ProjectionResultManifest(
        schema_version=REPLAY_SCHEMA_VERSION,
        projection_name=request.projection_name,
        projection_version=request.projection_version,
        run_id=run_id,
        pod_id=pod_id,
        trial_id=trial_id,
        dyad_id=dyad_id,
        source=_source_summary(events),
        projector_config=request.projector_config,
        projection_semantic_hash=projection.semantic_hash,
        warnings=_collect_warnings(projection),
    )
    return ProjectionReplayResult(manifest=manifest, projection=projection)


def replay_projection(
    stream_path: str | os.PathLike[str],
    request: ProjectionRequest,
    *,
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
) -> ProjectionReplayResult:
    """Strictly read one stream and replay one exact-trial projection."""

    if not isinstance(request, ProjectionRequest):
        raise TypeError("request must be ProjectionRequest")
    _, events = _read_stream(
        stream_path,
        expected_run_id=request.expected_run_id,
        expected_pod_id=request.expected_pod_id,
        max_record_bytes=max_record_bytes,
    )
    return _replay_loaded_events(events, request)


def replay_validated_events(
    events: Iterable[EventEnvelope], request: ProjectionRequest
) -> ProjectionReplayResult:
    """Replay in-memory events after applying the same integrity contracts."""

    if not isinstance(request, ProjectionRequest):
        raise TypeError("request must be ProjectionRequest")
    validated = _validate_in_memory_events(
        events,
        expected_run_id=request.expected_run_id,
        expected_pod_id=request.expected_pod_id,
    )
    return _replay_loaded_events(validated, request)


@dataclass(frozen=True, slots=True)
class PendingLinkage(SemanticProjection):
    """Unfinished call/artifact/action relationships at a trial boundary."""

    open_model_call_ids: tuple[str, ...]
    awaiting_artifact_call_ids: tuple[str, ...]
    awaiting_proposal_call_ids: tuple[str, ...]
    open_proposal_event_ids: tuple[str, ...]
    open_committed_action_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in fields(self):
            values = getattr(self, field.name)
            if len(values) != len(set(values)) or values != tuple(sorted(values)):
                raise ValueError(
                    f"PendingLinkage.{field.name} must be sorted and unique"
                )
            for value in values:
                _nonempty(value, f"PendingLinkage.{field.name}")

    @property
    def empty(self) -> bool:
        return not any(
            (
                self.open_model_call_ids,
                self.awaiting_artifact_call_ids,
                self.awaiting_proposal_call_ids,
                self.open_proposal_event_ids,
                self.open_committed_action_event_ids,
            )
        )


@dataclass(frozen=True, slots=True)
class TrialResumeInspection(SemanticProjection):
    """Read-only resume classification for one trial or invalid source record."""

    schema_version: str
    run_id: str | None
    pod_id: str | None
    trial_id: str | None
    dyad_id: str | None
    classification: ResumeClassification
    started_event_id: str | None
    terminal_event_id: str | None
    last_event_id: str | None
    last_event_type: str | None
    last_sequence_num: int | None
    last_content_hash: str | None
    reason: str
    safe_boundary_event_id: str | None
    safe_boundary_event_type: str | None
    pending_linkage: PendingLinkage

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported trial inspection schema version")
        _nonempty(self.reason, "reason")
        if self.classification not in {
            "completed",
            "failed",
            "restart_required",
            "resumable_safe_boundary",
            "invalid",
        }:
            raise ValueError("unknown resume classification")
        if self.run_id is not None:
            _canonical_uuid(self.run_id, "run_id")
        if self.last_sequence_num is not None and self.last_sequence_num < 0:
            raise ValueError("last_sequence_num must be non-negative")
        if self.last_content_hash is not None:
            _sha256(self.last_content_hash, "last_content_hash")
        boundary_present = self.safe_boundary_event_id is not None
        if boundary_present != (self.safe_boundary_event_type is not None):
            raise ValueError("safe boundary ID and type must be present together")
        if self.classification == "resumable_safe_boundary" and not boundary_present:
            raise ValueError("resumable classification requires a safe boundary")
        if self.classification != "resumable_safe_boundary" and boundary_present:
            raise ValueError("only resumable classification may name a safe boundary")
        terminal_expected = self.classification in {"completed", "failed"}
        if terminal_expected != (self.terminal_event_id is not None):
            raise ValueError("terminal classification/event invariant failed")


@dataclass(frozen=True, slots=True)
class StreamResumeInspection(SemanticProjection):
    """Inspection of one source file without path timestamp metadata."""

    schema_version: str
    source_name: str
    source: SourceStreamSummary | None
    trials: tuple[TrialResumeInspection, ...]
    valid: bool
    error_type: str | None
    error_reason: str | None

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported stream inspection schema version")
        _nonempty(self.source_name, "source_name")
        error_present = self.error_type is not None or self.error_reason is not None
        if (self.error_type is None) != (self.error_reason is None):
            raise ValueError("stream error type and reason must be present together")
        if self.valid and error_present:
            raise ValueError("valid stream inspection cannot contain an error")
        if self.valid and any(trial.classification == "invalid" for trial in self.trials):
            raise ValueError("valid stream cannot contain invalid trial inspection")


@dataclass(frozen=True, slots=True)
class ClassificationCount(SemanticProjection):
    classification: ResumeClassification
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("classification count must be non-negative")


@dataclass(frozen=True, slots=True)
class MultiStreamInspectionSummary(SemanticProjection):
    """Deterministically ordered ownership-checked multi-stream inspection."""

    schema_version: str
    run_id: str | None
    stream_count: int
    trial_count: int
    valid: bool
    classification_counts: tuple[ClassificationCount, ...]
    streams: tuple[StreamResumeInspection, ...]

    def __post_init__(self) -> None:
        if self.schema_version != REPLAY_SCHEMA_VERSION:
            raise ValueError("unsupported multi-stream schema version")
        if self.run_id is not None:
            _canonical_uuid(self.run_id, "run_id")
        if self.stream_count != len(self.streams):
            raise ValueError("stream_count does not match streams")
        if self.trial_count != sum(len(stream.trials) for stream in self.streams):
            raise ValueError("trial_count does not match stream trial records")
        if self.valid != all(stream.valid for stream in self.streams):
            raise ValueError("multi-stream valid flag does not match streams")
        if sum(item.count for item in self.classification_counts) != self.trial_count:
            raise ValueError("classification counts do not equal trial_count")


@dataclass(slots=True)
class _LinkState:
    starts: dict[str, tuple[EventEnvelope, ModelCallStartedPayload]]
    terminals: dict[str, EventEnvelope]
    completed: dict[str, tuple[EventEnvelope, ModelCallCompletedPayload]]
    captured_hashes: dict[str, list[str]]
    awaiting_proposal_calls: set[str]
    proposals: dict[str, tuple[EventEnvelope, ActionProposedPayload]]
    proposal_calls: dict[str, str]
    commitments: dict[str, tuple[EventEnvelope, ActionCommittedPayload]]
    errors: list[str]


def _new_link_state() -> _LinkState:
    return _LinkState({}, {}, {}, {}, set(), {}, {}, {}, [])


def _link_error(state: _LinkState, event: EventEnvelope, message: str) -> None:
    state.errors.append(f"{event.event_type} {event.event_id}: {message}")


def _analyze_trial_linkage(
    trial_events: Sequence[EventEnvelope],
) -> tuple[PendingLinkage, tuple[str, ...]]:
    state = _new_link_state()
    open_commits: dict[str, EventEnvelope] = {}

    for event in trial_events:
        payload = event.payload
        if isinstance(payload, OpaqueEventPayload):
            if event.event_type in _LINK_EVENT_TYPES:
                _link_error(
                    state,
                    event,
                    "required call/action payload schema is opaque",
                )
            continue

        if isinstance(payload, ModelCallStartedPayload):
            call_id = payload.model_call_id
            if call_id in state.starts:
                _link_error(state, event, f"duplicate start for call {call_id}")
                continue
            if event.model_call_id != call_id or event.actor_id != payload.actor_id:
                _link_error(state, event, "envelope call/actor identity mismatch")
            state.starts[call_id] = (event, payload)
            continue

        if isinstance(payload, (ModelCallCompletedPayload, ModelCallFailedPayload)):
            call_id = payload.model_call_id
            start_pair = state.starts.get(call_id)
            if start_pair is None:
                _link_error(state, event, f"terminal call {call_id} has no start")
                continue
            start_event, start = start_pair
            if call_id in state.terminals:
                _link_error(state, event, f"call {call_id} has duplicate terminals")
                continue
            if (
                payload.started_event_id != start_event.event_id
                or payload.actor_id != start.actor_id
                or payload.purpose != start.purpose
                or payload.generation_config_hash != start.generation_config.sha256
            ):
                _link_error(state, event, "terminal does not match its call start")
            if event.model_call_id != call_id or event.actor_id != payload.actor_id:
                _link_error(state, event, "envelope call/actor identity mismatch")
            state.terminals[call_id] = event
            if isinstance(payload, ModelCallCompletedPayload):
                state.completed[call_id] = (event, payload)
                state.captured_hashes.setdefault(call_id, [])
                if payload.purpose in {"actor_action", "counterpart_action"}:
                    state.awaiting_proposal_calls.add(call_id)
            continue

        if isinstance(payload, ActivationCapturedPayload):
            call_id = payload.artifact.source_model_call_id
            completed_pair = state.completed.get(call_id)
            if completed_pair is None:
                _link_error(state, event, f"activation call {call_id} is not completed")
                continue
            _, completed = completed_pair
            captured = state.captured_hashes.setdefault(call_id, [])
            captured.append(payload.artifact.artifact_hash)
            expected = completed.activation_artifact_hashes
            if tuple(captured) != expected[: len(captured)]:
                _link_error(
                    state,
                    event,
                    "activation order/hash is not a prefix of completed-call metadata",
                )
            start = state.starts[call_id][1]
            if (
                event.model_call_id != call_id
                or event.actor_id != start.actor_id
                or payload.artifact.tokenizer_id != start.tokenizer_id
                or payload.artifact.model_revision != start.model_revision
            ):
                _link_error(state, event, "activation identity/metadata mismatch")
            continue

        if isinstance(payload, ActionProposedPayload):
            call_id = payload.model_call_id
            completed_pair = state.completed.get(call_id)
            if completed_pair is None:
                _link_error(state, event, f"action call {call_id} is not completed")
                continue
            completed_event, completed = completed_pair
            if call_id in state.proposal_calls:
                _link_error(state, event, f"call {call_id} proposes multiple actions")
            if (
                payload.model_call_event_id != completed_event.event_id
                or payload.actor_id != completed.actor_id
                or event.model_call_id != call_id
                or event.actor_id != payload.actor_id
            ):
                _link_error(state, event, "proposal does not match completed call")
            expected = completed.activation_artifact_hashes
            if tuple(state.captured_hashes.get(call_id, ())) != expected:
                _link_error(state, event, "proposal precedes complete activation capture")
            state.awaiting_proposal_calls.discard(call_id)
            state.proposals[event.event_id] = (event, payload)
            state.proposal_calls[call_id] = event.event_id
            continue

        if isinstance(payload, ActionCommittedPayload):
            proposal_pair = state.proposals.get(payload.proposed_event_id)
            if proposal_pair is None:
                _link_error(state, event, "commit has no matching proposal")
                continue
            _, proposal = proposal_pair
            if payload.proposed_event_id in {
                committed.proposed_event_id
                for _, committed in state.commitments.values()
            }:
                _link_error(state, event, "proposal is committed more than once")
            if (
                payload.action_id != proposal.action_id
                or payload.actor_id != proposal.actor_id
                or payload.model_call_id != proposal.model_call_id
                or event.actor_id != payload.actor_id
                or event.model_call_id != payload.model_call_id
            ):
                _link_error(state, event, "commit identity does not match proposal")
            if (
                payload.protocol_decision_event_id is None
                and payload.action_hash != proposal.action_hash
            ):
                _link_error(state, event, "commit action hash does not match proposal")
            state.proposals.pop(payload.proposed_event_id, None)
            state.commitments[event.event_id] = (event, payload)
            open_commits[event.event_id] = event
            continue

        if isinstance(payload, TurnAdvancedPayload):
            commitment_pair = state.commitments.get(payload.committed_action_event_id)
            if commitment_pair is None:
                _link_error(state, event, "turn has no matching committed action")
                continue
            _, commitment = commitment_pair
            if (
                payload.from_actor_id != commitment.actor_id
                or event.actor_id != payload.from_actor_id
            ):
                _link_error(state, event, "turn actor does not match commitment")
            if payload.committed_action_event_id not in open_commits:
                _link_error(state, event, "committed action advances more than once")
            open_commits.pop(payload.committed_action_event_id, None)

    open_calls = tuple(
        sorted(set(state.starts).difference(state.terminals))
    )
    awaiting_artifacts = tuple(
        sorted(
            call_id
            for call_id, (_, completed) in state.completed.items()
            if tuple(state.captured_hashes.get(call_id, ()))
            != completed.activation_artifact_hashes
        )
    )
    pending = PendingLinkage(
        open_model_call_ids=open_calls,
        awaiting_artifact_call_ids=awaiting_artifacts,
        awaiting_proposal_call_ids=tuple(sorted(state.awaiting_proposal_calls)),
        open_proposal_event_ids=tuple(sorted(state.proposals)),
        open_committed_action_event_ids=tuple(sorted(open_commits)),
    )
    return pending, tuple(state.errors)


def _empty_pending_linkage() -> PendingLinkage:
    return PendingLinkage((), (), (), (), ())


def _pending_reason(pending: PendingLinkage) -> str:
    parts: list[str] = []
    for label, values in (
        ("open model calls", pending.open_model_call_ids),
        ("awaiting activation artifacts", pending.awaiting_artifact_call_ids),
        ("awaiting action proposals", pending.awaiting_proposal_call_ids),
        ("open proposals", pending.open_proposal_event_ids),
        ("committed actions awaiting turn advance", pending.open_committed_action_event_ids),
    ):
        if values:
            parts.append(f"{label}: {', '.join(values)}")
    return "; ".join(parts)


def _invalid_trial_inspection(
    *,
    run_id: str | None,
    pod_id: str | None,
    trial_id: str | None,
    dyad_id: str | None,
    started_event_id: str | None,
    last_event: EventEnvelope | None,
    reason: str,
    pending: PendingLinkage | None = None,
) -> TrialResumeInspection:
    return TrialResumeInspection(
        schema_version=REPLAY_SCHEMA_VERSION,
        run_id=run_id,
        pod_id=pod_id,
        trial_id=trial_id,
        dyad_id=dyad_id,
        classification="invalid",
        started_event_id=started_event_id,
        terminal_event_id=None,
        last_event_id=None if last_event is None else last_event.event_id,
        last_event_type=None if last_event is None else last_event.event_type,
        last_sequence_num=None if last_event is None else last_event.sequence_num,
        last_content_hash=None if last_event is None else last_event.content_hash,
        reason=reason,
        safe_boundary_event_id=None,
        safe_boundary_event_type=None,
        pending_linkage=pending or _empty_pending_linkage(),
    )


def _classify_trial(
    events: Sequence[EventEnvelope], trial_id: str
) -> TrialResumeInspection:
    trial_events = tuple(event for event in events if event.trial_id == trial_id)
    if not trial_events:
        return _invalid_trial_inspection(
            run_id=None,
            pod_id=None,
            trial_id=trial_id,
            dyad_id=None,
            started_event_id=None,
            last_event=None,
            reason="trial has no events",
        )
    first, last = trial_events[0], trial_events[-1]
    run_id, pod_id = first.run_id, first.pod_id
    dyads = {event.dyad_id for event in trial_events if event.dyad_id is not None}
    dyad_id = next(iter(dyads)) if len(dyads) == 1 else None
    if len(dyads) > 1:
        return _invalid_trial_inspection(
            run_id=run_id,
            pod_id=pod_id,
            trial_id=trial_id,
            dyad_id=None,
            started_event_id=(first.event_id if first.event_type == "TrialStarted" else None),
            last_event=last,
            reason="trial changes dyad identity",
        )
    if first.event_type != "TrialStarted" or not isinstance(
        first.payload, TrialStartedPayload
    ):
        return _invalid_trial_inspection(
            run_id=run_id,
            pod_id=pod_id,
            trial_id=trial_id,
            dyad_id=dyad_id,
            started_event_id=None,
            last_event=last,
            reason="trial does not begin with a typed TrialStarted payload",
        )

    pending, link_errors = _analyze_trial_linkage(trial_events)
    if link_errors:
        return _invalid_trial_inspection(
            run_id=run_id,
            pod_id=pod_id,
            trial_id=trial_id,
            dyad_id=dyad_id,
            started_event_id=first.event_id,
            last_event=last,
            reason="; ".join(link_errors),
            pending=pending,
        )

    common = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "run_id": run_id,
        "pod_id": pod_id,
        "trial_id": trial_id,
        "dyad_id": dyad_id,
        "started_event_id": first.event_id,
        "last_event_id": last.event_id,
        "last_event_type": last.event_type,
        "last_sequence_num": last.sequence_num,
        "last_content_hash": last.content_hash,
        "pending_linkage": pending,
    }
    if isinstance(last.payload, TrialCompletedPayload):
        return TrialResumeInspection(
            classification="completed",
            terminal_event_id=last.event_id,
            reason="trial is sealed by a typed TrialCompleted event",
            safe_boundary_event_id=None,
            safe_boundary_event_type=None,
            **common,
        )
    if isinstance(last.payload, TrialFailedPayload):
        return TrialResumeInspection(
            classification="failed",
            terminal_event_id=last.event_id,
            reason="trial is sealed by a typed TrialFailed event",
            safe_boundary_event_id=None,
            safe_boundary_event_type=None,
            **common,
        )
    if last.event_type in {"TrialCompleted", "TrialFailed"}:
        return _invalid_trial_inspection(
            run_id=run_id,
            pod_id=pod_id,
            trial_id=trial_id,
            dyad_id=dyad_id,
            started_event_id=first.event_id,
            last_event=last,
            reason=f"terminal {last.event_type} payload schema is opaque or invalid",
            pending=pending,
        )
    if not pending.empty:
        return TrialResumeInspection(
            classification="restart_required",
            terminal_event_id=None,
            reason=(
                "trial ends with unfinished call/artifact/action linkage: "
                + _pending_reason(pending)
            ),
            safe_boundary_event_id=None,
            safe_boundary_event_type=None,
            **common,
        )
    if last.event_type in SAFE_RESUME_BOUNDARY_EVENT_TYPES:
        return TrialResumeInspection(
            classification="resumable_safe_boundary",
            terminal_event_id=None,
            reason=(
                f"{last.event_type} is explicitly allowlisted and no "
                "call/artifact/action linkage is open; this is a structural "
                "boundary, not proof of serializable Concordia state"
            ),
            safe_boundary_event_id=last.event_id,
            safe_boundary_event_type=last.event_type,
            **common,
        )
    return TrialResumeInspection(
        classification="restart_required",
        terminal_event_id=None,
        reason=(
            f"last event type {last.event_type} is not an explicitly allowlisted "
            "resume boundary"
        ),
        safe_boundary_event_id=None,
        safe_boundary_event_type=None,
        **common,
    )


def _inspect_loaded_stream(
    events: Sequence[EventEnvelope], *, source_name: str
) -> StreamResumeInspection:
    source = _source_summary(events)
    trial_ids = sorted(
        {event.trial_id for event in events if event.trial_id is not None}
    )
    trials = tuple(_classify_trial(events, trial_id) for trial_id in trial_ids)
    valid = all(trial.classification != "invalid" for trial in trials)
    return StreamResumeInspection(
        schema_version=REPLAY_SCHEMA_VERSION,
        source_name=source_name,
        source=source,
        trials=trials,
        valid=valid,
        error_type=None,
        error_reason=None,
    )


def inspect_stream(
    stream_path: str | os.PathLike[str],
    *,
    source_name: str | None = None,
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
) -> StreamResumeInspection:
    """Inspect one regular stream without writing, appending, or repairing it."""

    path = _safe_regular_stream_path(stream_path)
    name = path.name if source_name is None else source_name
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ReplayPathError("source_name must be a non-empty safe string")
    try:
        _, events = _read_stream(
            path,
            expected_run_id=None,
            expected_pod_id=None,
            max_record_bytes=max_record_bytes,
        )
    except ReplaySourceError as exc:
        cause = exc.__cause__
        invalid = _invalid_trial_inspection(
            run_id=None,
            pod_id=None,
            trial_id=None,
            dyad_id=None,
            started_event_id=None,
            last_event=None,
            reason=str(exc),
        )
        return StreamResumeInspection(
            schema_version=REPLAY_SCHEMA_VERSION,
            source_name=name,
            source=None,
            trials=(invalid,),
            valid=False,
            error_type=(
                type(cause).__name__ if cause is not None else type(exc).__name__
            ),
            error_reason=str(exc),
        )
    return _inspect_loaded_stream(events, source_name=name)


def _classification_counts(
    streams: Sequence[StreamResumeInspection],
) -> tuple[ClassificationCount, ...]:
    ordering: tuple[ResumeClassification, ...] = (
        "completed",
        "failed",
        "resumable_safe_boundary",
        "restart_required",
        "invalid",
    )
    counts = {
        classification: sum(
            trial.classification == classification
            for stream in streams
            for trial in stream.trials
        )
        for classification in ordering
    }
    return tuple(
        ClassificationCount(classification=classification, count=counts[classification])
        for classification in ordering
    )


def _summarize_stream_inspections(
    inspections: Sequence[StreamResumeInspection],
) -> MultiStreamInspectionSummary:
    ordered = tuple(sorted(inspections, key=lambda item: item.source_name))
    if len({item.source_name for item in ordered}) != len(ordered):
        raise ReplayOwnershipError("multi-stream inspection has duplicate source names")
    run_ids = {
        stream.source.run_id
        for stream in ordered
        if stream.source is not None and stream.source.run_id is not None
    }
    if len(run_ids) > 1:
        raise ReplayOwnershipError(
            f"multi-stream inspection mixes run ownership: {', '.join(sorted(run_ids))}"
        )
    ownership: dict[tuple[str, str, str], str] = {}
    for stream in ordered:
        for trial in stream.trials:
            if trial.run_id is None or trial.pod_id is None or trial.trial_id is None:
                continue
            key = (trial.run_id, trial.pod_id, trial.trial_id)
            previous = ownership.get(key)
            if previous is not None:
                raise ReplayOwnershipError(
                    "duplicate (run,pod,trial) ownership across streams: "
                    f"{key} in {previous!r} and {stream.source_name!r}"
                )
            ownership[key] = stream.source_name
    return MultiStreamInspectionSummary(
        schema_version=REPLAY_SCHEMA_VERSION,
        run_id=None if not run_ids else next(iter(run_ids)),
        stream_count=len(ordered),
        trial_count=sum(len(stream.trials) for stream in ordered),
        valid=all(stream.valid for stream in ordered),
        classification_counts=_classification_counts(ordered),
        streams=ordered,
    )


def inspect_streams(
    stream_paths: Iterable[str | os.PathLike[str]],
    *,
    max_streams: int = DEFAULT_MAX_STREAMS,
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
) -> MultiStreamInspectionSummary:
    """Inspect explicit stream paths in deterministic order with ownership checks."""

    _validate_positive_int(max_streams, "max_streams")
    if isinstance(stream_paths, (str, bytes, os.PathLike)):
        raise TypeError("stream_paths must be an iterable of paths")
    paths = tuple(stream_paths)
    if len(paths) > max_streams:
        raise ReplayPathError(
            f"received {len(paths)} streams; maximum is {max_streams}"
        )
    resolved = tuple(sorted(_safe_regular_stream_path(path) for path in paths))
    if len(set(resolved)) != len(resolved):
        raise ReplayPathError("stream path list contains duplicates")
    inspections = tuple(
        inspect_stream(
            path,
            source_name=path.name,
            max_record_bytes=max_record_bytes,
        )
        for path in resolved
    )
    return _summarize_stream_inspections(inspections)


def inspect_stream_directory(
    directory: str | os.PathLike[str],
    *,
    filenames: Iterable[str],
    max_streams: int = DEFAULT_MAX_STREAMS,
    max_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
) -> MultiStreamInspectionSummary:
    """Inspect only explicitly named regular files under one safe directory."""

    _validate_positive_int(max_streams, "max_streams")
    requested = Path(directory).expanduser()
    if requested.is_symlink():
        raise ReplayPathError("stream directory must not be a symlink")
    try:
        root = requested.resolve(strict=True)
        root_stat = os.stat(root, follow_symlinks=False)
    except OSError as exc:
        raise ReplayPathError(f"stream directory is unavailable: {requested}") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ReplayPathError(f"stream directory is not a directory: {root}")
    if isinstance(filenames, (str, bytes)):
        raise TypeError("filenames must be an iterable of explicit names")
    names = tuple(filenames)
    if len(names) > max_streams:
        raise ReplayPathError(
            f"received {len(names)} filenames; maximum is {max_streams}"
        )
    normalized: list[str] = []
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or Path(name).name != name
            or len(name.encode("utf-8")) > MAX_STREAM_FILENAME_BYTES
        ):
            raise ReplayPathError(f"unsafe explicit stream filename {name!r}")
        normalized.append(name)
    if len(set(normalized)) != len(normalized):
        raise ReplayPathError("explicit filename list contains duplicates")
    paths = tuple(root / name for name in sorted(normalized))
    for path in paths:
        _safe_regular_stream_path(path)
    inspections = tuple(
        inspect_stream(
            path,
            source_name=path.name,
            max_record_bytes=max_record_bytes,
        )
        for path in paths
    )
    return _summarize_stream_inspections(inspections)


__all__ = [
    "DEFAULT_MAX_RECORD_BYTES",
    "DEFAULT_MAX_STREAMS",
    "MAX_STREAM_FILENAME_BYTES",
    "PROJECTION_VERSION",
    "REPLAY_SCHEMA_VERSION",
    "SAFE_RESUME_BOUNDARY_EVENT_TYPES",
    "ClassificationCount",
    "MultiStreamInspectionSummary",
    "PendingLinkage",
    "ProjectionReplayResult",
    "ProjectionRequest",
    "ProjectionResultManifest",
    "ProjectorConfigEntry",
    "ProjectorDefinition",
    "ReplayError",
    "ReplayOwnershipError",
    "ReplayPathError",
    "ReplayProjectionError",
    "ReplayRequestError",
    "ReplaySourceError",
    "ReplayWarning",
    "SourceLaneTerminal",
    "SourceStreamSummary",
    "StreamResumeInspection",
    "TrialResumeInspection",
    "create_projector",
    "inspect_stream",
    "inspect_stream_directory",
    "inspect_streams",
    "projector_registry",
    "replay_projection",
    "replay_validated_events",
]
