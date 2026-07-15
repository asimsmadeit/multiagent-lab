"""Immutable event envelopes for reproducible experiment records.

Events distinguish three kinds of provenance:

* facts record observations, actions, outcomes, and captured artifacts;
* activities record model calls, interventions, and protocol decisions;
* derived annotations record labels, monitor scores, and quality-control results.

The envelope deliberately has no dependency on the negotiation runner.  Payload
models register against an event type and payload schema version, allowing the
runtime, readers, and future schema upgraders to evolve independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any, ClassVar, TypeVar
from uuid import UUID, uuid4, uuid5

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    ValidationInfo,
    field_validator,
    model_validator,
)

EVENT_ENVELOPE_SCHEMA_VERSION = "1.0.0"
ACTIVATION_CAPTURED_PAYLOAD_SCHEMA_VERSION = "1.0.0"

_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "event_type",
        "run_id",
        "pod_id",
        "trial_id",
        "dyad_id",
        "sequence_num",
        "recorded_at",
        "actor_id",
        "actor_role",
        "model_call_id",
        "parent_event_ids",
        "previous_event_hash",
        "payload",
        "payload_schema_version",
        "content_hash",
    }
)
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STABLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_EVENT_TYPE_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9]{0,127}$")
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+/-]{0,127}$")
_IDENTITY_NAMESPACE = UUID("f107ec6e-a79c-5fb2-a7c7-dda4fb6ed925")


class PayloadRegistrationError(ValueError):
    """Raised when two payload models claim the same wire identity."""


class UnknownPayloadTypeError(ValueError):
    """Raised when no typed payload model is registered for an event."""


class EventHashMismatchError(ValueError):
    """Raised when an envelope's supplied content hash is not reproducible."""


def _canonical_uuid(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a canonical UUID string")
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a canonical UUID string") from exc
    if str(parsed) != value:
        raise ValueError(f"{field_name} must use lowercase canonical UUID form")
    return value


def _stable_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _STABLE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a non-empty stable identifier containing only "
            "letters, digits, '.', '_', ':', '/', or '-'"
        )
    return value


def _schema_version(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _VERSION_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} is not a valid schema-version identifier")
    return value


def _sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return value


def _reject_non_finite(value: Any, path: str = "payload") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            _reject_non_finite(getattr(value, name), f"{path}.{name}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


def _reject_mutable_containers(value: Any, path: str = "payload") -> None:
    if isinstance(value, (list, dict, set, bytearray)):
        raise TypeError(
            f"{path} uses a mutable container; use frozen models and tuples"
        )
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            _reject_mutable_containers(getattr(value, name), f"{path}.{name}")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _reject_mutable_containers(item, f"{path}[{index}]")


def _canonical_json(value: Mapping[str, Any]) -> str:
    _reject_non_finite(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _parse_json_object(raw: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value!r} is not permitted")

    parsed = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(parsed, dict):
        raise ValueError("payload JSON must contain an object")
    return parsed


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class EventPayload(_StrictFrozenModel):
    """Base class for immutable, event-specific payload models."""

    EVENT_TYPE: ClassVar[str | None] = None
    PAYLOAD_SCHEMA_VERSION: ClassVar[str | None] = None

    @model_validator(mode="after")
    def _finite_values_only(self) -> EventPayload:
        _reject_non_finite(self)
        _reject_mutable_containers(self)
        return self

    def to_payload_dict(self) -> dict[str, Any]:
        """Return this payload's JSON-compatible wire representation."""

        return self.model_dump(mode="json")


class OpaqueEventPayload(EventPayload):
    """Losslessly preserves an unregistered payload for forward-compatible reads.

    Opaque payloads can be reserialized and hash-checked, but projectors should
    reject them until a typed model or schema upgrader has been installed.
    """

    original_event_type: str
    original_schema_version: str
    canonical_payload_json: str

    @field_validator("original_event_type")
    @classmethod
    def _valid_event_type(cls, value: str) -> str:
        if not _EVENT_TYPE_PATTERN.fullmatch(value):
            raise ValueError("original_event_type is invalid")
        return value

    @field_validator("original_schema_version")
    @classmethod
    def _valid_payload_version(cls, value: str) -> str:
        return _schema_version(value, "original_schema_version")

    @field_validator("canonical_payload_json")
    @classmethod
    def _valid_canonical_payload(cls, value: str) -> str:
        parsed = _parse_json_object(value)
        canonical = _canonical_json(parsed)
        if value != canonical:
            raise ValueError("canonical_payload_json is not canonical JSON")
        return value

    @classmethod
    def from_payload_dict(
        cls,
        event_type: str,
        payload_schema_version: str,
        payload: Mapping[str, Any],
    ) -> OpaqueEventPayload:
        return cls(
            original_event_type=event_type,
            original_schema_version=payload_schema_version,
            canonical_payload_json=_canonical_json(dict(payload)),
        )

    def to_payload_dict(self) -> dict[str, Any]:
        return _parse_json_object(self.canonical_payload_json)


PayloadT = TypeVar("PayloadT", bound=EventPayload)
_PAYLOAD_REGISTRY: dict[tuple[str, str], type[EventPayload]] = {}


def register_payload(
    event_type: str,
    payload_schema_version: str,
):
    """Register a frozen payload model for a wire-level event identity."""

    if not isinstance(event_type, str) or not _EVENT_TYPE_PATTERN.fullmatch(event_type):
        raise PayloadRegistrationError(f"invalid event type {event_type!r}")
    try:
        _schema_version(payload_schema_version, "payload_schema_version")
    except (TypeError, ValueError) as exc:
        raise PayloadRegistrationError(str(exc)) from exc

    def decorator(payload_type: type[PayloadT]) -> type[PayloadT]:
        if not isinstance(payload_type, type) or not issubclass(payload_type, EventPayload):
            raise PayloadRegistrationError("payload model must subclass EventPayload")
        key = (event_type, payload_schema_version)
        existing = _PAYLOAD_REGISTRY.get(key)
        if existing is not None and existing is not payload_type:
            raise PayloadRegistrationError(
                f"payload {event_type}@{payload_schema_version} is already registered "
                f"to {existing.__name__}"
            )
        payload_type.EVENT_TYPE = event_type
        payload_type.PAYLOAD_SCHEMA_VERSION = payload_schema_version
        _PAYLOAD_REGISTRY[key] = payload_type
        return payload_type

    return decorator


def registered_payload_types() -> Mapping[tuple[str, str], type[EventPayload]]:
    """Return a read-only snapshot of the typed payload registry."""

    return MappingProxyType(dict(_PAYLOAD_REGISTRY))


def parse_payload(
    event_type: str,
    payload_schema_version: str,
    payload: Mapping[str, Any],
    *,
    preserve_unknown: bool = False,
) -> EventPayload:
    """Parse a payload strictly, optionally preserving unknown future schemas."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a JSON object")
    payload_type = _PAYLOAD_REGISTRY.get((event_type, payload_schema_version))
    if payload_type is None:
        if preserve_unknown:
            return OpaqueEventPayload.from_payload_dict(
                event_type, payload_schema_version, payload
            )
        raise UnknownPayloadTypeError(
            f"no payload model registered for {event_type}@{payload_schema_version}"
        )
    return payload_type.model_validate_json(_canonical_json(dict(payload)))


class ArtifactReference(_StrictFrozenModel):
    """Content-addressed activation artifact metadata stored in an event payload."""

    artifact_hash: str
    hook_name: str
    layer: int = Field(ge=0)
    token_selection: str
    aggregation: str
    shape: tuple[int, ...]
    dtype: str
    tokenizer_id: str
    model_revision: str
    source_model_call_id: str
    external_uri: str | None = None
    external_checksum: str | None = None

    @field_validator("artifact_hash")
    @classmethod
    def _valid_artifact_hash(cls, value: str) -> str:
        return _sha256(value, "artifact_hash")

    @field_validator(
        "hook_name",
        "token_selection",
        "aggregation",
        "dtype",
        "tokenizer_id",
        "model_revision",
        "source_model_call_id",
    )
    @classmethod
    def _valid_metadata_identifier(cls, value: str, info: ValidationInfo) -> str:
        return _stable_id(value, info.field_name or "artifact metadata")

    @field_validator("shape")
    @classmethod
    def _valid_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(isinstance(size, bool) or size < 0 for size in value):
            raise ValueError("shape must contain non-negative integer dimensions")
        return value

    @field_validator("external_checksum")
    @classmethod
    def _valid_external_checksum(cls, value: str | None) -> str | None:
        return None if value is None else _sha256(value, "external_checksum")

    @model_validator(mode="after")
    def _external_reference_is_checksummed(self) -> ArtifactReference:
        if (self.external_uri is None) != (self.external_checksum is None):
            raise ValueError("external_uri and external_checksum must be supplied together")
        if self.external_uri == "":
            raise ValueError("external_uri must not be empty")
        return self


@register_payload("ActivationCaptured", ACTIVATION_CAPTURED_PAYLOAD_SCHEMA_VERSION)
class ActivationCapturedPayload(EventPayload):
    """Typed payload linking one model call to one activation artifact."""

    artifact: ArtifactReference

    @property
    def artifact_refs(self) -> tuple[ArtifactReference, ...]:
        return (self.artifact,)


class EventEnvelope(_StrictFrozenModel):
    """Strict canonical envelope for one append-only experiment event.

    Hashing excludes ``recorded_at`` because it is provenance rather than
    scientific ordering, and excludes both hash fields.  Ordering comes from
    ``sequence_num``; stream validation is responsible for detecting cross-event
    duplicates, gaps, missing parents, and non-monotonic trial lanes.
    """

    schema_version: str = EVENT_ENVELOPE_SCHEMA_VERSION
    event_id: str
    event_type: str
    run_id: str
    pod_id: str
    trial_id: str | None
    dyad_id: str | None
    sequence_num: int = Field(ge=0)
    recorded_at: datetime
    actor_id: str | None
    actor_role: str | None
    model_call_id: str | None
    parent_event_ids: tuple[str, ...] = ()
    previous_event_hash: str | None
    payload: SerializeAsAny[EventPayload]
    payload_schema_version: str
    content_hash: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _parse_wire_payload(cls, value: Any, info: ValidationInfo) -> Any:
        if not isinstance(value, Mapping):
            return value
        parsed = dict(value)
        recorded_at = parsed.get("recorded_at")
        if isinstance(recorded_at, str):
            try:
                parsed["recorded_at"] = datetime.fromisoformat(
                    recorded_at[:-1] + "+00:00"
                    if recorded_at.endswith("Z")
                    else recorded_at
                )
            except ValueError:
                # Leave malformed input for Pydantic to report against the field.
                pass
        parent_event_ids = parsed.get("parent_event_ids")
        if isinstance(parent_event_ids, list):
            parsed["parent_event_ids"] = tuple(parent_event_ids)
        raw_payload = value.get("payload")
        if isinstance(raw_payload, EventPayload):
            return parsed
        if not isinstance(raw_payload, Mapping):
            return parsed
        event_type = value.get("event_type")
        payload_version = value.get("payload_schema_version")
        if not isinstance(event_type, str) or not isinstance(payload_version, str):
            return parsed
        preserve_unknown = bool(
            info.context and info.context.get("preserve_unknown_payloads", False)
        )
        parsed["payload"] = parse_payload(
            event_type,
            payload_version,
            raw_payload,
            preserve_unknown=preserve_unknown,
        )
        return parsed

    @field_validator("schema_version")
    @classmethod
    def _supported_envelope_version(cls, value: str) -> str:
        if value != EVENT_ENVELOPE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported event envelope schema {value!r}; expected "
                f"{EVENT_ENVELOPE_SCHEMA_VERSION!r}"
            )
        return value

    @field_validator("event_id", "run_id")
    @classmethod
    def _valid_uuid_ids(cls, value: str, info: ValidationInfo) -> str:
        return _canonical_uuid(value, info.field_name or "UUID")

    @field_validator("event_type")
    @classmethod
    def _valid_event_type(cls, value: str) -> str:
        if not _EVENT_TYPE_PATTERN.fullmatch(value):
            raise ValueError("event_type must be a PascalCase identifier")
        return value

    @field_validator(
        "pod_id", "trial_id", "dyad_id", "actor_id", "actor_role", "model_call_id"
    )
    @classmethod
    def _valid_stable_ids(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _stable_id(value, info.field_name or "identifier")

    @field_validator("parent_event_ids")
    @classmethod
    def _valid_parent_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        parents = tuple(_canonical_uuid(parent, "parent_event_ids") for parent in value)
        if len(parents) != len(set(parents)):
            raise ValueError("parent_event_ids must not contain duplicates")
        return parents

    @field_validator("recorded_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware UTC")
        if value.utcoffset() != timedelta(0):
            raise ValueError("recorded_at must be UTC, not merely timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("payload_schema_version")
    @classmethod
    def _valid_payload_schema_version(cls, value: str) -> str:
        return _schema_version(value, "payload_schema_version")

    @field_validator("previous_event_hash", "content_hash")
    @classmethod
    def _valid_hashes(cls, value: str | None, info: ValidationInfo) -> str | None:
        return None if value is None else _sha256(value, info.field_name or "hash")

    @model_validator(mode="after")
    def _validate_identity_and_hash(self) -> EventEnvelope:
        if self.event_id in self.parent_event_ids:
            raise ValueError("an event cannot be its own parent")
        if (self.actor_id is None) != (self.actor_role is None):
            raise ValueError("actor_id and actor_role must be supplied together")
        if self.sequence_num == 0 and self.previous_event_hash is not None:
            raise ValueError("sequence 0 must not have a previous_event_hash")
        if self.sequence_num > 0 and self.previous_event_hash is None:
            raise ValueError("positive sequence_num requires previous_event_hash")

        if isinstance(self.payload, OpaqueEventPayload):
            if self.payload.original_event_type != self.event_type:
                raise ValueError("opaque payload event type does not match envelope")
            if self.payload.original_schema_version != self.payload_schema_version:
                raise ValueError("opaque payload schema does not match envelope")
        else:
            if self.payload.EVENT_TYPE != self.event_type:
                raise ValueError("registered payload event type does not match envelope")
            if self.payload.PAYLOAD_SCHEMA_VERSION != self.payload_schema_version:
                raise ValueError("registered payload schema does not match envelope")

        expected_hash = self.compute_content_hash()
        if self.content_hash is None:
            object.__setattr__(self, "content_hash", expected_hash)
        elif self.content_hash != expected_hash:
            raise EventHashMismatchError(
                f"event {self.event_id} content_hash mismatch: supplied "
                f"{self.content_hash}, expected {expected_hash}"
            )
        return self

    @property
    def artifact_refs(self) -> tuple[ArtifactReference, ...]:
        refs = getattr(self.payload, "artifact_refs", ())
        return tuple(refs)

    def semantic_dict(self) -> dict[str, Any]:
        """Return the canonical scientific content, excluding time and hashes."""

        value = self.to_dict()
        value.pop("recorded_at")
        value.pop("previous_event_hash")
        value.pop("content_hash")
        return value

    def canonical_semantic_json(self) -> str:
        return _canonical_json(self.semantic_dict())

    def compute_content_hash(self) -> str:
        previous = self.previous_event_hash or ""
        material = previous + self.canonical_semantic_json()
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def verify_content_hash(self) -> None:
        expected = self.compute_content_hash()
        if self.content_hash != expected:
            raise EventHashMismatchError(
                f"event {self.event_id} content_hash mismatch: supplied "
                f"{self.content_hash}, expected {expected}"
            )

    def semantically_equal(self, other: object) -> bool:
        return isinstance(other, EventEnvelope) and (
            self.canonical_semantic_json() == other.canonical_semantic_json()
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSONL wire envelope with canonical UTC formatting."""

        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "pod_id": self.pod_id,
            "trial_id": self.trial_id,
            "dyad_id": self.dyad_id,
            "sequence_num": self.sequence_num,
            "recorded_at": self.recorded_at.isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            ),
            "actor_id": self.actor_id,
            "actor_role": self.actor_role,
            "model_call_id": self.model_call_id,
            "parent_event_ids": list(self.parent_event_ids),
            "previous_event_hash": self.previous_event_hash,
            "payload": self.payload.to_payload_dict(),
            "payload_schema_version": self.payload_schema_version,
            "content_hash": self.content_hash,
        }

    def to_json(self) -> str:
        """Serialize this envelope as canonical compact JSON."""

        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        preserve_unknown_payloads: bool = False,
    ) -> EventEnvelope:
        """Validate an exact wire envelope, failing closed on unknown fields."""

        if not isinstance(value, Mapping):
            raise TypeError("event envelope must be a mapping")
        fields = frozenset(value)
        if fields != _EVENT_FIELDS:
            missing = sorted(_EVENT_FIELDS - fields)
            unknown = sorted(fields - _EVENT_FIELDS)
            raise ValueError(
                f"event envelope fields are not exact; missing={missing}, unknown={unknown}"
            )
        return cls.model_validate_json(
            _canonical_json(dict(value)),
            context={"preserve_unknown_payloads": preserve_unknown_payloads},
        )

    @classmethod
    def from_json(
        cls,
        value: str | bytes,
        *,
        preserve_unknown_payloads: bool = False,
    ) -> EventEnvelope:
        """Parse canonical or ordinary JSON and apply the strict wire contract."""

        if isinstance(value, bytes):
            value = value.decode("utf-8")
        parsed = _parse_json_object(value)
        return cls.from_dict(
            parsed, preserve_unknown_payloads=preserve_unknown_payloads
        )


def new_run_id() -> str:
    """Allocate a unique canonical UUID for a run."""

    return str(uuid4())


def new_event_id() -> str:
    """Allocate a unique canonical UUID for an event."""

    return str(uuid4())


def deterministic_trial_id(
    run_id: str, scenario_instance_id: str, trial_seed: int
) -> str:
    """Derive a stable trial UUID from run, scenario instance, and seed."""

    _canonical_uuid(run_id, "run_id")
    _stable_id(scenario_instance_id, "scenario_instance_id")
    if isinstance(trial_seed, bool) or not isinstance(trial_seed, int):
        raise TypeError("trial_seed must be an integer")
    return str(uuid5(_IDENTITY_NAMESPACE, f"trial:{run_id}:{scenario_instance_id}:{trial_seed}"))


def deterministic_dyad_id(
    run_id: str, trial_family_id: str, actor_ids: tuple[str, str]
) -> str:
    """Derive an order-independent stable UUID for a dyad."""

    _canonical_uuid(run_id, "run_id")
    _stable_id(trial_family_id, "trial_family_id")
    if len(actor_ids) != 2 or actor_ids[0] == actor_ids[1]:
        raise ValueError("actor_ids must contain two distinct actor identities")
    actors = sorted(_stable_id(actor, "actor_ids") for actor in actor_ids)
    return str(uuid5(_IDENTITY_NAMESPACE, f"dyad:{run_id}:{trial_family_id}:{actors[0]}:{actors[1]}"))


def deterministic_model_call_id(
    run_id: str,
    trial_id: str,
    sequence_num: int,
    actor_id: str,
    purpose: str,
) -> str:
    """Allocate a reproducible call identity before model inference begins."""

    _canonical_uuid(run_id, "run_id")
    _stable_id(trial_id, "trial_id")
    _stable_id(actor_id, "actor_id")
    _stable_id(purpose, "purpose")
    if isinstance(sequence_num, bool) or not isinstance(sequence_num, int):
        raise TypeError("sequence_num must be an integer")
    if sequence_num < 0:
        raise ValueError("sequence_num must be non-negative")
    return str(
        uuid5(
            _IDENTITY_NAMESPACE,
            f"call:{run_id}:{trial_id}:{sequence_num}:{actor_id}:{purpose}",
        )
    )


__all__ = [
    "ACTIVATION_CAPTURED_PAYLOAD_SCHEMA_VERSION",
    "EVENT_ENVELOPE_SCHEMA_VERSION",
    "ActivationCapturedPayload",
    "ArtifactReference",
    "EventEnvelope",
    "EventHashMismatchError",
    "EventPayload",
    "OpaqueEventPayload",
    "PayloadRegistrationError",
    "UnknownPayloadTypeError",
    "deterministic_dyad_id",
    "deterministic_model_call_id",
    "deterministic_trial_id",
    "new_event_id",
    "new_run_id",
    "parse_payload",
    "register_payload",
    "registered_payload_types",
]
