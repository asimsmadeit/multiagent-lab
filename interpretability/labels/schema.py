"""Canonical label records and explicit compatibility projection policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping


LABEL_SCHEMA_VERSION = "1.2.0"
_LABEL_RECORD_FIELDS = frozenset({
    "schema_version",
    "subject_actor_id",
    "behavior_target",
    "value",
    "status",
    "source",
    "target_event_id",
    "evaluation_event_id",
    "evaluator_version",
    "evidence_event_ids",
    "confidence",
    "severity",
    "evaluation_error",
    "fallback_reason",
    "metadata",
    "label_id",
})


def _require_exact_label_fields(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("serialized label record must be a mapping")
    actual = set(value)
    missing = sorted(_LABEL_RECORD_FIELDS - actual)
    unknown = sorted(actual - _LABEL_RECORD_FIELDS)
    if missing:
        raise ValueError(
            "serialized label record is missing fields: " + ", ".join(missing)
        )
    if unknown:
        raise ValueError(
            "serialized label record has unknown fields: " + ", ".join(unknown)
        )


class LabelValue(str, Enum):
    """Tri-state label value."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class LabelStatus(str, Enum):
    """Whether and how an assessment is available."""

    AVAILABLE = "available"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    CONFLICTED = "conflicted"


class LabelSource(str, Enum):
    """Independent source of an assessment."""

    RULE = "rule"
    GM = "gm"
    MODEL_JUDGE = "model_judge"
    HUMAN = "human"
    AGENT_PERCEPTION = "agent_perception"
    UNKNOWN = "unknown"


class BehaviorTarget(str, Enum):
    """Construct being assessed; constructs are not silently merged."""

    FACTUAL_DECEPTION = "factual_deception"
    DECEPTIVE_INTENT = "deceptive_intent"
    STRATEGIC_OMISSION = "strategic_omission"
    COMMITMENT_VIOLATION = "commitment_violation"
    MANIPULATION = "manipulation"
    CONSISTENCY = "consistency"
    COUNTERPART_DECEPTION_ESTIMATE = "counterpart_deception_estimate"


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("label metadata keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("label metadata floats must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"label metadata must be JSON-safe, got {type(value).__name__}")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _label_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"label_{digest[:24]}"


@dataclass(frozen=True)
class LabelRecord:
    """One append-only assessment from one source about one subject event."""

    subject_actor_id: str
    behavior_target: BehaviorTarget
    value: LabelValue
    status: LabelStatus
    source: LabelSource
    target_event_id: str | None
    evaluation_event_id: str | None
    evaluator_version: str
    evidence_event_ids: tuple[str, ...] = ()
    confidence: float | None = None
    severity: float | None = None
    evaluation_error: str | None = None
    fallback_reason: str | None = None
    metadata: Mapping[str, Any] | tuple[tuple[str, Any], ...] = field(
        default_factory=dict
    )
    schema_version: str = LABEL_SCHEMA_VERSION
    label_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != LABEL_SCHEMA_VERSION:
            raise ValueError("unsupported label record schema version")
        if not isinstance(self.subject_actor_id, str) or not self.subject_actor_id:
            raise ValueError("subject_actor_id must be non-empty")
        if not isinstance(self.evaluator_version, str) or not self.evaluator_version:
            raise ValueError("evaluator_version must be non-empty")
        for name in ("target_event_id", "evaluation_event_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be null or a non-empty string")
        evidence = tuple(self.evidence_event_ids)
        if any(not isinstance(item, str) or not item for item in evidence):
            raise ValueError("evidence_event_ids must contain non-empty strings")
        object.__setattr__(self, "evidence_event_ids", evidence)
        object.__setattr__(self, "metadata", _freeze_json(dict(self.metadata)))
        if self.status is LabelStatus.AVAILABLE:
            if self.value is LabelValue.UNKNOWN:
                raise ValueError("an available label must have a true or false value")
            if self.source is LabelSource.UNKNOWN:
                raise ValueError("an available label must have a known source")
            if not self.target_event_id:
                raise ValueError("an available label must identify its target event")
        elif self.value is not LabelValue.UNKNOWN:
            raise ValueError("a non-available label must have value='unknown'")
        if (
            self.source is LabelSource.AGENT_PERCEPTION
            and self.behavior_target
            is not BehaviorTarget.COUNTERPART_DECEPTION_ESTIMATE
        ):
            raise ValueError(
                "agent perception may only assess counterpart deception estimates"
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.severity is not None and not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be between 0 and 1")
        if len(set(evidence)) != len(evidence):
            raise ValueError("evidence_event_ids must not contain duplicates")
        if self.status is LabelStatus.UNKNOWN and not (
            self.evaluation_error or self.fallback_reason
        ):
            raise ValueError("an unknown label must retain a reason or error")

        payload = self.to_dict(include_id=False)
        object.__setattr__(self, "label_id", _label_id(payload))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        """Return a JSON-safe representation."""
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "subject_actor_id": self.subject_actor_id,
            "behavior_target": self.behavior_target.value,
            "value": self.value.value,
            "status": self.status.value,
            "source": self.source.value,
            "target_event_id": self.target_event_id,
            "evaluation_event_id": self.evaluation_event_id,
            "evaluator_version": self.evaluator_version,
            "evidence_event_ids": list(self.evidence_event_ids),
            "confidence": self.confidence,
            "severity": self.severity,
            "evaluation_error": self.evaluation_error,
            "fallback_reason": self.fallback_reason,
            "metadata": _thaw_json(self.metadata),
        }
        if include_id:
            result["label_id"] = self.label_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LabelRecord":
        """Validate and restore a serialized record, including its stable ID."""
        if not isinstance(value, Mapping):
            raise TypeError("serialized label record must be a mapping")
        if value.get("schema_version") != LABEL_SCHEMA_VERSION:
            raise ValueError("unsupported label record schema version")
        serialized_id = value.get("label_id")
        if not isinstance(serialized_id, str) or not serialized_id:
            raise ValueError("label_id is required for persisted label records")
        _require_exact_label_fields(value)
        if not isinstance(value["evidence_event_ids"], list):
            raise TypeError("serialized evidence_event_ids must be an array")
        if not isinstance(value["metadata"], Mapping):
            raise TypeError("serialized label metadata must be a mapping")
        record = cls(
            schema_version=value["schema_version"],
            subject_actor_id=value["subject_actor_id"],
            behavior_target=BehaviorTarget(value["behavior_target"]),
            value=LabelValue(value["value"]),
            status=LabelStatus(value["status"]),
            source=LabelSource(value["source"]),
            target_event_id=value["target_event_id"],
            evaluation_event_id=value["evaluation_event_id"],
            evaluator_version=value["evaluator_version"],
            evidence_event_ids=tuple(value["evidence_event_ids"]),
            confidence=value["confidence"],
            severity=value["severity"],
            evaluation_error=value["evaluation_error"],
            fallback_reason=value["fallback_reason"],
            metadata=dict(value["metadata"]),
        )
        if serialized_id != record.label_id:
            raise ValueError("serialized label_id does not match record content")
        return record


@dataclass(frozen=True)
class LabelSourcePolicy:
    """Versioned source order for one compatibility projection."""

    behavior_target: BehaviorTarget
    source_order: tuple[LabelSource, ...]
    policy_version: str
    allow_fallback: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_order", tuple(self.source_order))
        if not self.policy_version:
            raise ValueError("policy_version must be non-empty")
        if not self.source_order:
            raise ValueError("source_order must not be empty")
        if len(set(self.source_order)) != len(self.source_order):
            raise ValueError("source_order must not contain duplicates")
        if LabelSource.AGENT_PERCEPTION in self.source_order:
            raise ValueError(
                "agent perception cannot be an actual-behavior projection source"
            )
        if len(self.source_order) > 1 and not self.allow_fallback:
            raise ValueError("multiple sources require allow_fallback=True")


@dataclass(frozen=True)
class LabelProjection:
    """Traceable scalar compatibility view over retained source records."""

    behavior_target: BehaviorTarget
    value: LabelValue
    status: LabelStatus
    policy_version: str
    selected_record_id: str | None
    selected_source: LabelSource | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "behavior_target": self.behavior_target.value,
            "value": self.value.value,
            "status": self.status.value,
            "policy_version": self.policy_version,
            "selected_record_id": self.selected_record_id,
            "selected_source": (
                None if self.selected_source is None else self.selected_source.value
            ),
            "reason": self.reason,
        }


def project_label(
    records: Iterable[LabelRecord],
    policy: LabelSourcePolicy,
    *,
    subject_actor_id: str,
    target_event_id: str | None,
) -> LabelProjection:
    """Project records without overwriting disagreement or coercing unknown."""
    candidates = [
        record
        for record in records
        if record.behavior_target is policy.behavior_target
        and record.subject_actor_id == subject_actor_id
        and record.target_event_id == target_event_id
    ]
    for source in policy.source_order:
        source_records = [record for record in candidates if record.source is source]
        available = [
            record for record in source_records
            if record.status is LabelStatus.AVAILABLE
        ]
        values = {record.value for record in available}
        if len(values) > 1:
            return LabelProjection(
                behavior_target=policy.behavior_target,
                value=LabelValue.UNKNOWN,
                status=LabelStatus.CONFLICTED,
                policy_version=policy.policy_version,
                selected_record_id=None,
                selected_source=source,
                reason="selected source contains conflicting available records",
            )
        if available:
            selected = sorted(available, key=lambda item: item.label_id)[0]
            return LabelProjection(
                behavior_target=policy.behavior_target,
                value=selected.value,
                status=LabelStatus.AVAILABLE,
                policy_version=policy.policy_version,
                selected_record_id=selected.label_id,
                selected_source=selected.source,
            )
        if source_records and not policy.allow_fallback:
            return LabelProjection(
                behavior_target=policy.behavior_target,
                value=LabelValue.UNKNOWN,
                status=LabelStatus.UNKNOWN,
                policy_version=policy.policy_version,
                selected_record_id=None,
                selected_source=source,
                reason="selected source has no available assessment",
            )

    return LabelProjection(
        behavior_target=policy.behavior_target,
        value=LabelValue.UNKNOWN,
        status=LabelStatus.UNKNOWN,
        policy_version=policy.policy_version,
        selected_record_id=None,
        selected_source=None,
        reason="no eligible source record",
    )
