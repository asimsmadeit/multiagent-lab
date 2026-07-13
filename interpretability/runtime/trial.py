"""Replayable trial state machine independent of model execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping


TRIAL_RUNTIME_SCHEMA_VERSION = "1.7.0"


class TrialState(str, Enum):
    CREATED = "created"
    COMPILED = "compiled"
    AGENTS_BUILT = "agents_built"
    INITIALIZED = "initialized"
    INTERVENTION_APPLIED = "intervention_applied"
    TURN_PROPOSED = "turn_proposed"
    ACTION_CAPTURED = "action_captured"
    ADJUDICATED = "adjudicated"
    BATCH_PROPOSED = "batch_proposed"
    BATCH_CAPTURED = "batch_captured"
    BATCH_ADJUDICATED = "batch_adjudicated"
    OBSERVED = "observed"
    COMPLETED = "completed"
    FAILED = "failed"


_ALLOWED_TRANSITIONS = {
    TrialState.CREATED: {TrialState.COMPILED, TrialState.FAILED},
    TrialState.COMPILED: {TrialState.AGENTS_BUILT, TrialState.FAILED},
    TrialState.AGENTS_BUILT: {TrialState.INITIALIZED, TrialState.FAILED},
    TrialState.INITIALIZED: {
        TrialState.INTERVENTION_APPLIED,
        TrialState.TURN_PROPOSED,
        TrialState.BATCH_PROPOSED,
        TrialState.FAILED,
    },
    TrialState.INTERVENTION_APPLIED: {
        TrialState.INTERVENTION_APPLIED,
        TrialState.TURN_PROPOSED,
        TrialState.BATCH_PROPOSED,
        TrialState.OBSERVED,
        TrialState.COMPLETED,
        TrialState.FAILED,
    },
    TrialState.TURN_PROPOSED: {TrialState.ACTION_CAPTURED, TrialState.FAILED},
    TrialState.ACTION_CAPTURED: {TrialState.ADJUDICATED, TrialState.FAILED},
    TrialState.BATCH_PROPOSED: {
        TrialState.BATCH_CAPTURED,
        TrialState.FAILED,
    },
    TrialState.BATCH_CAPTURED: {
        TrialState.BATCH_ADJUDICATED,
        TrialState.FAILED,
    },
    TrialState.BATCH_ADJUDICATED: {
        TrialState.OBSERVED,
        TrialState.COMPLETED,
        TrialState.FAILED,
    },
    TrialState.ADJUDICATED: {
        TrialState.INTERVENTION_APPLIED,
        TrialState.OBSERVED,
        TrialState.COMPLETED,
        TrialState.FAILED,
    },
    TrialState.OBSERVED: {
        TrialState.INTERVENTION_APPLIED,
        TrialState.TURN_PROPOSED,
        TrialState.BATCH_PROPOSED,
        TrialState.COMPLETED,
        TrialState.FAILED,
    },
    TrialState.COMPLETED: set(),
    TrialState.FAILED: set(),
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("trial event payload mapping keys must be strings")
            frozen[key] = _freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("trial event payload floats must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"trial event payload is not JSON-safe: {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _required_identifier(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_index(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _validate_identifier_list(
    payload: Mapping[str, Any],
    key: str,
    *,
    minimum: int = 0,
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be an array of identifiers")
    identifiers = tuple(value)
    if len(identifiers) < minimum or any(
        not isinstance(item, str) or not item for item in identifiers
    ):
        raise ValueError(f"{key} must contain non-empty string identifiers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{key} must not contain duplicate identifiers")
    return identifiers


def _validate_index_list(
    payload: Mapping[str, Any],
    key: str,
    *,
    length: int,
) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{key} must contain one index per batch actor")
    indices = tuple(value)
    if any(type(item) is not int or item < 0 for item in indices):
        raise ValueError(f"{key} must contain non-negative integers")
    return indices


def _validate_text_list(
    payload: Mapping[str, Any],
    key: str,
    *,
    length: int,
) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{key} must contain one string per batch actor")
    strings = tuple(value)
    if any(not isinstance(item, str) for item in strings):
        raise ValueError(f"{key} must contain strings")
    return strings


def _validate_boolean_list(
    payload: Mapping[str, Any],
    key: str,
    *,
    length: int,
) -> tuple[bool, ...]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{key} must contain one boolean per batch actor")
    booleans = tuple(value)
    if any(type(item) is not bool for item in booleans):
        raise ValueError(f"{key} must contain booleans")
    return booleans


def _validate_nested_identifier_lists(
    payload: Mapping[str, Any],
    key: str,
    *,
    length: int,
) -> tuple[tuple[str, ...], ...]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{key} must contain one identifier array per batch actor")
    result = []
    for identifiers in value:
        if not isinstance(identifiers, (list, tuple)) or any(
            not isinstance(item, str) or not item for item in identifiers
        ):
            raise ValueError(f"{key} must contain identifier arrays")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{key} arrays must not contain duplicates")
        result.append(tuple(identifiers))
    return tuple(result)


def _validate_state_payload(
    to_state: TrialState,
    payload: Mapping[str, Any],
) -> None:
    """Validate the required record references for one target state."""
    if not isinstance(payload, Mapping):
        raise TypeError("trial transition payload must be a mapping")
    if to_state is TrialState.COMPILED:
        _required_identifier(payload, "scenario_instance_id")
    elif to_state is TrialState.AGENTS_BUILT:
        _validate_identifier_list(payload, "actors", minimum=2)
    elif to_state is TrialState.INITIALIZED:
        _required_index(payload, "round_index")
        next_actor = payload.get("next_actor")
        if next_actor is not None and (
            not isinstance(next_actor, str) or not next_actor
        ):
            raise ValueError("next_actor must be null or a non-empty string")
    elif to_state is TrialState.INTERVENTION_APPLIED:
        _required_identifier(payload, "application_receipt_id")
        _required_identifier(payload, "intervention_design_id")
        _required_identifier(payload, "intervention_family")
        _required_identifier(payload, "target_actor_id")
        _required_index(payload, "round_index")
        _required_index(payload, "committed_action_boundary")
        _required_identifier(payload, "content_hash")
        _required_identifier(payload, "source")
        _required_identifier(payload, "status")
        observation = payload.get("observation")
        if observation is not None and not isinstance(observation, str):
            raise ValueError("observation must be null or a string")
        if "evidence_call_id" not in payload:
            raise ValueError("evidence_call_id must be explicitly present")
        evidence_call_id = payload["evidence_call_id"]
        if evidence_call_id is not None and (
            not isinstance(evidence_call_id, str) or not evidence_call_id
        ):
            raise ValueError(
                "evidence_call_id must be null or a non-empty string"
            )
    elif to_state is TrialState.TURN_PROPOSED:
        _required_identifier(payload, "actor_id")
        _required_index(payload, "round_index")
        _required_identifier(payload, "generation_call_id")
        _required_index(payload, "attempt")
    elif to_state is TrialState.ACTION_CAPTURED:
        _required_identifier(payload, "actor_id")
        _required_index(payload, "round_index")
        if not isinstance(payload.get("output_text"), str):
            raise ValueError("output_text must be a string")
        _required_identifier(payload, "generation_record_id")
    elif to_state is TrialState.BATCH_PROPOSED:
        _required_index(payload, "round_index")
        actors = _validate_identifier_list(payload, "actor_ids", minimum=2)
        calls = _validate_identifier_list(
            payload, "generation_call_ids", minimum=2
        )
        if len(calls) != len(actors):
            raise ValueError(
                "generation_call_ids must contain one call per batch actor"
            )
        _validate_index_list(payload, "attempts", length=len(actors))
    elif to_state is TrialState.BATCH_CAPTURED:
        _required_index(payload, "round_index")
        actors = _validate_identifier_list(payload, "actor_ids", minimum=2)
        records = _validate_identifier_list(
            payload, "generation_record_ids", minimum=2
        )
        if len(records) != len(actors):
            raise ValueError(
                "generation_record_ids must contain one record per batch actor"
            )
        _validate_text_list(payload, "output_texts", length=len(actors))
    elif to_state is TrialState.ADJUDICATED:
        _required_identifier(payload, "actor_id")
        _required_identifier(payload, "interaction_event_id")
        _required_identifier(payload, "resolution_id")
        _required_identifier(payload, "action_id")
        if not isinstance(payload.get("accepted"), bool):
            raise ValueError("accepted must be a boolean")
    elif to_state is TrialState.BATCH_ADJUDICATED:
        _required_index(payload, "round_index")
        actors = _validate_identifier_list(payload, "actor_ids", minimum=2)
        for key in (
            "interaction_event_ids",
            "resolution_ids",
            "action_ids",
        ):
            identifiers = _validate_identifier_list(payload, key, minimum=2)
            if len(identifiers) != len(actors):
                raise ValueError(f"{key} must contain one ID per batch actor")
        _validate_boolean_list(payload, "accepted", length=len(actors))
        _validate_nested_identifier_lists(
            payload, "label_record_ids", length=len(actors)
        )
    elif to_state is TrialState.OBSERVED:
        if "actor_ids" in payload:
            actors = _validate_identifier_list(payload, "actor_ids", minimum=2)
            events = _validate_identifier_list(
                payload, "interaction_event_ids", minimum=2
            )
            if len(events) != len(actors):
                raise ValueError(
                    "interaction_event_ids must contain one ID per batch actor"
                )
            _validate_nested_identifier_lists(
                payload, "label_record_ids", length=len(actors)
            )
        else:
            _required_identifier(payload, "actor_id")
            _required_identifier(payload, "interaction_event_id")
            _validate_identifier_list(payload, "label_record_ids")
        if not isinstance(payload.get("observation"), str):
            raise ValueError("observation must be a string")
    elif to_state is TrialState.COMPLETED:
        _required_identifier(payload, "reason")
        for key in ("outcome_id", "interaction_event_id"):
            value = payload.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{key} must be null or a non-empty string")
        if "label_record_ids" in payload:
            _validate_identifier_list(payload, "label_record_ids")
        if "interaction_event_ids" in payload:
            _validate_identifier_list(
                payload, "interaction_event_ids", minimum=2
            )
        if "committed_actions" in payload:
            _required_index(payload, "committed_actions")
    elif to_state is TrialState.FAILED:
        _required_identifier(payload, "error_type")
        _required_identifier(payload, "error")
        if "interaction_event_id" in payload:
            _required_identifier(payload, "interaction_event_id")


@dataclass(frozen=True)
class TrialRuntimeEvent:
    """One state transition with references to canonical scientific records."""

    run_id: str
    trial_id: str
    attempt: int
    sequence: int
    from_state: TrialState
    to_state: TrialState
    payload: Mapping[str, Any]
    schema_version: str = TRIAL_RUNTIME_SCHEMA_VERSION
    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or not self.run_id
            or not isinstance(self.trial_id, str)
            or not self.trial_id
        ):
            raise ValueError("run_id and trial_id must be non-empty")
        if type(self.attempt) is not int or type(self.sequence) is not int:
            raise TypeError("attempt and sequence must be integers")
        if self.attempt < 0 or self.sequence < 0:
            raise ValueError("attempt and sequence must be non-negative")
        if self.to_state not in _ALLOWED_TRANSITIONS[self.from_state]:
            raise ValueError(
                f"invalid trial transition: {self.from_state.value} -> "
                f"{self.to_state.value}"
            )
        if self.schema_version != TRIAL_RUNTIME_SCHEMA_VERSION:
            raise ValueError("unsupported trial runtime schema version")
        _validate_state_payload(self.to_state, self.payload)
        object.__setattr__(self, "payload", _freeze(self.payload))
        digest = hashlib.sha256(
            _canonical_json(self.to_dict(include_id=False)).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "event_id", f"trial_event_{digest[:24]}")

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "attempt": self.attempt,
            "sequence": self.sequence,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "payload": _thaw(self.payload),
        }
        if include_id:
            result["event_id"] = self.event_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialRuntimeEvent":
        if not isinstance(value, Mapping):
            raise TypeError("serialized trial event must be a mapping")
        expected = {
            "schema_version",
            "run_id",
            "trial_id",
            "attempt",
            "sequence",
            "from_state",
            "to_state",
            "payload",
            "event_id",
        }
        missing = expected.difference(value)
        unknown = set(value).difference(expected)
        if missing:
            raise ValueError(
                "serialized trial event is missing fields: "
                + ", ".join(sorted(missing))
            )
        if unknown:
            raise ValueError(
                "serialized trial event has unknown fields: "
                + ", ".join(sorted(unknown))
            )
        if value.get("schema_version") != TRIAL_RUNTIME_SCHEMA_VERSION:
            raise ValueError("unsupported trial runtime schema version")
        serialized_id = value.get("event_id")
        if not isinstance(serialized_id, str) or not serialized_id:
            raise ValueError("event_id is required for persisted trial events")
        event = cls(
            schema_version=value["schema_version"],
            run_id=value["run_id"],
            trial_id=value["trial_id"],
            attempt=value["attempt"],
            sequence=value["sequence"],
            from_state=TrialState(value["from_state"]),
            to_state=TrialState(value["to_state"]),
            payload=value.get("payload", {}),
        )
        if serialized_id != event.event_id:
            raise ValueError("serialized trial event ID does not match content")
        return event


class TrialRunner:
    """Append-only state coordinator; model calls happen outside this object."""

    def __init__(self, *, run_id: str, trial_id: str, attempt: int = 0) -> None:
        if (
            not isinstance(run_id, str)
            or not run_id
            or not isinstance(trial_id, str)
            or not trial_id
            or type(attempt) is not int
            or attempt < 0
        ):
            raise ValueError("valid run_id, trial_id, and attempt are required")
        self.run_id = run_id
        self.trial_id = trial_id
        self.attempt = attempt
        self._state = TrialState.CREATED
        self._events: list[TrialRuntimeEvent] = []

    @property
    def state(self) -> TrialState:
        return self._state

    @property
    def events(self) -> tuple[TrialRuntimeEvent, ...]:
        return tuple(self._events)

    def transition(
        self,
        to_state: TrialState,
        payload: Mapping[str, Any] | None = None,
    ) -> TrialRuntimeEvent:
        event = TrialRuntimeEvent(
            run_id=self.run_id,
            trial_id=self.trial_id,
            attempt=self.attempt,
            sequence=len(self._events),
            from_state=self._state,
            to_state=to_state,
            payload=payload or {},
        )
        self._validate_event_lineage(event)
        self._events.append(event)
        self._state = to_state
        return event

    def get_state(self) -> dict[str, Any]:
        return {
            "schema_version": TRIAL_RUNTIME_SCHEMA_VERSION,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "attempt": self.attempt,
            "state": self._state.value,
            "events": [event.to_dict() for event in self._events],
        }

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> "TrialRunner":
        if not isinstance(value, Mapping):
            raise TypeError("serialized trial runner must be a mapping")
        expected = {
            "schema_version",
            "run_id",
            "trial_id",
            "attempt",
            "state",
            "events",
        }
        missing = expected.difference(value)
        unknown = set(value).difference(expected)
        if missing:
            raise ValueError(
                "serialized trial runner is missing fields: "
                + ", ".join(sorted(missing))
            )
        if unknown:
            raise ValueError(
                "serialized trial runner has unknown fields: "
                + ", ".join(sorted(unknown))
            )
        if value.get("schema_version") != TRIAL_RUNTIME_SCHEMA_VERSION:
            raise ValueError("unsupported trial runner schema version")
        events = value.get("events")
        if not isinstance(events, (list, tuple)):
            raise ValueError("events must be an array of persisted trial events")
        runner = cls(
            run_id=value["run_id"],
            trial_id=value["trial_id"],
            attempt=value["attempt"],
        )
        for serialized in events:
            event = TrialRuntimeEvent.from_dict(serialized)
            if event.run_id != runner.run_id or event.trial_id != runner.trial_id:
                raise ValueError("trial event belongs to another run or trial")
            if event.attempt != runner.attempt:
                raise ValueError("trial event attempt does not match runner")
            if event.sequence != len(runner._events):
                raise ValueError("trial event sequence is not contiguous")
            if event.from_state is not runner._state:
                raise ValueError("trial event state lineage is invalid")
            runner._validate_event_lineage(event)
            runner._events.append(event)
            runner._state = event.to_state
        if TrialState(value["state"]) is not runner._state:
            raise ValueError("serialized trial state does not match event replay")
        return runner

    def _validate_event_lineage(self, event: TrialRuntimeEvent) -> None:
        """Bind per-state payload references to the immediately prior boundary."""
        if not self._events:
            return
        previous = self._events[-1]
        payload = _thaw(event.payload)
        prior_payload = _thaw(previous.payload)
        lineage_previous = previous
        if (
            previous.to_state is TrialState.INTERVENTION_APPLIED
            and event.to_state in {TrialState.OBSERVED, TrialState.COMPLETED}
        ):
            lineage_previous = next(
                (
                    candidate for candidate in reversed(self._events)
                    if candidate.to_state
                    is not TrialState.INTERVENTION_APPLIED
                ),
                previous,
            )
            if (
                event.to_state is TrialState.OBSERVED
                and lineage_previous.to_state not in {
                    TrialState.ADJUDICATED,
                    TrialState.BATCH_ADJUDICATED,
                }
            ):
                raise ValueError(
                    'Post-action intervention lacks an adjudicated boundary'
                )
            prior_payload = _thaw(lineage_previous.payload)
        if event.to_state is TrialState.ACTION_CAPTURED:
            if payload["actor_id"] != prior_payload["actor_id"]:
                raise ValueError("captured actor_id does not match proposed actor")
            if payload["round_index"] != prior_payload["round_index"]:
                raise ValueError("captured round_index does not match proposal")
            if payload["generation_record_id"] != prior_payload["generation_call_id"]:
                raise ValueError("captured generation record does not match proposed call")
        elif event.to_state is TrialState.BATCH_CAPTURED:
            if payload["actor_ids"] != prior_payload["actor_ids"]:
                raise ValueError("captured batch actors do not match proposed actors")
            if payload["round_index"] != prior_payload["round_index"]:
                raise ValueError("captured batch round does not match proposal")
            if (
                payload["generation_record_ids"]
                != prior_payload["generation_call_ids"]
            ):
                raise ValueError(
                    "captured batch records do not match proposed calls"
                )
        elif event.to_state is TrialState.ADJUDICATED:
            if payload["actor_id"] != prior_payload["actor_id"]:
                raise ValueError("adjudicated actor_id does not match captured actor")
        elif event.to_state is TrialState.BATCH_ADJUDICATED:
            if payload["actor_ids"] != prior_payload["actor_ids"]:
                raise ValueError("adjudicated batch actors do not match captured actors")
            if payload["round_index"] != prior_payload["round_index"]:
                raise ValueError("adjudicated batch round does not match capture")
        elif event.to_state is TrialState.OBSERVED:
            if lineage_previous.to_state is TrialState.BATCH_ADJUDICATED:
                if payload["actor_ids"] != prior_payload["actor_ids"]:
                    raise ValueError(
                        "observed batch actors do not match adjudicated actors"
                    )
                if (
                    payload["interaction_event_ids"]
                    != prior_payload["interaction_event_ids"]
                ):
                    raise ValueError(
                        "batch observation does not match adjudicated interactions"
                    )
            else:
                if payload["actor_id"] != prior_payload["actor_id"]:
                    raise ValueError(
                        "observed actor_id does not match adjudicated actor"
                    )
                if payload["interaction_event_id"] != prior_payload[
                    "interaction_event_id"
                ]:
                    raise ValueError(
                        "observation does not match adjudicated interaction"
                    )
        elif (
            event.to_state is TrialState.COMPLETED
            and lineage_previous.to_state is TrialState.ADJUDICATED
            and payload.get("interaction_event_id") is not None
            and payload["interaction_event_id"]
            != prior_payload["interaction_event_id"]
        ):
            raise ValueError("completion does not match adjudicated interaction")
        elif (
            event.to_state is TrialState.COMPLETED
            and lineage_previous.to_state is TrialState.BATCH_ADJUDICATED
            and payload.get("interaction_event_ids") is not None
            and payload["interaction_event_ids"]
            != prior_payload["interaction_event_ids"]
        ):
            raise ValueError(
                "completion does not match adjudicated batch interactions"
            )

    def project_transcript(self) -> tuple[dict[str, Any], ...]:
        """Project captured actions/observations without making model calls."""
        rows: list[dict[str, Any]] = []
        for event in self._events:
            if event.to_state not in {
                TrialState.ACTION_CAPTURED,
                TrialState.BATCH_CAPTURED,
                TrialState.OBSERVED,
            }:
                continue
            payload = _thaw(event.payload)
            if event.to_state is TrialState.BATCH_CAPTURED:
                rows.extend({
                    "trial_runtime_event_id": event.event_id,
                    "state": event.to_state.value,
                    "actor_id": actor_id,
                    "text": output_text,
                    "generation_record_id": generation_record_id,
                    "interaction_event_id": None,
                    "label_record_ids": [],
                } for actor_id, output_text, generation_record_id in zip(
                    payload["actor_ids"],
                    payload["output_texts"],
                    payload["generation_record_ids"],
                ))
                continue
            rows.append({
                "trial_runtime_event_id": event.event_id,
                "state": event.to_state.value,
                "actor_id": payload.get("actor_id"),
                "text": payload.get("output_text", payload.get("observation")),
                "generation_record_id": payload.get("generation_record_id"),
                "interaction_event_id": payload.get(
                    "interaction_event_id",
                    payload.get("interaction_event_ids"),
                ),
                "label_record_ids": payload.get("label_record_ids", []),
            })
        return tuple(rows)
