"""Pure deterministic projections over validated experiment event streams.

Projectors in this module rebuild read models only.  They do not call a model,
read artifacts, consult process configuration, or use wall-clock time.  Event
``recorded_at`` values are intentionally absent from every projection's
semantic identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel

from interpretability.events.payloads import (
    ActionCommittedPayload,
    ActionProposedPayload,
    ActivationIntervenedPayload,
    AgentBuiltPayload,
    BehaviorLabeledPayload,
    LabelProvenance,
    LabelValue,
    ModelCallCompletedPayload,
    ModelCallFailedPayload,
    ModelCallStartedPayload,
    MonitorScoredPayload,
    ObservationDeliveredPayload,
    OutcomeResolvedPayload,
    PrivateViewAssignedPayload,
    ProtocolDecisionAppliedPayload,
    QualityControlAppliedPayload,
    TrialCompletedPayload,
    TrialFailedPayload,
    TrialStartedPayload,
    TurnAdvancedPayload,
)
from interpretability.events.schema import (
    ActivationCapturedPayload,
    ArtifactReference,
    EventEnvelope,
    OpaqueEventPayload,
)


class ProjectionError(ValueError):
    """Base class for invalid or unprojectable event streams."""


class ProjectionIdentityError(ProjectionError):
    """Raised when a stream mixes run, pod, trial, dyad, or actor identity."""


class ProjectionSequenceError(ProjectionError):
    """Raised for duplicate IDs, sequence gaps, or broken hash links."""


class ProjectionLinkError(ProjectionError):
    """Raised when event lineage is absent, out of order, or inconsistent."""


class ProjectionLifecycleError(ProjectionError):
    """Raised when trial events violate open/sealed lifecycle rules."""


class UnprojectableEventError(ProjectionError):
    """Raised when a projection needs an opaque payload schema."""


def _semantic_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _semantic_value(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, BaseModel):
        return _semantic_value(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return _semantic_value(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_semantic_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        f"projection contains unsupported semantic value {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _semantic_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class SemanticProjection:
    """Canonical serialization shared by immutable projection values."""

    def semantic_dict(self) -> dict[str, Any]:
        value = _semantic_value(self)
        if not isinstance(value, dict):  # pragma: no cover - dataclass contract
            raise TypeError("semantic projection must serialize as an object")
        return value

    def canonical_semantic_json(self) -> str:
        return _canonical_json(self)

    @property
    def semantic_hash(self) -> str:
        return hashlib.sha256(
            self.canonical_semantic_json().encode("utf-8")
        ).hexdigest()

    @property
    def semantic_sha256(self) -> str:
        """Alias spelling useful in manifests."""

        return self.semantic_hash


@dataclass(frozen=True, slots=True)
class ProjectionWarning(SemanticProjection):
    field_name: str
    code: str
    detail: str
    source_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class BehaviorLabelProjection(SemanticProjection):
    event_id: str
    label_id: str
    target_event_id: str
    target_actor_id: str
    label_name: str
    value: LabelValue
    provenance: LabelProvenance
    parent_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MonitorScoreProjection(SemanticProjection):
    event_id: str
    monitor_id: str
    monitor_version: str
    target_event_id: str
    target_actor_id: str
    score: float
    threshold: float | None
    flagged: bool
    evidence_event_ids: tuple[str, ...]
    parent_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QualityControlProjection(SemanticProjection):
    event_id: str
    qc_id: str
    qc_version: str
    target_event_id: str
    passed: bool
    flags: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    parent_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutcomeProjection(SemanticProjection):
    event_id: str
    outcome_id: str
    trial_id: str
    resolver_id: str
    resolver_version: str
    outcome_schema_version: str
    outcome_json: str
    outcome_hash: str
    source_event_ids: tuple[str, ...]
    success: bool | None
    score: float | None
    parent_event_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TranscriptAction(SemanticProjection):
    action_id: str
    action_hash: str
    actor_id: str
    actor_role: str
    model_call_id: str
    proposed_event_id: str
    committed_event_id: str
    turn_event_id: str
    turn_id: str
    next_actor_id: str
    next_actor_role: str
    next_sequence_num: int
    protocol_decision_event_id: str | None
    annotation_event_ids: tuple[str, ...]
    action_text: None = None
    warnings: tuple[ProjectionWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class TranscriptProjection(SemanticProjection):
    run_id: str | None
    pod_id: str | None
    trial_id: str | None
    dyad_id: str | None
    actions: tuple[TranscriptAction, ...]
    labels: tuple[BehaviorLabelProjection, ...]
    monitor_scores: tuple[MonitorScoreProjection, ...]
    quality_controls: tuple[QualityControlProjection, ...]


@dataclass(frozen=True, slots=True)
class PrivateViewProjection(SemanticProjection):
    event_id: str
    sequence_num: int
    scenario_instance_id: str
    view_id: str
    recipient_actor_id: str
    recipient_role: str
    view_schema_version: str
    view_hash: str
    source_scenario_event_id: str
    private_view_content: None = None
    warnings: tuple[ProjectionWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class ObservationProjection(SemanticProjection):
    event_id: str
    sequence_num: int
    observation_id: str
    recipient_actor_id: str
    recipient_role: str
    source_actor_id: str | None
    source_event_id: str
    content_hash: str
    visibility: str
    sequence_in_recipient_view: int
    observation_content: None = None
    warnings: tuple[ProjectionWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentViewProjection(SemanticProjection):
    run_id: str | None
    pod_id: str | None
    trial_id: str | None
    dyad_id: str | None
    actor_id: str
    through_sequence_num: int | None
    private_views: tuple[PrivateViewProjection, ...]
    observations: tuple[ObservationProjection, ...]


@dataclass(frozen=True, slots=True)
class ProjectedEventFact(SemanticProjection):
    event_id: str
    event_type: str
    sequence_num: int
    actor_id: str | None
    actor_role: str | None
    model_call_id: str | None
    parent_event_ids: tuple[str, ...]
    payload_schema_version: str
    payload_json: str
    payload_hash: str


@dataclass(frozen=True, slots=True)
class CurrentTurnProjection(SemanticProjection):
    turn_id: str
    turn_event_id: str
    actor_id: str
    actor_role: str
    prior_actor_id: str
    prior_actor_role: str
    committed_action_event_id: str
    next_sequence_num: int


@dataclass(frozen=True, slots=True)
class TrialStateProjection(SemanticProjection):
    run_id: str | None
    pod_id: str | None
    trial_id: str | None
    dyad_id: str | None
    lifecycle_status: str
    started_event_id: str | None
    terminal_event_id: str | None
    current_turn: CurrentTurnProjection | None
    commitments: tuple[ProjectedEventFact, ...]
    interventions: tuple[ProjectedEventFact, ...]
    labels: tuple[BehaviorLabelProjection, ...]
    monitor_scores: tuple[MonitorScoreProjection, ...]
    quality_controls: tuple[QualityControlProjection, ...]
    outcomes: tuple[OutcomeProjection, ...]


@dataclass(frozen=True, slots=True)
class ProjectedActivationSample(SemanticProjection):
    run_id: str
    pod_id: str
    trial_id: str
    dyad_id: str | None
    actor_id: str
    actor_role: str
    call_purpose: str
    model_call_id: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    generation_config_json: str
    generation_config_hash: str
    model_call_started_event_id: str
    model_call_completed_event_id: str
    activation_event_ids: tuple[str, ...]
    activation_artifacts: tuple[ArtifactReference, ...]
    action_id: str
    action_hash: str
    action_proposed_event_id: str
    action_committed_event_id: str
    turn_event_id: str
    turn_id: str
    annotation_event_ids: tuple[str, ...]
    sample_type: str = "negotiation"
    round_num: None = None
    prompt: None = None
    response: None = None
    activation_values: None = None
    warnings: tuple[ProjectionWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class ActivationSampleProjection(SemanticProjection):
    run_id: str | None
    pod_id: str | None
    trial_id: str | None
    dyad_id: str | None
    samples: tuple[ProjectedActivationSample, ...]
    labels: tuple[BehaviorLabelProjection, ...]
    monitor_scores: tuple[MonitorScoreProjection, ...]
    quality_controls: tuple[QualityControlProjection, ...]


@dataclass(frozen=True, slots=True)
class DyadTurnProjection(SemanticProjection):
    ordinal: int
    dyad_id: str
    trial_id: str
    turn_id: str
    turn_event_id: str
    actor_id: str
    actor_role: str
    counterpart_actor_id: str
    counterpart_role: str
    action_id: str
    action_hash: str
    action_event_id: str
    model_call_id: str
    reception_event_ids: tuple[str, ...]
    response_action_event_id: str | None
    response_model_call_id: str | None
    annotation_event_ids: tuple[str, ...]
    warnings: tuple[ProjectionWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class DyadProjection(SemanticProjection):
    run_id: str | None
    pod_id: str | None
    trial_id: str | None
    dyad_id: str | None
    actor_ids: tuple[str, ...]
    actor_roles: tuple[tuple[str, str], ...]
    turns: tuple[DyadTurnProjection, ...]


@dataclass(frozen=True, slots=True)
class MetricInputProjection(SemanticProjection):
    run_id: str | None
    pod_id: str | None
    trial_id: str | None
    dyad_id: str | None
    labels: tuple[BehaviorLabelProjection, ...]
    monitor_scores: tuple[MonitorScoreProjection, ...]
    outcomes: tuple[OutcomeProjection, ...]


@dataclass(frozen=True, slots=True)
class _ValidatedStream:
    events: tuple[EventEnvelope, ...]
    by_id: Mapping[str, EventEnvelope]
    run_id: str | None
    pod_id: str | None
    trial_id: str | None
    dyad_id: str | None
    actor_roles: Mapping[str, str]
    external_parent_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _InteractionIndex:
    starts_by_call: Mapping[str, EventEnvelope]
    completed_by_call: Mapping[str, EventEnvelope]
    activations_by_call: Mapping[str, tuple[EventEnvelope, ...]]
    proposals_by_event: Mapping[str, EventEnvelope]
    commitments_by_event: Mapping[str, EventEnvelope]
    commitments_by_proposal: Mapping[str, EventEnvelope]
    turns_by_commitment: Mapping[str, EventEnvelope]


_TRIAL_TERMINALS = frozenset({"TrialCompleted", "TrialFailed"})
_ANNOTATION_TYPES = frozenset(
    {"BehaviorLabeled", "MonitorScored", "QualityControlApplied"}
)
_INTERVENTION_TYPES = frozenset(
    {
        "InterventionScheduled",
        "BeliefIntervened",
        "ActivationIntervened",
        "ProtocolDecisionApplied",
    }
)
_CALL_ACTION_TYPES = frozenset(
    {
        "ModelCallStarted",
        "ModelCallCompleted",
        "ModelCallFailed",
        "ActivationCaptured",
        "ActionProposed",
        "ActionCommitted",
        "TurnAdvanced",
        "ProtocolDecisionApplied",
    }
)


def _expect_payload(
    event: EventEnvelope, payload_type: type[BaseModel]
) -> Any:
    if isinstance(event.payload, OpaqueEventPayload):
        raise UnprojectableEventError(
            f"event {event.event_id} ({event.event_type}) uses opaque payload "
            f"schema {event.payload_schema_version}; install an upgrader before "
            "projecting it"
        )
    if not isinstance(event.payload, payload_type):
        raise UnprojectableEventError(
            f"event {event.event_id} ({event.event_type}) has unexpected payload "
            f"type {type(event.payload).__name__}"
        )
    return event.payload


def _remember_role(
    roles: dict[str, str], actor_id: str, role: str, *, event_id: str
) -> None:
    existing = roles.get(actor_id)
    if existing is not None and existing != role:
        raise ProjectionIdentityError(
            f"actor {actor_id} changes role from {existing} to {role} at "
            f"event {event_id}"
        )
    roles[actor_id] = role


def _payload_json(event: EventEnvelope) -> str:
    return json.dumps(
        event.payload.to_payload_dict(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _event_fact(event: EventEnvelope) -> ProjectedEventFact:
    payload_json = _payload_json(event)
    return ProjectedEventFact(
        event_id=event.event_id,
        event_type=event.event_type,
        sequence_num=event.sequence_num,
        actor_id=event.actor_id,
        actor_role=event.actor_role,
        model_call_id=event.model_call_id,
        parent_event_ids=event.parent_event_ids,
        payload_schema_version=event.payload_schema_version,
        payload_json=payload_json,
        payload_hash=hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
    )


def _validate_payload_identity(
    event: EventEnvelope,
    *,
    actor_roles: dict[str, str],
    declared_actors: set[str] | None,
) -> set[str] | None:
    """Validate envelope/payload identities without interpreting event content."""

    payload = event.payload
    payload_run_id = getattr(payload, "run_id", event.run_id)
    if payload_run_id != event.run_id:
        raise ProjectionIdentityError(
            f"event {event.event_id} payload run_id {payload_run_id} does not "
            f"match envelope run_id {event.run_id}"
        )
    payload_trial_id = getattr(payload, "trial_id", event.trial_id)
    if payload_trial_id != event.trial_id:
        raise ProjectionIdentityError(
            f"event {event.event_id} payload trial_id {payload_trial_id} does "
            f"not match envelope trial_id {event.trial_id}"
        )

    if event.actor_id is not None and event.actor_role is not None:
        _remember_role(
            actor_roles,
            event.actor_id,
            event.actor_role,
            event_id=event.event_id,
        )

    if isinstance(payload, TrialStartedPayload):
        if payload.dyad_id != event.dyad_id:
            raise ProjectionIdentityError(
                f"TrialStarted {event.event_id} payload/envelope dyad mismatch"
            )
        declared_actors = set(payload.actor_ids)

    actor_id = getattr(payload, "actor_id", None)
    if actor_id is not None:
        if event.actor_id != actor_id:
            raise ProjectionIdentityError(
                f"event {event.event_id} payload actor {actor_id} does not match "
                f"envelope actor {event.actor_id}"
            )
        if declared_actors is not None and actor_id not in declared_actors:
            raise ProjectionIdentityError(
                f"event {event.event_id} actor {actor_id} was not declared by "
                "TrialStarted"
            )

    if isinstance(payload, AgentBuiltPayload):
        if event.actor_role != payload.role:
            raise ProjectionIdentityError(
                f"AgentBuilt {event.event_id} role {payload.role} does not match "
                f"envelope role {event.actor_role}"
            )
        _remember_role(actor_roles, payload.actor_id, payload.role, event_id=event.event_id)

    if isinstance(payload, PrivateViewAssignedPayload):
        if event.actor_id != payload.recipient_actor_id:
            raise ProjectionIdentityError(
                f"PrivateViewAssigned {event.event_id} recipient does not match "
                "envelope actor"
            )
        if event.actor_role != payload.recipient_role:
            raise ProjectionIdentityError(
                f"PrivateViewAssigned {event.event_id} recipient role does not "
                "match envelope role"
            )
        _remember_role(
            actor_roles,
            payload.recipient_actor_id,
            payload.recipient_role,
            event_id=event.event_id,
        )

    if isinstance(payload, ObservationDeliveredPayload):
        if event.actor_id != payload.recipient_actor_id:
            raise ProjectionIdentityError(
                f"ObservationDelivered {event.event_id} recipient does not match "
                "envelope actor"
            )

    if isinstance(payload, TurnAdvancedPayload):
        if event.actor_id != payload.from_actor_id:
            raise ProjectionIdentityError(
                f"TurnAdvanced {event.event_id} from_actor_id does not match "
                "envelope actor"
            )
        if declared_actors is not None and (
            payload.from_actor_id not in declared_actors
            or payload.to_actor_id not in declared_actors
        ):
            raise ProjectionIdentityError(
                f"TurnAdvanced {event.event_id} references an undeclared actor"
            )

    payload_model_call_id = None
    if isinstance(
        payload,
        (
            ModelCallStartedPayload,
            ModelCallCompletedPayload,
            ModelCallFailedPayload,
            ActionProposedPayload,
            ActionCommittedPayload,
            ActivationIntervenedPayload,
        ),
    ):
        payload_model_call_id = payload.model_call_id
    elif isinstance(payload, ActivationCapturedPayload):
        payload_model_call_id = payload.artifact.source_model_call_id
    if payload_model_call_id is not None and event.model_call_id != payload_model_call_id:
        raise ProjectionIdentityError(
            f"event {event.event_id} payload model call {payload_model_call_id} "
            f"does not match envelope model call {event.model_call_id}"
        )
    return declared_actors


def _validate_stream(
    events: Iterable[EventEnvelope],
    *,
    expected_run_id: str | None,
    expected_trial_id: str | None,
    external_parent_ids: frozenset[str],
    required_event_types: frozenset[str],
) -> _ValidatedStream:
    if isinstance(events, (str, bytes)):
        raise TypeError("events must be an iterable of EventEnvelope objects")

    collected: list[EventEnvelope] = []
    by_id: dict[str, EventEnvelope] = {}
    lanes: dict[str | None, tuple[int, str]] = {}
    actor_roles: dict[str, str] = {}
    run_id: str | None = None
    pod_id: str | None = None
    trial_id: str | None = None
    dyad_id: str | None = None
    trial_open = False
    trial_terminal: str | None = None
    declared_actors: set[str] | None = None

    for position, event in enumerate(events):
        if not isinstance(event, EventEnvelope):
            raise TypeError(
                f"events[{position}] must be EventEnvelope, got "
                f"{type(event).__name__}"
            )
        try:
            event.verify_content_hash()
        except Exception as exc:
            raise ProjectionSequenceError(
                f"event {event.event_id} failed content-hash verification"
            ) from exc

        if run_id is None:
            run_id = event.run_id
            pod_id = event.pod_id
        elif event.run_id != run_id or event.pod_id != pod_id:
            raise ProjectionIdentityError(
                f"event {event.event_id} changes stream identity from "
                f"{run_id}/{pod_id} to {event.run_id}/{event.pod_id}"
            )
        if expected_run_id is not None and event.run_id != expected_run_id:
            raise ProjectionIdentityError(
                f"event {event.event_id} belongs to run {event.run_id}, expected "
                f"{expected_run_id}"
            )

        if event.trial_id is not None:
            if trial_id is None:
                trial_id = event.trial_id
            elif event.trial_id != trial_id:
                raise ProjectionIdentityError(
                    f"projection mixes trials {trial_id} and {event.trial_id}"
                )
            if expected_trial_id is not None and event.trial_id != expected_trial_id:
                raise ProjectionIdentityError(
                    f"event {event.event_id} belongs to trial {event.trial_id}, "
                    f"expected {expected_trial_id}"
                )
            if event.dyad_id is not None:
                if dyad_id is None:
                    dyad_id = event.dyad_id
                elif event.dyad_id != dyad_id:
                    raise ProjectionIdentityError(
                        f"event {event.event_id} changes dyad from {dyad_id} to "
                        f"{event.dyad_id}"
                    )
        elif event.event_type == "TrialStarted" or event.event_type in _TRIAL_TERMINALS:
            raise ProjectionLifecycleError(
                f"{event.event_type} event {event.event_id} requires trial_id"
            )

        if event.event_id in by_id or event.event_id in external_parent_ids:
            raise ProjectionSequenceError(f"duplicate event_id {event.event_id}")
        for parent_id in event.parent_event_ids:
            if parent_id not in by_id and parent_id not in external_parent_ids:
                raise ProjectionLinkError(
                    f"event {event.event_id} references missing or later parent "
                    f"{parent_id}"
                )

        lane = lanes.get(event.trial_id)
        if lane is None:
            if event.sequence_num != 0 or event.previous_event_hash is not None:
                raise ProjectionSequenceError(
                    f"first event in lane {event.trial_id!r} must use sequence 0 "
                    "with no previous hash"
                )
        else:
            expected_sequence = lane[0] + 1
            if event.sequence_num != expected_sequence:
                raise ProjectionSequenceError(
                    f"lane {event.trial_id!r} expected sequence "
                    f"{expected_sequence}, got {event.sequence_num}"
                )
            if event.previous_event_hash != lane[1]:
                raise ProjectionSequenceError(
                    f"event {event.event_id} has an incorrect previous hash"
                )

        if isinstance(event.payload, OpaqueEventPayload):
            structural = {"TrialStarted", *_TRIAL_TERMINALS}
            if event.event_type in required_event_types or event.event_type in structural:
                raise UnprojectableEventError(
                    f"event {event.event_id} ({event.event_type}) is required by "
                    "this projection but its payload schema is opaque"
                )

        if event.trial_id is not None:
            if trial_terminal is not None:
                raise ProjectionLifecycleError(
                    f"event {event.event_id} mutates trial {event.trial_id} after "
                    f"terminal event {trial_terminal}"
                )
            if event.event_type == "TrialStarted":
                if trial_open:
                    raise ProjectionLifecycleError(
                        f"duplicate TrialStarted for trial {event.trial_id}"
                    )
                trial_open = True
            elif not trial_open:
                raise ProjectionLifecycleError(
                    f"event {event.event_id} occurs before TrialStarted for "
                    f"trial {event.trial_id}"
                )

        declared_actors = _validate_payload_identity(
            event,
            actor_roles=actor_roles,
            declared_actors=declared_actors,
        )

        collected.append(event)
        by_id[event.event_id] = event
        lanes[event.trial_id] = (event.sequence_num, event.content_hash or "")
        if event.event_type in _TRIAL_TERMINALS:
            trial_terminal = event.event_id
            trial_open = False

    if expected_run_id is not None and run_id is not None and run_id != expected_run_id:
        raise ProjectionIdentityError(
            f"stream run {run_id} does not match expected {expected_run_id}"
        )
    if (
        expected_trial_id is not None
        and trial_id is not None
        and trial_id != expected_trial_id
    ):
        raise ProjectionIdentityError(
            f"stream trial {trial_id} does not match expected {expected_trial_id}"
        )
    validated = _ValidatedStream(
        events=tuple(collected),
        by_id=by_id,
        run_id=run_id,
        pod_id=pod_id,
        trial_id=trial_id,
        dyad_id=dyad_id,
        actor_roles=actor_roles,
        external_parent_ids=external_parent_ids,
    )
    _validate_interaction_links(validated, required_event_types=required_event_types)
    _validate_annotation_links(validated, required_event_types=required_event_types)
    return validated


def _require_prior_event(
    stream: _ValidatedStream,
    event_id: str,
    *,
    before: EventEnvelope,
    relation: str,
    expected_types: tuple[str, ...] = (),
) -> EventEnvelope:
    linked = stream.by_id.get(event_id)
    if linked is None:
        raise ProjectionLinkError(
            f"event {before.event_id} {relation} references missing event {event_id}"
        )
    try:
        linked_position = stream.events.index(linked)
        before_position = stream.events.index(before)
    except ValueError as exc:  # pragma: no cover - internal index contract
        raise ProjectionLinkError("event index is internally inconsistent") from exc
    if linked_position >= before_position:
        raise ProjectionLinkError(
            f"event {before.event_id} {relation} must reference an earlier event"
        )
    if expected_types and linked.event_type not in expected_types:
        choices = ", ".join(expected_types)
        raise ProjectionLinkError(
            f"event {before.event_id} {relation} references {linked.event_type}; "
            f"expected {choices}"
        )
    return linked


def _validate_interaction_links(
    stream: _ValidatedStream, *, required_event_types: frozenset[str]
) -> None:
    """Fail closed on call/action lineage whenever a projector consumes it."""

    opaque_call_action = any(
        isinstance(event.payload, OpaqueEventPayload)
        and event.event_type in _CALL_ACTION_TYPES
        for event in stream.events
    )
    if (
        opaque_call_action
        and not required_event_types.intersection(_CALL_ACTION_TYPES)
    ):
        # The caller does not consume this future call/action schema.  Treat it
        # like any other unrelated opaque event instead of guessing its links.
        return
    starts: dict[str, EventEnvelope] = {}
    terminal_calls: dict[str, EventEnvelope] = {}
    activations: dict[str, list[EventEnvelope]] = {}
    proposals: dict[str, EventEnvelope] = {}
    proposal_calls: dict[str, EventEnvelope] = {}
    commitments: dict[str, EventEnvelope] = {}
    turns: dict[str, EventEnvelope] = {}

    for event in stream.events:
        payload = event.payload
        if isinstance(payload, ModelCallStartedPayload):
            if payload.model_call_id in starts:
                raise ProjectionLinkError(
                    f"duplicate ModelCallStarted for call {payload.model_call_id}"
                )
            starts[payload.model_call_id] = event
        elif isinstance(payload, (ModelCallCompletedPayload, ModelCallFailedPayload)):
            started = starts.get(payload.model_call_id)
            if started is None:
                raise ProjectionLinkError(
                    f"{event.event_type} {event.event_id} has no earlier "
                    "ModelCallStarted"
                )
            started_payload = _expect_payload(started, ModelCallStartedPayload)
            if payload.started_event_id != started.event_id:
                raise ProjectionLinkError(
                    f"{event.event_type} {event.event_id} points to the wrong "
                    "ModelCallStarted event"
                )
            if (
                payload.actor_id != started_payload.actor_id
                or payload.purpose != started_payload.purpose
            ):
                raise ProjectionLinkError(
                    f"{event.event_type} {event.event_id} changes call actor/purpose"
                )
            if payload.generation_config_hash != started_payload.generation_config.sha256:
                raise ProjectionLinkError(
                    f"{event.event_type} {event.event_id} generation config hash "
                    "does not match its start"
                )
            if payload.model_call_id in terminal_calls:
                raise ProjectionLinkError(
                    f"model call {payload.model_call_id} has multiple terminal events"
                )
            terminal_calls[payload.model_call_id] = event
        elif isinstance(payload, ActivationCapturedPayload):
            call_id = payload.artifact.source_model_call_id
            terminal = terminal_calls.get(call_id)
            if terminal is None or terminal.event_type != "ModelCallCompleted":
                raise ProjectionLinkError(
                    f"ActivationCaptured {event.event_id} has no earlier completed "
                    f"model call {call_id}"
                )
            start_payload = _expect_payload(starts[call_id], ModelCallStartedPayload)
            if payload.artifact.tokenizer_id != start_payload.tokenizer_id:
                raise ProjectionLinkError(
                    f"ActivationCaptured {event.event_id} tokenizer does not match "
                    "its model call"
                )
            if payload.artifact.model_revision != start_payload.model_revision:
                raise ProjectionLinkError(
                    f"ActivationCaptured {event.event_id} model revision does not "
                    "match its model call"
                )
            if event.actor_id != start_payload.actor_id:
                raise ProjectionLinkError(
                    f"ActivationCaptured {event.event_id} actor does not match "
                    "its model call"
                )
            activations.setdefault(call_id, []).append(event)
        elif isinstance(payload, ActionProposedPayload):
            if payload.action_id in proposals:
                raise ProjectionLinkError(
                    f"duplicate proposed action ID {payload.action_id}"
                )
            if payload.model_call_id in proposal_calls:
                raise ProjectionLinkError(
                    f"model call {payload.model_call_id} proposes multiple actions"
                )
            completed = terminal_calls.get(payload.model_call_id)
            if completed is None or completed.event_type != "ModelCallCompleted":
                raise ProjectionLinkError(
                    f"ActionProposed {event.event_id} has no completed model call "
                    f"{payload.model_call_id}"
                )
            if payload.model_call_event_id != completed.event_id:
                raise ProjectionLinkError(
                    f"ActionProposed {event.event_id} points to the wrong model "
                    "completion event"
                )
            completed_payload = _expect_payload(completed, ModelCallCompletedPayload)
            if payload.actor_id != completed_payload.actor_id:
                raise ProjectionLinkError(
                    f"ActionProposed {event.event_id} actor does not match its call"
                )
            started_payload = _expect_payload(
                starts[payload.model_call_id], ModelCallStartedPayload
            )
            for observation_id in payload.source_observation_event_ids:
                _require_prior_event(
                    stream,
                    observation_id,
                    before=event,
                    relation="source_observation_event_ids",
                    expected_types=("ObservationDelivered",),
                )
                if observation_id not in started_payload.input_event_ids:
                    raise ProjectionLinkError(
                        f"ActionProposed {event.event_id} source observation "
                        f"{observation_id} was not an input to its model call"
                    )
            proposals[payload.action_id] = event
            proposal_calls[payload.model_call_id] = event
        elif isinstance(payload, ActionCommittedPayload):
            if payload.action_id in commitments:
                raise ProjectionLinkError(
                    f"duplicate committed action ID {payload.action_id}"
                )
            proposal = _require_prior_event(
                stream,
                payload.proposed_event_id,
                before=event,
                relation="proposed_event_id",
                expected_types=("ActionProposed",),
            )
            proposal_payload = _expect_payload(proposal, ActionProposedPayload)
            if (
                payload.action_id != proposal_payload.action_id
                or payload.actor_id != proposal_payload.actor_id
                or payload.model_call_id != proposal_payload.model_call_id
            ):
                raise ProjectionLinkError(
                    f"ActionCommitted {event.event_id} does not match its proposal"
                )
            if payload.protocol_decision_event_id is not None:
                decision_event = _require_prior_event(
                    stream,
                    payload.protocol_decision_event_id,
                    before=event,
                    relation="protocol_decision_event_id",
                    expected_types=("ProtocolDecisionApplied",),
                )
                decision = _expect_payload(
                    decision_event, ProtocolDecisionAppliedPayload
                )
                if decision.target_action_event_id != proposal.event_id:
                    raise ProjectionLinkError(
                        f"protocol decision {decision_event.event_id} targets a "
                        "different action proposal"
                    )
                expected_hash = decision.modified_action_hash or proposal_payload.action_hash
                if payload.action_hash != expected_hash:
                    raise ProjectionLinkError(
                        f"ActionCommitted {event.event_id} does not apply its "
                        "protocol decision hash"
                    )
            elif payload.action_hash != proposal_payload.action_hash:
                raise ProjectionLinkError(
                    f"ActionCommitted {event.event_id} action hash does not match "
                    "its proposal"
                )
            commitments[payload.action_id] = event
        elif isinstance(payload, TurnAdvancedPayload):
            if payload.committed_action_event_id in turns:
                raise ProjectionLinkError(
                    f"committed action {payload.committed_action_event_id} has "
                    "multiple TurnAdvanced events"
                )
            commitment = _require_prior_event(
                stream,
                payload.committed_action_event_id,
                before=event,
                relation="committed_action_event_id",
                expected_types=("ActionCommitted",),
            )
            committed = _expect_payload(commitment, ActionCommittedPayload)
            if payload.from_actor_id != committed.actor_id:
                raise ProjectionLinkError(
                    f"TurnAdvanced {event.event_id} actor does not match committed "
                    "action"
                )
            turns[payload.committed_action_event_id] = event
        elif isinstance(payload, ProtocolDecisionAppliedPayload):
            target = _require_prior_event(
                stream,
                payload.target_action_event_id,
                before=event,
                relation="target_action_event_id",
                expected_types=("ActionProposed",),
            )
            _expect_payload(target, ActionProposedPayload)
            if payload.trial_id != target.trial_id:
                raise ProjectionLinkError(
                    f"ProtocolDecisionApplied {event.event_id} crosses trials"
                )
        elif isinstance(payload, ActivationIntervenedPayload):
            source = _require_prior_event(
                stream,
                payload.source_activation_event_id,
                before=event,
                relation="source_activation_event_id",
                expected_types=("ActivationCaptured",),
            )
            _expect_payload(source, ActivationCapturedPayload)

    for call_id, terminal_event in terminal_calls.items():
        if not isinstance(terminal_event.payload, ModelCallCompletedPayload):
            continue
        completed_payload = terminal_event.payload
        activation_events = activations.get(call_id, [])
        captured_hashes = tuple(
            _expect_payload(item, ActivationCapturedPayload).artifact.artifact_hash
            for item in activation_events
        )
        if len(captured_hashes) != len(set(captured_hashes)):
            raise ProjectionLinkError(
                f"model call {call_id} captures a duplicate activation artifact"
            )
        if captured_hashes != completed_payload.activation_artifact_hashes:
            raise ProjectionLinkError(
                f"model call {call_id} activation events do not match the hashes "
                "declared by ModelCallCompleted"
            )


def _validate_annotation_links(
    stream: _ValidatedStream, *, required_event_types: frozenset[str]
) -> None:
    if not required_event_types.intersection(
        {*_ANNOTATION_TYPES, "OutcomeResolved"}
    ):
        return
    positions = {event.event_id: index for index, event in enumerate(stream.events)}

    def source(event: EventEnvelope, event_id: str, relation: str) -> EventEnvelope | None:
        linked = stream.by_id.get(event_id)
        if linked is None:
            if event_id in stream.external_parent_ids:
                return None
            raise ProjectionLinkError(
                f"event {event.event_id} {relation} references missing event "
                f"{event_id}"
            )
        if positions[linked.event_id] >= positions[event.event_id]:
            raise ProjectionLinkError(
                f"event {event.event_id} {relation} references a later event"
            )
        return linked

    for event in stream.events:
        payload = event.payload
        if isinstance(payload, BehaviorLabeledPayload):
            target = source(event, payload.target_event_id, "target_event_id")
            for source_id in payload.provenance.source_event_ids:
                source(event, source_id, "provenance.source_event_ids")
            if (
                target is not None
                and target.actor_id is not None
                and target.actor_id != payload.target_actor_id
            ):
                raise ProjectionLinkError(
                    f"BehaviorLabeled {event.event_id} target actor does not match "
                    "the target event"
                )
        elif isinstance(payload, MonitorScoredPayload):
            target = source(event, payload.target_event_id, "target_event_id")
            for source_id in payload.evidence_event_ids:
                source(event, source_id, "evidence_event_ids")
            if (
                target is not None
                and target.actor_id is not None
                and target.actor_id != payload.target_actor_id
            ):
                raise ProjectionLinkError(
                    f"MonitorScored {event.event_id} target actor does not match "
                    "the target event"
                )
        elif isinstance(payload, QualityControlAppliedPayload):
            source(event, payload.target_event_id, "target_event_id")
            for source_id in payload.source_event_ids:
                source(event, source_id, "source_event_ids")
        elif isinstance(payload, OutcomeResolvedPayload):
            for source_id in payload.source_event_ids:
                source(event, source_id, "source_event_ids")


def _interaction_index(stream: _ValidatedStream) -> _InteractionIndex:
    starts: dict[str, EventEnvelope] = {}
    completed: dict[str, EventEnvelope] = {}
    activations: dict[str, list[EventEnvelope]] = {}
    proposals: dict[str, EventEnvelope] = {}
    commitments: dict[str, EventEnvelope] = {}
    commitments_by_proposal: dict[str, EventEnvelope] = {}
    turns: dict[str, EventEnvelope] = {}
    for event in stream.events:
        payload = event.payload
        if isinstance(payload, ModelCallStartedPayload):
            starts[payload.model_call_id] = event
        elif isinstance(payload, ModelCallCompletedPayload):
            completed[payload.model_call_id] = event
        elif isinstance(payload, ActivationCapturedPayload):
            activations.setdefault(
                payload.artifact.source_model_call_id, []
            ).append(event)
        elif isinstance(payload, ActionProposedPayload):
            proposals[event.event_id] = event
        elif isinstance(payload, ActionCommittedPayload):
            commitments[event.event_id] = event
            commitments_by_proposal[payload.proposed_event_id] = event
        elif isinstance(payload, TurnAdvancedPayload):
            turns[payload.committed_action_event_id] = event
    return _InteractionIndex(
        starts_by_call=starts,
        completed_by_call=completed,
        activations_by_call={key: tuple(value) for key, value in activations.items()},
        proposals_by_event=proposals,
        commitments_by_event=commitments,
        commitments_by_proposal=commitments_by_proposal,
        turns_by_commitment=turns,
    )


def _label_projection(event: EventEnvelope) -> BehaviorLabelProjection:
    payload = _expect_payload(event, BehaviorLabeledPayload)
    return BehaviorLabelProjection(
        event_id=event.event_id,
        label_id=payload.label_id,
        target_event_id=payload.target_event_id,
        target_actor_id=payload.target_actor_id,
        label_name=payload.label_name,
        value=payload.value,
        provenance=payload.provenance,
        parent_event_ids=event.parent_event_ids,
    )


def _monitor_projection(event: EventEnvelope) -> MonitorScoreProjection:
    payload = _expect_payload(event, MonitorScoredPayload)
    return MonitorScoreProjection(
        event_id=event.event_id,
        monitor_id=payload.monitor_id,
        monitor_version=payload.monitor_version,
        target_event_id=payload.target_event_id,
        target_actor_id=payload.target_actor_id,
        score=payload.score,
        threshold=payload.threshold,
        flagged=payload.flagged,
        evidence_event_ids=payload.evidence_event_ids,
        parent_event_ids=event.parent_event_ids,
    )


def _qc_projection(event: EventEnvelope) -> QualityControlProjection:
    payload = _expect_payload(event, QualityControlAppliedPayload)
    return QualityControlProjection(
        event_id=event.event_id,
        qc_id=payload.qc_id,
        qc_version=payload.qc_version,
        target_event_id=payload.target_event_id,
        passed=payload.passed,
        flags=payload.flags,
        source_event_ids=payload.source_event_ids,
        parent_event_ids=event.parent_event_ids,
    )


def _outcome_projection(event: EventEnvelope) -> OutcomeProjection:
    payload = _expect_payload(event, OutcomeResolvedPayload)
    return OutcomeProjection(
        event_id=event.event_id,
        outcome_id=payload.outcome_id,
        trial_id=payload.trial_id,
        resolver_id=payload.resolver_id,
        resolver_version=payload.resolver_version,
        outcome_schema_version=payload.outcome.schema_version,
        outcome_json=payload.outcome.canonical_json,
        outcome_hash=payload.outcome.sha256,
        source_event_ids=payload.source_event_ids,
        success=payload.success,
        score=payload.score,
        parent_event_ids=event.parent_event_ids,
    )


def _annotations(
    stream: _ValidatedStream,
) -> tuple[
    tuple[BehaviorLabelProjection, ...],
    tuple[MonitorScoreProjection, ...],
    tuple[QualityControlProjection, ...],
]:
    labels: list[BehaviorLabelProjection] = []
    monitors: list[MonitorScoreProjection] = []
    quality_controls: list[QualityControlProjection] = []
    for event in stream.events:
        if isinstance(event.payload, BehaviorLabeledPayload):
            labels.append(_label_projection(event))
        elif isinstance(event.payload, MonitorScoredPayload):
            monitors.append(_monitor_projection(event))
        elif isinstance(event.payload, QualityControlAppliedPayload):
            quality_controls.append(_qc_projection(event))
    return tuple(labels), tuple(monitors), tuple(quality_controls)


def _annotation_ids_for(
    target_ids: set[str],
    labels: tuple[BehaviorLabelProjection, ...],
    monitors: tuple[MonitorScoreProjection, ...],
    quality_controls: tuple[QualityControlProjection, ...],
) -> tuple[str, ...]:
    return tuple(
        annotation.event_id
        for annotations in (labels, monitors, quality_controls)
        for annotation in annotations
        if annotation.target_event_id in target_ids
    )


class _ProjectorBase:
    _required_event_types: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        *,
        expected_run_id: str | None = None,
        expected_trial_id: str | None = None,
        external_parent_ids: Iterable[str] = (),
    ) -> None:
        if expected_run_id is not None and not expected_run_id:
            raise ValueError("expected_run_id must be a non-empty string")
        if expected_trial_id is not None and not expected_trial_id:
            raise ValueError("expected_trial_id must be a non-empty string")
        external = frozenset(external_parent_ids)
        if any(not isinstance(item, str) or not item for item in external):
            raise ValueError("external_parent_ids must contain non-empty strings")
        self._expected_run_id = expected_run_id
        self._expected_trial_id = expected_trial_id
        self._external_parent_ids = external
        self._last_result: SemanticProjection | None = None

    @property
    def last_result(self) -> SemanticProjection | None:
        return self._last_result

    def reset(self) -> None:
        self._last_result = None

    def _validate(self, events: Iterable[EventEnvelope]) -> _ValidatedStream:
        self.reset()
        return _validate_stream(
            events,
            expected_run_id=self._expected_run_id,
            expected_trial_id=self._expected_trial_id,
            external_parent_ids=self._external_parent_ids,
            required_event_types=self._required_event_types,
        )

    def _finish(self, result: SemanticProjection) -> Any:
        self._last_result = result
        return result


class TranscriptProjector(_ProjectorBase):
    """Project committed public actions in authoritative turn order."""

    _required_event_types = _CALL_ACTION_TYPES | _ANNOTATION_TYPES

    def project(self, events: Iterable[EventEnvelope]) -> TranscriptProjection:
        stream = self._validate(events)
        index = _interaction_index(stream)
        labels, monitors, quality_controls = _annotations(stream)
        actions: list[TranscriptAction] = []
        for event in stream.events:
            if not isinstance(event.payload, ActionCommittedPayload):
                continue
            payload = event.payload
            turn_event = index.turns_by_commitment.get(event.event_id)
            if turn_event is None:
                raise ProjectionLinkError(
                    f"committed action {event.event_id} has no TurnAdvanced event"
                )
            turn = _expect_payload(turn_event, TurnAdvancedPayload)
            actor_role = stream.actor_roles.get(payload.actor_id)
            next_role = stream.actor_roles.get(turn.to_actor_id)
            if actor_role is None or next_role is None:
                raise ProjectionIdentityError(
                    f"turn {turn_event.event_id} lacks stable roles for both actors"
                )
            target_ids = {
                payload.proposed_event_id,
                event.event_id,
                turn_event.event_id,
            }
            actions.append(
                TranscriptAction(
                    action_id=payload.action_id,
                    action_hash=payload.action_hash,
                    actor_id=payload.actor_id,
                    actor_role=actor_role,
                    model_call_id=payload.model_call_id,
                    proposed_event_id=payload.proposed_event_id,
                    committed_event_id=event.event_id,
                    turn_event_id=turn_event.event_id,
                    turn_id=turn.turn_id,
                    next_actor_id=turn.to_actor_id,
                    next_actor_role=next_role,
                    next_sequence_num=turn.next_sequence_num,
                    protocol_decision_event_id=payload.protocol_decision_event_id,
                    annotation_event_ids=_annotation_ids_for(
                        target_ids, labels, monitors, quality_controls
                    ),
                    warnings=(
                        ProjectionWarning(
                            field_name="action_text",
                            code="content_unavailable",
                            detail=(
                                "ActionCommitted stores a content hash, not raw "
                                "public text"
                            ),
                            source_event_id=event.event_id,
                        ),
                    ),
                )
            )
        return self._finish(
            TranscriptProjection(
                run_id=stream.run_id,
                pod_id=stream.pod_id,
                trial_id=stream.trial_id,
                dyad_id=stream.dyad_id,
                actions=tuple(actions),
                labels=labels,
                monitor_scores=monitors,
                quality_controls=quality_controls,
            )
        )


class AgentViewProjector(_ProjectorBase):
    """Project exactly the private-view assignments and observations for an actor."""

    _required_event_types = frozenset(
        {"PrivateViewAssigned", "ObservationDelivered"}
    )

    def __init__(
        self,
        actor_id: str,
        *,
        through_sequence_num: int | None = None,
        expected_run_id: str | None = None,
        expected_trial_id: str | None = None,
        external_parent_ids: Iterable[str] = (),
    ) -> None:
        if not isinstance(actor_id, str) or not actor_id:
            raise ValueError("actor_id must be a non-empty string")
        if through_sequence_num is not None and (
            isinstance(through_sequence_num, bool)
            or not isinstance(through_sequence_num, int)
            or through_sequence_num < 0
        ):
            raise ValueError("through_sequence_num must be a non-negative integer")
        super().__init__(
            expected_run_id=expected_run_id,
            expected_trial_id=expected_trial_id,
            external_parent_ids=external_parent_ids,
        )
        self._actor_id = actor_id
        self._through_sequence_num = through_sequence_num

    def project(
        self,
        events: Iterable[EventEnvelope],
        *,
        through_sequence_num: int | None = None,
    ) -> AgentViewProjection:
        if through_sequence_num is not None and (
            isinstance(through_sequence_num, bool)
            or not isinstance(through_sequence_num, int)
            or through_sequence_num < 0
        ):
            raise ValueError("through_sequence_num must be a non-negative integer")
        cutoff = (
            self._through_sequence_num
            if through_sequence_num is None
            else through_sequence_num
        )
        stream = self._validate(events)
        private_views: list[PrivateViewProjection] = []
        observations: list[ObservationProjection] = []
        observation_positions: set[int] = set()
        for event in stream.events:
            if cutoff is not None and event.sequence_num > cutoff:
                continue
            if isinstance(event.payload, PrivateViewAssignedPayload):
                payload = event.payload
                if payload.recipient_actor_id != self._actor_id:
                    continue
                private_views.append(
                    PrivateViewProjection(
                        event_id=event.event_id,
                        sequence_num=event.sequence_num,
                        scenario_instance_id=payload.scenario_instance_id,
                        view_id=payload.view_id,
                        recipient_actor_id=payload.recipient_actor_id,
                        recipient_role=payload.recipient_role,
                        view_schema_version=payload.view_schema_version,
                        view_hash=payload.view_hash,
                        source_scenario_event_id=payload.source_scenario_event_id,
                        warnings=(
                            ProjectionWarning(
                                field_name="private_view_content",
                                code="content_unavailable",
                                detail=(
                                    "PrivateViewAssigned stores a content hash, "
                                    "not private-view content"
                                ),
                                source_event_id=event.event_id,
                            ),
                        ),
                    )
                )
            elif isinstance(event.payload, ObservationDeliveredPayload):
                payload = event.payload
                if payload.recipient_actor_id != self._actor_id:
                    continue
                if payload.sequence_in_recipient_view in observation_positions:
                    raise ProjectionSequenceError(
                        f"actor {self._actor_id} has duplicate observation-view "
                        f"sequence {payload.sequence_in_recipient_view}"
                    )
                observation_positions.add(payload.sequence_in_recipient_view)
                role = stream.actor_roles.get(self._actor_id)
                if role is None:
                    raise ProjectionIdentityError(
                        f"actor {self._actor_id} has no stable role"
                    )
                observations.append(
                    ObservationProjection(
                        event_id=event.event_id,
                        sequence_num=event.sequence_num,
                        observation_id=payload.observation_id,
                        recipient_actor_id=payload.recipient_actor_id,
                        recipient_role=role,
                        source_actor_id=payload.source_actor_id,
                        source_event_id=payload.source_event_id,
                        content_hash=payload.content_hash,
                        visibility=payload.visibility,
                        sequence_in_recipient_view=payload.sequence_in_recipient_view,
                        warnings=(
                            ProjectionWarning(
                                field_name="observation_content",
                                code="content_unavailable",
                                detail=(
                                    "ObservationDelivered stores a content hash, "
                                    "not observation content"
                                ),
                                source_event_id=event.event_id,
                            ),
                        ),
                    )
                )
        expected_positions = set(range(len(observation_positions)))
        if observation_positions != expected_positions:
            raise ProjectionSequenceError(
                f"actor {self._actor_id} observation-view sequence has gaps: "
                f"observed {sorted(observation_positions)}"
            )
        return self._finish(
            AgentViewProjection(
                run_id=stream.run_id,
                pod_id=stream.pod_id,
                trial_id=stream.trial_id,
                dyad_id=stream.dyad_id,
                actor_id=self._actor_id,
                through_sequence_num=cutoff,
                private_views=tuple(private_views),
                observations=tuple(observations),
            )
        )


class TrialStateProjector(_ProjectorBase):
    """Rebuild lifecycle, turn, commitments, interventions, and annotations."""

    _required_event_types = (
        _CALL_ACTION_TYPES
        | _ANNOTATION_TYPES
        | _INTERVENTION_TYPES
        | frozenset({"TrialStarted", "TrialCompleted", "TrialFailed", "OutcomeResolved"})
    )

    def project(self, events: Iterable[EventEnvelope]) -> TrialStateProjection:
        stream = self._validate(events)
        labels, monitors, quality_controls = _annotations(stream)
        commitments: list[ProjectedEventFact] = []
        interventions: list[ProjectedEventFact] = []
        outcomes: list[OutcomeProjection] = []
        started_event_id: str | None = None
        terminal_event_id: str | None = None
        lifecycle_status = "empty"
        current_turn: CurrentTurnProjection | None = None

        for event in stream.events:
            payload = event.payload
            if isinstance(payload, TrialStartedPayload):
                started_event_id = event.event_id
                lifecycle_status = "open"
            elif isinstance(payload, TrialCompletedPayload):
                outcome_event = stream.by_id.get(payload.outcome_event_id)
                if outcome_event is None or not isinstance(
                    outcome_event.payload, OutcomeResolvedPayload
                ):
                    raise ProjectionLinkError(
                        f"TrialCompleted {event.event_id} references a missing or "
                        "non-outcome event"
                    )
                if payload.outcome_event_id not in {
                    outcome.event_id for outcome in outcomes
                }:
                    raise ProjectionLinkError(
                        f"TrialCompleted {event.event_id} references an outcome "
                        "that did not precede it"
                    )
                for action_event_id in payload.terminal_action_event_ids:
                    action_event = stream.by_id.get(action_event_id)
                    if action_event is None or not isinstance(
                        action_event.payload, ActionCommittedPayload
                    ):
                        raise ProjectionLinkError(
                            f"TrialCompleted {event.event_id} terminal action "
                            f"{action_event_id} is missing or not committed"
                        )
                captured_hashes = {
                    candidate.payload.artifact.artifact_hash
                    for candidate in stream.events
                    if isinstance(candidate.payload, ActivationCapturedPayload)
                    and candidate.sequence_num < event.sequence_num
                }
                missing_artifacts = set(payload.required_artifact_hashes).difference(
                    captured_hashes
                )
                if missing_artifacts:
                    raise ProjectionLinkError(
                        f"TrialCompleted {event.event_id} requires unrecorded "
                        f"activation artifacts: {', '.join(sorted(missing_artifacts))}"
                    )
                terminal_event_id = event.event_id
                lifecycle_status = "completed"
            elif isinstance(payload, TrialFailedPayload):
                if payload.last_event_id is not None:
                    last_event = stream.by_id.get(payload.last_event_id)
                    if last_event is None or last_event.sequence_num >= event.sequence_num:
                        raise ProjectionLinkError(
                            f"TrialFailed {event.event_id} last_event_id does not "
                            "reference an earlier event"
                        )
                terminal_event_id = event.event_id
                lifecycle_status = "failed"
            elif isinstance(payload, ActionCommittedPayload):
                commitments.append(_event_fact(event))
            elif event.event_type in _INTERVENTION_TYPES:
                interventions.append(_event_fact(event))
            elif isinstance(payload, OutcomeResolvedPayload):
                outcomes.append(_outcome_projection(event))
            elif isinstance(payload, TurnAdvancedPayload):
                actor_role = stream.actor_roles.get(payload.to_actor_id)
                prior_role = stream.actor_roles.get(payload.from_actor_id)
                if actor_role is None or prior_role is None:
                    raise ProjectionIdentityError(
                        f"TurnAdvanced {event.event_id} lacks actor roles"
                    )
                current_turn = CurrentTurnProjection(
                    turn_id=payload.turn_id,
                    turn_event_id=event.event_id,
                    actor_id=payload.to_actor_id,
                    actor_role=actor_role,
                    prior_actor_id=payload.from_actor_id,
                    prior_actor_role=prior_role,
                    committed_action_event_id=payload.committed_action_event_id,
                    next_sequence_num=payload.next_sequence_num,
                )

        if len(outcomes) > 1:
            raise ProjectionLinkError(
                f"trial {stream.trial_id} has multiple OutcomeResolved events"
            )
        return self._finish(
            TrialStateProjection(
                run_id=stream.run_id,
                pod_id=stream.pod_id,
                trial_id=stream.trial_id,
                dyad_id=stream.dyad_id,
                lifecycle_status=lifecycle_status,
                started_event_id=started_event_id,
                terminal_event_id=terminal_event_id,
                current_turn=current_turn,
                commitments=tuple(commitments),
                interventions=tuple(interventions),
                labels=labels,
                monitor_scores=monitors,
                quality_controls=quality_controls,
                outcomes=tuple(outcomes),
            )
        )


class ActivationSampleProjector(_ProjectorBase):
    """Project activation-bearing negotiation actions with exact call lineage."""

    _required_event_types = _CALL_ACTION_TYPES | _ANNOTATION_TYPES

    def project(self, events: Iterable[EventEnvelope]) -> ActivationSampleProjection:
        stream = self._validate(events)
        index = _interaction_index(stream)
        labels, monitors, quality_controls = _annotations(stream)
        samples: list[ProjectedActivationSample] = []
        consumed_activation_events: set[str] = set()

        for committed_event in stream.events:
            if not isinstance(committed_event.payload, ActionCommittedPayload):
                continue
            committed = committed_event.payload
            activation_events = index.activations_by_call.get(
                committed.model_call_id, ()
            )
            if not activation_events:
                continue
            started_event = index.starts_by_call.get(committed.model_call_id)
            completed_event = index.completed_by_call.get(committed.model_call_id)
            proposal_event = index.proposals_by_event.get(committed.proposed_event_id)
            turn_event = index.turns_by_commitment.get(committed_event.event_id)
            if (
                started_event is None
                or completed_event is None
                or proposal_event is None
                or turn_event is None
            ):
                raise ProjectionLinkError(
                    f"activation-bearing action {committed_event.event_id} has "
                    "incomplete call/action/turn lineage"
                )
            started = _expect_payload(started_event, ModelCallStartedPayload)
            completed = _expect_payload(completed_event, ModelCallCompletedPayload)
            proposal = _expect_payload(proposal_event, ActionProposedPayload)
            turn = _expect_payload(turn_event, TurnAdvancedPayload)
            if started.purpose not in {"actor_action", "counterpart_action"}:
                raise ProjectionLinkError(
                    f"activation-bearing committed action {committed_event.event_id} "
                    f"uses non-negotiation call purpose {started.purpose}"
                )
            if stream.run_id is None or stream.pod_id is None or stream.trial_id is None:
                raise ProjectionIdentityError(
                    "activation samples require run, pod, and trial identity"
                )
            role = stream.actor_roles.get(committed.actor_id)
            if role is None:
                raise ProjectionIdentityError(
                    f"activation-bearing actor {committed.actor_id} has no role"
                )
            artifacts = tuple(
                _expect_payload(event, ActivationCapturedPayload).artifact
                for event in activation_events
            )
            consumed_activation_events.update(
                event.event_id for event in activation_events
            )
            lineage_ids = {
                started_event.event_id,
                completed_event.event_id,
                *(event.event_id for event in activation_events),
                proposal_event.event_id,
                committed_event.event_id,
                turn_event.event_id,
            }
            unavailable = (
                ProjectionWarning(
                    field_name="round_num",
                    code="legacy_field_unavailable",
                    detail=(
                        "The event schema records stable turn_id, not a fabricated "
                        "legacy numeric round"
                    ),
                    source_event_id=turn_event.event_id,
                ),
                ProjectionWarning(
                    field_name="prompt",
                    code="content_unavailable",
                    detail="ModelCallStarted stores prompt identity and hash, not text",
                    source_event_id=started_event.event_id,
                ),
                ProjectionWarning(
                    field_name="response",
                    code="content_unavailable",
                    detail="ActionCommitted stores action identity and hash, not text",
                    source_event_id=committed_event.event_id,
                ),
                ProjectionWarning(
                    field_name="activation_values",
                    code="artifact_not_loaded",
                    detail=(
                        "The pure projector preserves artifact references but does "
                        "not read activation files"
                    ),
                    source_event_id=activation_events[0].event_id,
                ),
            )
            samples.append(
                ProjectedActivationSample(
                    run_id=stream.run_id,
                    pod_id=stream.pod_id,
                    trial_id=stream.trial_id,
                    dyad_id=stream.dyad_id,
                    actor_id=committed.actor_id,
                    actor_role=role,
                    call_purpose=started.purpose,
                    model_call_id=committed.model_call_id,
                    model_id=started.model_id,
                    model_revision=started.model_revision,
                    tokenizer_id=started.tokenizer_id,
                    tokenizer_revision=started.tokenizer_revision,
                    generation_config_json=started.generation_config.canonical_json,
                    generation_config_hash=completed.generation_config_hash,
                    model_call_started_event_id=started_event.event_id,
                    model_call_completed_event_id=completed_event.event_id,
                    activation_event_ids=tuple(
                        event.event_id for event in activation_events
                    ),
                    activation_artifacts=artifacts,
                    action_id=proposal.action_id,
                    action_hash=committed.action_hash,
                    action_proposed_event_id=proposal_event.event_id,
                    action_committed_event_id=committed_event.event_id,
                    turn_event_id=turn_event.event_id,
                    turn_id=turn.turn_id,
                    annotation_event_ids=_annotation_ids_for(
                        lineage_ids, labels, monitors, quality_controls
                    ),
                    warnings=unavailable,
                )
            )

        orphaned: list[str] = []
        for call_id, activation_events in index.activations_by_call.items():
            started_event = index.starts_by_call.get(call_id)
            if started_event is None:
                continue
            started = _expect_payload(started_event, ModelCallStartedPayload)
            if started.purpose in {"actor_action", "counterpart_action"}:
                orphaned.extend(
                    event.event_id
                    for event in activation_events
                    if event.event_id not in consumed_activation_events
                )
        if orphaned:
            raise ProjectionLinkError(
                "negotiation activation events are not linked to a committed "
                f"action: {', '.join(orphaned)}"
            )
        return self._finish(
            ActivationSampleProjection(
                run_id=stream.run_id,
                pod_id=stream.pod_id,
                trial_id=stream.trial_id,
                dyad_id=stream.dyad_id,
                samples=tuple(samples),
                labels=labels,
                monitor_scores=monitors,
                quality_controls=quality_controls,
            )
        )


class DyadProjector(_ProjectorBase):
    """Pair each committed turn with its stable dyad counterpart and reception."""

    _required_event_types = (
        _CALL_ACTION_TYPES
        | _ANNOTATION_TYPES
        | frozenset({"TrialStarted", "AgentBuilt", "ObservationDelivered"})
    )

    def project(self, events: Iterable[EventEnvelope]) -> DyadProjection:
        stream = self._validate(events)
        index = _interaction_index(stream)
        labels, monitors, quality_controls = _annotations(stream)
        trial_start: TrialStartedPayload | None = None
        for event in stream.events:
            if isinstance(event.payload, TrialStartedPayload):
                trial_start = event.payload
                break
        if trial_start is None:
            if stream.events:
                raise ProjectionLifecycleError("dyad projection requires TrialStarted")
            return self._finish(
                DyadProjection(
                    run_id=None,
                    pod_id=None,
                    trial_id=None,
                    dyad_id=None,
                    actor_ids=(),
                    actor_roles=(),
                    turns=(),
                )
            )
        if len(trial_start.actor_ids) != 2:
            raise ProjectionIdentityError(
                f"dyad {trial_start.dyad_id} must declare exactly two actors"
            )
        actor_ids = trial_start.actor_ids
        roles: list[tuple[str, str]] = []
        for actor_id in actor_ids:
            role = stream.actor_roles.get(actor_id)
            if role is None:
                raise ProjectionIdentityError(
                    f"dyad actor {actor_id} has no stable role"
                )
            roles.append((actor_id, role))

        observations_by_source: dict[str, list[EventEnvelope]] = {}
        for event in stream.events:
            if isinstance(event.payload, ObservationDeliveredPayload):
                observations_by_source.setdefault(
                    event.payload.source_event_id, []
                ).append(event)

        turns: list[DyadTurnProjection] = []
        for event in stream.events:
            if not isinstance(event.payload, ActionCommittedPayload):
                continue
            committed = event.payload
            turn_event = index.turns_by_commitment.get(event.event_id)
            if turn_event is None:
                raise ProjectionLinkError(
                    f"dyad action {event.event_id} has no TurnAdvanced event"
                )
            turn = _expect_payload(turn_event, TurnAdvancedPayload)
            if turn.to_actor_id not in actor_ids or committed.actor_id not in actor_ids:
                raise ProjectionIdentityError(
                    f"turn {turn_event.event_id} is outside declared dyad"
                )
            expected_counterpart = (
                actor_ids[1] if committed.actor_id == actor_ids[0] else actor_ids[0]
            )
            if turn.to_actor_id != expected_counterpart:
                raise ProjectionIdentityError(
                    f"turn {turn_event.event_id} does not advance to the other "
                    "dyad actor"
                )
            receptions = tuple(
                observation
                for observation in observations_by_source.get(event.event_id, ())
                if isinstance(observation.payload, ObservationDeliveredPayload)
                and observation.payload.recipient_actor_id == expected_counterpart
            )
            if len(receptions) > 1:
                raise ProjectionLinkError(
                    f"action {event.event_id} has multiple deliveries to "
                    f"{expected_counterpart}"
                )

            response_event: EventEnvelope | None = None
            if receptions:
                reception_id = receptions[0].event_id
                for proposal_event in index.proposals_by_event.values():
                    proposal = _expect_payload(
                        proposal_event, ActionProposedPayload
                    )
                    if (
                        proposal.actor_id == expected_counterpart
                        and reception_id in proposal.source_observation_event_ids
                    ):
                        candidate = index.commitments_by_proposal.get(
                            proposal_event.event_id
                        )
                        if candidate is not None:
                            if response_event is not None:
                                raise ProjectionLinkError(
                                    f"reception {reception_id} links to multiple "
                                    "committed responses"
                                )
                            response_event = candidate

            warning: tuple[ProjectionWarning, ...] = ()
            if not receptions:
                warning = (
                    ProjectionWarning(
                        field_name="reception_event_ids",
                        code="reception_unavailable",
                        detail=(
                            "No ObservationDelivered event links this committed "
                            "action to its counterpart"
                        ),
                        source_event_id=event.event_id,
                    ),
                )
            response_payload = (
                None
                if response_event is None
                else _expect_payload(response_event, ActionCommittedPayload)
            )
            target_ids = {
                event.event_id,
                committed.proposed_event_id,
                turn_event.event_id,
                *(reception.event_id for reception in receptions),
            }
            turns.append(
                DyadTurnProjection(
                    ordinal=len(turns),
                    dyad_id=trial_start.dyad_id,
                    trial_id=trial_start.trial_id,
                    turn_id=turn.turn_id,
                    turn_event_id=turn_event.event_id,
                    actor_id=committed.actor_id,
                    actor_role=stream.actor_roles[committed.actor_id],
                    counterpart_actor_id=expected_counterpart,
                    counterpart_role=stream.actor_roles[expected_counterpart],
                    action_id=committed.action_id,
                    action_hash=committed.action_hash,
                    action_event_id=event.event_id,
                    model_call_id=committed.model_call_id,
                    reception_event_ids=tuple(
                        reception.event_id for reception in receptions
                    ),
                    response_action_event_id=(
                        None if response_event is None else response_event.event_id
                    ),
                    response_model_call_id=(
                        None
                        if response_payload is None
                        else response_payload.model_call_id
                    ),
                    annotation_event_ids=_annotation_ids_for(
                        target_ids, labels, monitors, quality_controls
                    ),
                    warnings=warning,
                )
            )
        return self._finish(
            DyadProjection(
                run_id=stream.run_id,
                pod_id=stream.pod_id,
                trial_id=stream.trial_id,
                dyad_id=stream.dyad_id,
                actor_ids=actor_ids,
                actor_roles=tuple(roles),
                turns=tuple(turns),
            )
        )


class MetricInputProjector(_ProjectorBase):
    """Project row-level label, monitor, and outcome inputs without aggregation."""

    _required_event_types = frozenset(
        {"BehaviorLabeled", "MonitorScored", "OutcomeResolved"}
    )

    def project(self, events: Iterable[EventEnvelope]) -> MetricInputProjection:
        stream = self._validate(events)
        labels: list[BehaviorLabelProjection] = []
        monitors: list[MonitorScoreProjection] = []
        outcomes: list[OutcomeProjection] = []
        for event in stream.events:
            if isinstance(event.payload, BehaviorLabeledPayload):
                labels.append(_label_projection(event))
            elif isinstance(event.payload, MonitorScoredPayload):
                monitors.append(_monitor_projection(event))
            elif isinstance(event.payload, OutcomeResolvedPayload):
                outcomes.append(_outcome_projection(event))
        return self._finish(
            MetricInputProjection(
                run_id=stream.run_id,
                pod_id=stream.pod_id,
                trial_id=stream.trial_id,
                dyad_id=stream.dyad_id,
                labels=tuple(labels),
                monitor_scores=tuple(monitors),
                outcomes=tuple(outcomes),
            )
        )


__all__ = [
    "ActivationSampleProjection",
    "ActivationSampleProjector",
    "AgentViewProjection",
    "AgentViewProjector",
    "BehaviorLabelProjection",
    "CurrentTurnProjection",
    "DyadProjection",
    "DyadProjector",
    "DyadTurnProjection",
    "MetricInputProjection",
    "MetricInputProjector",
    "MonitorScoreProjection",
    "ObservationProjection",
    "OutcomeProjection",
    "PrivateViewProjection",
    "ProjectedActivationSample",
    "ProjectedEventFact",
    "ProjectionError",
    "ProjectionIdentityError",
    "ProjectionLifecycleError",
    "ProjectionLinkError",
    "ProjectionSequenceError",
    "ProjectionWarning",
    "QualityControlProjection",
    "SemanticProjection",
    "TranscriptAction",
    "TranscriptProjection",
    "TranscriptProjector",
    "TrialStateProjection",
    "TrialStateProjector",
    "UnprojectableEventError",
]
