"""Versioned typed payload catalog for the initial event-sourced runtime.

Payloads store stable identities, hashes, typed status, and canonical versioned
documents.  They deliberately do not store hidden chain-of-thought, raw prompt
or output text, credentials, or mutable unversioned blobs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from interpretability.events.schema import EventPayload, register_payload

PAYLOAD_SCHEMA_VERSION = "1.0.0"

_UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_HASH_PATTERN = r"^[0-9a-f]{64}$"
_STABLE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$"
_VERSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._+/-]{0,127}$"
_SENSITIVE_CONFIG_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
    }
)

EventId = Annotated[str, StringConstraints(pattern=_UUID_PATTERN)]
RunId = EventId
StableId = Annotated[str, StringConstraints(pattern=_STABLE_ID_PATTERN)]
VersionId = Annotated[str, StringConstraints(pattern=_VERSION_PATTERN)]
Sha256 = Annotated[str, StringConstraints(pattern=_HASH_PATTERN)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]

ModelCallPurpose = Literal[
    "actor_action",
    "counterpart_action",
    "tom_inference",
    "component_analysis",
    "belief_verification",
    "plausibility_probe",
    "judge",
    "monitor",
]


def _unique_tuple(value: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(value) != len(set(value)):
        raise ValueError("ordered references must not contain duplicates")
    return value


OrderedEventIds = Annotated[tuple[EventId, ...], AfterValidator(_unique_tuple)]
NonEmptyEventIds = Annotated[
    tuple[EventId, ...], Field(min_length=1), AfterValidator(_unique_tuple)
]
OrderedHashes = Annotated[tuple[Sha256, ...], AfterValidator(_unique_tuple)]
OrderedStableIds = Annotated[tuple[StableId, ...], AfterValidator(_unique_tuple)]
NonEmptyStableIds = Annotated[
    tuple[StableId, ...], Field(min_length=1), AfterValidator(_unique_tuple)
]


class _StrictValue(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _scan_canonical_value(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string key")
            normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
            if normalized in _SENSITIVE_CONFIG_KEYS:
                raise ValueError(f"{path}.{key} is a prohibited secret-bearing field")
            _scan_canonical_value(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _scan_canonical_value(item, f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"{path} contains a non-finite or unsupported JSON value")


def _canonical_json(value: Mapping[str, Any]) -> str:
    _scan_canonical_value(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


class CanonicalJsonDocument(_StrictValue):
    """A canonical, versioned, content-verified JSON object without secrets."""

    schema_version: VersionId
    canonical_json: str
    sha256: Sha256

    @model_validator(mode="after")
    def _verify_document(self) -> CanonicalJsonDocument:
        try:
            parsed = json.loads(
                self.canonical_json,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value}")
                ),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("canonical_json is not valid strict JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("canonical_json must contain a JSON object")
        canonical = _canonical_json(parsed)
        if canonical != self.canonical_json:
            raise ValueError("canonical_json is not compact sorted canonical JSON")
        expected_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if self.sha256 != expected_hash:
            raise ValueError("canonical_json SHA-256 mismatch")
        return self

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, schema_version: str
    ) -> CanonicalJsonDocument:
        if not isinstance(value, Mapping):
            raise TypeError("canonical document input must be a mapping")
        canonical = _canonical_json(dict(value))
        return cls(
            schema_version=schema_version,
            canonical_json=canonical,
            sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )


class UsageRecord(_StrictValue):
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    total_tokens: NonNegativeInt
    cached_input_tokens: NonNegativeInt = 0
    latency_seconds: Annotated[float, Field(ge=0.0)] | None = None
    cost_usd: Annotated[float, Field(ge=0.0)] | None = None

    @model_validator(mode="after")
    def _token_totals_match(self) -> UsageRecord:
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached_input_tokens cannot exceed input_tokens")
        return self


class LabelValue(_StrictValue):
    kind: Literal["boolean", "score", "category"]
    boolean_value: bool | None = None
    score_value: Probability | None = None
    category_value: StableId | None = None

    @model_validator(mode="after")
    def _exactly_one_typed_value(self) -> LabelValue:
        present = {
            "boolean": self.boolean_value is not None,
            "score": self.score_value is not None,
            "category": self.category_value is not None,
        }
        if sum(present.values()) != 1 or not present[self.kind]:
            raise ValueError("label kind must select exactly one matching value")
        return self


class LabelProvenance(_StrictValue):
    source: Literal[
        "rules",
        "judge",
        "human",
        "monitor",
        "scenario_ground_truth",
    ]
    method_id: StableId
    method_version: VersionId
    source_event_ids: NonEmptyEventIds
    evaluation_succeeded: bool
    fallback_reason_hash: Sha256 | None = None

    @model_validator(mode="after")
    def _fallback_is_explicit(self) -> LabelProvenance:
        if self.evaluation_succeeded and self.fallback_reason_hash is not None:
            raise ValueError("successful evaluation cannot have a fallback reason")
        if not self.evaluation_succeeded and self.fallback_reason_hash is None:
            raise ValueError("failed evaluation requires a fallback reason hash")
        return self


@register_payload("RunStarted", PAYLOAD_SCHEMA_VERSION)
class RunStartedPayload(EventPayload):
    run_id: RunId
    orchestrator_id: StableId
    code_revision: StableId
    run_seed: NonNegativeInt
    parent_run_id: RunId | None = None


@register_payload("RunConfigFrozen", PAYLOAD_SCHEMA_VERSION)
class RunConfigFrozenPayload(EventPayload):
    run_id: RunId
    config: CanonicalJsonDocument
    code_revision: StableId
    working_tree_dirty: bool
    python_version: StableId
    package_lock_hash: Sha256
    scenario_spec_hashes: OrderedHashes = ()


@register_payload("RunCompleted", PAYLOAD_SCHEMA_VERSION)
class RunCompletedPayload(EventPayload):
    run_id: RunId
    status: Literal["completed"] = "completed"
    completed_trials: NonNegativeInt
    failed_trials: NonNegativeInt
    open_trials: Literal[0] = 0
    terminal_event_ids: OrderedEventIds = ()


@register_payload("RunFailed", PAYLOAD_SCHEMA_VERSION)
class RunFailedPayload(EventPayload):
    run_id: RunId
    status: Literal["failed"] = "failed"
    error_type: StableId
    error_message_hash: Sha256
    resumable: bool
    failed_event_id: EventId | None = None
    open_trial_ids: OrderedStableIds = ()


@register_payload("TrialStarted", PAYLOAD_SCHEMA_VERSION)
class TrialStartedPayload(EventPayload):
    trial_id: StableId
    scenario_instance_id: StableId
    dyad_id: StableId
    attempt: PositiveInt
    actor_ids: NonEmptyStableIds
    source_scenario_event_id: EventId


@register_payload("TrialCompleted", PAYLOAD_SCHEMA_VERSION)
class TrialCompletedPayload(EventPayload):
    trial_id: StableId
    status: Literal["completed"] = "completed"
    outcome_event_id: EventId
    terminal_action_event_ids: OrderedEventIds = ()
    required_artifact_hashes: OrderedHashes = ()


@register_payload("TrialFailed", PAYLOAD_SCHEMA_VERSION)
class TrialFailedPayload(EventPayload):
    trial_id: StableId
    status: Literal["failed"] = "failed"
    error_type: StableId
    error_message_hash: Sha256
    resumable: bool
    last_event_id: EventId | None = None


@register_payload("ScenarioInstantiated", PAYLOAD_SCHEMA_VERSION)
class ScenarioInstantiatedPayload(EventPayload):
    scenario_instance_id: StableId
    scenario_type: StableId
    scenario_schema_version: VersionId
    scenario_spec_hash: Sha256
    parameters: CanonicalJsonDocument
    trial_seed: NonNegativeInt

    @model_validator(mode="after")
    def _parameter_schema_matches(self) -> ScenarioInstantiatedPayload:
        if self.parameters.schema_version != self.scenario_schema_version:
            raise ValueError("parameter schema version must match scenario schema")
        return self


@register_payload("PrivateViewAssigned", PAYLOAD_SCHEMA_VERSION)
class PrivateViewAssignedPayload(EventPayload):
    scenario_instance_id: StableId
    view_id: StableId
    recipient_actor_id: StableId
    recipient_role: StableId
    view_schema_version: VersionId
    view_hash: Sha256
    source_scenario_event_id: EventId


@register_payload("InterventionScheduled", PAYLOAD_SCHEMA_VERSION)
class InterventionScheduledPayload(EventPayload):
    intervention_id: StableId
    scenario_instance_id: StableId
    target_actor_id: StableId
    intervention_type: StableId
    scheduled_sequence_num: NonNegativeInt
    specification: CanonicalJsonDocument
    source_event_ids: NonEmptyEventIds


@register_payload("AgentBuilt", PAYLOAD_SCHEMA_VERSION)
class AgentBuiltPayload(EventPayload):
    actor_id: StableId
    role: StableId
    entity_id: StableId
    model_id: StableId
    model_revision: StableId
    component_names: NonEmptyStableIds
    component_config: CanonicalJsonDocument
    scenario_event_id: EventId


@register_payload("ObservationDelivered", PAYLOAD_SCHEMA_VERSION)
class ObservationDeliveredPayload(EventPayload):
    observation_id: StableId
    recipient_actor_id: StableId
    source_actor_id: StableId | None
    source_event_id: EventId
    content_hash: Sha256
    visibility: Literal["public", "private"]
    sequence_in_recipient_view: NonNegativeInt


@register_payload("ComponentContextProduced", PAYLOAD_SCHEMA_VERSION)
class ComponentContextProducedPayload(EventPayload):
    context_id: StableId
    actor_id: StableId
    component_name: StableId
    component_index: NonNegativeInt
    model_call_id: StableId
    input_event_ids: OrderedEventIds
    context_hash: Sha256
    context_schema_version: VersionId
    contains_hidden_chain_of_thought: Literal[False] = False


@register_payload("ToMStateUpdated", PAYLOAD_SCHEMA_VERSION)
class ToMStateUpdatedPayload(EventPayload):
    state_id: StableId
    actor_id: StableId
    counterpart_actor_id: StableId
    state_schema_version: VersionId
    state_hash: Sha256
    evidence_event_ids: NonEmptyEventIds
    source_model_call_id: StableId

    @model_validator(mode="after")
    def _counterpart_is_distinct(self) -> ToMStateUpdatedPayload:
        if self.actor_id == self.counterpart_actor_id:
            raise ValueError("counterpart_actor_id must differ from actor_id")
        return self


@register_payload("ModelCallStarted", PAYLOAD_SCHEMA_VERSION)
class ModelCallStartedPayload(EventPayload):
    model_call_id: StableId
    purpose: ModelCallPurpose
    actor_id: StableId
    model_id: StableId
    model_revision: StableId
    tokenizer_id: StableId
    tokenizer_revision: StableId
    prompt_id: StableId
    prompt_hash: Sha256
    input_event_ids: OrderedEventIds
    generation_config: CanonicalJsonDocument
    started_by_event_id: EventId | None = None


@register_payload("ModelCallCompleted", PAYLOAD_SCHEMA_VERSION)
class ModelCallCompletedPayload(EventPayload):
    model_call_id: StableId
    purpose: ModelCallPurpose
    actor_id: StableId
    status: Literal["completed"] = "completed"
    started_event_id: EventId
    output_id: StableId
    output_hash: Sha256
    token_ids_hash: Sha256 | None
    generation_config_hash: Sha256
    usage: UsageRecord
    finish_reason: Literal["stop", "length", "tool", "content_filter", "other"]
    activation_artifact_hashes: OrderedHashes = ()


@register_payload("ModelCallFailed", PAYLOAD_SCHEMA_VERSION)
class ModelCallFailedPayload(EventPayload):
    model_call_id: StableId
    purpose: ModelCallPurpose
    actor_id: StableId
    status: Literal["failed"] = "failed"
    started_event_id: EventId
    generation_config_hash: Sha256
    error_type: StableId
    error_message_hash: Sha256
    retryable: bool


@register_payload("ActionProposed", PAYLOAD_SCHEMA_VERSION)
class ActionProposedPayload(EventPayload):
    action_id: StableId
    actor_id: StableId
    status: Literal["proposed"] = "proposed"
    model_call_id: StableId
    model_call_event_id: EventId
    action_spec_id: StableId
    action_hash: Sha256
    source_observation_event_ids: OrderedEventIds = ()


@register_payload("ActionCommitted", PAYLOAD_SCHEMA_VERSION)
class ActionCommittedPayload(EventPayload):
    action_id: StableId
    actor_id: StableId
    status: Literal["committed"] = "committed"
    proposed_event_id: EventId
    model_call_id: StableId
    action_hash: Sha256
    protocol_decision_event_id: EventId | None = None


@register_payload("TurnAdvanced", PAYLOAD_SCHEMA_VERSION)
class TurnAdvancedPayload(EventPayload):
    turn_id: StableId
    trial_id: StableId
    from_actor_id: StableId
    to_actor_id: StableId
    committed_action_event_id: EventId
    next_sequence_num: NonNegativeInt

    @model_validator(mode="after")
    def _turn_changes_actor(self) -> TurnAdvancedPayload:
        if self.from_actor_id == self.to_actor_id:
            raise ValueError("turn must advance to a different actor")
        return self


@register_payload("BehaviorLabeled", PAYLOAD_SCHEMA_VERSION)
class BehaviorLabeledPayload(EventPayload):
    label_id: StableId
    target_event_id: EventId
    target_actor_id: StableId
    label_name: StableId
    value: LabelValue
    provenance: LabelProvenance

    @model_validator(mode="after")
    def _target_is_in_provenance(self) -> BehaviorLabeledPayload:
        if self.target_event_id not in self.provenance.source_event_ids:
            raise ValueError("label provenance must reference the target event")
        return self


@register_payload("MonitorScored", PAYLOAD_SCHEMA_VERSION)
class MonitorScoredPayload(EventPayload):
    monitor_id: StableId
    monitor_version: VersionId
    target_event_id: EventId
    target_actor_id: StableId
    score: Probability
    threshold: Probability | None
    flagged: bool
    evidence_event_ids: NonEmptyEventIds

    @model_validator(mode="after")
    def _flag_matches_threshold(self) -> MonitorScoredPayload:
        if self.threshold is None and self.flagged:
            raise ValueError("flagged monitor score requires a threshold")
        if self.threshold is not None and self.flagged != (self.score >= self.threshold):
            raise ValueError("flagged must equal score >= threshold")
        if self.target_event_id not in self.evidence_event_ids:
            raise ValueError("monitor evidence must reference the target event")
        return self


@register_payload("OutcomeResolved", PAYLOAD_SCHEMA_VERSION)
class OutcomeResolvedPayload(EventPayload):
    outcome_id: StableId
    trial_id: StableId
    resolver_id: StableId
    resolver_version: VersionId
    outcome: CanonicalJsonDocument
    source_event_ids: NonEmptyEventIds
    success: bool | None = None
    score: float | None = None


@register_payload("QualityControlApplied", PAYLOAD_SCHEMA_VERSION)
class QualityControlAppliedPayload(EventPayload):
    qc_id: StableId
    qc_version: VersionId
    target_event_id: EventId
    passed: bool
    flags: OrderedStableIds
    source_event_ids: NonEmptyEventIds

    @model_validator(mode="after")
    def _flags_match_status(self) -> QualityControlAppliedPayload:
        if self.passed and self.flags:
            raise ValueError("passed quality control cannot have failure flags")
        if not self.passed and not self.flags:
            raise ValueError("failed quality control requires at least one flag")
        if self.target_event_id not in self.source_event_ids:
            raise ValueError("quality-control provenance must reference target event")
        return self


@register_payload("BeliefIntervened", PAYLOAD_SCHEMA_VERSION)
class BeliefIntervenedPayload(EventPayload):
    intervention_id: StableId
    actor_id: StableId
    counterpart_actor_id: StableId
    source_state_event_id: EventId
    result_state_id: StableId
    result_state_schema_version: VersionId
    result_state_hash: Sha256
    method_id: StableId
    parameters: CanonicalJsonDocument
    source_model_call_id: StableId | None = None

    @model_validator(mode="after")
    def _belief_target_is_distinct(self) -> BeliefIntervenedPayload:
        if self.actor_id == self.counterpart_actor_id:
            raise ValueError("belief intervention counterpart must differ from actor")
        return self


@register_payload("ActivationIntervened", PAYLOAD_SCHEMA_VERSION)
class ActivationIntervenedPayload(EventPayload):
    intervention_id: StableId
    actor_id: StableId
    model_call_id: StableId
    source_activation_event_id: EventId
    hook_name: StableId
    layer: NonNegativeInt
    direction_hash: Sha256
    magnitude: float
    method_id: StableId
    is_control: bool


@register_payload("ProtocolDecisionApplied", PAYLOAD_SCHEMA_VERSION)
class ProtocolDecisionAppliedPayload(EventPayload):
    decision_id: StableId
    module_name: StableId
    trial_id: StableId
    target_action_event_id: EventId
    decision: Literal["accept", "reject", "modify", "defer"]
    reason_code: StableId
    source_event_ids: NonEmptyEventIds
    modified_action_hash: Sha256 | None = None

    @model_validator(mode="after")
    def _modified_action_is_explicit(self) -> ProtocolDecisionAppliedPayload:
        if self.target_action_event_id not in self.source_event_ids:
            raise ValueError("protocol decision must reference its target action")
        if (self.decision == "modify") != (self.modified_action_hash is not None):
            raise ValueError("only a modify decision requires modified_action_hash")
        return self


INITIAL_PAYLOAD_TYPES: tuple[type[EventPayload], ...] = (
    RunStartedPayload,
    RunConfigFrozenPayload,
    RunCompletedPayload,
    RunFailedPayload,
    TrialStartedPayload,
    TrialCompletedPayload,
    TrialFailedPayload,
    ScenarioInstantiatedPayload,
    PrivateViewAssignedPayload,
    InterventionScheduledPayload,
    AgentBuiltPayload,
    ObservationDeliveredPayload,
    ComponentContextProducedPayload,
    ToMStateUpdatedPayload,
    ModelCallStartedPayload,
    ModelCallCompletedPayload,
    ModelCallFailedPayload,
    ActionProposedPayload,
    ActionCommittedPayload,
    TurnAdvancedPayload,
    BehaviorLabeledPayload,
    MonitorScoredPayload,
    OutcomeResolvedPayload,
    QualityControlAppliedPayload,
    BeliefIntervenedPayload,
    ActivationIntervenedPayload,
    ProtocolDecisionAppliedPayload,
)


__all__ = [
    "INITIAL_PAYLOAD_TYPES",
    "PAYLOAD_SCHEMA_VERSION",
    "ActionCommittedPayload",
    "ActionProposedPayload",
    "ActivationIntervenedPayload",
    "AgentBuiltPayload",
    "BehaviorLabeledPayload",
    "BeliefIntervenedPayload",
    "CanonicalJsonDocument",
    "ComponentContextProducedPayload",
    "InterventionScheduledPayload",
    "LabelProvenance",
    "LabelValue",
    "ModelCallCompletedPayload",
    "ModelCallFailedPayload",
    "ModelCallStartedPayload",
    "MonitorScoredPayload",
    "ObservationDeliveredPayload",
    "OutcomeResolvedPayload",
    "PrivateViewAssignedPayload",
    "ProtocolDecisionAppliedPayload",
    "QualityControlAppliedPayload",
    "RunCompletedPayload",
    "RunConfigFrozenPayload",
    "RunFailedPayload",
    "RunStartedPayload",
    "ScenarioInstantiatedPayload",
    "ToMStateUpdatedPayload",
    "TrialCompletedPayload",
    "TrialFailedPayload",
    "TrialStartedPayload",
    "TurnAdvancedPayload",
    "UsageRecord",
]
