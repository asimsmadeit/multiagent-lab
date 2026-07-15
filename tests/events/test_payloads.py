"""Permanent construction and validation contract for the typed payload catalog."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from interpretability.events.payloads import (
    INITIAL_PAYLOAD_TYPES,
    PAYLOAD_SCHEMA_VERSION,
    ActionCommittedPayload,
    ActionProposedPayload,
    ActivationIntervenedPayload,
    AgentBuiltPayload,
    BehaviorLabeledPayload,
    BeliefIntervenedPayload,
    CanonicalJsonDocument,
    ComponentContextProducedPayload,
    InterventionScheduledPayload,
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
    RunCompletedPayload,
    RunConfigFrozenPayload,
    RunFailedPayload,
    RunStartedPayload,
    ScenarioInstantiatedPayload,
    ToMStateUpdatedPayload,
    TrialCompletedPayload,
    TrialFailedPayload,
    TrialStartedPayload,
    TurnAdvancedPayload,
    UsageRecord,
)
from interpretability.events.schema import (
    ActivationCapturedPayload,
    EventEnvelope,
    EventPayload,
    OpaqueEventPayload,
    UnknownPayloadTypeError,
    parse_payload,
    registered_payload_types,
)

E1 = "00000000-0000-4000-8000-000000000001"
E2 = "00000000-0000-4000-8000-000000000002"
E3 = "00000000-0000-4000-8000-000000000003"
RUN_ID = "00000000-0000-4000-8000-000000000004"
HASH_A = "a" * 64
HASH_B = "b" * 64


def canonical_documents() -> tuple[
    CanonicalJsonDocument, CanonicalJsonDocument, CanonicalJsonDocument
]:
    generation = CanonicalJsonDocument.from_mapping(
        {"max_tokens": 128, "temperature": 0.5},
        schema_version="generation/1.0",
    )
    scenario = CanonicalJsonDocument.from_mapping(
        {"roles": ["seller", "buyer"], "stakes": 10},
        schema_version="scenario/1.0",
    )
    outcome = CanonicalJsonDocument.from_mapping(
        {"agreement": True, "value": 5}, schema_version="outcome/1.0"
    )
    return generation, scenario, outcome


def payload_examples() -> tuple[EventPayload, ...]:
    config, scenario, outcome = canonical_documents()
    usage = UsageRecord(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cached_input_tokens=2,
        latency_seconds=0.2,
        cost_usd=0.01,
    )
    provenance = LabelProvenance(
        source="rules",
        method_id="rules-v1",
        method_version="1.0",
        source_event_ids=(E1,),
        evaluation_succeeded=True,
    )
    return (
        RunStartedPayload(
            run_id=RUN_ID,
            orchestrator_id="runner-1",
            code_revision="abc123",
            run_seed=1,
        ),
        RunConfigFrozenPayload(
            run_id=RUN_ID,
            config=config,
            code_revision="abc123",
            working_tree_dirty=False,
            python_version="3.12.0",
            package_lock_hash=HASH_A,
            scenario_spec_hashes=(HASH_B,),
        ),
        RunCompletedPayload(
            run_id=RUN_ID,
            completed_trials=1,
            failed_trials=0,
            terminal_event_ids=(E1,),
        ),
        RunFailedPayload(
            run_id=RUN_ID,
            error_type="RuntimeError",
            error_message_hash=HASH_A,
            resumable=True,
            failed_event_id=E1,
            open_trial_ids=("trial-1",),
        ),
        TrialStartedPayload(
            trial_id="trial-1",
            scenario_instance_id="scenario-1",
            dyad_id="dyad-1",
            attempt=1,
            actor_ids=("agent-a", "agent-b"),
            source_scenario_event_id=E1,
        ),
        TrialCompletedPayload(
            trial_id="trial-1",
            outcome_event_id=E1,
            terminal_action_event_ids=(E2,),
            required_artifact_hashes=(HASH_A,),
        ),
        TrialFailedPayload(
            trial_id="trial-1",
            error_type="RuntimeError",
            error_message_hash=HASH_A,
            resumable=False,
            last_event_id=E1,
        ),
        ScenarioInstantiatedPayload(
            scenario_instance_id="scenario-1",
            scenario_type="hidden-value",
            scenario_schema_version="scenario/1.0",
            scenario_spec_hash=HASH_A,
            parameters=scenario,
            trial_seed=3,
        ),
        PrivateViewAssignedPayload(
            scenario_instance_id="scenario-1",
            view_id="view-a",
            recipient_actor_id="agent-a",
            recipient_role="seller",
            view_schema_version="view/1.0",
            view_hash=HASH_A,
            source_scenario_event_id=E1,
        ),
        InterventionScheduledPayload(
            intervention_id="int-1",
            scenario_instance_id="scenario-1",
            target_actor_id="agent-a",
            intervention_type="belief",
            scheduled_sequence_num=2,
            specification=config,
            source_event_ids=(E1,),
        ),
        AgentBuiltPayload(
            actor_id="agent-a",
            role="seller",
            entity_id="entity-a",
            model_id="model-a",
            model_revision="rev-a",
            component_names=("memory", "instructions"),
            component_config=config,
            scenario_event_id=E1,
        ),
        ObservationDeliveredPayload(
            observation_id="obs-1",
            recipient_actor_id="agent-a",
            source_actor_id="agent-b",
            source_event_id=E1,
            content_hash=HASH_A,
            visibility="private",
            sequence_in_recipient_view=0,
        ),
        ComponentContextProducedPayload(
            context_id="ctx-1",
            actor_id="agent-a",
            component_name="memory",
            component_index=0,
            model_call_id="call-1",
            input_event_ids=(E1,),
            context_hash=HASH_A,
            context_schema_version="context/1.0",
        ),
        ToMStateUpdatedPayload(
            state_id="state-1",
            actor_id="agent-a",
            counterpart_actor_id="agent-b",
            state_schema_version="tom/1.0",
            state_hash=HASH_A,
            evidence_event_ids=(E1,),
            source_model_call_id="call-1",
        ),
        ModelCallStartedPayload(
            model_call_id="call-1",
            purpose="actor_action",
            actor_id="agent-a",
            model_id="model-a",
            model_revision="rev-a",
            tokenizer_id="tok-a",
            tokenizer_revision="tok-rev",
            prompt_id="prompt-1",
            prompt_hash=HASH_A,
            input_event_ids=(E1,),
            generation_config=config,
            started_by_event_id=E2,
        ),
        ModelCallCompletedPayload(
            model_call_id="call-1",
            purpose="actor_action",
            actor_id="agent-a",
            started_event_id=E1,
            output_id="output-1",
            output_hash=HASH_A,
            token_ids_hash=HASH_B,
            generation_config_hash=config.sha256,
            usage=usage,
            finish_reason="stop",
            activation_artifact_hashes=(HASH_A,),
        ),
        ModelCallFailedPayload(
            model_call_id="call-1",
            purpose="actor_action",
            actor_id="agent-a",
            started_event_id=E1,
            generation_config_hash=config.sha256,
            error_type="TimeoutError",
            error_message_hash=HASH_A,
            retryable=True,
        ),
        ActionProposedPayload(
            action_id="action-1",
            actor_id="agent-a",
            model_call_id="call-1",
            model_call_event_id=E1,
            action_spec_id="spec-1",
            action_hash=HASH_A,
            source_observation_event_ids=(E2,),
        ),
        ActionCommittedPayload(
            action_id="action-1",
            actor_id="agent-a",
            proposed_event_id=E1,
            model_call_id="call-1",
            action_hash=HASH_A,
            protocol_decision_event_id=E2,
        ),
        TurnAdvancedPayload(
            turn_id="turn-1",
            trial_id="trial-1",
            from_actor_id="agent-a",
            to_actor_id="agent-b",
            committed_action_event_id=E1,
            next_sequence_num=3,
        ),
        BehaviorLabeledPayload(
            label_id="label-1",
            target_event_id=E1,
            target_actor_id="agent-a",
            label_name="actual-deception",
            value=LabelValue(kind="boolean", boolean_value=True),
            provenance=provenance,
        ),
        MonitorScoredPayload(
            monitor_id="monitor-1",
            monitor_version="1.0",
            target_event_id=E1,
            target_actor_id="agent-a",
            score=0.8,
            threshold=0.5,
            flagged=True,
            evidence_event_ids=(E1, E2),
        ),
        OutcomeResolvedPayload(
            outcome_id="outcome-1",
            trial_id="trial-1",
            resolver_id="rules",
            resolver_version="1.0",
            outcome=outcome,
            source_event_ids=(E1,),
            success=True,
            score=1.0,
        ),
        QualityControlAppliedPayload(
            qc_id="qc-1",
            qc_version="1.0",
            target_event_id=E1,
            passed=True,
            flags=(),
            source_event_ids=(E1,),
        ),
        BeliefIntervenedPayload(
            intervention_id="int-1",
            actor_id="agent-a",
            counterpart_actor_id="agent-b",
            source_state_event_id=E1,
            result_state_id="state-2",
            result_state_schema_version="tom/1.0",
            result_state_hash=HASH_A,
            method_id="replace-belief",
            parameters=config,
            source_model_call_id="call-1",
        ),
        ActivationIntervenedPayload(
            intervention_id="int-2",
            actor_id="agent-a",
            model_call_id="call-1",
            source_activation_event_id=E1,
            hook_name="blocks.2.hook",
            layer=2,
            direction_hash=HASH_A,
            magnitude=-1.5,
            method_id="steer",
            is_control=False,
        ),
        ProtocolDecisionAppliedPayload(
            decision_id="decision-1",
            module_name="validation",
            trial_id="trial-1",
            target_action_event_id=E1,
            decision="accept",
            reason_code="valid",
            source_event_ids=(E1,),
            modified_action_hash=None,
        ),
    )


EXAMPLES = payload_examples()
EXAMPLE_BY_TYPE = {type(payload): payload for payload in EXAMPLES}


def model_from_wire(payload_type: type[EventPayload], value: dict[str, Any]):
    return payload_type.model_validate_json(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    )


@pytest.mark.parametrize(
    "payload", EXAMPLES, ids=lambda payload: payload.EVENT_TYPE or type(payload).__name__
)
def test_every_payload_registry_and_event_envelope_round_trip(
    payload: EventPayload,
) -> None:
    registry = registered_payload_types()
    assert payload.EVENT_TYPE is not None
    assert payload.PAYLOAD_SCHEMA_VERSION == PAYLOAD_SCHEMA_VERSION
    assert registry[(payload.EVENT_TYPE, PAYLOAD_SCHEMA_VERSION)] is type(payload)
    assert parse_payload(
        payload.EVENT_TYPE, PAYLOAD_SCHEMA_VERSION, payload.to_payload_dict()
    ) == payload

    actor_id = getattr(payload, "actor_id", None)
    if actor_id is None:
        actor_id = getattr(payload, "recipient_actor_id", None)
    trial_id = getattr(payload, "trial_id", None)
    envelope = EventEnvelope(
        event_id=E3,
        event_type=payload.EVENT_TYPE,
        run_id=RUN_ID,
        pod_id="pod-payload-tests",
        trial_id=trial_id,
        dyad_id=None if trial_id is None else "dyad-1",
        sequence_num=0,
        recorded_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
        actor_id=actor_id,
        actor_role=None if actor_id is None else "actor",
        model_call_id=getattr(payload, "model_call_id", None),
        parent_event_ids=(),
        previous_event_hash=None,
        payload=payload,
        payload_schema_version=PAYLOAD_SCHEMA_VERSION,
    )
    assert EventEnvelope.from_json(envelope.to_json()) == envelope


def test_catalog_is_complete_unique_and_does_not_replace_activation_payload() -> None:
    registry = registered_payload_types()
    assert len(EXAMPLES) == len(INITIAL_PAYLOAD_TYPES) == 27
    assert {type(payload) for payload in EXAMPLES} == set(INITIAL_PAYLOAD_TYPES)
    assert len({payload.EVENT_TYPE for payload in EXAMPLES}) == 27
    assert ActivationCapturedPayload not in INITIAL_PAYLOAD_TYPES
    assert registry[("ActivationCaptured", "1.0.0")] is ActivationCapturedPayload


@pytest.mark.parametrize(
    "payload", EXAMPLES, ids=lambda payload: payload.EVENT_TYPE or type(payload).__name__
)
def test_all_required_fields_are_required_and_extra_fields_fail(
    payload: EventPayload,
) -> None:
    payload_type = type(payload)
    wire = payload.to_payload_dict()
    for field_name, field_info in payload_type.model_fields.items():
        if not field_info.is_required():
            continue
        missing = dict(wire)
        missing.pop(field_name)
        with pytest.raises(ValidationError):
            model_from_wire(payload_type, missing)

    extra = dict(wire)
    extra["unversioned_blob"] = {"arbitrary": True}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model_from_wire(payload_type, extra)


def test_canonical_document_hash_canonicality_and_alias_isolation() -> None:
    source = {"z": [1, {"ok": True}], "a": {"value": 2}}
    document = CanonicalJsonDocument.from_mapping(source, schema_version="test/1.0")
    source["z"][1]["ok"] = False
    source["a"]["value"] = 999

    assert document.canonical_json == '{"a":{"value":2},"z":[1,{"ok":true}]}'
    assert document.sha256 == hashlib.sha256(
        document.canonical_json.encode("utf-8")
    ).hexdigest()
    assert json.loads(document.canonical_json)["z"][1]["ok"] is True


def test_duplicate_keys_noncanonical_json_and_hash_mismatch_fail() -> None:
    duplicate = '{"a":1,"a":2}'
    with pytest.raises(ValidationError, match="valid strict JSON"):
        CanonicalJsonDocument(
            schema_version="test/1.0",
            canonical_json=duplicate,
            sha256=hashlib.sha256(duplicate.encode()).hexdigest(),
        )
    spaced = '{ "a": 1 }'
    with pytest.raises(ValidationError, match="not compact"):
        CanonicalJsonDocument(
            schema_version="test/1.0",
            canonical_json=spaced,
            sha256=hashlib.sha256(spaced.encode()).hexdigest(),
        )
    with pytest.raises(ValidationError, match="SHA-256 mismatch"):
        CanonicalJsonDocument(
            schema_version="test/1.0",
            canonical_json='{"a":1}',
            sha256=HASH_A,
        )


@pytest.mark.parametrize(
    "value",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {"nested": [1, {"value": -float("inf")}]},
        {"value": object()},
        {"value": b"bytes"},
        {"value": {1, 2}},
        {1: "non-string key"},
        ["not", "an", "object"],
    ],
)
def test_nonfinite_unsupported_and_nonobject_documents_fail(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        CanonicalJsonDocument.from_mapping(value, schema_version="test/1.0")


@pytest.mark.parametrize(
    "secret_key",
    [
        "api_key",
        "API-Key",
        "access_token",
        "refresh-token",
        "password",
        "private_key",
        "authorization",
        "credentials",
        "secret",
    ],
)
def test_secret_keys_are_rejected_recursively(secret_key: str) -> None:
    with pytest.raises(ValueError, match="prohibited secret-bearing field"):
        CanonicalJsonDocument.from_mapping(
            {"safe": {"nested": [{secret_key: "must-not-persist"}]}},
            schema_version="test/1.0",
        )
    assert CanonicalJsonDocument.from_mapping(
        {"max_tokens": 32, "tokenizer_id": "tok-v1"},
        schema_version="test/1.0",
    )


def test_document_size_participates_in_identity_without_aliasing() -> None:
    small = CanonicalJsonDocument.from_mapping(
        {"text": "x" * 1024}, schema_version="test/1.0"
    )
    large = CanonicalJsonDocument.from_mapping(
        {"text": "x" * 65536}, schema_version="test/1.0"
    )
    assert len(large.canonical_json) > len(small.canonical_json)
    assert large.sha256 != small.sha256


@pytest.mark.parametrize(
    ("payload_type", "field_name", "invalid"),
    [
        (RunStartedPayload, "run_id", "not-a-uuid"),
        (TrialStartedPayload, "source_scenario_event_id", "not-a-uuid"),
        (TrialStartedPayload, "trial_id", "contains spaces"),
        (RunConfigFrozenPayload, "package_lock_hash", "A" * 64),
        (PrivateViewAssignedPayload, "view_schema_version", "bad version"),
        (ModelCallStartedPayload, "prompt_hash", "short"),
        (ActionProposedPayload, "model_call_event_id", "not-a-uuid"),
        (ActionCommittedPayload, "proposed_event_id", "not-a-uuid"),
        (ObservationDeliveredPayload, "recipient_actor_id", ""),
        (ModelCallStartedPayload, "model_call_id", "call with spaces"),
    ],
)
def test_uuid_stable_id_hash_version_and_link_fields_are_strict(
    payload_type: type[EventPayload], field_name: str, invalid: Any
) -> None:
    wire = EXAMPLE_BY_TYPE[payload_type].to_payload_dict()
    wire[field_name] = invalid
    with pytest.raises(ValidationError):
        model_from_wire(payload_type, wire)


def test_reference_order_is_preserved_and_duplicates_fail() -> None:
    payload = ModelCallStartedPayload(
        **{
            **EXAMPLE_BY_TYPE[ModelCallStartedPayload].model_dump(),
            "input_event_ids": (E2, E1),
        }
    )
    assert payload.input_event_ids == (E2, E1)
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        ModelCallStartedPayload(
            **{
                **EXAMPLE_BY_TYPE[ModelCallStartedPayload].model_dump(),
                "input_event_ids": (E1, E1),
            }
        )
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        AgentBuiltPayload(
            **{
                **EXAMPLE_BY_TYPE[AgentBuiltPayload].model_dump(),
                "component_names": ("memory", "memory"),
            }
        )


def test_actor_recipient_source_and_model_call_links_are_required() -> None:
    cases = [
        (ObservationDeliveredPayload, "source_event_id"),
        (ObservationDeliveredPayload, "recipient_actor_id"),
        (ModelCallStartedPayload, "actor_id"),
        (ModelCallStartedPayload, "model_call_id"),
        (ModelCallCompletedPayload, "started_event_id"),
        (ModelCallFailedPayload, "started_event_id"),
        (ActionProposedPayload, "model_call_event_id"),
        (ActionCommittedPayload, "proposed_event_id"),
        (BeliefIntervenedPayload, "source_state_event_id"),
        (ActivationIntervenedPayload, "source_activation_event_id"),
    ]
    for payload_type, field_name in cases:
        wire = EXAMPLE_BY_TYPE[payload_type].to_payload_dict()
        wire.pop(field_name)
        with pytest.raises(ValidationError):
            model_from_wire(payload_type, wire)


def test_component_order_visibility_and_no_hidden_cot_contract() -> None:
    component = EXAMPLE_BY_TYPE[ComponentContextProducedPayload].model_dump()
    with pytest.raises(ValidationError):
        ComponentContextProducedPayload(**{**component, "component_index": -1})
    with pytest.raises(ValidationError):
        ComponentContextProducedPayload(
            **{**component, "contains_hidden_chain_of_thought": True}
        )
    observation = EXAMPLE_BY_TYPE[ObservationDeliveredPayload].model_dump()
    with pytest.raises(ValidationError):
        ObservationDeliveredPayload(**{**observation, "visibility": "secret"})


@pytest.mark.parametrize("field", ["latency_seconds", "cost_usd"])
def test_usage_is_finite_nonnegative_and_totals_match(field: str) -> None:
    with pytest.raises(ValidationError, match="total_tokens"):
        UsageRecord(input_tokens=3, output_tokens=2, total_tokens=6)
    with pytest.raises(ValidationError):
        UsageRecord(input_tokens=-1, output_tokens=1, total_tokens=0)
    with pytest.raises(ValidationError):
        UsageRecord(
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            **{field: float("inf")},
        )
    with pytest.raises(ValidationError, match="cannot exceed"):
        UsageRecord(
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            cached_input_tokens=2,
        )


def test_model_call_terminal_status_purpose_config_and_usage_contracts() -> None:
    completed = EXAMPLE_BY_TYPE[ModelCallCompletedPayload].model_dump()
    failed = EXAMPLE_BY_TYPE[ModelCallFailedPayload].model_dump()
    started = EXAMPLE_BY_TYPE[ModelCallStartedPayload].model_dump()
    with pytest.raises(ValidationError):
        ModelCallCompletedPayload(**{**completed, "status": "failed"})
    with pytest.raises(ValidationError):
        ModelCallFailedPayload(**{**failed, "status": "completed"})
    with pytest.raises(ValidationError):
        ModelCallStartedPayload(**{**started, "purpose": "hidden_reasoning"})
    with pytest.raises(ValidationError):
        ModelCallCompletedPayload(
            **{**completed, "generation_config_hash": "bad"}
        )


def test_action_proposal_commit_and_turn_linkage_contracts() -> None:
    proposed = EXAMPLE_BY_TYPE[ActionProposedPayload].model_dump()
    committed = EXAMPLE_BY_TYPE[ActionCommittedPayload].model_dump()
    turn = EXAMPLE_BY_TYPE[TurnAdvancedPayload].model_dump()
    with pytest.raises(ValidationError):
        ActionProposedPayload(**{**proposed, "status": "committed"})
    with pytest.raises(ValidationError):
        ActionCommittedPayload(**{**committed, "status": "proposed"})
    with pytest.raises(ValidationError, match="different actor"):
        TurnAdvancedPayload(**{**turn, "to_actor_id": turn["from_actor_id"]})


def test_label_value_source_and_provenance_contracts() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        LabelValue(kind="boolean", score_value=0.5)
    with pytest.raises(ValidationError, match="exactly one"):
        LabelValue(kind="score", score_value=0.5, boolean_value=True)
    with pytest.raises(ValidationError, match="fallback reason"):
        LabelProvenance(
            source="rules",
            method_id="rules",
            method_version="1.0",
            source_event_ids=(E1,),
            evaluation_succeeded=False,
        )
    with pytest.raises(ValidationError):
        LabelProvenance(
            source="self_report",
            method_id="rules",
            method_version="1.0",
            source_event_ids=(E1,),
            evaluation_succeeded=True,
        )
    payload = EXAMPLE_BY_TYPE[BehaviorLabeledPayload].model_dump()
    provenance = LabelProvenance(
        source="rules",
        method_id="rules",
        method_version="1.0",
        source_event_ids=(E2,),
        evaluation_succeeded=True,
    )
    with pytest.raises(ValidationError, match="reference the target"):
        BehaviorLabeledPayload(**{**payload, "provenance": provenance})


def test_monitor_score_threshold_and_evidence_are_consistent() -> None:
    payload = EXAMPLE_BY_TYPE[MonitorScoredPayload].model_dump()
    with pytest.raises(ValidationError, match="score >= threshold"):
        MonitorScoredPayload(**{**payload, "score": 0.4, "flagged": True})
    with pytest.raises(ValidationError, match="requires a threshold"):
        MonitorScoredPayload(**{**payload, "threshold": None, "flagged": True})
    with pytest.raises(ValidationError, match="reference the target"):
        MonitorScoredPayload(**{**payload, "evidence_event_ids": (E2,)})
    for invalid in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValidationError):
            MonitorScoredPayload(**{**payload, "score": invalid})


def test_quality_control_flags_status_and_provenance_are_consistent() -> None:
    payload = EXAMPLE_BY_TYPE[QualityControlAppliedPayload].model_dump()
    with pytest.raises(ValidationError, match="cannot have failure flags"):
        QualityControlAppliedPayload(**{**payload, "flags": ("malformed",)})
    with pytest.raises(ValidationError, match="requires at least one flag"):
        QualityControlAppliedPayload(**{**payload, "passed": False, "flags": ()})
    with pytest.raises(ValidationError, match="reference target"):
        QualityControlAppliedPayload(**{**payload, "source_event_ids": (E2,)})


def test_run_trial_outcome_and_scenario_terminal_consistency() -> None:
    run = EXAMPLE_BY_TYPE[RunCompletedPayload].model_dump()
    trial = EXAMPLE_BY_TYPE[TrialCompletedPayload].model_dump()
    scenario = EXAMPLE_BY_TYPE[ScenarioInstantiatedPayload].model_dump()
    outcome = EXAMPLE_BY_TYPE[OutcomeResolvedPayload].model_dump()
    with pytest.raises(ValidationError):
        RunCompletedPayload(**{**run, "status": "failed"})
    with pytest.raises(ValidationError):
        RunCompletedPayload(**{**run, "open_trials": 1})
    with pytest.raises(ValidationError):
        TrialCompletedPayload(**{**trial, "status": "failed"})
    mismatched = CanonicalJsonDocument.from_mapping(
        {"x": 1}, schema_version="different/1.0"
    )
    with pytest.raises(ValidationError, match="must match scenario"):
        ScenarioInstantiatedPayload(**{**scenario, "parameters": mismatched})
    with pytest.raises(ValidationError):
        OutcomeResolvedPayload(**{**outcome, "score": float("inf")})


def test_every_intervention_link_and_cross_field_contract() -> None:
    scheduled = EXAMPLE_BY_TYPE[InterventionScheduledPayload].model_dump()
    belief = EXAMPLE_BY_TYPE[BeliefIntervenedPayload].model_dump()
    activation = EXAMPLE_BY_TYPE[ActivationIntervenedPayload].model_dump()
    protocol = EXAMPLE_BY_TYPE[ProtocolDecisionAppliedPayload].model_dump()
    with pytest.raises(ValidationError):
        InterventionScheduledPayload(
            **{**scheduled, "scheduled_sequence_num": -1}
        )
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        InterventionScheduledPayload(**{**scheduled, "source_event_ids": (E1, E1)})
    with pytest.raises(ValidationError, match="differ from actor"):
        BeliefIntervenedPayload(
            **{**belief, "counterpart_actor_id": belief["actor_id"]}
        )
    with pytest.raises(ValidationError):
        ActivationIntervenedPayload(**{**activation, "layer": -1})
    with pytest.raises(ValidationError):
        ActivationIntervenedPayload(**{**activation, "magnitude": float("inf")})
    with pytest.raises(ValidationError, match="reference its target"):
        ProtocolDecisionAppliedPayload(**{**protocol, "source_event_ids": (E2,)})
    with pytest.raises(ValidationError, match="modify decision"):
        ProtocolDecisionAppliedPayload(**{**protocol, "decision": "modify"})
    with pytest.raises(ValidationError, match="modify decision"):
        ProtocolDecisionAppliedPayload(
            **{**protocol, "decision": "accept", "modified_action_hash": HASH_A}
        )


@pytest.mark.parametrize(
    ("payload_type", "raw_field"),
    [
        (PrivateViewAssignedPayload, "private_facts"),
        (ModelCallStartedPayload, "prompt_text"),
        (ModelCallStartedPayload, "api_key"),
        (ModelCallCompletedPayload, "output_text"),
        (ComponentContextProducedPayload, "chain_of_thought"),
        (ToMStateUpdatedPayload, "hidden_reasoning"),
    ],
)
def test_raw_private_prompt_output_and_cot_fields_are_not_accepted(
    payload_type: type[EventPayload], raw_field: str
) -> None:
    wire = EXAMPLE_BY_TYPE[payload_type].to_payload_dict()
    wire[raw_field] = "must not be persisted"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model_from_wire(payload_type, wire)


def test_unknown_opaque_payload_behavior_remains_unchanged() -> None:
    opaque = OpaqueEventPayload.from_payload_dict(
        "FutureCatalogEvent", "2.0.0", {"future": [1, 2]}
    )
    envelope = EventEnvelope(
        event_id=E3,
        event_type="FutureCatalogEvent",
        run_id=RUN_ID,
        pod_id="pod-payload-tests",
        trial_id=None,
        dyad_id=None,
        sequence_num=0,
        recorded_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
        actor_id=None,
        actor_role=None,
        model_call_id=None,
        parent_event_ids=(),
        previous_event_hash=None,
        payload=opaque,
        payload_schema_version="2.0.0",
    )
    with pytest.raises((UnknownPayloadTypeError, ValidationError)):
        EventEnvelope.from_json(envelope.to_json())
    preserved = EventEnvelope.from_json(
        envelope.to_json(), preserve_unknown_payloads=True
    )
    assert isinstance(preserved.payload, OpaqueEventPayload)
    assert preserved.to_json() == envelope.to_json()

