"""Permanent deterministic projection and adversarial lineage tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pytest
from pydantic import ValidationError

from interpretability.events.payloads import (
    ActionCommittedPayload,
    ActionProposedPayload,
    ActivationIntervenedPayload,
    AgentBuiltPayload,
    BehaviorLabeledPayload,
    BeliefIntervenedPayload,
    CanonicalJsonDocument,
    InterventionScheduledPayload,
    LabelProvenance,
    LabelValue,
    ModelCallCompletedPayload,
    ModelCallStartedPayload,
    MonitorScoredPayload,
    ObservationDeliveredPayload,
    OutcomeResolvedPayload,
    PrivateViewAssignedPayload,
    ProtocolDecisionAppliedPayload,
    QualityControlAppliedPayload,
    RunStartedPayload,
    ScenarioInstantiatedPayload,
    ToMStateUpdatedPayload,
    TrialCompletedPayload,
    TrialFailedPayload,
    TrialStartedPayload,
    TurnAdvancedPayload,
    UsageRecord,
)
from interpretability.events.projectors import (
    ActivationSampleProjector,
    AgentViewProjector,
    DyadProjector,
    MetricInputProjector,
    ProjectionIdentityError,
    ProjectionLifecycleError,
    ProjectionLinkError,
    ProjectionSequenceError,
    TranscriptProjector,
    TrialStateProjector,
    UnprojectableEventError,
)
from interpretability.events.reader import EventReader
from interpretability.events.schema import (
    ActivationCapturedPayload,
    ArtifactReference,
    EventEnvelope,
    EventPayload,
    OpaqueEventPayload,
)
from interpretability.events.writer import EventWriter

RUN_ID = "70000000-0000-4000-8000-000000000001"
OTHER_RUN_ID = "80000000-0000-4000-8000-000000000001"
POD_ID = "pod-projectors"
TRIAL_ID = "trial-projectors"
DYAD_ID = "dyad-projectors"
ALICE = "alice"
BOB = "bob"
SELLER = "seller"
BUYER = "buyer"
CALL_ALICE = "call-alice-action"
CALL_BOB = "call-bob-action"
RECORDED_AT = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def event_id(index: int) -> str:
    return f"90000000-0000-4000-8000-{index:012x}"


def document(value: dict[str, Any], version: str = "1.0.0") -> CanonicalJsonDocument:
    return CanonicalJsonDocument.from_mapping(value, schema_version=version)


@dataclass(frozen=True)
class FullTrial:
    events: tuple[EventEnvelope, ...]
    named: dict[str, EventEnvelope]


class StreamBuilder:
    """Build canonical multi-lane envelopes with deterministic identities."""

    def __init__(self, *, recorded_at: datetime = RECORDED_AT) -> None:
        self.events: list[EventEnvelope] = []
        self.named: dict[str, EventEnvelope] = {}
        self.recorded_at = recorded_at

    def add(
        self,
        name: str,
        event_type: str,
        payload: EventPayload,
        *,
        trial_id: str | None = TRIAL_ID,
        dyad_id: str | None = DYAD_ID,
        actor_id: str | None = None,
        actor_role: str | None = None,
        model_call_id: str | None = None,
        parent_event_ids: Iterable[str] = (),
    ) -> EventEnvelope:
        lane = [event for event in self.events if event.trial_id == trial_id]
        parents = tuple(dict.fromkeys(parent_event_ids))
        envelope = EventEnvelope(
            event_id=event_id(len(self.events) + 1),
            event_type=event_type,
            run_id=RUN_ID,
            pod_id=POD_ID,
            trial_id=trial_id,
            dyad_id=dyad_id if trial_id is not None else None,
            sequence_num=len(lane),
            recorded_at=self.recorded_at + timedelta(seconds=len(self.events)),
            actor_id=actor_id,
            actor_role=actor_role,
            model_call_id=model_call_id,
            parent_event_ids=parents,
            previous_event_hash=None if not lane else lane[-1].content_hash,
            payload=payload,
            payload_schema_version=payload.PAYLOAD_SCHEMA_VERSION or "1.0.0",
        )
        self.events.append(envelope)
        self.named[name] = envelope
        return envelope


def artifact(
    artifact_hash: str,
    *,
    layer: int,
    call_id: str,
    tokenizer_id: str,
    model_revision: str,
) -> ArtifactReference:
    return ArtifactReference(
        artifact_hash=artifact_hash,
        hook_name=f"blocks.{layer}.hook_resid_post",
        layer=layer,
        token_selection="generated_last",
        aggregation="none",
        shape=(1, 8),
        dtype="float32",
        tokenizer_id=tokenizer_id,
        model_revision=model_revision,
        source_model_call_id=call_id,
    )


def build_full_trial(*, first_action_hash: str | None = None) -> FullTrial:
    builder = StreamBuilder()
    scenario_hash = digest("scenario")
    action_a_hash = first_action_hash or digest("alice-action")
    action_b_hash = digest("bob-action")
    activation_a2 = digest("alice-layer-2")
    activation_a5 = digest("alice-layer-5")
    activation_b3 = digest("bob-layer-3")

    run_started = builder.add(
        "run_started",
        "RunStarted",
        RunStartedPayload(
            run_id=RUN_ID,
            orchestrator_id="projector-tests",
            code_revision="revision-1",
            run_seed=7,
        ),
        trial_id=None,
        dyad_id=None,
    )
    scenario = builder.add(
        "scenario",
        "ScenarioInstantiated",
        ScenarioInstantiatedPayload(
            scenario_instance_id="scenario-instance",
            scenario_type="hidden_value",
            scenario_schema_version="1.0.0",
            scenario_spec_hash=scenario_hash,
            parameters=document({"reserve_price": 80}),
            trial_seed=11,
        ),
        trial_id=None,
        dyad_id=None,
        parent_event_ids=(run_started.event_id,),
    )
    started = builder.add(
        "trial_started",
        "TrialStarted",
        TrialStartedPayload(
            trial_id=TRIAL_ID,
            scenario_instance_id="scenario-instance",
            dyad_id=DYAD_ID,
            attempt=1,
            actor_ids=(ALICE, BOB),
            source_scenario_event_id=scenario.event_id,
        ),
        parent_event_ids=(scenario.event_id,),
    )
    private_a = builder.add(
        "private_a",
        "PrivateViewAssigned",
        PrivateViewAssignedPayload(
            scenario_instance_id="scenario-instance",
            view_id="view-alice",
            recipient_actor_id=ALICE,
            recipient_role=SELLER,
            view_schema_version="1.0.0",
            view_hash=digest("alice-private-view"),
            source_scenario_event_id=scenario.event_id,
        ),
        actor_id=ALICE,
        actor_role=SELLER,
        parent_event_ids=(started.event_id, scenario.event_id),
    )
    private_b = builder.add(
        "private_b",
        "PrivateViewAssigned",
        PrivateViewAssignedPayload(
            scenario_instance_id="scenario-instance",
            view_id="view-bob",
            recipient_actor_id=BOB,
            recipient_role=BUYER,
            view_schema_version="1.0.0",
            view_hash=digest("bob-private-view"),
            source_scenario_event_id=scenario.event_id,
        ),
        actor_id=BOB,
        actor_role=BUYER,
        parent_event_ids=(private_a.event_id, scenario.event_id),
    )
    agent_a = builder.add(
        "agent_a",
        "AgentBuilt",
        AgentBuiltPayload(
            actor_id=ALICE,
            role=SELLER,
            entity_id="entity-alice",
            model_id="model-a",
            model_revision="model-a-rev",
            component_names=("observation", "strategy", "reasoning"),
            component_config=document({"module_order": ["observation", "strategy"]}),
            scenario_event_id=scenario.event_id,
        ),
        actor_id=ALICE,
        actor_role=SELLER,
        parent_event_ids=(private_b.event_id, scenario.event_id),
    )
    agent_b = builder.add(
        "agent_b",
        "AgentBuilt",
        AgentBuiltPayload(
            actor_id=BOB,
            role=BUYER,
            entity_id="entity-bob",
            model_id="model-b",
            model_revision="model-b-rev",
            component_names=("observation", "strategy"),
            component_config=document({"module_order": ["observation", "strategy"]}),
            scenario_event_id=scenario.event_id,
        ),
        actor_id=BOB,
        actor_role=BUYER,
        parent_event_ids=(agent_a.event_id, scenario.event_id),
    )
    initial_a = builder.add(
        "initial_a",
        "ObservationDelivered",
        ObservationDeliveredPayload(
            observation_id="observation-alice-private",
            recipient_actor_id=ALICE,
            source_actor_id=None,
            source_event_id=private_a.event_id,
            content_hash=digest("alice-initial-observation"),
            visibility="private",
            sequence_in_recipient_view=0,
        ),
        actor_id=ALICE,
        actor_role=SELLER,
        parent_event_ids=(agent_b.event_id, private_a.event_id),
    )
    initial_b = builder.add(
        "initial_b",
        "ObservationDelivered",
        ObservationDeliveredPayload(
            observation_id="observation-bob-private",
            recipient_actor_id=BOB,
            source_actor_id=None,
            source_event_id=private_b.event_id,
            content_hash=digest("bob-initial-observation"),
            visibility="private",
            sequence_in_recipient_view=0,
        ),
        actor_id=BOB,
        actor_role=BUYER,
        parent_event_ids=(initial_a.event_id, private_b.event_id),
    )

    generation_a = document({"temperature": 0.3, "top_p": 0.9, "seed": 101})
    call_a_started = builder.add(
        "call_a_started",
        "ModelCallStarted",
        ModelCallStartedPayload(
            model_call_id=CALL_ALICE,
            purpose="actor_action",
            actor_id=ALICE,
            model_id="model-a",
            model_revision="model-a-rev",
            tokenizer_id="tokenizer-a",
            tokenizer_revision="tokenizer-a-rev",
            prompt_id="prompt-alice-1",
            prompt_hash=digest("prompt-alice-1"),
            input_event_ids=(initial_a.event_id,),
            generation_config=generation_a,
        ),
        actor_id=ALICE,
        actor_role=SELLER,
        model_call_id=CALL_ALICE,
        parent_event_ids=(initial_a.event_id,),
    )
    call_a_completed = builder.add(
        "call_a_completed",
        "ModelCallCompleted",
        ModelCallCompletedPayload(
            model_call_id=CALL_ALICE,
            purpose="actor_action",
            actor_id=ALICE,
            started_event_id=call_a_started.event_id,
            output_id="output-alice-1",
            output_hash=action_a_hash,
            token_ids_hash=digest("tokens-alice-1"),
            generation_config_hash=generation_a.sha256,
            usage=UsageRecord(input_tokens=10, output_tokens=4, total_tokens=14),
            finish_reason="stop",
            activation_artifact_hashes=(activation_a2, activation_a5),
        ),
        actor_id=ALICE,
        actor_role=SELLER,
        model_call_id=CALL_ALICE,
        parent_event_ids=(call_a_started.event_id,),
    )
    capture_a2 = builder.add(
        "capture_a2",
        "ActivationCaptured",
        ActivationCapturedPayload(
            artifact=artifact(
                activation_a2,
                layer=2,
                call_id=CALL_ALICE,
                tokenizer_id="tokenizer-a",
                model_revision="model-a-rev",
            )
        ),
        actor_id=ALICE,
        actor_role=SELLER,
        model_call_id=CALL_ALICE,
        parent_event_ids=(call_a_completed.event_id,),
    )
    capture_a5 = builder.add(
        "capture_a5",
        "ActivationCaptured",
        ActivationCapturedPayload(
            artifact=artifact(
                activation_a5,
                layer=5,
                call_id=CALL_ALICE,
                tokenizer_id="tokenizer-a",
                model_revision="model-a-rev",
            )
        ),
        actor_id=ALICE,
        actor_role=SELLER,
        model_call_id=CALL_ALICE,
        parent_event_ids=(capture_a2.event_id,),
    )
    proposed_a = builder.add(
        "proposed_a",
        "ActionProposed",
        ActionProposedPayload(
            action_id="action-alice-1",
            actor_id=ALICE,
            model_call_id=CALL_ALICE,
            model_call_event_id=call_a_completed.event_id,
            action_spec_id="negotiation-action",
            action_hash=action_a_hash,
            source_observation_event_ids=(initial_a.event_id,),
        ),
        actor_id=ALICE,
        actor_role=SELLER,
        model_call_id=CALL_ALICE,
        parent_event_ids=(capture_a5.event_id, call_a_completed.event_id),
    )
    committed_a = builder.add(
        "committed_a",
        "ActionCommitted",
        ActionCommittedPayload(
            action_id="action-alice-1",
            actor_id=ALICE,
            proposed_event_id=proposed_a.event_id,
            model_call_id=CALL_ALICE,
            action_hash=action_a_hash,
        ),
        actor_id=ALICE,
        actor_role=SELLER,
        model_call_id=CALL_ALICE,
        parent_event_ids=(proposed_a.event_id,),
    )
    turn_a = builder.add(
        "turn_a",
        "TurnAdvanced",
        TurnAdvancedPayload(
            turn_id="turn-alice-1",
            trial_id=TRIAL_ID,
            from_actor_id=ALICE,
            to_actor_id=BOB,
            committed_action_event_id=committed_a.event_id,
            next_sequence_num=1,
        ),
        actor_id=ALICE,
        actor_role=SELLER,
        parent_event_ids=(committed_a.event_id,),
    )
    reception_b = builder.add(
        "reception_b",
        "ObservationDelivered",
        ObservationDeliveredPayload(
            observation_id="observation-bob-from-alice",
            recipient_actor_id=BOB,
            source_actor_id=ALICE,
            source_event_id=committed_a.event_id,
            content_hash=action_a_hash,
            visibility="public",
            sequence_in_recipient_view=1,
        ),
        actor_id=BOB,
        actor_role=BUYER,
        parent_event_ids=(turn_a.event_id, committed_a.event_id),
    )
    tom_b = builder.add(
        "tom_b",
        "ToMStateUpdated",
        ToMStateUpdatedPayload(
            state_id="tom-bob-after-alice",
            actor_id=BOB,
            counterpart_actor_id=ALICE,
            state_schema_version="1.0.0",
            state_hash=digest("tom-bob-state"),
            evidence_event_ids=(reception_b.event_id,),
            source_model_call_id="call-bob-tom",
        ),
        actor_id=BOB,
        actor_role=BUYER,
        parent_event_ids=(reception_b.event_id,),
    )
    belief_b = builder.add(
        "belief_b",
        "BeliefIntervened",
        BeliefIntervenedPayload(
            intervention_id="belief-intervention-bob",
            actor_id=BOB,
            counterpart_actor_id=ALICE,
            source_state_event_id=tom_b.event_id,
            result_state_id="tom-bob-intervened",
            result_state_schema_version="1.0.0",
            result_state_hash=digest("tom-bob-intervened"),
            method_id="belief-shift",
            parameters=document({"trust_delta": -0.2}),
        ),
        actor_id=BOB,
        actor_role=BUYER,
        parent_event_ids=(tom_b.event_id,),
    )
    scheduled = builder.add(
        "scheduled",
        "InterventionScheduled",
        InterventionScheduledPayload(
            intervention_id="schedule-bob-steering",
            scenario_instance_id="scenario-instance",
            target_actor_id=BOB,
            intervention_type="activation-steering",
            scheduled_sequence_num=20,
            specification=document({"layer": 3, "magnitude": 0.5}),
            source_event_ids=(belief_b.event_id,),
        ),
        parent_event_ids=(belief_b.event_id,),
    )
    activation_intervention = builder.add(
        "activation_intervention",
        "ActivationIntervened",
        ActivationIntervenedPayload(
            intervention_id="activation-intervention-bob",
            actor_id=BOB,
            model_call_id=CALL_BOB,
            source_activation_event_id=capture_a2.event_id,
            hook_name="blocks.2.hook_resid_post",
            layer=2,
            direction_hash=digest("steering-direction"),
            magnitude=0.5,
            method_id="residual-steering",
            is_control=False,
        ),
        actor_id=BOB,
        actor_role=BUYER,
        model_call_id=CALL_BOB,
        parent_event_ids=(scheduled.event_id, capture_a2.event_id),
    )

    generation_b = document({"temperature": 0.2, "top_p": 0.8, "seed": 202})
    call_b_started = builder.add(
        "call_b_started",
        "ModelCallStarted",
        ModelCallStartedPayload(
            model_call_id=CALL_BOB,
            purpose="counterpart_action",
            actor_id=BOB,
            model_id="model-b",
            model_revision="model-b-rev",
            tokenizer_id="tokenizer-b",
            tokenizer_revision="tokenizer-b-rev",
            prompt_id="prompt-bob-1",
            prompt_hash=digest("prompt-bob-1"),
            input_event_ids=(reception_b.event_id,),
            generation_config=generation_b,
            started_by_event_id=activation_intervention.event_id,
        ),
        actor_id=BOB,
        actor_role=BUYER,
        model_call_id=CALL_BOB,
        parent_event_ids=(activation_intervention.event_id, reception_b.event_id),
    )
    call_b_completed = builder.add(
        "call_b_completed",
        "ModelCallCompleted",
        ModelCallCompletedPayload(
            model_call_id=CALL_BOB,
            purpose="counterpart_action",
            actor_id=BOB,
            started_event_id=call_b_started.event_id,
            output_id="output-bob-1",
            output_hash=action_b_hash,
            token_ids_hash=digest("tokens-bob-1"),
            generation_config_hash=generation_b.sha256,
            usage=UsageRecord(input_tokens=11, output_tokens=3, total_tokens=14),
            finish_reason="stop",
            activation_artifact_hashes=(activation_b3,),
        ),
        actor_id=BOB,
        actor_role=BUYER,
        model_call_id=CALL_BOB,
        parent_event_ids=(call_b_started.event_id,),
    )
    capture_b3 = builder.add(
        "capture_b3",
        "ActivationCaptured",
        ActivationCapturedPayload(
            artifact=artifact(
                activation_b3,
                layer=3,
                call_id=CALL_BOB,
                tokenizer_id="tokenizer-b",
                model_revision="model-b-rev",
            )
        ),
        actor_id=BOB,
        actor_role=BUYER,
        model_call_id=CALL_BOB,
        parent_event_ids=(call_b_completed.event_id,),
    )
    proposed_b = builder.add(
        "proposed_b",
        "ActionProposed",
        ActionProposedPayload(
            action_id="action-bob-1",
            actor_id=BOB,
            model_call_id=CALL_BOB,
            model_call_event_id=call_b_completed.event_id,
            action_spec_id="negotiation-action",
            action_hash=action_b_hash,
            source_observation_event_ids=(reception_b.event_id,),
        ),
        actor_id=BOB,
        actor_role=BUYER,
        model_call_id=CALL_BOB,
        parent_event_ids=(capture_b3.event_id, call_b_completed.event_id),
    )
    protocol = builder.add(
        "protocol",
        "ProtocolDecisionApplied",
        ProtocolDecisionAppliedPayload(
            decision_id="protocol-accept-bob",
            module_name="negotiation-protocol",
            trial_id=TRIAL_ID,
            target_action_event_id=proposed_b.event_id,
            decision="accept",
            reason_code="valid-offer",
            source_event_ids=(proposed_b.event_id,),
        ),
        parent_event_ids=(proposed_b.event_id,),
    )
    committed_b = builder.add(
        "committed_b",
        "ActionCommitted",
        ActionCommittedPayload(
            action_id="action-bob-1",
            actor_id=BOB,
            proposed_event_id=proposed_b.event_id,
            model_call_id=CALL_BOB,
            action_hash=action_b_hash,
            protocol_decision_event_id=protocol.event_id,
        ),
        actor_id=BOB,
        actor_role=BUYER,
        model_call_id=CALL_BOB,
        parent_event_ids=(protocol.event_id, proposed_b.event_id),
    )
    turn_b = builder.add(
        "turn_b",
        "TurnAdvanced",
        TurnAdvancedPayload(
            turn_id="turn-bob-1",
            trial_id=TRIAL_ID,
            from_actor_id=BOB,
            to_actor_id=ALICE,
            committed_action_event_id=committed_b.event_id,
            next_sequence_num=2,
        ),
        actor_id=BOB,
        actor_role=BUYER,
        parent_event_ids=(committed_b.event_id,),
    )
    reception_a = builder.add(
        "reception_a",
        "ObservationDelivered",
        ObservationDeliveredPayload(
            observation_id="observation-alice-from-bob",
            recipient_actor_id=ALICE,
            source_actor_id=BOB,
            source_event_id=committed_b.event_id,
            content_hash=action_b_hash,
            visibility="public",
            sequence_in_recipient_view=1,
        ),
        actor_id=ALICE,
        actor_role=SELLER,
        parent_event_ids=(turn_b.event_id, committed_b.event_id),
    )

    label_rules = builder.add(
        "label_rules",
        "BehaviorLabeled",
        BehaviorLabeledPayload(
            label_id="label-alice-rules",
            target_event_id=committed_a.event_id,
            target_actor_id=ALICE,
            label_name="actual_deception",
            value=LabelValue(kind="boolean", boolean_value=True),
            provenance=LabelProvenance(
                source="rules",
                method_id="hidden-value-rules",
                method_version="1.0.0",
                source_event_ids=(committed_a.event_id,),
                evaluation_succeeded=True,
            ),
        ),
        parent_event_ids=(reception_a.event_id, committed_a.event_id),
    )
    label_judge = builder.add(
        "label_judge",
        "BehaviorLabeled",
        BehaviorLabeledPayload(
            label_id="label-alice-judge",
            target_event_id=committed_a.event_id,
            target_actor_id=ALICE,
            label_name="actual_deception",
            value=LabelValue(kind="score", score_value=0.8),
            provenance=LabelProvenance(
                source="judge",
                method_id="judge-v1",
                method_version="1.0.0",
                source_event_ids=(committed_a.event_id,),
                evaluation_succeeded=True,
            ),
        ),
        parent_event_ids=(label_rules.event_id, committed_a.event_id),
    )
    label_b = builder.add(
        "label_b",
        "BehaviorLabeled",
        BehaviorLabeledPayload(
            label_id="label-bob-rules",
            target_event_id=committed_b.event_id,
            target_actor_id=BOB,
            label_name="actual_deception",
            value=LabelValue(kind="boolean", boolean_value=False),
            provenance=LabelProvenance(
                source="rules",
                method_id="hidden-value-rules",
                method_version="1.0.0",
                source_event_ids=(committed_b.event_id,),
                evaluation_succeeded=True,
            ),
        ),
        parent_event_ids=(label_judge.event_id, committed_b.event_id),
    )
    monitor = builder.add(
        "monitor",
        "MonitorScored",
        MonitorScoredPayload(
            monitor_id="deception-monitor",
            monitor_version="1.0.0",
            target_event_id=committed_a.event_id,
            target_actor_id=ALICE,
            score=0.75,
            threshold=0.5,
            flagged=True,
            evidence_event_ids=(committed_a.event_id,),
        ),
        parent_event_ids=(label_b.event_id, committed_a.event_id),
    )
    qc = builder.add(
        "qc",
        "QualityControlApplied",
        QualityControlAppliedPayload(
            qc_id="qc-alice-action",
            qc_version="1.0.0",
            target_event_id=committed_a.event_id,
            passed=True,
            flags=(),
            source_event_ids=(committed_a.event_id,),
        ),
        parent_event_ids=(monitor.event_id, committed_a.event_id),
    )
    outcome = builder.add(
        "outcome",
        "OutcomeResolved",
        OutcomeResolvedPayload(
            outcome_id="outcome-main",
            trial_id=TRIAL_ID,
            resolver_id="negotiation-outcome",
            resolver_version="1.0.0",
            outcome=document({"agreement": True, "price": 95}),
            source_event_ids=(committed_a.event_id, committed_b.event_id),
            success=True,
            score=0.9,
        ),
        parent_event_ids=(qc.event_id, committed_a.event_id, committed_b.event_id),
    )
    builder.add(
        "trial_completed",
        "TrialCompleted",
        TrialCompletedPayload(
            trial_id=TRIAL_ID,
            outcome_event_id=outcome.event_id,
            terminal_action_event_ids=(committed_a.event_id, committed_b.event_id),
            required_artifact_hashes=(activation_a2, activation_a5, activation_b3),
        ),
        parent_event_ids=(outcome.event_id, capture_a2.event_id, capture_a5.event_id, capture_b3.event_id),
    )
    return FullTrial(tuple(builder.events), builder.named)


@pytest.fixture(scope="module")
def full_trial() -> FullTrial:
    return build_full_trial()


def clone_event(event: EventEnvelope, **updates: Any) -> EventEnvelope:
    values = {
        "schema_version": event.schema_version,
        "event_id": event.event_id,
        "event_type": event.event_type,
        "run_id": event.run_id,
        "pod_id": event.pod_id,
        "trial_id": event.trial_id,
        "dyad_id": event.dyad_id,
        "sequence_num": event.sequence_num,
        "recorded_at": event.recorded_at,
        "actor_id": event.actor_id,
        "actor_role": event.actor_role,
        "model_call_id": event.model_call_id,
        "parent_event_ids": event.parent_event_ids,
        "previous_event_hash": event.previous_event_hash,
        "payload": event.payload,
        "payload_schema_version": event.payload_schema_version,
        "content_hash": None,
    }
    values.update(updates)
    return EventEnvelope(**values)


def normalize(events: Iterable[EventEnvelope]) -> tuple[EventEnvelope, ...]:
    """Rebuild order/hash lanes after deleting or inserting attack events."""

    source = tuple(events)
    retained_ids = {event.event_id for event in source}
    lanes: dict[str | None, list[EventEnvelope]] = {}
    result: list[EventEnvelope] = []
    for event in source:
        lane = lanes.setdefault(event.trial_id, [])
        rebuilt = clone_event(
            event,
            sequence_num=len(lane),
            previous_event_hash=None if not lane else lane[-1].content_hash,
            parent_event_ids=tuple(
                parent for parent in event.parent_event_ids if parent in retained_ids
            ),
        )
        lane.append(rebuilt)
        result.append(rebuilt)
    return tuple(result)


def replace_named(
    trial: FullTrial, name: str, replacement: EventEnvelope
) -> tuple[EventEnvelope, ...]:
    return normalize(
        replacement if event.event_id == trial.named[name].event_id else event
        for event in trial.events
    )


def without_named(trial: FullTrial, *names: str) -> tuple[EventEnvelope, ...]:
    removed = {trial.named[name].event_id for name in names}
    return normalize(event for event in trial.events if event.event_id not in removed)


def insert_after(
    trial: FullTrial,
    name: str,
    inserted: EventEnvelope,
) -> tuple[EventEnvelope, ...]:
    result: list[EventEnvelope] = []
    target_id = trial.named[name].event_id
    for event in trial.events:
        result.append(event)
        if event.event_id == target_id:
            result.append(inserted)
    return normalize(result)


def test_full_trial_round_trips_through_real_writer_and_reader(
    tmp_path: Path, full_trial: FullTrial
) -> None:
    path = tmp_path / "events.jsonl"
    with EventWriter(path, run_id=RUN_ID, pod_id=POD_ID, fsync_mode="never") as writer:
        receipts = [writer.append(event) for event in full_trial.events]

    reader = EventReader(path)
    located = tuple(reader.iter_events())
    report = reader.validate(require_sealed_trials=True)

    assert tuple(item.event for item in located) == full_trial.events
    assert [receipt.line_number for receipt in receipts] == list(
        range(1, len(full_trial.events) + 1)
    )
    assert report.completed_trials == (TRIAL_ID,)
    assert report.projectable


def test_transcript_projects_only_committed_actions_turns_roles_and_annotations(
    full_trial: FullTrial,
) -> None:
    projection = TranscriptProjector().project(full_trial.events)

    assert projection.run_id == RUN_ID
    assert projection.trial_id == TRIAL_ID
    assert [action.action_id for action in projection.actions] == [
        "action-alice-1",
        "action-bob-1",
    ]
    first, second = projection.actions
    assert (first.actor_id, first.actor_role, first.next_actor_id) == (
        ALICE,
        SELLER,
        BOB,
    )
    assert (first.turn_id, first.model_call_id) == ("turn-alice-1", CALL_ALICE)
    assert first.proposed_event_id == full_trial.named["proposed_a"].event_id
    assert first.committed_event_id == full_trial.named["committed_a"].event_id
    assert first.turn_event_id == full_trial.named["turn_a"].event_id
    assert first.action_text is None
    assert {warning.field_name for warning in first.warnings} == {"action_text"}
    assert (second.actor_id, second.actor_role, second.next_actor_role) == (
        BOB,
        BUYER,
        SELLER,
    )
    assert second.protocol_decision_event_id == full_trial.named["protocol"].event_id
    assert len(projection.labels) == 3
    assert [label.label_name for label in projection.labels].count(
        "actual_deception"
    ) == 3
    assert full_trial.named["label_rules"].event_id in first.annotation_event_ids
    assert full_trial.named["label_judge"].event_id in first.annotation_event_ids
    assert full_trial.named["monitor"].event_id in first.annotation_event_ids
    assert full_trial.named["qc"].event_id in first.annotation_event_ids


def test_agent_view_filters_recipient_and_supports_sequence_cutoff(
    full_trial: FullTrial,
) -> None:
    complete = AgentViewProjector(BOB).project(full_trial.events)
    cutoff = full_trial.named["initial_b"].sequence_num
    early = AgentViewProjector(BOB, through_sequence_num=cutoff).project(
        full_trial.events
    )
    override = AgentViewProjector(BOB).project(
        full_trial.events, through_sequence_num=cutoff
    )

    assert [view.view_id for view in complete.private_views] == ["view-bob"]
    assert [item.observation_id for item in complete.observations] == [
        "observation-bob-private",
        "observation-bob-from-alice",
    ]
    assert [item.sequence_in_recipient_view for item in complete.observations] == [0, 1]
    assert all(item.recipient_actor_id == BOB for item in complete.observations)
    assert all(item.observation_content is None for item in complete.observations)
    assert len(early.observations) == 1
    assert early == override
    assert early.through_sequence_num == cutoff
    assert all(event.sequence_num <= cutoff for event in early.observations)


def test_trial_state_rebuilds_lifecycle_turn_interventions_and_annotations(
    full_trial: FullTrial,
) -> None:
    state = TrialStateProjector().project(full_trial.events)

    assert state.lifecycle_status == "completed"
    assert state.started_event_id == full_trial.named["trial_started"].event_id
    assert state.terminal_event_id == full_trial.named["trial_completed"].event_id
    assert state.current_turn is not None
    assert (state.current_turn.actor_id, state.current_turn.actor_role) == (
        ALICE,
        SELLER,
    )
    assert len(state.commitments) == 2
    assert {fact.event_type for fact in state.interventions} == {
        "BeliefIntervened",
        "InterventionScheduled",
        "ActivationIntervened",
        "ProtocolDecisionApplied",
    }
    assert len(state.labels) == 3
    assert len(state.monitor_scores) == 1
    assert len(state.quality_controls) == 1
    assert len(state.outcomes) == 1
    assert state.outcomes[0].success is True
    assert state.outcomes[0].outcome_json == '{"agreement":true,"price":95}'


def test_activation_samples_preserve_exact_multilayer_lineage_and_missingness(
    full_trial: FullTrial,
) -> None:
    projection = ActivationSampleProjector().project(full_trial.events)

    assert len(projection.samples) == 2
    alice, bob = projection.samples
    assert alice.actor_id == ALICE
    assert alice.call_purpose == "actor_action"
    assert alice.model_call_id == CALL_ALICE
    assert alice.model_call_started_event_id == full_trial.named["call_a_started"].event_id
    assert alice.model_call_completed_event_id == full_trial.named["call_a_completed"].event_id
    assert alice.action_proposed_event_id == full_trial.named["proposed_a"].event_id
    assert alice.action_committed_event_id == full_trial.named["committed_a"].event_id
    assert alice.turn_event_id == full_trial.named["turn_a"].event_id
    assert [reference.layer for reference in alice.activation_artifacts] == [2, 5]
    assert alice.activation_event_ids == (
        full_trial.named["capture_a2"].event_id,
        full_trial.named["capture_a5"].event_id,
    )
    assert bob.call_purpose == "counterpart_action"
    assert [reference.layer for reference in bob.activation_artifacts] == [3]
    assert alice.sample_type == "negotiation"
    assert (alice.round_num, alice.prompt, alice.response, alice.activation_values) == (
        None,
        None,
        None,
        None,
    )
    assert {warning.field_name for warning in alice.warnings} == {
        "round_num",
        "prompt",
        "response",
        "activation_values",
    }
    assert {warning.code for warning in alice.warnings} == {
        "legacy_field_unavailable",
        "content_unavailable",
        "artifact_not_loaded",
    }
    assert full_trial.named["label_rules"].event_id in alice.annotation_event_ids
    assert full_trial.named["label_judge"].event_id in alice.annotation_event_ids


def test_dyad_pairs_send_reception_response_and_stable_roles(
    full_trial: FullTrial,
) -> None:
    projection = DyadProjector().project(full_trial.events)

    assert projection.dyad_id == DYAD_ID
    assert projection.actor_ids == (ALICE, BOB)
    assert projection.actor_roles == ((ALICE, SELLER), (BOB, BUYER))
    assert len(projection.turns) == 2
    sent, response = projection.turns
    assert (sent.actor_id, sent.counterpart_actor_id) == (ALICE, BOB)
    assert (sent.actor_role, sent.counterpart_role) == (SELLER, BUYER)
    assert sent.reception_event_ids == (full_trial.named["reception_b"].event_id,)
    assert sent.response_action_event_id == full_trial.named["committed_b"].event_id
    assert sent.response_model_call_id == CALL_BOB
    assert response.reception_event_ids == (full_trial.named["reception_a"].event_id,)
    assert response.response_action_event_id is None


def test_metric_inputs_preserve_duplicate_label_sources_without_aggregation(
    full_trial: FullTrial,
) -> None:
    projection = MetricInputProjector().project(full_trial.events)

    alice_labels = [
        label for label in projection.labels if label.target_actor_id == ALICE
    ]
    assert len(alice_labels) == 2
    assert [label.provenance.source for label in alice_labels] == ["rules", "judge"]
    assert [label.value.kind for label in alice_labels] == ["boolean", "score"]
    assert len(projection.monitor_scores) == 1
    assert projection.monitor_scores[0].score == 0.75
    assert len(projection.outcomes) == 1
    assert projection.outcomes[0].score == 0.9
    assert set(projection.semantic_dict()) == {
        "run_id",
        "pod_id",
        "trial_id",
        "dyad_id",
        "labels",
        "monitor_scores",
        "outcomes",
    }


PROJECTOR_FACTORIES: tuple[Callable[[], Any], ...] = (
    TranscriptProjector,
    lambda: AgentViewProjector(ALICE),
    TrialStateProjector,
    ActivationSampleProjector,
    DyadProjector,
    MetricInputProjector,
)


@pytest.mark.parametrize("factory", PROJECTOR_FACTORIES)
def test_projectors_are_generator_safe_idempotent_and_resettable(
    factory: Callable[[], Any], full_trial: FullTrial
) -> None:
    projector = factory()
    first = projector.project(event for event in full_trial.events)
    first_json = first.canonical_semantic_json()
    first_hash = first.semantic_hash
    second = projector.project(full_trial.events)

    assert second.canonical_semantic_json() == first_json
    assert second.semantic_hash == first_hash
    assert second.semantic_sha256 == first_hash
    assert projector.last_result == second
    projector.reset()
    assert projector.last_result is None


def test_semantic_identity_ignores_recorded_at_but_detects_action_change(
    full_trial: FullTrial,
) -> None:
    shifted = tuple(
        event.model_copy(update={"recorded_at": event.recorded_at + timedelta(days=2)})
        for event in full_trial.events
    )
    baseline = TranscriptProjector().project(full_trial.events)
    retimed = TranscriptProjector().project(shifted)
    changed = TranscriptProjector().project(
        build_full_trial(first_action_hash=digest("materially-different-action")).events
    )

    assert baseline.canonical_semantic_json() == retimed.canonical_semantic_json()
    assert baseline.semantic_hash == retimed.semantic_hash
    assert changed.canonical_semantic_json() != baseline.canonical_semantic_json()
    assert changed.semantic_hash != baseline.semantic_hash


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("run_id", OTHER_RUN_ID),
        ("pod_id", "another-pod"),
        ("trial_id", "another-trial"),
        ("dyad_id", "another-dyad"),
    ),
)
def test_rejects_mixed_stream_identities(
    field_name: str, value: str, full_trial: FullTrial
) -> None:
    target = full_trial.named["agent_a"]
    attacked = replace_named(
        full_trial,
        "agent_a",
        clone_event(target, **{field_name: value}),
    )

    with pytest.raises(ProjectionIdentityError):
        MetricInputProjector().project(attacked)


def test_expected_run_and_trial_guards(full_trial: FullTrial) -> None:
    with pytest.raises(ProjectionIdentityError):
        TranscriptProjector(expected_run_id=OTHER_RUN_ID).project(full_trial.events)
    with pytest.raises(ProjectionIdentityError):
        TrialStateProjector(expected_trial_id="wrong-trial").project(full_trial.events)
    with pytest.raises(ValueError):
        TranscriptProjector(expected_run_id="")
    with pytest.raises(ValueError):
        TranscriptProjector(expected_trial_id="")


def test_rejects_duplicate_event_ids_sequence_gaps_and_wrong_hash_links(
    full_trial: FullTrial,
) -> None:
    run_started = full_trial.named["run_started"]
    with pytest.raises(ProjectionSequenceError, match="duplicate event_id"):
        MetricInputProjector().project((run_started, run_started))

    target = full_trial.named["agent_a"]
    target_position = full_trial.events.index(target)
    gap = clone_event(target, sequence_num=target.sequence_num + 1)
    with pytest.raises(ProjectionSequenceError, match="expected sequence"):
        MetricInputProjector().project(
            (*full_trial.events[:target_position], gap)
        )

    wrong_link = clone_event(target, previous_event_hash=digest("wrong-link"))
    with pytest.raises(ProjectionSequenceError, match="previous hash"):
        MetricInputProjector().project(
            (*full_trial.events[:target_position], wrong_link)
        )


def test_rejects_content_hash_tampering(full_trial: FullTrial) -> None:
    target = full_trial.named["committed_a"]
    tampered = target.model_copy(update={"content_hash": digest("tampered")})
    attacked = tuple(
        tampered if event.event_id == target.event_id else event
        for event in full_trial.events
    )

    with pytest.raises(ProjectionSequenceError, match="content-hash"):
        TranscriptProjector().project(attacked)


def test_rejects_missing_and_later_parents_and_schema_rejects_duplicates(
    full_trial: FullTrial,
) -> None:
    target = full_trial.named["agent_a"]
    target_position = full_trial.events.index(target)
    missing = clone_event(target, parent_event_ids=(event_id(999),))
    later = clone_event(
        target,
        parent_event_ids=(full_trial.named["committed_a"].event_id,),
    )
    with pytest.raises(ProjectionLinkError, match="missing or later parent"):
        MetricInputProjector().project((*full_trial.events[:target_position], missing))
    with pytest.raises(ProjectionLinkError, match="missing or later parent"):
        MetricInputProjector().project((*full_trial.events[:target_position], later))

    parent = target.parent_event_ids[0]
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        clone_event(target, parent_event_ids=(parent, parent))


def test_rejects_trial_events_before_start_and_mutation_after_terminal(
    full_trial: FullTrial,
) -> None:
    agent = full_trial.named["agent_a"]
    before_start = clone_event(
        agent,
        sequence_num=0,
        previous_event_hash=None,
        parent_event_ids=(),
    )
    with pytest.raises(ProjectionLifecycleError, match="before TrialStarted"):
        TrialStateProjector().project((before_start,))

    terminal = full_trial.named["trial_completed"]
    late = EventEnvelope(
        event_id=event_id(998),
        event_type="ObservationDelivered",
        run_id=RUN_ID,
        pod_id=POD_ID,
        trial_id=TRIAL_ID,
        dyad_id=DYAD_ID,
        sequence_num=terminal.sequence_num + 1,
        recorded_at=terminal.recorded_at + timedelta(seconds=1),
        actor_id=ALICE,
        actor_role=SELLER,
        model_call_id=None,
        parent_event_ids=(terminal.event_id,),
        previous_event_hash=terminal.content_hash,
        payload=ObservationDeliveredPayload(
            observation_id="late-observation",
            recipient_actor_id=ALICE,
            source_actor_id=BOB,
            source_event_id=full_trial.named["committed_b"].event_id,
            content_hash=digest("late"),
            visibility="public",
            sequence_in_recipient_view=2,
        ),
        payload_schema_version="1.0.0",
    )
    with pytest.raises(ProjectionLifecycleError, match="after terminal"):
        TrialStateProjector().project((*full_trial.events, late))


@pytest.mark.parametrize(
    ("name", "payload_update"),
    (
        ("agent_a", {"role": BUYER}),
        ("initial_a", {"recipient_actor_id": BOB}),
        ("proposed_a", {"actor_id": BOB}),
        ("call_a_completed", {"model_call_id": "different-call"}),
    ),
)
def test_rejects_payload_envelope_actor_role_recipient_and_call_mismatches(
    name: str, payload_update: dict[str, Any], full_trial: FullTrial
) -> None:
    target = full_trial.named[name]
    payload = target.payload.model_copy(update=payload_update)
    attacked = replace_named(full_trial, name, clone_event(target, payload=payload))

    with pytest.raises(ProjectionIdentityError):
        MetricInputProjector().project(attacked)


def test_rejects_turn_actor_and_private_view_role_mismatches(
    full_trial: FullTrial,
) -> None:
    turn = full_trial.named["turn_a"]
    wrong_turn = clone_event(
        turn,
        payload=turn.payload.model_copy(update={"from_actor_id": BOB}),
    )
    with pytest.raises(ProjectionIdentityError):
        TranscriptProjector().project(replace_named(full_trial, "turn_a", wrong_turn))

    private = full_trial.named["private_a"]
    wrong_private = clone_event(
        private,
        payload=private.payload.model_copy(update={"recipient_role": BUYER}),
    )
    with pytest.raises(ProjectionIdentityError):
        AgentViewProjector(ALICE).project(
            replace_named(full_trial, "private_a", wrong_private)
        )


def test_rejects_artifact_and_action_model_call_mismatches(
    full_trial: FullTrial,
) -> None:
    capture = full_trial.named["capture_a2"]
    capture_payload = capture.payload.model_copy(
        update={
            "artifact": capture.payload.artifact.model_copy(
                update={"source_model_call_id": "missing-call"}
            )
        }
    )
    wrong_capture = clone_event(
        capture,
        payload=capture_payload,
        model_call_id="missing-call",
    )
    with pytest.raises(ProjectionLinkError, match="no earlier completed"):
        ActivationSampleProjector().project(
            replace_named(full_trial, "capture_a2", wrong_capture)
        )

    proposal = full_trial.named["proposed_a"]
    proposal_payload = proposal.payload.model_copy(
        update={"model_call_id": "missing-call"}
    )
    wrong_proposal = clone_event(
        proposal,
        payload=proposal_payload,
        model_call_id="missing-call",
    )
    with pytest.raises(ProjectionLinkError, match="no completed model call"):
        ActivationSampleProjector().project(
            replace_named(full_trial, "proposed_a", wrong_proposal)
        )


def test_rejects_call_terminal_start_config_and_actor_inconsistency(
    full_trial: FullTrial,
) -> None:
    completed = full_trial.named["call_a_completed"]
    wrong_start = clone_event(
        completed,
        payload=completed.payload.model_copy(
            update={"started_event_id": full_trial.named["call_b_started"].event_id}
        ),
    )
    with pytest.raises(ProjectionLinkError, match="wrong ModelCallStarted"):
        ActivationSampleProjector().project(
            replace_named(full_trial, "call_a_completed", wrong_start)
        )

    wrong_config = clone_event(
        completed,
        payload=completed.payload.model_copy(
            update={"generation_config_hash": digest("other-config")}
        ),
    )
    with pytest.raises(ProjectionLinkError, match="generation config hash"):
        ActivationSampleProjector().project(
            replace_named(full_trial, "call_a_completed", wrong_config)
        )


def lineage_attack(full_trial: FullTrial, attack: str) -> tuple[EventEnvelope, ...]:
    duplicate_id = event_id(700 + len(attack))
    if attack == "missing_start":
        return without_named(full_trial, "call_a_started")
    if attack == "duplicate_start":
        original = full_trial.named["call_a_started"]
        duplicate = clone_event(
            original,
            event_id=duplicate_id,
            parent_event_ids=(original.event_id,),
        )
        return insert_after(full_trial, "call_a_started", duplicate)
    if attack == "missing_terminal":
        return without_named(full_trial, "call_a_completed")
    if attack == "duplicate_terminal":
        original = full_trial.named["call_a_completed"]
        duplicate = clone_event(
            original,
            event_id=duplicate_id,
            parent_event_ids=(original.event_id,),
        )
        return insert_after(full_trial, "call_a_completed", duplicate)
    if attack == "missing_activation":
        return without_named(full_trial, "capture_a5")
    if attack == "duplicate_activation":
        original = full_trial.named["capture_a2"]
        duplicate = clone_event(
            original,
            event_id=duplicate_id,
            parent_event_ids=(original.event_id,),
        )
        return insert_after(full_trial, "capture_a2", duplicate)
    if attack == "missing_proposal":
        return without_named(full_trial, "proposed_a")
    if attack == "duplicate_proposal":
        original = full_trial.named["proposed_a"]
        duplicate = clone_event(
            original,
            event_id=duplicate_id,
            parent_event_ids=(original.event_id,),
        )
        return insert_after(full_trial, "proposed_a", duplicate)
    if attack == "missing_commit":
        return without_named(full_trial, "committed_a")
    if attack == "duplicate_commit":
        original = full_trial.named["committed_a"]
        duplicate = clone_event(
            original,
            event_id=duplicate_id,
            parent_event_ids=(original.event_id,),
        )
        return insert_after(full_trial, "committed_a", duplicate)
    if attack == "missing_turn":
        return without_named(full_trial, "turn_a")
    if attack == "duplicate_turn":
        original = full_trial.named["turn_a"]
        duplicate = clone_event(
            original,
            event_id=duplicate_id,
            parent_event_ids=(original.event_id,),
        )
        return insert_after(full_trial, "turn_a", duplicate)
    raise AssertionError(f"unknown lineage attack {attack}")


@pytest.mark.parametrize(
    "attack",
    (
        "missing_start",
        "duplicate_start",
        "missing_terminal",
        "duplicate_terminal",
        "missing_activation",
        "duplicate_activation",
        "missing_proposal",
        "duplicate_proposal",
        "missing_commit",
        "duplicate_commit",
        "missing_turn",
        "duplicate_turn",
    ),
)
def test_rejects_missing_and_duplicate_call_action_activation_turn_lineage(
    attack: str, full_trial: FullTrial
) -> None:
    with pytest.raises(ProjectionLinkError):
        ActivationSampleProjector().project(lineage_attack(full_trial, attack))


def test_rejects_action_hash_artifact_metadata_and_activation_order_mismatch(
    full_trial: FullTrial,
) -> None:
    committed = full_trial.named["committed_a"]
    wrong_action = clone_event(
        committed,
        payload=committed.payload.model_copy(
            update={"action_hash": digest("different-commit")}
        ),
    )
    with pytest.raises(ProjectionLinkError, match="action hash"):
        TranscriptProjector().project(
            replace_named(full_trial, "committed_a", wrong_action)
        )

    capture = full_trial.named["capture_a2"]
    wrong_artifact = clone_event(
        capture,
        payload=capture.payload.model_copy(
            update={
                "artifact": capture.payload.artifact.model_copy(
                    update={"tokenizer_id": "wrong-tokenizer"}
                )
            }
        ),
    )
    with pytest.raises(ProjectionLinkError, match="tokenizer"):
        ActivationSampleProjector().project(
            replace_named(full_trial, "capture_a2", wrong_artifact)
        )

    completed = full_trial.named["call_a_completed"]
    reversed_hashes = tuple(reversed(completed.payload.activation_artifact_hashes))
    wrong_order = clone_event(
        completed,
        payload=completed.payload.model_copy(
            update={"activation_artifact_hashes": reversed_hashes}
        ),
    )
    with pytest.raises(ProjectionLinkError, match="activation events"):
        ActivationSampleProjector().project(
            replace_named(full_trial, "call_a_completed", wrong_order)
        )


def test_missing_artifact_is_loud_but_projected_references_need_no_filesystem(
    full_trial: FullTrial, tmp_path: Path
) -> None:
    projection = ActivationSampleProjector().project(full_trial.events)
    assert not tuple(tmp_path.iterdir())
    assert projection.samples[0].activation_artifacts
    assert any(
        warning.code == "artifact_not_loaded"
        for warning in projection.samples[0].warnings
    )

    with pytest.raises(ProjectionLinkError, match="activation events"):
        ActivationSampleProjector().project(
            without_named(full_trial, "capture_a5")
        )


def test_agent_view_rejects_observation_gaps_and_duplicates(
    full_trial: FullTrial,
) -> None:
    with pytest.raises(ProjectionSequenceError, match="has gaps"):
        AgentViewProjector(BOB).project(without_named(full_trial, "initial_b"))

    reception = full_trial.named["reception_b"]
    duplicate_position = clone_event(
        reception,
        payload=reception.payload.model_copy(
            update={"sequence_in_recipient_view": 0}
        ),
    )
    with pytest.raises(ProjectionSequenceError, match="duplicate"):
        AgentViewProjector(BOB).project(
            replace_named(full_trial, "reception_b", duplicate_position)
        )


@pytest.mark.parametrize("cutoff", (-1, True, "3", 1.5))
def test_agent_view_rejects_invalid_cutoffs(cutoff: Any, full_trial: FullTrial) -> None:
    with pytest.raises(ValueError, match="through_sequence_num"):
        AgentViewProjector(ALICE, through_sequence_num=cutoff)
    with pytest.raises(ValueError, match="through_sequence_num"):
        AgentViewProjector(ALICE).project(
            full_trial.events, through_sequence_num=cutoff
        )


def test_opaque_unrelated_events_are_ignored_but_required_types_fail(
    full_trial: FullTrial,
) -> None:
    outcome = full_trial.named["outcome"]
    unrelated = EventEnvelope(
        event_id=event_id(996),
        event_type="FutureProjectionFact",
        run_id=RUN_ID,
        pod_id=POD_ID,
        trial_id=TRIAL_ID,
        dyad_id=DYAD_ID,
        sequence_num=outcome.sequence_num + 1,
        recorded_at=outcome.recorded_at + timedelta(microseconds=1),
        actor_id=None,
        actor_role=None,
        model_call_id=None,
        parent_event_ids=(outcome.event_id,),
        previous_event_hash=outcome.content_hash,
        payload=OpaqueEventPayload.from_payload_dict(
            "FutureProjectionFact", "9.0.0", {"future": True}
        ),
        payload_schema_version="9.0.0",
    )
    with_unrelated = insert_after(full_trial, "outcome", unrelated)
    baseline = MetricInputProjector().project(full_trial.events)
    projected = MetricInputProjector().project(with_unrelated)
    assert projected == baseline

    required = clone_event(
        unrelated,
        event_id=event_id(995),
        event_type="BehaviorLabeled",
        payload=OpaqueEventPayload.from_payload_dict(
            "BehaviorLabeled", "9.0.0", {}
        ),
    )
    with pytest.raises(UnprojectableEventError, match="required"):
        MetricInputProjector().project(insert_after(full_trial, "outcome", required))


def test_external_parent_ids_support_annotation_projection() -> None:
    external = event_id(990)
    builder = StreamBuilder()
    started = builder.add(
        "trial_started",
        "TrialStarted",
        TrialStartedPayload(
            trial_id=TRIAL_ID,
            scenario_instance_id="external-scenario",
            dyad_id=DYAD_ID,
            attempt=1,
            actor_ids=(ALICE, BOB),
            source_scenario_event_id=external,
        ),
        parent_event_ids=(external,),
    )
    builder.add(
        "external_label",
        "BehaviorLabeled",
        BehaviorLabeledPayload(
            label_id="external-label",
            target_event_id=external,
            target_actor_id=ALICE,
            label_name="posthoc-label",
            value=LabelValue(kind="boolean", boolean_value=True),
            provenance=LabelProvenance(
                source="human",
                method_id="reviewer",
                method_version="1.0.0",
                source_event_ids=(external,),
                evaluation_succeeded=True,
            ),
        ),
        parent_event_ids=(started.event_id, external),
    )

    with pytest.raises(ProjectionLinkError, match="missing or later parent"):
        MetricInputProjector().project(builder.events)
    projection = MetricInputProjector(
        external_parent_ids=(external,)
    ).project(builder.events)
    assert [label.label_id for label in projection.labels] == ["external-label"]

    with pytest.raises(ValueError, match="external_parent_ids"):
        MetricInputProjector(external_parent_ids=("",))
    with pytest.raises(ValueError, match="external_parent_ids"):
        MetricInputProjector(external_parent_ids=(123,))  # type: ignore[arg-type]


def test_empty_open_failed_and_complete_trial_states(full_trial: FullTrial) -> None:
    empty_transcript = TranscriptProjector().project(())
    empty_view = AgentViewProjector(ALICE).project(())
    empty_state = TrialStateProjector().project(())
    empty_samples = ActivationSampleProjector().project(())
    empty_dyad = DyadProjector().project(())
    empty_metrics = MetricInputProjector().project(())

    assert empty_transcript.actions == ()
    assert empty_view.observations == ()
    assert empty_state.lifecycle_status == "empty"
    assert empty_samples.samples == ()
    assert empty_dyad.turns == ()
    assert empty_metrics.labels == ()

    outcome_position = full_trial.events.index(full_trial.named["outcome"])
    open_state = TrialStateProjector().project(
        full_trial.events[:outcome_position]
    )
    assert open_state.lifecycle_status == "open"
    assert open_state.terminal_event_id is None
    assert len(open_state.commitments) == 2

    builder = StreamBuilder()
    started = builder.add(
        "started",
        "TrialStarted",
        TrialStartedPayload(
            trial_id=TRIAL_ID,
            scenario_instance_id="failed-scenario",
            dyad_id=DYAD_ID,
            attempt=1,
            actor_ids=(ALICE, BOB),
            source_scenario_event_id=event_id(980),
        ),
    )
    failed = builder.add(
        "failed",
        "TrialFailed",
        TrialFailedPayload(
            trial_id=TRIAL_ID,
            error_type="ModelFailure",
            error_message_hash=digest("failure"),
            resumable=False,
            last_event_id=started.event_id,
        ),
        parent_event_ids=(started.event_id,),
    )
    failed_state = TrialStateProjector().project(builder.events)
    assert failed_state.lifecycle_status == "failed"
    assert failed_state.terminal_event_id == failed.event_id

    complete_state = TrialStateProjector().project(full_trial.events)
    assert complete_state.lifecycle_status == "completed"


@pytest.mark.parametrize("factory", PROJECTOR_FACTORIES)
def test_projectors_reject_non_iterable_and_non_envelope_inputs(
    factory: Callable[[], Any],
) -> None:
    with pytest.raises(TypeError):
        factory().project("not-events")
    with pytest.raises(TypeError, match="must be EventEnvelope"):
        factory().project((object(),))


@pytest.mark.parametrize(
    ("name", "mutator"),
    (
        (
            "label_rules",
            lambda payload: payload.model_copy(update={"target_actor_id": BOB}),
        ),
        (
            "monitor",
            lambda payload: payload.model_copy(update={"target_actor_id": BOB}),
        ),
    ),
)
def test_rejects_annotation_target_actor_mismatch(
    name: str,
    mutator: Callable[[Any], EventPayload],
    full_trial: FullTrial,
) -> None:
    target = full_trial.named[name]
    attacked = replace_named(
        full_trial,
        name,
        clone_event(target, payload=mutator(target.payload)),
    )
    with pytest.raises(ProjectionLinkError, match="target actor"):
        MetricInputProjector().project(attacked)


def test_rejects_missing_label_monitor_qc_and_outcome_sources(
    full_trial: FullTrial,
) -> None:
    missing = event_id(979)

    label_event = full_trial.named["label_rules"]
    label_payload = label_event.payload
    label_provenance = label_payload.provenance.model_copy(
        update={
            "source_event_ids": (
                label_payload.target_event_id,
                missing,
            )
        }
    )
    bad_label = clone_event(
        label_event,
        payload=label_payload.model_copy(update={"provenance": label_provenance}),
    )

    monitor_event = full_trial.named["monitor"]
    bad_monitor = clone_event(
        monitor_event,
        payload=monitor_event.payload.model_copy(
            update={
                "evidence_event_ids": (
                    monitor_event.payload.target_event_id,
                    missing,
                )
            }
        ),
    )

    qc_event = full_trial.named["qc"]
    bad_qc = clone_event(
        qc_event,
        payload=qc_event.payload.model_copy(
            update={"source_event_ids": (qc_event.payload.target_event_id, missing)}
        ),
    )

    outcome_event = full_trial.named["outcome"]
    bad_outcome = clone_event(
        outcome_event,
        payload=outcome_event.payload.model_copy(
            update={"source_event_ids": (missing,)}
        ),
    )

    for name, event in (
        ("label_rules", bad_label),
        ("monitor", bad_monitor),
        ("qc", bad_qc),
        ("outcome", bad_outcome),
    ):
        with pytest.raises(ProjectionLinkError, match="missing event"):
            TrialStateProjector().project(replace_named(full_trial, name, event))


def test_trial_terminal_requires_recorded_outcome_actions_and_artifacts(
    full_trial: FullTrial,
) -> None:
    terminal = full_trial.named["trial_completed"]
    missing = event_id(978)
    attacks = (
        terminal.payload.model_copy(update={"outcome_event_id": missing}),
        terminal.payload.model_copy(update={"terminal_action_event_ids": (missing,)}),
        terminal.payload.model_copy(
            update={"required_artifact_hashes": (digest("absent-artifact"),)}
        ),
    )
    for payload in attacks:
        with pytest.raises(ProjectionLinkError):
            TrialStateProjector().project(
                replace_named(
                    full_trial,
                    "trial_completed",
                    clone_event(terminal, payload=payload),
                )
            )


def test_failed_terminal_last_event_must_precede_failure() -> None:
    builder = StreamBuilder()
    started = builder.add(
        "started",
        "TrialStarted",
        TrialStartedPayload(
            trial_id=TRIAL_ID,
            scenario_instance_id="failed-scenario",
            dyad_id=DYAD_ID,
            attempt=1,
            actor_ids=(ALICE, BOB),
            source_scenario_event_id=event_id(977),
        ),
    )
    failed_id = event_id(976)
    builder.add(
        "failed",
        "TrialFailed",
        TrialFailedPayload(
            trial_id=TRIAL_ID,
            error_type="ModelFailure",
            error_message_hash=digest("failure"),
            resumable=False,
            last_event_id=failed_id,
        ),
        parent_event_ids=(started.event_id,),
    )
    failed = clone_event(builder.events[-1], event_id=failed_id)
    attacked = normalize((started, failed))

    with pytest.raises(ProjectionLinkError, match="last_event_id"):
        TrialStateProjector().project(attacked)
