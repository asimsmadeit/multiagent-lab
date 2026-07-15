"""Lineage-query and PROV-export tests for events provenance (Plan 2)."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone

import pytest

from interpretability.events.payloads import (
    ActionCommittedPayload,
    ActionProposedPayload,
    BehaviorLabeledPayload,
    CanonicalJsonDocument,
    LabelProvenance,
    LabelValue,
    ModelCallCompletedPayload,
    ModelCallStartedPayload,
    RunConfigFrozenPayload,
    RunStartedPayload,
    ScenarioInstantiatedPayload,
    TrialStartedPayload,
    UsageRecord,
)
from interpretability.events.provenance import (
    PROV_DIALECT,
    DuplicateEventIdError,
    LineageCycleError,
    ProvenanceError,
    UnknownEventError,
    prov_document,
    trace_event,
)
from interpretability.events.schema import (
    ActivationCapturedPayload,
    ArtifactReference,
    EventEnvelope,
    EventPayload,
)

RUN_ID = "70000000-0000-4000-8000-0000000000aa"
POD_ID = "pod-provenance"
TRIAL_ID = "trial-provenance"
DYAD_ID = "dyad-provenance"
ALICE = "alice"
CALL = "call-alice-action"
BASE_TIME = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def eid(index: int) -> str:
    return f"91000000-0000-4000-8000-{index:012x}"


def document(value: dict, version: str = "1.0.0") -> CanonicalJsonDocument:
    return CanonicalJsonDocument.from_mapping(value, schema_version=version)


class ChainBuilder:
    """Minimal deterministic envelope builder for provenance fixtures."""

    def __init__(self) -> None:
        self.events: list[EventEnvelope] = []
        self.named: dict[str, EventEnvelope] = {}
        self._lanes: dict[str | None, list[EventEnvelope]] = {}

    def add(
        self,
        name: str,
        event_type: str,
        payload: EventPayload,
        *,
        trial_id: str | None = None,
        actor_id: str | None = None,
        actor_role: str | None = None,
        model_call_id: str | None = None,
        parents: tuple[str, ...] = (),
    ) -> EventEnvelope:
        lane = self._lanes.setdefault(trial_id, [])
        envelope = EventEnvelope(
            event_id=eid(len(self.events) + 1),
            event_type=event_type,
            run_id=RUN_ID,
            pod_id=POD_ID,
            trial_id=trial_id,
            dyad_id=DYAD_ID if trial_id is not None else None,
            sequence_num=len(lane),
            recorded_at=BASE_TIME + timedelta(seconds=len(self.events)),
            actor_id=actor_id,
            actor_role=actor_role,
            model_call_id=model_call_id,
            parent_event_ids=parents,
            previous_event_hash=None if not lane else lane[-1].content_hash,
            payload=payload,
            payload_schema_version=payload.PAYLOAD_SCHEMA_VERSION or "1.0.0",
        )
        lane.append(envelope)
        self.events.append(envelope)
        self.named[name] = envelope
        return envelope


ARTIFACT_HASH = digest("alice-activation-artifact")


def build_chain() -> ChainBuilder:
    """run -> config -> scenario -> trial -> call -> action -> label."""
    builder = ChainBuilder()
    run_started = builder.add(
        "run_started",
        "RunStarted",
        RunStartedPayload(
            run_id=RUN_ID,
            orchestrator_id="provenance-tests",
            code_revision="revision-1",
            run_seed=3,
        ),
    )
    config = builder.add(
        "config",
        "RunConfigFrozen",
        RunConfigFrozenPayload(
            run_id=RUN_ID,
            config=document({"model": "model-a", "seed": 3}),
            code_revision="revision-1",
            working_tree_dirty=False,
            python_version="3.12.13",
            package_lock_hash=digest("lock"),
            scenario_spec_hashes=(digest("spec"),),
        ),
        parents=(run_started.event_id,),
    )
    scenario = builder.add(
        "scenario",
        "ScenarioInstantiated",
        ScenarioInstantiatedPayload(
            scenario_instance_id="scenario-instance",
            scenario_type="hidden_value",
            scenario_schema_version="1.0.0",
            scenario_spec_hash=digest("spec"),
            parameters=document({"reserve_price": 70}),
            trial_seed=11,
        ),
        parents=(config.event_id,),
    )
    trial_started = builder.add(
        "trial_started",
        "TrialStarted",
        TrialStartedPayload(
            trial_id=TRIAL_ID,
            scenario_instance_id="scenario-instance",
            dyad_id=DYAD_ID,
            attempt=1,
            actor_ids=(ALICE, "bob"),
            source_scenario_event_id=scenario.event_id,
        ),
        trial_id=TRIAL_ID,
        parents=(scenario.event_id,),
    )
    call_started = builder.add(
        "call_started",
        "ModelCallStarted",
        ModelCallStartedPayload(
            model_call_id=CALL,
            purpose="actor_action",
            actor_id=ALICE,
            model_id="model-a",
            model_revision="model-a-rev",
            tokenizer_id="tokenizer-a",
            tokenizer_revision="tokenizer-a-rev",
            prompt_id="prompt-1",
            prompt_hash=digest("prompt"),
            input_event_ids=(trial_started.event_id,),
            generation_config=document({"temperature": 0.7}),
        ),
        trial_id=TRIAL_ID,
        actor_id=ALICE,
        actor_role="seller",
        model_call_id=CALL,
        parents=(trial_started.event_id,),
    )
    activation = builder.add(
        "activation",
        "ActivationCaptured",
        ActivationCapturedPayload(
            artifact=ArtifactReference(
                artifact_hash=ARTIFACT_HASH,
                hook_name="blocks.14.hook_resid_post",
                layer=14,
                token_selection="generated_last",
                aggregation="none",
                shape=(1, 8),
                dtype="float32",
                tokenizer_id="tokenizer-a",
                model_revision="model-a-rev",
                source_model_call_id=CALL,
            )
        ),
        trial_id=TRIAL_ID,
        actor_id=ALICE,
        actor_role="seller",
        model_call_id=CALL,
        parents=(call_started.event_id,),
    )
    call_completed = builder.add(
        "call_completed",
        "ModelCallCompleted",
        ModelCallCompletedPayload(
            model_call_id=CALL,
            purpose="actor_action",
            actor_id=ALICE,
            started_event_id=call_started.event_id,
            output_id="output-1",
            output_hash=digest("output"),
            token_ids_hash=digest("tokens"),
            generation_config_hash=digest("generation-config"),
            usage=UsageRecord(input_tokens=9, output_tokens=5, total_tokens=14),
            finish_reason="stop",
            activation_artifact_hashes=(ARTIFACT_HASH,),
        ),
        trial_id=TRIAL_ID,
        actor_id=ALICE,
        actor_role="seller",
        model_call_id=CALL,
        parents=(call_started.event_id,),
    )
    proposed = builder.add(
        "proposed",
        "ActionProposed",
        ActionProposedPayload(
            action_id="action-1",
            actor_id=ALICE,
            model_call_id=CALL,
            model_call_event_id=call_started.event_id,
            action_spec_id="spec-1",
            action_hash=digest("action"),
        ),
        trial_id=TRIAL_ID,
        actor_id=ALICE,
        actor_role="seller",
        model_call_id=CALL,
        parents=(call_completed.event_id,),
    )
    committed = builder.add(
        "committed",
        "ActionCommitted",
        ActionCommittedPayload(
            action_id="action-1",
            actor_id=ALICE,
            proposed_event_id=proposed.event_id,
            model_call_id=CALL,
            action_hash=digest("action"),
        ),
        trial_id=TRIAL_ID,
        actor_id=ALICE,
        actor_role="seller",
        model_call_id=CALL,
        parents=(proposed.event_id,),
    )
    builder.add(
        "label",
        "BehaviorLabeled",
        BehaviorLabeledPayload(
            label_id="label-1",
            target_event_id=committed.event_id,
            target_actor_id=ALICE,
            label_name="actual_deception",
            value=LabelValue(kind="boolean", boolean_value=True),
            provenance=LabelProvenance(
                source="rules",
                method_id="rule-evaluator",
                method_version="1.0.0",
                source_event_ids=(committed.event_id,),
                evaluation_succeeded=True,
            ),
        ),
        trial_id=TRIAL_ID,
        parents=(committed.event_id,),
    )
    return builder


# ---------------------------------------------------------------------------
# trace_event
# ---------------------------------------------------------------------------


def test_trace_collects_ancestors_descendants_call_and_classified_events():
    chain = build_chain()
    named = chain.named
    trace = trace_event(chain.events, named["committed"].event_id)

    assert trace.ancestor_event_ids == (
        named["proposed"].event_id,
        named["call_completed"].event_id,
        named["call_started"].event_id,
        named["trial_started"].event_id,
        named["scenario"].event_id,
        named["config"].event_id,
        named["run_started"].event_id,
    )
    assert trace.descendant_event_ids == (named["label"].event_id,)
    assert trace.same_call_event_ids == (
        named["call_started"].event_id,
        named["activation"].event_id,
        named["call_completed"].event_id,
        named["proposed"].event_id,
    )
    assert trace.artifact_hashes == (ARTIFACT_HASH,)
    assert trace.scenario_event_ids == (named["scenario"].event_id,)
    assert trace.config_event_ids == (named["config"].event_id,)
    assert trace.label_event_ids == (named["label"].event_id,)
    assert trace.actor_ids == (ALICE,)
    assert trace.missing_parent_ids == ()

    closure = trace.closure_event_ids()
    assert closure[0] == named["committed"].event_id
    assert len(closure) == len(set(closure)) == len(chain.events)


def test_trace_of_the_root_orders_descendants_by_depth_then_sequence():
    chain = build_chain()
    named = chain.named
    trace = trace_event(chain.events, named["run_started"].event_id)
    assert trace.ancestor_event_ids == ()
    assert trace.descendant_event_ids == (
        named["config"].event_id,
        named["scenario"].event_id,
        named["trial_started"].event_id,
        named["call_started"].event_id,
        named["activation"].event_id,
        named["call_completed"].event_id,
        named["proposed"].event_id,
        named["committed"].event_id,
        named["label"].event_id,
    )
    # The root has no model call, so no same-call siblings.
    assert trace.same_call_event_ids == ()


def test_trace_is_independent_of_input_order():
    chain = build_chain()
    target = chain.named["committed"].event_id
    baseline = trace_event(chain.events, target)
    shuffled = list(chain.events)
    random.Random(5).shuffle(shuffled)
    assert trace_event(shuffled, target) == baseline


def test_trace_records_missing_parents_instead_of_failing():
    chain = build_chain()
    orphan_parent = "99000000-0000-4000-8000-00000000dead"
    payload = RunStartedPayload(
        run_id=RUN_ID,
        orchestrator_id="provenance-tests",
        code_revision="revision-1",
        run_seed=4,
    )
    orphan_env = EventEnvelope(
        event_id="93000000-0000-4000-8000-000000000001",
        event_type="RunStarted",
        run_id=RUN_ID,
        pod_id=POD_ID,
        trial_id=None,
        dyad_id=None,
        sequence_num=0,
        recorded_at=BASE_TIME + timedelta(minutes=2),
        actor_id=None,
        actor_role=None,
        model_call_id=None,
        parent_event_ids=(orphan_parent,),
        previous_event_hash=None,
        payload=payload,
        payload_schema_version=payload.PAYLOAD_SCHEMA_VERSION or "1.0.0",
    )
    trace = trace_event(chain.events + [orphan_env], orphan_env.event_id)
    assert trace.ancestor_event_ids == ()
    assert trace.missing_parent_ids == (orphan_parent,)


def test_trace_fails_closed_on_bad_streams():
    chain = build_chain()
    with pytest.raises(UnknownEventError):
        trace_event(chain.events, "99000000-0000-4000-8000-000000000bad")
    with pytest.raises(DuplicateEventIdError):
        trace_event(
            chain.events + [chain.events[0]], chain.events[0].event_id
        )
    with pytest.raises(ProvenanceError):
        trace_event([42], "anything")  # type: ignore[list-item]


def _cycle_pair() -> list[EventEnvelope]:
    id_a = "92000000-0000-4000-8000-00000000000a"
    id_b = "92000000-0000-4000-8000-00000000000b"
    payload = RunStartedPayload(
        run_id=RUN_ID,
        orchestrator_id="provenance-tests",
        code_revision="revision-1",
        run_seed=5,
    )

    def envelope(
        event_id: str,
        parent_id: str,
        sequence: int,
        previous_event_hash: str | None,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_id=event_id,
            event_type="RunStarted",
            run_id=RUN_ID,
            pod_id=POD_ID,
            trial_id=None,
            dyad_id=None,
            sequence_num=sequence,
            recorded_at=BASE_TIME + timedelta(minutes=1, seconds=sequence),
            actor_id=None,
            actor_role=None,
            model_call_id=None,
            parent_event_ids=(parent_id,),
            previous_event_hash=previous_event_hash,
            payload=payload,
            payload_schema_version=payload.PAYLOAD_SCHEMA_VERSION or "1.0.0",
        )

    first = envelope(id_a, id_b, 0, None)
    second = envelope(id_b, id_a, 1, first.content_hash)
    return [first, second]


def test_any_cycle_in_the_stream_fails_closed_even_off_target():
    cycle = _cycle_pair()
    with pytest.raises(LineageCycleError):
        trace_event(cycle, cycle[0].event_id)
    # A cycle elsewhere in the stream is still corruption for any query.
    chain = build_chain()
    with pytest.raises(LineageCycleError):
        trace_event(chain.events + cycle, chain.named["committed"].event_id)


# ---------------------------------------------------------------------------
# prov_document
# ---------------------------------------------------------------------------


def test_prov_document_entities_relations_and_determinism():
    chain = build_chain()
    named = chain.named
    target = named["committed"].event_id
    doc = prov_document(chain.events, target)

    assert doc["prov_dialect"] == PROV_DIALECT
    assert doc["target_event_id"] == target
    assert doc["missing_parent_ids"] == []

    assert set(doc["entity"]) == {
        f"artifact:{ARTIFACT_HASH}",
        f"event:{named['scenario'].event_id}",
        f"event:{named['config'].event_id}",
    }
    assert doc["entity"][f"artifact:{ARTIFACT_HASH}"] == {
        "prov:type": "activation_artifact"
    }
    committed_activity = doc["activity"][f"event:{target}"]
    assert committed_activity["prov:type"] == "ActionCommitted"
    assert committed_activity["model_call_id"] == CALL
    assert doc["agent"] == {f"actor:{ALICE}": {"prov:type": "agent"}}

    assert doc["wasGeneratedBy"] == [
        [f"artifact:{ARTIFACT_HASH}", f"event:{named['activation'].event_id}"]
    ]
    informed = doc["wasInformedBy"]
    assert [
        f"event:{named['label'].event_id}",
        f"event:{target}",
    ] in informed
    assert all(len(pair) == 2 for pair in informed)
    assert [
        f"event:{target}",
        f"actor:{ALICE}",
    ] in doc["wasAssociatedWith"]

    shuffled = list(chain.events)
    random.Random(9).shuffle(shuffled)
    assert json.dumps(prov_document(shuffled, target), sort_keys=True) == (
        json.dumps(doc, sort_keys=True)
    )


def test_prov_document_scopes_relations_to_the_closure():
    chain = build_chain()
    named = chain.named
    # Tracing the label keeps the whole ancestor chain but has no same-call
    # events (the label has no model_call_id), so the activation event is
    # absent and no artifact entity may appear.
    doc = prov_document(chain.events, named["label"].event_id)
    assert f"artifact:{ARTIFACT_HASH}" not in doc["entity"]
    assert doc["wasGeneratedBy"] == []
    assert f"event:{named['activation'].event_id}" not in doc["activity"]
