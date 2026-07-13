"""Replay and failure contracts for the trial state machine."""

from __future__ import annotations

import json

import pytest

from interpretability.runtime.trial import (
    TRIAL_RUNTIME_SCHEMA_VERSION,
    TrialRunner,
    TrialRuntimeEvent,
    TrialState,
)


def _advance_one_turn(runner: TrialRunner) -> None:
    runner.transition(TrialState.TURN_PROPOSED, {
        "actor_id": "seller",
        "round_index": 0,
        "generation_call_id": "call-1",
        "attempt": runner.attempt,
    })
    runner.transition(
        TrialState.ACTION_CAPTURED,
        {
            "actor_id": "seller",
            "round_index": 0,
            "output_text": "I offer $70.",
            "generation_record_id": "call-1",
        },
    )
    runner.transition(
        TrialState.ADJUDICATED,
        {
            "interaction_event_id": "event-1",
            "actor_id": "seller",
            "resolution_id": "resolution-1",
            "accepted": True,
            "action_id": "action-1",
            "label_record_ids": ["label-1"],
        },
    )
    runner.transition(
        TrialState.OBSERVED,
        {
            "actor_id": "seller",
            "observation": "The offer was delivered.",
            "interaction_event_id": "event-1",
            "label_record_ids": ["label-1"],
        },
    )


def test_trial_state_replays_identically_and_projects_without_callbacks() -> None:
    runner = TrialRunner(run_id="run-1", trial_id="trial-1", attempt=2)
    runner.transition(TrialState.COMPILED, {"scenario_instance_id": "scenario-1"})
    runner.transition(TrialState.AGENTS_BUILT, {"actors": ["seller", "buyer"]})
    runner.transition(TrialState.INITIALIZED, {"round_index": 0})
    _advance_one_turn(runner)
    runner.transition(TrialState.COMPLETED, {
        "outcome_id": "outcome-1",
        "reason": "agreement",
    })

    serialized = json.loads(json.dumps(runner.get_state()))
    restored = TrialRunner.from_state(serialized)

    assert restored.get_state() == runner.get_state()
    assert (
            restored.get_state()["schema_version"]
            == TRIAL_RUNTIME_SCHEMA_VERSION
            == "1.7.0"
        )
    assert restored.project_transcript() == runner.project_transcript()
    assert restored.project_transcript()[0]["generation_record_id"] == "call-1"
    assert restored.project_transcript()[1]["label_record_ids"] == ["label-1"]


def test_interrupted_trial_resumes_at_exact_persisted_boundary() -> None:
    runner = TrialRunner(run_id="run-1", trial_id="trial-1")
    runner.transition(TrialState.COMPILED, {"scenario_instance_id": "scenario-1"})
    runner.transition(TrialState.AGENTS_BUILT, {"actors": ["seller", "buyer"]})
    runner.transition(TrialState.INITIALIZED, {"round_index": 0})
    runner.transition(TrialState.TURN_PROPOSED, {
        "actor_id": "seller",
        "round_index": 0,
        "generation_call_id": "c",
        "attempt": 0,
    })
    restored = TrialRunner.from_state(json.loads(json.dumps(runner.get_state())))

    restored.transition(
        TrialState.ACTION_CAPTURED,
        {
            "actor_id": "seller",
            "round_index": 0,
            "output_text": "offer",
            "generation_record_id": "c",
        },
    )

    assert restored.events[-1].sequence == 4
    assert restored.events[-1].from_state is TrialState.TURN_PROPOSED


def test_invalid_transition_and_tampered_replay_fail_closed() -> None:
    runner = TrialRunner(run_id="run-1", trial_id="trial-1")
    with pytest.raises(ValueError, match="invalid trial transition"):
        runner.transition(TrialState.ACTION_CAPTURED)

    runner.transition(TrialState.COMPILED, {"scenario_instance_id": "scenario-1"})
    payload = runner.get_state()
    payload["events"][0]["payload"]["scenario_instance_id"] = "tampered"
    with pytest.raises(ValueError, match="event ID"):
        TrialRunner.from_state(payload)

    payload = runner.get_state()
    del payload["events"][0]["event_id"]
    payload["events"][0]["payload"]["scenario_instance_id"] = "tampered"
    with pytest.raises(ValueError, match="missing fields: event_id"):
        TrialRunner.from_state(payload)

    payload = runner.get_state()
    del payload["events"][0]["schema_version"]
    with pytest.raises(ValueError, match="missing fields: schema_version"):
        TrialRunner.from_state(payload)

    fresh = TrialRunner(run_id="run-2", trial_id="trial-2")
    with pytest.raises(ValueError, match="finite"):
        fresh.transition(TrialState.COMPILED, {
            "scenario_instance_id": "scenario-2",
            "invalid": float("nan"),
        })
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        fresh.transition(TrialState.COMPILED, {
            "scenario_instance_id": "scenario-2",
            1: "colliding key",
        })


def test_failure_is_terminal_and_retains_error_provenance() -> None:
    runner = TrialRunner(run_id="run-1", trial_id="trial-1")
    runner.transition(TrialState.FAILED, {"error_type": "RuntimeError", "error": "offline"})

    assert runner.state is TrialState.FAILED
    assert runner.events[-1].payload["error_type"] == "RuntimeError"
    with pytest.raises(ValueError, match="invalid trial transition"):
        runner.transition(TrialState.COMPILED)


def test_intervention_application_is_replayable_before_action() -> None:
    runner = TrialRunner(run_id="run-1", trial_id="trial-1")
    runner.transition(TrialState.COMPILED, {"scenario_instance_id": "scenario-1"})
    runner.transition(TrialState.AGENTS_BUILT, {"actors": ["seller", "buyer"]})
    runner.transition(TrialState.INITIALIZED, {"round_index": 0})
    receipt = runner.transition(TrialState.INTERVENTION_APPLIED, {
        "application_receipt_id": "receipt-1",
        "intervention_design_id": "design-1",
        "intervention_family": "scripted_observation",
        "target_actor_id": "seller",
        "round_index": 0,
        "committed_action_boundary": 0,
        "content_hash": "sha256:" + ("a" * 64),
        "source": "registered:test/v1",
        "status": "applied",
        "observation": "A deterministic public observation.",
        "evidence_call_id": None,
    })
    restored = TrialRunner.from_state(
        json.loads(json.dumps(runner.get_state()))
    )
    restored.transition(TrialState.TURN_PROPOSED, {
        "actor_id": "seller",
        "round_index": 0,
        "generation_call_id": "call-1",
        "attempt": 0,
    })

    assert receipt.to_state is TrialState.INTERVENTION_APPLIED
    assert restored.events[-1].from_state is TrialState.INTERVENTION_APPLIED

    missing_evidence = TrialRunner(run_id="run-2", trial_id="trial-2")
    missing_evidence.transition(
        TrialState.COMPILED, {"scenario_instance_id": "scenario-2"}
    )
    missing_evidence.transition(
        TrialState.AGENTS_BUILT, {"actors": ["seller", "buyer"]}
    )
    missing_evidence.transition(TrialState.INITIALIZED, {"round_index": 0})
    with pytest.raises(ValueError, match="explicitly present"):
        missing_evidence.transition(TrialState.INTERVENTION_APPLIED, {
            "application_receipt_id": "receipt-2",
            "intervention_design_id": "design-2",
            "intervention_family": "probe",
            "target_actor_id": "seller",
            "round_index": 0,
            "committed_action_boundary": 0,
            "content_hash": "sha256:" + ("b" * 64),
            "source": "registered:test/v1",
            "status": "applied",
            "observation": None,
        })


def test_trial_restore_rejects_unknown_root_and_event_fields() -> None:
    runner = TrialRunner(run_id="run-1", trial_id="trial-1")
    runner.transition(TrialState.COMPILED, {"scenario_instance_id": "scenario-1"})

    unknown_root = runner.get_state()
    unknown_root["ignored"] = True
    with pytest.raises(ValueError, match="unknown fields: ignored"):
        TrialRunner.from_state(unknown_root)

    unknown_event = runner.get_state()
    unknown_event["events"][0]["ignored"] = True
    with pytest.raises(ValueError, match="unknown fields: ignored"):
        TrialRunner.from_state(unknown_event)


def test_simultaneous_batch_boundaries_replay_and_project_both_calls() -> None:
    runner = TrialRunner(run_id="run-batch", trial_id="trial-batch")
    runner.transition(TrialState.COMPILED, {"scenario_instance_id": "scenario-1"})
    runner.transition(TrialState.AGENTS_BUILT, {"actors": ["seller", "buyer"]})
    runner.transition(TrialState.INITIALIZED, {"round_index": 0})
    runner.transition(TrialState.BATCH_PROPOSED, {
        "round_index": 0,
        "actor_ids": ["seller", "buyer"],
        "generation_call_ids": ["call-seller", "call-buyer"],
        "attempts": [0, 0],
    })
    runner.transition(TrialState.BATCH_CAPTURED, {
        "round_index": 0,
        "actor_ids": ["seller", "buyer"],
        "generation_record_ids": ["call-seller", "call-buyer"],
        "output_texts": ["I offer $70.", "I offer $60."],
    })
    runner.transition(TrialState.BATCH_ADJUDICATED, {
        "round_index": 0,
        "actor_ids": ["seller", "buyer"],
        "interaction_event_ids": ["event-seller", "event-buyer"],
        "resolution_ids": ["resolution-seller", "resolution-buyer"],
        "action_ids": ["action-seller", "action-buyer"],
        "accepted": [True, True],
        "label_record_ids": [["label-seller"], []],
    })
    runner.transition(TrialState.OBSERVED, {
        "actor_ids": ["seller", "buyer"],
        "observation": "Both simultaneous offers were committed.",
        "interaction_event_ids": ["event-seller", "event-buyer"],
        "label_record_ids": [["label-seller"], []],
    })

    restored = TrialRunner.from_state(json.loads(json.dumps(runner.get_state())))

    assert restored.get_state() == runner.get_state()
    captures = [
        row for row in restored.project_transcript()
        if row["state"] == "batch_captured"
    ]
    assert [row["actor_id"] for row in captures] == ["seller", "buyer"]
    assert [row["generation_record_id"] for row in captures] == [
        "call-seller", "call-buyer"
    ]


def test_simultaneous_batch_lineage_rejects_reordered_or_mismatched_calls() -> None:
    runner = TrialRunner(run_id="run-batch", trial_id="trial-batch")
    runner.transition(TrialState.COMPILED, {"scenario_instance_id": "scenario-1"})
    runner.transition(TrialState.AGENTS_BUILT, {"actors": ["seller", "buyer"]})
    runner.transition(TrialState.INITIALIZED, {"round_index": 0})
    runner.transition(TrialState.BATCH_PROPOSED, {
        "round_index": 0,
        "actor_ids": ["seller", "buyer"],
        "generation_call_ids": ["call-seller", "call-buyer"],
        "attempts": [0, 0],
    })

    with pytest.raises(ValueError, match="batch actors"):
        runner.transition(TrialState.BATCH_CAPTURED, {
            "round_index": 0,
            "actor_ids": ["buyer", "seller"],
            "generation_record_ids": ["call-buyer", "call-seller"],
            "output_texts": ["buyer", "seller"],
        })
    with pytest.raises(ValueError, match="proposed calls"):
        runner.transition(TrialState.BATCH_CAPTURED, {
            "round_index": 0,
            "actor_ids": ["seller", "buyer"],
            "generation_record_ids": ["call-other", "call-buyer"],
            "output_texts": ["seller", "buyer"],
        })


@pytest.mark.parametrize(
    ("from_state", "to_state", "required_field"),
    [
        (TrialState.CREATED, TrialState.COMPILED, "scenario_instance_id"),
        (TrialState.COMPILED, TrialState.AGENTS_BUILT, "actors"),
        (TrialState.AGENTS_BUILT, TrialState.INITIALIZED, "round_index"),
        (TrialState.INITIALIZED, TrialState.TURN_PROPOSED, "actor_id"),
        (TrialState.TURN_PROPOSED, TrialState.ACTION_CAPTURED, "actor_id"),
        (TrialState.INITIALIZED, TrialState.BATCH_PROPOSED, "round_index"),
        (TrialState.BATCH_PROPOSED, TrialState.BATCH_CAPTURED, "round_index"),
        (TrialState.BATCH_CAPTURED, TrialState.BATCH_ADJUDICATED, "round_index"),
        (TrialState.ACTION_CAPTURED, TrialState.ADJUDICATED, "actor_id"),
        (TrialState.ADJUDICATED, TrialState.OBSERVED, "actor_id"),
        (TrialState.OBSERVED, TrialState.COMPLETED, "reason"),
        (TrialState.CREATED, TrialState.FAILED, "error_type"),
    ],
)
def test_every_state_rejects_missing_required_payload(
    from_state,
    to_state,
    required_field,
) -> None:
    with pytest.raises(ValueError, match=required_field):
        TrialRuntimeEvent(
            run_id="run-1",
            trial_id="trial-1",
            attempt=0,
            sequence=0,
            from_state=from_state,
            to_state=to_state,
            payload={},
        )


def test_trial_lineage_rejects_cross_actor_round_call_and_event_references() -> None:
    runner = TrialRunner(run_id="run-1", trial_id="trial-1")
    runner.transition(TrialState.COMPILED, {"scenario_instance_id": "scenario-1"})
    runner.transition(TrialState.AGENTS_BUILT, {"actors": ["seller", "buyer"]})
    runner.transition(TrialState.INITIALIZED, {"round_index": 0})
    runner.transition(TrialState.TURN_PROPOSED, {
        "actor_id": "seller",
        "round_index": 0,
        "generation_call_id": "call-1",
        "attempt": 0,
    })

    with pytest.raises(ValueError, match="captured actor_id"):
        runner.transition(TrialState.ACTION_CAPTURED, {
            "actor_id": "buyer",
            "round_index": 0,
            "output_text": "offer",
            "generation_record_id": "call-1",
        })
    with pytest.raises(ValueError, match="generation record"):
        runner.transition(TrialState.ACTION_CAPTURED, {
            "actor_id": "seller",
            "round_index": 0,
            "output_text": "offer",
            "generation_record_id": "call-other",
        })
