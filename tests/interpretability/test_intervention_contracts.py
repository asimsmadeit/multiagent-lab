"""Contracts for probe and scripted-observation runtime interventions."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from interpretability.runtime.interventions import (
    INTERVENTION_APPLICATION_LOG_SCHEMA_VERSION,
    INTERVENTION_APPLICATION_SCHEMA_VERSION,
    INTERVENTION_DESIGN_SCHEMA_VERSION,
    INTERVENTION_SCHEDULE_SCHEMA_VERSION,
    PROBE_INTERVENTION_SCHEMA_VERSION,
    SCRIPTED_OBSERVATION_SCHEMA_VERSION,
    InterventionApplicationLog,
    InterventionApplicationReceipt,
    InterventionApplicationStatus,
    InterventionDesign,
    InterventionFamily,
    InterventionSchedule,
    ProbeInterventionPlan,
    ProbeInterventionSpec,
    ProbeKind,
    ProbeLabelStatus,
    ScriptedObservationKind,
    ScriptedObservationPlan,
    ScriptedObservationSpec,
    calculate_intervention_progress,
    content_sha256,
)


def _probe(**changes) -> ProbeInterventionPlan:
    values = {
        "run_id": "run-1",
        "trial_id": "trial-1",
        "scenario_instance_id": "scenario-instance-1",
        "kind": ProbeKind.BELIEF_VERIFICATION,
        "target_actor_id": "Negotiator",
        "scheduled_round": 0,
        "committed_action_boundary": 0,
        "sequence": 0,
        "enabled": True,
        "source": "registered:belief-verification/v1",
        "content": "State the private value you were given.",
    }
    values.update(changes)
    return ProbeInterventionPlan(**values)


def _observation(**changes) -> ScriptedObservationPlan:
    values = {
        "run_id": "run-1",
        "trial_id": "trial-1",
        "scenario_instance_id": "scenario-instance-1",
        "kind": ScriptedObservationKind.REGISTERED_TEMPLATE,
        "target_actor_id": "Negotiator",
        "scheduled_round": 1,
        "committed_action_boundary": 2,
        "sequence": 0,
        "enabled": True,
        "source": "registered:belief-shift/v1",
        "content": "The counterpart has received a competing offer.",
    }
    values.update(changes)
    return ScriptedObservationPlan(**values)


def _probe_spec(**changes) -> ProbeInterventionSpec:
    values = {
        "kind": ProbeKind.BELIEF_VERIFICATION,
        "target_actor_id": "Negotiator",
        "scheduled_round": 0,
        "committed_action_boundary": 0,
        "sequence": 0,
        "enabled": True,
        "source": "registered:belief-verification/v1",
        "content": "State the private value you were given.",
    }
    values.update(changes)
    return ProbeInterventionSpec(**values)


def _observation_spec(**changes) -> ScriptedObservationSpec:
    values = {
        "kind": ScriptedObservationKind.REGISTERED_TEMPLATE,
        "target_actor_id": "Negotiator",
        "scheduled_round": 1,
        "committed_action_boundary": 2,
        "sequence": 0,
        "enabled": True,
        "source": "registered:belief-shift/v1",
        "content": "The counterpart has received a competing offer.",
    }
    values.update(changes)
    return ScriptedObservationSpec(**values)


def _schedule(*plans) -> InterventionSchedule:
    return InterventionSchedule(
        run_id="run-1",
        trial_id="trial-1",
        scenario_instance_id="scenario-instance-1",
        plans=plans or (_probe(), _observation()),
    )


def _receipt(
    schedule: InterventionSchedule,
    plan=None,
) -> InterventionApplicationReceipt:
    selected = plan or next(
        item for item in schedule.plans
        if isinstance(item, ProbeInterventionPlan)
    )
    return InterventionApplicationReceipt.for_plan(
        schedule,
        selected,
        status=InterventionApplicationStatus.APPLIED,
        evidence_call_id=(
            "call-probe-1"
            if isinstance(selected, ProbeInterventionPlan)
            else None
        ),
        label_status=(
            ProbeLabelStatus.UNKNOWN
            if isinstance(selected, ProbeInterventionPlan)
            else ProbeLabelStatus.INAPPLICABLE
        ),
    )


def test_plan_schedule_receipt_and_log_json_round_trip() -> None:
    schedule = _schedule()
    probe = schedule.plans[0]
    receipt = _receipt(schedule, probe)
    application_log = InterventionApplicationLog.empty(schedule).append(receipt)

    restored_schedule = InterventionSchedule.from_dict(
        json.loads(json.dumps(schedule.to_dict()))
    )
    restored_receipt = InterventionApplicationReceipt.from_dict(
        json.loads(json.dumps(receipt.to_dict()))
    )
    restored_log = InterventionApplicationLog.from_dict(
        json.loads(json.dumps(application_log.to_dict()))
    )

    assert restored_schedule == schedule
    assert restored_receipt == receipt
    assert restored_log == application_log
    assert restored_schedule.plans[0] == ProbeInterventionPlan.from_dict(
        probe.to_dict()
    )
    assert restored_schedule.plans[1] == ScriptedObservationPlan.from_dict(
        schedule.plans[1].to_dict()
    )


def test_unbound_specs_and_design_json_round_trip() -> None:
    design = InterventionDesign(specs=(_observation_spec(), _probe_spec()))
    restored = InterventionDesign.from_dict(
        json.loads(json.dumps(design.to_dict()))
    )

    assert restored == design
    assert ProbeInterventionSpec.from_dict(
        _probe_spec().to_dict()
    ) == _probe_spec()
    assert ScriptedObservationSpec.from_dict(
        _observation_spec().to_dict()
    ) == _observation_spec()
    assert restored.specs[0].family is InterventionFamily.PROBE


def test_schema_versions_are_explicit_and_current_only() -> None:
    design = InterventionDesign(specs=(_probe_spec(), _observation_spec()))
    schedule = _schedule()
    receipt = _receipt(schedule)
    application_log = InterventionApplicationLog.empty(schedule).append(receipt)

    assert schedule.schema_version == INTERVENTION_SCHEDULE_SCHEMA_VERSION
    assert design.schema_version == INTERVENTION_DESIGN_SCHEMA_VERSION
    assert receipt.schema_version == INTERVENTION_APPLICATION_SCHEMA_VERSION
    assert (
        application_log.schema_version
        == INTERVENTION_APPLICATION_LOG_SCHEMA_VERSION
    )
    assert schedule.plans[0].schema_version == PROBE_INTERVENTION_SCHEMA_VERSION
    assert (
        schedule.plans[1].schema_version
        == SCRIPTED_OBSERVATION_SCHEMA_VERSION
    )

    for record_type, payload in (
        (InterventionDesign, design.to_dict()),
        (InterventionSchedule, schedule.to_dict()),
        (InterventionApplicationReceipt, receipt.to_dict()),
        (InterventionApplicationLog, application_log.to_dict()),
    ):
        payload["schema_version"] = "0.9.0"
        with pytest.raises(ValueError, match="schema_version"):
            record_type.from_dict(payload)


def test_content_hashes_use_exact_utf8_content() -> None:
    probe = _probe(content="café\n")
    assert probe.content_hash == content_sha256("café\n")
    assert probe.content_hash.startswith("sha256:")
    assert len(probe.content_hash) == len("sha256:") + 64
    assert _probe(content="café").design_id != probe.design_id


def test_frozen_records_isolate_mutable_constructor_inputs() -> None:
    probe = _probe()
    observation = _observation()
    mutable_plans = [observation, probe]
    schedule = _schedule(*mutable_plans)
    mutable_plans.clear()

    assert len(schedule.plans) == 2
    assert isinstance(schedule.plans, tuple)
    with pytest.raises(FrozenInstanceError):
        schedule.run_id = "changed"

    payload = schedule.to_dict()
    restored = InterventionSchedule.from_dict(payload)
    payload["plans"][0]["content"] = "mutated after restore"
    assert restored.to_dict() == schedule.to_dict()


def test_schedule_and_log_order_are_deterministic() -> None:
    probe = _probe()
    observation = _observation()
    forward = _schedule(probe, observation)
    reverse = _schedule(observation, probe)

    assert forward.plans == reverse.plans
    assert forward.schedule_id == reverse.schedule_id

    first_receipt = _receipt(forward, probe)
    second_receipt = _receipt(forward, observation)
    first_log = InterventionApplicationLog(
        run_id=forward.run_id,
        trial_id=forward.trial_id,
        scenario_instance_id=forward.scenario_instance_id,
        schedule_id=forward.schedule_id,
        receipts=(first_receipt, second_receipt),
    )
    second_log = replace(
        first_log,
        receipts=(second_receipt, first_receipt),
    )
    assert first_log.receipts == second_log.receipts
    assert first_log.log_id == second_log.log_id

    forward_design = InterventionDesign(
        specs=(_probe_spec(), _observation_spec())
    )
    reverse_design = InterventionDesign(
        specs=(_observation_spec(), _probe_spec())
    )
    assert forward_design.specs == reverse_design.specs
    assert forward_design.design_id == reverse_design.design_id


@pytest.mark.parametrize(
    "change",
    [
        {"kind": ProbeKind.PLAUSIBILITY},
        {"target_actor_id": "Counterpart"},
        {"scheduled_round": 2},
        {"committed_action_boundary": 3},
        {"sequence": 1},
        {"enabled": False},
        {"source": "custom:verification"},
        {"content": "A different scientific prompt."},
    ],
)
def test_unbound_design_identity_changes_with_trial_relevant_inputs(change) -> None:
    original = InterventionDesign(specs=(_probe_spec(),))
    changed = InterventionDesign(specs=(_probe_spec(**change),))
    assert changed.design_id != original.design_id


def test_binding_after_compilation_preserves_unbound_design_identity() -> None:
    design = InterventionDesign(specs=(_probe_spec(), _observation_spec()))

    first = design.bind(
        run_id="run-1",
        trial_id="compiled-trial-1",
        scenario_instance_id="compiled-scenario-1",
    )
    second = design.bind(
        run_id="run-2",
        trial_id="compiled-trial-2",
        scenario_instance_id="compiled-scenario-2",
    )

    assert first.intervention_design_id == design.design_id
    assert second.intervention_design_id == design.design_id
    assert tuple(item.spec_id for item in first.plans) == tuple(
        item.spec_id for item in design.specs
    )
    assert tuple(item.to_spec() for item in first.plans) == design.specs
    assert first.schedule_id != second.schedule_id
    assert tuple(item.design_id for item in first.plans) != tuple(
        item.design_id for item in second.plans
    )


@pytest.mark.parametrize(
    "change",
    [
        {"trial_id": "trial-2"},
        {"scenario_instance_id": "scenario-instance-2"},
        {"enabled": False},
        {"source": "custom:test"},
        {"scheduled_round": 3},
    ],
)
def test_probe_design_identity_changes_with_scientific_inputs(change) -> None:
    assert _probe(**change).design_id != _probe().design_id


def test_observation_design_binds_scenario_trial_source_kind_and_content() -> None:
    original = _observation()
    variants = (
        _observation(trial_id="trial-2"),
        _observation(scenario_instance_id="scenario-instance-2"),
        _observation(source="custom:operator"),
        _observation(kind=ScriptedObservationKind.CUSTOM),
        _observation(content="Different observation."),
    )
    assert all(item.design_id != original.design_id for item in variants)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.__setitem__("content", "tampered"), "content hash"),
        (lambda value: value.__setitem__("design_id", "wrong"), "design_id"),
        (lambda value: value.__setitem__("enabled", 1), "enabled"),
        (lambda value: value.__setitem__("scheduled_round", True), "scheduled_round"),
        (lambda value: value.__setitem__("kind", False), "probe kind"),
        (lambda value: value.__setitem__("unknown", "field"), "unknown fields"),
    ],
)
def test_probe_restore_rejects_tampering_unknown_fields_and_types(
    mutation,
    match,
) -> None:
    payload = _probe().to_dict()
    mutation(payload)
    with pytest.raises((TypeError, ValueError), match=match):
        ProbeInterventionPlan.from_dict(payload)


def test_observation_restore_rejects_hash_tampering_and_unknown_fields() -> None:
    payload = _observation().to_dict()
    payload["content_hash"] = "sha256:" + ("0" * 64)
    with pytest.raises(ValueError, match="content hash"):
        ScriptedObservationPlan.from_dict(payload)


def test_unbound_design_restore_rejects_tampering_and_duplicate_specs() -> None:
    spec = _probe_spec()
    with pytest.raises(ValueError, match="duplicate spec_ids"):
        InterventionDesign(specs=(spec, spec))

    payload = InterventionDesign(specs=(spec,)).to_dict()
    payload["specs"][0]["content"] = "tampered"
    with pytest.raises(ValueError, match="content hash"):
        InterventionDesign.from_dict(payload)

    payload = InterventionDesign(specs=(spec,)).to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        InterventionDesign.from_dict(payload)

    payload = _observation().to_dict()
    payload["extra"] = None
    with pytest.raises(ValueError, match="unknown fields"):
        ScriptedObservationPlan.from_dict(payload)


def test_schedule_rejects_duplicate_plans_and_ambiguous_order_slots() -> None:
    probe = _probe()
    with pytest.raises(ValueError, match="duplicate design_ids"):
        _schedule(probe, probe)

    same_slot = _observation(
        scheduled_round=probe.scheduled_round,
        committed_action_boundary=probe.committed_action_boundary,
        sequence=probe.sequence,
    )
    with pytest.raises(ValueError, match="unique sequence"):
        _schedule(probe, same_slot)


def test_receipt_has_stable_application_identity_and_exactly_once_log() -> None:
    schedule = _schedule()
    receipt = _receipt(schedule)
    same_application = _receipt(schedule)

    assert receipt.application_id == same_application.application_id
    assert receipt.receipt_id == same_application.receipt_id
    with pytest.raises(ValueError, match="duplicate receipt"):
        InterventionApplicationLog(
            run_id=schedule.run_id,
            trial_id=schedule.trial_id,
            scenario_instance_id=schedule.scenario_instance_id,
            schedule_id=schedule.schedule_id,
            receipts=(receipt, same_application),
        )
    with pytest.raises(ValueError, match="duplicate receipt"):
        InterventionApplicationLog.empty(schedule).append(receipt).append(receipt)

    conflicting_evidence = replace(receipt, evidence_call_id="call-probe-2")
    assert conflicting_evidence.receipt_id != receipt.receipt_id
    assert conflicting_evidence.application_id == receipt.application_id
    with pytest.raises(ValueError, match="duplicate application"):
        InterventionApplicationLog(
            run_id=schedule.run_id,
            trial_id=schedule.trial_id,
            scenario_instance_id=schedule.scenario_instance_id,
            schedule_id=schedule.schedule_id,
            receipts=(receipt, conflicting_evidence),
        )


def test_receipt_rejects_boolean_label_status_and_probe_without_evidence() -> None:
    schedule = _schedule()
    probe = schedule.plans[0]

    with pytest.raises(TypeError, match="inapplicable or unknown"):
        InterventionApplicationReceipt(
            run_id=schedule.run_id,
            trial_id=schedule.trial_id,
            scenario_instance_id=schedule.scenario_instance_id,
            schedule_id=schedule.schedule_id,
            design_id=probe.design_id,
            family=InterventionFamily.PROBE,
            status=InterventionApplicationStatus.APPLIED,
            applied_round=0,
            committed_action_boundary=0,
            evidence_call_id="call-probe",
            label_status=True,
        )
    with pytest.raises(ValueError, match="evidence_call_id"):
        InterventionApplicationReceipt.for_plan(
            schedule,
            probe,
            status=InterventionApplicationStatus.APPLIED,
            evidence_call_id=None,
            label_status=ProbeLabelStatus.UNKNOWN,
        )


def test_receipt_restore_requires_ids_and_rejects_tampering() -> None:
    schedule = _schedule()
    receipt = _receipt(schedule)

    payload = receipt.to_dict()
    del payload["receipt_id"]
    with pytest.raises(ValueError, match="missing fields"):
        InterventionApplicationReceipt.from_dict(payload)

    payload = receipt.to_dict()
    payload["evidence_call_id"] = "call-tampered"
    with pytest.raises(ValueError, match="receipt_id"):
        InterventionApplicationReceipt.from_dict(payload)

    payload = receipt.to_dict()
    payload["label_status"] = False
    with pytest.raises(TypeError, match="probe label status"):
        InterventionApplicationReceipt.from_dict(payload)


def test_disabled_plan_requires_explicit_skipped_receipt() -> None:
    disabled = _probe(enabled=False)
    schedule = _schedule(disabled)
    application_log = InterventionApplicationLog.empty(schedule)
    progress = calculate_intervention_progress(
        schedule,
        application_log,
        current_round=0,
        committed_action_boundary=0,
    )
    assert progress.disabled == (disabled,)
    assert not progress.pending

    receipt = InterventionApplicationReceipt.for_plan(
        schedule,
        disabled,
        status=InterventionApplicationStatus.SKIPPED_DISABLED,
        evidence_call_id=None,
        label_status=ProbeLabelStatus.INAPPLICABLE,
    )
    progress = calculate_intervention_progress(
        schedule,
        application_log.append(receipt),
        current_round=0,
        committed_action_boundary=0,
    )
    assert progress.skipped == (disabled,)
    assert not progress.disabled


def test_enabled_terminal_skip_is_explicit_and_has_no_model_evidence() -> None:
    plan = _observation()
    schedule = _schedule(plan)
    receipt = InterventionApplicationReceipt.for_plan(
        schedule,
        plan,
        status=InterventionApplicationStatus.SKIPPED_TERMINAL,
        evidence_call_id=None,
        label_status=ProbeLabelStatus.INAPPLICABLE,
    )

    restored = InterventionApplicationReceipt.from_dict(
        json.loads(json.dumps(receipt.to_dict()))
    )
    progress = calculate_intervention_progress(
        schedule,
        InterventionApplicationLog.empty(schedule).append(restored),
        current_round=0,
        committed_action_boundary=0,
    )

    assert restored == receipt
    assert progress.terminal_skipped == (plan,)
    assert not progress.applied
    assert not progress.future
    with pytest.raises(ValueError, match="disabled intervention"):
        InterventionApplicationReceipt.for_plan(
            _schedule(_probe(enabled=False)),
            _probe(enabled=False),
            status=InterventionApplicationStatus.SKIPPED_TERMINAL,
            evidence_call_id=None,
            label_status=ProbeLabelStatus.INAPPLICABLE,
        )


def test_progress_is_pure_resume_friendly_and_fails_missed_boundaries_closed() -> None:
    probe = _probe()
    observation = _observation()
    schedule = _schedule(observation, probe)
    empty = InterventionApplicationLog.empty(schedule)

    first = calculate_intervention_progress(
        schedule,
        empty,
        current_round=0,
        committed_action_boundary=0,
    )
    repeated = calculate_intervention_progress(
        schedule,
        empty,
        current_round=0,
        committed_action_boundary=0,
    )
    assert first == repeated
    assert first.pending == (probe,)
    assert first.future == (observation,)
    assert not first.applied

    after_probe = empty.append(_receipt(schedule, probe))
    resumed = calculate_intervention_progress(
        schedule,
        after_probe,
        current_round=1,
        committed_action_boundary=2,
    )
    assert resumed.applied == (probe,)
    assert resumed.pending == (observation,)

    missed = calculate_intervention_progress(
        schedule,
        after_probe,
        current_round=2,
        committed_action_boundary=3,
    )
    assert missed.overdue == (observation,)
    assert not missed.pending


def test_progress_rejects_receipt_for_wrong_boundary_or_schedule() -> None:
    schedule = _schedule()
    receipt = _receipt(schedule)
    wrong_boundary = replace(receipt, applied_round=1)
    application_log = InterventionApplicationLog(
        run_id=schedule.run_id,
        trial_id=schedule.trial_id,
        scenario_instance_id=schedule.scenario_instance_id,
        schedule_id=schedule.schedule_id,
        receipts=(wrong_boundary,),
    )
    with pytest.raises(ValueError, match="boundary"):
        calculate_intervention_progress(
            schedule,
            application_log,
            current_round=0,
            committed_action_boundary=0,
        )

    other = _schedule(_probe(content="other"))
    with pytest.raises(ValueError, match="does not belong"):
        calculate_intervention_progress(
            other,
            InterventionApplicationLog.empty(schedule),
            current_round=0,
            committed_action_boundary=0,
        )
