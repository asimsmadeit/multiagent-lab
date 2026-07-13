"""Hostile and round-trip tests for the public activation dataset boundary."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import click
from click.testing import CliRunner
import numpy as np
import pytest
import torch

from interpretability.core.dataset_builder import ActivationSample, DatasetBuilder
from interpretability.data import (
    load_activation_dataset,
    load_activation_recovery_checkpoint,
    save_activation_dataset,
    save_activation_recovery_checkpoint,
)
from interpretability.data.activation_dataset import _dataset_hash
from interpretability.data.activation_recovery import _recovery_hash
from interpretability.labels import (
    BehaviorTarget,
    LabelRecord,
    LabelSource,
    LabelStatus,
    LabelValue,
)
from interpretability.merge_parallel_results import (
    _merge_canonical_records,
    merge_parallel_activations,
)
from interpretability.runtime.model_call import (
    CallPurpose,
    CaptureMode,
    GenerationRecord,
    SamplingSettings,
    make_activation_artifact_refs,
    make_call_id,
)
from interpretability.runtime.interventions import (
    InterventionApplicationLog,
    InterventionApplicationReceipt,
    InterventionApplicationStatus,
    InterventionDesign,
    InterventionSchedule,
    ProbeInterventionSpec,
    ProbeKind,
    ProbeLabelStatus,
)
from negotiation.domain import ActionKind, NegotiationAction, Offer
from negotiation.game_master.adjudication import EventStatus, InteractionEvent


def _settings() -> SamplingSettings:
    return SamplingSettings(
        max_tokens=32,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        seed=17,
        do_sample=True,
    )


def _row(index: int) -> tuple[ActivationSample, GenerationRecord, InteractionEvent, LabelRecord]:
    trial_id = f"trial-{index}"
    is_deceptive = bool(index % 2)
    activation = torch.tensor(
        [float(index), 1.0, 2.0, 3.0], dtype=torch.float32
    )
    action = NegotiationAction(
        action_ref=f"action-{index}",
        actor_id="Negotiator",
        kind=ActionKind.OFFER,
        offer=Offer("Negotiator", "Counterpart", {"price": 70 + index}),
        raw_text=f"I offer ${70 + index} under the stated terms.",
    )
    event = InteractionEvent(
        negotiation_id=f"negotiation-{index}",
        action=action,
        status=EventStatus.COMMITTED,
        round_index=0,
        action_sequence=0,
        committed_turn_index=0,
        decisions=(),
        module_context={"scenario": "ultimatum_bluff"},
    )
    call_id = make_call_id(
        run_id="dataset-run",
        trial_id=trial_id,
        attempt=0,
        sequence=0,
        purpose=CallPurpose.ACTOR_ACTION,
        actor_id="Negotiator",
    )
    generation = GenerationRecord(
        call_id=call_id,
        run_id="dataset-run",
        trial_id=trial_id,
        attempt=0,
        sequence=0,
        actor_id="Negotiator",
        purpose=CallPurpose.ACTOR_ACTION,
        assembled_prompt=f"Private prompt {index}",
        input_token_ids=(1, 2),
        requested_sampling=_settings(),
        effective_sampling=_settings(),
        generation_path="offline_generation_pass",
        output_token_ids=(10 + index,),
        retained_token_ids=(10 + index,),
        output_text=action.raw_text,
        terminator=None,
        model_revision="offline-model@abc123",
        tokenizer_revision="offline-tokenizer@def456",
        concordia_version="2.4.0",
        capture_mode=CaptureMode.GENERATION_PASS,
        activation_position="last_retained_response_token",
        activation_artifacts=make_activation_artifact_refs(
            {"blocks.2.hook_resid_post": activation}, 0
        ),
        retained_token_index=0,
    )
    label = LabelRecord(
        subject_actor_id="Negotiator",
        behavior_target=BehaviorTarget.FACTUAL_DECEPTION,
        value=LabelValue.TRUE if is_deceptive else LabelValue.FALSE,
        status=LabelStatus.AVAILABLE,
        source=LabelSource.RULE,
        target_event_id=event.event_id,
        evaluation_event_id=None,
        evaluator_version="offline-rule/v1",
        evidence_event_ids=(event.event_id,),
        confidence=1.0,
    )
    sample = ActivationSample(
        trial_id=trial_id,
        round_num=0,
        agent_name="Negotiator",
        activations={"blocks.2.hook_resid_post": activation},
        actual_deception=float(is_deceptive),
        perceived_deception=None if index == 0 else 0.2,
        prompt=generation.assembled_prompt,
        response=generation.output_text,
        scenario_type="ultimatum_bluff",
        emergent_scenario="ultimatum_bluff",
        sample_type="negotiation",
        semantic_phase="discussion",
        experiment_mode="emergent",
        experiment_track="single_agent_white_box",
        sampling_config={
            "requested": generation.requested_sampling.to_dict(),
            "effective": generation.effective_sampling.to_dict(),
            "generation_path": generation.generation_path,
        },
        generation_record_id=generation.call_id,
        interaction_event_id=event.event_id,
        label_record_ids=[label.label_id],
        actual_deception_projection=float(is_deceptive),
        trial_family_id=f"family-{index}",
        scenario_instance_id=f"scenario-instance-{index}",
        role_assignment_id=f"role-{index}",
        order_assignment_id=f"order-{index}",
        counterpart_assignment_id=f"counterpart-{index}",
        surface_assignment_id=f"surface-{index}",
        counterbalance_id=f"counterbalance-{index}",
        first_mover_id="Negotiator",
        role_assignment={"actor": "seller", "counterpart": "buyer"},
        surface_assignment={"variant": f"surface-{index}"},
        counterpart_name="Counterpart",
        counterpart_idx=({0: 1, 1: 0}.get(index % 3)),
        tom_state={1: {"available": index > 0}},
        sae_features=({4: 1.5} if index % 3 == 1 else None),
        sae_top_features=([4] if index % 3 == 1 else None),
    )
    return sample, generation, event, label


def _builder(start: int = 0) -> DatasetBuilder:
    rows = [_row(start + index) for index in range(3)]
    builder = DatasetBuilder(
        generation_records=[row[1] for row in rows],
        interaction_events=[row[2] for row in rows],
        label_records=[row[3] for row in rows],
        experiment_track="single_agent_white_box",
        captured_actor_ids=("Negotiator",),
        split_seed=23,
    )
    for sample, *_ in rows:
        builder.add_sample(sample)
    return builder


def _intervention_builder(start: int = 0) -> DatasetBuilder:
    rows = [_row(start + index) for index in range(3)]
    design = InterventionDesign(specs=(ProbeInterventionSpec(
        kind=ProbeKind.BELIEF_VERIFICATION,
        target_actor_id="Negotiator",
        scheduled_round=0,
        committed_action_boundary=0,
        sequence=0,
        enabled=True,
        source="registered:belief-verification/v1",
        content="State your current belief before negotiating.",
    ),))
    generation_records = []
    samples = []
    schedules = []
    application_logs = []
    for index, (sample, action_generation, _event, _label) in enumerate(rows):
        sample.counterpart_idx = None
        schedule = design.bind(
            run_id="dataset-run",
            trial_id=str(sample.trial_id),
            scenario_instance_id=str(sample.scenario_instance_id),
        )
        probe_call_id = make_call_id(
            run_id="dataset-run",
            trial_id=str(sample.trial_id),
            attempt=0,
            sequence=1,
            purpose=CallPurpose.BELIEF_VERIFICATION,
            actor_id="Negotiator",
        )
        probe_text = f"My pre-negotiation belief for trial {index}."
        probe_generation = GenerationRecord(
            call_id=probe_call_id,
            run_id="dataset-run",
            trial_id=str(sample.trial_id),
            attempt=0,
            sequence=1,
            actor_id="Negotiator",
            purpose=CallPurpose.BELIEF_VERIFICATION,
            assembled_prompt=design.specs[0].content,
            input_token_ids=(1, 2),
            requested_sampling=_settings(),
            effective_sampling=_settings(),
            generation_path="offline_generation_pass",
            output_token_ids=(40 + index,),
            retained_token_ids=(40 + index,),
            output_text=probe_text,
            terminator=None,
            model_revision="offline-model@abc123",
            tokenizer_revision="offline-tokenizer@def456",
            concordia_version="2.4.0",
            capture_mode=CaptureMode.GENERATION_PASS,
            activation_position="last_retained_response_token",
            activation_artifacts=make_activation_artifact_refs(
                sample.activations, 0
            ),
            retained_token_index=0,
        )
        receipt = InterventionApplicationReceipt.for_plan(
            schedule,
            schedule.plans[0],
            status=InterventionApplicationStatus.APPLIED,
            evidence_call_id=probe_generation.call_id,
            label_status=ProbeLabelStatus.UNKNOWN,
        )
        application_log = InterventionApplicationLog.empty(schedule).append(
            receipt
        )
        receipt_ids = [item.receipt_id for item in application_log.receipts]
        sample.intervention_design_id = design.design_id
        sample.intervention_application_receipt_ids = receipt_ids

        probe_sample = copy.deepcopy(sample)
        probe_sample.round_num = -1
        probe_sample.actual_deception = None
        probe_sample.perceived_deception = None
        probe_sample.emergent_ground_truth = None
        probe_sample.actual_deception_projection = None
        probe_sample.prompt = probe_generation.assembled_prompt
        probe_sample.response = probe_generation.output_text
        probe_sample.sample_type = "pre_verification"
        probe_sample.semantic_phase = ProbeKind.BELIEF_VERIFICATION.value
        probe_sample.generation_record_id = probe_generation.call_id
        probe_sample.interaction_event_id = None
        probe_sample.label_record_ids = []
        probe_sample.is_verification_probe = True
        probe_sample.plausibility_response = None
        probe_sample.counterpart_idx = None

        generation_records.extend((action_generation, probe_generation))
        samples.extend((sample, probe_sample))
        schedules.append(schedule)
        application_logs.append(application_log)

    builder = DatasetBuilder(
        generation_records=generation_records,
        interaction_events=[row[2] for row in rows],
        label_records=[row[3] for row in rows],
        intervention_designs=(design, design),
        intervention_schedules=schedules,
        intervention_application_logs=application_logs,
        experiment_track="single_agent_white_box",
        captured_actor_ids=("Negotiator",),
        split_seed=23,
    )
    for sample in samples:
        builder.add_sample(sample)
    return builder


def _analysis_builder() -> DatasetBuilder:
    rows = [_row(100 + index) for index in range(12)]
    for index, row in enumerate(rows):
        row[0].trial_family_id = f"analysis-family-{index // 4}"
        row[0].sae_features = None
        row[0].sae_top_features = None
    builder = DatasetBuilder(
        generation_records=[row[1] for row in rows],
        interaction_events=[row[2] for row in rows],
        label_records=[row[3] for row in rows],
        experiment_track="single_agent_white_box",
        captured_actor_ids=("Negotiator",),
        split_seed=23,
    )
    for sample, *_ in rows:
        builder.add_sample(sample)
    return builder


def _save(builder: DatasetBuilder, path: Path) -> Path:
    return builder.save(
        path,
        model_name="offline-model",
        model_revision="offline-model@abc123",
        tokenizer_name="offline-tokenizer",
        tokenizer_revision="offline-tokenizer@def456",
    )


def test_recovery_restores_typed_canonical_records_for_republication(
    tmp_path: Path,
) -> None:
    rows = [_row(index) for index in range(3)]
    sample, generation, event, label = rows[0]
    path = save_activation_recovery_checkpoint(
        tmp_path / "recovery.json",
        samples=[row[0] for row in rows],
        generation_records=[row[1] for row in rows],
        interaction_events=[row[2] for row in rows],
        label_records=[row[3] for row in rows],
        experiment_track="single_agent_white_box",
        captured_actor_ids=("Negotiator",),
        pod_id=0,
        trial_id_offset=0,
        current_trial_id=1,
        reason="awaiting more split components",
    )

    restored = load_activation_recovery_checkpoint(path)

    assert isinstance(restored["generation_records"][0], GenerationRecord)
    assert isinstance(restored["interaction_events"][0], InteractionEvent)
    assert isinstance(restored["label_records"][0], LabelRecord)
    assert restored["generation_records"][0].capture_mode is CaptureMode.GENERATION_PASS
    assert restored["activation_samples"][0].generation_record_id == generation.call_id

    builder = DatasetBuilder(
        generation_records=restored["generation_records"],
        interaction_events=restored["interaction_events"],
        label_records=restored["label_records"],
        experiment_track="single_agent_white_box",
        captured_actor_ids=("Negotiator",),
    )
    for restored_sample in restored["activation_samples"]:
        builder.add_sample(restored_sample)
    republished = load_activation_dataset(
        _save(builder, tmp_path / "republished.json")
    )
    assert republished["config"]["provenance"]["capture"][
        "capture_modes"
    ] == ["generation_pass"]


def test_text_only_recovery_allows_zero_capture_manifest_and_rows(
    tmp_path: Path,
) -> None:
    path = save_activation_recovery_checkpoint(
        tmp_path / "text-only-recovery.json",
        samples=(),
        experiment_track="text_only",
        captured_actor_ids=(),
        pod_id=0,
        trial_id_offset=0,
        current_trial_id=0,
        reason="text-only transcript recovery",
    )

    restored = load_activation_recovery_checkpoint(path)
    assert restored["activation_samples"] == []
    assert restored["runner_state"]["captured_actor_ids"] == []
    with pytest.raises(ValueError, match="cannot declare captured"):
        save_activation_recovery_checkpoint(
            tmp_path / "bad-text-only.json",
            samples=(),
            experiment_track="text_only",
            captured_actor_ids=("Negotiator",),
            pod_id=0,
            trial_id_offset=0,
            current_trial_id=0,
            reason="invalid",
        )


def test_multitrial_intervention_recovery_restores_publishable_lineage(
    tmp_path: Path,
) -> None:
    from interpretability.evaluation import InterpretabilityRunner
    from interpretability.tracks import ExperimentTrack

    source = _intervention_builder()
    runner = object.__new__(InterpretabilityRunner)
    runner.model = SimpleNamespace(
        model_name="offline-model@abc123",
        tokenizer=SimpleNamespace(name_or_path="offline-tokenizer@def456"),
    )
    runner.activation_samples = list(source.samples)
    runner.generation_records = list(source.generation_records)
    runner.label_records = list(source.label_records)
    runner.interaction_events = list(source.interaction_events)
    runner.intervention_designs = list(source.intervention_designs)
    runner.intervention_schedules = list(source.intervention_schedules)
    runner.intervention_application_logs = list(
        source.intervention_application_logs
    )
    runner.experiment_track = ExperimentTrack.SINGLE_AGENT_WHITE_BOX
    runner.captured_actor_ids = ("Negotiator",)
    runner._pod_id = 0
    runner._trial_id_offset = 0
    runner._trial_id = 3

    recovery_path = runner.write_activation_recovery(
        tmp_path / "multi-trial-interventions.json",
        reason="offline resume test",
    )
    restored = load_activation_recovery_checkpoint(recovery_path)

    assert len(restored["intervention_designs"]) == 1
    assert len(restored["intervention_schedules"]) == 3
    assert len(restored["intervention_application_logs"]) == 3
    assert all(
        isinstance(item, InterventionDesign)
        for item in restored["intervention_designs"]
    )
    assert all(
        isinstance(item, InterventionSchedule)
        for item in restored["intervention_schedules"]
    )
    assert all(
        isinstance(item, InterventionApplicationLog)
        for item in restored["intervention_application_logs"]
    )

    resumed = object.__new__(InterpretabilityRunner)
    resumed.model = runner.model
    resumed.restore_activation_checkpoint(recovery_path)
    assert len(resumed.intervention_schedules) == 3
    published_path = resumed.save_dataset(
        str(tmp_path / "resumed-publication.json")
    )
    published = load_activation_dataset(published_path)
    assert len(published["intervention_designs"]) == 1
    assert len(published["intervention_schedules"]) == 3
    assert len(published["intervention_application_logs"]) == 3

    suffix_sample = copy.deepcopy(source.samples[0])
    suffix_sample.trial_id = "legacy-suffix"
    suffix_sample.trial_family_id = "legacy-suffix-family"
    suffix_sample.scenario_instance_id = None
    suffix_sample.generation_record_id = None
    suffix_sample.interaction_event_id = None
    suffix_sample.label_record_ids = []
    suffix_sample.intervention_design_id = None
    suffix_sample.intervention_application_receipt_ids = []
    suffix_sample.sample_type = "unclassified"
    runner.activation_samples.append(suffix_sample)
    suffix_path = runner.write_activation_recovery(
        tmp_path / "legacy-suffix.json",
        sample_start=len(source.samples),
        generation_start=len(source.generation_records),
        label_start=len(source.label_records),
        event_start=len(source.interaction_events),
        reason="separate non-headline suffix",
    )
    suffix = load_activation_recovery_checkpoint(suffix_path)
    assert suffix["intervention_designs"] == []
    assert suffix["intervention_schedules"] == []
    assert suffix["intervention_application_logs"] == []


def test_text_only_recovery_retains_uncaptured_probe_intervention_graph(
    tmp_path: Path,
) -> None:
    design = InterventionDesign(specs=(ProbeInterventionSpec(
        kind=ProbeKind.BELIEF_VERIFICATION,
        target_actor_id="Negotiator",
        scheduled_round=0,
        committed_action_boundary=0,
        sequence=0,
        enabled=True,
        source="registered:text-only-belief/v1",
        content="State your current belief before negotiating.",
    ),))
    schedule = design.bind(
        run_id="text-only-run",
        trial_id="text-only-trial",
        scenario_instance_id="text-only-instance",
    )
    call_id = make_call_id(
        run_id="text-only-run",
        trial_id="text-only-trial",
        attempt=0,
        sequence=0,
        purpose=CallPurpose.BELIEF_VERIFICATION,
        actor_id="Negotiator",
    )
    generation = GenerationRecord(
        call_id=call_id,
        run_id="text-only-run",
        trial_id="text-only-trial",
        attempt=0,
        sequence=0,
        actor_id="Negotiator",
        purpose=CallPurpose.BELIEF_VERIFICATION,
        assembled_prompt=design.specs[0].content,
        input_token_ids=(1, 2),
        requested_sampling=_settings(),
        effective_sampling=_settings(),
        generation_path="offline_text_only",
        output_token_ids=(7,),
        retained_token_ids=(7,),
        output_text="My current belief is uncertain.",
        terminator=None,
        model_revision="offline-model@abc123",
        tokenizer_revision="offline-tokenizer@def456",
        concordia_version="2.4.0",
        capture_mode=CaptureMode.NONE,
        retained_token_index=0,
    )
    receipt = InterventionApplicationReceipt.for_plan(
        schedule,
        schedule.plans[0],
        status=InterventionApplicationStatus.APPLIED,
        evidence_call_id=generation.call_id,
        label_status=ProbeLabelStatus.UNKNOWN,
    )
    application_log = InterventionApplicationLog.empty(schedule).append(
        receipt
    )

    path = save_activation_recovery_checkpoint(
        tmp_path / "text-only-intervention.json",
        samples=(),
        generation_records=(generation,),
        intervention_designs=(design,),
        intervention_schedules=(schedule,),
        intervention_application_logs=(application_log,),
        experiment_track="text_only",
        captured_actor_ids=(),
        pod_id=0,
        trial_id_offset=0,
        current_trial_id=0,
        reason="text-only intervention evidence",
    )
    restored = load_activation_recovery_checkpoint(path)

    assert restored["activation_samples"] == []
    assert restored["generation_records"][0].capture_mode is CaptureMode.NONE
    assert restored["intervention_designs"] == [design]
    assert restored["intervention_schedules"] == [schedule]
    assert restored["intervention_application_logs"] == [application_log]


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("design_content", "content hash"),
        ("missing_schedule_log", "without a trial schedule"),
        ("row_receipts", "receipts disagree"),
        ("unknown_log_field", "unknown fields"),
    ],
)
def test_multitrial_intervention_recovery_revalidates_after_rehash(
    tmp_path: Path,
    tamper: str,
    match: str,
) -> None:
    source = _intervention_builder()
    path = save_activation_recovery_checkpoint(
        tmp_path / f"recovery-intervention-{tamper}.json",
        samples=source.samples,
        generation_records=source.generation_records,
        interaction_events=source.interaction_events,
        label_records=source.label_records,
        intervention_designs=source.intervention_designs,
        intervention_schedules=source.intervention_schedules,
        intervention_application_logs=source.intervention_application_logs,
        experiment_track="single_agent_white_box",
        captured_actor_ids=("Negotiator",),
        pod_id=0,
        trial_id_offset=0,
        current_trial_id=3,
        reason="hostile aggregate fixture",
    )

    def mutate(manifest):
        if tamper == "design_content":
            manifest["intervention_designs"][0]["specs"][0][
                "content"
            ] = "tampered recovery content"
        elif tamper == "missing_schedule_log":
            removed = manifest["intervention_schedules"].pop()
            manifest["intervention_application_logs"] = [
                item
                for item in manifest["intervention_application_logs"]
                if item["schedule_id"] != removed["schedule_id"]
            ]
        elif tamper == "row_receipts":
            manifest["samples"][0][
                "intervention_application_receipt_ids"
            ] = []
        else:
            manifest["intervention_application_logs"][0][
                "future_field"
            ] = None

    _rewrite_recovery_bundle(path, mutate)
    with pytest.raises(ValueError, match=match):
        load_activation_recovery_checkpoint(path)
    with pytest.raises(ValueError, match="requires captured"):
        save_activation_recovery_checkpoint(
            tmp_path / "bad-white-box.json",
            samples=(),
            experiment_track="single_agent_white_box",
            captured_actor_ids=(),
            pod_id=0,
            trial_id_offset=0,
            current_trial_id=0,
            reason="invalid",
        )


def _rewrite_bundle(
    manifest_path: Path,
    mutate,
) -> None:
    outer = json.loads(manifest_path.read_text(encoding="utf-8"))
    array_path = manifest_path.with_name(outer["array_file"])
    with np.load(array_path, allow_pickle=False) as bundle:
        arrays = {name: np.array(bundle[name], copy=True) for name in bundle.files}
    mutate(outer, outer["manifest"], arrays)
    np.savez_compressed(array_path, **arrays)
    outer["array_sha256"] = hashlib.sha256(array_path.read_bytes()).hexdigest()
    outer["arrays"] = {
        name: {"shape": list(array.shape), "dtype": str(array.dtype)}
        for name, array in arrays.items()
    }
    outer["manifest"]["dataset_hash"] = _dataset_hash(
        arrays, outer["manifest"]
    )
    manifest_path.write_text(
        json.dumps(outer, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _rewrite_recovery_bundle(manifest_path: Path, mutate) -> None:
    outer = json.loads(manifest_path.read_text(encoding="utf-8"))
    array_path = manifest_path.with_name(outer["array_file"])
    with np.load(array_path, allow_pickle=False) as bundle:
        arrays = {name: np.array(bundle[name], copy=True) for name in bundle.files}
    mutate(outer["manifest"])
    outer["manifest"]["recovery_hash"] = _recovery_hash(
        outer["manifest"], arrays
    )
    manifest_path.write_text(
        json.dumps(outer, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_safe_round_trip_preserves_lineage_split_and_provenance(tmp_path: Path) -> None:
    manifest_path = _save(_builder(), tmp_path / "activations.json")
    data = load_activation_dataset(manifest_path)

    assert manifest_path.suffix == ".json"
    assert manifest_path.with_suffix(".npz").exists()
    assert data["activations"][2].shape == (3, 4)
    assert data["labels"]["agent_labels"][0] is None
    assert set(data["labels"]["split_partitions"]) == {
        "train", "development", "test"
    }
    assert data["split_manifest"]["locked"] is True
    assert len(set(data["labels"]["connected_group_ids"])) == 3
    assert data["metadata"][0]["generation_record_id"].startswith("call_")
    assert data["metadata"][0]["interaction_event_id"].startswith("event_")
    assert data["metadata"][0]["trial_family_id"] == "family-0"
    assert data["metadata"][0]["role_assignment_id"] == "role-0"
    assert data["metadata"][0]["tom_state"]["1"]["available"] is False
    assert data["config"]["provenance"]["model"]["revision"].endswith("abc123")
    assert data["config"]["provenance"]["capture"]["capture_modes"] == [
        "generation_pass"
    ]
    assert data["sae_available_mask"] == [False, True, False]
    assert data["sae_top_features"] == [[], [4], []]


def test_public_dataset_embeds_complete_intervention_lineage(
    tmp_path: Path,
) -> None:
    manifest_path = _save(
        _intervention_builder(),
        tmp_path / "intervention-lineage.json",
    )
    data = load_activation_dataset(manifest_path)

    assert data["config"]["dataset_schema_version"] == "4.1.0"
    assert len(data["intervention_designs"]) == 1
    assert len(data["intervention_schedules"]) == 3
    assert len(data["intervention_application_logs"]) == 3
    design = InterventionDesign.from_dict(data["intervention_designs"][0])
    schedules = {
        item.trial_id: item
        for item in map(
            InterventionSchedule.from_dict,
            data["intervention_schedules"],
        )
    }
    logs = {
        item.trial_id: item
        for item in map(
            InterventionApplicationLog.from_dict,
            data["intervention_application_logs"],
        )
    }
    generation_by_id = {
        item["call_id"]: GenerationRecord.from_dict(item)
        for item in data["generation_records"]
    }
    probe_rows = [
        row for row in data["metadata"]
        if row["sample_type"] == "pre_verification"
    ]

    assert len(probe_rows) == 3
    for row in data["metadata"]:
        schedule = schedules[str(row["trial_id"])]
        application_log = logs[str(row["trial_id"])]
        assert schedule.intervention_design_id == design.design_id
        assert row["intervention_design_id"] == design.design_id
        assert row["intervention_application_receipt_ids"] == [
            receipt.receipt_id for receipt in application_log.receipts
        ]
    for row in probe_rows:
        application_log = logs[str(row["trial_id"])]
        receipt = application_log.receipts[0]
        generation = generation_by_id[receipt.evidence_call_id]
        assert generation.call_id == row["generation_record_id"]
        assert generation.purpose is CallPurpose.BELIEF_VERIFICATION
        assert generation.actor_id == row["agent_name"]
        assert row["interaction_event_id"] is None
        assert row["label_record_ids"] == []
        assert row["actual_deception"] is None
        assert row["is_verification_probe"] is True


def test_public_dataset_rejects_unreceipted_terminal_plan(tmp_path: Path) -> None:
    manifest_path = _save(
        _intervention_builder(),
        tmp_path / "missing-terminal-receipt.json",
    )

    def remove_receipt(_outer, manifest, _arrays):
        schedule = InterventionSchedule.from_dict(
            manifest["intervention_schedules"][0]
        )
        old_log = InterventionApplicationLog.from_dict(
            next(
                raw_log
                for raw_log in manifest["intervention_application_logs"]
                if raw_log["schedule_id"] == schedule.schedule_id
            )
        )
        empty_log = InterventionApplicationLog.empty(schedule)
        manifest["intervention_application_logs"] = [
            empty_log.to_dict()
            if raw_log["log_id"] == old_log.log_id else raw_log
            for raw_log in manifest["intervention_application_logs"]
        ]
        for row in manifest["metadata"]:
            if str(row["trial_id"]) == schedule.trial_id:
                row["intervention_application_receipt_ids"] = []

    _rewrite_bundle(manifest_path, remove_receipt)
    with pytest.raises(ValueError, match="terminal receipt per plan"):
        load_activation_dataset(manifest_path)


def test_interpretability_runner_publishes_aggregated_intervention_records(
    tmp_path: Path,
) -> None:
    from interpretability.evaluation import InterpretabilityRunner

    source = _intervention_builder()
    runner = object.__new__(InterpretabilityRunner)
    runner.model = SimpleNamespace(
        model_name="offline-model@abc123",
        tokenizer=SimpleNamespace(name_or_path="offline-tokenizer@def456"),
    )
    runner.activation_samples = list(source.samples)
    runner.generation_records = list(source.generation_records)
    runner.label_records = list(source.label_records)
    runner.interaction_events = list(source.interaction_events)
    runner.intervention_designs = list(source.intervention_designs)
    runner.intervention_schedules = list(source.intervention_schedules)
    runner.intervention_application_logs = list(
        source.intervention_application_logs
    )
    runner.experiment_track = "single_agent_white_box"
    runner.captured_actor_ids = ("Negotiator",)
    runner._pod_id = 0
    runner._trial_id_offset = 0

    published = load_activation_dataset(InterpretabilityRunner.save_dataset(
        runner,
        str(tmp_path / "runner-interventions.json"),
    ))

    assert len(published["intervention_designs"]) == 1
    assert len(published["intervention_schedules"]) == 3
    assert len(published["intervention_application_logs"]) == 3


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("row_design", "design disagrees"),
        ("row_receipts", "receipts disagree"),
        ("design_content", "content hash"),
        ("probe_evidence", "probe GenerationRecords do not exactly match"),
        ("probe_actor", "does not exactly match its scheduled plan"),
    ],
)
def test_public_dataset_revalidates_intervention_graph_after_rehash(
    tmp_path: Path,
    tamper: str,
    match: str,
) -> None:
    manifest_path = _save(
        _intervention_builder(),
        tmp_path / f"intervention-{tamper}.json",
    )

    def mutate(_outer, manifest, _arrays):
        if tamper == "row_design":
            manifest["metadata"][0]["intervention_design_id"] = "wrong-design"
        elif tamper == "row_receipts":
            manifest["metadata"][0][
                "intervention_application_receipt_ids"
            ] = []
        elif tamper == "design_content":
            manifest["intervention_designs"][0]["specs"][0][
                "content"
            ] = "tampered content"
        else:
            schedule = InterventionSchedule.from_dict(
                manifest["intervention_schedules"][0]
            )
            old_log = InterventionApplicationLog.from_dict(
                next(
                    raw_log
                    for raw_log in manifest["intervention_application_logs"]
                    if raw_log["schedule_id"] == schedule.schedule_id
                )
            )
            if tamper == "probe_evidence":
                evidence_call_id = next(
                    record["call_id"]
                    for record in manifest["generation_records"]
                    if record["trial_id"] == schedule.trial_id
                    and record["purpose"] == CallPurpose.ACTOR_ACTION.value
                )
            else:
                old_probe_payload = next(
                    record
                    for record in manifest["generation_records"]
                    if record["trial_id"] == schedule.trial_id
                    and record["purpose"]
                    == CallPurpose.BELIEF_VERIFICATION.value
                )
                old_probe = GenerationRecord.from_dict(old_probe_payload)
                wrong_actor_call = replace(
                    old_probe,
                    call_id=make_call_id(
                        run_id=old_probe.run_id,
                        trial_id=old_probe.trial_id,
                        attempt=old_probe.attempt,
                        sequence=old_probe.sequence,
                        purpose=old_probe.purpose,
                        actor_id="Counterpart",
                    ),
                    actor_id="Counterpart",
                )
                manifest["generation_records"] = [
                    wrong_actor_call.to_dict()
                    if record["call_id"] == old_probe.call_id else record
                    for record in manifest["generation_records"]
                ]
                for row in manifest["metadata"]:
                    if row.get("generation_record_id") == old_probe.call_id:
                        row["generation_record_id"] = wrong_actor_call.call_id
                evidence_call_id = wrong_actor_call.call_id
            wrong_receipt = InterventionApplicationReceipt.for_plan(
                schedule,
                schedule.plans[0],
                status=InterventionApplicationStatus.APPLIED,
                evidence_call_id=evidence_call_id,
                label_status=ProbeLabelStatus.UNKNOWN,
            )
            wrong_log = InterventionApplicationLog(
                run_id=old_log.run_id,
                trial_id=old_log.trial_id,
                scenario_instance_id=old_log.scenario_instance_id,
                schedule_id=old_log.schedule_id,
                receipts=(wrong_receipt,),
            )
            manifest["intervention_application_logs"] = [
                wrong_log.to_dict()
                if raw_log["schedule_id"] == schedule.schedule_id
                else raw_log
                for raw_log in manifest["intervention_application_logs"]
            ]
            for row in manifest["metadata"]:
                if str(row["trial_id"]) == schedule.trial_id:
                    row["intervention_application_receipt_ids"] = [
                        wrong_receipt.receipt_id
                    ]

    _rewrite_bundle(manifest_path, mutate)
    with pytest.raises(ValueError, match=match):
        load_activation_dataset(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_manifest", "missing fields: merge_info"),
        ("unknown_manifest", "unknown fields: future_field"),
        ("missing_layer", "missing fields: source_dtype"),
        ("unknown_layer", "unknown fields: future_field"),
    ],
)
def test_activation_dataset_restore_requires_exact_current_manifest(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    manifest_path = _save(_builder(), tmp_path / f"strict-{mutation}.json")

    def mutate(_outer, manifest, _arrays):
        if mutation == "missing_manifest":
            del manifest["merge_info"]
        elif mutation == "unknown_manifest":
            manifest["future_field"] = None
        elif mutation == "missing_layer":
            del manifest["layer_arrays"][0]["source_dtype"]
        else:
            manifest["layer_arrays"][0]["future_field"] = None

    _rewrite_bundle(manifest_path, mutate)
    with pytest.raises(ValueError, match=match):
        load_activation_dataset(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_manifest", "missing=.*experiment_progress"),
        ("unknown_manifest", "unknown=.*future_field"),
        ("missing_runner", "missing=.*current_trial_id"),
        ("unknown_runner", "unknown=.*future_field"),
        ("missing_sample", "sample fields are malformed"),
        ("unknown_sample", "sample fields are malformed"),
    ],
)
def test_activation_recovery_restore_requires_exact_current_schema(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    rows = [_row(index) for index in range(3)]
    manifest_path = save_activation_recovery_checkpoint(
        tmp_path / f"recovery-{mutation}.json",
        samples=[row[0] for row in rows],
        generation_records=[row[1] for row in rows],
        interaction_events=[row[2] for row in rows],
        label_records=[row[3] for row in rows],
        experiment_track="single_agent_white_box",
        captured_actor_ids=("Negotiator",),
        pod_id=0,
        trial_id_offset=0,
        current_trial_id=1,
        reason="strict fixture",
    )

    def mutate(manifest):
        if mutation == "missing_manifest":
            del manifest["experiment_progress"]
        elif mutation == "unknown_manifest":
            manifest["future_field"] = None
        elif mutation == "missing_runner":
            del manifest["runner_state"]["current_trial_id"]
        elif mutation == "unknown_runner":
            manifest["runner_state"]["future_field"] = None
        elif mutation == "missing_sample":
            del manifest["samples"][0]["timestamp"]
        else:
            manifest["samples"][0]["future_field"] = None

    _rewrite_recovery_bundle(manifest_path, mutate)
    with pytest.raises(ValueError, match=match):
        load_activation_recovery_checkpoint(manifest_path)


def test_safe_publication_rejects_legacy_and_incomplete_provenance(tmp_path: Path) -> None:
    builder = _builder()
    with pytest.raises(PermissionError, match="trusted_legacy=True"):
        builder.save(tmp_path / "unsafe.pt")
    legacy_path = builder.save(
        tmp_path / "reviewed.pt",
        trusted_legacy=True,
    )
    assert legacy_path.suffix == ".pt"
    with pytest.raises(PermissionError, match="trusted=True"):
        load_activation_dataset(legacy_path)
    assert load_activation_dataset(legacy_path, trusted_legacy=True)["activations"]

    dataset = builder.build(
        model_name="offline-model",
        model_revision="offline-model@abc123",
        tokenizer_name="offline-tokenizer",
        tokenizer_revision="offline-tokenizer@def456",
    )
    dataset["config"]["provenance"]["model"]["revision"] = "unresolved"
    with pytest.raises(ValueError, match="unresolved"):
        save_activation_dataset(tmp_path / "unresolved.json", dataset)


def test_dataset_hash_is_stable_across_mapping_insertion_order(tmp_path: Path) -> None:
    dataset = _builder().build(
        model_name="offline-model",
        model_revision="offline-model@abc123",
        tokenizer_name="offline-tokenizer",
        tokenizer_revision="offline-tokenizer@def456",
    )
    reordered = copy.deepcopy(dataset)
    reordered["labels"] = dict(reversed(list(reordered["labels"].items())))
    reordered["config"] = dict(reversed(list(reordered["config"].items())))
    reordered["activations"] = dict(
        reversed(list(reordered["activations"].items()))
    )
    first = save_activation_dataset(tmp_path / "first.json", dataset)[1]
    second = save_activation_dataset(tmp_path / "second.json", reordered)[1]
    assert load_activation_dataset(first)["config"]["dataset_hash"] == (
        load_activation_dataset(second)["config"]["dataset_hash"]
    )


@pytest.mark.parametrize("tamper", ["missing_id", "prompt_content"])
def test_loader_restores_and_revalidates_canonical_records(
    tmp_path: Path,
    tamper: str,
) -> None:
    manifest_path = _save(_builder(), tmp_path / f"{tamper}.json")

    def mutate(_outer, manifest, _arrays):
        record = manifest["generation_records"][0]
        if tamper == "missing_id":
            del record["call_id"]
        else:
            record["assembled_prompt"] = "tampered prompt"

    _rewrite_bundle(manifest_path, mutate)
    with pytest.raises(ValueError, match="call_id|prompt_hash"):
        load_activation_dataset(manifest_path)


@pytest.mark.parametrize(
    ("collection", "match"),
    [
        ("generation_records", "generation record has unknown fields"),
        ("label_records", "label record has unknown fields"),
        ("interaction_events", "interaction event has unknown fields"),
    ],
)
def test_dataset_rejects_rehashed_unknown_canonical_record_fields(
    tmp_path: Path,
    collection: str,
    match: str,
) -> None:
    manifest_path = _save(_builder(), tmp_path / f"unknown-{collection}.json")

    def mutate(_outer, manifest, _arrays):
        manifest[collection][0]["future_field"] = None

    _rewrite_bundle(manifest_path, mutate)
    with pytest.raises(ValueError, match=match):
        load_activation_dataset(manifest_path)


@pytest.mark.parametrize(
    ("collection", "match"),
    [
        ("generation_records", "generation record has unknown fields"),
        ("label_records", "label record has unknown fields"),
        ("interaction_events", "interaction event has unknown fields"),
    ],
)
def test_recovery_rejects_rehashed_unknown_canonical_record_fields(
    tmp_path: Path,
    collection: str,
    match: str,
) -> None:
    rows = [_row(index) for index in range(3)]
    manifest_path = save_activation_recovery_checkpoint(
        tmp_path / f"recovery-unknown-{collection}.json",
        samples=[row[0] for row in rows],
        generation_records=[row[1] for row in rows],
        interaction_events=[row[2] for row in rows],
        label_records=[row[3] for row in rows],
        experiment_track="single_agent_white_box",
        captured_actor_ids=("Negotiator",),
        pod_id=0,
        trial_id_offset=0,
        current_trial_id=1,
        reason="strict canonical fixture",
    )

    def mutate(manifest):
        manifest[collection][0]["future_field"] = None

    _rewrite_recovery_bundle(manifest_path, mutate)
    with pytest.raises(ValueError, match=match):
        load_activation_recovery_checkpoint(manifest_path)


def test_loader_rejects_registry_substitution_even_when_rehashed(tmp_path: Path) -> None:
    manifest_path = _save(_builder(), tmp_path / "registry.json")

    def mutate(_outer, manifest, _arrays):
        manifest["schema_registry"]["generation_record"] = "attacker/9"
        checksum = hashlib.sha256(
            json.dumps(
                manifest["schema_registry"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        manifest["schema_registry_checksum"] = checksum
        manifest["config"]["schema_registry_checksum"] = checksum

    _rewrite_bundle(manifest_path, mutate)
    with pytest.raises(ValueError, match="current registry"):
        load_activation_dataset(manifest_path)


@pytest.mark.parametrize("tamper", ["dtype", "shape", "extra_member", "nonfinite"])
def test_loader_rejects_semantic_array_tampering_after_rehash(
    tmp_path: Path,
    tamper: str,
) -> None:
    manifest_path = _save(_builder(), tmp_path / f"array-{tamper}.json")

    def mutate(_outer, _manifest, arrays):
        if tamper == "dtype":
            arrays["activation_0000"] = arrays["activation_0000"].astype(np.float64)
        elif tamper == "shape":
            arrays["activation_0000"] = arrays["activation_0000"][:, :0]
        elif tamper == "extra_member":
            arrays["undeclared"] = np.ones((3, 1), dtype=np.float32)
        else:
            arrays["activation_0000"][0, 0] = np.nan

    _rewrite_bundle(manifest_path, mutate)
    with pytest.raises((TypeError, ValueError), match="dtype|shape|undeclared|finite"):
        load_activation_dataset(manifest_path)


def test_loader_rejects_sae_mask_top_feature_mismatch(tmp_path: Path) -> None:
    manifest_path = _save(_builder(), tmp_path / "sae.json")

    def mutate(_outer, _manifest, arrays):
        arrays["sae_available_mask"][1] = False

    _rewrite_bundle(manifest_path, mutate)
    with pytest.raises(ValueError, match="unavailable SAE rows"):
        load_activation_dataset(manifest_path)


def test_negotiation_lineage_and_three_component_split_fail_closed(tmp_path: Path) -> None:
    dataset = _builder().build(
        model_name="offline-model",
        model_revision="offline-model@abc123",
        tokenizer_name="offline-tokenizer",
        tokenizer_revision="offline-tokenizer@def456",
    )
    dataset["metadata"][0]["generation_record_id"] = None
    with pytest.raises(ValueError, match="GenerationRecord reference"):
        save_activation_dataset(tmp_path / "missing-lineage.json", dataset)

    one_row = _builder().samples[0]
    one_row.sample_type = "serialization_fixture"
    one_row.generation_record_id = None
    one_row.interaction_event_id = None
    one_row.label_record_ids = []
    builder = DatasetBuilder(
        experiment_track="single_agent_white_box",
        captured_actor_ids=("Negotiator",),
        capture_modes=("generation_pass",),
    )
    builder.add_sample(one_row)
    with pytest.raises(ValueError, match="three independent"):
        builder.save(
            tmp_path / "too-small.json",
            model_name="offline-model",
            model_revision="offline-model@abc123",
            tokenizer_name="offline-tokenizer",
            tokenizer_revision="offline-tokenizer@def456",
        )


def test_merge_preserves_canonical_records_and_remaps_counterparts(tmp_path: Path) -> None:
    first = _save(_builder(0), tmp_path / "pod-a.json")
    second = _save(_builder(3), tmp_path / "pod-b.json")
    merged_path = merge_parallel_activations(
        [str(first), str(second)],
        output_dir=str(tmp_path / "merged"),
        timestamp="fixed",
        verbose=False,
    )
    merged = load_activation_dataset(merged_path)

    assert merged["config"]["n_samples"] == 6
    assert len(merged["generation_records"]) == 6
    assert len(merged["interaction_events"]) == 6
    assert len(merged["label_records"]) == 6
    assert merged["labels"]["counterpart_idxs"] == [1, 0, None, 4, 3, None]
    assert merged["merge_info"]["source_dataset_hashes"] == [
        load_activation_dataset(first)["config"]["dataset_hash"],
        load_activation_dataset(second)["config"]["dataset_hash"],
    ]


def test_merge_deduplicates_designs_and_preserves_trial_intervention_logs(
    tmp_path: Path,
) -> None:
    first = _save(_intervention_builder(0), tmp_path / "intervention-a.json")
    second = _save(_intervention_builder(3), tmp_path / "intervention-b.json")

    merged_path = merge_parallel_activations(
        [str(first), str(second)],
        output_dir=str(tmp_path / "merged-interventions"),
        timestamp="fixed",
        verbose=False,
    )
    merged = load_activation_dataset(merged_path)

    assert merged["config"]["n_samples"] == 12
    assert len(merged["intervention_designs"]) == 1
    assert len(merged["intervention_schedules"]) == 6
    assert len(merged["intervention_application_logs"]) == 6
    assert len({
        row["intervention_application_receipt_ids"][0]
        for row in merged["metadata"]
    }) == 6


def test_merge_rejects_incompatible_tracks_and_conflicting_canonical_ids(
    tmp_path: Path,
) -> None:
    first = _save(_builder(0), tmp_path / "pod-a.json")
    incompatible_builder = _builder(3)
    incompatible_builder.experiment_track = "theory_of_mind"
    for sample in incompatible_builder.samples:
        sample.experiment_track = "theory_of_mind"
        sample.modules_enabled = ["theory_of_mind"]
    second = _save(incompatible_builder, tmp_path / "pod-b.json")
    with pytest.raises(ValueError, match="provenance/config is incompatible"):
        merge_parallel_activations(
            [str(first), str(second)],
            output_dir=str(tmp_path / "merged"),
            verbose=False,
        )

    with pytest.raises(ValueError, match="conflicting duplicate canonical ID"):
        _merge_canonical_records(
            [
                {"generation_records": [{"call_id": "call-1", "value": 1}]},
                {"generation_records": [{"call_id": "call-1", "value": 2}]},
            ],
            collection="generation_records",
            id_key="call_id",
        )


def test_metadata_objects_fail_with_explicit_json_projection_error(tmp_path: Path) -> None:
    builder = _builder()
    builder.samples[0].tom_state = {"unsafe": object()}
    with pytest.raises(TypeError, match="metadata must be JSON-safe"):
        _save(builder, tmp_path / "bad-metadata.json")


def test_analysis_discovery_selects_safe_manifests_and_explicit_legacy_only(
    tmp_path: Path,
) -> None:
    from interpretability.analyze_data import find_data_files, load_and_merge

    manifest_path = _save(_builder(), tmp_path / "activations.json")
    (tmp_path / "probe_results.json").write_text("{}\n", encoding="utf-8")
    legacy_path = tmp_path / "reviewed.pt"
    torch.save({"labels": {"gm_labels": [0.0]}}, legacy_path)

    discovered = find_data_files([str(tmp_path)])
    assert discovered == sorted([str(manifest_path), str(legacy_path)])
    loaded = load_and_merge([str(manifest_path)], verbose=False)
    assert loaded["config"]["dataset_hash"] == load_activation_dataset(
        manifest_path
    )["config"]["dataset_hash"]


def test_cli_training_helper_round_trips_safe_bundle_through_full_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import interpretability.cli as cli_module
    from interpretability.probes.train_probes import run_full_analysis

    manifest_path = _save(_analysis_builder(), tmp_path / "analysis.json")
    monkeypatch.setattr(cli_module, "run_full_analysis", run_full_analysis, raising=False)
    results = cli_module._train_probes_on_data(
        str(manifest_path),
        None,
        trusted_legacy=False,
    )

    assert results["best_probe"] is not None
    assert "SplitManifest" in results["split_unit"]


def test_cli_validates_and_persists_track_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import interpretability.cli as cli_module

    constructed: list[dict[str, object]] = []

    def forbidden_runner(**kwargs):
        constructed.append(kwargs)
        raise RuntimeError("model construction sentinel")

    monkeypatch.setattr(cli_module, "_lazy_import", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli_module, "get_emergent_scenarios", lambda: ["ultimatum_bluff"], raising=False
    )
    monkeypatch.setattr(
        cli_module, "get_instructed_scenarios", lambda: ["ultimatum_bluff"], raising=False
    )
    monkeypatch.setattr(
        cli_module, "InterpretabilityRunner", forbidden_runner, raising=False
    )
    runner = CliRunner()

    invalid_output = tmp_path / "invalid"
    invalid = runner.invoke(
        cli_module.cli,
        [
            "run",
            "--fast",
            "--experiment-track",
            "theory_of_mind",
            "--output",
            str(invalid_output),
        ],
    )
    assert invalid.exit_code != 0
    assert constructed == []
    assert not (invalid_output / "experiment_track_manifest.json").exists()

    valid_output = tmp_path / "valid"
    valid = runner.invoke(
        cli_module.cli,
        [
            "run",
            "--fast",
            "--experiment-track",
            "single_agent_white_box",
            "--output",
            str(valid_output),
        ],
    )
    assert valid.exit_code != 0
    assert isinstance(valid.exception, RuntimeError)
    assert len(constructed) == 1
    assert "Mode: emergent" in valid.output
    assert "LEGACY/NON-HEADLINE" not in valid.output
    manifest = json.loads(
        (valid_output / "experiment_track_manifest.json").read_text("utf-8")
    )
    assert manifest["experiment_track"] == "single_agent_white_box"
    assert manifest["captured_actor_ids"] == ["Negotiator", "Counterpart"]
    assert manifest["headline_capture_policy"] == "one_logical_actor_per_trial"
    assert manifest["headline_captured_actor_count_per_trial"] == 1
    assert manifest["experiment_mode"] == "emergent"
    assert manifest["headline_probe_rows"] == "emergent"

    both_output = tmp_path / "both"
    both = runner.invoke(
        cli_module.cli,
        [
            "run",
            "--mode",
            "both",
            "--fast",
            "--output",
            str(both_output),
        ],
    )
    assert both.exit_code != 0
    assert isinstance(both.exception, RuntimeError)
    assert "LEGACY/NON-HEADLINE" in both.output
    assert len(constructed) == 2
    both_manifest = json.loads(
        (both_output / "experiment_track_manifest.json").read_text("utf-8")
    )
    assert both_manifest["headline_probe_rows"] == "emergent_only"

    ultrafast_output = tmp_path / "ultrafast"
    ultrafast_result = runner.invoke(
        cli_module.cli,
        [
            "run",
            "--ultrafast",
            "--output",
            str(ultrafast_output),
        ],
    )
    assert ultrafast_result.exit_code != 0
    assert isinstance(ultrafast_result.exception, RuntimeError)
    ultrafast_manifest = json.loads(
        (ultrafast_output / "experiment_track_manifest.json").read_text("utf-8")
    )
    assert ultrafast_manifest["experiment_track"] == (
        "single_agent_white_box"
    )
    assert ultrafast_manifest["enabled_modules"] == []


def test_cli_separates_emergent_publication_from_instructed_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import interpretability.cli as cli_module
    from interpretability.evaluation import InterpretabilityRunner

    created_runners = []
    trained_paths: list[Path] = []

    def runner_factory(**kwargs):
        runner = object.__new__(InterpretabilityRunner)
        runner.model = SimpleNamespace(
            model_name="offline-model@abc123",
            tokenizer=SimpleNamespace(
                name_or_path="offline-tokenizer@def456"
            ),
        )
        runner.activation_samples = []
        runner.generation_records = []
        runner.label_records = []
        runner.interaction_events = []
        runner.experiment_track = kwargs["experiment_track"]
        runner.captured_actor_ids = tuple(kwargs["captured_actor_ids"])
        runner._pod_id = 0
        runner._trial_id_offset = 0
        runner._trial_id = 0
        runner.safe_save_calls = []
        def tracked_save(filepath, **save_kwargs):
            runner.safe_save_calls.append(Path(filepath))
            assert all(
                sample.sample_type == "negotiation"
                and sample.trial_family_id
                for sample in runner.activation_samples
            )
            return InterpretabilityRunner.save_dataset(
                runner, filepath, **save_kwargs
            )

        runner.save_dataset = tracked_save
        created_runners.append(runner)
        return runner

    def fake_emergent(*, runner, **_kwargs):
        builder = _builder()
        runner.activation_samples.extend(builder.samples)
        runner.generation_records.extend(builder.generation_records)
        runner.label_records.extend(builder.label_records)
        runner.interaction_events.extend(builder.interaction_events)
        runner._trial_id = 3
        return {"scientific_status": "headline"}

    def fake_instructed(*, runner, **_kwargs):
        runner._trial_id += 1
        runner.activation_samples.append(
            ActivationSample(
                trial_id=runner._trial_id,
                round_num=0,
                agent_name="Negotiator",
                activations={
                    "blocks.2.hook_resid_post": torch.tensor(
                        [9.0, 8.0, 7.0, 6.0]
                    )
                },
                actual_deception=1.0,
                perceived_deception=None,
                prompt="Explicitly instructed compatibility prompt",
                response="Compatibility response",
                experiment_mode="instructed",
                experiment_track="single_agent_white_box",
                # Deliberately no family/lineage: this cannot be published.
            )
        )
        return {"scientific_status": "legacy_non_headline"}

    monkeypatch.setattr(cli_module, "_lazy_import", lambda: None)
    monkeypatch.setattr(
        cli_module,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli_module, "get_emergent_scenarios", lambda: ["ultimatum_bluff"],
        raising=False,
    )
    monkeypatch.setattr(
        cli_module, "get_instructed_scenarios", lambda: ["ultimatum_bluff"],
        raising=False,
    )
    monkeypatch.setattr(
        cli_module, "InterpretabilityRunner", runner_factory, raising=False
    )
    monkeypatch.setattr(
        cli_module, "_run_emergent_experiment", fake_emergent
    )
    monkeypatch.setattr(
        cli_module, "_run_instructed_experiment", fake_instructed
    )
    monkeypatch.setattr(
        cli_module,
        "_train_probes_on_data",
        lambda path, *_args, **_kwargs: (
            trained_paths.append(Path(path)) or {"best_probe": None}
        ),
    )
    monkeypatch.setattr(cli_module, "_print_summary", lambda *_a, **_k: None)
    cli_runner = CliRunner()

    both_dir = tmp_path / "both-e2e"
    both = cli_runner.invoke(
        cli_module.cli,
        ["run", "--mode", "both", "--fast", "--output", str(both_dir)],
    )
    assert both.exit_code == 0, both.output
    both_runner = created_runners[-1]
    assert len(both_runner.safe_save_calls) == 1
    headline_path = both_runner.safe_save_calls[0]
    assert load_activation_dataset(headline_path)["config"]["n_samples"] == 3
    recovery_path = next(both_dir.glob("instructed_legacy_recovery_*.json"))
    recovery = load_activation_recovery_checkpoint(recovery_path)
    assert len(recovery["activation_samples"]) == 1
    assert recovery["activation_samples"][0].sample_type == "unclassified"
    assert trained_paths[-1] == headline_path

    instructed_dir = tmp_path / "instructed-e2e"
    trained_before = len(trained_paths)
    instructed = cli_runner.invoke(
        cli_module.cli,
        [
            "run",
            "--mode",
            "instructed",
            "--fast",
            "--output",
            str(instructed_dir),
        ],
    )
    assert instructed.exit_code == 0, instructed.output
    instructed_runner = created_runners[-1]
    assert instructed_runner.safe_save_calls == []
    instructed_recovery = next(
        instructed_dir.glob("instructed_legacy_recovery_*.json")
    )
    assert load_activation_recovery_checkpoint(instructed_recovery)[
        "activation_samples"
    ][0].sample_type == "unclassified"
    assert len(trained_paths) == trained_before
    assert "Headline probe training unavailable" in instructed.output


def test_requested_causal_validation_failure_is_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import interpretability.cli as cli_module

    manifest_path = _save(_builder(), tmp_path / "causal-input.json")
    monkeypatch.setattr(
        cli_module,
        "filter_causal_samples",
        lambda activations, labels, metadata, **kwargs: (
            activations,
            np.asarray(labels),
            metadata,
            np.asarray(["group-a", "group-b", "group-c"]),
        ),
        raising=False,
    )
    monkeypatch.setattr(cli_module, "np", np, raising=False)
    runner = SimpleNamespace(model=SimpleNamespace())

    with pytest.raises(click.ClickException, match="cannot access"):
        cli_module._run_causal_validation(
            runner,
            manifest_path,
            {"best_probe": {"layer": 2}},
            3,
            tmp_path,
        )
