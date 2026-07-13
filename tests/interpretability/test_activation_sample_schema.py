"""Ownership and compatibility tests for the canonical activation schema."""

from dataclasses import fields

import torch

from interpretability.core.dataset_builder import (
    ActivationSample as BuilderActivationSample,
)
from interpretability.data import ActivationSample
from interpretability.evaluation import ActivationSample as RunnerActivationSample


LEGACY_RUNNER_FIELDS = {
    "trial_id",
    "round_num",
    "agent_name",
    "activations",
    "prompt",
    "response",
    "perceived_deception",
    "emotion_intensity",
    "trust_level",
    "cooperation_intent",
    "actual_deception",
    "commitment_violation",
    "manipulation_score",
    "consistency_score",
    "scenario_type",
    "modules_enabled",
    "pod_id",
    "gm_modules_enabled",
    "counterpart_idx",
    "counterpart_name",
    "trial_outcome",
    "joint_value",
    "agent_utility",
    "condition_id",
    "followup_activations",
    "emergent_scenario",
    "incentive_condition",
    "scenario_params",
    "emergent_ground_truth",
    "sae_features",
    "sae_top_features",
    "sae_sparsity",
    "timestamp",
    "experiment_mode",
    "tom_state",
    "deepeval_false_claims",
    "deepeval_omission",
    "deepeval_framing",
    "deepeval_commitment",
    "deepeval_confidence",
    "deepeval_reasoning",
    "component_reasoning",
    "dialogue_history",
    "is_verification_probe",
    "plausibility_response",
    "sample_type",
    "counterpart_type",
    "belief_shift_injected",
    "belief_shift_type",
    "belief_shift_round",
    "framing_variant",
    "activation_position",
    "sampling_config",
    "gt_regex",
    "gt_llm_rules",
    "gt_deepeval",
}


def _minimal_sample(trial_id=1):
    return ActivationSample(
        trial_id=trial_id,
        round_num=0,
        agent_name="Negotiator",
        activations={"blocks.2.hook_resid_post": torch.tensor([1.0, 2.0])},
        actual_deception=0.0,
        perceived_deception=0.25,
    )


def test_runner_and_builder_export_the_one_canonical_class():
    assert RunnerActivationSample is ActivationSample
    assert BuilderActivationSample is ActivationSample
    assert ActivationSample.__module__ == "interpretability.data.schema"
    assert LEGACY_RUNNER_FIELDS <= {field.name for field in fields(ActivationSample)}


def test_minimal_builder_and_full_legacy_keyword_construction_are_supported():
    minimal = _minimal_sample(trial_id="trial-a")
    full = ActivationSample(
        trial_id=7,
        round_num=3,
        agent_name="Seller",
        activations={"blocks.4.hook_resid_post": torch.ones(3)},
        prompt="Assembled context",
        response="I can accept 70.",
        perceived_deception=0.2,
        emotion_intensity=0.1,
        trust_level=0.6,
        cooperation_intent=0.7,
        actual_deception=1.0,
        commitment_violation=0.0,
        manipulation_score=0.4,
        consistency_score=0.9,
        scenario_type="hidden_value",
        modules_enabled=["theory_of_mind"],
        scenario_params={"true_value": 40},
        sampling_config={"seed": 9},
        generation_record_id="generation-7-3",
        interaction_event_id="event-7-3",
        label_record_ids=["label-rules", "label-judge"],
        actual_deception_projection=1.0,
        execution_protocol="alternating",
        intervention_design_id="intervention-design-1",
        intervention_application_receipt_ids=["receipt-1"],
    )

    assert minimal.trial_id == "trial-a"
    assert minimal.prompt == ""
    assert minimal.trust_level is None
    assert minimal.consistency_score == 1.0
    assert minimal.sample_type == "unclassified"
    assert minimal.qc_status == "unreviewed"
    assert minimal.qc_version is None
    assert minimal.timestamp is None
    assert full.trial_id == 7
    assert full.actual_deception == 1.0
    assert full.actual_deception_projection == 1.0
    assert full.label_record_ids == ["label-rules", "label-judge"]
    assert full.execution_protocol == "alternating"
    assert full.intervention_application_receipt_ids == ["receipt-1"]


def test_timestamp_is_explicit_and_never_implicitly_reads_the_wall_clock():
    first = _minimal_sample(trial_id="first")
    second = _minimal_sample(trial_id="second")
    explicit = ActivationSample(
        trial_id="explicit",
        round_num=0,
        agent_name="Negotiator",
        activations={"blocks.2.hook_resid_post": torch.ones(2)},
        actual_deception=None,
        perceived_deception=None,
        timestamp="2035-04-03T02:01:00+00:00",
    )

    assert first.timestamp is None
    assert second.timestamp is None
    assert explicit.timestamp == "2035-04-03T02:01:00+00:00"


def test_mutable_defaults_are_isolated_between_samples():
    first = _minimal_sample()
    second = _minimal_sample()

    first.modules_enabled.append("theory_of_mind")
    first.gm_modules_enabled.append("protocol")
    first.scenario_params["private"] = 1
    first.sampling_config["seed"] = 42
    first.label_record_ids.append("label-1")
    first.qc_flags.append("too_short")
    first.intervention_application_receipt_ids.append("receipt-1")

    assert second.modules_enabled == []
    assert second.gm_modules_enabled == []
    assert second.scenario_params == {}
    assert second.sampling_config == {}
    assert second.label_record_ids == []
    assert second.qc_flags == []
    assert second.intervention_application_receipt_ids == []


def test_torch_round_trip_preserves_canonical_identity_and_references(tmp_path):
    sample = _minimal_sample(trial_id="trial-round-trip")
    sample.generation_record_id = "generation-1"
    sample.interaction_event_id = "event-1"
    sample.label_record_ids.extend(["label-1", "label-2"])
    sample.actual_deception_projection = 0.75
    path = tmp_path / "sample.pt"

    torch.save(sample, path)
    restored = torch.load(path, weights_only=False)

    assert type(restored) is ActivationSample
    assert restored.trial_id == "trial-round-trip"
    assert torch.equal(
        restored.activations["blocks.2.hook_resid_post"],
        torch.tensor([1.0, 2.0]),
    )
    assert restored.generation_record_id == "generation-1"
    assert restored.interaction_event_id == "event-1"
    assert restored.label_record_ids == ["label-1", "label-2"]
    assert restored.actual_deception_projection == 0.75
