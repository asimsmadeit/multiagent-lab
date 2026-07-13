import numpy as np
import pytest
import torch
from contextlib import contextmanager
from types import SimpleNamespace

from interpretability.causal.causal_validation import (
    SteeringVector,
    _split_train_test_indices,
    filter_causal_samples,
    run_full_causal_validation,
    steering_behavioral_test,
)
from interpretability.causal.design import (
    CausalDesignManifest,
    ControlKind,
    DirectionVectorIdentity,
    InterventionKind,
)


def test_filter_causal_samples_excludes_intervention_probes_and_builds_groups():
    activations = {3: torch.arange(36, dtype=torch.float32).reshape(6, 6)}
    labels = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    metadata = [
        {"sample_type": "pre_verification", "trial_id": 10},
        {"sample_type": "negotiation", "trial_id": 10},
        {"sample_type": "negotiation", "trial_id": 10},
        {"sample_type": "post_plausibility", "trial_id": 11},
        {"sample_type": "negotiation", "trial_id": 11},
        {"sample_type": "negotiation", "trial_id": 11},
    ]

    filtered, filtered_labels, filtered_meta, groups = filter_causal_samples(
        activations,
        labels,
        metadata,
        pod_ids=[2] * 6,
    )

    assert filtered[3].shape == (4, 6)
    assert filtered_labels.tolist() == [0.0, 1.0, 1.0, 0.0]
    assert all(row["sample_type"] == "negotiation" for row in filtered_meta)
    assert groups.tolist() == ["2:10", "2:10", "2:11", "2:11"]


def test_filter_causal_samples_uses_round_numbers_for_legacy_data():
    activations = {1: np.arange(20).reshape(5, 4)}
    filtered, labels, metadata, groups = filter_causal_samples(
        activations,
        [0, 1, 0, 1, 0],
        round_nums=[-1, 0, 1, -2, 2],
        trial_ids=[1, 1, 1, 1, 2],
    )

    assert filtered[1].shape[0] == 3
    assert labels.tolist() == [1, 0, 0]
    assert metadata == []
    assert groups.tolist() == ["0:1", "0:1", "0:2"]


def test_group_split_never_places_one_trial_on_both_sides():
    groups = np.repeat(["a", "b", "c", "d"], 3)
    train_idx, test_idx = _split_train_test_indices(
        len(groups), groups=groups, test_size=0.5, random_state=7
    )

    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))


def test_full_causal_validation_records_grouped_sampling_controls():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(6), 4)
    labels = np.repeat([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 4)
    activations = {2: rng.standard_normal((len(labels), 12))}

    result = run_full_causal_validation(
        model=None,
        activations=activations,
        labels=labels,
        best_layer=2,
        group_ids=groups,
        random_state=19,
        verbose=False,
    )

    assert result["sampling_controls"] == {
        "random_state": 19,
        "split_unit": "trial",
        "cross_sample_prompt_alignment_required": True,
    }
    assert result["tests"]["selectivity"]["details"]["split_unit"] == "trial"
    assert result["tests"]["probe_faithfulness"]["details"]["split_unit"] == "trial"
    assert result["causal_evidence_strength"] is None
    assert result["causal_claim_ready"] is False
    assert result["overall_passed"] is False
    assert result["aggregation_status"] == (
        "unavailable_without_preregistered_hierarchy"
    )


def test_behavioral_steering_without_versioned_scorer_is_unavailable() -> None:
    result = steering_behavioral_test(
        model=object(),
        steering_vector=SteeringVector(
            direction=np.ones(4),
            layer=2,
            method="mass_mean",
        ),
        test_prompts=["prompt"],
        n_samples=1,
        verbose=False,
    )

    assert result.passed is False
    assert result.n_samples_tested == 0
    assert result.details["available"] is False
    assert result.details["skipped"] is True
    assert "versioned" in result.message


def test_behavioral_steering_retains_every_completion_and_scorer_version() -> None:
    class FakeTokenizer:
        def decode(self, tokens, *, skip_special_tokens):
            assert skip_special_tokens is True
            return f"completion-{int(tokens[-1])}"

    class FakeModel:
        cfg = SimpleNamespace(n_ctx=128)
        tokenizer = FakeTokenizer()

        def parameters(self):
            yield torch.nn.Parameter(torch.ones(1))

        def to_tokens(self, prompt, *, truncate):
            assert prompt and truncate is True
            return torch.tensor([[1, 2]])

        @contextmanager
        def hooks(self, *, fwd_hooks):
            assert len(fwd_hooks) == 1
            yield

        def generate(self, tokens, **kwargs):
            assert kwargs["max_new_tokens"] > 0
            return torch.cat((tokens, torch.tensor([[3]])), dim=1)

    result = steering_behavioral_test(
        model=FakeModel(),
        steering_vector=SteeringVector(
            direction=np.ones(4),
            layer=2,
            method="mass_mean",
        ),
        test_prompts=["prompt-a", "prompt-b"],
        magnitudes=[-1.0, 0.0, 1.0],
        n_samples=2,
        random_direction_control=False,
        scorer_fn=lambda response, idx: idx == 1,
        scorer_version="test-scorer/1",
        n_perm=10,
        verbose=False,
    )

    assert result.details["available"] is True
    assert result.details["scoring_method"] == "custom_scorer_fn"
    assert result.details["scorer_version"] == "test-scorer/1"
    for measurement in result.details["deception_sweep"].values():
        assert measurement["n_generated"] == 2
        assert measurement["n_scoring_failed"] == 0
        assert len(measurement["completions"]) == 2
        assert [row["score"] for row in measurement["completions"]] == [False, True]


def test_full_validation_retains_and_checks_locked_design_identity() -> None:
    rng = np.random.default_rng(8)
    labels = np.repeat([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 4)
    activations = {2: rng.standard_normal((len(labels), 12))}
    manifest = CausalDesignManifest(
        study_id="study",
        dataset_hash="sha256:" + "d" * 64,
        intervention=InterventionKind.STEER,
        intervention_adapter_version="test-hook-adapter/1",
        direction_source="outer-train",
        primary_direction_identity=DirectionVectorIdentity.from_vector(
            "outer-train", np.asarray([1.0, 0.0, 0.0, 0.0])
        ),
        layer=2,
        token_stage="last_response_token",
        coefficients=(-1.0, 0.0, 1.0),
        outcomes=("logit",),
        independent_unit="trial",
        group_key="trial_id",
        controls=(ControlKind.ZERO_HOOK,),
        repetitions=2,
        random_seed=19,
    )

    result = run_full_causal_validation(
        model=None,
        activations=activations,
        labels=labels,
        best_layer=2,
        group_ids=np.repeat(np.arange(6), 4),
        random_state=19,
        design_manifest=manifest,
        verbose=False,
    )

    assert result["design_manifest"]["design_id"] == manifest.design_id
    assert result["causal_claim_ready"] is False
    with pytest.raises(ValueError, match="best_layer"):
        run_full_causal_validation(
            model=None,
            activations=activations,
            labels=labels,
            best_layer=3,
            group_ids=np.repeat(np.arange(6), 4),
            random_state=19,
            design_manifest=manifest,
            verbose=False,
        )
