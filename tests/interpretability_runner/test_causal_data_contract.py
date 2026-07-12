import numpy as np
import torch

from interpretability.causal.causal_validation import (
    _split_train_test_indices,
    filter_causal_samples,
    run_full_causal_validation,
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
