"""Synthetic guarantees for locked-test nested grouped probe evaluation."""

import numpy as np
import pytest

from interpretability.data import SplitManifest
from interpretability.probes.artifacts import load_headline_probe_artifact
from interpretability.probes.metrics import evaluate_nested_grouped_layers
from interpretability.probes.train_probes import (
    find_best_layer,
    train_probes_per_layer,
)


def _synthetic_data():
    rng = np.random.default_rng(17)
    group_names = [f"trial-{index}" for index in range(9)]
    groups = np.repeat(group_names, 2)
    partitions = np.repeat(
        ["train"] * 3 + ["development"] * 3 + ["test"] * 3,
        2,
    )
    labels = np.tile([0.0, 1.0], 9)
    signal = labels[:, None] * 8.0 + rng.normal(scale=0.05, size=(18, 1))
    layer_zero = np.hstack((signal, rng.normal(size=(18, 5))))
    layer_one = rng.normal(size=(18, 6))
    test_mask = partitions == "test"
    layer_zero[test_mask] += 50.0
    layer_one[test_mask] += 50.0
    return {0: layer_zero, 1: layer_one}, labels, partitions, groups


def _evaluate(labels=None):
    activations, original_labels, partitions, groups = _synthetic_data()
    return evaluate_nested_grouped_layers(
        activations,
        original_labels if labels is None else labels,
        partition_labels=partitions,
        groups=groups,
        hyperparameters=(
            {"C": 0.1, "n_components": 3},
            {"C": 1.0, "n_components": 3},
        ),
        random_state=3,
    )


def test_test_label_changes_cannot_change_layer_or_hyperparameter_selection():
    activations, labels, partitions, _ = _synthetic_data()
    changed_labels = labels.copy()
    changed_labels[partitions == "test"] = 1.0 - changed_labels[
        partitions == "test"
    ]

    original = _evaluate(labels)
    changed = _evaluate(changed_labels)

    assert original.selected_layer == changed.selected_layer == 0
    assert original.selected_hyperparameters == changed.selected_hyperparameters
    assert original.selection_table == changed.selection_table
    assert original.test_metrics["auc"] != changed.test_metrics["auc"]


def test_preprocessing_is_fit_only_on_rows_available_to_each_stage():
    activations, _, partitions, _ = _synthetic_data()
    result = _evaluate()
    selected_X = activations[result.selected_layer]
    train_mask = partitions == "train"
    fit_mask = np.isin(partitions, ["train", "development"])

    assert np.allclose(
        result.selection_pipeline.named_steps["scaler"].mean_,
        selected_X[train_mask].mean(axis=0),
    )
    assert np.allclose(
        result.pipeline.named_steps["scaler"].mean_,
        selected_X[fit_mask].mean(axis=0),
    )
    assert not np.allclose(
        result.pipeline.named_steps["scaler"].mean_,
        selected_X.mean(axis=0),
    )


def test_outer_groups_are_disjoint_and_provenance_is_explicit():
    result = _evaluate()
    provenance = result.split_provenance
    partition_groups = provenance["partition_group_ids"]

    assert set(partition_groups["train"]).isdisjoint(
        partition_groups["development"]
    )
    assert set(partition_groups["train"]).isdisjoint(partition_groups["test"])
    assert set(partition_groups["development"]).isdisjoint(
        partition_groups["test"]
    )
    assert provenance["selection_fit_partitions"] == ["train"]
    assert provenance["final_fit_partitions"] == ["train", "development"]
    assert provenance["test_evaluations"] == 1


def test_saved_final_pipeline_reproduces_locked_test_predictions(tmp_path):
    activations, _, _, _ = _synthetic_data()
    result = _evaluate()
    _, manifest_path = result.save_artifact(tmp_path / "headline")
    restored = load_headline_probe_artifact(manifest_path)

    predictions = restored.predict_proba(
        activations[result.selected_layer][result.test_indices]
    )[:, 1]

    assert np.allclose(
        predictions,
        result.test_probabilities,
        rtol=1e-13,
        atol=1e-15,
    )


def test_selection_and_locked_test_metrics_are_separate_contracts():
    result = _evaluate()
    serialized = result.to_dict()

    assert serialized["selection_table"]
    assert all(
        row["selection_partition"] == "development"
        and "test_metrics" not in row
        for row in serialized["selection_table"]
    )
    assert set(serialized["test_metrics"]) == {"auc", "accuracy", "r2"}
    assert "development_metrics" not in serialized["test_metrics"]


def test_manifest_assignments_can_drive_connected_outer_split():
    activations, labels, _, groups = _synthetic_data()
    records = [
        {
            "trial_id": str(trial_id),
            "trial_family_id": f"family-{str(trial_id)}",
            "dyad_id": f"dyad-{str(trial_id)}",
        }
        for trial_id in sorted(set(groups))
    ]
    manifest = SplitManifest.build(records, seed=11)

    result = evaluate_nested_grouped_layers(
        activations,
        labels,
        split_manifest=manifest,
        trial_ids=groups,
        hyperparameters=({"C": 1.0, "n_components": 3},),
    )

    assert result.split_provenance["manifest_id"] == manifest.manifest_id
    assert result.split_provenance["split_source"] == "split_manifest"


def test_headline_evaluation_fails_without_groups_or_three_partitions():
    activations, labels, partitions, groups = _synthetic_data()
    with pytest.raises(ValueError, match="requires explicit"):
        evaluate_nested_grouped_layers(
            activations,
            labels,
            partition_labels=partitions,
        )

    missing_development = partitions.copy()
    missing_development[missing_development == "development"] = "train"
    with pytest.raises(ValueError, match="non-empty train, development, and test"):
        evaluate_nested_grouped_layers(
            activations,
            labels,
            partition_labels=missing_development,
            groups=groups,
        )


def test_legacy_layer_helpers_require_development_selection_for_headlines():
    activations, labels, partitions, groups = _synthetic_data()

    with pytest.raises(ValueError, match="Headline per-layer training requires"):
        train_probes_per_layer(activations, labels, groups=groups)

    layer_results = train_probes_per_layer(
        activations,
        labels,
        groups=groups,
        partition_labels=partitions,
        n_components=3,
    )
    selection = find_best_layer(layer_results)

    assert selection["headline_eligible"] is True
    assert selection["selection_partition"] == "development"

    diagnostic = train_probes_per_layer(
        activations,
        labels,
        groups=groups,
        diagnostic_non_headline=True,
        n_components=3,
    )
    with pytest.raises(ValueError, match="cannot produce a headline layer"):
        find_best_layer(diagnostic)
    diagnostic_selection = find_best_layer(
        diagnostic,
        diagnostic_non_headline=True,
    )
    assert diagnostic_selection["headline_eligible"] is False


@pytest.mark.parametrize(
    "invalid_labels, message",
    [
        (np.array([0.0, 1.0] * 8 + [np.nan, 1.0]), "finite and available"),
        (np.array([0.0, 1.0] * 8 + [0.4, 1.0]), "explicit binary"),
        (np.array([0.0, 1.0] * 8 + [-1.0, 1.0]), "explicit binary"),
    ],
)
def test_headline_evaluation_rejects_unknown_or_nonbinary_labels(
    invalid_labels,
    message,
):
    with pytest.raises(ValueError, match=message):
        _evaluate(invalid_labels)


def test_headline_evaluation_rejects_nonfinite_activations_and_hyperparameters():
    activations, labels, partitions, groups = _synthetic_data()
    activations[0][0, 0] = np.nan
    with pytest.raises(ValueError, match="activations must be finite"):
        evaluate_nested_grouped_layers(
            activations,
            labels,
            partition_labels=partitions,
            groups=groups,
        )

    activations, labels, partitions, groups = _synthetic_data()
    with pytest.raises(ValueError, match="positive finite"):
        evaluate_nested_grouped_layers(
            activations,
            labels,
            partition_labels=partitions,
            groups=groups,
            hyperparameters=({"C": float("nan"), "n_components": 3},),
        )
