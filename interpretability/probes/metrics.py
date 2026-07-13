"""Leakage-safe nested grouped evaluation for headline probe results."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from interpretability.data import SplitManifest


_PARTITIONS = ("train", "development", "test")


@dataclass
class NestedGroupedEvaluation:
    """Fitted headline probe and strictly separated evaluation artifacts."""

    pipeline: Pipeline
    selection_pipeline: Pipeline
    selected_layer: int
    selected_hyperparameters: Dict[str, Any]
    selection_table: list[Dict[str, Any]]
    test_metrics: Dict[str, float]
    split_provenance: Dict[str, Any]
    test_indices: np.ndarray
    test_probabilities: np.ndarray
    dataset_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable metrics without embedding fitted estimators."""
        return {
            "selected_layer": self.selected_layer,
            "selected_hyperparameters": dict(self.selected_hyperparameters),
            "selection_table": list(self.selection_table),
            "test_metrics": dict(self.test_metrics),
            "dataset_sha256": self.dataset_sha256,
            "split_provenance": dict(self.split_provenance),
        }

    def save_artifact(self, base_path: str | Path) -> tuple[Path, Path]:
        """Save the fitted headline probe through the pickle-free public API."""
        from interpretability.probes.artifacts import (
            save_headline_probe_artifact,
        )

        metadata = self.to_dict()
        split_provenance = metadata.pop("split_provenance")
        return save_headline_probe_artifact(
            base_path,
            self.pipeline,
            metadata=metadata,
            split_provenance=split_provenance,
        )


def resolve_manifest_row_groups(
    manifest: SplitManifest,
    trial_ids: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve locked partitions and connected family/dyad groups per row."""
    assignment_by_trial = {row.trial_id: row for row in manifest.assignments}
    normalized_trials = np.asarray([str(value) for value in trial_ids])
    missing = sorted(set(normalized_trials) - set(assignment_by_trial))
    if missing:
        raise ValueError(
            "SplitManifest has no assignments for trial_ids: " + ", ".join(missing)
        )

    # Manifest validation guarantees connected families/dyads stay in one
    # partition. Reconstruct stable connected-component IDs for grouped fits.
    unique_trials = sorted(set(normalized_trials))
    parents = {trial_id: trial_id for trial_id in unique_trials}

    def find(trial_id: str) -> str:
        while parents[trial_id] != trial_id:
            parents[trial_id] = parents[parents[trial_id]]
            trial_id = parents[trial_id]
        return trial_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    family_owner = {}
    dyad_owner = {}
    for trial_id in unique_trials:
        assignment = assignment_by_trial[trial_id]
        if assignment.trial_family_id in family_owner:
            union(trial_id, family_owner[assignment.trial_family_id])
        else:
            family_owner[assignment.trial_family_id] = trial_id
        if assignment.dyad_id in dyad_owner:
            union(trial_id, dyad_owner[assignment.dyad_id])
        else:
            dyad_owner[assignment.dyad_id] = trial_id

    groups = np.asarray([find(trial_id) for trial_id in normalized_trials])
    partitions = np.asarray([
        assignment_by_trial[trial_id].partition for trial_id in normalized_trials
    ])
    return partitions, groups


def _resolve_outer_split(
    n_rows: int,
    *,
    partition_labels: Optional[Sequence[str]],
    groups: Optional[Sequence[Any]],
    split_manifest: Optional[SplitManifest],
    trial_ids: Optional[Sequence[Any]],
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if split_manifest is not None:
        if partition_labels is not None or groups is not None:
            raise ValueError(
                "Provide SplitManifest+trial_ids or explicit partitions+groups, not both"
            )
        if trial_ids is None or len(trial_ids) != n_rows:
            raise ValueError("SplitManifest evaluation requires one trial_id per row")
        split_manifest.validate()
        partitions_array, groups_array = resolve_manifest_row_groups(
            split_manifest, trial_ids
        )
        manifest_id = split_manifest.manifest_id
        split_source = "split_manifest"
    else:
        if partition_labels is None or groups is None:
            raise ValueError(
                "Headline evaluation requires explicit train/development/test "
                "partitions and connected trial-family/dyad groups"
            )
        if len(partition_labels) != n_rows or len(groups) != n_rows:
            raise ValueError("partitions and groups must have one value per row")
        partitions_array = np.asarray(partition_labels)
        groups_array = np.asarray(groups)
        split_payload = {
            "partitions": partitions_array.tolist(),
            "groups": [str(value) for value in groups_array.tolist()],
        }
        manifest_id = hashlib.sha256(
            json.dumps(
                split_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        split_source = "explicit_partitions_and_groups"

    observed = set(partitions_array.tolist())
    if observed != set(_PARTITIONS):
        raise ValueError(
            "Headline evaluation requires non-empty train, development, and test "
            f"partitions; observed {sorted(observed)}"
        )
    group_partitions = {}
    for group, partition in zip(groups_array.tolist(), partitions_array.tolist()):
        group_partitions.setdefault(group, set()).add(partition)
    crossing = sorted(
        str(group) for group, values in group_partitions.items() if len(values) > 1
    )
    if crossing:
        raise ValueError(
            "Connected trial-family/dyad groups cross outer partitions: "
            + ", ".join(crossing)
        )

    indices = {
        partition: np.flatnonzero(partitions_array == partition)
        for partition in _PARTITIONS
    }
    provenance = {
        "split_source": split_source,
        "manifest_id": manifest_id,
        "split_locked": True,
        "partition_sizes": {
            partition: int(len(indices[partition])) for partition in _PARTITIONS
        },
        "partition_group_ids": {
            partition: sorted(set(groups_array[indices[partition]].astype(str)))
            for partition in _PARTITIONS
        },
        "selection_fit_partitions": ["train"],
        "final_fit_partitions": ["train", "development"],
        "test_evaluations": 1,
    }
    return indices, provenance


def _pipeline(
    *,
    n_components: int,
    c_value: float,
    random_state: int,
) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
        ("probe", LogisticRegression(
            C=c_value,
            max_iter=2000,
            random_state=random_state,
        )),
    ])


def _partition_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "r2": float(r2_score(y_true, probabilities)),
    }


def _dataset_sha256(
    layer_arrays: Mapping[int, np.ndarray],
    labels: np.ndarray,
) -> str:
    """Bind a fitted probe to exact normalized activation and label bytes."""
    digest = hashlib.sha256(b"headline-probe-dataset/1\0")
    digest.update(np.ascontiguousarray(labels, dtype=np.int8).tobytes())
    for layer in sorted(layer_arrays):
        array = np.ascontiguousarray(layer_arrays[layer], dtype=np.float64)
        header = json.dumps(
            {"layer": layer, "shape": list(array.shape), "dtype": "float64"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(header + b"\0" + array.tobytes())
    return digest.hexdigest()


def evaluate_nested_grouped_layers(
    activations_by_layer: Mapping[int, np.ndarray],
    labels: Sequence[float] | np.ndarray,
    *,
    partition_labels: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[Any]] = None,
    split_manifest: Optional[SplitManifest] = None,
    trial_ids: Optional[Sequence[Any]] = None,
    hyperparameters: Optional[Sequence[Mapping[str, Any]]] = None,
    random_state: int = 42,
) -> NestedGroupedEvaluation:
    """Select on development, refit on train+development, test once.

    Scaler, PCA, and the probe are contained in one sklearn ``Pipeline`` so
    every preprocessing parameter is fitted only on the rows available to that
    stage. The locked test partition never contributes to selection.
    """
    if not activations_by_layer:
        raise ValueError("activations_by_layer cannot be empty")
    raw_labels = np.asarray(labels, dtype=float)
    if raw_labels.ndim != 1 or not len(raw_labels):
        raise ValueError("Binary probe labels must be a non-empty 1D array")
    if not bool(np.isfinite(raw_labels).all()):
        raise ValueError("Binary probe labels must be finite and available")
    unique_labels = set(raw_labels.tolist())
    if not unique_labels.issubset({0.0, 1.0}):
        raise ValueError(
            "Logistic headline probes require explicit binary 0/1 labels; "
            "unknown, severity, and out-of-range values need another target"
        )
    y = raw_labels.astype(int)
    n_rows = len(y)
    layer_arrays = {
        int(layer): np.asarray(values, dtype=float)
        for layer, values in activations_by_layer.items()
    }
    dataset_sha256 = _dataset_sha256(layer_arrays, y)
    if any(len(values) != n_rows for values in layer_arrays.values()):
        raise ValueError("Every activation layer must align with labels")
    for layer, values in layer_arrays.items():
        if values.ndim != 2 or values.shape[1] < 1:
            raise ValueError(f"Layer {layer} activations must be a non-empty 2D array")
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"Layer {layer} activations must be finite")
    split_indices, provenance = _resolve_outer_split(
        n_rows,
        partition_labels=partition_labels,
        groups=groups,
        split_manifest=split_manifest,
        trial_ids=trial_ids,
    )
    train_indices = split_indices["train"]
    development_indices = split_indices["development"]
    test_indices = split_indices["test"]
    for partition, indices in split_indices.items():
        if len(np.unique(y[indices])) < 2:
            raise ValueError(f"{partition} partition must contain both label classes")

    if hyperparameters is None:
        hyperparameters = (
            {"C": 0.1, "n_components": 8},
            {"C": 1.0, "n_components": 8},
            {"C": 10.0, "n_components": 8},
        )
    if not hyperparameters:
        raise ValueError("At least one hyperparameter candidate is required")

    selection_table = []
    candidate_artifacts = []
    for layer in sorted(layer_arrays):
        X = layer_arrays[layer]
        for raw_parameters in hyperparameters:
            c_value = float(raw_parameters.get("C", 1.0))
            requested_components = int(raw_parameters.get("n_components", 8))
            if not math.isfinite(c_value) or c_value <= 0.0:
                raise ValueError("Probe C must be a positive finite value")
            if requested_components < 1:
                raise ValueError("Probe n_components must be positive")
            n_components = min(
                requested_components,
                X.shape[1],
                max(1, len(train_indices) - 1),
            )
            fitted = _pipeline(
                n_components=n_components,
                c_value=c_value,
                random_state=random_state,
            )
            fitted.fit(X[train_indices], y[train_indices])
            development_probabilities = fitted.predict_proba(
                X[development_indices]
            )[:, 1]
            development_metrics = _partition_metrics(
                y[development_indices], development_probabilities
            )
            row = {
                "layer": layer,
                "hyperparameters": {
                    "C": c_value,
                    "n_components": n_components,
                },
                "selection_partition": "development",
                "development_metrics": development_metrics,
            }
            selection_table.append(row)
            candidate_artifacts.append((row, fitted))

    selected_row, selection_pipeline = max(
        candidate_artifacts,
        key=lambda item: (
            item[0]["development_metrics"]["auc"],
            item[0]["development_metrics"]["accuracy"],
            -item[0]["layer"],
            -item[0]["hyperparameters"]["n_components"],
            -item[0]["hyperparameters"]["C"],
        ),
    )
    selected_layer = int(selected_row["layer"])
    selected_parameters = dict(selected_row["hyperparameters"])
    final_pipeline = _pipeline(
        n_components=selected_parameters["n_components"],
        c_value=selected_parameters["C"],
        random_state=random_state,
    )
    fit_indices = np.concatenate((train_indices, development_indices))
    selected_X = layer_arrays[selected_layer]
    final_pipeline.fit(selected_X[fit_indices], y[fit_indices])

    # This is the only locked-test prediction in the evaluation contract.
    test_probabilities = final_pipeline.predict_proba(selected_X[test_indices])[:, 1]
    test_metrics = _partition_metrics(y[test_indices], test_probabilities)
    provenance["selected_development_metrics"] = dict(
        selected_row["development_metrics"]
    )
    provenance["selected_layer"] = selected_layer
    provenance["dataset_sha256"] = dataset_sha256

    return NestedGroupedEvaluation(
        pipeline=final_pipeline,
        selection_pipeline=selection_pipeline,
        selected_layer=selected_layer,
        selected_hyperparameters=selected_parameters,
        selection_table=selection_table,
        test_metrics=test_metrics,
        split_provenance=provenance,
        test_indices=test_indices,
        test_probabilities=test_probabilities,
        dataset_sha256=dataset_sha256,
    )
