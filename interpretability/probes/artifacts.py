"""Pickle-free fitted artifacts for headline logistic probes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from interpretability.data import load_array_bundle, save_array_bundle


HEADLINE_PROBE_SCHEMA_VERSION = "1.2.0"
_ARTIFACT_TYPE = "headline_logistic_probe"
_BASE_STEPS = ("scaler", "probe")
_PCA_STEPS = ("scaler", "pca", "probe")
_PARTITION_NAMES = ("train", "development", "test")
_SUPPORTED_SOLVERS = {
    "lbfgs",
    "liblinear",
    "newton-cg",
    "newton-cholesky",
    "sag",
    "saga",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class HeadlineProbeArtifact:
    """Loaded inference pipeline with its non-executable scientific metadata."""

    pipeline: Pipeline
    metadata: dict[str, Any]
    split_provenance: dict[str, Any]

    @property
    def selected_layer(self) -> int:
        """Activation layer expected by this probe."""
        return int(self.metadata["selected_layer"])

    def predict_proba(self, activations: np.ndarray) -> np.ndarray:
        """Return class probabilities using the reconstructed pipeline."""
        return self.pipeline.predict_proba(activations)

    def predict(self, activations: np.ndarray) -> np.ndarray:
        """Return class predictions using the reconstructed pipeline."""
        return self.pipeline.predict(activations)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _contract_checksum(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(contract).encode("utf-8")).hexdigest()


def _json_copy(value: Any, *, field_name: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must contain finite JSON values") from error


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    field_name: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{field_name} keys do not match schema: expected "
            f"{sorted(expected)}, observed {sorted(observed)}"
        )


def _require_finite_numeric(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biuf":
        raise TypeError(f"probe state array {name!r} must have numeric dtype")
    if not np.isfinite(array).all():
        raise ValueError(f"probe state array {name!r} contains non-finite values")
    return array


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and bool(np.isfinite(value))
    )


def _validate_metric_set(metrics: Any, *, field_name: str) -> None:
    if not isinstance(metrics, dict) or set(metrics) != {"auc", "accuracy", "r2"}:
        raise ValueError(f"{field_name} must contain exactly auc, accuracy, and r2")
    if not all(_is_finite_number(value) for value in metrics.values()):
        raise ValueError(f"{field_name} values must be finite numbers")
    if not 0.0 <= float(metrics["auc"]) <= 1.0:
        raise ValueError(f"{field_name}.auc must be between zero and one")
    if not 0.0 <= float(metrics["accuracy"]) <= 1.0:
        raise ValueError(f"{field_name}.accuracy must be between zero and one")
    if float(metrics["r2"]) > 1.0:
        raise ValueError(f"{field_name}.r2 cannot exceed one")


def _fitted_array(estimator: Any, attribute: str, array_name: str) -> np.ndarray:
    if not hasattr(estimator, attribute):
        raise ValueError(
            f"pipeline step is not fitted: missing {type(estimator).__name__}."
            f"{attribute}"
        )
    return _require_finite_numeric(array_name, getattr(estimator, attribute))


def _validate_metadata(
    metadata: Mapping[str, Any],
    split_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_copy = _json_copy(dict(metadata), field_name="metadata")
    provenance_copy = _json_copy(
        dict(split_provenance), field_name="split_provenance"
    )
    required_metadata = {
        "selected_layer",
        "selected_hyperparameters",
        "selection_table",
        "test_metrics",
        "dataset_sha256",
    }
    missing_metadata = required_metadata - set(metadata_copy)
    if missing_metadata:
        raise ValueError(
            "headline probe metadata is missing: "
            + ", ".join(sorted(missing_metadata))
        )
    selected_layer = metadata_copy["selected_layer"]
    if (
        isinstance(selected_layer, bool)
        or not isinstance(selected_layer, int)
        or selected_layer < 0
    ):
        raise TypeError("metadata.selected_layer must be a non-negative integer")
    if not isinstance(metadata_copy["selected_hyperparameters"], dict):
        raise TypeError("metadata.selected_hyperparameters must be an object")
    selection_table = metadata_copy["selection_table"]
    if not isinstance(selection_table, list) or not selection_table:
        raise ValueError("metadata.selection_table must be a non-empty array")
    _validate_metric_set(
        metadata_copy["test_metrics"], field_name="metadata.test_metrics"
    )
    dataset_sha256 = metadata_copy["dataset_sha256"]
    if not isinstance(dataset_sha256, str) or not _SHA256_PATTERN.fullmatch(
        dataset_sha256
    ):
        raise ValueError("metadata.dataset_sha256 must be a 64-hex digest")
    for index, row in enumerate(selection_table):
        if not isinstance(row, dict) or set(row) != {
            "layer", "hyperparameters", "selection_partition",
            "development_metrics",
        }:
            raise ValueError(
                f"metadata.selection_table[{index}] has an invalid schema"
            )
        if (
            isinstance(row["layer"], bool)
            or not isinstance(row["layer"], int)
            or row["layer"] < 0
        ):
            raise ValueError("selection row layer must be a non-negative integer")
        if row["selection_partition"] != "development":
            raise ValueError("all selection rows must use the development partition")
        parameters = row["hyperparameters"]
        if not isinstance(parameters, dict) or set(parameters) != {
            "C", "n_components"
        }:
            raise ValueError("selection row hyperparameters have an invalid schema")
        if not _is_finite_number(parameters["C"]) or parameters["C"] <= 0:
            raise ValueError("selection row C must be positive and finite")
        components = parameters["n_components"]
        if components is not None and (
            isinstance(components, bool)
            or not isinstance(components, int)
            or components < 1
        ):
            raise ValueError("selection row n_components must be null or positive")
        _validate_metric_set(
            row["development_metrics"],
            field_name=f"metadata.selection_table[{index}].development_metrics",
        )

    selected_matches = [
        row for row in selection_table
        if row["layer"] == selected_layer
        and row["hyperparameters"] == metadata_copy["selected_hyperparameters"]
    ]
    if len(selected_matches) != 1:
        raise ValueError(
            "selected layer/hyperparameters must identify exactly one selection row"
        )

    required_provenance = {
        "split_source",
        "partition_sizes",
        "partition_group_ids",
        "selection_fit_partitions",
        "final_fit_partitions",
        "test_evaluations",
        "selected_layer",
        "manifest_id",
        "split_locked",
        "dataset_sha256",
    }
    missing_provenance = required_provenance - set(provenance_copy)
    if missing_provenance:
        raise ValueError(
            "headline split provenance is missing: "
            + ", ".join(sorted(missing_provenance))
        )
    if provenance_copy["selected_layer"] != selected_layer:
        raise ValueError("metadata and split provenance selected_layer disagree")
    if provenance_copy["selection_fit_partitions"] != ["train"]:
        raise ValueError("selection fit provenance must name only train")
    if provenance_copy["final_fit_partitions"] != ["train", "development"]:
        raise ValueError("final fit provenance must name train and development")
    if provenance_copy["test_evaluations"] != 1:
        raise ValueError("headline artifact must record exactly one test evaluation")
    if provenance_copy["split_locked"] is not True:
        raise ValueError("headline split provenance must be locked")
    if (
        not isinstance(provenance_copy["manifest_id"], str)
        or not _SHA256_PATTERN.fullmatch(provenance_copy["manifest_id"])
    ):
        raise ValueError("split provenance manifest_id must be a 64-hex digest")
    if provenance_copy["dataset_sha256"] != dataset_sha256:
        raise ValueError("metadata and split provenance dataset hashes disagree")
    if not isinstance(provenance_copy["split_source"], str) or not (
        provenance_copy["split_source"]
    ):
        raise TypeError("split_provenance.split_source must be a non-empty string")
    for field_name in ("partition_sizes", "partition_group_ids"):
        partition_value = provenance_copy[field_name]
        if not isinstance(partition_value, dict) or set(partition_value) != {
            "train",
            "development",
            "test",
        }:
            raise ValueError(
                f"split_provenance.{field_name} must cover all outer partitions"
            )
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in provenance_copy["partition_sizes"].values()
    ):
        raise ValueError("all split partition sizes must be positive integers")
    partition_groups = provenance_copy["partition_group_ids"]
    if any(
        not isinstance(group_ids, list)
        or not group_ids
        or any(
            not isinstance(group_id, str) or not group_id
            for group_id in group_ids
        )
        for group_ids in partition_groups.values()
    ):
        raise ValueError("all split partitions must name non-empty string group IDs")
    group_sets = [set(partition_groups[name]) for name in _PARTITION_NAMES]
    if any(
        group_sets[left] & group_sets[right]
        for left in range(len(group_sets))
        for right in range(left + 1, len(group_sets))
    ):
        raise ValueError("split provenance group IDs cross outer partitions")
    selected_development = provenance_copy.get("selected_development_metrics")
    if selected_development != selected_matches[0]["development_metrics"]:
        raise ValueError(
            "selected development metrics do not match the selected table row"
        )
    return metadata_copy, provenance_copy


def _serialize_pipeline(
    pipeline: Pipeline,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if type(pipeline) is not Pipeline:
        raise TypeError("headline artifact requires an sklearn Pipeline")
    step_names = tuple(name for name, _ in pipeline.steps)
    if step_names not in (_BASE_STEPS, _PCA_STEPS):
        raise TypeError(
            "unsupported headline pipeline steps; expected scaler, optional pca, "
            "then probe"
        )
    if pipeline.memory is not None:
        raise TypeError("pipeline caching is not supported in public artifacts")

    scaler = pipeline.named_steps["scaler"]
    probe = pipeline.named_steps["probe"]
    if type(scaler) is not StandardScaler:
        raise TypeError("scaler step must be exactly StandardScaler")
    if type(probe) is not LogisticRegression:
        raise TypeError("probe step must be exactly LogisticRegression")
    if not scaler.with_mean or not scaler.with_std:
        raise TypeError(
            "public headline artifacts require mean-centering and variance scaling"
        )

    scaler_mean = _fitted_array(scaler, "mean_", "scaler_mean")
    scaler_scale = _fitted_array(scaler, "scale_", "scaler_scale")
    scaler_var = _fitted_array(scaler, "var_", "scaler_var")
    if scaler_mean.ndim != 1 or scaler_mean.size == 0:
        raise ValueError("scaler_mean must be a non-empty vector")
    input_features = int(scaler_mean.size)
    if scaler_scale.shape != (input_features,) or scaler_var.shape != (
        input_features,
    ):
        raise ValueError("StandardScaler fitted vectors have inconsistent shapes")
    if np.any(scaler_scale <= 0) or np.any(scaler_var < 0):
        raise ValueError("StandardScaler fitted scale/variance state is invalid")
    scaler_seen = _fitted_array(
        scaler, "n_samples_seen_", "scaler_n_samples_seen"
    )
    if scaler_seen.shape not in ((), (input_features,)):
        raise ValueError("StandardScaler n_samples_seen_ has an unsupported shape")

    arrays: dict[str, np.ndarray] = {
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "scaler_var": scaler_var,
        "scaler_n_samples_seen": scaler_seen,
    }
    pipeline_contract: dict[str, Any] = {
        "steps": list(step_names),
        "input_features": input_features,
        "scaler": {
            "with_mean": True,
            "with_std": True,
            "n_samples_seen_shape": list(scaler_seen.shape),
        },
        "pca": None,
    }

    transformed_features = input_features
    if "pca" in pipeline.named_steps:
        pca = pipeline.named_steps["pca"]
        if type(pca) is not PCA:
            raise TypeError("pca step must be exactly PCA")
        pca_components = _fitted_array(pca, "components_", "pca_components")
        if pca_components.ndim != 2 or pca_components.shape[1] != input_features:
            raise ValueError("PCA components do not align with scaler output")
        transformed_features = int(pca_components.shape[0])
        if transformed_features == 0:
            raise ValueError("PCA must retain at least one component")
        arrays.update({
            "pca_components": pca_components,
            "pca_mean": _fitted_array(pca, "mean_", "pca_mean"),
            "pca_explained_variance": _fitted_array(
                pca, "explained_variance_", "pca_explained_variance"
            ),
            "pca_explained_variance_ratio": _fitted_array(
                pca,
                "explained_variance_ratio_",
                "pca_explained_variance_ratio",
            ),
            "pca_singular_values": _fitted_array(
                pca, "singular_values_", "pca_singular_values"
            ),
            "pca_noise_variance": _fitted_array(
                pca, "noise_variance_", "pca_noise_variance"
            ),
        })
        if arrays["pca_mean"].shape != (input_features,):
            raise ValueError("PCA mean does not align with scaler output")
        for name in (
            "pca_explained_variance",
            "pca_explained_variance_ratio",
            "pca_singular_values",
        ):
            if arrays[name].shape != (transformed_features,):
                raise ValueError(f"{name} does not align with PCA components")
        if arrays["pca_noise_variance"].shape != ():
            raise ValueError("PCA noise_variance_ must be scalar")
        pipeline_contract["pca"] = {
            "n_components": transformed_features,
            "n_samples": int(pca.n_samples_),
            "whiten": bool(pca.whiten),
        }

    probe_classes = _fitted_array(probe, "classes_", "probe_classes")
    probe_coef = _fitted_array(probe, "coef_", "probe_coef")
    probe_intercept = _fitted_array(probe, "intercept_", "probe_intercept")
    probe_iterations = _fitted_array(probe, "n_iter_", "probe_n_iter")
    if probe_classes.shape != (2,):
        raise ValueError("headline logistic probe must have exactly two classes")
    if not np.array_equal(probe_classes, np.asarray([0, 1])):
        raise ValueError("headline logistic probe classes must be exactly [0, 1]")
    if probe_coef.shape != (1, transformed_features):
        raise ValueError("logistic coefficients do not align with pipeline output")
    if probe_intercept.shape != (1,) or probe_iterations.shape != (1,):
        raise ValueError("binary logistic fitted state has inconsistent shapes")
    if probe_iterations.dtype.kind not in "iu" or np.any(probe_iterations <= 0):
        raise ValueError("binary logistic iteration state must be positive integers")
    if not _is_finite_number(probe.C) or float(probe.C) <= 0:
        raise ValueError("logistic C must be a positive finite number")
    if str(probe.solver) not in _SUPPORTED_SOLVERS:
        raise TypeError(f"unsupported logistic solver: {probe.solver!r}")
    arrays.update({
        "probe_classes": probe_classes,
        "probe_coef": probe_coef,
        "probe_intercept": probe_intercept,
        "probe_n_iter": probe_iterations,
    })
    pipeline_contract["transformed_features"] = transformed_features
    pipeline_contract["probe"] = {
        "C": float(probe.C),
        "fit_intercept": bool(probe.fit_intercept),
        "solver": str(probe.solver),
        "positive_class": 1,
    }
    return arrays, pipeline_contract


def _validate_metadata_pipeline_consistency(
    metadata: Mapping[str, Any],
    pipeline_contract: Mapping[str, Any],
) -> None:
    selected = metadata["selected_hyperparameters"]
    if "C" not in selected or not _is_finite_number(selected["C"]):
        raise ValueError("selected_hyperparameters.C must be a finite number")
    if not np.isclose(
        float(selected["C"]),
        float(pipeline_contract["probe"]["C"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("selected logistic C disagrees with fitted pipeline")
    selected_components = selected.get("n_components")
    pca_contract = pipeline_contract["pca"]
    fitted_components = (
        None if pca_contract is None else pca_contract["n_components"]
    )
    if selected_components != fitted_components:
        raise ValueError("selected PCA components disagree with fitted pipeline")


def save_headline_probe_artifact(
    base_path: str | Path,
    pipeline: Pipeline,
    *,
    metadata: Mapping[str, Any],
    split_provenance: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Save one whitelisted fitted headline pipeline as JSON plus NPZ arrays."""
    metadata_copy, provenance_copy = _validate_metadata(
        metadata, split_provenance
    )
    arrays, pipeline_contract = _serialize_pipeline(pipeline)
    _validate_metadata_pipeline_consistency(metadata_copy, pipeline_contract)
    from interpretability.schema_registry import schema_registry_checksum

    contract = {
        "artifact_type": _ARTIFACT_TYPE,
        "schema_version": HEADLINE_PROBE_SCHEMA_VERSION,
        "schema_registry_checksum": schema_registry_checksum(),
        "sklearn_version": sklearn.__version__,
        "pipeline": pipeline_contract,
        "metadata": metadata_copy,
        "split_provenance": provenance_copy,
    }
    manifest = {
        "contract": contract,
        "contract_sha256": _contract_checksum(contract),
    }
    return save_array_bundle(base_path, arrays, manifest)


def _validated_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        manifest,
        {"contract", "contract_sha256"},
        field_name="probe artifact manifest",
    )
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise TypeError("probe artifact contract must be an object")
    _require_exact_keys(
        contract,
        {
            "artifact_type",
            "schema_version",
            "schema_registry_checksum",
            "sklearn_version",
            "pipeline",
            "metadata",
            "split_provenance",
        },
        field_name="probe artifact contract",
    )
    if contract["artifact_type"] != _ARTIFACT_TYPE:
        raise ValueError("unsupported artifact type")
    if contract["schema_version"] != HEADLINE_PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported headline probe schema version")
    from interpretability.schema_registry import schema_registry_checksum

    if contract["schema_registry_checksum"] != schema_registry_checksum():
        raise ValueError("headline probe schema-registry checksum mismatch")
    if not isinstance(contract["sklearn_version"], str) or not contract[
        "sklearn_version"
    ]:
        raise TypeError("artifact sklearn_version must be a non-empty string")
    if manifest["contract_sha256"] != _contract_checksum(contract):
        raise ValueError("probe artifact manifest checksum mismatch")
    return contract


def _validate_loaded_arrays(
    arrays: Mapping[str, np.ndarray],
    pipeline_contract: Mapping[str, Any],
) -> None:
    if not isinstance(pipeline_contract, dict):
        raise TypeError("pipeline contract must be an object")
    _require_exact_keys(
        pipeline_contract,
        {
            "steps",
            "input_features",
            "transformed_features",
            "scaler",
            "pca",
            "probe",
        },
        field_name="pipeline contract",
    )
    steps = tuple(pipeline_contract["steps"])
    if steps not in (_BASE_STEPS, _PCA_STEPS):
        raise ValueError("artifact declares unsupported pipeline steps")
    input_features = pipeline_contract["input_features"]
    transformed_features = pipeline_contract["transformed_features"]
    if (
        isinstance(input_features, bool)
        or not isinstance(input_features, int)
        or input_features <= 0
        or isinstance(transformed_features, bool)
        or not isinstance(transformed_features, int)
        or transformed_features <= 0
    ):
        raise ValueError("pipeline feature counts must be positive integers")

    expected_shapes: dict[str, tuple[int, ...]] = {
        "scaler_mean": (input_features,),
        "scaler_scale": (input_features,),
        "scaler_var": (input_features,),
        "probe_classes": (2,),
        "probe_coef": (1, transformed_features),
        "probe_intercept": (1,),
        "probe_n_iter": (1,),
    }
    scaler_contract = pipeline_contract["scaler"]
    if not isinstance(scaler_contract, dict):
        raise TypeError("scaler contract must be an object")
    _require_exact_keys(
        scaler_contract,
        {"with_mean", "with_std", "n_samples_seen_shape"},
        field_name="scaler contract",
    )
    if (
        scaler_contract["with_mean"] is not True
        or scaler_contract["with_std"] is not True
    ):
        raise ValueError("artifact scaler contract is unsupported")
    seen_shape = tuple(scaler_contract["n_samples_seen_shape"])
    if seen_shape not in ((), (input_features,)):
        raise ValueError("artifact scaler sample-count shape is invalid")
    expected_shapes["scaler_n_samples_seen"] = seen_shape

    pca_contract = pipeline_contract["pca"]
    if steps == _PCA_STEPS:
        if not isinstance(pca_contract, dict):
            raise TypeError("PCA pipeline requires a PCA contract")
        _require_exact_keys(
            pca_contract,
            {"n_components", "n_samples", "whiten"},
            field_name="PCA contract",
        )
        if pca_contract["n_components"] != transformed_features:
            raise ValueError("PCA component count disagrees with transformed features")
        if (
            isinstance(pca_contract["n_samples"], bool)
            or not isinstance(pca_contract["n_samples"], int)
            or pca_contract["n_samples"] <= 0
            or not isinstance(pca_contract["whiten"], bool)
        ):
            raise ValueError("PCA fitted-count and whitening state is invalid")
        expected_shapes.update({
            "pca_components": (transformed_features, input_features),
            "pca_mean": (input_features,),
            "pca_explained_variance": (transformed_features,),
            "pca_explained_variance_ratio": (transformed_features,),
            "pca_singular_values": (transformed_features,),
            "pca_noise_variance": (),
        })
    elif pca_contract is not None or transformed_features != input_features:
        raise ValueError("non-PCA pipeline declares inconsistent transformed state")

    probe_contract = pipeline_contract["probe"]
    if not isinstance(probe_contract, dict):
        raise TypeError("probe contract must be an object")
    _require_exact_keys(
        probe_contract,
        {"C", "fit_intercept", "solver", "positive_class"},
        field_name="probe contract",
    )
    if not _is_finite_number(probe_contract["C"]) or probe_contract["C"] <= 0:
        raise ValueError("artifact logistic C must be a positive finite number")
    if not isinstance(probe_contract["fit_intercept"], bool):
        raise TypeError("artifact logistic fit_intercept must be boolean")
    if probe_contract["solver"] not in _SUPPORTED_SOLVERS:
        raise ValueError("artifact logistic solver is unsupported")
    if probe_contract["positive_class"] != 1:
        raise ValueError("artifact positive class must be one")
    if set(arrays) != set(expected_shapes):
        raise ValueError("probe artifact array members do not match pipeline schema")
    for name, expected_shape in expected_shapes.items():
        array = _require_finite_numeric(name, arrays[name])
        if array.shape != expected_shape:
            raise ValueError(
                f"probe state array {name!r} has shape {array.shape}, "
                f"expected {expected_shape}"
            )
    if np.any(arrays["scaler_scale"] <= 0) or np.any(
        arrays["scaler_var"] < 0
    ):
        raise ValueError("artifact scaler scale/variance state is invalid")
    if not np.array_equal(arrays["probe_classes"], np.asarray([0, 1])):
        raise ValueError("artifact logistic classes must be exactly [0, 1]")
    if arrays["probe_n_iter"].dtype.kind not in "iu" or np.any(
        arrays["probe_n_iter"] <= 0
    ):
        raise ValueError("artifact logistic iterations must be positive integers")


def _reconstruct_pipeline(
    arrays: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> Pipeline:
    pipeline_contract = contract["pipeline"]
    _validate_loaded_arrays(arrays, pipeline_contract)
    input_features = int(pipeline_contract["input_features"])

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.mean_ = arrays["scaler_mean"].copy()
    scaler.scale_ = arrays["scaler_scale"].copy()
    scaler.var_ = arrays["scaler_var"].copy()
    seen = arrays["scaler_n_samples_seen"].copy()
    scaler.n_samples_seen_ = seen.item() if seen.shape == () else seen
    scaler.n_features_in_ = input_features
    steps: list[tuple[str, Any]] = [("scaler", scaler)]

    pca_contract = pipeline_contract["pca"]
    if pca_contract is not None:
        pca = PCA(
            n_components=int(pca_contract["n_components"]),
            whiten=bool(pca_contract["whiten"]),
        )
        pca.components_ = arrays["pca_components"].copy()
        pca.mean_ = arrays["pca_mean"].copy()
        pca.explained_variance_ = arrays["pca_explained_variance"].copy()
        pca.explained_variance_ratio_ = arrays[
            "pca_explained_variance_ratio"
        ].copy()
        pca.singular_values_ = arrays["pca_singular_values"].copy()
        pca.noise_variance_ = arrays["pca_noise_variance"].item()
        pca.n_components_ = int(pca_contract["n_components"])
        pca.n_features_in_ = input_features
        pca.n_samples_ = int(pca_contract["n_samples"])
        steps.append(("pca", pca))

    probe_contract = pipeline_contract["probe"]
    probe = LogisticRegression(
        C=float(probe_contract["C"]),
        fit_intercept=bool(probe_contract["fit_intercept"]),
        solver=str(probe_contract["solver"]),
    )
    probe.classes_ = arrays["probe_classes"].copy()
    probe.coef_ = arrays["probe_coef"].copy()
    probe.intercept_ = arrays["probe_intercept"].copy()
    probe.n_iter_ = arrays["probe_n_iter"].copy()
    probe.n_features_in_ = int(pipeline_contract["transformed_features"])
    steps.append(("probe", probe))
    return Pipeline(steps)


def load_headline_probe_artifact(
    manifest_path: str | Path,
) -> HeadlineProbeArtifact:
    """Load and reconstruct a fitted headline probe without pickle execution."""
    arrays, manifest = load_array_bundle(manifest_path)
    contract = _validated_contract(manifest)
    metadata, provenance = _validate_metadata(
        contract["metadata"], contract["split_provenance"]
    )
    pipeline = _reconstruct_pipeline(arrays, contract)
    _validate_metadata_pipeline_consistency(metadata, contract["pipeline"])
    return HeadlineProbeArtifact(
        pipeline=pipeline,
        metadata=metadata,
        split_provenance=provenance,
    )


__all__: Sequence[str] = (
    "HEADLINE_PROBE_SCHEMA_VERSION",
    "HeadlineProbeArtifact",
    "load_headline_probe_artifact",
    "save_headline_probe_artifact",
)
