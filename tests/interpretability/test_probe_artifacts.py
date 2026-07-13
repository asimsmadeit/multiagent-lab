"""Security and fidelity contracts for public fitted probe artifacts."""

import hashlib
import json

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from interpretability.probes.artifacts import (
    HEADLINE_PROBE_SCHEMA_VERSION,
    load_headline_probe_artifact,
    save_headline_probe_artifact,
)
from interpretability.probes.metrics import evaluate_nested_grouped_layers


def _training_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(73)
    activations = rng.normal(size=(96, 7))
    logits = (
        1.7 * activations[:, 0]
        - 0.8 * activations[:, 2]
        + 0.3 * activations[:, 5]
    )
    labels = (logits > np.median(logits)).astype(int)
    return activations, labels


def _pipeline(*, use_pca: bool) -> Pipeline:
    activations, labels = _training_data()
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=4, random_state=9)))
    steps.append(("probe", LogisticRegression(C=0.7, random_state=9)))
    return Pipeline(steps).fit(activations, labels)


def _metadata(*, use_pca: bool) -> dict:
    return {
        "selected_layer": 12,
        "selected_hyperparameters": {
            "C": 0.7,
            "n_components": 4 if use_pca else None,
        },
        "selection_table": [
            {
                "layer": 12,
                "hyperparameters": {
                    "C": 0.7,
                    "n_components": 4 if use_pca else None,
                },
                "selection_partition": "development",
                "development_metrics": {
                    "auc": 0.91,
                    "accuracy": 0.83,
                    "r2": 0.24,
                },
            }
        ],
        "test_metrics": {"auc": 0.89, "accuracy": 0.84, "r2": 0.31},
        "dataset_id": "synthetic-fixture",
        "dataset_sha256": "a" * 64,
    }


def _provenance() -> dict:
    return {
        "split_source": "split_manifest",
        "manifest_id": "b" * 64,
        "split_locked": True,
        "dataset_sha256": "a" * 64,
        "partition_sizes": {"train": 40, "development": 24, "test": 32},
        "partition_group_ids": {
            "train": ["group-a"],
            "development": ["group-b"],
            "test": ["group-c"],
        },
        "selection_fit_partitions": ["train"],
        "final_fit_partitions": ["train", "development"],
        "test_evaluations": 1,
        "selected_development_metrics": {
            "auc": 0.91,
            "accuracy": 0.83,
            "r2": 0.24,
        },
        "selected_layer": 12,
    }


@pytest.mark.parametrize("use_pca", [False, True])
def test_logistic_pipeline_roundtrip_reproduces_inference_exactly(
    tmp_path,
    monkeypatch,
    use_pca,
):
    activations, _ = _training_data()
    pipeline = _pipeline(use_pca=use_pca)
    _, manifest_path = save_headline_probe_artifact(
        tmp_path / f"probe-{use_pca}",
        pipeline,
        metadata=_metadata(use_pca=use_pca),
        split_provenance=_provenance(),
    )

    def forbid_torch_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("safe probe loading must never invoke torch.load")

    monkeypatch.setattr("torch.load", forbid_torch_load)
    loaded = load_headline_probe_artifact(manifest_path)

    np.testing.assert_allclose(
        loaded.predict_proba(activations),
        pipeline.predict_proba(activations),
        rtol=1e-13,
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        loaded.predict(activations), pipeline.predict(activations)
    )
    assert tuple(loaded.pipeline.named_steps) == (
        ("scaler", "pca", "probe")
        if use_pca
        else ("scaler", "probe")
    )
    assert loaded.selected_layer == 12
    assert loaded.metadata["dataset_id"] == "synthetic-fixture"
    assert loaded.split_provenance == _provenance()


def test_nested_evaluation_has_explicit_safe_artifact_api(tmp_path):
    rng = np.random.default_rng(81)
    groups = np.repeat([f"trial-{index}" for index in range(12)], 2)
    partitions = np.repeat(
        ["train"] * 4 + ["development"] * 4 + ["test"] * 4,
        2,
    )
    labels = np.tile([0.0, 1.0], 12)
    signal = labels[:, None] * 5.0 + rng.normal(scale=0.1, size=(24, 1))
    activations = {6: np.hstack((signal, rng.normal(size=(24, 5))))}
    evaluation = evaluate_nested_grouped_layers(
        activations,
        labels,
        partition_labels=partitions,
        groups=groups,
        hyperparameters=({"C": 1.0, "n_components": 3},),
    )

    _, manifest_path = evaluation.save_artifact(tmp_path / "nested-headline")
    loaded = load_headline_probe_artifact(manifest_path)
    locked_activations = activations[6][evaluation.test_indices]

    np.testing.assert_allclose(
        loaded.predict_proba(locked_activations)[:, 1],
        evaluation.test_probabilities,
        rtol=1e-13,
        atol=1e-15,
    )
    assert loaded.metadata["selected_hyperparameters"] == (
        evaluation.selected_hyperparameters
    )
    assert loaded.split_provenance == evaluation.split_provenance


def test_artifact_rejects_unsupported_steps_and_estimators(tmp_path):
    activations, labels = _training_data()
    unsupported_step = Pipeline([
        ("scaler", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=2)),
        ("probe", LogisticRegression()),
    ]).fit(activations, labels)
    wrong_step_name = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression()),
    ]).fit(activations, labels)

    for pipeline in (unsupported_step, wrong_step_name):
        with pytest.raises(TypeError, match="unsupported headline pipeline steps"):
            save_headline_probe_artifact(
                tmp_path / "unsupported",
                pipeline,
                metadata=_metadata(use_pca=False),
                split_provenance=_provenance(),
            )


def test_artifact_rejects_nonfinite_fitted_state_on_save(tmp_path):
    pipeline = _pipeline(use_pca=False)
    pipeline.named_steps["probe"].coef_[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        save_headline_probe_artifact(
            tmp_path / "nonfinite-save",
            pipeline,
            metadata=_metadata(use_pca=False),
            split_provenance=_provenance(),
        )


def test_artifact_rejects_array_checksum_and_shape_tampering(tmp_path):
    array_path, manifest_path = save_headline_probe_artifact(
        tmp_path / "tamper",
        _pipeline(use_pca=True),
        metadata=_metadata(use_pca=True),
        split_provenance=_provenance(),
    )
    original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_arrays = array_path.read_bytes()

    array_path.write_bytes(original_arrays + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_headline_probe_artifact(manifest_path)

    array_path.write_bytes(original_arrays)
    manifest_path.write_text(
        json.dumps(original_manifest, sort_keys=True), encoding="utf-8"
    )
    original_manifest["arrays"]["probe_coef"]["shape"] = [2, 4]
    manifest_path.write_text(
        json.dumps(original_manifest, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        load_headline_probe_artifact(manifest_path)


def test_artifact_rejects_manifest_and_schema_tampering(tmp_path):
    _, manifest_path = save_headline_probe_artifact(
        tmp_path / "manifest-tamper",
        _pipeline(use_pca=False),
        metadata=_metadata(use_pca=False),
        split_provenance=_provenance(),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["manifest"]["contract"]["metadata"]["selected_layer"] = 99
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest checksum"):
        load_headline_probe_artifact(manifest_path)

    payload["manifest"]["contract"]["schema_version"] = "999.0.0"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported headline probe schema"):
        load_headline_probe_artifact(manifest_path)


def test_artifact_rejects_nonfinite_arrays_even_with_valid_file_checksum(tmp_path):
    array_path, manifest_path = save_headline_probe_artifact(
        tmp_path / "nonfinite-load",
        _pipeline(use_pca=False),
        metadata=_metadata(use_pca=False),
        split_provenance=_provenance(),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(array_path, allow_pickle=False) as saved:
        arrays = {name: np.asarray(saved[name]) for name in saved.files}
    arrays["probe_coef"] = arrays["probe_coef"].copy()
    arrays["probe_coef"][0, 0] = np.inf
    np.savez_compressed(array_path, **arrays)
    payload["array_sha256"] = hashlib.sha256(array_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite"):
        load_headline_probe_artifact(manifest_path)


def test_public_manifest_records_versioned_non_pickle_contract(tmp_path):
    array_path, manifest_path = save_headline_probe_artifact(
        tmp_path / "contract",
        _pipeline(use_pca=True),
        metadata=_metadata(use_pca=True),
        split_provenance=_provenance(),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = payload["manifest"]["contract"]

    assert contract["schema_version"] == HEADLINE_PROBE_SCHEMA_VERSION
    from interpretability.schema_registry import schema_registry_checksum

    assert contract["schema_registry_checksum"] == schema_registry_checksum()
    assert contract["artifact_type"] == "headline_logistic_probe"
    assert contract["pipeline"]["steps"] == ["scaler", "pca", "probe"]
    assert array_path.suffix == ".npz"
    assert manifest_path.suffix == ".json"
    assert not list(tmp_path.glob("*.pkl"))
    assert not list(tmp_path.glob("*.pt"))


def test_artifact_rejects_substituted_schema_registry_even_if_rehashed(
    tmp_path,
) -> None:
    _, manifest_path = save_headline_probe_artifact(
        tmp_path / "registry-tamper",
        _pipeline(use_pca=False),
        metadata=_metadata(use_pca=False),
        split_provenance=_provenance(),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = payload["manifest"]["contract"]
    contract["schema_registry_checksum"] = "f" * 64
    payload["manifest"]["contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="schema-registry checksum"):
        load_headline_probe_artifact(manifest_path)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda metadata, provenance: metadata.update(selection_table=[]),
            "non-empty",
        ),
        (
            lambda metadata, provenance: metadata["selection_table"][0].update(
                selection_partition="test"
            ),
            "development partition",
        ),
        (
            lambda metadata, provenance: metadata["test_metrics"].update(auc=9),
            "between zero and one",
        ),
        (
            lambda metadata, provenance: provenance.update(
                dataset_sha256="c" * 64
            ),
            "dataset hashes disagree",
        ),
    ],
)
def test_artifact_rejects_forged_scientific_metadata(
    tmp_path, mutator, match
) -> None:
    metadata = _metadata(use_pca=False)
    provenance = _provenance()
    mutator(metadata, provenance)

    with pytest.raises(ValueError, match=match):
        save_headline_probe_artifact(
            tmp_path / "forged",
            _pipeline(use_pca=False),
            metadata=metadata,
            split_provenance=provenance,
        )


def test_artifact_rejects_noncanonical_binary_classes(tmp_path) -> None:
    pipeline = _pipeline(use_pca=False)
    pipeline.named_steps["probe"].classes_ = np.asarray([5, 9])

    with pytest.raises(ValueError, match=r"exactly \[0, 1\]"):
        save_headline_probe_artifact(
            tmp_path / "wrong-classes",
            pipeline,
            metadata=_metadata(use_pca=False),
            split_provenance=_provenance(),
        )
