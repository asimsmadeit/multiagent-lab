"""Contracts for preregistered causal intervention manifests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from interpretability.causal.design import (
    CAUSAL_DESIGN_SCHEMA_VERSION,
    CausalDesignManifest,
    ControlKind,
    DirectionVectorIdentity,
    InterventionKind,
)


PRIMARY = np.asarray([1.0, 0.0, 0.0, 0.0])
POSITIVE = np.asarray([0.0, 0.0, 0.0, 1.0])


def _manifest(**overrides) -> CausalDesignManifest:
    values = {
        "study_id": "study-1",
        "dataset_hash": "sha256:" + "d" * 64,
        "intervention": InterventionKind.STEER,
        "intervention_adapter_version": "fake-hook-adapter/1",
        "direction_source": "outer-train-fold",
        "primary_direction_identity": DirectionVectorIdentity.from_vector(
            "outer-train-fold", PRIMARY
        ),
        "layer": 12,
        "token_stage": "last_response_token",
        "coefficients": (-1.0, 0.0, 1.0),
        "outcomes": ("logit", "deception_behavior", "fluency", "utility"),
        "independent_unit": "trial_family",
        "group_key": "trial_family_id",
        "controls": (
            ControlKind.ZERO_HOOK,
            ControlKind.NORM_MATCHED_RANDOM,
            ControlKind.LABEL_SHUFFLED,
            ControlKind.POSITIVE_HOOK,
        ),
        "repetitions": 20,
        "random_seed": 42,
        "scorer_version": "hidden-value-rule/2",
        "prompt_ids": ("prompt-1", "prompt-2"),
        "positive_control_source": "known-hook-effect/1",
        "positive_direction_identity": DirectionVectorIdentity.from_vector(
            "known-hook-effect/1", POSITIVE
        ),
    }
    values.update(overrides)
    return CausalDesignManifest(**values)


def test_causal_design_is_stable_json_round_trip_and_tamper_evident() -> None:
    manifest = _manifest()
    payload = json.loads(json.dumps(manifest.to_dict()))

    assert CausalDesignManifest.from_dict(payload) == manifest
    payload["layer"] = 13
    with pytest.raises(ValueError, match="design_id"):
        CausalDesignManifest.from_dict(payload)

    missing_id = manifest.to_dict()
    del missing_id["design_id"]
    with pytest.raises(ValueError, match="require design_id"):
        CausalDesignManifest.from_dict(missing_id)

    missing_registry = manifest.to_dict()
    del missing_registry["schema_registry_checksum"]
    with pytest.raises(ValueError, match="schema-registry checksum"):
        CausalDesignManifest.from_dict(missing_registry)


def test_behavioral_design_requires_scorer_and_matched_controls() -> None:
    with pytest.raises(ValueError, match="controls"):
        _manifest(
            controls=(ControlKind.ZERO_HOOK,),
            positive_control_source=None,
            positive_direction_identity=None,
        )
    with pytest.raises(ValueError, match="scorer_version"):
        _manifest(scorer_version=None)


def test_unlocked_or_ungrouped_confirmatory_design_fails_closed() -> None:
    with pytest.raises(ValueError, match="locked"):
        _manifest(locked=False)
    with pytest.raises(ValueError, match="independent_unit"):
        _manifest(independent_unit="")
    with pytest.raises(TypeError, match="boolean"):
        _manifest(locked=1)


def test_named_controls_are_frozen_into_manifest_identity() -> None:
    with pytest.raises(ValueError, match="declared direction IDs"):
        _manifest(
            controls=(*_manifest().controls, ControlKind.NUISANCE_DIRECTION),
        )
    with pytest.raises(ValueError, match="positive_control_source"):
        _manifest(positive_control_source=None)

    manifest = _manifest(
        controls=(*_manifest().controls, ControlKind.NUISANCE_DIRECTION),
        nuisance_direction_ids=("role/1", "surface/1"),
        nuisance_direction_identities=(
            DirectionVectorIdentity.from_vector(
                "role/1", np.asarray([0.0, 1.0, 0.0, 0.0])
            ),
            DirectionVectorIdentity.from_vector(
                "surface/1", np.asarray([0.0, 0.0, 1.0, 0.0])
            ),
        ),
    )
    assert manifest.to_dict()["nuisance_direction_ids"] == ["role/1", "surface/1"]


def test_direction_hashes_and_dimensions_are_bound_into_design_identity() -> None:
    float32_identity = DirectionVectorIdentity.from_vector(
        "outer-train-fold", PRIMARY.astype(np.float32)
    )
    float64_identity = DirectionVectorIdentity.from_vector(
        "outer-train-fold", PRIMARY.astype(np.float64)
    )

    assert float32_identity == float64_identity
    assert float64_identity.dimension == 4
    assert len(float64_identity.sha256) == len("sha256:") + 64

    changed = _manifest(
        primary_direction_identity=DirectionVectorIdentity.from_vector(
            "outer-train-fold", np.asarray([0.0, 1.0, 0.0, 0.0])
        )
    )
    assert changed.design_id != _manifest().design_id


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("schema_version", "1.1.0", "schema_version"),
        ("dataset_hash", "sha256:not-a-digest", "canonical SHA256"),
        ("schema_registry_checksum", "f" * 64, "checksum mismatch"),
        ("coefficients", (0.0, float("nan")), "finite"),
        ("coefficients", (0.0, 0.0), "duplicates"),
        ("outcomes", ("logit", "logit"), "duplicates"),
        ("prompt_ids", ("prompt-1", "prompt-1"), "duplicates"),
        ("random_seed", -1, "non-negative"),
        ("n_bootstrap", 0, "positive integer"),
        ("n_permutations", True, "positive integer"),
        ("alpha", float("nan"), "alpha"),
        ("alpha", 1.0, "alpha"),
    ],
)
def test_manifest_rejects_hostile_locked_fields(
    field_name: str, bad_value, match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _manifest(**{field_name: bad_value})


def test_serialized_manifest_rejects_missing_or_wrong_schema_and_locked_type() -> None:
    payload = _manifest().to_dict()
    payload.pop("schema_version")
    with pytest.raises(ValueError, match="schema_version"):
        CausalDesignManifest.from_dict(payload)

    payload = _manifest().to_dict()
    payload["locked"] = "true"
    with pytest.raises(TypeError, match="boolean"):
        CausalDesignManifest.from_dict(payload)

    assert _manifest().schema_version == CAUSAL_DESIGN_SCHEMA_VERSION == "1.3.0"


def test_direction_identity_rejects_nonfinite_or_wrong_dimension_vectors() -> None:
    with pytest.raises(ValueError, match="finite"):
        DirectionVectorIdentity.from_vector("source", [1.0, float("inf")])
    with pytest.raises(ValueError, match="one-dimensional"):
        DirectionVectorIdentity.from_vector("source", [[1.0, 2.0]])
