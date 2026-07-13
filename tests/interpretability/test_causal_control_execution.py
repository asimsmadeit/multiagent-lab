"""Execution contracts for every control in a locked causal design."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from interpretability.causal.causal_validation import run_full_causal_validation
from interpretability.causal.design import (
    CausalDesignManifest,
    ControlKind,
    DirectionVectorIdentity,
    InterventionKind,
)
from interpretability.causal.execution import (
    CausalExecutionInputs,
    InterventionApplicationReceipt,
    execute_manifest_controls,
)


OUTCOMES = (
    "logit",
    "deception_behavior",
    "fluency",
    "utility",
    "counterpart_outcome",
)

PRIMARY = np.asarray([1.0, 0.0, 0.0, 0.0])
NUISANCE = {
    "role-identity/1": np.asarray([0.0, 1.0, 0.0, 0.0]),
    "surface-form/1": np.asarray([0.0, 0.0, 1.0, 0.0]),
}
POSITIVE = np.asarray([0.0, 0.0, 0.0, 1.0])


def _manifest(**overrides) -> CausalDesignManifest:
    values = {
        "study_id": "control-study",
        "dataset_hash": "sha256:" + "d" * 64,
        "intervention": InterventionKind.STEER,
        "intervention_adapter_version": "fake-hook-adapter/1",
        "direction_source": "outer-train-fold-0",
        "primary_direction_identity": DirectionVectorIdentity.from_vector(
            "outer-train-fold-0", PRIMARY
        ),
        "layer": 3,
        "token_stage": "last_response_token",
        "coefficients": (-1.0, 1.0),
        "outcomes": OUTCOMES,
        "independent_unit": "trial_family",
        "group_key": "trial_family_id",
        "controls": tuple(ControlKind),
        "repetitions": 2,
        "random_seed": 71,
        "n_bootstrap": 50,
        "n_permutations": 50,
        "alpha": 0.05,
        "scorer_version": "behavior-rule/3",
        "prompt_ids": ("prompt-a", "prompt-b", "prompt-c"),
        "nuisance_direction_ids": ("role-identity/1", "surface-form/1"),
        "nuisance_direction_identities": tuple(
            DirectionVectorIdentity.from_vector(source, vector)
            for source, vector in NUISANCE.items()
        ),
        "positive_control_source": "known-hook-effect/1",
        "positive_direction_identity": DirectionVectorIdentity.from_vector(
            "known-hook-effect/1", POSITIVE
        ),
    }
    values.update(overrides)
    return CausalDesignManifest(**values)


class FakeCausalModel:
    """Deterministic fake that records the complete paired execution grid."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def measure_causal_outcomes(self, **kwargs):
        self.calls.append(
            {
                **kwargs,
                "direction": (
                    None
                    if kwargs["direction"] is None
                    else np.asarray(kwargs["direction"]).copy()
                ),
            }
        )
        direction = kwargs["direction"]
        if kwargs["hook_mode"] in {"none", "zero", "sham"} or direction is None:
            intervention = 0.0
        else:
            weights = np.arange(1, len(direction) + 1, dtype=float)
            intervention = float(np.dot(direction, weights)) * kwargs["coefficient"]
        paired_noise = (
            kwargs["prompt_index"] * 0.1
            + kwargs["repeat"] * 0.01
            + (kwargs["seed"] % 17) * 0.0001
        )
        receipt = InterventionApplicationReceipt(
            design_id=kwargs["design_id"],
            intervention_adapter_version=kwargs[
                "intervention_adapter_version"
            ],
            applied=kwargs["hook_mode"] != "none",
            direction_id=kwargs["direction_id"],
            coefficient=kwargs["coefficient"],
            layer=kwargs["layer"],
            token_stage=kwargs["token_stage"],
            prompt_id=kwargs["prompt_id"],
            repeat=kwargs["repeat"],
            seed=kwargs["seed"],
            condition_id=kwargs["condition_id"],
            hook_mode=kwargs["hook_mode"],
        ).to_dict()
        return {
            "application_receipt": receipt,
            "logit": paired_noise + intervention,
            "deception_behavior": paired_noise + intervention * 0.1,
            "fluency": 1.0 + paired_noise - abs(intervention) * 0.01,
            "utility": 10.0 + paired_noise + intervention * 0.2,
            "counterpart_outcome": 5.0 + paired_noise - intervention * 0.1,
        }


def _direction_data() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(9)
    activations = rng.standard_normal((6, 4))
    labels = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    groups = ("train-a", "train-a", "train-b", "train-b", "train-c", "train-c")
    return activations, labels, groups


def _inputs(
    manifest: CausalDesignManifest,
    model: FakeCausalModel,
    **overrides,
) -> CausalExecutionInputs:
    activations, labels, direction_groups = _direction_data()
    values = {
        "design_id": manifest.design_id,
        "dataset_hash": manifest.dataset_hash,
        "intervention": manifest.intervention.value,
        "intervention_adapter_version": (
            manifest.intervention_adapter_version
        ),
        "direction_source": manifest.direction_source,
        "layer": manifest.layer,
        "token_stage": manifest.token_stage,
        "coefficients": manifest.coefficients,
        "repetitions": manifest.repetitions,
        "random_seed": manifest.random_seed,
        "independent_unit": manifest.independent_unit,
        "group_key": manifest.group_key,
        "prompts": ("alpha", "beta", "gamma"),
        "prompt_ids": manifest.prompt_ids,
        "group_ids": ("family-a", "family-b", "family-c"),
        "primary_direction": PRIMARY,
        "nuisance_directions": NUISANCE,
        "positive_direction": POSITIVE,
        "positive_direction_source": "known-hook-effect/1",
        "direction_activations": activations,
        "direction_labels": labels,
        "direction_group_ids": direction_groups,
        "outcome_provenance": {
            "logit": {"measurement_version": "logit-readout/1"},
            "deception_behavior": {"scorer_version": "behavior-rule/3"},
            "fluency": {"scorer_version": "fluency/1"},
            "utility": {"scorer_version": "utility/1"},
            "counterpart_outcome": {"scorer_version": "counterpart/1"},
        },
        "measurement_fn": model.measure_causal_outcomes,
    }
    values.update(overrides)
    return CausalExecutionInputs(**values)


def test_every_declared_control_executes_on_the_same_paired_grid() -> None:
    manifest = _manifest()
    model = FakeCausalModel()

    report = execute_manifest_controls(manifest, _inputs(manifest, model))

    expected_conditions = {
        "baseline_no_hook",
        "target",
        "zero_hook",
        "sham_hook",
        "norm_matched_random",
        "label_shuffled",
        "nuisance_direction:role-identity/1",
        "nuisance_direction:surface-form/1",
        "positive_hook",
    }
    assert set(report["conditions"]) == expected_conditions
    assert set(report["control_status"]) == {item.value for item in ControlKind}
    assert all(item["available"] for item in report["control_status"].values())
    assert report["status"] == "complete"
    assert report["paired_grid_verified"] is True
    assert report["application_receipts_verified"] is True
    assert report["causal_claim_ready"] is False
    assert report["causal_evidence_strength"] is None

    expected_rows = 3 * manifest.repetitions * len(manifest.coefficients)
    pair_sets = []
    for condition in report["conditions"].values():
        assert condition["expected_rows"] == expected_rows
        assert len(condition["rows"]) == expected_rows
        pair_sets.append({row["pair_id"] for row in condition["rows"]})
        for outcome in OUTCOMES:
            outcome_report = condition["outcomes"][outcome]
            assert outcome_report["available"] is True
            assert "source" in outcome_report["scorer_provenance"]
            for coefficient_report in outcome_report["by_coefficient"].values():
                assert coefficient_report["available"] is True
                assert len(coefficient_report["paired_rows"]) == 6
                assert coefficient_report["estimate"]["n_rows"] == 6
                assert coefficient_report["estimate"]["n_clusters"] == 3
        for row in condition["rows"]:
            receipt = row["application_receipt"]
            assert receipt is not None
            assert receipt["applied"] is (
                condition["source"] != "baseline"
            )
    assert all(pair_set == pair_sets[0] for pair_set in pair_sets[1:])
    json.dumps(report)


def test_random_controls_repeats_and_pair_seeds_are_deterministic() -> None:
    manifest = _manifest()
    first_model = FakeCausalModel()
    second_model = FakeCausalModel()

    first = execute_manifest_controls(manifest, _inputs(manifest, first_model))
    second = execute_manifest_controls(manifest, _inputs(manifest, second_model))

    assert first == second
    random_directions = first["conditions"]["norm_matched_random"][
        "direction_by_repeat"
    ]
    assert len({row["direction_id"] for row in random_directions}) == 2
    assert [row["norm"] for row in random_directions] == pytest.approx([1.0, 1.0])

    seeds_by_pair: dict[str, set[int]] = {}
    for condition in first["conditions"].values():
        for row in condition["rows"]:
            seeds_by_pair.setdefault(row["pair_id"], set()).add(row["seed"])
    assert all(len(seeds) == 1 for seeds in seeds_by_pair.values())


def test_missing_control_inputs_remain_structured_and_other_controls_run() -> None:
    manifest = _manifest()
    model = FakeCausalModel()
    inputs = _inputs(
        manifest,
        model,
        nuisance_directions={},
        positive_direction=None,
        positive_direction_source=None,
        direction_activations=None,
        direction_labels=None,
        direction_group_ids=(),
    )

    report = execute_manifest_controls(manifest, inputs)

    assert report["status"] == "partial"
    for control in (
        "label_shuffled",
        "nuisance_direction",
        "positive_hook",
    ):
        status = report["control_status"][control]
        assert status["available"] is False
        assert status["condition_ids"]
        assert status["reasons"]
    for control in ("zero_hook", "sham_hook", "norm_matched_random"):
        assert report["control_status"][control]["available"] is True


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("design_id", "causal_design_wrong", "design_id"),
        ("dataset_hash", "sha256:other", "dataset_hash"),
        ("intervention", "patch", "intervention"),
        ("intervention_adapter_version", "other/1", "adapter"),
        ("direction_source", "test-fold", "direction_source"),
        ("layer", 4, "layer"),
        ("token_stage", "mean_pool", "token_stage"),
        ("coefficients", (-2.0, 2.0), "coefficients"),
        ("repetitions", 3, "repetitions"),
        ("random_seed", 72, "random_seed"),
        ("independent_unit", "dyad", "independent_unit"),
        ("group_key", "dyad_id", "group_key"),
        ("prompt_ids", ("x", "y", "z"), "prompt_ids"),
    ],
)
def test_locked_manifest_identity_mismatches_are_rejected(
    field_name: str,
    bad_value,
    match: str,
) -> None:
    manifest = _manifest()
    model = FakeCausalModel()
    inputs = replace(_inputs(manifest, model), **{field_name: bad_value})

    with pytest.raises(ValueError, match=match):
        execute_manifest_controls(manifest, inputs)


def test_measurement_failure_retains_paired_rows_and_is_not_estimated() -> None:
    manifest = _manifest(outcomes=("logit",))
    model = FakeCausalModel()

    def incomplete_measurement(**kwargs):
        measured = model.measure_causal_outcomes(**kwargs)
        if kwargs["condition_id"] == "positive_hook" and kwargs["repeat"] == 1:
            return {}
        return measured

    report = execute_manifest_controls(
        manifest,
        _inputs(manifest, model, measurement_fn=incomplete_measurement),
    )
    positive = report["conditions"]["positive_hook"]["outcomes"]["logit"]

    assert positive["available"] is False
    coefficient = positive["by_coefficient"]["1.0"]
    assert coefficient["available"] is False
    assert len(coefficient["paired_rows"]) == 6
    assert any(not row["available"] for row in coefficient["paired_rows"])


def test_control_and_scorer_source_mismatches_are_rejected() -> None:
    manifest = _manifest()
    model = FakeCausalModel()

    with pytest.raises(ValueError, match="positive direction"):
        execute_manifest_controls(
            manifest,
            _inputs(
                manifest,
                model,
                positive_direction_source="unlocked-positive-source",
            ),
        )
    with pytest.raises(ValueError, match="nuisance direction IDs"):
        execute_manifest_controls(
            manifest,
            _inputs(
                manifest,
                model,
                nuisance_directions={
                    **NUISANCE,
                    "undeclared/1": np.ones(4),
                },
            ),
        )
    provenance = {
        key: dict(value)
        for key, value in _inputs(manifest, model).outcome_provenance.items()
    }
    provenance["deception_behavior"]["scorer_version"] = "other-scorer/1"
    with pytest.raises(ValueError, match="scorer version"):
        execute_manifest_controls(
            manifest,
            _inputs(manifest, model, outcome_provenance=provenance),
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("intervention_adapter_version", "tampered-adapter/1"),
        ("applied", False),
        ("direction_id", "tampered-direction"),
        ("coefficient", 19.0),
        ("layer", 99),
        ("token_stage", "tampered-stage"),
        ("prompt_id", "tampered-prompt"),
        ("repeat", 99),
        ("seed", 99),
        ("condition_id", "tampered-condition"),
    ],
)
def test_validly_rehashed_but_wrong_application_receipt_fails_closed(
    field_name: str, replacement
) -> None:
    manifest = _manifest(outcomes=("logit",))
    model = FakeCausalModel()

    def tampered_measurement(**kwargs):
        measured = model.measure_causal_outcomes(**kwargs)
        receipt = InterventionApplicationReceipt.from_dict(
            measured["application_receipt"]
        )
        measured["application_receipt"] = replace(
            receipt, **{field_name: replacement}
        ).to_dict()
        return measured

    report = execute_manifest_controls(
        manifest,
        _inputs(manifest, model, measurement_fn=tampered_measurement),
    )

    assert report["status"] == "partial"
    assert report["application_receipts_verified"] is False
    assert any(
        "does not match" in str(row["measurement_error"])
        for condition in report["conditions"].values()
        for row in condition["rows"]
    )


def test_receipt_content_tampering_is_detected_by_receipt_id() -> None:
    manifest = _manifest(outcomes=("logit",))
    model = FakeCausalModel()

    def tampered_measurement(**kwargs):
        measured = model.measure_causal_outcomes(**kwargs)
        measured["application_receipt"]["coefficient"] += 0.5
        return measured

    report = execute_manifest_controls(
        manifest,
        _inputs(manifest, model, measurement_fn=tampered_measurement),
    )

    assert report["status"] == "partial"
    assert any(
        "receipt ID" in str(row["measurement_error"])
        for condition in report["conditions"].values()
        for row in condition["rows"]
    )


def test_constant_outcomes_without_application_receipts_cannot_complete() -> None:
    manifest = _manifest(outcomes=("logit",))
    model = FakeCausalModel()

    report = execute_manifest_controls(
        manifest,
        _inputs(
            manifest,
            model,
            measurement_fn=lambda **_: {"logit": 0.0},
        ),
    )

    assert report["status"] == "partial"
    assert report["application_receipts_verified"] is False
    assert all(
        row["outcomes"]["logit"] is None
        for condition in report["conditions"].values()
        for row in condition["rows"]
    )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        (
            "primary_direction",
            np.asarray([0.0, 1.0, 0.0, 0.0]),
            "primary direction",
        ),
        (
            "nuisance_directions",
            {
                **NUISANCE,
                "role-identity/1": np.asarray([1.0, 1.0, 0.0, 0.0]),
            },
            "nuisance direction",
        ),
        (
            "positive_direction",
            np.asarray([1.0, 0.0, 0.0, 1.0]),
            "positive direction",
        ),
    ],
)
def test_supplied_vectors_must_match_locked_hash_and_dimension(
    field_name: str, bad_value, match: str
) -> None:
    manifest = _manifest()
    model = FakeCausalModel()

    with pytest.raises(ValueError, match=match):
        execute_manifest_controls(
            manifest,
            _inputs(manifest, model, **{field_name: bad_value}),
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "match"),
    [
        ("primary_direction", [1.0, float("nan"), 0.0, 0.0], "finite"),
        (
            "direction_activations",
            np.asarray([[0.0, 1.0, 2.0, float("inf")]]),
            "finite",
        ),
        ("direction_labels", np.asarray([0.0, 0.5, 1.0]), "binary"),
        ("random_seed", -1, "non-negative"),
        ("prompt_ids", ("prompt-a", "prompt-a"), "unique"),
    ],
)
def test_execution_inputs_reject_hostile_numeric_and_identity_values(
    field_name: str, bad_value, match: str
) -> None:
    manifest = _manifest()
    model = FakeCausalModel()

    with pytest.raises(ValueError, match=match):
        _inputs(manifest, model, **{field_name: bad_value})


def test_nonfinite_measurement_outcome_fails_the_row() -> None:
    manifest = _manifest(outcomes=("logit",))
    model = FakeCausalModel()

    def nonfinite_measurement(**kwargs):
        measured = model.measure_causal_outcomes(**kwargs)
        measured["logit"] = float("inf")
        return measured

    report = execute_manifest_controls(
        manifest,
        _inputs(manifest, model, measurement_fn=nonfinite_measurement),
    )

    assert report["status"] == "partial"
    assert any(
        "must be finite" in str(row["measurement_error"])
        for condition in report["conditions"].values()
        for row in condition["rows"]
    )


def test_comprehensive_path_retains_controls_but_never_claims_readiness() -> None:
    manifest = _manifest()
    model = FakeCausalModel()
    rng = np.random.default_rng(3)
    labels = np.repeat([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 4)
    activations = {3: rng.standard_normal((len(labels), 12))}

    report = run_full_causal_validation(
        model=None,
        activations=activations,
        labels=labels,
        best_layer=3,
        group_ids=np.repeat(np.arange(6), 4),
        design_manifest=manifest,
        control_execution_inputs=_inputs(manifest, model),
        random_state=manifest.random_seed,
        verbose=False,
    )

    controls = report["control_execution"]
    assert controls["status"] == "complete"
    assert controls["design_id"] == manifest.design_id
    assert report["causal_claim_ready"] is False
    assert report["causal_evidence_strength"] is None
    assert controls["causal_claim_ready"] is False
    assert all("evidence strength" not in item.lower() for item in controls["interpretation_limits"])


def test_comprehensive_path_rejects_seed_mismatch_and_lists_missing_controls() -> None:
    manifest = _manifest()
    rng = np.random.default_rng(5)
    labels = np.repeat([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 4)
    activations = {3: rng.standard_normal((len(labels), 12))}

    with pytest.raises(ValueError, match="random_state"):
        run_full_causal_validation(
            model=None,
            activations=activations,
            labels=labels,
            best_layer=3,
            design_manifest=manifest,
            random_state=manifest.random_seed + 1,
            verbose=False,
        )

    unavailable = run_full_causal_validation(
        model=None,
        activations=activations,
        labels=labels,
        best_layer=3,
        group_ids=np.repeat(np.arange(6), 4),
        design_manifest=manifest,
        random_state=manifest.random_seed,
        verbose=False,
    )["control_execution"]
    assert unavailable["status"] == "unavailable"
    assert set(unavailable["control_status"]) == {
        control.value for control in manifest.controls
    }
    assert all(
        status["available"] is False
        for status in unavailable["control_status"].values()
    )


def test_row_level_independence_is_rejected_by_the_manifest() -> None:
    with pytest.raises(ValueError, match="never an individual row"):
        _manifest(independent_unit="row")
