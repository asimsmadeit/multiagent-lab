"""Tests for paired clustered causal uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from interpretability.causal.statistics import paired_clustered_estimate


def test_clustered_estimate_is_seeded_paired_and_json_safe() -> None:
    clusters = np.repeat(["family-a", "family-b", "family-c", "family-d"], 3)
    baseline = np.zeros(12)
    intervention = np.repeat([1.0, 2.0, 3.0, 4.0], 3)

    first = paired_clustered_estimate(
        baseline,
        intervention,
        clusters,
        cluster_unit="trial_family",
        n_bootstrap=200,
        n_permutations=500,
        random_seed=7,
    )
    second = paired_clustered_estimate(
        baseline,
        intervention,
        clusters,
        cluster_unit="trial_family",
        n_bootstrap=200,
        n_permutations=500,
        random_seed=7,
    )

    assert first == second
    assert first.effect == pytest.approx(2.5)
    assert first.n_rows == 12
    assert first.n_clusters == 4
    assert first.to_dict()["cluster_unit"] == "trial_family"
    assert first.to_dict()["bootstrap_ci"]["clustered"] is True
    assert first.to_dict()["sign_permutation"]["p_value"] == first.p_value


def test_cluster_weighting_uses_independent_units_not_row_counts() -> None:
    # The large first cluster still contributes one of three cluster means.
    clusters = np.array(["large"] * 8 + ["small-a", "small-b"])
    baseline = np.zeros(10)
    intervention = np.array([9.0] * 8 + [0.0, 0.0])

    result = paired_clustered_estimate(
        baseline,
        intervention,
        clusters,
        cluster_unit="dyad",
        n_bootstrap=50,
        n_permutations=50,
    )

    assert result.effect == pytest.approx(3.0)
    assert result.effect != pytest.approx(intervention.mean())


def test_clustered_estimate_fails_closed_for_unaligned_or_too_few_units() -> None:
    with pytest.raises(ValueError, match="align"):
        paired_clustered_estimate(
            [0.0, 1.0],
            [1.0],
            ["a", "b"],
            cluster_unit="trial",
        )


@pytest.mark.parametrize(
    ("cluster_ids", "match"),
    [
        (["family-a", None, "family-c"], "missing"),
        (["family-a", "", "family-c"], "missing"),
        (["family-a", float("nan"), "family-c"], "finite"),
        ([["family-a"], ["family-b"], ["family-c"]], "one-dimensional"),
    ],
)
def test_clustered_estimate_rejects_missing_or_nonfinite_cluster_ids(
    cluster_ids, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        paired_clustered_estimate(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            cluster_ids,
            cluster_unit="trial_family",
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"cluster_unit": "row"}, "row-level"),
        ({"n_bootstrap": 0}, "positive integer"),
        ({"n_bootstrap": True}, "positive integer"),
        ({"n_permutations": 0}, "positive integer"),
        ({"random_seed": -1}, "non-negative"),
        ({"alpha": float("nan")}, "alpha"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
    ],
)
def test_clustered_estimate_rejects_invalid_inference_configuration(
    overrides, match: str
) -> None:
    kwargs = {
        "cluster_unit": "trial_family",
        "n_bootstrap": 10,
        "n_permutations": 10,
        "random_seed": 7,
        "alpha": 0.05,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        paired_clustered_estimate(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            ["a", "b", "c"],
            **kwargs,
        )


@pytest.mark.parametrize(
    ("baseline", "intervention"),
    [
        ([0.0, float("nan"), 0.0], [1.0, 1.0, 1.0]),
        ([0.0, 0.0, 0.0], [1.0, float("inf"), 1.0]),
    ],
)
def test_clustered_estimate_rejects_nonfinite_paired_values(
    baseline, intervention
) -> None:
    with pytest.raises(ValueError, match="finite"):
        paired_clustered_estimate(
            baseline,
            intervention,
            ["a", "b", "c"],
            cluster_unit="trial_family",
        )
    with pytest.raises(ValueError, match="three independent"):
        paired_clustered_estimate(
            [0.0, 0.0],
            [1.0, 1.0],
            ["a", "b"],
            cluster_unit="trial",
        )
