"""Exact-value tests for ToM literal/functional/faithfulness metrics."""

from __future__ import annotations

import math

import pytest

from negotiation.components.tom.evaluation import (
    GroupedEstimate,
    adaptation_latency,
    brier_score,
    cluster_bootstrap_mean,
    expected_calibration_error,
    next_action_log_loss,
    paired_condition_effect,
    regret_vs_oracle,
    top_k_accuracy,
    update_direction_accuracy,
    update_moved_toward,
)
from negotiation.components.tom.schema import (
    BeliefDistribution,
    BeliefUpdate,
    EpistemicStatus,
    Evidence,
    EvidenceChannel,
    EvidenceVisibility,
    GroundTruthKind,
    UpdateMethod,
)


def dist(
    probabilities: tuple[float, ...],
    categories: tuple[str, ...] = ("accept", "counter", "unknown"),
    target: str = "next_action",
) -> BeliefDistribution:
    return BeliefDistribution(
        target=target,
        categories=categories,
        probabilities=probabilities,
        unknown_category="unknown",
        epistemic_status=EpistemicStatus.UPDATED,
        ground_truth_kind=GroundTruthKind.OBJECTIVE,
    )


def _update(prior: tuple[float, ...], posterior: tuple[float, ...]) -> BeliefUpdate:
    evidence = Evidence(
        observer_id="Seller",
        source_actor_id="Buyer",
        source_event_id="event-1",
        source_call_id="call-1",
        turn=1,
        features=(("requested_evidence", True),),
        channel=EvidenceChannel.OBSERVABLE,
        visibility=EvidenceVisibility.PUBLIC,
        visible_to=(),
        reliability=0.9,
        extractor_version="rules-1",
        source_text_hash="sha256:" + "a" * 64,
        source_span=(3, 12),
        summary="counterpart requested documentation",
    )
    return BeliefUpdate(
        prior=dist(prior),
        evidence=(evidence,),
        likelihoods=(0.9, 0.05, 0.05),
        posterior=dist(posterior),
        method=UpdateMethod.BAYESIAN,
        updater_version="bayes-1",
        observation_model_version="controlled-table-1",
    )


# ---------------------------------------------------------------------------
# Literal metrics
# ---------------------------------------------------------------------------


def test_log_loss_matches_hand_computation() -> None:
    predictions = [dist((0.5, 0.25, 0.25)), dist((0.1, 0.8, 0.1))]
    outcomes = ["accept", "counter"]
    expected = (-math.log(0.5) - math.log(0.8)) / 2
    assert next_action_log_loss(predictions, outcomes) == pytest.approx(expected)


def test_unseen_outcome_scores_against_the_unknown_bucket() -> None:
    prediction = dist((0.6, 0.3, 0.1))
    loss = next_action_log_loss([prediction], ["walk_away"])
    assert loss == pytest.approx(-math.log(0.1))
    assert brier_score([prediction], ["walk_away"]) == pytest.approx(
        0.6**2 + 0.3**2 + 0.9**2
    )


def test_missing_unknown_bucket_is_unconstructable() -> None:
    # The schema already refuses distributions whose unknown bucket is not a
    # member category, so the evaluation-level fail-closed branch is pure
    # defense in depth: closed-world hypothesis spaces cannot exist at all.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BeliefDistribution(
            target="next_action",
            categories=("accept", "counter"),
            probabilities=(0.5, 0.5),
            unknown_category="unknown",
            epistemic_status=EpistemicStatus.UPDATED,
            ground_truth_kind=GroundTruthKind.OBJECTIVE,
        )


def test_brier_score_exact_value() -> None:
    prediction = dist((0.7, 0.2, 0.1))
    expected = (0.7 - 1.0) ** 2 + 0.2**2 + 0.1**2
    assert brier_score([prediction], ["accept"]) == pytest.approx(expected)


def test_top_k_accuracy_and_validation() -> None:
    predictions = [dist((0.5, 0.4, 0.1)), dist((0.1, 0.2, 0.7))]
    outcomes = ["counter", "accept"]
    assert top_k_accuracy(predictions, outcomes, k=1) == 0.0
    assert top_k_accuracy(predictions, outcomes, k=2) == pytest.approx(0.5)
    assert top_k_accuracy(predictions, outcomes, k=3) == 1.0
    with pytest.raises(ValueError, match="positive integer"):
        top_k_accuracy(predictions, outcomes, k=0)


def test_expected_calibration_error_two_bins() -> None:
    # Bin [0, 0.5): one sample, confidence 0.4, wrong -> |0 - 0.4| = 0.4
    # Bin [0.5, 1]: one sample, confidence 0.9, right -> |1 - 0.9| = 0.1
    predictions = [
        dist((0.4, 0.35, 0.25)),
        dist((0.9, 0.05, 0.05)),
    ]
    outcomes = ["counter", "accept"]
    ece = expected_calibration_error(predictions, outcomes, n_bins=2)
    assert ece == pytest.approx(0.5 * 0.4 + 0.5 * 0.1)
    with pytest.raises(ValueError, match="at least 2"):
        expected_calibration_error(predictions, outcomes, n_bins=1)


def test_length_and_type_validation() -> None:
    with pytest.raises(ValueError, match="equal length"):
        next_action_log_loss([dist((0.5, 0.3, 0.2))], [])
    with pytest.raises(ValueError, match="at least one"):
        brier_score([], [])
    with pytest.raises(ValueError, match="non-empty strings"):
        top_k_accuracy([dist((0.5, 0.3, 0.2))], [""])


# ---------------------------------------------------------------------------
# Update direction and functional metrics
# ---------------------------------------------------------------------------


def test_update_direction() -> None:
    toward = _update((0.4, 0.4, 0.2), (0.7, 0.2, 0.1))
    away = _update((0.4, 0.4, 0.2), (0.2, 0.6, 0.2))
    assert update_moved_toward(toward, "accept")
    assert not update_moved_toward(away, "accept")
    assert update_direction_accuracy(
        [toward, away], ["accept", "accept"]
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="hypothesis space"):
        update_moved_toward(toward, "walk_away")


def test_regret_and_adaptation_latency() -> None:
    assert regret_vs_oracle([1.0, 2.0], [3.0, 2.5]) == pytest.approx(1.25)
    with pytest.raises(ValueError, match="finite"):
        regret_vs_oracle([float("nan")], [1.0])
    assert adaptation_latency([False, False, True, True], shift_index=1) == 1
    assert adaptation_latency([True, False, False], shift_index=1) is None
    assert adaptation_latency([False, True], shift_index=1) == 0
    with pytest.raises(ValueError, match="inside the flag sequence"):
        adaptation_latency([True], shift_index=1)


# ---------------------------------------------------------------------------
# Cluster bootstrap and paired effects
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    values = [0.0, 0.2, 0.8, 1.0, 0.4, 0.6]
    groups = ["t1", "t1", "t2", "t2", "t3", "t3"]
    first = cluster_bootstrap_mean(values, groups, n_bootstrap=500, seed=9)
    second = cluster_bootstrap_mean(values, groups, n_bootstrap=500, seed=9)
    assert first == second
    assert isinstance(first, GroupedEstimate)
    assert first.mean == pytest.approx(0.5)
    assert first.ci_low <= first.mean <= first.ci_high
    assert first.n_groups == 3 and first.n_samples == 6


def test_cluster_bootstrap_requires_group_structure() -> None:
    with pytest.raises(ValueError, match="two distinct groups"):
        cluster_bootstrap_mean([1.0, 2.0], ["t1", "t1"])
    with pytest.raises(ValueError, match="equal length"):
        cluster_bootstrap_mean([1.0], ["t1", "t2"])
    with pytest.raises(ValueError, match="at least 100"):
        cluster_bootstrap_mean([1.0, 2.0], ["t1", "t2"], n_bootstrap=10)


def test_paired_condition_effect_is_the_mean_paired_difference() -> None:
    effect = paired_condition_effect(
        [0.9, 0.8, 0.7, 0.6],
        [0.5, 0.6, 0.6, 0.5],
        ["t1", "t1", "t2", "t2"],
        n_bootstrap=500,
        seed=3,
    )
    assert effect.mean == pytest.approx((0.4 + 0.2 + 0.1 + 0.1) / 4)
    assert effect.ci_low <= effect.mean <= effect.ci_high
    with pytest.raises(ValueError, match="must align"):
        paired_condition_effect([1.0], [1.0, 2.0], ["t1", "t2"])
