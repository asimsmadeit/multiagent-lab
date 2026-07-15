"""Exact-value tests for dyadic feature families and detection metrics."""

from __future__ import annotations

import numpy as np
import pytest

from interpretability.dyads.evaluation import (
    TransferCell,
    TransferMatrix,
    detection_lead_times,
    false_alarms_per_dyad,
    grouped_metric_estimate,
    roc_auc,
    tpr_at_fpr,
)
from interpretability.dyads.features import (
    change_point,
    fit_linear_alignment,
    fuse_scores,
    relational_features,
    score_slope,
    shuffled_pairing_control,
    time_to_first_alert,
)

# ---------------------------------------------------------------------------
# Feature families
# ---------------------------------------------------------------------------


def test_score_fusion_methods_and_bounds() -> None:
    sender = [0.2, 0.8]
    receiver = [0.6, 0.4]
    assert fuse_scores(sender, receiver, "mean").tolist() == pytest.approx(
        [0.4, 0.6]
    )
    assert fuse_scores(sender, receiver, "max").tolist() == [0.6, 0.8]
    product = fuse_scores(sender, receiver, "product")
    assert product.tolist() == pytest.approx([0.12, 0.32])
    with pytest.raises(ValueError, match="must align"):
        fuse_scores([0.1], [0.1, 0.2])
    with pytest.raises(ValueError, match="requires scores in"):
        fuse_scores([1.5, 0.2], [0.1, 0.2], "product")
    with pytest.raises(ValueError, match="unknown fusion"):
        fuse_scores(sender, receiver, "median")  # type: ignore[arg-type]


def test_linear_alignment_recovers_an_exact_map() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(24, 4))
    true_map = rng.normal(size=(4, 4))
    target = source @ true_map
    alignment = fit_linear_alignment(source, target)
    assert alignment.n_fit_rows == 24
    aligned = alignment.apply(source)
    assert np.allclose(aligned, target, atol=1e-8)
    features = relational_features(aligned, target)
    assert features["cosine"] == pytest.approx([1.0] * 24)
    assert features["l2_difference"] == pytest.approx([0.0] * 24, abs=1e-7)
    with pytest.raises(ValueError, match="does not match the fit"):
        alignment.apply(rng.normal(size=(3, 5)))
    with pytest.raises(ValueError, match="at least as many rows"):
        fit_linear_alignment(rng.normal(size=(3, 4)), rng.normal(size=(3, 4)))


def test_relational_features_reject_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="matched shapes"):
        relational_features([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="no defined direction"):
        relational_features([[0.0, 0.0]], [[1.0, 0.0]])


def test_shuffled_pairing_moves_rows_across_dyads() -> None:
    receiver = np.arange(8.0).reshape(4, 2)
    dyads = ["d1", "d1", "d2", "d2"]
    shuffled = shuffled_pairing_control(receiver, dyads, seed=5)
    assert sorted(map(tuple, shuffled.tolist())) == sorted(
        map(tuple, receiver.tolist())
    )
    moved_across = any(
        dyads[i] != dyads[receiver.tolist().index(row)]
        for i, row in enumerate(shuffled.tolist())
    )
    assert moved_across
    again = shuffled_pairing_control(receiver, dyads, seed=5)
    assert np.array_equal(shuffled, again)
    with pytest.raises(ValueError, match="at least two dyads"):
        shuffled_pairing_control(receiver, ["d1"] * 4)


def test_temporal_features_exact_values() -> None:
    assert score_slope([0.0, 0.1, 0.2, 0.3]) == pytest.approx(0.1)
    assert change_point([0.1, 0.15, 0.9, 0.85]) == 2
    assert time_to_first_alert([0.1, 0.4, 0.8], threshold=0.5) == 2
    assert time_to_first_alert([0.1, 0.2], threshold=0.5) is None
    with pytest.raises(ValueError, match="at least two turns"):
        score_slope([0.5])


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------


def test_roc_auc_exact_with_ties() -> None:
    labels = [True, True, False, False]
    assert roc_auc([0.9, 0.8, 0.2, 0.1], labels) == 1.0
    assert roc_auc([0.1, 0.2, 0.8, 0.9], labels) == 0.0
    assert roc_auc([0.5, 0.5, 0.5, 0.5], labels) == 0.5
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([0.1, 0.2], [True, True])


def test_tpr_at_fpr_picks_the_highest_recall_legal_threshold() -> None:
    scores = [0.95, 0.9, 0.85, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    labels = [True, True, True, True, False, True, False, False, False,
              False]
    point = tpr_at_fpr(scores, labels, target_fpr=0.2)
    assert point.threshold == pytest.approx(0.5)
    assert point.achieved_fpr == pytest.approx(0.2)
    assert point.tpr == pytest.approx(1.0)
    strict = tpr_at_fpr(scores, labels, target_fpr=0.05)
    assert strict.achieved_fpr == 0.0
    assert strict.tpr == pytest.approx(0.8)
    with pytest.raises(ValueError, match="strictly between"):
        tpr_at_fpr(scores, labels, target_fpr=0.0)


def test_tpr_at_fpr_flags_nothing_when_no_threshold_is_legal() -> None:
    # The single negative carries the top score, so every observed
    # threshold that flags anything flags it too.
    point = tpr_at_fpr([0.9, 0.8, 0.7], [False, True, True], target_fpr=0.4)
    assert point.tpr == 0.0
    assert point.achieved_fpr == 0.0


def test_false_alarms_per_dyad_counts_only_false_positives() -> None:
    flagged = [True, True, False, True]
    labels = [True, False, False, False]
    dyads = ["d1", "d1", "d2", "d2"]
    assert false_alarms_per_dyad(flagged, labels, dyads) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="must align"):
        false_alarms_per_dyad([True], labels, dyads)


def test_detection_lead_times_signs_and_misses() -> None:
    leads = detection_lead_times(
        {"d1": 1, "d2": None, "d3": 4},
        {"d1": 3, "d2": 2, "d3": 2},
    )
    assert leads == {"d1": 2, "d2": None, "d3": -2}
    with pytest.raises(ValueError, match="same dyads"):
        detection_lead_times({"d1": 1}, {"d2": 1})


def test_transfer_matrix_gap() -> None:
    matrix = TransferMatrix(
        metric_name="auroc",
        cells=(
            TransferCell("ultimatum", "ultimatum", 0.9),
            TransferCell("alliance", "alliance", 0.8),
            TransferCell("ultimatum", "alliance", 0.6),
            TransferCell("alliance", "ultimatum", 0.7),
        ),
    )
    assert matrix.in_distribution_mean() == pytest.approx(0.85)
    assert matrix.held_out_mean() == pytest.approx(0.65)
    assert matrix.transfer_gap() == pytest.approx(0.2)
    with pytest.raises(ValueError, match="duplicate transfer"):
        TransferMatrix(
            metric_name="auroc",
            cells=(
                TransferCell("a", "a", 0.5),
                TransferCell("a", "a", 0.6),
            ),
        )


def test_grouped_metric_estimate_clusters_by_dyad() -> None:
    estimate = grouped_metric_estimate(
        [0.0, 0.2, 0.8, 1.0, 0.4, 0.6],
        ["d1", "d1", "d2", "d2", "d3", "d3"],
        n_bootstrap=500,
        seed=11,
    )
    assert estimate.mean == pytest.approx(0.5)
    assert estimate.n_groups == 3
    assert estimate.ci_low <= estimate.mean <= estimate.ci_high
