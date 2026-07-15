"""Literal, functional, and faithfulness ToM metrics (Plan 3, Phase 8).

Pure, deterministic scoring over typed belief records. Three rules hold
throughout:

1. An outcome category the distribution never enumerated scores against the
   distribution's declared unknown bucket — closed-world confidence is not
   rewarded, and a missing unknown bucket is an error, never a silent zero.
2. Turns are repeated measurements: uncertainty is estimated by cluster
   bootstrap over trial/dyad groups, not by treating rows as independent.
3. Every function validates its inputs and fails closed on mismatched
   lengths, empty data, or non-finite values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from negotiation.components.tom.schema import BeliefDistribution, BeliefUpdate

EVALUATION_VERSION = "tom-evaluation/1.0.0"

_MIN_PROBABILITY = 1e-12


def _checked_pairs(
    predictions: Sequence[BeliefDistribution],
    outcomes: Sequence[str],
) -> list[tuple[BeliefDistribution, str]]:
    if len(predictions) != len(outcomes):
        raise ValueError("predictions and outcomes must have equal length")
    if not predictions:
        raise ValueError("at least one prediction is required")
    pairs: list[tuple[BeliefDistribution, str]] = []
    for prediction, outcome in zip(predictions, outcomes):
        if not isinstance(prediction, BeliefDistribution):
            raise ValueError("predictions must be BeliefDistribution records")
        if type(outcome) is not str or not outcome:
            raise ValueError("outcomes must be non-empty strings")
        pairs.append((prediction, outcome))
    return pairs


def _resolved_category(prediction: BeliefDistribution, outcome: str) -> str:
    """Map an outcome onto the prediction's category space, fail closed."""
    if outcome in prediction.categories:
        return outcome
    if prediction.unknown_category in prediction.categories:
        return prediction.unknown_category
    raise ValueError(
        f"outcome {outcome!r} is outside the hypothesis space of target "
        f"{prediction.target!r} and no unknown bucket is available"
    )


def next_action_log_loss(
    predictions: Sequence[BeliefDistribution],
    outcomes: Sequence[str],
) -> float:
    """Mean negative log probability (nats) of the realized categories."""
    total = 0.0
    pairs = _checked_pairs(predictions, outcomes)
    for prediction, outcome in pairs:
        probability = prediction.probability(
            _resolved_category(prediction, outcome)
        )
        total += -math.log(max(probability, _MIN_PROBABILITY))
    return total / len(pairs)


def brier_score(
    predictions: Sequence[BeliefDistribution],
    outcomes: Sequence[str],
) -> float:
    """Mean multiclass Brier score over the declared hypothesis space."""
    total = 0.0
    pairs = _checked_pairs(predictions, outcomes)
    for prediction, outcome in pairs:
        realized = _resolved_category(prediction, outcome)
        total += sum(
            (probability - (1.0 if category == realized else 0.0)) ** 2
            for category, probability in zip(
                prediction.categories, prediction.probabilities
            )
        )
    return total / len(pairs)


def _top_categories(prediction: BeliefDistribution) -> list[str]:
    return [
        category
        for _, category in sorted(
            zip(prediction.probabilities, prediction.categories),
            key=lambda pair: (-pair[0], pair[1]),
        )
    ]


def top_k_accuracy(
    predictions: Sequence[BeliefDistribution],
    outcomes: Sequence[str],
    k: int = 1,
) -> float:
    """Fraction of outcomes inside each prediction's top-k categories."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    pairs = _checked_pairs(predictions, outcomes)
    hits = sum(
        1
        for prediction, outcome in pairs
        if _resolved_category(prediction, outcome)
        in _top_categories(prediction)[:k]
    )
    return hits / len(pairs)


def expected_calibration_error(
    predictions: Sequence[BeliefDistribution],
    outcomes: Sequence[str],
    n_bins: int = 10,
) -> float:
    """Standard top-label ECE with equal-width confidence bins."""
    if type(n_bins) is not int or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least 2")
    pairs = _checked_pairs(predictions, outcomes)
    confidences = np.empty(len(pairs))
    correct = np.empty(len(pairs))
    for index, (prediction, outcome) in enumerate(pairs):
        top = _top_categories(prediction)[0]
        confidences[index] = prediction.probability(top)
        correct[index] = float(
            _resolved_category(prediction, outcome) == top
        )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1.0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        if not mask.any():
            continue
        ece += (mask.mean()) * abs(
            correct[mask].mean() - confidences[mask].mean()
        )
    return float(ece)


def update_moved_toward(update: BeliefUpdate, category: str) -> bool:
    """Whether one update increased the probability of ``category``."""
    if not isinstance(update, BeliefUpdate):
        raise ValueError("update must be a BeliefUpdate")
    if category not in update.posterior.categories:
        raise ValueError(
            f"category {category!r} is not in the update's hypothesis space"
        )
    return update.posterior.probability(category) > update.prior.probability(
        category
    )


def update_direction_accuracy(
    updates: Sequence[BeliefUpdate],
    diagnostic_categories: Sequence[str],
) -> float:
    """Fraction of updates that moved toward their diagnostic category."""
    if len(updates) != len(diagnostic_categories):
        raise ValueError("updates and categories must have equal length")
    if not updates:
        raise ValueError("at least one update is required")
    moved = sum(
        1
        for update, category in zip(updates, diagnostic_categories)
        if update_moved_toward(update, category)
    )
    return moved / len(updates)


def regret_vs_oracle(
    realized_utilities: Sequence[float],
    oracle_utilities: Sequence[float],
) -> float:
    """Mean shortfall of realized utility against a per-trial oracle."""
    if len(realized_utilities) != len(oracle_utilities):
        raise ValueError("realized and oracle utilities must align")
    if not realized_utilities:
        raise ValueError("at least one utility pair is required")
    deltas = []
    for realized, oracle in zip(realized_utilities, oracle_utilities):
        for value in (realized, oracle):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("utilities must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError("utilities must be finite")
        deltas.append(float(oracle) - float(realized))
    return sum(deltas) / len(deltas)


def adaptation_latency(
    correct_flags: Sequence[bool],
    shift_index: int,
) -> int | None:
    """Turns from a partner-policy shift to the first correct prediction.

    Returns ``None`` when the predictor never becomes correct after the
    shift — an explicit non-adaptation outcome, not an infinite latency.
    """
    if type(shift_index) is not int or shift_index < 0:
        raise ValueError("shift_index must be a non-negative integer")
    if shift_index >= len(correct_flags):
        raise ValueError("shift_index must fall inside the flag sequence")
    for offset, flag in enumerate(correct_flags[shift_index:]):
        if type(flag) is not bool:
            raise ValueError("correct_flags must contain booleans")
        if flag:
            return offset
    return None


@dataclass(frozen=True)
class GroupedEstimate:
    """A cluster-bootstrap point estimate with a percentile interval."""

    mean: float
    ci_low: float
    ci_high: float
    n_samples: int
    n_groups: int
    n_bootstrap: int
    confidence: float


def cluster_bootstrap_mean(
    values: Sequence[float],
    groups: Sequence[str],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> GroupedEstimate:
    """Bootstrap the mean by resampling whole trial/dyad groups."""
    if len(values) != len(groups):
        raise ValueError("values and groups must have equal length")
    if not values:
        raise ValueError("at least one value is required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if type(n_bootstrap) is not int or n_bootstrap < 100:
        raise ValueError("n_bootstrap must be an integer of at least 100")
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    group_ids = np.asarray([str(group) for group in groups])
    unique_groups = np.unique(group_ids)
    if len(unique_groups) < 2:
        raise ValueError(
            "cluster bootstrap requires at least two distinct groups"
        )
    member_values = {
        group: array[group_ids == group] for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap)
    for draw in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups))
        means[draw] = float(
            np.concatenate([member_values[group] for group in sampled]).mean()
        )
    alpha = (1.0 - confidence) / 2.0
    return GroupedEstimate(
        mean=float(array.mean()),
        ci_low=float(np.quantile(means, alpha)),
        ci_high=float(np.quantile(means, 1.0 - alpha)),
        n_samples=len(array),
        n_groups=len(unique_groups),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )


def paired_condition_effect(
    condition_scores: Sequence[float],
    baseline_scores: Sequence[float],
    groups: Sequence[str],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> GroupedEstimate:
    """Cluster-bootstrapped mean of paired per-trial condition differences.

    One helper serves the three faithfulness contrasts: sufficiency
    (belief-only minus full), completeness (full minus belief-only), and
    necessity (corrupted minus intact) — callers choose the pairing and the
    sign convention.
    """
    if len(condition_scores) != len(baseline_scores):
        raise ValueError("condition and baseline scores must align")
    differences = [
        float(condition) - float(baseline)
        for condition, baseline in zip(condition_scores, baseline_scores)
    ]
    return cluster_bootstrap_mean(
        differences,
        groups,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )


__all__ = [
    "EVALUATION_VERSION",
    "GroupedEstimate",
    "adaptation_latency",
    "brier_score",
    "cluster_bootstrap_mean",
    "expected_calibration_error",
    "next_action_log_loss",
    "paired_condition_effect",
    "regret_vs_oracle",
    "top_k_accuracy",
    "update_direction_accuracy",
    "update_moved_toward",
]
