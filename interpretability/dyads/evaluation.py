"""Dyadic detection metrics (Plan 4, Phase 8).

Deployment-shaped reporting: TPR at fixed low false-positive rates, exact
tie-aware AUROC, false alarms per dyad, and detection lead time. Grouped
uncertainty reuses the ToM evaluation module's cluster bootstrap so every
interval in the framework clusters at the same trial/dyad unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from negotiation.components.tom.evaluation import (
    GroupedEstimate,
    cluster_bootstrap_mean,
)

DYAD_EVALUATION_VERSION = "dyad-evaluation/1.0.0"


def _scores_and_labels(
    scores: Sequence[float], labels: Sequence[bool]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=float)
    flags = np.asarray(labels)
    if values.ndim != 1 or values.shape[0] < 2:
        raise ValueError("at least two scored samples are required")
    if flags.shape != values.shape:
        raise ValueError("scores and labels must align")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if flags.dtype != bool:
        raise ValueError("labels must be booleans")
    if flags.all() or (~flags).all():
        raise ValueError("both classes are required")
    return values, flags


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Exact tie-aware AUROC (Mann-Whitney with 0.5 tie credit)."""
    values, flags = _scores_and_labels(scores, labels)
    positives = values[flags]
    negatives = values[~flags]
    wins = 0.0
    for positive in positives:
        wins += float((positive > negatives).sum())
        wins += 0.5 * float((positive == negatives).sum())
    return wins / (positives.size * negatives.size)


@dataclass(frozen=True)
class OperatingPoint:
    """One deployable threshold with its achieved rates."""

    target_fpr: float
    threshold: float
    achieved_fpr: float
    tpr: float
    n_positives: int
    n_negatives: int


def tpr_at_fpr(
    scores: Sequence[float],
    labels: Sequence[bool],
    target_fpr: float,
) -> OperatingPoint:
    """Highest-recall threshold whose FPR does not exceed the target.

    The threshold flags ``score >= threshold``; it is chosen from observed
    scores so the achieved FPR is exact on the calibration data.
    """
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr must lie strictly between 0 and 1")
    values, flags = _scores_and_labels(scores, labels)
    negatives = np.sort(values[~flags])[::-1]
    positives = values[flags]
    candidates = np.unique(values)[::-1]
    best: OperatingPoint | None = None
    for threshold in candidates:
        fpr = float((negatives >= threshold).mean())
        if fpr > target_fpr:
            continue
        point = OperatingPoint(
            target_fpr=target_fpr,
            threshold=float(threshold),
            achieved_fpr=fpr,
            tpr=float((positives >= threshold).mean()),
            n_positives=int(positives.size),
            n_negatives=int(negatives.size),
        )
        if best is None or point.tpr > best.tpr:
            best = point
    if best is None:
        # Even the strictest observed threshold overshoots: flag nothing.
        return OperatingPoint(
            target_fpr=target_fpr,
            threshold=float(np.max(values)) + 1.0,
            achieved_fpr=0.0,
            tpr=0.0,
            n_positives=int(positives.size),
            n_negatives=int(negatives.size),
        )
    return best


def false_alarms_per_dyad(
    flagged: Sequence[bool],
    labels: Sequence[bool],
    dyad_ids: Sequence[str],
) -> float:
    """Mean count of false alarms per dyad (repeated monitoring burden)."""
    flags = np.asarray(flagged)
    truth = np.asarray(labels)
    ids = [str(dyad_id) for dyad_id in dyad_ids]
    if flags.dtype != bool or truth.dtype != bool:
        raise ValueError("flagged and labels must be booleans")
    if not (flags.shape == truth.shape and flags.shape[0] == len(ids)):
        raise ValueError("flagged, labels, and dyad_ids must align")
    if not ids:
        raise ValueError("at least one sample is required")
    false_alarm = flags & ~truth
    per_dyad: dict[str, int] = {}
    for dyad_id, alarm in zip(ids, false_alarm):
        per_dyad[dyad_id] = per_dyad.get(dyad_id, 0) + int(alarm)
    return float(np.mean(list(per_dyad.values())))


def detection_lead_times(
    first_alert_ordinal: Mapping[str, int | None],
    onset_ordinal: Mapping[str, int],
) -> dict[str, int | None]:
    """Per-dyad turns of warning before deception onset.

    Positive values alert before the onset turn, zero alerts on it, and
    negative values alert only afterwards. ``None`` means the monitor never
    alerted — an explicit miss, not an infinite lead.
    """
    if set(first_alert_ordinal) != set(onset_ordinal):
        raise ValueError("alert and onset maps must cover the same dyads")
    leads: dict[str, int | None] = {}
    for dyad_id, onset in onset_ordinal.items():
        if type(onset) is not int or onset < 0:
            raise ValueError("onset ordinals must be non-negative integers")
        alert = first_alert_ordinal[dyad_id]
        if alert is None:
            leads[dyad_id] = None
            continue
        if type(alert) is not int or alert < 0:
            raise ValueError("alert ordinals must be non-negative integers")
        leads[dyad_id] = onset - alert
    return leads


@dataclass(frozen=True)
class TransferCell:
    """One train-scope -> evaluation-scope result."""

    train_scope: str
    eval_scope: str
    metric: float


@dataclass(frozen=True)
class TransferMatrix:
    """Within-distribution diagonal versus held-out off-diagonal results."""

    metric_name: str
    cells: tuple[TransferCell, ...]

    def __post_init__(self) -> None:
        if not self.cells:
            raise ValueError("a transfer matrix requires at least one cell")
        seen = {(cell.train_scope, cell.eval_scope) for cell in self.cells}
        if len(seen) != len(self.cells):
            raise ValueError("duplicate transfer cells")

    def in_distribution_mean(self) -> float:
        values = [
            cell.metric
            for cell in self.cells
            if cell.train_scope == cell.eval_scope
        ]
        if not values:
            raise ValueError("no in-distribution cells present")
        return float(np.mean(values))

    def held_out_mean(self) -> float:
        values = [
            cell.metric
            for cell in self.cells
            if cell.train_scope != cell.eval_scope
        ]
        if not values:
            raise ValueError("no held-out cells present")
        return float(np.mean(values))

    def transfer_gap(self) -> float:
        return self.in_distribution_mean() - self.held_out_mean()


def grouped_metric_estimate(
    per_sample_values: Sequence[float],
    dyad_ids: Sequence[str],
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> GroupedEstimate:
    """Dyad-clustered bootstrap interval for any per-sample metric."""
    return cluster_bootstrap_mean(
        per_sample_values,
        [str(dyad_id) for dyad_id in dyad_ids],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )


__all__ = [
    "DYAD_EVALUATION_VERSION",
    "OperatingPoint",
    "TransferCell",
    "TransferMatrix",
    "detection_lead_times",
    "false_alarms_per_dyad",
    "grouped_metric_estimate",
    "roc_auc",
    "tpr_at_fpr",
]
