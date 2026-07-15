"""Train-only calibrated monitor fusion (Plan 6, Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from interpretability.monitoring.base import Monitor, MonitorBase
from interpretability.monitoring.schema import (
    MonitorInput,
    MonitorOutput,
    MonitorView,
)


@dataclass(frozen=True)
class CalibratedFusionMonitor(MonitorBase):
    """Weighted fusion of frozen member monitors.

    Weights are fitted on the monitor-training partition only (least
    squares against binary labels, clipped to non-negative and
    renormalized) and are frozen thereafter. Members must already be
    frozen; fusing an unfitted member is a configuration error upstream.
    """

    members: tuple[Monitor, ...] = ()
    weights: tuple[float, ...] = ()
    required_views: tuple[MonitorView, ...] = ()

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError("fusion requires at least two member monitors")
        if len(self.weights) != len(self.members):
            raise ValueError("one weight per member is required")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("weights must be non-negative")
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("weights must sum to one")
        views = sorted(
            {
                view
                for member in self.members
                for view in member.required_views
            },
            key=lambda view: view.value,
        )
        object.__setattr__(self, "required_views", tuple(views))

    @staticmethod
    def fit_weights(
        member_train_scores: Sequence[Sequence[float]],
        labels: Sequence[bool],
    ) -> tuple[float, ...]:
        """Least-squares member weights from TRAINING scores only."""
        matrix = np.asarray(member_train_scores, dtype=float).T
        target = np.asarray([1.0 if label else 0.0 for label in labels])
        if matrix.ndim != 2 or matrix.shape[0] != target.shape[0]:
            raise ValueError("scores and labels must align")
        if matrix.shape[0] < matrix.shape[1] + 1:
            raise ValueError("fitting requires more rows than members")
        if len(set(labels)) < 2:
            raise ValueError("both classes are required to fit weights")
        raw, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        clipped = np.clip(raw, 0.0, None)
        if clipped.sum() <= 0:
            clipped = np.ones_like(clipped)
        normalized = clipped / clipped.sum()
        return tuple(float(weight) for weight in normalized)

    def score(self, monitor_input: MonitorInput) -> MonitorOutput:
        member_outputs = [
            member.score(monitor_input) for member in self.members
        ]
        fused = sum(
            weight * output.suspicion_score
            for weight, output in zip(self.weights, member_outputs)
        )
        reasons = tuple(
            f"{output.monitor_id}:{output.suspicion_score:.3f}"
            for output in member_outputs
        )
        return self.emit(
            monitor_input, float(fused), reason_codes=reasons
        )


__all__ = ["CalibratedFusionMonitor"]
