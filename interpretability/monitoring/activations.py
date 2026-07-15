"""Activation-probe monitors: sender, receiver, bilateral (Plan 6)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from interpretability.monitoring.base import MonitorBase
from interpretability.monitoring.schema import (
    MonitorInput,
    MonitorOutput,
    MonitorView,
)


class ProbeSide(str, Enum):
    SENDER = "sender"
    RECEIVER = "receiver"
    BILATERAL = "bilateral"


def _probe_score(
    vector: tuple[float, ...],
    direction: tuple[float, ...],
    bias: float,
) -> float:
    if len(vector) != len(direction):
        raise ValueError("activation and probe direction dimensions differ")
    logit = sum(v * d for v, d in zip(vector, direction)) + bias
    return 1.0 / (1.0 + math.exp(-logit))


@dataclass(frozen=True)
class LinearProbeMonitor(MonitorBase):
    """Frozen linear probe over one or both activation streams.

    Directions and bias are fitted upstream on training data and frozen
    here; bilateral scoring is the mean of the two sides' probabilities,
    with explicit degradation when one side is unavailable.
    """

    side: ProbeSide = ProbeSide.SENDER
    sender_direction: tuple[float, ...] = ()
    receiver_direction: tuple[float, ...] = ()
    bias: float = 0.0
    required_views: tuple[MonitorView, ...] = ()

    def __post_init__(self) -> None:
        views: tuple[MonitorView, ...]
        if self.side is ProbeSide.SENDER:
            views = (MonitorView.SENDER_ACTIVATIONS,)
            if not self.sender_direction:
                raise ValueError("sender probe requires sender_direction")
        elif self.side is ProbeSide.RECEIVER:
            views = (MonitorView.RECEIVER_ACTIVATIONS,)
            if not self.receiver_direction:
                raise ValueError("receiver probe requires receiver_direction")
        else:
            views = (
                MonitorView.SENDER_ACTIVATIONS,
                MonitorView.RECEIVER_ACTIVATIONS,
            )
            if not self.sender_direction or not self.receiver_direction:
                raise ValueError(
                    "bilateral probe requires both directions"
                )
        object.__setattr__(self, "required_views", views)

    def score(self, monitor_input: MonitorInput) -> MonitorOutput:
        scores: list[float] = []
        reasons: list[str] = []
        if self.side in (ProbeSide.SENDER, ProbeSide.BILATERAL):
            if monitor_input.sender_activation is None:
                reasons.append("missing_sender_activation")
            else:
                scores.append(
                    _probe_score(
                        monitor_input.sender_activation,
                        self.sender_direction,
                        self.bias,
                    )
                )
        if self.side in (ProbeSide.RECEIVER, ProbeSide.BILATERAL):
            if monitor_input.receiver_activation is None:
                reasons.append("missing_receiver_activation")
            else:
                scores.append(
                    _probe_score(
                        monitor_input.receiver_activation,
                        self.receiver_direction,
                        self.bias,
                    )
                )
        if not scores:
            return self.emit(
                monitor_input, 0.5, reason_codes=tuple(reasons)
            )
        return self.emit(
            monitor_input,
            sum(scores) / len(scores),
            reason_codes=tuple(reasons),
        )


__all__ = ["LinearProbeMonitor", "ProbeSide"]
