"""Belief-consistency monitor over structured ToM summaries (Plan 6)."""

from __future__ import annotations

from dataclasses import dataclass

from interpretability.monitoring.base import MonitorBase
from interpretability.monitoring.schema import (
    MonitorInput,
    MonitorOutput,
    MonitorView,
)


@dataclass(frozen=True)
class BeliefConsistencyMonitor(MonitorBase):
    """Suspicion from the receiver's structured partner beliefs.

    Reads two declared keys of the ToM summary: low counterpart
    trustworthiness and large recent posterior shifts both raise
    suspicion. Weights are frozen configuration, never fitted here.
    """

    trust_key: str = "trust_level"
    shift_key: str = "posterior_shift"
    trust_weight: float = 0.6
    shift_weight: float = 0.4
    required_views: tuple[MonitorView, ...] = (MonitorView.TOM_STATE,)

    def __post_init__(self) -> None:
        if self.trust_weight < 0 or self.shift_weight < 0:
            raise ValueError("weights must be non-negative")
        if abs(self.trust_weight + self.shift_weight - 1.0) > 1e-9:
            raise ValueError("weights must sum to one")

    def score(self, monitor_input: MonitorInput) -> MonitorOutput:
        summary = monitor_input.tom_summary
        if summary is None:
            return self.emit(
                monitor_input, 0.5, reason_codes=("missing_tom_state",)
            )
        reasons: list[str] = []
        trust = summary.get(self.trust_key)
        if trust is None:
            trust_component = 0.5
            reasons.append("missing_trust_signal")
        else:
            trust_component = min(max(1.0 - trust, 0.0), 1.0)
        shift = summary.get(self.shift_key)
        if shift is None:
            shift_component = 0.5
            reasons.append("missing_shift_signal")
        else:
            shift_component = min(max(abs(shift), 0.0), 1.0)
        suspicion = (
            self.trust_weight * trust_component
            + self.shift_weight * shift_component
        )
        return self.emit(
            monitor_input, suspicion, reason_codes=tuple(reasons)
        )


__all__ = ["BeliefConsistencyMonitor"]
