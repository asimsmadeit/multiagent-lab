"""Monitor and control-protocol runtime contracts (Plan 6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from interpretability.monitoring.schema import (
    MonitorInput,
    MonitorOutput,
    MonitorView,
    ProtocolDecision,
    ThresholdCalibration,
    frozen_monitor_hash,
)


@runtime_checkable
class Monitor(Protocol):
    """One frozen scorer over view-gated inputs."""

    monitor_id: str
    monitor_version: str
    required_views: tuple[MonitorView, ...]

    def score(self, monitor_input: MonitorInput) -> MonitorOutput:
        """Return normalized suspicion for one input."""


@runtime_checkable
class ControlProtocol(Protocol):
    """Maps one monitor output to one declared control decision."""

    protocol_id: str
    protocol_version: str

    def decide(
        self,
        output: MonitorOutput,
        *,
        proposed_action_event_id: str,
    ) -> ProtocolDecision:
        """Return the decision for one proposed action."""


@dataclass(frozen=True)
class MonitorBase:
    """Shared emission logic: coverage, calibration, and freezing."""

    monitor_id: str
    monitor_version: str
    calibration: ThresholdCalibration | None = None

    required_views: tuple[MonitorView, ...] = ()

    def _validate_calibration(self) -> None:
        if self.calibration is None:
            return
        if (
            self.calibration.monitor_id != self.monitor_id
            or self.calibration.monitor_version != self.monitor_version
        ):
            raise ValueError(
                "calibration belongs to a different monitor or version"
            )

    def emit(
        self,
        monitor_input: MonitorInput,
        suspicion_score: float,
        *,
        reason_codes: tuple[str, ...] = (),
        source_event_ids: tuple[str, ...] = (),
    ) -> MonitorOutput:
        self._validate_calibration()
        coverage = {
            view: monitor_input.has_data(view)
            for view in self.required_views
        }
        missing = tuple(
            sorted(
                (view for view, present in coverage.items() if not present),
                key=lambda view: view.value,
            )
        )
        threshold = (
            self.calibration.threshold if self.calibration else None
        )
        threshold_version = (
            self.calibration.calibration_id if self.calibration else None
        )
        flagged = threshold is not None and suspicion_score >= threshold
        return MonitorOutput(
            monitor_id=self.monitor_id,
            monitor_version=self.monitor_version,
            trial_id=monitor_input.trial_id,
            target_event_id=monitor_input.target_event_id,
            suspicion_score=float(suspicion_score),
            threshold=threshold,
            threshold_version=threshold_version,
            flagged=flagged,
            reason_codes=reason_codes,
            coverage=coverage,
            missing_views=missing,
            source_event_ids=source_event_ids,
        )

    def frozen_hash(self, parameters: dict) -> str:
        return frozen_monitor_hash(
            self.monitor_id, self.monitor_version, self.calibration, parameters
        )


__all__ = ["ControlProtocol", "Monitor", "MonitorBase"]
