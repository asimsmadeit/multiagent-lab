"""Dyadic example tables from event projections (Plan 4, Phase 6).

Builders join the events layer's ``DyadProjection`` turns with behavior
labels, quality control, and capture availability into five versioned row
families: ``sender_turn``, ``receiver_processing``, ``send_receive_pair``,
``temporal_window``, and ``dyad_outcome``.

Three rules from the cross-plan contract are enforced mechanically:

1. Label sources never collapse silently — a target with several labels for
   the selected (name, source) pair is a ``conflict`` row, not a coin flip.
2. Missingness is explicit — absent QC, absent labels, and absent capture
   stages are recorded states, never empty-string defaults.
3. Rows carry family/partition identities so downstream loaders can reject
   splits that were not built family-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from interpretability.events.projectors import (
    BehaviorLabelProjection,
    DyadProjection,
    DyadTurnProjection,
    OutcomeProjection,
    QualityControlProjection,
)

DYAD_PROJECTION_VERSION = "dyad-projections/1.0.0"

LABEL_STATUS_LABELED = "labeled"
LABEL_STATUS_UNLABELED = "unlabeled"
LABEL_STATUS_CONFLICT = "conflict"

QC_STATUS_PASSED = "passed"
QC_STATUS_FAILED = "failed"
QC_STATUS_MISSING = "missing"


def _require_identifier(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class LabelSelection:
    """Which behavior label defines the prediction target for a table."""

    label_name: str
    source: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.label_name, "label_name")
        if self.source is not None:
            _require_identifier(self.source, "source")

    def matches(self, label: BehaviorLabelProjection) -> bool:
        if label.label_name != self.label_name:
            return False
        return self.source is None or label.provenance.source == self.source


@dataclass(frozen=True)
class ResolvedLabel:
    """Outcome of selecting one label for one target event."""

    status: str
    value: bool | float | str | None
    label_event_ids: tuple[str, ...]
    sources: tuple[str, ...]


def _label_scalar(label: BehaviorLabelProjection) -> bool | float | str:
    value = label.value
    if value.kind == "boolean":
        return bool(value.boolean_value)
    if value.kind == "score":
        return float(value.score_value)
    return str(value.category_value)


def resolve_label(
    labels: Sequence[BehaviorLabelProjection],
    selection: LabelSelection,
) -> ResolvedLabel:
    """Select at most one label; several matches are an explicit conflict."""
    matching = [label for label in labels if selection.matches(label)]
    if not matching:
        return ResolvedLabel(
            status=LABEL_STATUS_UNLABELED,
            value=None,
            label_event_ids=(),
            sources=(),
        )
    if len(matching) > 1:
        return ResolvedLabel(
            status=LABEL_STATUS_CONFLICT,
            value=None,
            label_event_ids=tuple(
                sorted(label.event_id for label in matching)
            ),
            sources=tuple(
                sorted({label.provenance.source for label in matching})
            ),
        )
    label = matching[0]
    return ResolvedLabel(
        status=LABEL_STATUS_LABELED,
        value=_label_scalar(label),
        label_event_ids=(label.event_id,),
        sources=(label.provenance.source,),
    )


def _qc_status(
    qc: QualityControlProjection | None,
) -> tuple[str, tuple[str, ...]]:
    if qc is None:
        return QC_STATUS_MISSING, ()
    return (
        QC_STATUS_PASSED if qc.passed else QC_STATUS_FAILED,
        tuple(qc.flags),
    )


@dataclass(frozen=True)
class _TrialScope:
    """Identity constants shared by every row built from one trial."""

    run_id: str | None
    pod_id: str | None
    trial_id: str
    dyad_id: str
    family_id: str
    partition: str | None


@dataclass(frozen=True)
class SenderTurnRow:
    """Predict the sender's behavioral label from the sender's turn."""

    projection_version: str
    scope: _TrialScope
    ordinal: int
    turn_id: str
    actor_id: str
    actor_role: str
    recipient_id: str
    recipient_role: str
    action_event_id: str
    model_call_id: str
    captured_stages: tuple[str, ...]
    label: ResolvedLabel
    qc_status: str
    qc_flags: tuple[str, ...]
    sample_type: str = "negotiation"


@dataclass(frozen=True)
class ReceiverProcessingRow:
    """Predict the received message's label from the receiver's side."""

    projection_version: str
    scope: _TrialScope
    ordinal: int
    sender_turn_id: str
    sender_action_event_id: str
    recipient_id: str
    recipient_role: str
    reception_event_ids: tuple[str, ...]
    response_model_call_id: str | None
    receiver_captured_stages: tuple[str, ...]
    label: ResolvedLabel
    qc_status: str
    qc_flags: tuple[str, ...]
    sample_type: str = "negotiation"


@dataclass(frozen=True)
class SendReceivePairRow:
    """Join one sent action with the counterpart's processing and response."""

    projection_version: str
    scope: _TrialScope
    ordinal: int
    sender_turn_id: str
    sender_action_event_id: str
    sender_model_call_id: str
    sender_captured_stages: tuple[str, ...]
    recipient_id: str
    reception_event_ids: tuple[str, ...]
    response_action_event_id: str | None
    response_model_call_id: str | None
    receiver_captured_stages: tuple[str, ...]
    label: ResolvedLabel
    qc_status: str
    qc_flags: tuple[str, ...]
    sample_type: str = "negotiation"


@dataclass(frozen=True)
class TemporalWindowRow:
    """The prior-K paired turns ending at (and labeled by) one turn."""

    projection_version: str
    scope: _TrialScope
    window_size: int
    final_ordinal: int
    turn_ids: tuple[str, ...]
    action_event_ids: tuple[str, ...]
    label: ResolvedLabel
    qc_status: str
    qc_flags: tuple[str, ...]
    sample_type: str = "negotiation"


@dataclass(frozen=True)
class DyadOutcomeRow:
    """Whole-trajectory target for agreement/exploitation prediction."""

    projection_version: str
    scope: _TrialScope
    outcome_event_id: str | None
    outcome_success: bool | None
    outcome_score: float | None
    n_turns: int
    n_labeled_deceptive: int
    n_label_conflicts: int
    label_name: str


def _scope(dyad: DyadProjection, family_id: str, partition: str | None) -> _TrialScope:
    if dyad.trial_id is None or dyad.dyad_id is None:
        raise ValueError("dyad projection must identify a trial and dyad")
    _require_identifier(family_id, "family_id")
    if partition is not None:
        _require_identifier(partition, "partition")
    return _TrialScope(
        run_id=dyad.run_id,
        pod_id=dyad.pod_id,
        trial_id=dyad.trial_id,
        dyad_id=dyad.dyad_id,
        family_id=family_id,
        partition=partition,
    )


def _indexed_annotations(
    labels: Sequence[BehaviorLabelProjection],
    quality_controls: Sequence[QualityControlProjection],
) -> tuple[
    dict[str, list[BehaviorLabelProjection]],
    dict[str, QualityControlProjection],
]:
    labels_by_target: dict[str, list[BehaviorLabelProjection]] = {}
    for label in labels:
        labels_by_target.setdefault(label.target_event_id, []).append(label)
    qc_by_target: dict[str, QualityControlProjection] = {}
    for qc in quality_controls:
        if qc.target_event_id in qc_by_target:
            raise ValueError(
                f"duplicate quality control for target {qc.target_event_id}"
            )
        qc_by_target[qc.target_event_id] = qc
    return labels_by_target, qc_by_target


def _stages(
    captured_stages_by_call: Mapping[str, Sequence[str]],
    model_call_id: str | None,
) -> tuple[str, ...]:
    if model_call_id is None:
        return ()
    return tuple(captured_stages_by_call.get(model_call_id, ()))


def _sorted_turns(dyad: DyadProjection) -> tuple[DyadTurnProjection, ...]:
    turns = tuple(sorted(dyad.turns, key=lambda turn: turn.ordinal))
    ordinals = [turn.ordinal for turn in turns]
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("dyad turns must have unique ordinals")
    return turns


def build_sender_turn_rows(
    dyad: DyadProjection,
    *,
    labels: Sequence[BehaviorLabelProjection],
    quality_controls: Sequence[QualityControlProjection],
    captured_stages_by_call: Mapping[str, Sequence[str]],
    selection: LabelSelection,
    family_id: str,
    partition: str | None = None,
) -> tuple[SenderTurnRow, ...]:
    scope = _scope(dyad, family_id, partition)
    labels_by_target, qc_by_target = _indexed_annotations(
        labels, quality_controls
    )
    rows = []
    for turn in _sorted_turns(dyad):
        qc_status, qc_flags = _qc_status(
            qc_by_target.get(turn.action_event_id)
        )
        rows.append(
            SenderTurnRow(
                projection_version=DYAD_PROJECTION_VERSION,
                scope=scope,
                ordinal=turn.ordinal,
                turn_id=turn.turn_id,
                actor_id=turn.actor_id,
                actor_role=turn.actor_role,
                recipient_id=turn.counterpart_actor_id,
                recipient_role=turn.counterpart_role,
                action_event_id=turn.action_event_id,
                model_call_id=turn.model_call_id,
                captured_stages=_stages(
                    captured_stages_by_call, turn.model_call_id
                ),
                label=resolve_label(
                    labels_by_target.get(turn.action_event_id, ()), selection
                ),
                qc_status=qc_status,
                qc_flags=qc_flags,
            )
        )
    return tuple(rows)


def build_receiver_processing_rows(
    dyad: DyadProjection,
    *,
    labels: Sequence[BehaviorLabelProjection],
    quality_controls: Sequence[QualityControlProjection],
    captured_stages_by_call: Mapping[str, Sequence[str]],
    selection: LabelSelection,
    family_id: str,
    partition: str | None = None,
) -> tuple[ReceiverProcessingRow, ...]:
    scope = _scope(dyad, family_id, partition)
    labels_by_target, qc_by_target = _indexed_annotations(
        labels, quality_controls
    )
    rows = []
    for turn in _sorted_turns(dyad):
        qc_status, qc_flags = _qc_status(
            qc_by_target.get(turn.action_event_id)
        )
        rows.append(
            ReceiverProcessingRow(
                projection_version=DYAD_PROJECTION_VERSION,
                scope=scope,
                ordinal=turn.ordinal,
                sender_turn_id=turn.turn_id,
                sender_action_event_id=turn.action_event_id,
                recipient_id=turn.counterpart_actor_id,
                recipient_role=turn.counterpart_role,
                reception_event_ids=turn.reception_event_ids,
                response_model_call_id=turn.response_model_call_id,
                receiver_captured_stages=_stages(
                    captured_stages_by_call, turn.response_model_call_id
                ),
                label=resolve_label(
                    labels_by_target.get(turn.action_event_id, ()), selection
                ),
                qc_status=qc_status,
                qc_flags=qc_flags,
            )
        )
    return tuple(rows)


def build_send_receive_pairs(
    dyad: DyadProjection,
    *,
    labels: Sequence[BehaviorLabelProjection],
    quality_controls: Sequence[QualityControlProjection],
    captured_stages_by_call: Mapping[str, Sequence[str]],
    selection: LabelSelection,
    family_id: str,
    partition: str | None = None,
) -> tuple[SendReceivePairRow, ...]:
    scope = _scope(dyad, family_id, partition)
    labels_by_target, qc_by_target = _indexed_annotations(
        labels, quality_controls
    )
    turns = _sorted_turns(dyad)
    known_action_events = {turn.action_event_id for turn in turns}
    rows = []
    for turn in turns:
        response_event = turn.response_action_event_id
        if response_event is not None and response_event not in (
            known_action_events
        ):
            raise ValueError(
                f"turn {turn.turn_id} references a response action outside "
                "this dyad projection"
            )
        qc_status, qc_flags = _qc_status(
            qc_by_target.get(turn.action_event_id)
        )
        rows.append(
            SendReceivePairRow(
                projection_version=DYAD_PROJECTION_VERSION,
                scope=scope,
                ordinal=turn.ordinal,
                sender_turn_id=turn.turn_id,
                sender_action_event_id=turn.action_event_id,
                sender_model_call_id=turn.model_call_id,
                sender_captured_stages=_stages(
                    captured_stages_by_call, turn.model_call_id
                ),
                recipient_id=turn.counterpart_actor_id,
                reception_event_ids=turn.reception_event_ids,
                response_action_event_id=response_event,
                response_model_call_id=turn.response_model_call_id,
                receiver_captured_stages=_stages(
                    captured_stages_by_call, turn.response_model_call_id
                ),
                label=resolve_label(
                    labels_by_target.get(turn.action_event_id, ()), selection
                ),
                qc_status=qc_status,
                qc_flags=qc_flags,
            )
        )
    return tuple(rows)


def build_temporal_windows(
    sender_rows: Sequence[SenderTurnRow],
    *,
    window_size: int,
) -> tuple[TemporalWindowRow, ...]:
    """Trailing windows over one dyad's sender rows, labeled by the last turn.

    A window never mixes trials/dyads: all rows must share one scope. Early
    turns with fewer than ``window_size`` predecessors are still emitted —
    truncation would silently drop exactly the early-warning cases the
    sequential track exists to measure.
    """
    if type(window_size) is not int or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    if not sender_rows:
        return ()
    scopes = {row.scope for row in sender_rows}
    if len(scopes) != 1:
        raise ValueError("temporal windows must not cross trial boundaries")
    ordered = sorted(sender_rows, key=lambda row: row.ordinal)
    rows = []
    for index, final_row in enumerate(ordered):
        window = ordered[max(0, index - window_size + 1): index + 1]
        rows.append(
            TemporalWindowRow(
                projection_version=DYAD_PROJECTION_VERSION,
                scope=final_row.scope,
                window_size=window_size,
                final_ordinal=final_row.ordinal,
                turn_ids=tuple(row.turn_id for row in window),
                action_event_ids=tuple(
                    row.action_event_id for row in window
                ),
                label=final_row.label,
                qc_status=final_row.qc_status,
                qc_flags=final_row.qc_flags,
            )
        )
    return tuple(rows)


def build_dyad_outcome_row(
    dyad: DyadProjection,
    *,
    sender_rows: Sequence[SenderTurnRow],
    outcomes: Sequence[OutcomeProjection],
    selection: LabelSelection,
    family_id: str,
    partition: str | None = None,
) -> DyadOutcomeRow:
    scope = _scope(dyad, family_id, partition)
    if len(outcomes) > 1:
        raise ValueError("a dyad outcome row accepts at most one outcome")
    outcome = outcomes[0] if outcomes else None
    if outcome is not None and outcome.trial_id != scope.trial_id:
        raise ValueError("outcome belongs to a different trial")
    deceptive = 0
    conflicts = 0
    for row in sender_rows:
        if row.scope != scope:
            raise ValueError("sender rows must share the dyad's scope")
        if row.label.status == LABEL_STATUS_CONFLICT:
            conflicts += 1
        elif row.label.status == LABEL_STATUS_LABELED and (
            row.label.value is True
            or (
                isinstance(row.label.value, float)
                and row.label.value > 0.5
            )
        ):
            deceptive += 1
    return DyadOutcomeRow(
        projection_version=DYAD_PROJECTION_VERSION,
        scope=scope,
        outcome_event_id=outcome.event_id if outcome else None,
        outcome_success=outcome.success if outcome else None,
        outcome_score=outcome.score if outcome else None,
        n_turns=len(sender_rows),
        n_labeled_deceptive=deceptive,
        n_label_conflicts=conflicts,
        label_name=selection.label_name,
    )


def negotiation_training_rows(rows: Sequence, *, require_qc_pass: bool = True):
    """Filter any row family down to primary-training eligibility."""
    kept = []
    for row in rows:
        if getattr(row, "sample_type", "negotiation") != "negotiation":
            continue
        if row.label.status != LABEL_STATUS_LABELED:
            continue
        if require_qc_pass and row.qc_status != QC_STATUS_PASSED:
            continue
        kept.append(row)
    return tuple(kept)


__all__ = [
    "DYAD_PROJECTION_VERSION",
    "LABEL_STATUS_CONFLICT",
    "LABEL_STATUS_LABELED",
    "LABEL_STATUS_UNLABELED",
    "QC_STATUS_FAILED",
    "QC_STATUS_MISSING",
    "QC_STATUS_PASSED",
    "DyadOutcomeRow",
    "LabelSelection",
    "ReceiverProcessingRow",
    "ResolvedLabel",
    "SendReceivePairRow",
    "SenderTurnRow",
    "TemporalWindowRow",
    "build_dyad_outcome_row",
    "build_receiver_processing_rows",
    "build_send_receive_pairs",
    "build_sender_turn_rows",
    "build_temporal_windows",
    "negotiation_training_rows",
    "resolve_label",
]
