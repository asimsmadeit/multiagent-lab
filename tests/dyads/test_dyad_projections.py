"""Example-table builder tests for dyadic projections (Plan 4, Phase 6)."""

from __future__ import annotations

import pytest

from interpretability.dyads.projections import (
    LABEL_STATUS_CONFLICT,
    LABEL_STATUS_LABELED,
    LABEL_STATUS_UNLABELED,
    QC_STATUS_MISSING,
    QC_STATUS_PASSED,
    LabelSelection,
    build_dyad_outcome_row,
    build_receiver_processing_rows,
    build_send_receive_pairs,
    build_sender_turn_rows,
    build_temporal_windows,
    negotiation_training_rows,
    resolve_label,
)
from interpretability.events.payloads import LabelProvenance, LabelValue
from interpretability.events.projectors import (
    BehaviorLabelProjection,
    DyadProjection,
    DyadTurnProjection,
    OutcomeProjection,
    QualityControlProjection,
)

SELECTION = LabelSelection(label_name="actual_deception")
STAGES = {"call-a1": ("generated_last",), "call-b1": ("message_read_span",)}


def uid(index: int) -> str:
    """Canonical-UUID event ids, required by label provenance."""
    return f"90000000-0000-4000-8000-{index:012x}"


def turn(
    ordinal: int,
    *,
    actor: str = "alice",
    action_event: str | None = None,
    response_event: str | None = "evt-response",
    response_call: str | None = "call-b1",
) -> DyadTurnProjection:
    counterpart = "bob" if actor == "alice" else "alice"
    event_id = action_event or uid(ordinal)
    return DyadTurnProjection(
        ordinal=ordinal,
        dyad_id="dyad-1",
        trial_id="trial-1",
        turn_id=f"turn-{ordinal}",
        turn_event_id=f"evt-turn-{ordinal}",
        actor_id=actor,
        actor_role="potential_deceiver" if actor == "alice" else "counterpart",
        counterpart_actor_id=counterpart,
        counterpart_role="counterpart" if actor == "alice" else (
            "potential_deceiver"
        ),
        action_id=f"action-{ordinal}",
        action_hash="a" * 64,
        action_event_id=event_id,
        model_call_id="call-a1" if actor == "alice" else "call-b1",
        reception_event_ids=(f"evt-obs-{ordinal}",),
        response_action_event_id=response_event,
        response_model_call_id=response_call,
        annotation_event_ids=(),
    )


def dyad(*turns: DyadTurnProjection) -> DyadProjection:
    return DyadProjection(
        run_id="run-1",
        pod_id="pod-1",
        trial_id="trial-1",
        dyad_id="dyad-1",
        actor_ids=("alice", "bob"),
        actor_roles=(
            ("alice", "potential_deceiver"),
            ("bob", "counterpart"),
        ),
        turns=tuple(turns),
    )


def label(
    target: str,
    *,
    source: str = "rules",
    event_id: str = "evt-label-1",
    boolean: bool | None = True,
    score: float | None = None,
) -> BehaviorLabelProjection:
    kind = "boolean" if boolean is not None else "score"
    return BehaviorLabelProjection(
        event_id=event_id,
        label_id=f"label-{event_id}",
        target_event_id=target,
        target_actor_id="alice",
        label_name="actual_deception",
        value=LabelValue(kind=kind, boolean_value=boolean, score_value=score),
        provenance=LabelProvenance(
            source=source,
            method_id="rule-evaluator",
            method_version="1.0.0",
            source_event_ids=(target,),
            evaluation_succeeded=True,
        ),
        parent_event_ids=(target,),
    )


def qc(target: str, *, passed: bool = True) -> QualityControlProjection:
    return QualityControlProjection(
        event_id=f"evt-qc-{target}",
        qc_id=f"qc-{target}",
        qc_version="qc/1.0",
        target_event_id=target,
        passed=passed,
        flags=() if passed else ("repetition_loop",),
        source_event_ids=(target,),
        parent_event_ids=(target,),
    )


def build_senders(dyad_projection, labels=(), quality=(), **overrides):
    kwargs = dict(
        labels=labels,
        quality_controls=quality,
        captured_stages_by_call=STAGES,
        selection=SELECTION,
        family_id="family-1",
        partition="train",
    )
    kwargs.update(overrides)
    return build_sender_turn_rows(dyad_projection, **kwargs)


# ---------------------------------------------------------------------------
# Label resolution
# ---------------------------------------------------------------------------


def test_resolve_label_states_and_source_filter() -> None:
    unlabeled = resolve_label((), SELECTION)
    assert unlabeled.status == LABEL_STATUS_UNLABELED
    assert unlabeled.value is None

    one = resolve_label((label(uid(1)),), SELECTION)
    assert one.status == LABEL_STATUS_LABELED and one.value is True

    scored = resolve_label(
        (label(uid(1), boolean=None, score=0.8, event_id="evt-label-s"),),
        SELECTION,
    )
    assert scored.value == pytest.approx(0.8)

    conflicting = resolve_label(
        (
            label(uid(1), source="rules", event_id="evt-label-1"),
            label(uid(1), source="judge", event_id="evt-label-2"),
        ),
        SELECTION,
    )
    assert conflicting.status == LABEL_STATUS_CONFLICT
    assert conflicting.value is None
    assert conflicting.sources == ("judge", "rules")

    filtered = resolve_label(
        (
            label(uid(1), source="rules", event_id="evt-label-1"),
            label(uid(1), source="judge", event_id="evt-label-2"),
        ),
        LabelSelection(label_name="actual_deception", source="judge"),
    )
    assert filtered.status == LABEL_STATUS_LABELED


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def test_sender_rows_join_labels_qc_and_capture() -> None:
    first = turn(0)
    projection = dyad(first, turn(1, actor="bob", response_event=None,
                                  response_call=None))
    rows = build_senders(
        projection,
        labels=(label(first.action_event_id),),
        quality=(qc(first.action_event_id),),
    )
    assert len(rows) == 2
    lead = rows[0]
    assert lead.scope.trial_id == "trial-1"
    assert lead.scope.family_id == "family-1"
    assert lead.scope.partition == "train"
    assert lead.captured_stages == ("generated_last",)
    assert lead.label.status == LABEL_STATUS_LABELED
    assert lead.qc_status == QC_STATUS_PASSED
    follower = rows[1]
    assert follower.label.status == LABEL_STATUS_UNLABELED
    assert follower.qc_status == QC_STATUS_MISSING


def test_sender_rows_fail_closed_on_duplicates_and_bad_scope() -> None:
    first = turn(0)
    projection = dyad(first)
    with pytest.raises(ValueError, match="duplicate quality control"):
        build_senders(
            projection,
            quality=(qc(first.action_event_id), qc(first.action_event_id)),
        )
    with pytest.raises(ValueError, match="family_id"):
        build_senders(projection, family_id="")
    with pytest.raises(ValueError, match="unique ordinals"):
        build_senders(dyad(first, turn(0, actor="bob")))


def test_receiver_and_pair_rows_bind_the_response_side() -> None:
    first = turn(0, response_event=uid(1))
    second = turn(1, actor="bob", action_event=uid(1),
                  response_event=None, response_call=None)
    projection = dyad(first, second)
    receiver_rows = build_receiver_processing_rows(
        projection,
        labels=(label(first.action_event_id),),
        quality_controls=(),
        captured_stages_by_call=STAGES,
        selection=SELECTION,
        family_id="family-1",
    )
    assert receiver_rows[0].recipient_id == "bob"
    assert receiver_rows[0].receiver_captured_stages == (
        "message_read_span",
    )
    assert receiver_rows[0].label.status == LABEL_STATUS_LABELED

    pairs = build_send_receive_pairs(
        projection,
        labels=(),
        quality_controls=(),
        captured_stages_by_call=STAGES,
        selection=SELECTION,
        family_id="family-1",
    )
    assert pairs[0].response_action_event_id == uid(1)
    assert pairs[0].sender_captured_stages == ("generated_last",)

    stray = dyad(turn(0, response_event=uid(999)))
    with pytest.raises(ValueError, match="outside this dyad projection"):
        build_send_receive_pairs(
            stray,
            labels=(),
            quality_controls=(),
            captured_stages_by_call=STAGES,
            selection=SELECTION,
            family_id="family-1",
        )


def test_temporal_windows_keep_early_turns_and_one_scope() -> None:
    projection = dyad(
        turn(0), turn(1, actor="bob", response_event=None,
                      response_call=None), turn(2)
    )
    senders = build_senders(projection)
    windows = build_temporal_windows(senders, window_size=2)
    assert [w.final_ordinal for w in windows] == [0, 1, 2]
    assert windows[0].turn_ids == ("turn-0",)
    assert windows[2].turn_ids == ("turn-1", "turn-2")
    with pytest.raises(ValueError, match="positive integer"):
        build_temporal_windows(senders, window_size=0)
    other_scope = build_senders(projection, family_id="family-2")
    with pytest.raises(ValueError, match="cross trial boundaries"):
        build_temporal_windows(senders + other_scope, window_size=2)
    assert build_temporal_windows((), window_size=2) == ()


def test_dyad_outcome_row_aggregates_labels_and_outcome() -> None:
    first, third = turn(0), turn(2)
    projection = dyad(
        first, turn(1, actor="bob", response_event=None,
                    response_call=None), third
    )
    senders = build_senders(
        projection,
        labels=(
            label(first.action_event_id),
            label(third.action_event_id, source="rules",
                  event_id="evt-label-3"),
            label(third.action_event_id, source="judge",
                  event_id="evt-label-4"),
        ),
    )
    outcome = OutcomeProjection(
        event_id="evt-outcome-1",
        outcome_id="outcome-1",
        trial_id="trial-1",
        resolver_id="resolver",
        resolver_version="1.0.0",
        outcome_schema_version="1.0.0",
        outcome_json="{}",
        outcome_hash="b" * 64,
        source_event_ids=(),
        success=True,
        score=0.75,
        parent_event_ids=(),
    )
    row = build_dyad_outcome_row(
        projection,
        sender_rows=senders,
        outcomes=(outcome,),
        selection=SELECTION,
        family_id="family-1",
        partition="train",
    )
    assert row.n_turns == 3
    assert row.n_labeled_deceptive == 1
    assert row.n_label_conflicts == 1
    assert row.outcome_success is True and row.outcome_score == 0.75

    with pytest.raises(ValueError, match="at most one outcome"):
        build_dyad_outcome_row(
            projection,
            sender_rows=senders,
            outcomes=(outcome, outcome),
            selection=SELECTION,
            family_id="family-1",
        )


def test_negotiation_training_filter_keeps_only_clean_labeled_rows() -> None:
    first, second = turn(0), turn(1, actor="bob", response_event=None,
                                  response_call=None)
    projection = dyad(first, second)
    rows = build_senders(
        projection,
        labels=(label(first.action_event_id),
                label(second.action_event_id, event_id="evt-label-2")),
        quality=(qc(first.action_event_id),
                 qc(second.action_event_id, passed=False)),
    )
    strict = negotiation_training_rows(rows)
    assert [row.turn_id for row in strict] == ["turn-0"]
    lenient = negotiation_training_rows(rows, require_qc_pass=False)
    assert len(lenient) == 2
