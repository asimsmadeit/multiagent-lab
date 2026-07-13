"""Offline contracts for special experiment schedulers."""

from __future__ import annotations

import datetime

import pytest

from interpretability.runtime.protocols import (
    SOLO_NO_RESPONSE_PROTOCOL_VERSION,
    SoloNoResponseScheduler,
)
from negotiation.domain import ActionKind, NegotiationAction
from negotiation.game_master.adjudication import NegotiationAdjudicator
from negotiation.game_master.components.gm_state import NegotiationStateTracker


def _adjudicator() -> NegotiationAdjudicator:
    tracker = NegotiationStateTracker(
        max_rounds=3,
        enable_deadlines=False,
        clock=lambda: datetime.datetime(2040, 1, 2, 3, 4),
    )
    return NegotiationAdjudicator(
        negotiation_id="solo-trial",
        participants=("Negotiator", "AbsentCounterpart"),
        state_tracker=tracker,
    )


def test_solo_scheduler_commits_explicit_idempotent_environment_event() -> None:
    adjudicator = _adjudicator()
    actor_resolution = adjudicator.submit(NegotiationAction(
        action_ref="actor-round-0",
        actor_id="Negotiator",
        kind=ActionKind.DISCLOSE,
        raw_text="My offer is $70.",
    ))
    scheduler = SoloNoResponseScheduler("AbsentCounterpart")

    environment = scheduler.submit_no_response(adjudicator, round_index=0)

    assert actor_resolution.accepted and environment.accepted
    assert environment.event.action.kind is ActionKind.DISCLOSE
    assert environment.event.action.raw_text.startswith(
        "[NO_RESPONSE_ENVIRONMENT]"
    )
    assert environment.event.action_sequence == 1
    assert environment.event.committed_turn_index == 1
    assert adjudicator.next_actor == "Negotiator"
    assert scheduler.public_observation(environment).startswith(
        "AbsentCounterpart: [NO_RESPONSE_ENVIRONMENT]"
    )
    assert scheduler.protocol_version == SOLO_NO_RESPONSE_PROTOCOL_VERSION


def test_solo_scheduler_fails_closed_outside_exact_environment_turn() -> None:
    adjudicator = _adjudicator()
    scheduler = SoloNoResponseScheduler("AbsentCounterpart")

    with pytest.raises(RuntimeError, match="not due"):
        scheduler.submit_no_response(adjudicator, round_index=0)
    with pytest.raises(ValueError, match="non-negative"):
        scheduler.submit_no_response(adjudicator, round_index=-1)
    with pytest.raises(ValueError, match="non-empty"):
        SoloNoResponseScheduler("")
