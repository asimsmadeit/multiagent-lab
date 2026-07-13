"""Typed scheduling helpers for non-default experiment protocols."""

from __future__ import annotations

from dataclasses import dataclass

from negotiation.domain import ActionKind, NegotiationAction
from negotiation.game_master.adjudication import (
    ActionResolution,
    NegotiationAdjudicator,
)


SOLO_NO_RESPONSE_PROTOCOL_VERSION = "solo-no-response/1"


@dataclass(frozen=True)
class SoloNoResponseScheduler:
    """Commit one deterministic environment event without a model call.

    The adjudicator still has two protocol participants so domain validation,
    round accounting, and replay retain their ordinary guarantees. The absent
    role's turn is supplied by this scheduler as an explicit environment
    event; it is never represented as an empty prompt or a generated response.
    """

    environment_actor_id: str
    protocol_version: str = SOLO_NO_RESPONSE_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.environment_actor_id, str)
            or not self.environment_actor_id.strip()
        ):
            raise ValueError("environment_actor_id must be a non-empty string")
        if self.protocol_version != SOLO_NO_RESPONSE_PROTOCOL_VERSION:
            raise ValueError("unsupported solo no-response protocol version")

    def submit_no_response(
        self,
        adjudicator: NegotiationAdjudicator,
        *,
        round_index: int,
    ) -> ActionResolution:
        """Commit the absent role's idempotent environment action."""
        if not isinstance(adjudicator, NegotiationAdjudicator):
            raise TypeError("adjudicator must be a NegotiationAdjudicator")
        if type(round_index) is not int or round_index < 0:
            raise ValueError("round_index must be a non-negative integer")
        if adjudicator.protocol != "alternating":
            raise ValueError(
                "solo no-response scheduling requires alternating adjudication"
            )
        if self.environment_actor_id not in adjudicator.participants:
            raise ValueError("environment actor is not an adjudication participant")
        if adjudicator.next_actor != self.environment_actor_id:
            raise RuntimeError(
                "no-response environment event is not due for this actor"
            )
        action = NegotiationAction(
            action_ref=(
                f"{self.protocol_version}:{adjudicator.negotiation_id}:"
                f"round:{round_index}"
            ),
            actor_id=self.environment_actor_id,
            # ``DISCLOSE`` is the domain's non-offer public-text action. The
            # raw text and scheduler version distinguish this environment
            # event from a participant disclosure.
            kind=ActionKind.DISCLOSE,
            raw_text=(
                "[NO_RESPONSE_ENVIRONMENT] The counterpart is absent; "
                "no response is available."
            ),
        )
        resolution = adjudicator.submit(action)
        if not resolution.accepted:
            raise RuntimeError("no-response environment event was rejected")
        return resolution

    @staticmethod
    def public_observation(resolution: ActionResolution) -> str:
        """Render the committed environment event for the acting agent."""
        if not isinstance(resolution, ActionResolution):
            raise TypeError("resolution must be an ActionResolution")
        if not resolution.accepted:
            raise ValueError("cannot observe a rejected environment event")
        return (
            f"{resolution.event.action.actor_id}: "
            f"{resolution.event.action.raw_text}"
        )


__all__ = [
    "SOLO_NO_RESPONSE_PROTOCOL_VERSION",
    "SoloNoResponseScheduler",
]
