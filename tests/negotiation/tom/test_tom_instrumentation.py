"""Contract tests for event-safe ToM publications (Plan 3)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from negotiation.components.tom.instrumentation import (
    EvidenceEventLink,
    PartnerBeliefPublication,
    ScopedModelCall,
    tom_state_updated_fields,
)
from negotiation.components.tom.schema import (
    BeliefDistribution,
    BeliefUpdate,
    EpistemicStatus,
    Evidence,
    EvidenceChannel,
    EvidenceVisibility,
    GroundTruthKind,
    PartnerBeliefState,
    UpdateMethod,
)

_TEXT_HASH = "sha256:" + "a" * 64
EVENT_UUID = "70000000-0000-4000-8000-000000000031"


def _distribution(
    target: str = "policy_type",
    categories: tuple[str, ...] = ("skeptical", "unknown"),
    probabilities: tuple[float, ...] = (0.6, 0.4),
    *,
    status: EpistemicStatus = EpistemicStatus.PRIOR,
) -> BeliefDistribution:
    return BeliefDistribution(
        target=target,
        categories=categories,
        probabilities=probabilities,
        unknown_category="unknown",
        epistemic_status=status,
        ground_truth_kind=GroundTruthKind.OBJECTIVE,
    )


def _evidence(
    *,
    event_id: str = "event-1",
    features: tuple[tuple[str, Any], ...] = (("requested_evidence", True),),
) -> Evidence:
    return Evidence(
        observer_id="Seller",
        source_actor_id="Buyer",
        source_event_id=event_id,
        source_call_id="call-1",
        turn=1,
        features=features,
        channel=EvidenceChannel.OBSERVABLE,
        visibility=EvidenceVisibility.PUBLIC,
        visible_to=(),
        reliability=0.9,
        extractor_version="rules-1",
        source_text_hash=_TEXT_HASH,
        source_span=(3, 12),
        summary="counterpart requested documentation",
    )


def _belief_update(evidence: Evidence | None = None) -> BeliefUpdate:
    return BeliefUpdate(
        prior=_distribution(probabilities=(0.5, 0.5)),
        evidence=(evidence or _evidence(),),
        likelihoods=(0.9, 0.1),
        posterior=_distribution(
            probabilities=(0.9, 0.1), status=EpistemicStatus.UPDATED
        ),
        method=UpdateMethod.BAYESIAN,
        updater_version="bayes-1",
        observation_model_version="controlled-table-1",
    )


def _partner_state(update: BeliefUpdate | None) -> PartnerBeliefState:
    return PartnerBeliefState(
        observer_id="Seller",
        counterpart_id="Buyer",
        state_version=1,
        policy_type=update.posterior if update else _distribution(),
        expected_next_action=_distribution(
            "next_action", ("accept", "counter", "unknown"), (0.2, 0.6, 0.2)
        ),
        reservation_value=_distribution(
            "reservation_value", ("high", "low", "unknown"), (0.4, 0.4, 0.2)
        ),
        goal_beliefs=(
            _distribution("goal.value", ("maximize_value", "unknown"), (0.8, 0.2)),
        ),
        constraint_beliefs=(
            _distribution("constraint.time", ("deadline", "unknown"), (0.3, 0.7)),
        ),
        fact_beliefs=(
            _distribution(
                "fact.quality", ("false", "true", "unknown"), (0.2, 0.5, 0.3)
            ),
        ),
        trustworthiness=_distribution(
            "trustworthiness", ("trustworthy", "unknown"), (0.5, 0.5)
        ),
        evidence_ids=(update.evidence_ids[0],) if update else (),
        update_ids=(update.update_id,) if update else (),
    )


def _link(evidence: Evidence, event_uuid: str = EVENT_UUID) -> EvidenceEventLink:
    return EvidenceEventLink.bind(
        trial_id="trial-1",
        through_turn=1,
        actor_id="Seller",
        counterpart_id="Buyer",
        evidence=evidence,
        evidence_event_id=event_uuid,
    )


def _call() -> ScopedModelCall:
    return ScopedModelCall(
        trial_id="trial-1", turn=1, actor_id="Seller", call_id="call-tom-1"
    )


def _publication(
    *,
    with_call: bool = True,
    updates: tuple[BeliefUpdate, ...] | None = None,
    links: tuple[EvidenceEventLink, ...] | None = None,
) -> PartnerBeliefPublication:
    update = _belief_update()
    resolved_updates = updates if updates is not None else (update,)
    if links is None:
        links = tuple(
            _link(item)
            for resolved in resolved_updates
            for item in resolved.evidence
        )
    return PartnerBeliefPublication.from_state(
        trial_id="trial-1",
        turn=1,
        state=_partner_state(resolved_updates[0] if resolved_updates else update),
        updates=resolved_updates,
        evidence_event_links=links,
        source_model_call=_call() if with_call else None,
    )


# ---------------------------------------------------------------------------
# Publication construction
# ---------------------------------------------------------------------------


def test_publication_binds_state_hash_and_sorts_distributions() -> None:
    update = _belief_update()
    state = _partner_state(update)
    publication = PartnerBeliefPublication.from_state(
        trial_id="trial-1",
        turn=1,
        state=state,
        updates=(update,),
        evidence_event_links=(_link(update.evidence[0]),),
        source_model_call=_call(),
    )
    assert publication.state_hash == state.state_hash
    assert publication.state_id.startswith("tom_state_")
    targets = [snapshot.target for snapshot in publication.distributions]
    assert targets == sorted(targets)
    assert publication.actor_id == "Seller"
    assert publication.counterpart_id == "Buyer"
    assert len(publication.updates) == 1
    assert len(publication.evidence) == 1


def test_publication_is_deterministic_for_identical_inputs() -> None:
    assert _publication() == _publication()


def test_event_links_must_exactly_cover_update_evidence() -> None:
    update = _belief_update()
    with pytest.raises(ValueError, match="exactly cover"):
        _publication(updates=(update,), links=())
    stray = _evidence(event_id="event-unrelated", features=(("offer", 12),))
    with pytest.raises(ValueError, match="exactly cover"):
        _publication(
            updates=(update,),
            links=(_link(update.evidence[0]), _link(stray)),
        )
    with pytest.raises(ValueError, match="unique evidence IDs"):
        _publication(
            updates=(update,),
            links=(_link(update.evidence[0]), _link(update.evidence[0])),
        )


def test_evidence_cannot_be_reused_across_updates() -> None:
    update = _belief_update()
    with pytest.raises(ValueError, match="reused across updates"):
        _publication(
            updates=(update, update), links=(_link(update.evidence[0]),)
        )


# ---------------------------------------------------------------------------
# Event-payload bridge
# ---------------------------------------------------------------------------


def test_bridge_fields_construct_a_real_event_payload() -> None:
    from interpretability.events.payloads import ToMStateUpdatedPayload

    publication = _publication()
    fields = publication.to_tom_state_updated_fields()
    payload = ToMStateUpdatedPayload(**fields)
    assert payload.actor_id == "Seller"
    assert payload.counterpart_actor_id == "Buyer"
    assert payload.state_hash == publication.state_hash.removeprefix("sha256:")
    assert payload.evidence_event_ids == (EVENT_UUID,)
    assert payload.source_model_call_id == "call-tom-1"
    assert payload.state_id == publication.state_id


def test_bridge_requires_evidence_and_a_real_call() -> None:
    no_call = _publication(with_call=False)
    with pytest.raises(ValueError, match="real source model call"):
        no_call.to_tom_state_updated_fields()
    # A prior-only state (no update or evidence lineage yet) publishes, but
    # cannot announce a state transition to the event layer.
    empty = PartnerBeliefPublication.from_state(
        trial_id="trial-1",
        turn=1,
        state=_partner_state(None),
        updates=(),
        evidence_event_links=(),
        source_model_call=_call(),
    )
    with pytest.raises(ValueError, match="at least one evidence"):
        empty.to_tom_state_updated_fields()


def test_functional_adapter_rejects_non_publications() -> None:
    publication = _publication()
    assert tom_state_updated_fields(publication) == (
        publication.to_tom_state_updated_fields()
    )
    with pytest.raises(TypeError, match="PartnerBeliefPublication"):
        tom_state_updated_fields(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Information-channel guards
# ---------------------------------------------------------------------------


def test_sensitive_identifiers_are_rejected() -> None:
    with pytest.raises((ValueError, ValidationError), match="prohibited"):
        ScopedModelCall(
            trial_id="trial-1", turn=1, actor_id="adjudicator", call_id="call-1"
        )
    with pytest.raises((ValueError, ValidationError), match="prohibited"):
        EvidenceEventLink.bind(
            trial_id="chain_of_thought",
            through_turn=1,
            actor_id="Seller",
            counterpart_id="Buyer",
            evidence=_evidence(),
            evidence_event_id=EVENT_UUID,
        )


def test_link_requires_distinct_actor_and_counterpart() -> None:
    with pytest.raises((ValueError, ValidationError), match="distinct"):
        EvidenceEventLink.bind(
            trial_id="trial-1",
            through_turn=1,
            actor_id="Seller",
            counterpart_id="Seller",
            evidence=_evidence(),
            evidence_event_id=EVENT_UUID,
        )
