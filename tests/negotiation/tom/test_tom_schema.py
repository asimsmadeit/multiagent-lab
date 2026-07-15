"""Contract tests for the canonical Theory of Mind v2 schema records."""

from __future__ import annotations

import json
import math
from typing import Any, Callable

import pytest
from pydantic import BaseModel, ValidationError

from negotiation.components.tom.schema import (
    TOM_SCHEMA_VERSION,
    BeliefDistribution,
    BeliefUpdate,
    EpistemicStatus,
    Evidence,
    EvidenceChannel,
    EvidenceVisibility,
    GroundTruthKind,
    PartnerBeliefState,
    RecursiveBeliefState,
    ToMDecisionTrace,
    UpdateMethod,
)


_TEXT_HASH = "sha256:" + "a" * 64


def _distribution(
    target: str = "policy_type",
    categories: tuple[str, ...] = ("skeptical", "unknown"),
    probabilities: tuple[float, ...] = (0.6, 0.4),
    *,
    status: EpistemicStatus = EpistemicStatus.PRIOR,
    ground_truth: GroundTruthKind = GroundTruthKind.OBJECTIVE,
    unknown_category: str = "unknown",
) -> BeliefDistribution:
    return BeliefDistribution(
        target=target,
        categories=categories,
        probabilities=probabilities,
        unknown_category=unknown_category,
        epistemic_status=status,
        ground_truth_kind=ground_truth,
    )


def _evidence(
    *,
    event_id: str = "event-1",
    observer_id: str = "Seller",
    source_actor_id: str = "Buyer",
    visibility: EvidenceVisibility = EvidenceVisibility.PUBLIC,
    visible_to: tuple[str, ...] = (),
    features: tuple[tuple[str, Any], ...] = (("requested_evidence", True),),
) -> Evidence:
    return Evidence(
        observer_id=observer_id,
        source_actor_id=source_actor_id,
        source_event_id=event_id,
        source_call_id="call-1",
        turn=1,
        features=features,
        channel=EvidenceChannel.OBSERVABLE,
        visibility=visibility,
        visible_to=visible_to,
        reliability=0.9,
        extractor_version="rules-1",
        source_text_hash=_TEXT_HASH,
        source_span=(3, 12),
        summary="counterpart requested documentation",
    )


def _belief_update() -> BeliefUpdate:
    return BeliefUpdate(
        prior=_distribution(probabilities=(0.5, 0.5)),
        evidence=(_evidence(),),
        likelihoods=(0.9, 0.1),
        posterior=_distribution(
            probabilities=(0.9, 0.1), status=EpistemicStatus.UPDATED
        ),
        method=UpdateMethod.BAYESIAN,
        updater_version="bayes-1",
        observation_model_version="controlled-table-1",
    )


def _partner_state() -> PartnerBeliefState:
    update = _belief_update()
    return PartnerBeliefState(
        observer_id="Seller",
        counterpart_id="Buyer",
        state_version=1,
        policy_type=update.posterior,
        expected_next_action=_distribution(
            "next_action",
            ("accept", "counter", "unknown"),
            (0.2, 0.6, 0.2),
        ),
        reservation_value=_distribution(
            "reservation_value",
            ("high", "low", "unknown"),
            (0.4, 0.4, 0.2),
        ),
        goal_beliefs=(
            _distribution(
                "goal.value", ("maximize_value", "unknown"), (0.8, 0.2)
            ),
        ),
        constraint_beliefs=(
            _distribution(
                "constraint.time", ("deadline", "unknown"), (0.3, 0.7)
            ),
        ),
        fact_beliefs=(
            _distribution(
                "fact.quality",
                ("false", "true", "unknown"),
                (0.2, 0.5, 0.3),
            ),
        ),
        trustworthiness=_distribution(
            "trustworthiness", ("trustworthy", "unknown"), (0.5, 0.5)
        ),
        evidence_ids=(update.evidence_ids[0],),
        update_ids=(update.update_id,),
    )


def _recursive_state(depth: int = 1) -> RecursiveBeliefState:
    state = _partner_state()
    path = (
        ("Seller", "Buyer")
        if depth == 1
        else ("Seller", "Buyer", "Seller")
    )
    return RecursiveBeliefState(
        depth=depth,
        root_observer_id="Seller",
        counterpart_id="Buyer",
        root_state_hash=state.state_hash,
        information_path=path,
        target_belief=state.fact_beliefs[0],
        evidence=(_evidence(),),
    )


def _decision_trace() -> ToMDecisionTrace:
    state = _partner_state()
    predicted_action = state.expected_next_action
    return ToMDecisionTrace(
        trial_id="trial-1",
        turn=2,
        actor_id="Seller",
        counterpart_id="Buyer",
        belief_state_hash=state.state_hash,
        belief_update_ids=state.update_ids,
        predicted_counterpart_action=predicted_action,
        legal_actions=("accept", "counter"),
        expected_utilities=(7.0, 9.0),
        conditional_predictions=(predicted_action, predicted_action),
        chosen_action="counter",
        intervention_condition="dynamic_tom",
        objective="expected_utility",
        advisor_version="advisor-1",
        recommendation_summary_hash="sha256:" + "b" * 64,
    )


def _identity(record: BaseModel) -> str:
    if isinstance(record, Evidence):
        return record.evidence_id
    if isinstance(record, BeliefUpdate):
        return record.update_id
    if isinstance(record, ToMDecisionTrace):
        return record.trace_id
    if isinstance(
        record,
        (BeliefDistribution, PartnerBeliefState, RecursiveBeliefState),
    ):
        return record.state_hash
    raise TypeError(type(record).__name__)


@pytest.mark.parametrize(
    "factory",
    [
        _evidence,
        _distribution,
        _partner_state,
        _recursive_state,
        _belief_update,
        _decision_trace,
    ],
    ids=[
        "evidence",
        "distribution",
        "partner-state",
        "recursive-state",
        "belief-update",
        "decision-trace",
    ],
)
def test_public_records_have_canonical_identity_and_json_round_trip(
    factory: Callable[[], BaseModel],
) -> None:
    record = factory()
    serialized = record.canonical_json()
    restored = type(record).model_validate_json(serialized)

    assert restored == record
    assert _identity(restored) == _identity(record)
    assert json.dumps(
        json.loads(serialized),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) == serialized
    assert record.schema_version == TOM_SCHEMA_VERSION == "2.0.0"


@pytest.mark.parametrize(
    ("factory", "field", "tampered_value"),
    [
        (_evidence, "summary", "different observed evidence"),
        (_distribution, "probabilities", (0.7, 0.3)),
        (_partner_state, "state_version", 2),
        (_recursive_state, "permitted_external_sources", ("PublicFeed",)),
        (_belief_update, "updater_version", "bayes-2"),
        (_decision_trace, "advisor_version", "advisor-2"),
    ],
)
def test_content_tampering_changes_scientific_identity(
    factory: Callable[[], BaseModel], field: str, tampered_value: Any
) -> None:
    original = factory()
    payload = original.model_dump()
    payload[field] = tampered_value
    tampered = type(original)(**payload)

    assert tampered.canonical_json() != original.canonical_json()
    assert _identity(tampered) != _identity(original)


@pytest.mark.parametrize(
    "factory",
    [_evidence, _distribution, _partner_state, _recursive_state,
     _belief_update, _decision_trace],
)
def test_public_records_are_frozen_and_forbid_extra_fields(
    factory: Callable[[], BaseModel],
) -> None:
    record = factory()
    with pytest.raises(ValidationError, match="frozen"):
        record.schema_version = "tampered"

    payload = record.model_dump()
    payload["future_field"] = "not in schema 2.0"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(record)(**payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"probabilities": (True, 0.0)}, "real number"),
        ({"probabilities": (float("nan"), 0.0)}, "finite"),
        ({"probabilities": (float("inf"), 0.0)}, "finite"),
        ({"probabilities": (float("-inf"), 0.0)}, "finite"),
        ({"probabilities": (-0.1, 1.1)}, "between zero and one"),
        ({"probabilities": (0.2, 0.2)}, "sum to one"),
        ({"probabilities": (0.2, 0.3, 0.5)}, "aligned"),
        ({"categories": ("unknown", "unknown")}, "duplicates"),
        ({"categories": ("unknown", "skeptical")}, "lexicographic"),
        ({"categories": ("skeptical", "other")}, "lexicographic"),
        ({"unknown_category": "other"}, "must be represented"),
        ({"categories": ("Unknown", "skeptical")}, "lowercase"),
    ],
)
def test_distribution_rejects_invalid_numeric_and_category_spaces(
    mutation: dict[str, Any], match: str
) -> None:
    values: dict[str, Any] = {
        "target": "policy_type",
        "categories": ("skeptical", "unknown"),
        "probabilities": (0.6, 0.4),
        "epistemic_status": EpistemicStatus.PRIOR,
        "ground_truth_kind": GroundTruthKind.OBJECTIVE,
    }
    values.update(mutation)

    with pytest.raises(ValidationError, match=match):
        BeliefDistribution(**values)


def test_distribution_requires_finite_space_and_unknown_bucket() -> None:
    with pytest.raises(ValidationError, match="at least 2 items"):
        _distribution(categories=("unknown",), probabilities=(1.0,))

    with pytest.raises(ValidationError, match="must be represented"):
        _distribution(
            categories=("skeptical", "unknown"),
            probabilities=(0.6, 0.4),
            unknown_category="other",
        )


def test_distribution_entropy_and_named_access_are_exact() -> None:
    uniform = _distribution(
        categories=("default", "skeptical", "unknown"),
        probabilities=(1 / 3, 1 / 3, 1 / 3),
    )
    certain = _distribution(probabilities=(1.0, 0.0))

    assert uniform.entropy == pytest.approx(math.log(3))
    assert certain.entropy == 0.0
    assert uniform.probability("skeptical") == pytest.approx(1 / 3)
    with pytest.raises(KeyError, match="absent"):
        uniform.probability("absent")


def test_evidence_identity_covers_provenance_and_visibility() -> None:
    public = _evidence()
    different_event = _evidence(event_id="event-2")
    explicit = _evidence(
        visibility=EvidenceVisibility.EXPLICIT,
        visible_to=("Buyer", "Seller"),
    )

    assert public.source_text_hash == _TEXT_HASH
    assert public.source_span == (3, 12)
    assert public.is_visible_to("UninvolvedObserver")
    assert explicit.is_visible_to("Buyer")
    assert not explicit.is_visible_to("UninvolvedObserver")
    assert len({public.evidence_id, different_event.evidence_id,
                explicit.evidence_id}) == 3


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"reliability": True}, "real number"),
        ({"reliability": float("nan")}, "finite"),
        ({"reliability": float("inf")}, "finite"),
        ({"reliability": -0.1}, "between zero and one"),
        ({"reliability": 1.1}, "between zero and one"),
        ({"turn": True}, "valid integer"),
        ({"turn": -1}, "greater than or equal to 0"),
        ({"features": (("score", float("nan")),)}, "finite"),
        ({"features": (("score", float("inf")),)}, "finite"),
        ({"features": (("z", 1), ("a", 2))}, "lexicographic"),
        ({"features": (("offer", 1), ("offer", 2))}, "duplicates"),
        ({"source_text_hash": "sha256:not-a-digest"}, "sha256 digest"),
        ({"source_span": (12, 3)}, "non-empty forward span"),
        ({"source_span": (3, 3)}, "non-empty forward span"),
        ({"source_span": (-1, 3)}, "non-empty forward span"),
        ({"source_span": (3, 12), "source_text_hash": None},
         "requires source_text_hash"),
        ({"parent_evidence_ids": ("z", "a")}, "lexicographic"),
        ({"parent_evidence_ids": ("evidence-1", "evidence-1")},
         "duplicates"),
    ],
)
def test_evidence_rejects_invalid_provenance_and_values(
    mutation: dict[str, Any], match: str
) -> None:
    payload = _evidence().model_dump()
    payload.update(mutation)

    with pytest.raises(ValidationError, match=match):
        Evidence(**payload)


@pytest.mark.parametrize(
    ("visibility", "visible_to", "match"),
    [
        (EvidenceVisibility.PUBLIC, ("Seller",), "must not enumerate"),
        (EvidenceVisibility.EXPLICIT, (), "must enumerate"),
        (EvidenceVisibility.EXPLICIT, ("Buyer",), "observer must be"),
        (EvidenceVisibility.EXPLICIT, ("Seller", "Buyer"), "lexicographic"),
        (EvidenceVisibility.EXPLICIT, ("Seller", "Seller"), "duplicates"),
        (EvidenceVisibility.ADJUDICATOR_ONLY, (), "must enumerate"),
    ],
)
def test_evidence_visibility_contracts(
    visibility: EvidenceVisibility, visible_to: tuple[str, ...], match: str
) -> None:
    payload = _evidence().model_dump()
    payload.update({"visibility": visibility, "visible_to": visible_to})

    with pytest.raises(ValidationError, match=match):
        Evidence(**payload)


def test_adjudicator_evidence_is_never_actor_visible() -> None:
    adjudicator = _evidence(
        visibility=EvidenceVisibility.ADJUDICATOR_ONLY,
        visible_to=("Seller",),
    )

    assert not adjudicator.is_visible_to("Seller")
    assert not adjudicator.is_visible_to("Buyer")


def test_partner_state_hash_covers_versions_and_lineage() -> None:
    state = _partner_state()
    version_payload = state.model_dump()
    version_payload["state_version"] = 2
    version_two = PartnerBeliefState(**version_payload)
    lineage_payload = state.model_dump()
    lineage_payload["evidence_ids"] = ("evidence-new",)
    different_lineage = PartnerBeliefState(**lineage_payload)

    assert state.state_hash == _partner_state().state_hash
    assert version_two.state_hash != state.state_hash
    assert different_lineage.state_hash != state.state_hash


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"counterpart_id": "Seller"}, "must be distinct"),
        ({"state_version": True}, "valid integer"),
        ({"state_version": -1}, "greater than or equal to 0"),
        ({"goal_beliefs": ()}, "at least 1 item"),
        ({"constraint_beliefs": ()}, "at least 1 item"),
        ({"fact_beliefs": ()}, "at least 1 item"),
        ({"evidence_ids": ("z", "a")}, "lexicographic"),
        ({"evidence_ids": ("same", "same")}, "duplicates"),
        ({"update_ids": ("z", "a")}, "lexicographic"),
        ({"update_ids": ("same", "same")}, "duplicates"),
    ],
)
def test_partner_state_rejects_incomplete_or_unstable_state(
    mutation: dict[str, Any], match: str
) -> None:
    payload = _partner_state().model_dump()
    payload.update(mutation)

    with pytest.raises(ValidationError, match=match):
        PartnerBeliefState(**payload)


def test_partner_state_rejects_duplicate_semantic_targets() -> None:
    state = _partner_state()
    payload = state.model_dump()
    payload["trustworthiness"] = _distribution(
        state.policy_type.target,
        state.policy_type.categories,
        state.policy_type.probabilities,
    )

    with pytest.raises(ValidationError, match="targets must be unique"):
        PartnerBeliefState(**payload)


@pytest.mark.parametrize("depth", [1, 2])
def test_recursive_state_accepts_only_declared_level_paths(depth: int) -> None:
    recursive = _recursive_state(depth)

    assert recursive.depth == depth
    assert len(recursive.information_path) == depth + 1
    assert recursive.root_state_hash == _partner_state().state_hash


@pytest.mark.parametrize(
    ("depth", "path", "match"),
    [
        (0, ("Seller",), "Input should be 1 or 2"),
        (3, ("Seller", "Buyer", "Seller", "Buyer"),
         "Input should be 1 or 2"),
        (True, ("Seller", "Buyer"), "integer one or two"),
        (1, ("Seller",), "does not match"),
        (1, ("Seller", "Buyer", "Seller"), "does not match"),
        (2, ("Seller", "Buyer"), "does not match"),
        (2, ("Seller", "Buyer", "Buyer"), "does not match"),
    ],
)
def test_recursive_state_rejects_unsupported_depth_and_path(
    depth: Any, path: tuple[str, ...], match: str
) -> None:
    state = _partner_state()
    with pytest.raises(ValidationError, match=match):
        RecursiveBeliefState(
            depth=depth,
            root_observer_id="Seller",
            counterpart_id="Buyer",
            root_state_hash=state.state_hash,
            information_path=path,
            target_belief=state.fact_beliefs[0],
            evidence=(_evidence(),),
        )


def test_recursive_state_rejects_private_and_adjudicator_knowledge() -> None:
    state = _partner_state()
    seller_private = _evidence(
        visibility=EvidenceVisibility.EXPLICIT,
        visible_to=("Seller",),
    )
    adjudicator = _evidence(
        visibility=EvidenceVisibility.ADJUDICATOR_ONLY,
        visible_to=("Seller",),
    )

    for evidence, match in (
        (seller_private, "unavailable along"),
        (adjudicator, "cannot enter ToM"),
    ):
        with pytest.raises(ValidationError, match=match):
            RecursiveBeliefState(
                depth=1,
                root_observer_id="Seller",
                counterpart_id="Buyer",
                root_state_hash=state.state_hash,
                information_path=("Seller", "Buyer"),
                target_belief=state.fact_beliefs[0],
                evidence=(evidence,),
            )


def test_recursive_state_accepts_shared_private_evidence() -> None:
    state = _partner_state()
    shared = _evidence(
        visibility=EvidenceVisibility.EXPLICIT,
        visible_to=("Buyer", "Seller"),
    )
    recursive = RecursiveBeliefState(
        depth=1,
        root_observer_id="Seller",
        counterpart_id="Buyer",
        root_state_hash=state.state_hash,
        information_path=("Seller", "Buyer"),
        target_belief=state.fact_beliefs[0],
        evidence=(shared,),
    )

    assert recursive.evidence == (shared,)


def test_recursive_state_checks_sources_and_external_source_declarations() -> None:
    state = _partner_state()
    public_feed = _evidence(source_actor_id="PublicFeed")
    allowed = RecursiveBeliefState(
        depth=1,
        root_observer_id="Seller",
        counterpart_id="Buyer",
        root_state_hash=state.state_hash,
        information_path=("Seller", "Buyer"),
        target_belief=state.fact_beliefs[0],
        evidence=(public_feed,),
        permitted_external_sources=("PublicFeed",),
    )
    assert allowed.permitted_external_sources == ("PublicFeed",)

    with pytest.raises(ValidationError, match="source is outside"):
        RecursiveBeliefState(
            depth=1,
            root_observer_id="Seller",
            counterpart_id="Buyer",
            root_state_hash=state.state_hash,
            information_path=("Seller", "Buyer"),
            target_belief=state.fact_beliefs[0],
            evidence=(public_feed,),
        )

    with pytest.raises(ValidationError, match="must be external"):
        RecursiveBeliefState(
            depth=1,
            root_observer_id="Seller",
            counterpart_id="Buyer",
            root_state_hash=state.state_hash,
            information_path=("Seller", "Buyer"),
            target_belief=state.fact_beliefs[0],
            evidence=(_evidence(),),
            permitted_external_sources=("Buyer",),
        )


def test_recursive_state_rejects_missing_duplicate_and_unordered_evidence() -> None:
    state = _partner_state()
    first = _evidence(event_id="event-1")
    second = _evidence(event_id="event-2")
    ordered = tuple(sorted((first, second), key=lambda item: item.evidence_id))

    with pytest.raises(ValidationError, match="at least 1 item"):
        RecursiveBeliefState(
            depth=1,
            root_observer_id="Seller",
            counterpart_id="Buyer",
            root_state_hash=state.state_hash,
            information_path=("Seller", "Buyer"),
            target_belief=state.fact_beliefs[0],
            evidence=(),
        )
    with pytest.raises(ValidationError, match="duplicates"):
        RecursiveBeliefState(
            depth=1,
            root_observer_id="Seller",
            counterpart_id="Buyer",
            root_state_hash=state.state_hash,
            information_path=("Seller", "Buyer"),
            target_belief=state.fact_beliefs[0],
            evidence=(first, first),
        )
    with pytest.raises(ValidationError, match="lexicographic"):
        RecursiveBeliefState(
            depth=1,
            root_observer_id="Seller",
            counterpart_id="Buyer",
            root_state_hash=state.state_hash,
            information_path=("Seller", "Buyer"),
            target_belief=state.fact_beliefs[0],
            evidence=tuple(reversed(ordered)),
        )


def test_belief_update_links_prior_evidence_posterior_and_entropy() -> None:
    update = _belief_update()

    assert update.prior_state_hash == update.prior.state_hash
    assert update.posterior_state_hash == update.posterior.state_hash
    assert update.evidence_ids == tuple(
        evidence.evidence_id for evidence in update.evidence
    )
    assert update.entropy_change == pytest.approx(
        update.posterior.entropy - update.prior.entropy
    )
    assert update.entropy_change < 0.0
    assert update.update_id == _belief_update().update_id


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"posterior": _distribution("other_target")}, "targets must match"),
        ({"posterior": _distribution(
            categories=("default", "skeptical", "unknown"),
            probabilities=(0.2, 0.7, 0.1),
            status=EpistemicStatus.UPDATED,
        )}, "categories must match"),
        ({"posterior": _distribution(
            probabilities=(0.8, 0.2), status=EpistemicStatus.UPDATED,
            ground_truth=GroundTruthKind.INFERRED,
        )}, "ground-truth semantics"),
        ({"likelihoods": (0.8,)}, "align with prior"),
        ({"likelihoods": (True, 0.2)}, "real number"),
        ({"likelihoods": (-0.1, 0.2)}, "non-negative"),
        ({"likelihoods": (float("nan"), 0.2)}, "finite"),
        ({"likelihoods": (float("inf"), 0.2)}, "finite"),
        ({"warnings": ("z warning", "a warning")}, "lexicographic"),
        ({"warnings": ("same warning", "same warning")}, "duplicates"),
    ],
)
def test_belief_update_rejects_inconsistent_linkage(
    mutation: dict[str, Any], match: str
) -> None:
    payload = _belief_update().model_dump()
    payload.update(mutation)

    with pytest.raises(ValidationError, match=match):
        BeliefUpdate(**payload)


def test_belief_update_rejects_duplicate_and_unordered_evidence() -> None:
    update = _belief_update()
    first = _evidence(event_id="event-1")
    second = _evidence(event_id="event-2")
    ordered = tuple(sorted((first, second), key=lambda item: item.evidence_id))

    duplicate = update.model_dump()
    duplicate["evidence"] = (first, first)
    with pytest.raises(ValidationError, match="duplicates"):
        BeliefUpdate(**duplicate)

    unordered = update.model_dump()
    unordered["evidence"] = tuple(reversed(ordered))
    with pytest.raises(ValidationError, match="lexicographic"):
        BeliefUpdate(**unordered)


def test_frozen_belief_update_may_record_no_new_evidence() -> None:
    prior = _distribution()
    frozen = BeliefUpdate(
        prior=prior,
        evidence=(),
        likelihoods=(1.0, 1.0),
        posterior=_distribution(status=EpistemicStatus.FROZEN),
        method=UpdateMethod.FROZEN_PRIOR,
        updater_version="frozen-1",
        observation_model_version="none-1",
    )

    assert frozen.evidence_ids == ()
    assert frozen.entropy_change == 0.0


def test_decision_trace_aligns_actions_utilities_predictions_and_binding() -> None:
    trace = _decision_trace()
    linked = trace.link_action("call-8")

    assert trace.chosen_action == trace.legal_actions[1]
    assert len(trace.expected_utilities) == len(trace.legal_actions)
    assert len(trace.conditional_predictions) == len(trace.legal_actions)
    assert not trace.action_linked
    assert linked.action_linked
    assert linked.chosen_action_call_id == "call-8"
    assert linked.trace_id != trace.trace_id
    assert trace.chosen_action_call_id is None
    with pytest.raises(ValueError, match="already linked"):
        linked.link_action("call-9")
    with pytest.raises(ValidationError):
        trace.link_action(" invalid-call ")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"counterpart_id": "Seller"}, "must be distinct"),
        ({"turn": True}, "valid integer"),
        ({"belief_state_hash": "not-a-hash"}, "sha256 digest"),
        ({"belief_update_ids": ()}, "at least 1 item"),
        ({"belief_update_ids": ("z", "a")}, "lexicographic"),
        ({"belief_update_ids": ("same", "same")}, "duplicates"),
        ({"legal_actions": ("counter", "accept")}, "lexicographic"),
        ({"legal_actions": ("accept", "accept")}, "duplicates"),
        ({"chosen_action": "reject"}, "must be legal"),
        ({"expected_utilities": (7.0,)}, "align with legal_actions"),
        ({"expected_utilities": (True, 9.0)}, "real number"),
        ({"expected_utilities": (float("nan"), 9.0)}, "finite"),
        ({"expected_utilities": (float("inf"), 9.0)}, "finite"),
        ({"conditional_predictions": (_distribution(
            "next_action",
            ("accept", "counter", "unknown"),
            (0.2, 0.6, 0.2),
        ),)}, "align with legal_actions"),
    ],
)
def test_decision_trace_rejects_invalid_action_alignment(
    mutation: dict[str, Any], match: str
) -> None:
    payload = _decision_trace().model_dump()
    payload.update(mutation)

    with pytest.raises(ValidationError, match=match):
        ToMDecisionTrace(**payload)


@pytest.mark.parametrize("mismatch", ["target", "categories"])
def test_decision_trace_rejects_incompatible_conditional_predictions(
    mismatch: str,
) -> None:
    trace = _decision_trace()
    if mismatch == "target":
        incompatible = _distribution(
            "different_action_target",
            trace.predicted_counterpart_action.categories,
            trace.predicted_counterpart_action.probabilities,
        )
    else:
        incompatible = _distribution(
            "next_action",
            ("accept", "reject", "unknown"),
            (0.2, 0.6, 0.2),
        )
    payload = trace.model_dump()
    payload["conditional_predictions"] = (
        incompatible,
        trace.predicted_counterpart_action,
    )

    with pytest.raises(ValidationError, match="share the declared action space"):
        ToMDecisionTrace(**payload)
