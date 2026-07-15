"""Permanent information-flow contracts for Theory of Mind recursion."""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import pytest
from pydantic import BaseModel, ValidationError

from negotiation.components.tom.recursion import (
    RECURSION_SCHEMA_VERSION,
    AccessBasis,
    AuthorizedEvidenceUse,
    DeterministicRecursionBuilder,
    EvidenceUseKind,
    InformationAccessRecord,
    RecursiveBeliefResult,
    RecursiveEvidenceInput,
    RecursiveTargetKind,
    ScorableRecursiveTarget,
    build_recursive_level,
    build_recursive_levels,
)
from negotiation.components.tom.schema import (
    BeliefDistribution,
    EpistemicStatus,
    Evidence,
    EvidenceChannel,
    EvidenceVisibility,
    GroundTruthKind,
    PartnerBeliefState,
)


def _belief(
    target: str,
    categories: tuple[str, ...],
    probabilities: tuple[float, ...],
    *,
    ground_truth: GroundTruthKind = GroundTruthKind.OBJECTIVE,
) -> BeliefDistribution:
    return BeliefDistribution(
        target=target,
        categories=categories,
        probabilities=probabilities,
        epistemic_status=EpistemicStatus.PRIOR,
        ground_truth_kind=ground_truth,
    )


def _state(
    *,
    policy_probabilities: tuple[float, ...] = (0.4, 0.3, 0.3),
    policy_ground_truth: GroundTruthKind = GroundTruthKind.OBJECTIVE,
) -> PartnerBeliefState:
    return PartnerBeliefState(
        observer_id="root",
        counterpart_id="counterpart",
        state_version=0,
        policy_type=_belief(
            "policy_type",
            ("default", "skeptical", "unknown"),
            policy_probabilities,
            ground_truth=policy_ground_truth,
        ),
        expected_next_action=_belief(
            "next_action",
            ("accept", "counter", "unknown"),
            (0.3, 0.4, 0.3),
        ),
        reservation_value=_belief(
            "reservation_value",
            ("high", "low", "unknown"),
            (0.3, 0.4, 0.3),
        ),
        goal_beliefs=(
            _belief("goal.value", ("maximize", "unknown"), (0.5, 0.5)),
        ),
        constraint_beliefs=(
            _belief(
                "constraint.time",
                ("deadline", "unknown"),
                (0.4, 0.6),
            ),
        ),
        fact_beliefs=(
            _belief(
                "fact.quality",
                ("false", "true", "unknown"),
                (0.2, 0.4, 0.4),
            ),
        ),
        trustworthiness=_belief(
            "trustworthiness",
            ("trustworthy", "unknown"),
            (0.5, 0.5),
        ),
    )


def _access(actor_id: str, **updates: Any) -> InformationAccessRecord:
    payload: dict[str, Any] = {
        "trial_id": "trial-1",
        "actor_id": actor_id,
        "through_turn": 2,
        "public_fact_ids": ("quality",),
        "role_view_fact_ids": (),
        "public_event_ids": ("event-a", "event-b", "event-c"),
        "delivered_event_ids": (),
        "adjudicator_only_fact_ids": ("secret_truth",),
        "adjudicator_only_event_ids": ("event-ground-truth",),
        "adjudicator_actor_ids": ("adjudicator",),
    }
    payload.update(updates)
    return InformationAccessRecord(**payload)


def _access_pair(**updates: Any) -> tuple[InformationAccessRecord, ...]:
    return (_access("root", **updates), _access("counterpart", **updates))


def _evidence(
    event_id: str = "event-a",
    *,
    feature_value: str = "offer",
    source_actor_id: str = "root",
    observer_id: str = "counterpart",
    turn: int = 1,
    reliability: float = 0.9,
    channel: EvidenceChannel = EvidenceChannel.OBSERVABLE,
    visibility: EvidenceVisibility = EvidenceVisibility.PUBLIC,
    visible_to: tuple[str, ...] = (),
    summary: str | None = None,
    feature_name: str = "atomic_action_category",
) -> Evidence:
    return Evidence(
        observer_id=observer_id,
        source_actor_id=source_actor_id,
        source_event_id=event_id,
        source_call_id=f"call-{event_id}",
        turn=turn,
        features=((feature_name, feature_value),),
        channel=channel,
        visibility=visibility,
        visible_to=visible_to,
        reliability=reliability,
        extractor_version="typed-features-1",
        summary=summary,
    )


def _target(
    *,
    subject_actor_id: str = "counterpart",
    target: str = "policy_type",
    kind: RecursiveTargetKind = RecursiveTargetKind.POLICY_TYPE,
    scoring_reference_id: str = "controlled-policy",
    ground_truth: GroundTruthKind = GroundTruthKind.OBJECTIVE,
    required_fact_ids: tuple[str, ...] = (),
) -> ScorableRecursiveTarget:
    return ScorableRecursiveTarget(
        target=target,
        kind=kind,
        subject_actor_id=subject_actor_id,
        scoring_reference_id=scoring_reference_id,
        expected_ground_truth_kind=ground_truth,
        required_fact_ids=required_fact_ids,
    )


def _soft(
    evidence: Evidence | None = None,
    *,
    target: str = "policy_type",
    likelihoods: tuple[tuple[str, float], ...] = (
        ("default", 0.2),
        ("skeptical", 0.7),
        ("unknown", 0.1),
    ),
    fact_ids: tuple[str, ...] = (),
    trial_id: str = "trial-1",
    calibration_id: str = "controlled-table-1",
    update_input_version: str = "likelihood-1",
) -> RecursiveEvidenceInput:
    return RecursiveEvidenceInput.soft(
        trial_id=trial_id,
        target=target,
        evidence=evidence or _evidence(),
        likelihoods=likelihoods,
        calibration_id=calibration_id,
        update_input_version=update_input_version,
        fact_ids=fact_ids,
    )


def _hard(
    evidence: Evidence | None = None,
    *,
    target: str = "next_action",
    category: str = "accept",
    fact_ids: tuple[str, ...] = (),
) -> RecursiveEvidenceInput:
    return RecursiveEvidenceInput.hard(
        trial_id="trial-1",
        target=target,
        evidence=evidence or _evidence(reliability=1.0),
        hard_category=category,
        fact_ids=fact_ids,
    )


def _build(
    *,
    depth: int = 1,
    target: ScorableRecursiveTarget | None = None,
    inputs: Iterable[RecursiveEvidenceInput] | None = None,
    access_records: Iterable[InformationAccessRecord] | None = None,
    root_state: PartnerBeliefState | None = None,
    trial_id: str = "trial-1",
    turn: int = 2,
    external_sources: tuple[str, ...] = (),
) -> RecursiveBeliefResult:
    return DeterministicRecursionBuilder().build_level(
        trial_id=trial_id,
        turn=turn,
        root_state=root_state or _state(),
        target=target or _target(),
        access_records=(
            access_records if access_records is not None else _access_pair()
        ),
        evidence_inputs=inputs if inputs is not None else (_soft(),),
        depth=depth,
        permitted_external_source_ids=external_sources,
    )


def _assert_rejected(
    function: Callable[[], Any],
    message: str | None = None,
) -> None:
    with pytest.raises((TypeError, ValueError, ValidationError)) as exc_info:
        function()
    if message is not None:
        assert message in str(exc_info.value)


def test_level_one_path_subject_and_exact_soft_posterior() -> None:
    result = _build(depth=1)

    assert result.state.depth == 1
    assert result.state.information_path == ("root", "counterpart")
    assert result.target.subject_actor_id == "counterpart"
    assert result.state.root_state_hash == _state().state_hash
    assert result.state.target_belief.probabilities == pytest.approx(
        (0.25, 0.65625, 0.09375),
        abs=1e-15,
    )
    assert result.state.target_belief.epistemic_status is EpistemicStatus.UPDATED
    assert result.state.target_belief.unknown_category == "unknown"


def test_level_two_path_uses_independent_evidence_about_root() -> None:
    level_two_input = _soft(
        _evidence("event-b", feature_value="request_evidence"),
        likelihoods=(
            ("default", 0.6),
            ("skeptical", 0.2),
            ("unknown", 0.2),
        ),
    )
    result = _build(
        depth=2,
        target=_target(
            subject_actor_id="root",
            scoring_reference_id="root-policy-belief",
        ),
        inputs=(level_two_input,),
    )

    assert result.state.depth == 2
    assert result.state.information_path == (
        "root",
        "counterpart",
        "root",
    )
    assert result.target.subject_actor_id == "root"
    assert result.state.target_belief.probabilities == pytest.approx(
        (2 / 3, 1 / 6, 1 / 6),
        abs=1e-15,
    )


def test_default_depth_is_maximum_two() -> None:
    target = _target(
        subject_actor_id="root",
        scoring_reference_id="root-policy-belief",
    )
    result = build_recursive_level(
        trial_id="trial-1",
        turn=2,
        root_state=_state(),
        target=target,
        access_records=_access_pair(),
        evidence_inputs=(_soft(),),
    )

    assert DeterministicRecursionBuilder().default_depth == 2
    assert DeterministicRecursionBuilder().maximum_depth == 2
    assert result.state.depth == 2


@pytest.mark.parametrize("depth", [0, -1, 3, True, 1.0, "2"])
def test_depth_rejects_zero_boolean_noninteger_or_above_two(depth: Any) -> None:
    _assert_rejected(lambda: _build(depth=depth), "depth")


@pytest.mark.parametrize(
    "payload",
    [{"default_depth": 1}, {"maximum_depth": 3}, {"default_depth": True}],
)
def test_builder_configuration_cannot_expand_or_change_depth(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        DeterministicRecursionBuilder(**payload)


@pytest.mark.parametrize(
    ("depth", "subject"),
    [(1, "root"), (2, "counterpart")],
)
def test_target_subject_must_match_level_semantics(
    depth: int,
    subject: str,
) -> None:
    _assert_rejected(
        lambda: _build(depth=depth, target=_target(subject_actor_id=subject)),
        "target subject",
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _target(
            target="belief",
            kind=RecursiveTargetKind.FACT,
            required_fact_ids=("quality",),
        ),
        lambda: _target(target="policy.other"),
        lambda: _target(
            target="fact.quality",
            kind=RecursiveTargetKind.FACT,
            required_fact_ids=(),
        ),
        lambda: _target(
            target="fact.",
            kind=RecursiveTargetKind.FACT,
            required_fact_ids=("quality",),
        ),
        lambda: _target(ground_truth=GroundTruthKind.NONE),
    ],
)
def test_targets_reject_generic_mismatched_or_unscorable_variables(
    factory: Callable[[], ScorableRecursiveTarget],
) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_target_must_exist_uniquely_in_root_state() -> None:
    absent = _target(
        target="fact.absent",
        kind=RecursiveTargetKind.FACT,
        required_fact_ids=("absent",),
    )

    _assert_rejected(
        lambda: _build(target=absent),
        "absent or not uniquely scorable",
    )


def test_target_ground_truth_kind_must_match_root_state() -> None:
    _assert_rejected(
        lambda: _build(
            target=_target(ground_truth=GroundTruthKind.INFERRED)
        ),
        "ground-truth semantics",
    )


def test_belief_schema_requires_declared_unknown_bucket() -> None:
    with pytest.raises(ValidationError, match="unknown category"):
        _belief("policy_type", ("default", "skeptical"), (0.5, 0.5))


def test_hard_observable_knowledge_produces_exact_point_mass() -> None:
    hard_input = _hard()
    result = _build(
        target=_target(
            target="next_action",
            kind=RecursiveTargetKind.NEXT_ACTION,
            scoring_reference_id="scripted-next-action",
        ),
        inputs=(hard_input,),
    )

    assert result.state.target_belief.probabilities == (1.0, 0.0, 0.0)
    assert result.evidence_uses[0].evidence_input.use_kind is (
        EvidenceUseKind.HARD_KNOWLEDGE
    )
    assert hard_input.hard_category == "accept"
    assert hard_input.likelihoods == ()


@pytest.mark.parametrize(
    "evidence",
    [
        _evidence(reliability=0.9),
        _evidence(
            reliability=1.0,
            channel=EvidenceChannel.LINGUISTIC,
        ),
        _evidence(
            reliability=1.0,
            channel=EvidenceChannel.MODEL_DERIVED,
        ),
    ],
)
def test_hard_knowledge_requires_reliable_observable_evidence(
    evidence: Evidence,
) -> None:
    with pytest.raises(ValidationError):
        _hard(evidence)


def test_contradictory_hard_categories_are_rejected() -> None:
    first = _hard(_evidence("event-a", reliability=1.0))
    second = _hard(
        _evidence("event-b", reliability=1.0),
        category="counter",
    )

    _assert_rejected(
        lambda: _build(
            target=_target(
                target="next_action",
                kind=RecursiveTargetKind.NEXT_ACTION,
            ),
            inputs=(first, second),
        ),
        "contradictory",
    )


def test_hard_category_must_be_in_target_hypotheses() -> None:
    _assert_rejected(
        lambda: _build(
            target=_target(
                target="next_action",
                kind=RecursiveTargetKind.NEXT_ACTION,
            ),
            inputs=(_hard(category="reject"),),
        ),
        "outside target hypotheses",
    )


def test_hard_and_soft_modes_cannot_be_mixed_for_one_target() -> None:
    hard_input = _hard(_evidence("event-a", reliability=1.0))
    soft_input = _soft(
        _evidence("event-b"),
        target="next_action",
        likelihoods=(
            ("accept", 0.2),
            ("counter", 0.7),
            ("unknown", 0.1),
        ),
    )

    _assert_rejected(
        lambda: _build(
            target=_target(
                target="next_action",
                kind=RecursiveTargetKind.NEXT_ACTION,
            ),
            inputs=(hard_input, soft_input),
        ),
        "separate targets",
    )


def test_multiple_soft_likelihoods_have_exact_bayesian_posterior() -> None:
    first = _soft()
    second = _soft(
        _evidence("event-b"),
        likelihoods=(
            ("default", 0.5),
            ("skeptical", 0.2),
            ("unknown", 0.3),
        ),
    )
    result = _build(inputs=(first, second))

    unnormalized = (0.4 * 0.2 * 0.5, 0.3 * 0.7 * 0.2, 0.3 * 0.1 * 0.3)
    total = math.fsum(unnormalized)
    assert result.state.target_belief.probabilities == pytest.approx(
        tuple(value / total for value in unnormalized),
        abs=1e-15,
    )
    assert math.fsum(result.state.target_belief.probabilities) == 1.0


def test_zero_prior_mass_is_not_resurrected() -> None:
    root_state = _state(policy_probabilities=(0.5, 0.5, 0.0))
    result = _build(
        root_state=root_state,
        inputs=(
            _soft(
                likelihoods=(
                    ("default", 0.2),
                    ("skeptical", 0.8),
                    ("unknown", 1.0),
                )
            ),
        ),
    )

    assert result.state.target_belief.probability("unknown") == 0.0


def test_soft_likelihoods_with_no_supported_prior_mass_fail_loudly() -> None:
    root_state = _state(policy_probabilities=(1.0, 0.0, 0.0))
    input_record = _soft(
        likelihoods=(
            ("default", 0.0),
            ("skeptical", 1.0),
            ("unknown", 0.0),
        )
    )

    _assert_rejected(
        lambda: _build(root_state=root_state, inputs=(input_record,)),
        "no positive posterior mass",
    )


@pytest.mark.parametrize(
    "value",
    [True, -0.1, 1.1, math.nan, math.inf, -math.inf],
)
def test_soft_likelihoods_reject_invalid_numeric_values(value: Any) -> None:
    with pytest.raises(ValidationError):
        _soft(
            likelihoods=(
                ("default", value),
                ("skeptical", 0.7),
                ("unknown", 0.1),
            )
        )


def test_soft_likelihood_categories_require_exact_canonical_order() -> None:
    with pytest.raises(ValidationError, match="lexicographic"):
        _soft(
            likelihoods=(
                ("skeptical", 0.7),
                ("default", 0.2),
                ("unknown", 0.1),
            )
        )
    with pytest.raises(ValidationError, match="duplicates"):
        _soft(
            likelihoods=(
                ("default", 0.2),
                ("default", 0.7),
                ("unknown", 0.1),
            )
        )
    missing_unknown = _soft(
        likelihoods=(("default", 0.2), ("skeptical", 0.8))
    )
    _assert_rejected(
        lambda: _build(inputs=(missing_unknown,)),
        "exactly align",
    )


def test_soft_input_requires_calibration_and_update_versions() -> None:
    payload = _soft().model_dump()
    payload["calibration_id"] = None
    with pytest.raises(ValidationError, match="calibration"):
        RecursiveEvidenceInput(**payload)

    payload = _soft().model_dump()
    payload["update_input_version"] = None
    with pytest.raises(ValidationError, match="update versions"):
        RecursiveEvidenceInput(**payload)

    payload = _soft().model_dump()
    payload["likelihoods"] = ()
    with pytest.raises(ValidationError, match="likelihoods"):
        RecursiveEvidenceInput(**payload)


def test_calibration_versions_are_retained_and_change_identity() -> None:
    first = _soft()
    second = _soft(calibration_id="controlled-table-2")

    assert first.calibration_id == "controlled-table-1"
    assert first.update_input_version == "likelihood-1"
    assert first.input_id != second.input_id


def test_both_levels_are_built_independently_not_by_confidence_decay() -> None:
    level_one_input = _soft()
    level_two_input = _soft(
        _evidence("event-b", feature_value="root_signal"),
        likelihoods=(
            ("default", 0.6),
            ("skeptical", 0.2),
            ("unknown", 0.2),
        ),
    )
    results = build_recursive_levels(
        trial_id="trial-1",
        turn=2,
        root_state=_state(),
        level_one_target=_target(),
        level_two_target=_target(
            subject_actor_id="root",
            scoring_reference_id="root-policy-belief",
        ),
        access_records=_access_pair(),
        level_one_evidence=(level_one_input,),
        level_two_evidence=(level_two_input,),
    )

    separately_built = (
        _build(inputs=(level_one_input,)),
        _build(
            depth=2,
            target=_target(
                subject_actor_id="root",
                scoring_reference_id="root-policy-belief",
            ),
            inputs=(level_two_input,),
        ),
    )
    assert results == separately_built
    assert results[0].state.target_belief.probabilities == pytest.approx(
        (0.25, 0.65625, 0.09375),
        abs=1e-15,
    )
    assert results[1].state.target_belief.probabilities == pytest.approx(
        (2 / 3, 1 / 6, 1 / 6),
        abs=1e-15,
    )
    assert results[0].state.target_belief.probabilities != (
        results[1].state.target_belief.probabilities
    )


def test_public_event_and_fact_authorization_explains_every_hop() -> None:
    fact_target = _target(
        target="fact.quality",
        kind=RecursiveTargetKind.FACT,
        scoring_reference_id="quality-truth",
        required_fact_ids=("quality",),
    )
    fact_input = _soft(
        target="fact.quality",
        fact_ids=("quality",),
        likelihoods=(
            ("false", 0.2),
            ("true", 0.7),
            ("unknown", 0.1),
        ),
    )
    result = _build(target=fact_target, inputs=(fact_input,))
    use = result.evidence_uses[0]

    assert use.authorized_actor_ids == ("counterpart", "root")
    assert use.information_path == ("root", "counterpart")
    assert use.event_access == (
        ("counterpart", AccessBasis.PUBLIC),
        ("root", AccessBasis.PUBLIC),
    )
    assert use.fact_access == (
        ("counterpart", "quality", AccessBasis.PUBLIC),
        ("root", "quality", AccessBasis.PUBLIC),
    )
    assert use.evidence_input.source_event_id == "event-a"
    assert use.evidence_input.source_actor_id == "root"
    assert use.evidence_input.observer_id == "counterpart"
    assert use.evidence_input.turn == 1
    assert result.trial_id == "trial-1"
    assert result.turn == 2


def test_role_view_fact_is_authorized_only_when_available_at_every_hop() -> None:
    target = _target(
        target="fact.quality",
        kind=RecursiveTargetKind.FACT,
        required_fact_ids=("quality",),
    )
    input_record = _soft(
        target="fact.quality",
        fact_ids=("quality",),
        likelihoods=(
            ("false", 0.2),
            ("true", 0.7),
            ("unknown", 0.1),
        ),
    )
    both_authorized = (
        _access(
            "root",
            public_fact_ids=(),
            role_view_fact_ids=("quality",),
        ),
        _access(
            "counterpart",
            public_fact_ids=(),
            role_view_fact_ids=("quality",),
        ),
    )
    result = _build(
        target=target,
        inputs=(input_record,),
        access_records=both_authorized,
    )

    assert result.evidence_uses[0].fact_access == (
        ("counterpart", "quality", AccessBasis.ROLE_VIEW),
        ("root", "quality", AccessBasis.ROLE_VIEW),
    )

    one_sided = (
        _access("root", public_fact_ids=(), role_view_fact_ids=()),
        _access(
            "counterpart",
            public_fact_ids=(),
            role_view_fact_ids=("quality",),
        ),
    )
    _assert_rejected(
        lambda: _build(
            target=target,
            inputs=(input_record,),
            access_records=one_sided,
        ),
        "fact is unavailable",
    )


def test_delivered_event_requires_delivery_and_visibility_at_every_hop() -> None:
    event_id = "event-private"
    delivered_evidence = _evidence(
        event_id,
        visibility=EvidenceVisibility.EXPLICIT,
        visible_to=("counterpart", "root"),
    )
    delivered_input = _soft(delivered_evidence)
    both_delivered = (
        _access(
            "root",
            public_event_ids=(),
            delivered_event_ids=(event_id,),
        ),
        _access(
            "counterpart",
            public_event_ids=(),
            delivered_event_ids=(event_id,),
        ),
    )
    result = _build(
        inputs=(delivered_input,),
        access_records=both_delivered,
    )

    assert result.evidence_uses[0].event_access == (
        ("counterpart", AccessBasis.DELIVERED),
        ("root", AccessBasis.DELIVERED),
    )

    one_sided = (
        _access("root", public_event_ids=(), delivered_event_ids=()),
        _access(
            "counterpart",
            public_event_ids=(),
            delivered_event_ids=(event_id,),
        ),
    )
    _assert_rejected(
        lambda: _build(
            inputs=(delivered_input,),
            access_records=one_sided,
        ),
        "event is unavailable",
    )

    counterpart_only = _evidence(
        event_id,
        visibility=EvidenceVisibility.EXPLICIT,
        visible_to=("counterpart",),
    )
    _assert_rejected(
        lambda: _build(
            inputs=(_soft(counterpart_only),),
            access_records=both_delivered,
        ),
        "complete path",
    )


def test_adjudicator_only_event_is_rejected() -> None:
    input_record = _soft(_evidence("event-ground-truth"))

    _assert_rejected(
        lambda: _build(inputs=(input_record,)),
        "adjudicator-only event",
    )


def test_adjudicator_only_fact_is_rejected() -> None:
    target = _target(
        target="fact.quality",
        kind=RecursiveTargetKind.FACT,
        required_fact_ids=("secret_truth",),
    )
    input_record = _soft(
        target="fact.quality",
        fact_ids=("secret_truth",),
        likelihoods=(
            ("false", 0.2),
            ("true", 0.7),
            ("unknown", 0.1),
        ),
    )

    _assert_rejected(
        lambda: _build(target=target, inputs=(input_record,)),
        "adjudicator-only fact",
    )


def test_adjudicator_visibility_or_source_is_rejected() -> None:
    adjudicator_evidence = _evidence(
        visibility=EvidenceVisibility.ADJUDICATOR_ONLY,
        visible_to=("counterpart", "root"),
    )
    with pytest.raises(ValidationError, match="adjudicator-only evidence"):
        _soft(adjudicator_evidence)

    adjudicator_source = _soft(_evidence(source_actor_id="adjudicator"))
    _assert_rejected(
        lambda: _build(inputs=(adjudicator_source,)),
        "adjudicator source",
    )


@pytest.mark.parametrize(
    "feature_name",
    [
        "raw_text",
        "raw-prompt",
        "private_prompt",
        "prompt_text",
        "chain_of_thought",
        "reasoning_trace",
        "actual_deception",
        "actual-deception-score",
        "ground.truth.deception",
        "deception_ground_truth",
    ],
)
def test_private_prompt_cot_and_actual_deception_features_are_rejected(
    feature_name: str,
) -> None:
    unsafe = _evidence(
        feature_name=feature_name,
        feature_value="secret",
    )

    with pytest.raises(ValidationError, match="cannot enter recursion"):
        _soft(unsafe)


def test_prose_and_action_words_are_never_parsed_for_update_semantics() -> None:
    baseline = _build()
    prose = _evidence(
        feature_value="default",
        summary=(
            "This prose claims the actor is private, skeptical, and certain."
        ),
    )
    prose_result = _build(inputs=(_soft(prose),))

    assert prose_result.state.target_belief.probabilities == (
        baseline.state.target_belief.probabilities
    )
    assert prose_result.state.target_belief.probability("skeptical") > (
        prose_result.state.target_belief.probability("default")
    )
    assert prose_result.result_id != baseline.result_id


def test_counterpart_view_and_level_two_source_semantics_fail_closed() -> None:
    wrong_observer = _soft(_evidence(observer_id="root"))
    _assert_rejected(
        lambda: _build(inputs=(wrong_observer,)),
        "counterpart's view",
    )

    wrong_level_two_source = _soft(_evidence(source_actor_id="counterpart"))
    _assert_rejected(
        lambda: _build(
            depth=2,
            target=_target(subject_actor_id="root"),
            inputs=(wrong_level_two_source,),
        ),
        "root actor",
    )


def test_external_source_is_explicit_and_only_valid_at_level_one() -> None:
    public_feed_input = _soft(_evidence(source_actor_id="public_feed"))
    result = _build(
        inputs=(public_feed_input,),
        external_sources=("public_feed",),
    )

    assert result.state.permitted_external_sources == ("public_feed",)
    _assert_rejected(
        lambda: _build(inputs=(public_feed_input,)),
        "outside the modeled path",
    )
    _assert_rejected(
        lambda: _build(
            depth=2,
            target=_target(subject_actor_id="root"),
            inputs=(public_feed_input,),
            external_sources=("public_feed",),
        ),
        "root actor",
    )


@pytest.mark.parametrize(
    "sources",
    [
        ("adjudicator",),
        ("root",),
        ("counterpart",),
        ("public_feed", "public_feed"),
        ("z_feed", "a_feed"),
    ],
)
def test_external_sources_reject_path_adjudicator_duplicate_or_noncanonical(
    sources: tuple[str, ...],
) -> None:
    _assert_rejected(
        lambda: _build(
            inputs=(_soft(_evidence(source_actor_id=sources[0])),),
            external_sources=sources,
        )
    )


def test_cross_trial_evidence_and_access_are_rejected() -> None:
    _assert_rejected(
        lambda: _build(inputs=(_soft(trial_id="trial-2"),)),
        "crosses trial",
    )
    wrong_access = (
        _access("root", trial_id="trial-2"),
        _access("counterpart", trial_id="trial-2"),
    )
    _assert_rejected(
        lambda: _build(access_records=wrong_access),
        "crosses trial",
    )


def test_future_evidence_and_access_turn_mismatch_are_rejected() -> None:
    _assert_rejected(
        lambda: _build(inputs=(_soft(_evidence(turn=3)),)),
        "future-turn",
    )
    _assert_rejected(lambda: _build(turn=1), "turn")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("evidence_id", "evidence_fake", "evidence ID"),
        ("source_event_id", "event-other", "source event"),
        ("source_actor_id", "counterpart", "source actor"),
        ("observer_id", "root", "observer"),
        ("turn", 2, "turn"),
    ],
)
def test_copied_evidence_lineage_must_match_nested_record(
    field: str,
    value: Any,
    message: str,
) -> None:
    payload = _soft().model_dump()
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        RecursiveEvidenceInput(**payload)


def test_fact_ids_are_canonical_unique_and_authorized() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        _soft(fact_ids=("quality", "quality"))
    with pytest.raises(ValidationError, match="lexicographic"):
        _soft(fact_ids=("z_fact", "a_fact"))

    inaccessible = _soft(fact_ids=("unavailable_fact",))
    _assert_rejected(
        lambda: _build(inputs=(inaccessible,)),
        "fact is unavailable",
    )


def test_duplicate_evidence_or_source_event_ids_are_rejected() -> None:
    input_record = _soft()
    _assert_rejected(
        lambda: _build(inputs=(input_record, input_record)),
        "evidence IDs must be unique",
    )

    same_event_different_evidence = _soft(
        _evidence(feature_value="counter")
    )
    assert same_event_different_evidence.evidence_id != input_record.evidence_id
    _assert_rejected(
        lambda: _build(
            inputs=(input_record, same_event_different_evidence)
        ),
        "event IDs must be unique",
    )


def test_access_actor_set_must_be_unique_and_complete() -> None:
    _assert_rejected(
        lambda: _build(
            access_records=(_access("root"), _access("root"))
        ),
        "unique actor IDs",
    )
    _assert_rejected(
        lambda: _build(access_records=(_access("root"),)),
        "exactly root and counterpart",
    )


def test_public_and_adjudicator_universe_must_agree() -> None:
    disagreement = (
        _access("root"),
        _access(
            "counterpart",
            public_event_ids=("event-a", "event-b"),
        ),
    )

    _assert_rejected(
        lambda: _build(access_records=disagreement),
        "disagree on public/adjudicator universe",
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: InformationAccessRecord(
            trial_id="trial-1",
            actor_id="root",
            through_turn=2,
            public_fact_ids=("quality",),
            role_view_fact_ids=("quality",),
        ),
        lambda: InformationAccessRecord(
            trial_id="trial-1",
            actor_id="root",
            through_turn=2,
            public_event_ids=("event-a",),
            delivered_event_ids=("event-a",),
        ),
        lambda: _access(
            "root",
            public_event_ids=("event-a", "event-a"),
        ),
        lambda: _access(
            "root",
            public_event_ids=("event-b", "event-a"),
        ),
        lambda: _access("adjudicator"),
        lambda: _access("root", adjudicator_actor_ids=()),
        lambda: _access("root", through_turn=True),
    ],
)
def test_access_schema_rejects_overlap_noncanonical_or_adjudicator_actor(
    factory: Callable[[], InformationAccessRecord],
) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_empty_or_wrong_typed_inputs_fail_clearly() -> None:
    _assert_rejected(
        lambda: _build(inputs=()),
        "insufficient authorized evidence",
    )
    _assert_rejected(
        lambda: _build(inputs=(object(),)),
        "RecursiveEvidenceInput",
    )
    _assert_rejected(
        lambda: _build(access_records=(object(),)),
        "InformationAccessRecord",
    )


def test_equivalent_input_and_access_order_has_identical_result_identity() -> None:
    first_input = _soft()
    second_input = _soft(
        _evidence("event-b"),
        likelihoods=(
            ("default", 0.5),
            ("skeptical", 0.2),
            ("unknown", 0.3),
        ),
    )
    first = _build(inputs=(first_input, second_input))
    second = _build(
        inputs=iter((second_input, first_input)),
        access_records=iter(reversed(_access_pair())),
    )

    assert first == second
    assert first.result_id == second.result_id
    assert tuple(item.evidence_id for item in first.state.evidence) == tuple(
        sorted(item.evidence_id for item in first.state.evidence)
    )
    assert tuple(item.use_id for item in first.evidence_uses) == tuple(
        sorted(item.use_id for item in first.evidence_uses)
    )


def test_schema_records_are_hash_stable_frozen_and_round_trip() -> None:
    result = _build()
    records: tuple[BaseModel, ...] = (
        _access("root"),
        _target(),
        _soft(),
        result.evidence_uses[0],
        result,
        DeterministicRecursionBuilder(),
    )

    for record in records:
        canonical_json = record.canonical_json()  # type: ignore[attr-defined]
        restored = type(record).model_validate_json(canonical_json)
        assert restored == record
        assert restored.content_hash() == record.content_hash()  # type: ignore[attr-defined]
        with pytest.raises(ValidationError, match="frozen"):
            record.schema_version = "tampered"  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("factory", "payload"),
    [
        (
            InformationAccessRecord,
            {**_access("root").model_dump(), "unexpected": True},
        ),
        (
            ScorableRecursiveTarget,
            {**_target().model_dump(), "unexpected": True},
        ),
        (
            RecursiveEvidenceInput,
            {**_soft().model_dump(), "unexpected": True},
        ),
        (
            AuthorizedEvidenceUse,
            {
                **_build().evidence_uses[0].model_dump(),
                "unexpected": True,
            },
        ),
        (
            RecursiveBeliefResult,
            {**_build().model_dump(), "unexpected": True},
        ),
        (
            DeterministicRecursionBuilder,
            {
                **DeterministicRecursionBuilder().model_dump(),
                "unexpected": True,
            },
        ),
    ],
)
def test_recursion_schemas_forbid_extra_fields(
    factory: type[BaseModel],
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        factory(**payload)


def test_schema_and_builder_versions_are_explicit() -> None:
    result = _build()

    assert RECURSION_SCHEMA_VERSION == "tom-recursion/1.0.0"
    assert _access("root").schema_version == RECURSION_SCHEMA_VERSION
    assert _target().schema_version == RECURSION_SCHEMA_VERSION
    assert _soft().schema_version == RECURSION_SCHEMA_VERSION
    assert result.evidence_uses[0].schema_version == RECURSION_SCHEMA_VERSION
    assert result.schema_version == RECURSION_SCHEMA_VERSION
    assert result.builder_version == "deterministic-recursion-1"


def test_result_rejects_missing_or_mismatched_explanation() -> None:
    result = _build()
    payload = result.model_dump()
    payload["evidence_uses"] = ()
    with pytest.raises(ValidationError, match="authorized evidence"):
        RecursiveBeliefResult(**payload)

    payload = result.model_dump()
    payload["target"] = _target(
        target="next_action",
        kind=RecursiveTargetKind.NEXT_ACTION,
    )
    with pytest.raises(ValidationError, match="target"):
        RecursiveBeliefResult(**payload)


def test_authorization_record_exposes_exact_access_and_parent_lineage() -> None:
    result = _build()
    use = result.evidence_uses[0]
    nested = use.evidence_input.evidence

    assert use.target_spec_id == result.target.target_spec_id
    assert use.access_record_ids == result.access_record_ids
    assert use.evidence_input.evidence_id == nested.evidence_id
    assert use.evidence_input.source_event_id == nested.source_event_id
    assert use.evidence_input.source_actor_id == nested.source_actor_id
    assert use.evidence_input.observer_id == nested.observer_id
    assert use.evidence_input.turn == nested.turn
    assert result.state.evidence == (nested,)
    assert result.state.target_belief.target == result.target.target
