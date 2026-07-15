"""Permanent contracts for deterministic Theory of Mind policy advice."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable

import pytest
from pydantic import BaseModel, ValidationError

from negotiation.components.tom.policy import (
    POLICY_SCHEMA_VERSION,
    ActionRecommendationScore,
    CounterpartResponseOutcome,
    ExclusionCode,
    LegalActionCandidate,
    PolicyInterventionCondition,
    PolicyObjective,
    PolicyRecommendationResult,
    PolicyRequest,
    ProtocolConstraints,
    ToMPolicyAdvisor,
    link_recommendation_action,
)
from negotiation.components.tom.schema import (
    BeliefDistribution,
    EpistemicStatus,
    GroundTruthKind,
    PartnerBeliefState,
    ToMDecisionTrace,
)


_RESPONSE_ACTIONS = ("accept", "counter", "unknown")


def _belief(
    target: str,
    categories: tuple[str, ...],
    probabilities: tuple[float, ...],
    *,
    status: EpistemicStatus = EpistemicStatus.PRIOR,
) -> BeliefDistribution:
    return BeliefDistribution(
        target=target,
        categories=categories,
        probabilities=probabilities,
        epistemic_status=status,
        ground_truth_kind=GroundTruthKind.OBJECTIVE,
    )


def _state(
    probabilities: tuple[float, ...] = (0.5, 0.3, 0.2),
    *,
    status: EpistemicStatus = EpistemicStatus.UPDATED,
    state_version: int = 1,
    evidence_ids: tuple[str, ...] = ("evidence-a",),
    update_ids: tuple[str, ...] = ("tom-update-a",),
    observer_id: str = "seller",
    counterpart_id: str = "buyer",
    expected_target: str = "next_action",
    response_categories: tuple[str, ...] = _RESPONSE_ACTIONS,
    policy_categories: tuple[str, ...] = (
        "default",
        "skeptical",
        "unknown",
    ),
) -> PartnerBeliefState:
    return PartnerBeliefState(
        observer_id=observer_id,
        counterpart_id=counterpart_id,
        state_version=state_version,
        policy_type=_belief(
            "policy_type",
            policy_categories,
            (0.3, 0.5, 0.2),
            status=status,
        ),
        expected_next_action=_belief(
            expected_target,
            response_categories,
            probabilities,
            status=status,
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
        evidence_ids=evidence_ids,
        update_ids=update_ids,
    )


def _outcome(
    counterpart_action: str,
    *,
    likelihood: Any = 1.0,
    actor_utility: Any = 0.0,
    counterpart_utility: Any = 0.0,
    acceptance: Any = 0.0,
    agreement: Any = 0.0,
    safety: Any = 0.0,
    regret: Any = 0.0,
) -> CounterpartResponseOutcome:
    return CounterpartResponseOutcome(
        counterpart_action=counterpart_action,
        response_likelihood=likelihood,
        acceptance_probability=acceptance,
        agreement_probability=agreement,
        actor_utility=actor_utility,
        counterpart_utility=counterpart_utility,
        safety_cost=safety,
        regret_cost=regret,
    )


def _candidate(
    action: str,
    *,
    actor_utilities: tuple[Any, ...] = (0.0, 0.0, 0.0),
    counterpart_utilities: tuple[Any, ...] = (0.0, 0.0, 0.0),
    likelihoods: tuple[Any, ...] = (1.0, 1.0, 1.0),
    acceptance: tuple[Any, ...] = (1.0, 0.0, 0.0),
    agreement: tuple[Any, ...] = (1.0, 0.0, 0.0),
    safety: tuple[Any, ...] = (0.0, 0.0, 0.0),
    regret: tuple[Any, ...] = (0.0, 0.0, 0.0),
    response_actions: tuple[str, ...] = _RESPONSE_ACTIONS,
    contract_version: str = "payoff-1",
) -> LegalActionCandidate:
    return LegalActionCandidate(
        action=action,
        contract_version=contract_version,
        responses=tuple(
            _outcome(
                counterpart_action,
                likelihood=likelihoods[index],
                actor_utility=actor_utilities[index],
                counterpart_utility=counterpart_utilities[index],
                acceptance=acceptance[index],
                agreement=agreement[index],
                safety=safety[index],
                regret=regret[index],
            )
            for index, counterpart_action in enumerate(response_actions)
        ),
    )


def _conditioned_candidate(action: str = "conditioned") -> LegalActionCandidate:
    return _candidate(
        action,
        actor_utilities=(10.0, 4.0, 1.0),
        counterpart_utilities=(4.0, 3.0, 1.0),
        likelihoods=(0.2, 1.0, 0.5),
        acceptance=(1.0, 0.2, 0.1),
        agreement=(0.8, 0.1, 0.0),
        safety=(1.0, 3.0, 2.0),
        regret=(2.0, 5.0, 7.0),
    )


def _objective(**updates: Any) -> PolicyObjective:
    payload: dict[str, Any] = {
        "objective_id": "actor_value",
        "version": "objective-1",
        "actor_utility_weight": 1.0,
    }
    payload.update(updates)
    return PolicyObjective(**payload)


def _request(
    state: PartnerBeliefState,
    candidates: tuple[LegalActionCandidate, ...],
    *,
    objective: PolicyObjective | None = None,
    constraints: ProtocolConstraints | None = None,
    condition: PolicyInterventionCondition = (
        PolicyInterventionCondition.DYNAMIC_TOM
    ),
    **updates: Any,
) -> PolicyRequest:
    payload: dict[str, Any] = {
        "trial_id": "trial-1",
        "turn": 2,
        "actor_id": state.observer_id,
        "counterpart_id": state.counterpart_id,
        "belief_state_hash": state.state_hash,
        "state_version": state.state_version,
        "legal_actions": tuple(item.action for item in candidates),
        "candidates": candidates,
        "objective": objective or _objective(),
        "intervention_condition": condition,
        "constraints": constraints,
    }
    payload.update(updates)
    return PolicyRequest(**payload)


def _recommend(
    state: PartnerBeliefState,
    candidates: tuple[LegalActionCandidate, ...],
    **kwargs: Any,
) -> PolicyRecommendationResult:
    return ToMPolicyAdvisor().recommend(
        state,
        _request(state, candidates, **kwargs),
    )


def _score(
    result: PolicyRecommendationResult,
    action: str,
) -> ActionRecommendationScore:
    return next(item for item in result.action_scores if item.action == action)


def _assert_rejected(
    function: Callable[[], Any],
    message: str | None = None,
) -> None:
    with pytest.raises((TypeError, ValueError, ValidationError)) as exc_info:
        function()
    if message is not None:
        assert message in str(exc_info.value)


def test_posterior_conditioning_and_all_decomposed_metrics_are_exact() -> None:
    state = _state()
    candidate = _conditioned_candidate()
    result = _recommend(state, (candidate,))
    score = result.action_scores[0]
    probabilities = (0.2, 0.6, 0.2)
    actor_utility = 4.6
    risk = math.sqrt(
        0.2 * (10.0 - actor_utility) ** 2
        + 0.6 * (4.0 - actor_utility) ** 2
        + 0.2 * (1.0 - actor_utility) ** 2
    )

    assert score.conditional_prediction.probabilities == probabilities
    assert score.expected_actor_utility == actor_utility
    assert score.expected_counterpart_utility == 2.8
    assert score.acceptance_probability == pytest.approx(0.34)
    assert score.agreement_probability == pytest.approx(0.22)
    assert score.expected_safety_cost == pytest.approx(2.4)
    assert score.expected_regret_cost == pytest.approx(4.8)
    assert score.response_entropy == pytest.approx(
        -math.fsum(value * math.log(value) for value in probabilities)
    )
    assert score.actor_utility_risk == pytest.approx(risk)
    assert score.objective_score == actor_utility
    assert score.conditional_prediction.target == "next_action"
    assert score.conditional_prediction.categories == _RESPONSE_ACTIONS


@pytest.mark.parametrize(
    ("weight_name", "metric_name", "sign"),
    [
        ("actor_utility_weight", "expected_actor_utility", 1.0),
        (
            "counterpart_utility_weight",
            "expected_counterpart_utility",
            1.0,
        ),
        ("acceptance_weight", "acceptance_probability", 1.0),
        ("agreement_weight", "agreement_probability", 1.0),
        ("safety_cost_weight", "expected_safety_cost", -1.0),
        ("regret_cost_weight", "expected_regret_cost", -1.0),
        ("uncertainty_weight", "response_entropy", -1.0),
        ("actor_risk_weight", "actor_utility_risk", -1.0),
    ],
)
def test_each_objective_weight_has_exact_declared_effect(
    weight_name: str,
    metric_name: str,
    sign: float,
) -> None:
    weights = {
        "actor_utility_weight": 0.0,
        "counterpart_utility_weight": 0.0,
        "acceptance_weight": 0.0,
        "agreement_weight": 0.0,
        "safety_cost_weight": 0.0,
        "regret_cost_weight": 0.0,
        "uncertainty_weight": 0.0,
        "actor_risk_weight": 0.0,
    }
    weights[weight_name] = 1.0
    objective = PolicyObjective(
        objective_id=f"weight_{weight_name}",
        version="objective-1",
        **weights,
    )
    score = _recommend(
        _state(),
        (_conditioned_candidate(),),
        objective=objective,
    ).action_scores[0]

    assert score.objective_score == pytest.approx(
        sign * getattr(score, metric_name)
    )


def test_combined_objective_score_is_exact_linear_composition() -> None:
    objective = PolicyObjective(
        objective_id="combined_value",
        version="objective-1",
        actor_utility_weight=2.0,
        counterpart_utility_weight=0.5,
        acceptance_weight=3.0,
        agreement_weight=4.0,
        safety_cost_weight=1.5,
        regret_cost_weight=0.25,
        uncertainty_weight=0.1,
        actor_risk_weight=0.2,
    )
    score = _recommend(
        _state(),
        (_conditioned_candidate(),),
        objective=objective,
    ).action_scores[0]
    expected = (
        2.0 * score.expected_actor_utility
        + 0.5 * score.expected_counterpart_utility
        + 3.0 * score.acceptance_probability
        + 4.0 * score.agreement_probability
        - 1.5 * score.expected_safety_cost
        - 0.25 * score.expected_regret_cost
        - 0.1 * score.response_entropy
        - 0.2 * score.actor_utility_risk
    )

    assert score.objective_score == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [True, -0.1, math.nan, math.inf, -math.inf],
)
@pytest.mark.parametrize(
    "weight_name",
    [
        "actor_utility_weight",
        "counterpart_utility_weight",
        "acceptance_weight",
        "agreement_weight",
        "safety_cost_weight",
        "regret_cost_weight",
        "uncertainty_weight",
        "actor_risk_weight",
    ],
)
def test_objective_weights_reject_boolean_negative_or_nonfinite(
    weight_name: str,
    value: Any,
) -> None:
    weights = {
        "actor_utility_weight": 0.0,
        weight_name: value,
    }
    with pytest.raises(ValidationError):
        PolicyObjective(
            objective_id="invalid_weight",
            version="objective-1",
            **weights,
        )


def test_all_zero_objective_is_invalid() -> None:
    with pytest.raises(ValidationError, match="positive weight"):
        PolicyObjective(
            objective_id="zero_weights",
            version="objective-1",
            actor_utility_weight=0.0,
        )


def test_computed_objective_overflow_fails_loudly() -> None:
    candidate = _candidate(
        "huge",
        actor_utilities=(1e308, 1e308, 1e308),
    )
    objective = _objective(actor_utility_weight=1e308)

    _assert_rejected(
        lambda: _recommend(
            _state(),
            (candidate,),
            objective=objective,
        ),
        "non-finite",
    )


def test_exact_score_tie_uses_lexicographic_action() -> None:
    alpha = _candidate("alpha", actor_utilities=(5.0, 5.0, 5.0))
    beta = _candidate("beta", actor_utilities=(5.0, 5.0, 5.0))
    result = _recommend(_state(), (alpha, beta))

    assert result.chosen_action == "alpha"
    assert tuple(item.objective_score for item in result.action_scores) == (
        5.0,
        5.0,
    )
    assert ToMPolicyAdvisor().tie_breaker == "lexicographic_action"


def test_replacement_state_flips_action_with_identical_contracts() -> None:
    safe = _candidate("safe", actor_utilities=(10.0, 0.0, 0.0))
    tough = _candidate("tough", actor_utilities=(0.0, 10.0, 0.0))
    accept_state = _state((0.7, 0.2, 0.1))
    counter_state = _state((0.2, 0.7, 0.1))
    first = _recommend(accept_state, (safe, tough))
    second = _recommend(
        counter_state,
        (safe, tough),
        condition=PolicyInterventionCondition.COUNTERFACTUAL_BELIEF,
    )

    assert first.chosen_action == "safe"
    assert second.chosen_action == "tough"
    assert {
        item.action: item.objective_score for item in first.action_scores
    } == {"safe": 7.0, "tough": 2.0}
    assert {
        item.action: item.objective_score for item in second.action_scores
    } == {"safe": 2.0, "tough": 7.0}
    assert first.request_id != second.request_id
    assert first.belief_state_hash != second.belief_state_hash


def test_safety_and_counterpart_weights_change_tradeoff_predictably() -> None:
    risky = _candidate(
        "risky",
        actor_utilities=(10.0, 10.0, 10.0),
        counterpart_utilities=(0.0, 0.0, 0.0),
        safety=(4.0, 4.0, 4.0),
    )
    safer = _candidate(
        "safer",
        actor_utilities=(8.0, 8.0, 8.0),
        counterpart_utilities=(6.0, 6.0, 6.0),
    )

    assert _recommend(_state(), (risky, safer)).chosen_action == "risky"
    assert _recommend(
        _state(),
        (risky, safer),
        objective=_objective(
            objective_id="safety_adjusted",
            safety_cost_weight=1.0,
        ),
    ).chosen_action == "safer"
    assert _recommend(
        _state(),
        (risky, safer),
        objective=_objective(
            objective_id="joint_value",
            actor_utility_weight=0.2,
            counterpart_utility_weight=1.0,
        ),
    ).chosen_action == "safer"


def test_uncertainty_and_risk_penalties_change_choice_predictably() -> None:
    certain = _candidate(
        "certain",
        actor_utilities=(6.0, 6.0, 6.0),
        likelihoods=(1.0, 0.0, 0.0),
    )
    uncertain = _candidate(
        "uncertain",
        actor_utilities=(10.0, 3.0, 3.0),
    )

    assert _recommend(_state(), (certain, uncertain)).chosen_action == (
        "uncertain"
    )
    assert _recommend(
        _state(),
        (certain, uncertain),
        objective=_objective(
            objective_id="risk_adjusted",
            actor_risk_weight=2.0,
        ),
    ).chosen_action == "certain"
    assert _recommend(
        _state(),
        (certain, uncertain),
        objective=_objective(
            objective_id="uncertainty_adjusted",
            uncertainty_weight=10.0,
        ),
    ).chosen_action == "certain"


def _constraint_candidates() -> tuple[LegalActionCandidate, ...]:
    alpha = _candidate(
        "alpha",
        actor_utilities=(10.0, 10.0, 10.0),
        counterpart_utilities=(1.0, 1.0, 1.0),
        acceptance=(0.2, 0.2, 0.2),
        agreement=(0.3, 0.3, 0.3),
        safety=(2.0, 2.0, 2.0),
    )
    beta = _candidate(
        "beta",
        actor_utilities=(5.0, 5.0, 5.0),
        counterpart_utilities=(6.0, 6.0, 6.0),
        acceptance=(0.9, 0.9, 0.9),
        agreement=(0.8, 0.8, 0.8),
        safety=(0.0, 0.0, 0.0),
    )
    return alpha, beta


@pytest.mark.parametrize(
    ("constraints", "expected_code"),
    [
        (
            ProtocolConstraints(
                constraint_set_id="permitted",
                version="constraints-1",
                permitted_actions=("beta",),
            ),
            ExclusionCode.NOT_PERMITTED,
        ),
        (
            ProtocolConstraints(
                constraint_set_id="forbidden",
                version="constraints-1",
                forbidden_actions=("alpha",),
            ),
            ExclusionCode.FORBIDDEN,
        ),
        (
            ProtocolConstraints(
                constraint_set_id="required",
                version="constraints-1",
                required_action="beta",
            ),
            ExclusionCode.NOT_REQUIRED_ACTION,
        ),
        (
            ProtocolConstraints(
                constraint_set_id="safety",
                version="constraints-1",
                max_expected_safety_cost=1.0,
            ),
            ExclusionCode.SAFETY_LIMIT,
        ),
        (
            ProtocolConstraints(
                constraint_set_id="acceptance",
                version="constraints-1",
                min_acceptance_probability=0.8,
            ),
            ExclusionCode.ACCEPTANCE_MINIMUM,
        ),
        (
            ProtocolConstraints(
                constraint_set_id="agreement",
                version="constraints-1",
                min_agreement_probability=0.7,
            ),
            ExclusionCode.AGREEMENT_MINIMUM,
        ),
        (
            ProtocolConstraints(
                constraint_set_id="counterpart_value",
                version="constraints-1",
                min_expected_counterpart_utility=5.0,
            ),
            ExclusionCode.COUNTERPART_UTILITY_MINIMUM,
        ),
    ],
)
def test_every_protocol_exclusion_code_is_enforced(
    constraints: ProtocolConstraints,
    expected_code: ExclusionCode,
) -> None:
    alpha, beta = _constraint_candidates()
    result = _recommend(
        _state(),
        (alpha, beta),
        constraints=constraints,
    )

    assert result.chosen_action == "beta"
    assert _score(result, "alpha").eligible is False
    assert expected_code in _score(result, "alpha").exclusion_codes
    assert result.trace.legal_actions == ("beta",)


def test_multiple_exclusion_codes_are_canonical_and_auditable() -> None:
    alpha, beta = _constraint_candidates()
    constraints = ProtocolConstraints(
        constraint_set_id="combined_limits",
        version="constraints-1",
        forbidden_actions=("alpha",),
        max_expected_safety_cost=1.0,
        min_acceptance_probability=0.8,
    )
    result = _recommend(
        _state(),
        (alpha, beta),
        constraints=constraints,
    )

    assert _score(result, "alpha").exclusion_codes == (
        ExclusionCode.ACCEPTANCE_MINIMUM,
        ExclusionCode.FORBIDDEN,
        ExclusionCode.SAFETY_LIMIT,
    )


def test_constraints_excluding_all_actions_fail_loudly() -> None:
    alpha, beta = _constraint_candidates()
    constraints = ProtocolConstraints(
        constraint_set_id="exclude_all",
        version="constraints-1",
        forbidden_actions=("alpha", "beta"),
    )

    _assert_rejected(
        lambda: _recommend(
            _state(),
            (alpha, beta),
            constraints=constraints,
        ),
        "exclude every legal action",
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ProtocolConstraints(
            constraint_set_id="overlap",
            version="constraints-1",
            permitted_actions=("alpha",),
            forbidden_actions=("alpha",),
        ),
        lambda: ProtocolConstraints(
            constraint_set_id="required_forbidden",
            version="constraints-1",
            required_action="alpha",
            forbidden_actions=("alpha",),
        ),
        lambda: ProtocolConstraints(
            constraint_set_id="required_unpermitted",
            version="constraints-1",
            permitted_actions=("beta",),
            required_action="alpha",
        ),
        lambda: ProtocolConstraints(
            constraint_set_id="duplicate",
            version="constraints-1",
            permitted_actions=("alpha", "alpha"),
        ),
        lambda: ProtocolConstraints(
            constraint_set_id="unordered",
            version="constraints-1",
            permitted_actions=("beta", "alpha"),
        ),
    ],
)
def test_protocol_constraints_reject_inconsistent_action_sets(
    factory: Callable[[], ProtocolConstraints],
) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_constraints_cannot_reference_unknown_legal_action() -> None:
    alpha = _candidate("alpha")
    constraints = ProtocolConstraints(
        constraint_set_id="unknown_action",
        version="constraints-1",
        forbidden_actions=("beta",),
    )

    with pytest.raises(ValidationError, match="outside legal space"):
        _request(_state(), (alpha,), constraints=constraints)


@pytest.mark.parametrize(
    ("request_updates", "message"),
    [
        ({"actor_id": "other"}, "observer"),
        ({"counterpart_id": "other"}, "counterpart"),
        ({"belief_state_hash": "sha256:" + "0" * 64}, "hash"),
        ({"state_version": 2}, "version"),
    ],
)
def test_request_must_match_state_identity(
    request_updates: dict[str, Any],
    message: str,
) -> None:
    state = _state()
    candidate = _candidate("alpha")
    request = _request(state, (candidate,), **request_updates)

    _assert_rejected(
        lambda: ToMPolicyAdvisor().recommend(state, request),
        message,
    )


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (_state(evidence_ids=()), "persisted updated"),
        (_state(update_ids=()), "persisted updated"),
        (_state(status=EpistemicStatus.PRIOR), "prior, oracle, or unupdated"),
        (_state(status=EpistemicStatus.ORACLE), "prior, oracle, or unupdated"),
        (_state(expected_target="next_move"), "expected-next-action"),
    ],
)
def test_unlinked_unupdated_or_oracle_state_is_rejected(
    state: PartnerBeliefState,
    message: str,
) -> None:
    candidate = _candidate("alpha")

    _assert_rejected(
        lambda: _recommend(state, (candidate,)),
        message,
    )


def test_zero_state_version_cannot_form_a_valid_request_link() -> None:
    state = _state(state_version=0)
    candidate = _candidate("alpha")

    with pytest.raises(ValidationError):
        _request(state, (candidate,))


def test_dynamic_and_frozen_conditions_require_matching_state_status() -> None:
    candidate = _candidate("alpha")
    updated = _state()
    frozen = _state(status=EpistemicStatus.FROZEN)

    assert _recommend(updated, (candidate,)).chosen_action == "alpha"
    assert _recommend(
        frozen,
        (candidate,),
        condition=PolicyInterventionCondition.FROZEN_PRIOR,
    ).chosen_action == "alpha"
    _assert_rejected(
        lambda: _recommend(frozen, (candidate,)),
        "dynamic_tom",
    )
    _assert_rejected(
        lambda: _recommend(
            updated,
            (candidate,),
            condition=PolicyInterventionCondition.FROZEN_PRIOR,
        ),
        "frozen_prior",
    )


@pytest.mark.parametrize(
    "condition",
    [
        PolicyInterventionCondition.SHUFFLED_TOM,
        PolicyInterventionCondition.COUNTERFACTUAL_BELIEF,
    ],
)
def test_nonoracle_belief_interventions_accept_valid_replacement_state(
    condition: PolicyInterventionCondition,
) -> None:
    result = _recommend(
        _state((0.2, 0.7, 0.1)),
        (_candidate("alpha"),),
        condition=condition,
    )

    assert result.trace.intervention_condition == condition.value


@pytest.mark.parametrize("condition", ["oracle_tom", "no_tom"])
def test_oracle_and_no_tom_cannot_enter_policy_advisor(
    condition: str,
) -> None:
    with pytest.raises(ValueError):
        PolicyInterventionCondition(condition)


def test_candidate_response_rows_must_align_exactly_with_state_hypotheses() -> None:
    candidate = _candidate(
        "alpha",
        response_actions=("accept", "reject", "unknown"),
    )

    _assert_rejected(
        lambda: _recommend(_state(), (candidate,)),
        "exactly align",
    )


@pytest.mark.parametrize(
    "response_actions",
    [
        ("accept", "accept", "unknown"),
        ("counter", "accept", "unknown"),
    ],
)
def test_candidate_response_rows_are_unique_and_canonical(
    response_actions: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError):
        _candidate("alpha", response_actions=response_actions)


def test_candidate_needs_positive_posterior_response_support() -> None:
    state = _state((1.0, 0.0, 0.0))
    candidate = _candidate("counter_only", likelihoods=(0.0, 1.0, 0.0))

    _assert_rejected(
        lambda: _recommend(state, (candidate,)),
        "no posterior response support",
    )


def test_candidate_rejects_all_zero_declared_likelihoods() -> None:
    with pytest.raises(ValidationError, match="positive support"):
        _candidate("alpha", likelihoods=(0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "value",
    [True, math.nan, math.inf, -math.inf],
)
@pytest.mark.parametrize(
    "field_name",
    ["actor_utility", "counterpart_utility"],
)
def test_payoff_utilities_reject_boolean_or_nonfinite(
    field_name: str,
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        _outcome("accept", **{field_name: value})


@pytest.mark.parametrize(
    "value",
    [True, -0.1, 1.1, math.nan, math.inf, -math.inf],
)
@pytest.mark.parametrize(
    "field_name",
    ["likelihood", "acceptance", "agreement"],
)
def test_response_probabilities_reject_invalid_values(
    field_name: str,
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        _outcome("accept", **{field_name: value})


@pytest.mark.parametrize("field_name", ["safety", "regret"])
@pytest.mark.parametrize(
    "value",
    [True, -0.1, math.nan, math.inf, -math.inf],
)
def test_costs_reject_boolean_negative_or_nonfinite(
    field_name: str,
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        _outcome("accept", **{field_name: value})


def test_legal_action_candidates_must_be_nonempty_unique_and_canonical() -> None:
    state = _state()
    alpha = _candidate("alpha")
    beta = _candidate("beta")

    with pytest.raises(ValidationError):
        PolicyRequest(
            trial_id="trial-1",
            turn=2,
            actor_id="seller",
            counterpart_id="buyer",
            belief_state_hash=state.state_hash,
            state_version=1,
            legal_actions=(),
            candidates=(),
            objective=_objective(),
            intervention_condition=PolicyInterventionCondition.DYNAMIC_TOM,
        )
    with pytest.raises(ValidationError, match="lexicographic"):
        _request(state, (beta, alpha))
    with pytest.raises(ValidationError, match="duplicates"):
        _request(state, (alpha, alpha))


def test_legal_actions_and_candidate_space_must_match_exactly() -> None:
    state = _state()
    alpha = _candidate("alpha")
    payload = _request(state, (alpha,)).model_dump()
    payload["legal_actions"] = ("beta",)

    with pytest.raises(ValidationError, match="match exactly"):
        PolicyRequest(**payload)


@pytest.mark.parametrize(
    "value",
    [
        "oracle_move",
        "private_offer",
        "deception_label",
        "adjudicator_action",
        "ground_truth_move",
    ],
)
def test_action_and_response_categories_reject_sensitive_labels(
    value: str,
) -> None:
    with pytest.raises(ValidationError):
        _candidate(value)
    with pytest.raises(ValidationError):
        _outcome(value)


@pytest.mark.parametrize(
    ("factory", "value"),
    [
        (lambda value: _objective(objective_id=value), "oracle_value"),
        (lambda value: _objective(version=value), "private-v1"),
        (
            lambda value: ProtocolConstraints(
                constraint_set_id=value,
                version="constraints-1",
            ),
            "deception_label",
        ),
        (
            lambda value: ProtocolConstraints(
                constraint_set_id="rules",
                version=value,
            ),
            "ground-truth-v1",
        ),
        (
            lambda value: _candidate("alpha", contract_version=value),
            "oracle-v1",
        ),
        (lambda value: ToMPolicyAdvisor(version=value), "adjudicator-v1"),
    ],
)
def test_semantic_and_provenance_fields_reject_sensitive_names(
    factory: Callable[[str], BaseModel],
    value: str,
) -> None:
    with pytest.raises(ValidationError):
        factory(value)


def test_sensitive_policy_hypothesis_cannot_enter_advisor() -> None:
    state = _state(policy_categories=("default", "oracle", "unknown"))

    _assert_rejected(
        lambda: _recommend(state, (_candidate("alpha"),)),
        "oracle/adjudicator/private/deception",
    )


def test_summary_is_bounded_token_safe_hashed_and_structured() -> None:
    first_action = "a_" + "x" * 100
    second_action = "z_" + "y" * 100
    candidates = (
        _candidate(first_action, actor_utilities=(5.0, 5.0, 5.0)),
        _candidate(second_action, actor_utilities=(7.0, 7.0, 7.0)),
    )
    state = _state()
    result = ToMPolicyAdvisor(max_summary_chars=192).recommend(
        state,
        _request(state, candidates),
    )
    tokens = result.context_summary.split()
    keys = {token.split("=", maxsplit=1)[0] for token in tokens[1:]}

    assert len(result.context_summary) <= 192
    assert tokens[0] == "tom_policy"
    assert {
        "chosen",
        "score",
        "uncertainty",
        "acceptance",
        "response",
        "alternative",
    }.issubset(keys)
    assert not result.context_summary.endswith(("=", ",", ":"))
    assert result.context_summary_hash == (
        "sha256:"
        + hashlib.sha256(result.context_summary.encode("utf-8")).hexdigest()
    )
    assert result.trace.recommendation_summary_hash == (
        result.context_summary_hash
    )


@pytest.mark.parametrize("max_chars", [True, 191, 513])
def test_summary_bound_configuration_is_strict(max_chars: Any) -> None:
    with pytest.raises(ValidationError):
        ToMPolicyAdvisor(max_summary_chars=max_chars)


def test_safe_provenance_names_do_not_affect_scores_or_choice() -> None:
    first = _candidate(
        "alpha",
        actor_utilities=(5.0, 5.0, 5.0),
        contract_version="payoff-1",
    )
    second = _candidate(
        "alpha",
        actor_utilities=(5.0, 5.0, 5.0),
        contract_version="payoff-2",
    )
    first_result = _recommend(_state(), (first,))
    second_result = _recommend(_state(), (second,))

    assert first_result.chosen_action == second_result.chosen_action == "alpha"
    assert first_result.action_scores == second_result.action_scores
    assert first_result.request_id != second_result.request_id


def test_result_and_trace_are_exactly_aligned_and_initially_unlinked() -> None:
    alpha = _candidate("alpha", actor_utilities=(5.0, 5.0, 5.0))
    beta = _candidate("beta", actor_utilities=(7.0, 7.0, 7.0))
    state = _state()
    result = _recommend(state, (alpha, beta))
    eligible = tuple(item for item in result.action_scores if item.eligible)

    assert result.trace.action_linked is False
    assert result.trace.chosen_action_call_id is None
    assert result.trace.chosen_action == result.chosen_action
    assert result.trace.belief_state_hash == state.state_hash
    assert result.trace.belief_update_ids == state.update_ids
    assert result.trace.legal_actions == tuple(item.action for item in eligible)
    assert result.trace.expected_utilities == tuple(
        item.objective_score for item in eligible
    )
    assert result.trace.conditional_predictions == tuple(
        item.conditional_prediction for item in eligible
    )
    assert result.trace.advisor_version == result.advisor_version


def test_action_trace_links_once_immutably_via_existing_contract() -> None:
    result = _recommend(_state(), (_candidate("alpha"),))
    linked = link_recommendation_action(result, "call-action-1")

    assert linked.action_linked is True
    assert linked.chosen_action_call_id == "call-action-1"
    assert result.trace.action_linked is False
    assert linked.trace_id != result.trace.trace_id
    with pytest.raises(ValueError, match="already linked"):
        linked.link_action("call-action-2")
    with pytest.raises(ValidationError):
        result.trace.link_action(" call-action-1 ")


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda result: {
                **result.model_dump(),
                "chosen_action": "alpha",
                "trace": result.trace.model_copy(
                    update={"chosen_action": "alpha"}
                ),
            },
            "deterministic score",
        ),
        (
            lambda result: {
                **result.model_dump(),
                "context_summary_hash": "sha256:" + "0" * 64,
            },
            "summary hash",
        ),
        (
            lambda result: {
                **result.model_dump(),
                "advisor_version": "different-advisor",
            },
            "advisor version",
        ),
        (
            lambda result: {
                **result.model_dump(),
                "belief_state_hash": "sha256:" + "0" * 64,
            },
            "belief state",
        ),
    ],
)
def test_result_rejects_trace_or_score_misalignment(
    mutator: Callable[[PolicyRecommendationResult], dict[str, Any]],
    message: str,
) -> None:
    result = _recommend(
        _state(),
        (
            _candidate("alpha", actor_utilities=(5.0, 5.0, 5.0)),
            _candidate("beta", actor_utilities=(7.0, 7.0, 7.0)),
        ),
    )

    with pytest.raises(ValidationError, match=message):
        PolicyRecommendationResult(**mutator(result))


def test_result_rejects_already_linked_trace() -> None:
    result = _recommend(_state(), (_candidate("alpha"),))
    payload = result.model_dump()
    payload["trace"] = result.trace.link_action("call-action-1")

    with pytest.raises(ValidationError, match="must be unlinked"):
        PolicyRecommendationResult(**payload)


def test_repeated_advice_is_deterministic_and_hash_stable() -> None:
    candidates = (
        _candidate("alpha", actor_utilities=(5.0, 5.0, 5.0)),
        _candidate("beta", actor_utilities=(7.0, 7.0, 7.0)),
    )
    first = _recommend(_state(), candidates)
    second = _recommend(_state(), candidates)

    assert first == second
    assert first.recommendation_id == second.recommendation_id
    assert first.trace.trace_id == second.trace.trace_id
    assert first.canonical_json() == second.canonical_json()


def test_models_are_frozen_canonical_and_json_round_trip() -> None:
    result = _recommend(_state(), (_candidate("alpha"),))
    records: tuple[BaseModel, ...] = (
        _outcome("accept"),
        _candidate("alpha"),
        _objective(),
        ProtocolConstraints(
            constraint_set_id="rules",
            version="constraints-1",
        ),
        _request(_state(), (_candidate("alpha"),)),
        result.action_scores[0],
        result,
        ToMPolicyAdvisor(),
    )

    for record in records:
        canonical = record.canonical_json()  # type: ignore[attr-defined]
        restored = type(record).model_validate_json(canonical)
        assert restored == record
        assert restored.content_hash() == record.content_hash()  # type: ignore[attr-defined]
        field_name = next(iter(type(record).model_fields))
        with pytest.raises(ValidationError, match="frozen"):
            setattr(record, field_name, getattr(record, field_name))


@pytest.mark.parametrize(
    ("factory", "payload"),
    [
        (
            CounterpartResponseOutcome,
            {**_outcome("accept").model_dump(), "unexpected": True},
        ),
        (
            LegalActionCandidate,
            {**_candidate("alpha").model_dump(), "unexpected": True},
        ),
        (
            PolicyObjective,
            {**_objective().model_dump(), "unexpected": True},
        ),
        (
            ProtocolConstraints,
            {
                **ProtocolConstraints(
                    constraint_set_id="rules",
                    version="constraints-1",
                ).model_dump(),
                "unexpected": True,
            },
        ),
        (
            PolicyRequest,
            {
                **_request(_state(), (_candidate("alpha"),)).model_dump(),
                "unexpected": True,
            },
        ),
        (
            ActionRecommendationScore,
            {
                **_recommend(
                    _state(), (_candidate("alpha"),)
                ).action_scores[0].model_dump(),
                "unexpected": True,
            },
        ),
        (
            PolicyRecommendationResult,
            {
                **_recommend(
                    _state(), (_candidate("alpha"),)
                ).model_dump(),
                "unexpected": True,
            },
        ),
        (
            ToMPolicyAdvisor,
            {**ToMPolicyAdvisor().model_dump(), "unexpected": True},
        ),
    ],
)
def test_policy_models_forbid_extra_fields(
    factory: type[BaseModel],
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        factory(**payload)


def test_policy_schema_and_provenance_ids_are_explicit() -> None:
    result = _recommend(_state(), (_candidate("alpha"),))

    assert POLICY_SCHEMA_VERSION == "tom-policy/1.0.0"
    assert _outcome("accept").schema_version == POLICY_SCHEMA_VERSION
    assert _candidate("alpha").candidate_id.startswith("policy_candidate_")
    assert _objective().objective_hash.startswith("sha256:")
    assert _request(_state(), (_candidate("alpha"),)).request_id.startswith(
        "policy_request_"
    )
    assert result.action_scores[0].score_id.startswith("action_score_")
    assert result.recommendation_id.startswith("policy_recommendation_")
    assert result.advisor_version == "tom-policy-advisor-1"


def test_runtime_type_checks_reject_untyped_inputs() -> None:
    state = _state()
    request = _request(state, (_candidate("alpha"),))

    _assert_rejected(
        lambda: ToMPolicyAdvisor().recommend(object(), request),
        "PartnerBeliefState",
    )
    _assert_rejected(
        lambda: ToMPolicyAdvisor().recommend(state, object()),
        "PolicyRequest",
    )
    _assert_rejected(
        lambda: link_recommendation_action(object(), "call-action-1"),
        "PolicyRecommendationResult",
    )
