"""Deterministic belief-to-action policy advice for Theory of Mind v2.

This module consumes only typed belief and payoff contracts.  It performs no
language-model calls, text interpretation, sampling, clock access, or global
state mutation.  The output is an inspectable recommendation and an unlinked
``ToMDecisionTrace`` ready to bind to exactly one acting-model call.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
import math
import re
from typing import Annotated, Any, Literal, Self
import unicodedata

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    model_validator,
)

from negotiation.components.tom.schema import (
    BeliefDistribution,
    EpistemicStatus,
    PartnerBeliefState,
    ToMDecisionTrace,
)


POLICY_SCHEMA_VERSION = "tom-policy/1.0.0"
_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_.:/-]*$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SENSITIVE_TOKENS = frozenset(
    {
        "adjudicator",
        "deception",
        "deceptive",
        "oracle",
        "private",
    }
)


def _stable_identifier(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be a string identifier")
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError("value must be a stable identifier")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("identifiers must use canonical NFC Unicode")
    return value


def _stable_category(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be a string category")
    if not _CATEGORY_PATTERN.fullmatch(value):
        raise ValueError("value must be a lowercase stable category")
    return value


def _information_safe_category(value: str, *, field_name: str) -> None:
    normalized = re.sub(r"[-./:]+", "_", value)
    tokens = frozenset(normalized.split("_"))
    if tokens & _SENSITIVE_TOKENS:
        raise ValueError(
            f"{field_name} cannot imply oracle/adjudicator/private/deception data"
        )
    if {"ground", "truth"}.issubset(tokens):
        raise ValueError(f"{field_name} cannot imply ground-truth input")


def _summary_identifier(value: str) -> str:
    """Return a compact, deterministic identifier without splitting tokens."""
    if len(value) <= 16:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{value[:7]}~{digest}"


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be numeric, not boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value must be finite")
    return 0.0 if result == 0.0 else result


def _non_negative(value: Any) -> float:
    result = _finite_number(value)
    if result < 0.0:
        raise ValueError("value must be non-negative")
    return result


def _probability(value: Any) -> float:
    result = _non_negative(value)
    if result > 1.0:
        raise ValueError("probability must not exceed one")
    return result


def _sha256(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("value must be a lowercase sha256 digest")
    return value


StableIdentifier = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=256),
    AfterValidator(_stable_identifier),
]
StableCategory = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=128),
    AfterValidator(_stable_category),
]
ShortSummary = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=512),
]
FiniteNumber = Annotated[float, BeforeValidator(_finite_number)]
NonNegativeNumber = Annotated[float, BeforeValidator(_non_negative)]
Probability = Annotated[float, BeforeValidator(_probability)]
Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True),
    AfterValidator(_sha256),
]


def _canonical_unique(values: tuple[str, ...], *, field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")
    if values != tuple(sorted(values)):
        raise ValueError(
            f"{field_name} must use canonical lexicographic ordering"
        )


def _finite_result(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"computed {name} is non-finite")
    return 0.0 if value == 0.0 else value


class _CanonicalModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def content_hash(self) -> str:
        material = f"{type(self).__name__}\0{self.canonical_json()}".encode()
        return f"sha256:{hashlib.sha256(material).hexdigest()}"


class PolicyInterventionCondition(str, Enum):
    """Supported non-oracle belief conditions at the policy boundary."""

    DYNAMIC_TOM = "dynamic_tom"
    FROZEN_PRIOR = "frozen_prior"
    SHUFFLED_TOM = "shuffled_tom"
    COUNTERFACTUAL_BELIEF = "counterfactual_belief"


class ExclusionCode(str, Enum):
    """Typed reasons why a candidate is excluded by protocol constraints."""

    NOT_PERMITTED = "not_permitted"
    FORBIDDEN = "forbidden"
    NOT_REQUIRED_ACTION = "not_required_action"
    SAFETY_LIMIT = "safety_limit"
    ACCEPTANCE_MINIMUM = "acceptance_minimum"
    AGREEMENT_MINIMUM = "agreement_minimum"
    COUNTERPART_UTILITY_MINIMUM = "counterpart_utility_minimum"


class CounterpartResponseOutcome(_CanonicalModel):
    """Typed payoff row for one hypothesized counterpart response."""

    schema_version: Literal["tom-policy/1.0.0"] = POLICY_SCHEMA_VERSION
    counterpart_action: StableCategory
    response_likelihood: Probability
    acceptance_probability: Probability
    agreement_probability: Probability
    actor_utility: FiniteNumber
    counterpart_utility: FiniteNumber
    safety_cost: NonNegativeNumber = 0.0
    regret_cost: NonNegativeNumber = 0.0

    @model_validator(mode="after")
    def _validate_response(self) -> Self:
        _information_safe_category(
            self.counterpart_action, field_name="counterpart_action"
        )
        return self


class LegalActionCandidate(_CanonicalModel):
    """One legal actor move with rows aligned to counterpart hypotheses."""

    schema_version: Literal["tom-policy/1.0.0"] = POLICY_SCHEMA_VERSION
    action: StableCategory
    responses: Annotated[
        tuple[CounterpartResponseOutcome, ...], Field(min_length=2)
    ]
    contract_version: StableIdentifier

    @model_validator(mode="after")
    def _validate_candidate(self) -> Self:
        _information_safe_category(self.action, field_name="action")
        _information_safe_category(
            self.contract_version, field_name="contract_version"
        )
        response_actions = tuple(item.counterpart_action for item in self.responses)
        _canonical_unique(response_actions, field_name="responses")
        if not any(item.response_likelihood > 0.0 for item in self.responses):
            raise ValueError("candidate response likelihoods need positive support")
        return self

    @property
    def candidate_id(self) -> str:
        return f"policy_candidate_{self.content_hash().removeprefix('sha256:')}"


class PolicyObjective(_CanonicalModel):
    """Versioned weights defining the scalar recommendation objective."""

    schema_version: Literal["tom-policy/1.0.0"] = POLICY_SCHEMA_VERSION
    objective_id: StableCategory
    version: StableIdentifier
    actor_utility_weight: NonNegativeNumber = 1.0
    counterpart_utility_weight: NonNegativeNumber = 0.0
    acceptance_weight: NonNegativeNumber = 0.0
    agreement_weight: NonNegativeNumber = 0.0
    safety_cost_weight: NonNegativeNumber = 0.0
    regret_cost_weight: NonNegativeNumber = 0.0
    uncertainty_weight: NonNegativeNumber = 0.0
    actor_risk_weight: NonNegativeNumber = 0.0

    @model_validator(mode="after")
    def _validate_objective(self) -> Self:
        _information_safe_category(self.objective_id, field_name="objective_id")
        _information_safe_category(self.version, field_name="objective version")
        weights = (
            self.actor_utility_weight,
            self.counterpart_utility_weight,
            self.acceptance_weight,
            self.agreement_weight,
            self.safety_cost_weight,
            self.regret_cost_weight,
            self.uncertainty_weight,
            self.actor_risk_weight,
        )
        if not any(weight > 0.0 for weight in weights):
            raise ValueError("objective must declare at least one positive weight")
        return self

    @property
    def objective_hash(self) -> str:
        return self.content_hash()


class ProtocolConstraints(_CanonicalModel):
    """Optional typed restrictions applied after score construction."""

    schema_version: Literal["tom-policy/1.0.0"] = POLICY_SCHEMA_VERSION
    constraint_set_id: StableCategory
    version: StableIdentifier
    permitted_actions: tuple[StableCategory, ...] = ()
    forbidden_actions: tuple[StableCategory, ...] = ()
    required_action: StableCategory | None = None
    max_expected_safety_cost: NonNegativeNumber | None = None
    min_acceptance_probability: Probability | None = None
    min_agreement_probability: Probability | None = None
    min_expected_counterpart_utility: FiniteNumber | None = None

    @model_validator(mode="after")
    def _validate_constraints(self) -> Self:
        _information_safe_category(
            self.constraint_set_id, field_name="constraint_set_id"
        )
        _information_safe_category(
            self.version, field_name="constraint version"
        )
        for field_name in ("permitted_actions", "forbidden_actions"):
            values = getattr(self, field_name)
            _canonical_unique(values, field_name=field_name)
            for value in values:
                _information_safe_category(value, field_name=field_name)
        if set(self.permitted_actions) & set(self.forbidden_actions):
            raise ValueError("permitted and forbidden actions must be disjoint")
        if self.required_action is not None:
            _information_safe_category(
                self.required_action, field_name="required_action"
            )
            if self.required_action in self.forbidden_actions:
                raise ValueError("required action cannot be forbidden")
            if (
                self.permitted_actions
                and self.required_action not in self.permitted_actions
            ):
                raise ValueError("required action must be permitted")
        return self

    @property
    def constraint_hash(self) -> str:
        return self.content_hash()


class PolicyRequest(_CanonicalModel):
    """Complete typed request linking one state to one legal action space."""

    schema_version: Literal["tom-policy/1.0.0"] = POLICY_SCHEMA_VERSION
    trial_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    actor_id: StableIdentifier
    counterpart_id: StableIdentifier
    belief_state_hash: Sha256Digest
    state_version: Annotated[StrictInt, Field(gt=0)]
    legal_actions: Annotated[
        tuple[StableCategory, ...], Field(min_length=1)
    ]
    candidates: Annotated[tuple[LegalActionCandidate, ...], Field(min_length=1)]
    objective: PolicyObjective
    intervention_condition: PolicyInterventionCondition
    constraints: ProtocolConstraints | None = None

    @model_validator(mode="after")
    def _validate_request(self) -> Self:
        if self.actor_id == self.counterpart_id:
            raise ValueError("actor and counterpart must be distinct")
        _canonical_unique(self.legal_actions, field_name="legal_actions")
        for action in self.legal_actions:
            _information_safe_category(action, field_name="legal_actions")
        candidate_actions = tuple(item.action for item in self.candidates)
        _canonical_unique(candidate_actions, field_name="candidates")
        if candidate_actions != self.legal_actions:
            raise ValueError(
                "legal_actions and candidate action space must match exactly"
            )
        if self.constraints is not None:
            named_actions = (
                *self.constraints.permitted_actions,
                *self.constraints.forbidden_actions,
            )
            if self.constraints.required_action is not None:
                named_actions = (*named_actions, self.constraints.required_action)
            unknown = set(named_actions) - set(self.legal_actions)
            if unknown:
                raise ValueError(
                    "protocol constraints reference actions outside legal space"
                )
        return self

    @property
    def request_id(self) -> str:
        return f"policy_request_{self.content_hash().removeprefix('sha256:')}"


class ActionRecommendationScore(_CanonicalModel):
    """Inspectably decomposed objective score for one candidate action."""

    schema_version: Literal["tom-policy/1.0.0"] = POLICY_SCHEMA_VERSION
    action: StableCategory
    conditional_prediction: BeliefDistribution
    expected_actor_utility: FiniteNumber
    expected_counterpart_utility: FiniteNumber
    acceptance_probability: Probability
    agreement_probability: Probability
    expected_safety_cost: NonNegativeNumber
    expected_regret_cost: NonNegativeNumber
    response_entropy: NonNegativeNumber
    actor_utility_risk: NonNegativeNumber
    objective_score: FiniteNumber
    eligible: StrictBool
    exclusion_codes: tuple[ExclusionCode, ...] = ()

    @model_validator(mode="after")
    def _validate_score(self) -> Self:
        _information_safe_category(self.action, field_name="action")
        codes = tuple(item.value for item in self.exclusion_codes)
        _canonical_unique(codes, field_name="exclusion_codes")
        if self.eligible and self.exclusion_codes:
            raise ValueError("eligible action cannot carry exclusion codes")
        if not self.eligible and not self.exclusion_codes:
            raise ValueError("ineligible action requires an exclusion code")
        return self

    @property
    def score_id(self) -> str:
        return f"action_score_{self.content_hash().removeprefix('sha256:')}"


class PolicyRecommendationResult(_CanonicalModel):
    """Immutable recommendation with an initially unlinked decision trace."""

    schema_version: Literal["tom-policy/1.0.0"] = POLICY_SCHEMA_VERSION
    request_id: StableIdentifier
    belief_state_hash: Sha256Digest
    belief_update_ids: tuple[StableIdentifier, ...]
    chosen_action: StableCategory
    action_scores: Annotated[
        tuple[ActionRecommendationScore, ...], Field(min_length=1)
    ]
    context_summary: ShortSummary
    context_summary_hash: Sha256Digest
    trace: ToMDecisionTrace
    advisor_version: StableIdentifier

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        _canonical_unique(
            self.belief_update_ids, field_name="belief_update_ids"
        )
        actions = tuple(item.action for item in self.action_scores)
        _canonical_unique(actions, field_name="action_scores")
        if self.chosen_action not in actions:
            raise ValueError("chosen action must have a score")
        if not any(
            item.action == self.chosen_action and item.eligible
            for item in self.action_scores
        ):
            raise ValueError("chosen action must be eligible")
        eligible = tuple(
            item for item in self.action_scores if item.eligible
        )
        expected_chosen = min(
            eligible,
            key=lambda item: (-item.objective_score, item.action),
        )
        if self.chosen_action != expected_chosen.action:
            raise ValueError(
                "chosen action does not follow deterministic score/tie ordering"
            )
        expected_hash = (
            "sha256:"
            + hashlib.sha256(self.context_summary.encode("utf-8")).hexdigest()
        )
        if self.context_summary_hash != expected_hash:
            raise ValueError("context summary hash does not match summary")
        if self.trace.action_linked:
            raise ValueError("new policy recommendation trace must be unlinked")
        if self.trace.belief_state_hash != self.belief_state_hash:
            raise ValueError("trace references a different belief state")
        if self.trace.belief_update_ids != self.belief_update_ids:
            raise ValueError("trace references different belief updates")
        if self.trace.chosen_action != self.chosen_action:
            raise ValueError("trace chosen action does not match result")
        if self.trace.recommendation_summary_hash != self.context_summary_hash:
            raise ValueError("trace summary hash does not match result")
        if self.trace.advisor_version != self.advisor_version:
            raise ValueError("trace advisor version does not match result")
        _information_safe_category(
            self.advisor_version, field_name="advisor_version"
        )
        for item in self.action_scores:
            if (
                item.conditional_prediction.target
                != self.trace.predicted_counterpart_action.target
                or item.conditional_prediction.categories
                != self.trace.predicted_counterpart_action.categories
            ):
                raise ValueError(
                    "action score prediction space does not match trace"
                )
        if self.trace.legal_actions != tuple(item.action for item in eligible):
            raise ValueError("trace legal actions do not match eligible scores")
        if self.trace.expected_utilities != tuple(
            item.objective_score for item in eligible
        ):
            raise ValueError("trace utilities do not match objective scores")
        if self.trace.conditional_predictions != tuple(
            item.conditional_prediction for item in eligible
        ):
            raise ValueError("trace predictions do not match eligible scores")
        return self

    @property
    def recommendation_id(self) -> str:
        return (
            "policy_recommendation_"
            + self.content_hash().removeprefix("sha256:")
        )

    def link_action(self, call_id: str) -> ToMDecisionTrace:
        """Delegate one immutable action link to the existing trace contract."""
        return self.trace.link_action(call_id)


class ToMPolicyAdvisor(_CanonicalModel):
    """Pure deterministic mapping from partner beliefs to action scores."""

    version: StableIdentifier = "tom-policy-advisor-1"
    max_summary_chars: Annotated[StrictInt, Field(ge=192, le=512)] = 512
    max_summary_responses: Annotated[StrictInt, Field(ge=1, le=5)] = 3
    max_summary_alternatives: Annotated[StrictInt, Field(ge=1, le=5)] = 3
    tie_breaker: Literal["lexicographic_action"] = "lexicographic_action"

    @model_validator(mode="after")
    def _validate_advisor(self) -> Self:
        _information_safe_category(self.version, field_name="advisor version")
        return self

    def recommend(
        self,
        state: PartnerBeliefState,
        request: PolicyRequest,
    ) -> PolicyRecommendationResult:
        """Score every candidate and emit an unlinked decision trace."""
        if not isinstance(state, PartnerBeliefState):
            raise TypeError("state must be a PartnerBeliefState")
        if not isinstance(request, PolicyRequest):
            raise TypeError("request must be a PolicyRequest")
        self._validate_state_linkage(state, request)

        scores = tuple(
            self._score_candidate(
                candidate,
                posterior=state.expected_next_action,
                objective=request.objective,
                constraints=request.constraints,
            )
            for candidate in request.candidates
        )
        eligible = tuple(item for item in scores if item.eligible)
        if not eligible:
            raise ValueError("protocol constraints exclude every legal action")
        chosen = min(
            eligible,
            key=lambda item: (-item.objective_score, item.action),
        )
        summary = self._context_summary(chosen, eligible)
        summary_hash = (
            "sha256:" + hashlib.sha256(summary.encode("utf-8")).hexdigest()
        )
        trace = ToMDecisionTrace(
            trial_id=request.trial_id,
            turn=request.turn,
            actor_id=request.actor_id,
            counterpart_id=request.counterpart_id,
            belief_state_hash=state.state_hash,
            belief_update_ids=state.update_ids,
            predicted_counterpart_action=state.expected_next_action,
            legal_actions=tuple(item.action for item in eligible),
            expected_utilities=tuple(item.objective_score for item in eligible),
            conditional_predictions=tuple(
                item.conditional_prediction for item in eligible
            ),
            chosen_action=chosen.action,
            intervention_condition=request.intervention_condition.value,
            objective=request.objective.objective_id,
            advisor_version=self.version,
            recommendation_summary_hash=summary_hash,
        )
        return PolicyRecommendationResult(
            request_id=request.request_id,
            belief_state_hash=state.state_hash,
            belief_update_ids=state.update_ids,
            chosen_action=chosen.action,
            action_scores=scores,
            context_summary=summary,
            context_summary_hash=summary_hash,
            trace=trace,
            advisor_version=self.version,
        )

    @staticmethod
    def _validate_state_linkage(
        state: PartnerBeliefState,
        request: PolicyRequest,
    ) -> None:
        if state.observer_id != request.actor_id:
            raise ValueError("state observer does not match request actor")
        if state.counterpart_id != request.counterpart_id:
            raise ValueError("state counterpart does not match request")
        if state.state_hash != request.belief_state_hash:
            raise ValueError("request belief state hash does not match state")
        if state.state_version != request.state_version:
            raise ValueError("request state version does not match state")
        if state.state_version <= 0 or not state.update_ids or not state.evidence_ids:
            raise ValueError("policy advice requires a persisted updated state")
        if state.expected_next_action.target != "next_action":
            raise ValueError("state expected-next-action target is mismatched")
        usable_statuses = {
            EpistemicStatus.UPDATED,
            EpistemicStatus.FROZEN,
        }
        if (
            state.policy_type.epistemic_status not in usable_statuses
            or state.expected_next_action.epistemic_status not in usable_statuses
        ):
            raise ValueError(
                "policy advice rejects prior, oracle, or unupdated beliefs"
            )
        if (
            request.intervention_condition
            is PolicyInterventionCondition.DYNAMIC_TOM
            and state.expected_next_action.epistemic_status
            is not EpistemicStatus.UPDATED
        ):
            raise ValueError("dynamic_tom requires an updated belief state")
        if (
            request.intervention_condition
            is PolicyInterventionCondition.FROZEN_PRIOR
            and state.expected_next_action.epistemic_status
            is not EpistemicStatus.FROZEN
        ):
            raise ValueError("frozen_prior requires a frozen belief state")
        distributions = (state.policy_type, state.expected_next_action)
        for distribution in distributions:
            _information_safe_category(
                distribution.target, field_name="belief target"
            )
            for category in distribution.categories:
                _information_safe_category(
                    category, field_name="belief category"
                )
        expected_actions = state.expected_next_action.categories
        for candidate in request.candidates:
            response_actions = tuple(
                item.counterpart_action for item in candidate.responses
            )
            if response_actions != expected_actions:
                raise ValueError(
                    "candidate response rows must exactly align with "
                    "expected-next-action hypotheses"
                )

    def _score_candidate(
        self,
        candidate: LegalActionCandidate,
        *,
        posterior: BeliefDistribution,
        objective: PolicyObjective,
        constraints: ProtocolConstraints | None,
    ) -> ActionRecommendationScore:
        raw_weights = tuple(
            probability * response.response_likelihood
            for probability, response in zip(
                posterior.probabilities,
                candidate.responses,
                strict=True,
            )
        )
        total = math.fsum(raw_weights)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(
                f"candidate {candidate.action!r} has no posterior response support"
            )
        probabilities_list = [value / total for value in raw_weights]
        largest = max(
            range(len(probabilities_list)),
            key=probabilities_list.__getitem__,
        )
        probabilities_list[largest] += 1.0 - math.fsum(probabilities_list)
        probabilities = tuple(probabilities_list)
        conditional = BeliefDistribution(
            target=posterior.target,
            categories=posterior.categories,
            probabilities=probabilities,
            unknown_category=posterior.unknown_category,
            epistemic_status=posterior.epistemic_status,
            ground_truth_kind=posterior.ground_truth_kind,
        )

        def expectation(attribute: str) -> float:
            value = math.fsum(
                probability * getattr(response, attribute)
                for probability, response in zip(
                    probabilities,
                    candidate.responses,
                    strict=True,
                )
            )
            return _finite_result(value, name=attribute)

        actor_utility = expectation("actor_utility")
        counterpart_utility = expectation("counterpart_utility")
        acceptance = expectation("acceptance_probability")
        agreement = expectation("agreement_probability")
        safety = expectation("safety_cost")
        regret = expectation("regret_cost")
        entropy = -math.fsum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        )
        entropy = _finite_result(entropy, name="response entropy")
        variance = math.fsum(
            probability * (response.actor_utility - actor_utility) ** 2
            for probability, response in zip(
                probabilities,
                candidate.responses,
                strict=True,
            )
        )
        variance = _finite_result(variance, name="actor utility variance")
        risk = _finite_result(math.sqrt(variance), name="actor utility risk")
        score = (
            objective.actor_utility_weight * actor_utility
            + objective.counterpart_utility_weight * counterpart_utility
            + objective.acceptance_weight * acceptance
            + objective.agreement_weight * agreement
            - objective.safety_cost_weight * safety
            - objective.regret_cost_weight * regret
            - objective.uncertainty_weight * entropy
            - objective.actor_risk_weight * risk
        )
        score = _finite_result(score, name="objective score")
        exclusions = self._exclusions(
            candidate.action,
            acceptance=acceptance,
            agreement=agreement,
            safety=safety,
            counterpart_utility=counterpart_utility,
            constraints=constraints,
        )
        return ActionRecommendationScore(
            action=candidate.action,
            conditional_prediction=conditional,
            expected_actor_utility=actor_utility,
            expected_counterpart_utility=counterpart_utility,
            acceptance_probability=acceptance,
            agreement_probability=agreement,
            expected_safety_cost=safety,
            expected_regret_cost=regret,
            response_entropy=entropy,
            actor_utility_risk=risk,
            objective_score=score,
            eligible=not exclusions,
            exclusion_codes=exclusions,
        )

    @staticmethod
    def _exclusions(
        action: str,
        *,
        acceptance: float,
        agreement: float,
        safety: float,
        counterpart_utility: float,
        constraints: ProtocolConstraints | None,
    ) -> tuple[ExclusionCode, ...]:
        if constraints is None:
            return ()
        codes: list[ExclusionCode] = []
        if (
            constraints.permitted_actions
            and action not in constraints.permitted_actions
        ):
            codes.append(ExclusionCode.NOT_PERMITTED)
        if action in constraints.forbidden_actions:
            codes.append(ExclusionCode.FORBIDDEN)
        if (
            constraints.required_action is not None
            and action != constraints.required_action
        ):
            codes.append(ExclusionCode.NOT_REQUIRED_ACTION)
        if (
            constraints.max_expected_safety_cost is not None
            and safety > constraints.max_expected_safety_cost
        ):
            codes.append(ExclusionCode.SAFETY_LIMIT)
        if (
            constraints.min_acceptance_probability is not None
            and acceptance < constraints.min_acceptance_probability
        ):
            codes.append(ExclusionCode.ACCEPTANCE_MINIMUM)
        if (
            constraints.min_agreement_probability is not None
            and agreement < constraints.min_agreement_probability
        ):
            codes.append(ExclusionCode.AGREEMENT_MINIMUM)
        if (
            constraints.min_expected_counterpart_utility is not None
            and counterpart_utility
            < constraints.min_expected_counterpart_utility
        ):
            codes.append(ExclusionCode.COUNTERPART_UTILITY_MINIMUM)
        return tuple(sorted(codes, key=lambda item: item.value))

    def _context_summary(
        self,
        chosen: ActionRecommendationScore,
        eligible: tuple[ActionRecommendationScore, ...],
    ) -> str:
        ranked_responses = sorted(
            zip(
                chosen.conditional_prediction.categories,
                chosen.conditional_prediction.probabilities,
                strict=True,
            ),
            key=lambda item: (-item[1], item[0]),
        )[: self.max_summary_responses]
        alternatives = sorted(
            (item for item in eligible if item.action != chosen.action),
            key=lambda item: (-item.objective_score, item.action),
        )[: self.max_summary_alternatives]
        response_text = ",".join(
            f"{_summary_identifier(category)}:{probability:.6g}"
            for category, probability in ranked_responses
        )
        alternative_text = ",".join(
            f"{_summary_identifier(item.action)}:{item.objective_score:.6g}"
            for item in alternatives
        ) or "none"
        leading_response = response_text.split(",", maxsplit=1)[0]
        leading_alternative = alternative_text.split(",", maxsplit=1)[0]
        required_tokens = (
            "tom_policy",
            f"chosen={_summary_identifier(chosen.action)}",
            f"score={chosen.objective_score:.6g}",
            f"uncertainty={chosen.response_entropy:.6g}",
            f"acceptance={chosen.acceptance_probability:.6g}",
            f"response={leading_response}",
            f"alternative={leading_alternative}",
        )
        summary = " ".join(required_tokens)
        if len(summary) > self.max_summary_chars:
            raise ValueError("summary bound is too small for required tokens")
        optional_tokens = (
            f"responses={response_text}",
            f"alternatives={alternative_text}",
        )
        for token in optional_tokens:
            extended = f"{summary} {token}"
            if len(extended) <= self.max_summary_chars:
                summary = extended
        return summary


def link_recommendation_action(
    recommendation: PolicyRecommendationResult,
    call_id: str,
) -> ToMDecisionTrace:
    """Link a recommendation trace to one acting call using schema semantics."""
    if not isinstance(recommendation, PolicyRecommendationResult):
        raise TypeError("recommendation must be a PolicyRecommendationResult")
    return recommendation.link_action(call_id)


__all__ = [
    "ActionRecommendationScore",
    "CounterpartResponseOutcome",
    "ExclusionCode",
    "LegalActionCandidate",
    "POLICY_SCHEMA_VERSION",
    "PolicyInterventionCondition",
    "PolicyObjective",
    "PolicyRecommendationResult",
    "PolicyRequest",
    "ProtocolConstraints",
    "ToMPolicyAdvisor",
    "link_recommendation_action",
]
