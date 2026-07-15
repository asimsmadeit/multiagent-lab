"""Deterministic ToM v2 engine: evidence -> beliefs -> policy (Plan 3).

The engine composes the gated v2 modules into one inspectable pipeline:

    projector (typed atoms -> Evidence)
      -> updater (prior x observation model -> BeliefUpdate)
        -> PartnerBeliefState (versioned, hash-addressed)
          -> ToMPolicyAdvisor (state x legal actions -> recommendation + trace)

Intervention conditions (Plan 3, Phase 7) are applied exactly here, at the
belief and belief-to-policy boundaries:

- ``dynamic_tom``: Bayesian updates from observed evidence.
- ``frozen_prior``: evidence recorded, never used.
- ``shuffled_tom``: a pre-recorded belief trajectory from another dyad
  replaces this dyad's updates; incoming evidence is acknowledged, not used.
- ``counterfactual_belief``: a normal update, then the policy-type posterior
  is clamped to a supplied distribution while transcript and evidence stand.
- ``oracle_tom``: the true controlled partner policy is pinned as the belief
  state; clearly an upper bound, never a deployable condition.
- ``tom_text_only``: qualitative prose renders, structured values are masked
  from the advisor, and no recommendation or decision trace exists.
- ``no_tom`` is the component's *absence* and is decided by the builder, so
  it never reaches this engine.

The engine is model-free and deterministic; nothing here calls a language
model or reads global randomness.
"""

from __future__ import annotations

from enum import Enum
import json
import re
from typing import Annotated, Any, Iterable, Mapping, Self, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from negotiation.components.tom.features import (
    AuxiliaryFeatureInput,
    DeterministicFeatureProjector,
    FeatureProjectionContext,
)
from negotiation.components.tom.policy import (
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
    BeliefUpdate,
    EpistemicStatus,
    Evidence,
    PartnerBeliefState,
    ToMDecisionTrace,
)
from negotiation.components.tom.updater import (
    BayesianUpdater,
    BeliefUpdater,
    FrozenPriorUpdater,
    ObservationModel,
)

ENGINE_VERSION = "tom-engine/1.0.0"

_Identifier = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=256)
]


class EngineCondition(str, Enum):
    """All seven preregistered conditions; ``no_tom`` is builder-level."""

    DYNAMIC_TOM = "dynamic_tom"
    FROZEN_PRIOR = "frozen_prior"
    SHUFFLED_TOM = "shuffled_tom"
    COUNTERFACTUAL_BELIEF = "counterfactual_belief"
    ORACLE_TOM = "oracle_tom"
    TOM_TEXT_ONLY = "tom_text_only"


#: How each engine condition presents at the typed policy boundary. An
#: oracle-pinned state is a frozen state there (beliefs never update), and
#: text-only never reaches the advisor at all, so its entry is inert.
POLICY_BOUNDARY_CONDITION: Mapping[EngineCondition, PolicyInterventionCondition] = {
    EngineCondition.DYNAMIC_TOM: PolicyInterventionCondition.DYNAMIC_TOM,
    EngineCondition.FROZEN_PRIOR: PolicyInterventionCondition.FROZEN_PRIOR,
    EngineCondition.SHUFFLED_TOM: PolicyInterventionCondition.SHUFFLED_TOM,
    EngineCondition.COUNTERFACTUAL_BELIEF: (
        PolicyInterventionCondition.COUNTERFACTUAL_BELIEF
    ),
    EngineCondition.ORACLE_TOM: PolicyInterventionCondition.FROZEN_PRIOR,
    EngineCondition.TOM_TEXT_ONLY: PolicyInterventionCondition.DYNAMIC_TOM,
}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class ToMEngineConfig(_FrozenModel):
    """Frozen per-trial engine configuration."""

    engine_version: str = ENGINE_VERSION
    trial_id: _Identifier
    observer_id: _Identifier
    counterpart_id: _Identifier
    condition: EngineCondition = EngineCondition.DYNAMIC_TOM
    policy_prior: BeliefDistribution
    action_mixture: dict[str, BeliefDistribution]
    reservation_prior: BeliefDistribution
    trustworthiness_prior: BeliefDistribution
    goal_priors: Annotated[
        tuple[BeliefDistribution, ...], Field(min_length=1)
    ]
    constraint_priors: Annotated[
        tuple[BeliefDistribution, ...], Field(min_length=1)
    ]
    fact_priors: Annotated[
        tuple[BeliefDistribution, ...], Field(min_length=1)
    ]
    oracle_policy_type: _Identifier | None = None
    counterfactual_clamp: BeliefDistribution | None = None
    shuffled_updates: tuple[BeliefUpdate, ...] = ()

    @model_validator(mode="after")
    def _validate_config(self) -> Self:
        if self.observer_id == self.counterpart_id:
            raise ValueError("observer and counterpart must be distinct")
        mixture_keys = set(self.action_mixture)
        policy_categories = set(self.policy_prior.categories)
        if mixture_keys != policy_categories:
            raise ValueError(
                "action_mixture must cover exactly the policy hypothesis "
                "space"
            )
        action_shapes = {
            (dist.target, dist.categories)
            for dist in self.action_mixture.values()
        }
        if len(action_shapes) != 1:
            raise ValueError(
                "action_mixture rows must share one target and category set"
            )
        if self.condition is EngineCondition.ORACLE_TOM:
            if self.oracle_policy_type is None:
                raise ValueError("oracle_tom requires oracle_policy_type")
            if self.oracle_policy_type not in policy_categories:
                raise ValueError(
                    "oracle_policy_type must be a policy hypothesis"
                )
        elif self.oracle_policy_type is not None:
            raise ValueError("oracle_policy_type is only valid for oracle_tom")
        if self.condition is EngineCondition.COUNTERFACTUAL_BELIEF:
            if self.counterfactual_clamp is None:
                raise ValueError(
                    "counterfactual_belief requires counterfactual_clamp"
                )
            clamp = self.counterfactual_clamp
            if (
                clamp.target != self.policy_prior.target
                or clamp.categories != self.policy_prior.categories
            ):
                raise ValueError(
                    "counterfactual_clamp must match the policy hypothesis "
                    "space"
                )
        elif self.counterfactual_clamp is not None:
            raise ValueError(
                "counterfactual_clamp is only valid for counterfactual_belief"
            )
        if self.condition is EngineCondition.SHUFFLED_TOM:
            if not self.shuffled_updates:
                raise ValueError(
                    "shuffled_tom requires a pre-recorded belief trajectory"
                )
            for update in self.shuffled_updates:
                if (
                    update.posterior.target != self.policy_prior.target
                    or update.posterior.categories
                    != self.policy_prior.categories
                ):
                    raise ValueError(
                        "shuffled updates must target the policy hypothesis "
                        "space"
                    )
        elif self.shuffled_updates:
            raise ValueError(
                "shuffled_updates are only valid for shuffled_tom"
            )
        return self


class EngineDecision(_FrozenModel):
    """One rendered advisory, with its recommendation when one exists."""

    engine_version: str = ENGINE_VERSION
    condition: EngineCondition
    guidance: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    belief_state_hash: _Identifier
    state_version: Annotated[int, Field(ge=1)]
    recommendation: PolicyRecommendationResult | None

    @model_validator(mode="after")
    def _validate_decision(self) -> Self:
        if self.condition is EngineCondition.TOM_TEXT_ONLY:
            if self.recommendation is not None:
                raise ValueError(
                    "tom_text_only masks structured values from the advisor"
                )
            if re.search(r"\d", self.guidance):
                raise ValueError(
                    "tom_text_only guidance must not carry numeric belief "
                    "values"
                )
        elif self.recommendation is None:
            raise ValueError("this condition requires a recommendation")
        return self


def _renormalized(values: Sequence[float]) -> tuple[float, ...]:
    total = float(sum(values))
    if total <= 0.0:
        raise ValueError("belief mixture collapsed to zero mass")
    return tuple(float(value) / total for value in values)


def _distribution(
    template: BeliefDistribution,
    probabilities: Sequence[float],
    status: EpistemicStatus,
) -> BeliefDistribution:
    return BeliefDistribution(
        target=template.target,
        categories=template.categories,
        probabilities=_renormalized(probabilities),
        unknown_category=template.unknown_category,
        epistemic_status=status,
        ground_truth_kind=template.ground_truth_kind,
    )


def _top_category(distribution: BeliefDistribution) -> str:
    paired = sorted(
        zip(distribution.probabilities, distribution.categories),
        key=lambda pair: (-pair[0], pair[1]),
    )
    return paired[0][1]


class ToMV2Engine:
    """Mutable per-trial belief engine over immutable typed records."""

    def __init__(
        self,
        config: ToMEngineConfig,
        *,
        observation_model: ObservationModel | None = None,
        updater: BeliefUpdater | None = None,
        projector: DeterministicFeatureProjector | None = None,
        advisor: ToMPolicyAdvisor | None = None,
    ) -> None:
        if not isinstance(config, ToMEngineConfig):
            raise TypeError("config must be a ToMEngineConfig")
        needs_model = config.condition in (
            EngineCondition.DYNAMIC_TOM,
            EngineCondition.COUNTERFACTUAL_BELIEF,
            EngineCondition.TOM_TEXT_ONLY,
        )
        if needs_model and observation_model is None:
            raise ValueError(
                f"{config.condition.value} requires an observation model for "
                "the policy_type target"
            )
        if observation_model is not None:
            if observation_model.target != config.policy_prior.target:
                raise ValueError(
                    "observation model must target the policy hypothesis"
                )
            if tuple(observation_model.categories) != tuple(
                config.policy_prior.categories
            ):
                raise ValueError(
                    "observation model categories must match the policy prior"
                )
        self._config = config
        self._observation_model = observation_model
        self._updater = updater or BayesianUpdater()
        self._frozen_updater = FrozenPriorUpdater()
        self._projector = projector or DeterministicFeatureProjector()
        self._advisor = advisor or ToMPolicyAdvisor()
        self._updates: list[BeliefUpdate] = []
        self._engine_notes: list[str] = []
        self._shuffle_cursor = 0
        self._last_recommendation: PolicyRecommendationResult | None = None
        self._state_version = 1
        if config.condition is EngineCondition.ORACLE_TOM:
            oracle_probs = tuple(
                1.0 if category == config.oracle_policy_type else 0.0
                for category in config.policy_prior.categories
            )
            self._policy_belief = _distribution(
                config.policy_prior, oracle_probs, EpistemicStatus.FROZEN
            )
        else:
            self._policy_belief = config.policy_prior

    # ------------------------------------------------------------------
    # Belief state
    # ------------------------------------------------------------------

    @property
    def config(self) -> ToMEngineConfig:
        return self._config

    @property
    def condition(self) -> EngineCondition:
        return self._config.condition

    @property
    def updates(self) -> tuple[BeliefUpdate, ...]:
        return tuple(self._updates)

    @property
    def engine_notes(self) -> tuple[str, ...]:
        return tuple(self._engine_notes)

    def _expected_next_action(
        self, policy_belief: BeliefDistribution
    ) -> BeliefDistribution:
        template = next(iter(self._config.action_mixture.values()))
        mixed = [0.0] * len(template.categories)
        for category, probability in zip(
            policy_belief.categories, policy_belief.probabilities
        ):
            row = self._config.action_mixture[category]
            for index, row_probability in enumerate(row.probabilities):
                mixed[index] += probability * row_probability
        if policy_belief.epistemic_status is EpistemicStatus.FROZEN:
            status = EpistemicStatus.FROZEN
        elif self._updates:
            status = EpistemicStatus.UPDATED
        else:
            status = EpistemicStatus.PRIOR
        return _distribution(template, mixed, status)

    @property
    def state(self) -> PartnerBeliefState:
        evidence_ids = sorted(
            {
                evidence_id
                for update in self._updates
                for evidence_id in update.evidence_ids
            }
        )
        update_ids = sorted({update.update_id for update in self._updates})
        return PartnerBeliefState(
            observer_id=self._config.observer_id,
            counterpart_id=self._config.counterpart_id,
            state_version=self._state_version,
            policy_type=self._policy_belief,
            expected_next_action=self._expected_next_action(
                self._policy_belief
            ),
            reservation_value=self._config.reservation_prior,
            goal_beliefs=self._config.goal_priors,
            constraint_beliefs=self._config.constraint_priors,
            fact_beliefs=self._config.fact_priors,
            trustworthiness=self._config.trustworthiness_prior,
            evidence_ids=tuple(evidence_ids),
            update_ids=tuple(update_ids),
        )

    # ------------------------------------------------------------------
    # Evidence ingestion (belief boundary)
    # ------------------------------------------------------------------

    def ingest_evidence(self, evidence: Iterable[Evidence]) -> BeliefUpdate:
        """Apply one evidence batch under the configured condition."""
        items = tuple(evidence)
        if not items:
            raise ValueError("at least one evidence record is required")
        condition = self._config.condition
        if condition is EngineCondition.SHUFFLED_TOM:
            if self._shuffle_cursor >= len(self._config.shuffled_updates):
                raise ValueError("shuffled belief trajectory is exhausted")
            update = self._config.shuffled_updates[self._shuffle_cursor]
            self._shuffle_cursor += 1
            self._engine_notes.append(
                f"shuffled_tom ignored {len(items)} live evidence record(s)"
            )
        elif condition in (
            EngineCondition.FROZEN_PRIOR,
            EngineCondition.ORACLE_TOM,
        ):
            update = self._frozen_updater.update(
                self._policy_belief, items, self._observation_model
            )
        else:
            update = self._updater.update(
                self._policy_belief, items, self._observation_model
            )
        posterior = update.posterior
        if condition is EngineCondition.COUNTERFACTUAL_BELIEF:
            posterior = self._config.counterfactual_clamp
            self._engine_notes.append(
                "counterfactual_clamp replaced the policy posterior"
            )
        self._updates.append(update)
        self._policy_belief = posterior
        self._state_version += 1
        return update

    def ingest_action(
        self,
        action: Any,
        context: FeatureProjectionContext,
        auxiliary_features: Iterable[AuxiliaryFeatureInput] = (),
    ) -> BeliefUpdate:
        """Project one typed observed action into evidence, then ingest."""
        evidence = self._projector.project(
            action, context, auxiliary_features
        )
        return self.ingest_evidence(evidence)

    # ------------------------------------------------------------------
    # Policy boundary
    # ------------------------------------------------------------------

    def recommend(
        self,
        *,
        turn: int,
        candidates: Sequence[LegalActionCandidate],
        objective: PolicyObjective,
        constraints: ProtocolConstraints | None = None,
    ) -> EngineDecision:
        """Score legal actions against the persisted belief state.

        The advisor requires update lineage, so the first advisory is only
        available after the first evidence batch has been ingested.
        """
        state = self.state
        if self._config.condition is EngineCondition.TOM_TEXT_ONLY:
            guidance = self._qualitative_guidance(state)
            self._last_recommendation = None
            return EngineDecision(
                condition=self._config.condition,
                guidance=guidance,
                belief_state_hash=state.state_hash,
                state_version=state.state_version,
                recommendation=None,
            )
        request = PolicyRequest(
            trial_id=self._config.trial_id,
            turn=turn,
            actor_id=self._config.observer_id,
            counterpart_id=self._config.counterpart_id,
            belief_state_hash=state.state_hash,
            state_version=state.state_version,
            legal_actions=tuple(
                candidate.action for candidate in candidates
            ),
            candidates=tuple(candidates),
            objective=objective,
            intervention_condition=POLICY_BOUNDARY_CONDITION[
                self._config.condition
            ],
            constraints=constraints,
        )
        recommendation = self._advisor.recommend(state, request)
        self._last_recommendation = recommendation
        header = f"Partner-model advisory [{self._config.condition.value}]"
        if self._config.condition is EngineCondition.ORACLE_TOM:
            header += " (oracle upper bound; not a deployable condition)"
        guidance = f"{header}\n{recommendation.context_summary}"
        return EngineDecision(
            condition=self._config.condition,
            guidance=guidance,
            belief_state_hash=state.state_hash,
            state_version=state.state_version,
            recommendation=recommendation,
        )

    def _qualitative_guidance(self, state: PartnerBeliefState) -> str:
        return (
            f"Partner-model advisory [{self._config.condition.value}]\n"
            f"Partner policy leaning: {_top_category(state.policy_type)}. "
            f"Most plausible counterpart response: "
            f"{_top_category(state.expected_next_action)}. "
            "Structured belief values are withheld under this condition."
        )

    def link_action(self, call_id: str) -> ToMDecisionTrace:
        """Bind the latest recommendation's trace to the acting call."""
        if self._last_recommendation is None:
            raise ValueError(
                "no linkable recommendation exists (none issued, or the "
                "condition is tom_text_only)"
            )
        linked = link_recommendation_action(self._last_recommendation, call_id)
        self._last_recommendation = None
        return linked

    # ------------------------------------------------------------------
    # Legacy compatibility (Plan 3, Phase 9)
    # ------------------------------------------------------------------

    def legacy_tom_state(self) -> dict[str, Any]:
        """Project the v2 state into the v1 ``get_state`` key layout.

        Legacy readers use ``.get`` with defaults, so keys whose v2 analogue
        does not exist (for example linguistic deception cues) are simply
        absent rather than fabricated. Earlier datasets built from v1
        heuristic summaries are NOT equivalent to these belief trajectories.
        """
        state = self.state
        trust = state.trustworthiness
        mental_model: dict[str, Any] = {
            "counterpart_id": self._config.counterpart_id,
            "goals": {},
            "personality_traits": {},
            "deception_indicators": {},
        }
        if "trustworthy" in trust.categories:
            mental_model["trust_level"] = trust.probability("trustworthy")
        return {
            "mental_models": {self._config.counterpart_id: mental_model},
            "belief_records": {},
            "emotion_history": [],
            "baseline_patterns": {},
            "deception_cue_history": [],
            "deception_indicators": [],
            "tom_version": "v2",
            "state_hash": state.state_hash,
            "v2": {
                "engine_version": ENGINE_VERSION,
                "condition": self._config.condition.value,
                "state_version": state.state_version,
                "state": state.model_dump(mode="json"),
                "updates": [
                    update.model_dump(mode="json")
                    for update in self._updates
                ],
                "engine_notes": list(self._engine_notes),
            },
        }

    def restore(
        self,
        state_dump: Mapping[str, Any],
        update_dumps: Sequence[Mapping[str, Any]],
    ) -> None:
        """Restore belief state and update history from a v2 snapshot."""
        # JSON-mode validation: snapshots are ``model_dump(mode="json")``
        # output, so tuples arrive as arrays, which strict Python-mode
        # validation would reject.
        state = PartnerBeliefState.model_validate_json(
            json.dumps(dict(state_dump))
        )
        if state.observer_id != self._config.observer_id or (
            state.counterpart_id != self._config.counterpart_id
        ):
            raise ValueError("snapshot identities do not match this engine")
        updates = [
            BeliefUpdate.model_validate_json(json.dumps(dict(dump)))
            for dump in update_dumps
        ]
        expected_ids = tuple(
            sorted({update.update_id for update in updates})
        )
        if tuple(state.update_ids) != expected_ids:
            raise ValueError("snapshot updates do not match the state lineage")
        self._updates = updates
        self._policy_belief = state.policy_type
        self._state_version = state.state_version


__all__ = [
    "ENGINE_VERSION",
    "POLICY_BOUNDARY_CONDITION",
    "EngineCondition",
    "EngineDecision",
    "ToMEngineConfig",
    "ToMV2Engine",
]
