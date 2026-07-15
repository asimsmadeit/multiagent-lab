"""ToM v2 engine and component-integration tests (Plan 3, Phases 6/7/9)."""

from __future__ import annotations

import dataclasses
import re

import pytest
from pydantic import ValidationError

from negotiation.components.theory_of_mind import TheoryOfMind
from negotiation.components.tom.engine import (
    EngineCondition,
    EngineDecision,
    ToMEngineConfig,
    ToMV2Engine,
)
from negotiation.components.tom.policy import (
    CounterpartResponseOutcome,
    LegalActionCandidate,
    PolicyObjective,
)
from negotiation.components.tom.schema import (
    BeliefDistribution,
    EpistemicStatus,
    Evidence,
    EvidenceChannel,
    EvidenceVisibility,
    GroundTruthKind,
)
from negotiation.components.tom.updater import BayesianUpdater

POLICY_CATS = ("cooperative", "skeptical", "unknown")
ACTION_CATS = ("accept", "counter", "unknown")
_TEXT_HASH = "sha256:" + "a" * 64


class RaisingModel:
    """Fails the test if any v2 path touches the language model."""

    def sample_text(self, *args, **kwargs):
        raise AssertionError("ToM v2 must never call the language model")


def dist(
    probabilities: tuple[float, ...],
    categories: tuple[str, ...] = POLICY_CATS,
    target: str = "policy_type",
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


def action_dist(probabilities: tuple[float, ...]) -> BeliefDistribution:
    return dist(probabilities, categories=ACTION_CATS, target="next_action")


def evidence(event_id: str, requested: bool) -> Evidence:
    return Evidence(
        observer_id="Seller",
        source_actor_id="Buyer",
        source_event_id=event_id,
        source_call_id=f"call-{event_id}",
        turn=1,
        features=(("requested_evidence", requested),),
        channel=EvidenceChannel.OBSERVABLE,
        visibility=EvidenceVisibility.PUBLIC,
        visible_to=(),
        reliability=0.9,
        extractor_version="rules-1",
        source_text_hash=_TEXT_HASH,
        source_span=(3, 12),
        summary="counterpart message",
    )


@dataclasses.dataclass(frozen=True)
class TableModel:
    """Exact likelihood table: requesting evidence is diagnostic of skeptics."""

    version: str = "table-1"
    target: str = "policy_type"
    categories: tuple[str, ...] = POLICY_CATS

    def likelihood(self, hypothesis: str, item: Evidence):
        requested = dict(item.features).get("requested_evidence")
        if requested is None:
            return None
        table = {
            True: {"cooperative": 0.2, "skeptical": 0.9, "unknown": 0.5},
            False: {"cooperative": 0.8, "skeptical": 0.1, "unknown": 0.5},
        }
        return table[bool(requested)][hypothesis]


MIXTURE = {
    "cooperative": action_dist((0.7, 0.2, 0.1)),
    "skeptical": action_dist((0.1, 0.8, 0.1)),
    "unknown": action_dist((0.34, 0.33, 0.33)),
}


def make_config(**overrides) -> ToMEngineConfig:
    fields: dict = {
        "trial_id": "trial-1",
        "observer_id": "Seller",
        "counterpart_id": "Buyer",
        "condition": EngineCondition.DYNAMIC_TOM,
        "policy_prior": dist((0.6, 0.2, 0.2)),
        "action_mixture": MIXTURE,
        "reservation_prior": dist(
            (0.4, 0.4, 0.2),
            categories=("high", "low", "unknown"),
            target="reservation_value",
        ),
        "trustworthiness_prior": dist(
            (0.5, 0.5),
            categories=("trustworthy", "unknown"),
            target="trustworthiness",
        ),
        "goal_priors": (
            dist(
                (0.7, 0.3),
                categories=("maximize_value", "unknown"),
                target="goal.value",
            ),
        ),
        "constraint_priors": (
            dist(
                (0.4, 0.6),
                categories=("deadline", "unknown"),
                target="constraint.time",
            ),
        ),
        "fact_priors": (
            dist(
                (0.3, 0.4, 0.3),
                categories=("false", "true", "unknown"),
                target="fact.quality",
            ),
        ),
    }
    fields.update(overrides)
    return ToMEngineConfig(**fields)


def make_engine(**overrides) -> ToMV2Engine:
    model = overrides.pop("observation_model", TableModel())
    return ToMV2Engine(make_config(**overrides), observation_model=model)


def candidate(
    action: str, utilities: dict[str, float]
) -> LegalActionCandidate:
    return LegalActionCandidate(
        action=action,
        responses=tuple(
            CounterpartResponseOutcome(
                counterpart_action=counterpart_action,
                response_likelihood={"accept": 0.4, "counter": 0.4,
                                     "unknown": 0.2}[counterpart_action],
                acceptance_probability=0.5,
                agreement_probability=0.5,
                actor_utility=utilities[counterpart_action],
                counterpart_utility=1.0,
            )
            for counterpart_action in ACTION_CATS
        ),
        contract_version="payoff-1",
    )


CANDIDATES = (
    candidate("concede_small", {"accept": 6.0, "counter": 5.0, "unknown": 4.0}),
    candidate("hold_firm", {"accept": 10.0, "counter": 0.0, "unknown": 2.0}),
)
OBJECTIVE = PolicyObjective(
    objective_id="actor_value", version="obj-1", actor_utility_weight=1.0
)


def recommend(engine: ToMV2Engine, turn: int = 1) -> EngineDecision:
    return engine.recommend(
        turn=turn, candidates=CANDIDATES, objective=OBJECTIVE
    )


# ---------------------------------------------------------------------------
# Configuration contracts
# ---------------------------------------------------------------------------


def test_condition_specific_configuration_is_enforced() -> None:
    with pytest.raises(ValidationError, match="requires oracle_policy_type"):
        make_config(condition=EngineCondition.ORACLE_TOM)
    with pytest.raises(ValidationError, match="only valid for oracle_tom"):
        make_config(oracle_policy_type="skeptical")
    with pytest.raises(ValidationError, match="requires counterfactual_clamp"):
        make_config(condition=EngineCondition.COUNTERFACTUAL_BELIEF)
    with pytest.raises(ValidationError, match="pre-recorded belief trajectory"):
        make_config(condition=EngineCondition.SHUFFLED_TOM)
    with pytest.raises(ValidationError, match="cover exactly the policy"):
        make_config(
            action_mixture={"cooperative": MIXTURE["cooperative"]}
        )
    with pytest.raises(ValidationError, match="must be distinct"):
        make_config(counterpart_id="Seller")
    with pytest.raises(ValueError, match="requires an observation model"):
        ToMV2Engine(make_config(), observation_model=None)


# ---------------------------------------------------------------------------
# Dynamic condition: the preregistered-direction check
# ---------------------------------------------------------------------------


def test_diagnostic_evidence_moves_posterior_and_advice_together() -> None:
    engine = make_engine()
    # The advisor requires update lineage: one cooperative-diagnostic
    # observation establishes the baseline belief and advisory.
    engine.ingest_evidence([evidence("event-1", requested=False)])
    before = recommend(engine)
    state_before = engine.state
    assert state_before.policy_type.probability("cooperative") == (
        pytest.approx(0.8)
    )
    assert before.recommendation is not None
    assert before.recommendation.chosen_action == "hold_firm"

    for index in range(2, 5):
        engine.ingest_evidence(
            [evidence(f"event-{index}", requested=True)]
        )

    state_after = engine.state
    top_policy = max(
        zip(
            state_after.policy_type.probabilities,
            state_after.policy_type.categories,
        )
    )[1]
    assert top_policy == "skeptical"
    top_action = max(
        zip(
            state_after.expected_next_action.probabilities,
            state_after.expected_next_action.categories,
        )
    )[1]
    assert top_action == "counter"

    after = recommend(engine, turn=2)
    assert after.recommendation is not None
    assert after.recommendation.chosen_action == "concede_small"
    assert after.state_version == state_after.state_version == 5
    assert len(engine.updates) == 4
    assert len(state_after.evidence_ids) == 4
    assert state_after.update_ids == tuple(sorted(state_after.update_ids))


# ---------------------------------------------------------------------------
# Intervention conditions
# ---------------------------------------------------------------------------


def test_frozen_prior_records_evidence_without_using_it() -> None:
    engine = make_engine(condition=EngineCondition.FROZEN_PRIOR)
    first = engine.ingest_evidence([evidence("event-1", requested=True)])
    before = recommend(engine)
    second = engine.ingest_evidence([evidence("event-2", requested=True)])
    for update in (first, second):
        assert update.posterior.probabilities == update.prior.probabilities
        assert "evidence_ignored_by_frozen_prior" in update.warnings
    after = recommend(engine, turn=2)
    assert (
        after.recommendation.chosen_action
        == before.recommendation.chosen_action
        == "hold_firm"
    )
    assert len(engine.state.evidence_ids) == 2


def test_oracle_pins_the_true_policy_as_an_upper_bound() -> None:
    engine = make_engine(
        condition=EngineCondition.ORACLE_TOM,
        oracle_policy_type="skeptical",
    )
    state = engine.state
    assert state.policy_type.probability("skeptical") == 1.0
    assert state.policy_type.epistemic_status is EpistemicStatus.FROZEN
    engine.ingest_evidence([evidence("event-1", requested=False)])
    assert engine.state.policy_type.probability("skeptical") == 1.0
    decision = recommend(engine)
    assert "oracle upper bound" in decision.guidance
    assert decision.recommendation.chosen_action == "concede_small"


def test_shuffled_replays_a_foreign_trajectory_then_exhausts() -> None:
    foreign = BayesianUpdater().update(
        dist((0.2, 0.6, 0.2)),
        [evidence("foreign-event", requested=True)],
        TableModel(),
    )
    engine = make_engine(
        condition=EngineCondition.SHUFFLED_TOM,
        shuffled_updates=(foreign,),
        observation_model=None,
    )
    update = engine.ingest_evidence([evidence("event-1", requested=False)])
    assert update is foreign
    assert engine.state.policy_type.probabilities == (
        foreign.posterior.probabilities
    )
    assert any("ignored" in note for note in engine.engine_notes)
    with pytest.raises(ValueError, match="exhausted"):
        engine.ingest_evidence([evidence("event-2", requested=False)])


def test_counterfactual_clamp_replaces_the_updated_posterior() -> None:
    clamp = dist((0.9, 0.05, 0.05), status=EpistemicStatus.UPDATED)
    engine = make_engine(
        condition=EngineCondition.COUNTERFACTUAL_BELIEF,
        counterfactual_clamp=clamp,
    )
    engine.ingest_evidence([evidence("event-1", requested=True)])
    assert engine.state.policy_type.probabilities == clamp.probabilities
    assert any("counterfactual_clamp" in note for note in engine.engine_notes)
    decision = recommend(engine)
    assert decision.recommendation.chosen_action == "hold_firm"


def test_text_only_masks_structured_values_from_the_advisor() -> None:
    engine = make_engine(condition=EngineCondition.TOM_TEXT_ONLY)
    engine.ingest_evidence([evidence("event-1", requested=True)])
    decision = recommend(engine)
    assert decision.recommendation is None
    assert re.search(r"\d", decision.guidance) is None
    assert "withheld" in decision.guidance
    with pytest.raises(ValueError, match="no linkable recommendation"):
        engine.link_action("call-1")


def test_link_action_binds_the_trace_exactly_once() -> None:
    engine = make_engine()
    engine.ingest_evidence([evidence("event-1", requested=True)])
    recommend(engine)
    trace = engine.link_action("call-77")
    assert trace.chosen_action_call_id == "call-77"
    with pytest.raises(ValueError, match="no linkable recommendation"):
        engine.link_action("call-78")


# ---------------------------------------------------------------------------
# Component integration (Phases 6 and 9)
# ---------------------------------------------------------------------------


def test_component_rejects_unknown_versions_and_gates_v2_surface() -> None:
    with pytest.raises(ValueError, match="tom_version"):
        TheoryOfMind(model=RaisingModel(), tom_version="v3")
    v1 = TheoryOfMind(model=RaisingModel())
    with pytest.raises(ValueError, match="requires tom_version v2"):
        v1.attach_v2_engine(make_engine())


def test_v2_component_never_calls_the_model_and_renders_guidance() -> None:
    component = TheoryOfMind(model=RaisingModel(), tom_version="v2")
    assert "engine not attached" in component.pre_act(None)
    component.pre_observe("Buyer: show me the inspection report")
    assert component.post_observe() == ""
    component.observe("Buyer: another message")  # v1 path would call model

    engine = make_engine()
    component.attach_v2_engine(engine)
    assert "partner policy leaning cooperative" in component.pre_act(None)

    engine.ingest_evidence([evidence("event-1", requested=True)])
    decision = component.prepare_v2_decision(
        turn=1, candidates=CANDIDATES, objective=OBJECTIVE
    )
    rendered = component.pre_act(None)
    assert decision.guidance in rendered
    assert "Partner-model advisory [dynamic_tom]" in rendered
    trace = component.link_v2_action("call-9")
    assert trace.chosen_action_call_id == "call-9"
    with pytest.raises(ValueError, match="already attached"):
        component.attach_v2_engine(make_engine())


def test_v2_get_state_retains_legacy_keys_and_round_trips() -> None:
    component = TheoryOfMind(model=RaisingModel(), tom_version="v2")
    engine = make_engine()
    component.attach_v2_engine(engine)
    engine.ingest_evidence([evidence("event-1", requested=True)])

    state = component.get_state()
    for legacy_key in (
        "mental_models",
        "belief_records",
        "emotion_history",
        "baseline_patterns",
        "deception_cue_history",
        "deception_indicators",
        "recursion_depth",
        "emotion_sensitivity",
        "empathy_level",
        "recent_emotional_trend",
        "last_observation",
    ):
        assert legacy_key in state
    assert state["tom_version"] == "v2"
    assert state["state_hash"].startswith("sha256:")
    buyer_model = state["mental_models"]["Buyer"]
    assert buyer_model["trust_level"] == pytest.approx(0.5)
    assert "deception_risk" not in buyer_model

    # Round trip into a component whose engine is attached afterwards: the
    # pending snapshot must be applied on attachment.
    restored = TheoryOfMind(model=RaisingModel(), tom_version="v2")
    restored.set_state(state)
    fresh_engine = make_engine()
    restored.attach_v2_engine(fresh_engine)
    assert fresh_engine.state.state_hash == engine.state.state_hash
    assert restored.get_state()["state_hash"] == state["state_hash"]


def test_builder_accepts_tom_version_and_defaults_to_v1() -> None:
    from negotiation.components.config import (
        TheoryOfMindModuleConfig,
        resolve_module_configs,
    )
    from negotiation.constants import ModuleType

    resolved, _ = resolve_module_configs(
        (ModuleType.THEORY_OF_MIND,),
        {"theory_of_mind": {"tom_version": "v2"}},
        trial_seed=7,
        actor_id="Alice",
    )
    config = resolved[ModuleType.THEORY_OF_MIND]
    assert config.tom_version == "v2"
    assert TheoryOfMindModuleConfig().tom_version == "v1"
    with pytest.raises(ValueError, match="tom_version"):
        TheoryOfMindModuleConfig(tom_version="v9")
    component = TheoryOfMind(
        model=RaisingModel(), **dataclasses.asdict(config)
    )
    assert component.tom_version == "v2"
