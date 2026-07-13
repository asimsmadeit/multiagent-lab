"""Offline contract tests for every negotiation agent component."""

from __future__ import annotations

import copy
import datetime
import json

import pytest

from negotiation.components import cultural_adaptation
from negotiation.components import negotiation_instructions
from negotiation.components import negotiation_memory
from negotiation.components import negotiation_strategy
from negotiation.components import strategy_evolution
from negotiation.components import swarm_intelligence
from negotiation.components import temporal_strategy
from negotiation.components import theory_of_mind
from negotiation.components import uncertainty_aware
from negotiation.components.contracts import (
    ComponentDiagnosticContract,
    ModelCallDeclaration,
    StatefulComponent,
)
from negotiation.utils import parsing


def _component_factories(model, memory_factory):
    return [
        lambda: negotiation_instructions.NegotiationInstructions('Alice', 'Close'),
        lambda: negotiation_memory.NegotiationMemory('Alice', memory_factory()),
        lambda: negotiation_strategy.BasicNegotiationStrategy('Alice'),
        lambda: cultural_adaptation.CulturalAdaptation(model),
        lambda: temporal_strategy.TemporalStrategy(model),
        lambda: swarm_intelligence.SwarmIntelligence(model),
        lambda: uncertainty_aware.UncertaintyAware(model, seed=17),
        lambda: strategy_evolution.StrategyEvolution(
            model, population_size=4, seed=17
        ),
        lambda: theory_of_mind.TheoryOfMind(model),
    ]


@pytest.mark.parametrize(
    'component_class',
    [
        cultural_adaptation.CulturalAdaptation,
        temporal_strategy.TemporalStrategy,
        swarm_intelligence.SwarmIntelligence,
        uncertainty_aware.UncertaintyAware,
        strategy_evolution.StrategyEvolution,
        theory_of_mind.TheoryOfMind,
    ],
)
def test_every_optional_component_declares_diagnostic_contract(
    component_class,
) -> None:
    contract = component_class.DIAGNOSTIC_CONTRACT
    assert isinstance(contract, ComponentDiagnosticContract)
    assert contract.inputs
    assert contract.outputs
    assert contract.state_fields
    assert contract.logging_fields
    assert set(contract.extra_model_calls) >= {
        'pre_act', 'post_act', 'pre_observe', 'post_observe'
    }
    json.dumps(contract.to_dict())


def test_diagnostic_contract_is_immutable_and_validated() -> None:
    contract = theory_of_mind.TheoryOfMind.DIAGNOSTIC_CONTRACT
    with pytest.raises(TypeError):
        contract.extra_model_calls['pre_act'] = 99
    with pytest.raises(ValueError, match='duplicate'):
        ComponentDiagnosticContract(
            inputs=('observation', 'observation'),
            outputs=('diagnostic',),
            state_fields=('state',),
            extra_model_calls={
                'pre_act': 0, 'post_act': 0, 'pre_observe': 0,
                'post_observe': 0,
            },
            logging_fields=('status',),
        )
    with pytest.raises(ValueError, match='ordered'):
        ModelCallDeclaration(2, 1, 'invalid')


def test_every_component_lifecycle_and_state_round_trip(
    model,
    memory_factory,
    action_spec,
) -> None:
    for factory in _component_factories(model, memory_factory):
        component = factory()
        if hasattr(type(component), 'DIAGNOSTIC_CONTRACT'):
            assert isinstance(component, StatefulComponent)
            assert set(component.DIAGNOSTIC_CONTRACT.state_fields) == set(
                component.get_state()
            )
        assert isinstance(component.pre_act(action_spec), str)
        assert isinstance(component.post_act('I offer $75 for the package.'), str)
        assert isinstance(
            component.pre_observe('Bob is concerned but offers $80.'), str
        )
        assert isinstance(component.post_observe(), str)
        component.update()

        state = copy.deepcopy(component.get_state())
        json.dumps(state)
        restored = factory()
        restored.set_state(copy.deepcopy(state))
        assert restored.get_state() == state, type(component).__name__


def test_conditional_model_call_contracts_are_executable_and_instrumented(
    model,
    action_spec,
) -> None:
    cultural = cultural_adaptation.CulturalAdaptation(model)
    before = len(model.prompts)
    cultural.pre_observe('short')
    assert len(model.prompts) - before == 0
    assert cultural.get_model_call_diagnostics()['pre_observe'] == {
        'branch': 'too_short',
        'observed_calls': 0,
        'expected_calls': 0,
    }
    before = len(model.prompts)
    cultural.pre_observe(
        'Bob uses a direct professional style with detailed written terms.'
    )
    actual = len(model.prompts) - before
    cultural.DIAGNOSTIC_CONTRACT.validate_model_calls('pre_observe', actual)
    assert actual == 1
    assert (
        cultural.get_model_call_diagnostics()['pre_observe']['observed_calls']
        == actual
    )

    disabled = cultural_adaptation.CulturalAdaptation(
        model,
        detect_culture=False,
    )
    before = len(model.prompts)
    disabled.pre_observe('This is long enough to trigger detection when enabled.')
    assert len(model.prompts) - before == 0
    assert disabled.get_model_call_diagnostics()['pre_observe']['branch'] == 'disabled'

    swarm = swarm_intelligence.SwarmIntelligence(
        model,
        enable_sub_agents=['market_analysis', 'game_theory'],
    )
    before = len(model.prompts)
    swarm.pre_act(action_spec)
    actual = len(model.prompts) - before
    swarm.DIAGNOSTIC_CONTRACT.validate_model_calls('pre_act', actual)
    assert actual == 2
    assert swarm.get_model_call_diagnostics()['pre_act'] == {
        'branch': '2_enabled_sub_agents',
        'observed_calls': 2,
        'expected_calls': 2,
    }

    empty_swarm = swarm_intelligence.SwarmIntelligence(
        model,
        enable_sub_agents=[],
    )
    before = len(model.prompts)
    empty_swarm.pre_act(action_spec)
    assert len(model.prompts) - before == 0
    assert empty_swarm.get_model_call_diagnostics()['pre_act'][
        'expected_calls'
    ] == 0


def test_compact_numeric_parsing_clamps_each_named_field() -> None:
    parsed = parsing.parse_named_floats(
        'trust:1.7 valence:-2 confidence:.75',
        {'trust': (0.0, 1.0), 'valence': (-1.0, 1.0), 'confidence': (0.0, 1.0)},
    )
    assert parsed == {'trust': 1.0, 'valence': -1.0, 'confidence': 0.75}


def test_offer_parser_prefers_offer_amount_over_unrelated_money() -> None:
    text = 'Our budget is $100, but in round 2 I offer $75 for delivery.'
    assert parsing.parse_offer_value(text) == 75.0
    assert parsing.signals_agreement("I don't accept that offer") is False
    assert parsing.signals_agreement('We accept your offer') is True


def test_strategy_evolution_is_seeded_and_keeps_multi_turn_episode(
    model,
    action_spec,
) -> None:
    first = strategy_evolution.StrategyEvolution(model, population_size=4, seed=9)
    second = strategy_evolution.StrategyEvolution(model, population_size=4, seed=9)
    assert first.get_state() == second.get_state()

    first.pre_act(action_spec)
    first.post_act('First move')
    episode = first._current_episode
    first.pre_observe('Bob asks for more detail.')
    first.pre_act(action_spec)
    first.post_act('Second move')
    assert first._current_episode is episode
    assert first._current_episode.actions_taken == ['First move', 'Second move']

    first.pre_observe('We accept your offer and have a deal.')
    assert first._current_episode is None
    assert first._total_negotiations == 1


def test_uncertainty_scenarios_are_seeded(model) -> None:
    first = uncertainty_aware.UncertaintyAware(model, seed=31)
    second = uncertainty_aware.UncertaintyAware(model, seed=31)
    assert first._generate_scenarios() == second._generate_scenarios()


def test_uncertainty_state_restores_behavioral_configuration(model) -> None:
    component = uncertainty_aware.UncertaintyAware(
        model,
        confidence_threshold=0.8,
        risk_tolerance=0.2,
        information_gathering_budget=0.3,
        seed=31,
    )
    restored = uncertainty_aware.UncertaintyAware(model, seed=1)
    restored.set_state(copy.deepcopy(component.get_state()))

    assert restored._confidence_threshold == pytest.approx(0.8)
    assert restored._risk_tolerance == pytest.approx(0.2)
    assert restored._info_budget == pytest.approx(0.3)
    assert restored.get_state() == component.get_state()


def test_strategy_state_restores_evolution_configuration(model) -> None:
    component = strategy_evolution.StrategyEvolution(
        model,
        population_size=4,
        mutation_rate=0.25,
        crossover_rate=0.6,
        learning_rate=0.02,
        seed=17,
    )
    restored = strategy_evolution.StrategyEvolution(
        model, population_size=4, seed=1
    )
    restored.set_state(copy.deepcopy(component.get_state()))

    assert restored._population_size == 4
    assert restored._mutation_rate == pytest.approx(0.25)
    assert restored._crossover_rate == pytest.approx(0.6)
    assert restored._learning_rate == pytest.approx(0.02)
    assert restored.get_state() == component.get_state()


def test_injectable_clocks_remove_wall_clock_variance(memory_factory) -> None:
    fixed = datetime.datetime(2030, 1, 2, 3, 4, 5)
    memory = negotiation_memory.NegotiationMemory(
        'Alice', memory_factory(), clock=lambda: fixed
    )
    memory.post_act('I offer $75.')
    assert memory._offer_history[0].timestamp == fixed

    relationship = temporal_strategy.RelationshipRecord('Bob')
    relationship.update_interaction(10.0, {}, now=fixed)
    assert relationship.last_interaction == fixed
    assert relationship.trust_score == 0.5
    assert relationship.get_relationship_strength(now=fixed) == pytest.approx(0.48)


def test_component_instances_do_not_share_mutable_state(model) -> None:
    first = swarm_intelligence.SwarmIntelligence(model)
    second = swarm_intelligence.SwarmIntelligence(model)
    first._sub_agents['market_analysis'].update_performance(1.0)
    assert second._sub_agents['market_analysis']._performance_history == []


def _swarm_analysis(agent_type: str, recommendation: str):
    return swarm_intelligence.SubAgentAnalysis(
        agent_type=agent_type,
        analysis=f'{agent_type} analysis',
        recommendations=[recommendation],
        confidence=1.0,
        key_factors=[],
        risks=[],
        opportunities=[],
    )


def test_swarm_consensus_threshold_and_iteration_limit_are_active(model) -> None:
    analyses = {
        'market_analysis': _swarm_analysis('market_analysis', 'Hold price'),
        'emotional_intelligence': _swarm_analysis(
            'emotional_intelligence', 'Acknowledge concern'
        ),
        'game_theory': _swarm_analysis('game_theory', 'Request reciprocity'),
        'diplomatic_relations': _swarm_analysis(
            'diplomatic_relations', 'Protect relationship'
        ),
    }
    weights = {name: 0.25 for name in analyses}

    bounded = swarm_intelligence.SwarmIntelligence(
        model, consensus_threshold=0.75, max_iterations=1
    )
    expanded = swarm_intelligence.SwarmIntelligence(
        model, consensus_threshold=0.75, max_iterations=3
    )
    low_threshold = swarm_intelligence.SwarmIntelligence(
        model, consensus_threshold=0.20, max_iterations=1
    )

    bounded_decision = bounded._build_collective_decision(analyses, weights)
    expanded_decision = expanded._build_collective_decision(analyses, weights)
    low_threshold_decision = low_threshold._build_collective_decision(
        analyses, weights
    )

    assert bounded_decision.iterations_used == 1
    assert bounded_decision.support_ratio == pytest.approx(0.25)
    assert bounded_decision.consensus_reached is False
    assert expanded_decision.iterations_used == 3
    assert expanded_decision.support_ratio == pytest.approx(0.75)
    assert expanded_decision.consensus_reached is True
    assert low_threshold_decision.consensus_reached is True
    assert low_threshold_decision.iterations_used == 1


def test_swarm_state_restores_consensus_configuration(model) -> None:
    component = swarm_intelligence.SwarmIntelligence(
        model, consensus_threshold=0.85, max_iterations=2
    )
    restored = swarm_intelligence.SwarmIntelligence(model)
    restored.set_state(copy.deepcopy(component.get_state()))

    assert restored._consensus_threshold == pytest.approx(0.85)
    assert restored._max_iterations == 2
    assert restored.get_state() == component.get_state()


class _TemporalExtractionModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def sample_text(self, prompt: str, **_kwargs: object) -> str:
        assert 'numeric_concession_or_unknown' in prompt
        return self.response


def test_temporal_observation_populates_evidence_bearing_metrics() -> None:
    fixed = datetime.datetime(2030, 2, 3, 4, 5, 6)
    component = temporal_strategy.TemporalStrategy(
        _TemporalExtractionModel(
            'Bob|agreement|promise kept; reputation positive|$75|5'
        ),
        clock=lambda: fixed,
    )

    component.observe('Bob completed the agreed exchange and conceded $5.')

    relationship = component._relationships['Bob']
    assert relationship.interaction_count == 1
    assert relationship.total_value_exchanged == pytest.approx(75.0)
    assert relationship.concession_history == [5.0]
    assert relationship.last_interaction == fixed
    assert relationship.trust_score == pytest.approx(0.55)
    assert relationship.reputation_score == pytest.approx(0.55)
    assert relationship.outcome_history[0]['promises_kept'] is True
    assert relationship.outcome_history[0]['source_observation'].startswith('Bob')


def test_temporal_observation_does_not_fabricate_unknown_metrics() -> None:
    component = temporal_strategy.TemporalStrategy(
        _TemporalExtractionModel('Bob|continued|neutral|unknown|unknown')
    )

    component.observe('Bob requested another round.')

    relationship = component._relationships['Bob']
    assert relationship.interaction_count == 1
    assert relationship.total_value_exchanged == 0.0
    assert relationship.concession_history == []
    assert relationship.trust_score == 0.5
    assert relationship.reputation_score == 0.5
    assert 'promises_kept' not in relationship.outcome_history[0]


def test_temporal_state_restores_behavioral_configuration(model) -> None:
    component = temporal_strategy.TemporalStrategy(
        model,
        discount_factor=0.75,
        reputation_weight=0.55,
        relationship_investment_threshold=0.8,
    )
    restored = temporal_strategy.TemporalStrategy(model)
    restored.set_state(copy.deepcopy(component.get_state()))

    assert restored._discount_factor == pytest.approx(0.75)
    assert restored._reputation_weight == pytest.approx(0.55)
    assert restored._investment_threshold == pytest.approx(0.8)
    assert restored.get_state() == component.get_state()


def test_cultural_state_and_public_diagnostics_preserve_configuration(model) -> None:
    component = cultural_adaptation.CulturalAdaptation(
        model,
        own_culture='east_asian',
        adaptation_level=0.4,
        detect_culture=False,
    )
    diagnostics = component.get_public_diagnostics()
    restored = cultural_adaptation.CulturalAdaptation(model)
    restored.set_state(copy.deepcopy(component.get_state()))

    assert diagnostics == {
        'own_culture': 'east_asian',
        'detected_counterpart_culture': None,
        'adaptation_level': 0.4,
        'detect_culture': False,
    }
    assert restored.get_own_culture() == 'east_asian'
    assert restored.get_public_diagnostics() == diagnostics
    assert restored.get_state() == component.get_state()


@pytest.mark.parametrize('style', ['unknown', '', 'collusive'])
def test_basic_strategy_rejects_unknown_style(style: str) -> None:
    with pytest.raises(ValueError):
        negotiation_strategy.BasicNegotiationStrategy('Alice', style)
