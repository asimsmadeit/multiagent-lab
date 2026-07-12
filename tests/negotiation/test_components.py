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


def test_every_component_lifecycle_and_state_round_trip(
    model,
    memory_factory,
    action_spec,
) -> None:
    for factory in _component_factories(model, memory_factory):
        component = factory()
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
    assert relationship.get_relationship_strength(now=fixed) == pytest.approx(0.505)


def test_component_instances_do_not_share_mutable_state(model) -> None:
    first = swarm_intelligence.SwarmIntelligence(model)
    second = swarm_intelligence.SwarmIntelligence(model)
    first._sub_agents['market_analysis'].update_performance(1.0)
    assert second._sub_agents['market_analysis']._performance_history == []


@pytest.mark.parametrize('style', ['unknown', '', 'collusive'])
def test_basic_strategy_rejects_unknown_style(style: str) -> None:
    with pytest.raises(ValueError):
        negotiation_strategy.BasicNegotiationStrategy('Alice', style)
