"""End-to-end fake-model construction tests for negotiation entities."""

from __future__ import annotations

import copy

import pytest

from negotiation import advanced_negotiator
from negotiation import base_negotiator
from negotiation.constants import MODULE_COMPONENT_NAMES, ModuleType
from negotiation.game_master import negotiation as gm_negotiation


ALL_MODULES = [module.value for module in ModuleType]


def test_base_agent_observes_acts_and_round_trips(
    model,
    memory_factory,
    action_spec,
) -> None:
    agent = base_negotiator.build_agent(
        model,
        memory_factory(),
        name='Alice',
        reservation_value=50,
    )
    assert (
        agent._context_components['BasicNegotiationStrategy']
        ._state.current_position > 0.0
    )
    agent.observe('Bob offers $70 for delivery.')
    action = agent.act(action_spec)
    assert action.startswith('Alice ')

    state = copy.deepcopy(agent.get_state())
    restored = base_negotiator.build_agent(
        model,
        memory_factory(),
        name='Alice',
        reservation_value=50,
    )
    restored.set_state(copy.deepcopy(state))
    restored_state = restored.get_state()
    for component_name in (
        'NegotiationInstructions',
        'NegotiationMemory',
        'BasicNegotiationStrategy',
    ):
        assert (
            restored_state['context_components'][component_name]
            == state['context_components'][component_name]
        )
    # The associative-memory dependency serializes a set as an unordered list,
    # so compare negotiation-owned state exactly and verify entity behavior.
    assert restored.act(action_spec).startswith('Alice ')


def test_advanced_agent_builds_and_runs_every_module(
    model,
    memory_factory,
    action_spec,
) -> None:
    agent = advanced_negotiator.build_agent(
        model,
        memory_factory(),
        name='Alice',
        reservation_value=50,
        modules=ALL_MODULES,
        module_configs={
            'strategy_evolution': {'population_size': 4, 'seed': 4},
            'uncertainty_aware': {'seed': 4},
        },
    )
    expected_names = set(MODULE_COMPONENT_NAMES.values())
    assert expected_names.issubset(agent._context_components)

    agent.observe('Bob is concerned and offers $70 for a long-term renewal.')
    assert agent.act(action_spec).startswith('Alice ')
    detected = gm_negotiation.gm_modules_registry.detect_agent_modules([agent])
    assert detected['Alice'] == set(ALL_MODULES)


def test_advanced_agent_rejects_invalid_module_inputs(model, memory_factory) -> None:
    with pytest.raises(ValueError, match='Unknown negotiation modules'):
        advanced_negotiator.build_agent(
            model, memory_factory(), modules=['not_a_module']
        )
    with pytest.raises(ValueError, match='configuration must be an object'):
        advanced_negotiator.Entity(params={
            'modules': 'theory_of_mind',
            'module_configs': {'theory_of_mind': 3},
        }).build(model, memory_factory())


def test_game_master_auto_detects_modules_and_completes_fake_act(
    model,
    memory_factory,
) -> None:
    alice = advanced_negotiator.build_agent(
        model,
        memory_factory(),
        name='Alice',
        modules=['theory_of_mind'],
    )
    bob = advanced_negotiator.build_agent(
        model,
        memory_factory(),
        name='Bob',
        modules=['uncertainty_aware'],
        module_configs={'uncertainty_aware': {'seed': 8}},
    )
    game_master = gm_negotiation.NegotiationGameMaster(
        params={
            'name': 'Mediator',
            'gm_modules': [],
            'gm_module_configs': {},
            'auto_detect_modules': True,
            'max_rounds': 5,
        },
        entities=[alice, bob],
    ).build(model, memory_factory())

    assert 'gm_module_social_intelligence' in game_master._context_components
    assert 'gm_module_uncertainty_management' in game_master._context_components
    instructions = game_master.get_component('instructions')
    assert instructions.get_state()['state'].startswith(
        'You are a professional negotiation mediator.'
    )
    make_observation = game_master.get_component('__make_observation__')
    assert 'gm_module_social_intelligence' in make_observation._components
    assert 'gm_module_uncertainty_management' in make_observation._components
    assert make_observation._allow_llm_fallback is False
    assert isinstance(game_master.act(), str)


def test_game_master_rejects_unknown_modules_and_invalid_participants(
    model,
    memory_factory,
) -> None:
    with pytest.raises(ValueError, match='two unique entities'):
        gm_negotiation.NegotiationGameMaster(
            params={'name': 'Mediator'}, entities=[]
        ).build(model, memory_factory())

    alice = base_negotiator.build_agent(model, memory_factory(), name='Alice')
    bob = base_negotiator.build_agent(model, memory_factory(), name='Bob')
    with pytest.raises(ValueError, match='Unknown game-master modules'):
        gm_negotiation.NegotiationGameMaster(
            params={'name': 'Mediator', 'gm_modules': ['unknown']},
            entities=[alice, bob],
        ).build(model, memory_factory())
