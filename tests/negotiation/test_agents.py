"""End-to-end fake-model construction tests for negotiation entities."""

from __future__ import annotations

import copy
import dataclasses
from contextlib import contextmanager

import pytest

import negotiation
from negotiation import advanced_negotiator
from negotiation import base_negotiator
from negotiation.components.config import (
    CulturalAdaptationConfig,
    MODULE_CONFIG_TYPES,
    StrategyEvolutionConfig,
    SwarmIntelligenceConfig,
    TemporalStrategyConfig,
    TheoryOfMindModuleConfig,
    UncertaintyAwareConfig,
    resolve_module_configs,
)
from negotiation.constants import (
    DEFAULT_MODULE_CONFIGS,
    MODULE_COMPONENT_NAMES,
    ModuleType,
)
from negotiation.game_master import negotiation as gm_negotiation
from negotiation.game_master import adjudication as gm_adjudication


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


def test_final_action_scope_excludes_component_reasoning_calls(
    model,
    memory_factory,
    action_spec,
) -> None:
    scope_active = False
    call_scope_flags = []

    class ScopeAwareModel:
        def sample_text(self, prompt, **kwargs):
            call_scope_flags.append(scope_active)
            return model.sample_text(prompt, **kwargs)

        def sample_choice(self, prompt, responses, **kwargs):
            call_scope_flags.append(scope_active)
            return model.sample_choice(prompt, responses, **kwargs)

    @contextmanager
    def action_scope():
        nonlocal scope_active
        assert scope_active is False
        scope_active = True
        try:
            yield
        finally:
            scope_active = False

    agent = base_negotiator.build_agent(
        ScopeAwareModel(),
        memory_factory(),
        name='Alice',
        action_call_scope_factory=action_scope,
    )

    agent.act(action_spec)

    assert call_scope_flags.count(True) == 1
    assert call_scope_flags.count(False) >= 3
    assert scope_active is False


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
    with pytest.raises(ValueError, match='disabled module'):
        advanced_negotiator.build_agent(
            model,
            memory_factory(),
            modules=['theory_of_mind'],
            module_configs={'cultural_adaptation': {}},
        )
    with pytest.raises(ValueError, match='Unknown module configuration'):
        advanced_negotiator.build_agent(
            model,
            memory_factory(),
            modules=['theory_of_mind'],
            module_configs={'typo_module': {}},
        )
    with pytest.raises(ValueError, match='Unknown theory_of_mind.*keys'):
        advanced_negotiator.build_agent(
            model,
            memory_factory(),
            modules=['theory_of_mind'],
            module_configs={'theory_of_mind': {'empathy_typo': 0.5}},
        )
    with pytest.raises(ValueError, match='duplicates'):
        advanced_negotiator.build_agent(
            model,
            memory_factory(),
            modules=['theory_of_mind', 'theory_of_mind'],
        )
    with pytest.raises(ValueError, match='seed'):
        advanced_negotiator.build_agent(
            model,
            memory_factory(),
            modules=['uncertainty_aware'],
            trial_seed=-1,
        )


def test_typed_defaults_reach_every_advanced_component(
    model,
    memory_factory,
) -> None:
    expected_types = {
        ModuleType.CULTURAL_ADAPTATION: CulturalAdaptationConfig,
        ModuleType.TEMPORAL_STRATEGY: TemporalStrategyConfig,
        ModuleType.SWARM_INTELLIGENCE: SwarmIntelligenceConfig,
        ModuleType.UNCERTAINTY_AWARE: UncertaintyAwareConfig,
        ModuleType.STRATEGY_EVOLUTION: StrategyEvolutionConfig,
        ModuleType.THEORY_OF_MIND: TheoryOfMindModuleConfig,
    }
    resolved, provenance = resolve_module_configs(
        tuple(ModuleType),
        {},
        trial_seed=41,
        actor_id='Alice',
    )
    assert dict(MODULE_CONFIG_TYPES) == expected_types
    for module_type, config in resolved.items():
        assert isinstance(config, expected_types[module_type])
        expected = dict(DEFAULT_MODULE_CONFIGS[module_type])
        if module_type in {
            ModuleType.UNCERTAINTY_AWARE,
            ModuleType.STRATEGY_EVOLUTION,
        }:
            expected['seed'] = provenance[module_type]['resolved_seed']
        assert dataclasses.asdict(config) == expected

    agent = advanced_negotiator.build_agent(
        model,
        memory_factory(),
        name='Alice',
        modules=ALL_MODULES,
        trial_seed=41,
    )
    cultural = agent._context_components['CulturalAdaptation']
    temporal = agent._context_components['TemporalStrategy']
    swarm = agent._context_components['SwarmIntelligence']
    uncertainty = agent._context_components['UncertaintyAware']
    evolution = agent._context_components['StrategyEvolution']
    tom = agent._context_components['TheoryOfMind']
    assert cultural._adaptation_level == pytest.approx(0.7)
    assert cultural._detect_culture is True
    assert temporal._discount_factor == pytest.approx(0.9)
    assert temporal._reputation_weight == pytest.approx(0.3)
    assert swarm._consensus_threshold == pytest.approx(0.7)
    assert swarm._max_iterations == 3
    assert set(swarm._sub_agents) == {
        'market_analysis', 'emotional_intelligence',
        'game_theory', 'diplomatic_relations',
    }
    assert uncertainty._confidence_threshold == pytest.approx(0.7)
    assert evolution._population_size == 20
    assert tom._max_recursion_depth == 3
    assert tom._emotion_sensitivity == pytest.approx(0.7)


def test_typed_configs_are_public_and_allow_disabling_tom_recursion() -> None:
    assert negotiation.CulturalAdaptationConfig is CulturalAdaptationConfig
    assert negotiation.TemporalStrategyConfig is TemporalStrategyConfig
    assert negotiation.SwarmIntelligenceConfig is SwarmIntelligenceConfig
    assert negotiation.UncertaintyAwareConfig is UncertaintyAwareConfig
    assert negotiation.StrategyEvolutionConfig is StrategyEvolutionConfig
    assert negotiation.TheoryOfMindModuleConfig is TheoryOfMindModuleConfig
    assert TheoryOfMindModuleConfig(max_recursion_depth=0).max_recursion_depth == 0


def test_public_builder_accepts_typed_module_config_objects(
    model,
    memory_factory,
) -> None:
    config = UncertaintyAwareConfig(
        confidence_threshold=0.85,
        risk_tolerance=0.2,
        seed=19,
    )
    agent = advanced_negotiator.build_agent(
        model,
        memory_factory(),
        name='Alice',
        modules=[ModuleType.UNCERTAINTY_AWARE],
        module_configs={ModuleType.UNCERTAINTY_AWARE: config},
        trial_seed=71,
    )

    component = agent._context_components['UncertaintyAware']
    assert component._confidence_threshold == pytest.approx(0.85)
    assert component._risk_tolerance == pytest.approx(0.2)
    assert component.get_seed_provenance()['source'] == 'module_config'


def test_public_builder_propagates_actor_scoped_trial_seed_provenance(
    model,
    memory_factory,
) -> None:
    alice = advanced_negotiator.build_adaptive_agent(
        model,
        memory_factory(),
        name='Alice',
        trial_seed=73,
    )
    bob = advanced_negotiator.build_adaptive_agent(
        model,
        memory_factory(),
        name='Bob',
        trial_seed=73,
    )
    alice_uncertainty = alice._context_components['UncertaintyAware']
    alice_evolution = alice._context_components['StrategyEvolution']
    bob_uncertainty = bob._context_components['UncertaintyAware']

    for component in (alice_uncertainty, alice_evolution):
        provenance = component.get_seed_provenance()
        assert provenance['source'] == 'trial_seed_derived'
        assert provenance['trial_seed'] == 73
        assert provenance['actor_id'] == 'Alice'
        assert provenance['resolved_seed'] is not None
    assert (
        alice_uncertainty.get_seed_provenance()['resolved_seed']
        != bob_uncertainty.get_seed_provenance()['resolved_seed']
    )

    explicit = advanced_negotiator.build_agent(
        model,
        memory_factory(),
        name='Alice',
        modules=['uncertainty_aware'],
        module_configs={'uncertainty_aware': {'seed': 5}},
        trial_seed=73,
    )
    assert explicit._context_components[
        'UncertaintyAware'
    ].get_seed_provenance()['source'] == 'module_config'


def test_default_stochastic_builders_are_deterministic_not_unseeded(
    model,
    memory_factory,
) -> None:
    collective = advanced_negotiator.build_collective_agent(
        model,
        memory_factory(),
        name='Collective',
    )
    adaptive = advanced_negotiator.build_adaptive_agent(
        model,
        memory_factory(),
        name='Adaptive',
    )

    for component in (
        collective._context_components['UncertaintyAware'],
        adaptive._context_components['UncertaintyAware'],
        adaptive._context_components['StrategyEvolution'],
    ):
        provenance = component.get_seed_provenance()
        assert provenance['source'] == 'trial_seed_derived'
        assert provenance['trial_seed'] == 0
        assert provenance['resolved_seed'] is not None

    with pytest.raises(ValueError, match='requires an explicit.*seed'):
        advanced_negotiator.build_agent(
            model,
            memory_factory(),
            modules=['uncertainty_aware'],
            trial_seed=None,
        )


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
    adjudicator = game_master.get_component('negotiation_adjudicator')
    assert isinstance(adjudicator, gm_adjudication.NegotiationAdjudicator)
    assert adjudicator.next_actor == 'Alice'
    assert 'Phase: opening' in adjudicator.pre_act(None)
    assert (
        game_master.get_component('negotiation_state').pre_act(None)
        != 'No active negotiations.'
    )
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

    for params in (
        {'name': 'Mediator', 'protocol': 'simultaneous'},
        {'name': 'Mediator', 'acting_order': 'model_choice'},
        {'name': 'Mediator', 'protocol': 'unknown'},
    ):
        with pytest.raises(ValueError, match='supports only protocol=alternating'):
            gm_negotiation.NegotiationGameMaster(
                params=params,
                entities=[alice, bob],
            ).build(model, memory_factory())


def test_game_master_reads_declared_agent_culture_through_public_api(
    model,
    memory_factory,
) -> None:
    alice = advanced_negotiator.build_agent(
        model,
        memory_factory(),
        name='Alice',
        modules=['cultural_adaptation'],
        module_configs={'cultural_adaptation': {'own_culture': 'east_asian'}},
    )
    bob = advanced_negotiator.build_agent(
        model,
        memory_factory(),
        name='Bob',
        modules=['cultural_adaptation'],
        module_configs={
            'cultural_adaptation': {'own_culture': 'northern_european'}
        },
    )

    game_master = gm_negotiation.NegotiationGameMaster(
        params={
            'name': 'Mediator',
            'gm_modules': ['cultural_awareness'],
            'auto_detect_modules': False,
        },
        entities=[alice, bob],
    ).build(model, memory_factory())
    cultural_gm = game_master.get_component('gm_module_cultural_awareness')

    assert cultural_gm.get_state()['participant_cultures'] == {
        'Alice': 'east_asian',
        'Bob': 'northern_european',
    }
