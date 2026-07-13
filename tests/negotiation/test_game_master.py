"""Offline tests for negotiation game-master state and modules."""

from __future__ import annotations

import copy
import datetime
import json

import pytest

from negotiation.game_master.components import gm_collective_intelligence
from negotiation.game_master.components import gm_cultural_awareness
from negotiation.game_master.components import gm_modules
from negotiation.game_master.components import gm_social_intelligence
from negotiation.game_master.components import gm_state
from negotiation.game_master.components import gm_strategy_evolution
from negotiation.game_master.components import gm_temporal_dynamics
from negotiation.game_master.components import gm_uncertainty_management
from negotiation.game_master.components import gm_validation


def _context(round_number: int = 3) -> gm_modules.ModuleContext:
    all_modules = {
        'cultural_adaptation',
        'theory_of_mind',
        'temporal_strategy',
        'uncertainty_aware',
        'swarm_intelligence',
        'strategy_evolution',
    }
    return gm_modules.ModuleContext(
        negotiation_id='neg-1',
        participants=['Alice', 'Bob', 'Cara'],
        current_phase='bargaining',
        current_round=round_number,
        active_modules={'Alice': all_modules, 'Bob': set(), 'Cara': set()},
        shared_data={},
    )


def _gm_factories():
    return [
        gm_cultural_awareness.CulturalAwarenessGM,
        gm_social_intelligence.SocialIntelligenceGM,
        gm_temporal_dynamics.TemporalDynamicsGM,
        gm_uncertainty_management.UncertaintyManagementGM,
        gm_collective_intelligence.CollectiveIntelligenceGM,
        gm_strategy_evolution.StrategyEvolutionGM,
    ]


def test_every_gm_module_validates_updates_and_round_trips() -> None:
    context = _context()
    for factory in _gm_factories():
        module = factory()
        if isinstance(module, gm_cultural_awareness.CulturalAwarenessGM):
            module.set_participant_culture('Alice', 'western_business')
            module.set_participant_culture('Bob', 'east_asian')

        module.set_module_state('audit_marker', {'round': 3})
        for actor, event in (
            ('Alice', 'I am frustrated but offer $75 and suggest working together.'),
            ('Bob', 'I agree, share the constraint, and promise delivery.'),
        ):
            valid, error = module.validate_action(actor, event, context)
            assert isinstance(valid, bool)
            assert error is None or isinstance(error, str)
            module.update_state(event, actor, context)

        assert isinstance(module.get_observation_context('Alice', context), str)
        assert isinstance(module.get_module_report(), str)
        state = copy.deepcopy(module.get_state())
        json.dumps(state)
        restored = factory()
        restored.set_state(copy.deepcopy(state))
        assert restored.get_state() == state, factory.__name__


def test_gm_module_instances_are_isolated() -> None:
    first = gm_social_intelligence.SocialIntelligenceGM()
    second = gm_social_intelligence.SocialIntelligenceGM()
    first.update_state('I am angry.', 'Alice', _context())
    assert second.get_state()['emotional_history'] == []


def test_state_tracker_enforces_participants_clock_and_round_trip() -> None:
    fixed = datetime.datetime(2030, 1, 2, 3, 4, 5)
    tracker = gm_state.NegotiationStateTracker(
        max_rounds=2,
        clock=lambda: fixed,
    )
    state = tracker.start_negotiation('neg-1', ['Alice', 'Bob'])
    offer = tracker.record_offer(
        'neg-1', 'Alice', 'Bob', 'initial', {'price': 75}
    )
    assert offer.timestamp == fixed
    with pytest.raises(ValueError, match='intended recipient'):
        tracker.accept_offer('neg-1', offer, 'Alice')

    tracker.accept_offer('neg-1', offer, 'Bob')
    tracker.terminate_negotiation('neg-1', 'agreement', success=True)
    assert state.phase == 'completed'
    snapshot = tracker.get_state()
    json.dumps(snapshot)
    restored = gm_state.NegotiationStateTracker()
    restored.set_state(copy.deepcopy(snapshot))
    assert restored.get_state() == snapshot


def test_validator_round_trip_preserves_enforcement() -> None:
    validator = gm_validation.NegotiationValidator(domain_type='price')
    validator.set_batna('Alice', {'role': 'seller', 'price': 50})
    agreement = validator.validate_agreement(
        ['Alice', 'Bob'],
        {'price': 75},
        {'market_price': 70},
    )
    assert agreement.validation_status == 'valid'
    assert validator.enforce_agreement(agreement.agreement_id)

    snapshot = validator.get_state()
    restored = gm_validation.NegotiationValidator(domain_type='price')
    restored.set_state(copy.deepcopy(snapshot))
    assert restored.get_state() == snapshot
    assert restored.enforce_agreement(agreement.agreement_id)


def test_contract_feasibility_uses_explicit_typed_constraints() -> None:
    validator = gm_validation.NegotiationValidator(
        domain_type='contract',
        enable_batna_check=False,
        enable_fairness_check=False,
    )
    valid, errors = validator.validate_offer(
        'Alice',
        {
            'duration': 12,
            'payment_terms': 'net 30',
            'start_date': '2030-02-01',
        },
        {
            'available_duration_months': 18,
            'earliest_start_date': '2030-01-01',
        },
    )
    assert valid is True
    assert errors == []

    invalid, errors = validator.validate_offer(
        'Alice',
        {
            'duration': 24,
            'payment_terms': 'net 30',
            'start_date': 'not-a-date',
        },
        {
            'available_duration_months': 18,
            'earliest_start_date': '2030-01-01',
        },
    )
    assert invalid is False
    assert 'Contract duration exceeds documented availability' in errors
    assert 'Contract start_date must use ISO YYYY-MM-DD format' in errors


def test_registry_contains_every_module_and_suggestions_are_stable() -> None:
    registered = set(gm_modules.NegotiationGMModuleRegistry.list_modules())
    assert registered == {
        'collective_intelligence',
        'cultural_awareness',
        'social_intelligence',
        'strategy_evolution',
        'temporal_dynamics',
        'uncertainty_management',
    }
    suggestions = gm_modules.suggest_gm_modules({
        'Alice': {'theory_of_mind', 'uncertainty_aware'},
    })
    assert suggestions == ['social_intelligence', 'uncertainty_management']


def test_module_detection_uses_public_component_mapping_only() -> None:
    class PublicAgent:
        name = 'Alice'

        @staticmethod
        def get_all_context_components():
            return {
                'TheoryOfMind': object(),
                'UncertaintyAware': object(),
                'unrelated': object(),
            }

        def __getattr__(self, name):
            if name == '_context_components':
                raise AssertionError('private component storage was accessed')
            raise AttributeError(name)

    assert gm_modules.detect_agent_modules([PublicAgent()]) == {
        'Alice': {'theory_of_mind', 'uncertainty_aware'}
    }
