"""Scientific and lifecycle contracts for evidence-linked ToM diagnostics."""

from __future__ import annotations

import copy
import json

import pytest

from negotiation.components.theory_of_mind import MentalModel, TheoryOfMind


def _diagnostic(component: TheoryOfMind, counterpart: str = 'Bob'):
    return component.get_counterpart_diagnostics()[counterpart]


def test_observation_analysis_owns_updates_and_reads_are_pure(
    model,
    action_spec,
) -> None:
    component = TheoryOfMind(model)
    before = copy.deepcopy(component.get_state())

    component.pre_observe('Bob: We can offer $80 by Friday.')
    assert len(model.prompts) == 0
    component.post_observe()
    assert len(model.prompts) == 3
    component.post_observe()
    assert len(model.prompts) == 3

    state = copy.deepcopy(component.get_state())
    first = component.pre_act(action_spec)
    second = component.pre_act(action_spec)
    first_diagnostics = component.get_counterpart_diagnostics()
    second_diagnostics = component.get_counterpart_diagnostics()

    assert before != state
    assert first == second
    assert first_diagnostics == second_diagnostics
    assert component.get_state() == state
    assert len(model.prompts) == 3


def test_changed_observation_changes_beliefs_advice_and_trust(model) -> None:
    component = TheoryOfMind(model)
    component.analyze_observation('Bob: I do not trust this unreliable process.')
    first = copy.deepcopy(_diagnostic(component))

    component.analyze_observation('Bob: I trust your reliable process.')
    second = _diagnostic(component)

    assert first['beliefs'][0]['proposition'] != second['beliefs'][0]['proposition']
    assert first['advice'] != second['advice']
    assert first['trust']['available'] is True
    assert second['trust']['available'] is True
    assert first['trust']['value'] < second['trust']['value']
    assert second['trust']['evidence']


def test_higher_order_beliefs_are_unknown_without_epistemic_evidence(model) -> None:
    component = TheoryOfMind(model, max_recursion_depth=3)
    component.analyze_observation('Bob: The delivery date is Friday.')

    beliefs = _diagnostic(component)['beliefs']
    assert beliefs[0]['available'] is True
    for belief in beliefs[1:]:
        assert belief['available'] is False
        assert belief['proposition'] is None
        assert belief['confidence'] is None
        assert belief['evidence'] == []
        assert belief['method'] == 'insufficient_evidence/v1'


def test_trust_absence_stays_none_until_evidence_arrives(model) -> None:
    component = TheoryOfMind(model)
    component.analyze_observation('Bob: The package price is $80.')
    absent = _diagnostic(component)['trust']
    assert absent == {
        'value': None,
        'available': False,
        'method': 'no_trust_evidence/v1',
        'evidence': [],
    }

    component.analyze_observation('Bob: I trust your dependable team.')
    observed = _diagnostic(component)['trust']
    assert observed['available'] is True
    assert observed['value'] is not None
    assert observed['method'] == 'lexical_trust_cues/v1'
    assert observed['evidence']


def test_state_json_round_trip_restores_next_render_without_model_call(
    model,
    action_spec,
) -> None:
    component = TheoryOfMind(model)
    component.analyze_observation('Bob: I think you believe delivery is Friday.')
    expected_render = component.pre_act(action_spec)
    expected_diagnostics = component.get_counterpart_diagnostics()
    snapshot = json.loads(json.dumps(component.get_state()))

    restored_model = type(model)()
    restored = TheoryOfMind(restored_model)
    restored.set_state(snapshot)

    assert restored.get_state() == snapshot
    assert restored.pre_act(action_spec) == expected_render
    assert restored.get_counterpart_diagnostics() == expected_diagnostics
    assert restored_model.prompts == []


def test_public_diagnostics_exclude_self_even_from_legacy_state(model) -> None:
    component = TheoryOfMind(model)
    component.analyze_observation('Bob: We can deliver Friday.')
    bob = component._mental_models['Bob']
    component._mental_models['self'] = MentalModel(
        counterpart_id='self',
        goals={},
        personality_traits={},
        emotional_state=copy.deepcopy(bob.emotional_state),
        strategies={},
        constraints=[],
        deception_indicators={'over_certainty': 1.0},
        trust=copy.deepcopy(bob.trust),
        evidence=['own output'],
        advice='own output',
        last_updated='current',
    )

    assert set(component.get_counterpart_diagnostics()) == {'Bob'}
    assert set(component.get_state()['mental_models']) == {'Bob'}


def test_deception_baseline_calibrates_repeated_cues(model) -> None:
    component = TheoryOfMind(model)
    statement = 'Honestly, this is absolutely unquestionably correct.'
    raw = component._raw_deception_cues(statement)
    component._baseline_patterns = dict(raw)

    assert component._detect_deception(statement) == pytest.approx(
        {name: 0.0 for name in raw}
    )
