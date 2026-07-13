"""Contract tests for the one-call ultrafast negotiator profile."""

from __future__ import annotations

from contextlib import contextmanager
import json

import pytest

from concordia.agents import entity_agent_with_logging

import negotiation
from negotiation import minimal_negotiator


class RecordingModel:
    """Small Concordia model stub that records every requested generation."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events
        self.prompts: list[str] = []
        self.choice_prompts: list[str] = []

    def sample_text(self, prompt: str, **_: object) -> str:
        if self.events is not None:
            self.events.append('model')
        self.prompts.append(prompt)
        if 'red folder' in prompt:
            return 'acknowledge the red-folder constraint'
        return 'make a concise offer'

    def sample_choice(
        self,
        prompt: str,
        responses: tuple[str, ...] | list[str],
        **_: object,
    ) -> tuple[int, str, dict[str, float]]:
        if self.events is not None:
            self.events.append('model')
        self.choice_prompts.append(prompt)
        return 0, responses[0], {}


def test_exactly_one_final_action_call_is_scoped(
    memory_factory,
    action_spec,
) -> None:
    events: list[str] = []
    model = RecordingModel(events)

    @contextmanager
    def action_scope():
        events.append('scope_enter')
        try:
            yield
        finally:
            events.append('scope_exit')

    agent = minimal_negotiator.build_agent(
        model,
        memory_factory(),
        name='Alice',
        goal='Protect the renewal while keeping price below $80',
        action_call_scope_factory=action_scope,
    )
    agent.observe('Bob put the contract in the red folder.')

    action = agent.act(action_spec)

    assert isinstance(
        agent, entity_agent_with_logging.EntityAgentWithLogging
    )
    assert agent.agent_profile == minimal_negotiator.AGENT_PROFILE
    assert events == ['scope_enter', 'model', 'scope_exit']
    assert len(model.prompts) == 1
    assert model.choice_prompts == []
    assert action == 'Alice acknowledge the red-folder constraint'
    prompt = model.prompts[0]
    assert 'Bob put the contract in the red folder.' in prompt
    assert 'Protect the renewal while keeping price below $80' in prompt
    assert 'do not narrate hidden reasoning' in prompt


def test_observation_memory_is_isolated_between_instances(
    memory_factory,
    action_spec,
) -> None:
    alice_model = RecordingModel()
    bob_model = RecordingModel()
    alice = minimal_negotiator.build_agent(
        alice_model, memory_factory(), name='Alice'
    )
    bob = minimal_negotiator.build_agent(
        bob_model, memory_factory(), name='Bob'
    )

    alice.observe('Alice-only fact: use the red folder.')
    bob.observe('Bob-only fact: use the blue folder.')
    alice.act(action_spec)
    bob.act(action_spec)

    assert 'Alice-only fact' in alice_model.prompts[0]
    assert 'Bob-only fact' not in alice_model.prompts[0]
    assert 'Bob-only fact' in bob_model.prompts[0]
    assert 'Alice-only fact' not in bob_model.prompts[0]


def test_json_state_round_trip_preserves_next_action_behavior(
    memory_factory,
    action_spec,
) -> None:
    original_model = RecordingModel()
    original = minimal_negotiator.build_agent(
        original_model,
        memory_factory(),
        name='Alice',
        goal='Respect the red folder constraint',
    )
    original.observe('The red folder contains the binding budget limit.')
    serialized = json.loads(json.dumps(original.get_state()))

    restored_model = RecordingModel()
    restored = minimal_negotiator.build_agent(
        restored_model,
        memory_factory(),
        name='Alice',
        goal='Respect the red folder constraint',
    )
    restored.set_state(serialized)

    assert restored.act(action_spec) == original.act(action_spec)
    assert 'binding budget limit' in restored_model.prompts[0]
    assert restored.get_state() == original.get_state()


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'message'),
    [
        ({'name': ''}, ValueError, 'name must be a non-empty string'),
        ({'name': 7}, TypeError, 'name must be a string'),
        ({'goal': '   '}, ValueError, 'goal must be a non-empty string'),
        ({'goal': None}, TypeError, 'goal must be a string'),
        (
            {'modules': ['theory_of_mind']},
            ValueError,
            'does not support optional negotiation modules',
        ),
        ({'modules': 'theory_of_mind'}, TypeError, 'empty sequence or None'),
        ({'modules': set()}, TypeError, 'empty sequence or None'),
        (
            {'action_call_scope_factory': object()},
            TypeError,
            'must be callable or None',
        ),
    ],
)
def test_invalid_profile_configuration_fails_closed(
    model,
    memory_factory,
    kwargs,
    exception,
    message,
) -> None:
    with pytest.raises(exception, match=message):
        minimal_negotiator.build_agent(model, memory_factory(), **kwargs)


def test_invalid_model_and_memory_fail_during_build(memory_factory) -> None:
    with pytest.raises(TypeError, match='sample_text'):
        minimal_negotiator.build_agent(object(), memory_factory())
    with pytest.raises(TypeError, match='AssociativeMemoryBank'):
        minimal_negotiator.build_agent(RecordingModel(), object())


def test_malformed_scope_fails_before_model_call(
    memory_factory,
    action_spec,
) -> None:
    model = RecordingModel()
    agent = minimal_negotiator.build_agent(
        model,
        memory_factory(),
        action_call_scope_factory=lambda: object(),
    )

    with pytest.raises(TypeError, match='synchronous context manager'):
        agent.act(action_spec)

    assert model.prompts == []
    assert model.choice_prompts == []


def test_act_component_disables_choice_randomization(memory_factory) -> None:
    agent = minimal_negotiator.build_agent(
        RecordingModel(),
        memory_factory(),
    )

    assert agent.get_state()['act_component']['randomize_choices'] is False


def test_profile_has_public_package_aliases() -> None:
    assert negotiation.build_ultrafast_agent is minimal_negotiator.build_agent
    assert (
        negotiation.ULTRAFAST_AGENT_PROFILE
        == minimal_negotiator.AGENT_PROFILE
        == 'ultrafast_minimal/1'
    )
