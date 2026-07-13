"""Minimal negotiator with one scoped model call per action.

This builder is intended for high-throughput experiments that need the acting
model's generation call to be the only model-dependent work performed by an
agent.  It deliberately excludes optional reasoning modules and retrieval that
would make additional model calls.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Any, cast

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.language_model import language_model

from negotiation.components import scoped_act
from negotiation.profiles import AgentProfile


AGENT_PROFILE = AgentProfile.ULTRAFAST_MINIMAL.value

_INSTRUCTIONS_COMPONENT = 'UltrafastNegotiationInstructions'
_GOAL_COMPONENT = 'UltrafastNegotiationGoal'
_OBSERVATION_TO_MEMORY_COMPONENT = 'observation_to_memory'
_OBSERVATION_COMPONENT = 'observation'
_OBSERVATION_HISTORY_LENGTH = 100


class UltrafastNegotiatorAgent(
    entity_agent_with_logging.EntityAgentWithLogging
):
    """Official Concordia entity tagged with its immutable profile identity."""

    agent_profile = AGENT_PROFILE


def _nonempty_text(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f'{name} must be a string.')
    normalized = value.strip()
    if not normalized:
        raise ValueError(f'{name} must be a non-empty string.')
    return normalized


def _validate_modules(modules: Sequence[str] | None) -> None:
    if modules is None:
        return
    if isinstance(modules, (str, bytes)) or not isinstance(modules, Sequence):
        raise TypeError('modules must be an empty sequence or None.')
    if modules:
        raise ValueError(
            f'{AGENT_PROFILE} does not support optional negotiation modules.'
        )


def _validating_scope_factory(
    factory: scoped_act.ActionScopeFactory,
) -> scoped_act.ActionScopeFactory:
    """Reject a malformed context manager before final generation starts."""

    def make_scope() -> AbstractContextManager[Any]:
        scope = factory()
        enter = getattr(scope, '__enter__', None)
        exit_method = getattr(scope, '__exit__', None)
        if not callable(enter) or not callable(exit_method):
            raise TypeError(
                'action_call_scope_factory must return a synchronous '
                'context manager.'
            )
        return cast(AbstractContextManager[Any], scope)

    return make_scope


def build_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    *,
    name: str = 'Negotiator',
    goal: str = 'Reach a mutually beneficial agreement',
    modules: Sequence[str] | None = (),
    action_call_scope_factory: scoped_act.ActionScopeFactory | None = None,
) -> UltrafastNegotiatorAgent:
    """Build the deterministic ``ultrafast_minimal/1`` agent profile.

    The context contains only constant instructions, a constant goal, recent
    observations, and their backing memory.  None of those components receives
    a language model, so a successful ``act`` performs exactly one model call:
    the final call made by :class:`ScopedConcatActComponent`.

    Args:
        model: Acting language model implementing Concordia's model protocol.
        memory_bank: Dedicated associative memory bank for this agent.
        name: Stable actor identity used in prompts and action prefixes.
        goal: Stable negotiation objective.
        modules: Must be empty; this profile has no optional reasoning modules.
        action_call_scope_factory: Optional factory for a fresh synchronous
            context manager around each final acting-model call.

    Returns:
        An official Concordia entity whose ``agent_profile`` attribute is
        ``ultrafast_minimal/1`` and whose state uses Concordia's normal
        ``get_state``/``set_state`` contract.
    """
    agent_name = _nonempty_text('name', name)
    agent_goal = _nonempty_text('goal', goal)
    _validate_modules(modules)
    if not isinstance(
        memory_bank,
        basic_associative_memory.AssociativeMemoryBank,
    ):
        raise TypeError('memory_bank must be an AssociativeMemoryBank.')
    if not callable(getattr(model, 'sample_text', None)):
        raise TypeError('model must provide a callable sample_text method.')
    if not callable(getattr(model, 'sample_choice', None)):
        raise TypeError('model must provide a callable sample_choice method.')
    if action_call_scope_factory is not None and not callable(
        action_call_scope_factory
    ):
        raise TypeError('action_call_scope_factory must be callable or None.')

    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)
    components = {
        _INSTRUCTIONS_COMPONENT: agent_components.constant.Constant(
            state=(
                f'You are {agent_name}. Negotiate directly and return only '
                'your next negotiation action; do not narrate hidden reasoning.'
            ),
            pre_act_label='Negotiation instructions',
        ),
        _GOAL_COMPONENT: agent_components.constant.Constant(
            state=agent_goal,
            pre_act_label='Negotiation goal',
        ),
        _OBSERVATION_TO_MEMORY_COMPONENT: (
            agent_components.observation.ObservationToMemory()
        ),
        _OBSERVATION_COMPONENT: agent_components.observation.LastNObservations(
            history_length=_OBSERVATION_HISTORY_LENGTH,
            pre_act_label='Recent negotiation observations',
        ),
        agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY: memory,
    }
    component_order = (
        _INSTRUCTIONS_COMPONENT,
        _GOAL_COMPONENT,
        _OBSERVATION_COMPONENT,
        _OBSERVATION_TO_MEMORY_COMPONENT,
        agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
    )
    scope_factory = (
        _validating_scope_factory(action_call_scope_factory)
        if action_call_scope_factory is not None
        else None
    )
    act_component = scoped_act.ScopedConcatActComponent(
        model=model,
        component_order=component_order,
        prefix_entity_name=True,
        randomize_choices=False,
        action_scope_factory=scope_factory,
    )
    agent = UltrafastNegotiatorAgent(
        agent_name=agent_name,
        act_component=act_component,
        context_components=components,
    )
    return agent


build_minimal_agent = build_agent


__all__ = [
    'AGENT_PROFILE',
    'UltrafastNegotiatorAgent',
    'build_agent',
    'build_minimal_agent',
]
