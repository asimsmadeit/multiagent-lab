"""Act component with an injectable scope around final action generation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from concordia.components.agent import concat_act_component
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


ActionScopeFactory = Callable[[], AbstractContextManager[Any]]


class ScopedConcatActComponent(concat_act_component.ConcatActComponent):
    """Scope only the final acting-model call, after component context reads."""

    def __init__(
        self,
        model: language_model.LanguageModel,
        component_order: Sequence[str] | None = None,
        prefix_entity_name: bool = True,
        randomize_choices: bool = True,
        action_scope_factory: ActionScopeFactory | None = None,
    ) -> None:
        super().__init__(
            model=model,
            component_order=component_order,
            prefix_entity_name=prefix_entity_name,
            randomize_choices=randomize_choices,
        )
        self._action_scope_factory = action_scope_factory

    def get_action_attempt(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        """Run final generation in the injected call scope, if configured."""
        scope = (
            self._action_scope_factory()
            if self._action_scope_factory is not None
            else nullcontext()
        )
        with scope:
            return super().get_action_attempt(contexts, action_spec)
