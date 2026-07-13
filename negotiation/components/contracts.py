"""Declarative diagnostics contracts for optional cognitive components."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclasses.dataclass(frozen=True)
class ModelCallDeclaration:
    """Executable lower/upper bound for one lifecycle phase's model calls."""

    minimum: int
    maximum: int
    condition: str = 'fixed'

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.minimum, self.maximum)
        ):
            raise TypeError('Model-call bounds must be integers.')
        if self.minimum < 0 or self.maximum < self.minimum:
            raise ValueError('Model-call bounds must be ordered and non-negative.')
        if not isinstance(self.condition, str) or not self.condition:
            raise ValueError('Model-call condition must be non-empty.')

    @classmethod
    def exact(cls, count: int) -> 'ModelCallDeclaration':
        return cls(count, count)

    def validate(self, observed_calls: int) -> None:
        if isinstance(observed_calls, bool) or not isinstance(observed_calls, int):
            raise TypeError('Observed model-call count must be an integer.')
        if not self.minimum <= observed_calls <= self.maximum:
            raise ValueError(
                f'Observed {observed_calls} model calls outside declared range '
                f'[{self.minimum}, {self.maximum}] ({self.condition}).'
            )

    def to_dict(self) -> dict[str, object]:
        return {
            'minimum': self.minimum,
            'maximum': self.maximum,
            'condition': self.condition,
        }


@dataclasses.dataclass(frozen=True)
class ComponentDiagnosticContract:
    """Auditable lifecycle and telemetry promises made by a component."""

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    state_fields: tuple[str, ...]
    extra_model_calls: Mapping[str, int | ModelCallDeclaration]
    logging_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        required_phases = {'pre_act', 'post_act', 'pre_observe', 'post_observe'}
        if not required_phases.issubset(self.extra_model_calls):
            missing = sorted(required_phases.difference(self.extra_model_calls))
            raise ValueError(f'Missing model-call declarations: {missing}')
        for field_name in ('inputs', 'outputs', 'state_fields', 'logging_fields'):
            values = tuple(getattr(self, field_name))
            if not values or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f'{field_name} must contain non-empty names.')
            if len(values) != len(set(values)):
                raise ValueError(f'{field_name} cannot contain duplicate names.')
            object.__setattr__(self, field_name, values)
        model_calls = {}
        for phase, value in self.extra_model_calls.items():
            if not isinstance(phase, str) or not phase:
                raise ValueError('Model-call phase names must be non-empty.')
            if isinstance(value, bool):
                raise TypeError('Model-call declarations cannot be booleans.')
            if isinstance(value, int):
                value = ModelCallDeclaration.exact(value)
            if not isinstance(value, ModelCallDeclaration):
                raise TypeError(
                    'Model-call declarations must be counts or call ranges.'
                )
            model_calls[phase] = value
        object.__setattr__(
            self,
            'extra_model_calls',
            MappingProxyType(model_calls),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        return {
            'inputs': list(self.inputs),
            'outputs': list(self.outputs),
            'state_fields': list(self.state_fields),
            'extra_model_calls': {
                phase: declaration.to_dict()
                for phase, declaration in self.extra_model_calls.items()
            },
            'logging_fields': list(self.logging_fields),
        }

    def validate_model_calls(self, phase: str, observed_calls: int) -> None:
        """Fail if an observed branch exceeds its declared lifecycle range."""
        try:
            declaration = self.extra_model_calls[phase]
        except KeyError as error:
            raise ValueError(f'Unknown lifecycle phase: {phase}') from error
        declaration.validate(observed_calls)


class ModelCallDiagnosticsMixin:
    """Branch-level call instrumentation shared by conditional components."""

    DIAGNOSTIC_CONTRACT: ComponentDiagnosticContract

    def _record_model_call_branch(
        self,
        phase: str,
        *,
        observed_calls: int,
        expected_calls: int,
        branch: str,
    ) -> None:
        self.DIAGNOSTIC_CONTRACT.validate_model_calls(phase, observed_calls)
        if observed_calls != expected_calls:
            raise RuntimeError(
                f'{type(self).__name__} {phase} branch {branch!r} expected '
                f'{expected_calls} model calls but observed {observed_calls}.'
            )
        diagnostics = dict(getattr(self, '_model_call_diagnostics', {}))
        diagnostics[phase] = {
            'branch': branch,
            'observed_calls': observed_calls,
            'expected_calls': expected_calls,
        }
        self._model_call_diagnostics = diagnostics

    def get_model_call_diagnostics(self) -> dict[str, dict[str, object]]:
        """Return the most recent instrumented branch for each lifecycle phase."""
        return {
            phase: dict(values)
            for phase, values in getattr(
                self, '_model_call_diagnostics', {}
            ).items()
        }


@runtime_checkable
class StatefulComponent(Protocol):
    """Structural state contract shared by cognitive components."""

    DIAGNOSTIC_CONTRACT: ComponentDiagnosticContract

    def get_state(self) -> Mapping[str, Any]:
        """Return a JSON-compatible behavior-restoring snapshot."""

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`get_state`."""
