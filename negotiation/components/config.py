"""Typed, strict configuration contracts for optional agent components."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from negotiation.constants import ModuleType


def _unit_interval(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{name} must be numeric.')
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f'{name} must be between 0 and 1.')


def _optional_seed(seed: int | None) -> None:
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError('seed must be a non-negative integer or None.')


@dataclasses.dataclass(frozen=True)
class CulturalAdaptationConfig:
    own_culture: str = 'western_business'
    adaptation_level: float = 0.7
    detect_culture: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.own_culture, str) or not self.own_culture:
            raise ValueError('own_culture must be a non-empty string.')
        _unit_interval('adaptation_level', self.adaptation_level)
        if type(self.detect_culture) is not bool:
            raise TypeError('detect_culture must be a boolean.')


@dataclasses.dataclass(frozen=True)
class TemporalStrategyConfig:
    discount_factor: float = 0.9
    reputation_weight: float = 0.3
    relationship_investment_threshold: float = 0.6

    def __post_init__(self) -> None:
        _unit_interval('discount_factor', self.discount_factor)
        _unit_interval('reputation_weight', self.reputation_weight)
        _unit_interval(
            'relationship_investment_threshold',
            self.relationship_investment_threshold,
        )


@dataclasses.dataclass(frozen=True)
class SwarmIntelligenceConfig:
    consensus_threshold: float = 0.7
    max_iterations: int = 3
    enable_sub_agents: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _unit_interval('consensus_threshold', self.consensus_threshold)
        if type(self.max_iterations) is not int or self.max_iterations < 1:
            raise ValueError('max_iterations must be a positive integer.')
        if self.enable_sub_agents is not None:
            if not isinstance(self.enable_sub_agents, Sequence) or isinstance(
                self.enable_sub_agents, (str, bytes)
            ):
                raise TypeError('enable_sub_agents must be an array of names or None.')
            names = tuple(self.enable_sub_agents)
            if any(not isinstance(name, str) or not name for name in names):
                raise ValueError('enable_sub_agents must contain non-empty names.')
            if len(set(names)) != len(names):
                raise ValueError('enable_sub_agents cannot contain duplicates.')
            object.__setattr__(self, 'enable_sub_agents', names)


@dataclasses.dataclass(frozen=True)
class UncertaintyAwareConfig:
    confidence_threshold: float = 0.7
    risk_tolerance: float = 0.3
    information_gathering_budget: float = 0.1
    seed: int | None = None

    def __post_init__(self) -> None:
        _unit_interval('confidence_threshold', self.confidence_threshold)
        _unit_interval('risk_tolerance', self.risk_tolerance)
        _unit_interval(
            'information_gathering_budget',
            self.information_gathering_budget,
        )
        _optional_seed(self.seed)


@dataclasses.dataclass(frozen=True)
class StrategyEvolutionConfig:
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    learning_rate: float = 0.01
    seed: int | None = None

    def __post_init__(self) -> None:
        if type(self.population_size) is not int or self.population_size < 1:
            raise ValueError('population_size must be a positive integer.')
        _unit_interval('mutation_rate', self.mutation_rate)
        _unit_interval('crossover_rate', self.crossover_rate)
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(float(self.learning_rate))
            or self.learning_rate <= 0
        ):
            raise ValueError('learning_rate must be positive and finite.')
        _optional_seed(self.seed)


@dataclasses.dataclass(frozen=True)
class TheoryOfMindModuleConfig:
    max_recursion_depth: int = 3
    emotion_sensitivity: float = 0.7
    empathy_level: float = 0.8
    tom_version: str = 'v1'

    def __post_init__(self) -> None:
        if type(self.max_recursion_depth) is not int or self.max_recursion_depth < 0:
            raise ValueError('max_recursion_depth must be a non-negative integer.')
        _unit_interval('emotion_sensitivity', self.emotion_sensitivity)
        _unit_interval('empathy_level', self.empathy_level)
        if self.tom_version not in ('v1', 'v2'):
            raise ValueError("tom_version must be 'v1' or 'v2'.")


ModuleConfig = (
    CulturalAdaptationConfig
    | TemporalStrategyConfig
    | SwarmIntelligenceConfig
    | UncertaintyAwareConfig
    | StrategyEvolutionConfig
    | TheoryOfMindModuleConfig
)


MODULE_CONFIG_TYPES: Mapping[ModuleType, type[Any]] = {
    ModuleType.CULTURAL_ADAPTATION: CulturalAdaptationConfig,
    ModuleType.TEMPORAL_STRATEGY: TemporalStrategyConfig,
    ModuleType.SWARM_INTELLIGENCE: SwarmIntelligenceConfig,
    ModuleType.UNCERTAINTY_AWARE: UncertaintyAwareConfig,
    ModuleType.STRATEGY_EVOLUTION: StrategyEvolutionConfig,
    ModuleType.THEORY_OF_MIND: TheoryOfMindModuleConfig,
}


def _derived_seed(
    trial_seed: int,
    actor_id: str,
    module_type: ModuleType,
) -> int:
    material = f'{trial_seed}:{actor_id}:{module_type.value}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(material).digest()[:8], 'big')


def resolve_module_configs(
    enabled_modules: Sequence[ModuleType],
    raw_configs: Mapping[str | ModuleType, Mapping[str, Any] | ModuleConfig],
    *,
    trial_seed: int | None,
    actor_id: str,
) -> tuple[dict[ModuleType, Any], dict[ModuleType, dict[str, Any]]]:
    """Construct every enabled config and reject ignored or misspelled keys."""
    _optional_seed(trial_seed)
    if not isinstance(actor_id, str) or not actor_id:
        raise ValueError('actor_id must be a non-empty string.')
    enabled = tuple(enabled_modules)
    normalized: dict[ModuleType, Mapping[str, Any] | ModuleConfig] = {}
    for raw_name, raw_config in raw_configs.items():
        try:
            module_type = (
                raw_name if isinstance(raw_name, ModuleType)
                else ModuleType(raw_name)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f'Unknown module configuration: {raw_name!r}') from error
        if module_type not in enabled:
            raise ValueError(
                f'Configuration supplied for disabled module: {module_type.value}'
            )
        config_type = MODULE_CONFIG_TYPES[module_type]
        if not isinstance(raw_config, (Mapping, config_type)):
            raise ValueError(
                f'{module_type.value} configuration must be an object or '
                f'{config_type.__name__}.'
            )
        normalized[module_type] = raw_config

    resolved: dict[ModuleType, Any] = {}
    provenance: dict[ModuleType, dict[str, Any]] = {}
    for module_type in enabled:
        config_type = MODULE_CONFIG_TYPES[module_type]
        raw_config = normalized.get(module_type, {})
        if isinstance(raw_config, config_type):
            config = raw_config
        else:
            values = dict(raw_config)
            allowed = {field.name for field in dataclasses.fields(config_type)}
            unknown = set(values).difference(allowed)
            if unknown:
                raise ValueError(
                    f'Unknown {module_type.value} configuration keys: '
                    f'{sorted(unknown)}'
                )
            try:
                config = config_type(**values)
            except TypeError as error:
                raise TypeError(
                    f'Invalid {module_type.value} configuration: {error}'
                ) from error
        if hasattr(config, 'seed'):
            explicit_seed = getattr(config, 'seed')
            if explicit_seed is not None:
                seed_source = 'module_config'
                resolved_seed = explicit_seed
            elif trial_seed is not None:
                seed_source = 'trial_seed_derived'
                resolved_seed = _derived_seed(trial_seed, actor_id, module_type)
                config = dataclasses.replace(config, seed=resolved_seed)
            else:
                raise ValueError(
                    f'{module_type.value} requires an explicit module seed or '
                    'trial_seed.'
                )
            provenance[module_type] = {
                'source': seed_source,
                'trial_seed': trial_seed,
                'resolved_seed': resolved_seed,
                'module': module_type.value,
                'actor_id': actor_id,
            }
        resolved[module_type] = config
    return resolved, provenance


__all__ = [
    'CulturalAdaptationConfig',
    'ModuleConfig',
    'MODULE_CONFIG_TYPES',
    'StrategyEvolutionConfig',
    'SwarmIntelligenceConfig',
    'TemporalStrategyConfig',
    'TheoryOfMindModuleConfig',
    'UncertaintyAwareConfig',
    'resolve_module_configs',
]
