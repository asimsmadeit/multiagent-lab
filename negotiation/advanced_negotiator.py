"""Advanced negotiation agent prefab with modular enhancements."""

from collections.abc import Mapping
import dataclasses
import json
from typing import Any, Optional, Sequence

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

# Import base negotiator and advanced components
from negotiation import base_negotiator
from negotiation.components import cultural_adaptation
from negotiation.components import temporal_strategy
from negotiation.components import swarm_intelligence
from negotiation.components import uncertainty_aware
from negotiation.components import strategy_evolution
from negotiation.components import theory_of_mind
from negotiation.components.config import resolve_module_configs
from negotiation.constants import MODULE_COMPONENT_NAMES, ModuleType
from negotiation.profiles import AgentProfile


AGENT_PROFILE = AgentProfile.ADVANCED.value


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
    """An advanced negotiation agent with optional enhancement modules.

    This prefab extends the base negotiator with advanced capabilities
    that can be enabled/disabled through configuration.

    Available modules:
    - cultural_adaptation: Adapt to different cultural negotiation styles
    - temporal_strategy: Multi-horizon planning and relationship management
    - swarm_intelligence: Collective decision-making through specialized sub-agents
    - uncertainty_aware: Probabilistic reasoning under incomplete information
    - strategy_evolution: Meta-learning and continual adaptation across negotiations
    - theory_of_mind: Emotional intelligence and recursive reasoning
    """

    description: str = (
        'An advanced negotiation agent with modular enhancements. '
        'Supports cultural adaptation, temporal planning, collective '
        'intelligence, and other sophisticated negotiation capabilities.'
    )

    params: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {
        'name': 'AdvancedNegotiator',
        'goal': 'Achieve optimal negotiation outcomes',
        'negotiation_style': 'integrative',
        'reservation_value': '0.0',
        'ethical_constraints': 'Be honest and fair. Respect cultural differences.',
        'modules': '',  # Comma-separated list of module names
        'module_configs': '{}',  # JSON string of module configurations
        'trial_seed': 0,
        'extra_components': {},
    })

    def build(
        self,
        model: language_model.LanguageModel,
        memory_bank: basic_associative_memory.AssociativeMemoryBank
    ) -> entity_agent_with_logging.EntityAgentWithLogging:
        """Build the advanced negotiation agent with selected modules.

        Args:
            model: Language model for reasoning
            memory_bank: Memory bank for storing experiences

        Returns:
            Configured advanced negotiation agent
        """
        # Parse module list from params
        modules_value = self.params.get('modules', '')
        if isinstance(modules_value, str):
            module_names = [
                module.strip() for module in modules_value.split(',')
                if module.strip()
            ]
        elif isinstance(modules_value, Sequence):
            module_names = [
                module.value if isinstance(module, ModuleType) else str(module).strip()
                for module in modules_value
            ]
        else:
            raise TypeError('modules must be a comma-separated string or sequence.')
        try:
            modules = tuple(ModuleType(name) for name in module_names)
        except ValueError as error:
            unknown_modules = set(module_names).difference(
                module.value for module in ModuleType
            )
            raise ValueError(
                f'Unknown negotiation modules: {sorted(unknown_modules)}'
            ) from error
        if len(set(modules)) != len(modules):
            raise ValueError('Negotiation modules cannot contain duplicates.')

        # Parse module configs from JSON string
        module_configs_value = self.params.get('module_configs', '{}')
        if isinstance(module_configs_value, Mapping):
            module_configs = dict(module_configs_value)
        elif isinstance(module_configs_value, str):
            try:
                module_configs = json.loads(module_configs_value)
            except json.JSONDecodeError as error:
                raise ValueError('module_configs must contain valid JSON.') from error
        else:
            raise TypeError('module_configs must be a mapping or JSON string.')
        if not isinstance(module_configs, dict):
            raise ValueError('module_configs JSON must decode to an object.')
        trial_seed = self.params.get('trial_seed')
        resolved_configs, seed_provenance = resolve_module_configs(
            modules,
            module_configs,
            trial_seed=trial_seed,
            actor_id=str(self.params.get('name', 'AdvancedNegotiator')),
        )

        # Build extra components for selected modules
        extra_components = {}

        # Helper to check module membership (works with strings or ModuleType)
        def has_module(module_type: ModuleType) -> bool:
            return module_type in modules

        # Add selected modules
        if has_module(ModuleType.CULTURAL_ADAPTATION):
            config = resolved_configs[ModuleType.CULTURAL_ADAPTATION]
            cultural = cultural_adaptation.CulturalAdaptation(
                model=model,
                **dataclasses.asdict(config),
            )
            extra_components[MODULE_COMPONENT_NAMES[ModuleType.CULTURAL_ADAPTATION]] = cultural

        if has_module(ModuleType.TEMPORAL_STRATEGY):
            config = resolved_configs[ModuleType.TEMPORAL_STRATEGY]
            temporal = temporal_strategy.TemporalStrategy(
                model=model,
                **dataclasses.asdict(config),
            )
            extra_components[MODULE_COMPONENT_NAMES[ModuleType.TEMPORAL_STRATEGY]] = temporal

        if has_module(ModuleType.SWARM_INTELLIGENCE):
            config = resolved_configs[ModuleType.SWARM_INTELLIGENCE]
            swarm = swarm_intelligence.SwarmIntelligence(
                model=model,
                **dataclasses.asdict(config),
            )
            extra_components[MODULE_COMPONENT_NAMES[ModuleType.SWARM_INTELLIGENCE]] = swarm

        if has_module(ModuleType.UNCERTAINTY_AWARE):
            config = resolved_configs[ModuleType.UNCERTAINTY_AWARE]
            uncertainty = uncertainty_aware.UncertaintyAware(
                model=model,
                **dataclasses.asdict(config),
                seed_provenance=seed_provenance[ModuleType.UNCERTAINTY_AWARE],
            )
            extra_components[MODULE_COMPONENT_NAMES[ModuleType.UNCERTAINTY_AWARE]] = uncertainty

        if has_module(ModuleType.STRATEGY_EVOLUTION):
            config = resolved_configs[ModuleType.STRATEGY_EVOLUTION]
            evolution = strategy_evolution.StrategyEvolution(
                model=model,
                **dataclasses.asdict(config),
                seed_provenance=seed_provenance[ModuleType.STRATEGY_EVOLUTION],
            )
            extra_components[MODULE_COMPONENT_NAMES[ModuleType.STRATEGY_EVOLUTION]] = evolution

        if has_module(ModuleType.THEORY_OF_MIND):
            config = resolved_configs[ModuleType.THEORY_OF_MIND]
            tom = theory_of_mind.TheoryOfMind(
                model=model,
                **dataclasses.asdict(config),
            )
            extra_components[MODULE_COMPONENT_NAMES[ModuleType.THEORY_OF_MIND]] = tom

        # Update params to include extra components
        enhanced_params = dict(self.params)
        enhanced_params['extra_components'] = extra_components

        # Create base negotiator with enhanced params
        enhanced_prefab = base_negotiator.Entity()
        enhanced_prefab.params = enhanced_params

        return enhanced_prefab.build(model, memory_bank)


def build_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    name: str = 'AdvancedNegotiator',
    goal: str = 'Achieve optimal negotiation outcomes',
    negotiation_style: str = 'integrative',
    reservation_value: float = 0.0,
    ethical_constraints: str = 'Be honest and fair. Respect cultural differences.',
    modules: Optional[Sequence[str | ModuleType]] = None,
    module_configs: Optional[Mapping[str, Any]] = None,
    trial_seed: int | None = 0,
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Convenience function to build an advanced negotiation agent.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        name: Name of the negotiation agent
        goal: Primary negotiation goal
        negotiation_style: Style of negotiation ('cooperative', 'competitive', 'integrative')
        reservation_value: Minimum acceptable value
        ethical_constraints: Ethical guidelines for negotiation
        modules: List of module names to enable (e.g., ['cultural_adaptation', 'theory_of_mind'])
        module_configs: Dictionary of module-specific configurations
        **kwargs: Additional parameters for the agent
        
    Returns:
        Configured advanced negotiation agent
        
    Available modules:
        - 'cultural_adaptation': Adapt to different cultural negotiation styles
        - 'temporal_strategy': Multi-horizon planning and relationship management
        - 'swarm_intelligence': Collective decision-making through specialized sub-agents
        - 'uncertainty_aware': Probabilistic reasoning under incomplete information
        - 'strategy_evolution': Meta-learning and continual adaptation
        - 'theory_of_mind': Emotional intelligence and recursive reasoning
        
    Example:
        ```python
        agent = build_agent(
            model=my_model,
            memory_bank=my_memory,
            name="Sophie",
            goal="Negotiate international trade agreement",
            modules=['cultural_adaptation', 'theory_of_mind'],
            module_configs={
                'cultural_adaptation': {'own_culture': 'western_business'},
                'theory_of_mind': {'max_recursion_depth': 2}
            }
        )
        ```
    """
    if modules is None:
        modules = []
    if module_configs is None:
        module_configs = {}
    
    params = {
        'name': name,
        'goal': goal,
        'negotiation_style': negotiation_style,
        'reservation_value': str(reservation_value),
        'ethical_constraints': ethical_constraints,
        'modules': ','.join(
            module.value if isinstance(module, ModuleType) else module
            for module in modules
        ),
        'module_configs': dict(module_configs),
        'trial_seed': trial_seed,
    }
    params.update(kwargs)
    
    prefab = Entity(params=params)
    return prefab.build(model=model, memory_bank=memory_bank)


def build_cultural_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    name: str = 'CulturalNegotiator',
    own_culture: str = 'western_business',
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent optimized for cross-cultural negotiations.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        name: Name of the negotiation agent
        own_culture: Agent's cultural background ('western_business', 'east_asian', etc.)
        **kwargs: Additional parameters for build_agent
        
    Returns:
        Agent with cultural adaptation capabilities
    """
    return build_agent(
        model=model,
        memory_bank=memory_bank,
        name=name,
        modules=['cultural_adaptation', 'theory_of_mind'],
        module_configs={
            'cultural_adaptation': {'own_culture': own_culture},
            'theory_of_mind': {'emotion_sensitivity': 0.8}
        },
        **kwargs
    )


def build_temporal_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    name: str = 'TemporalNegotiator',
    discount_factor: float = 0.9,
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent optimized for long-term relationship management.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        name: Name of the negotiation agent
        discount_factor: How much to value future outcomes (0-1)
        **kwargs: Additional parameters for build_agent
        
    Returns:
        Agent with temporal strategy capabilities
    """
    return build_agent(
        model=model,
        memory_bank=memory_bank,
        name=name,
        modules=['temporal_strategy', 'theory_of_mind'],
        module_configs={
            'temporal_strategy': {'discount_factor': discount_factor},
        },
        **kwargs
    )


def build_collective_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    name: str = 'CollectiveNegotiator',
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent optimized for multi-party negotiations.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        name: Name of the negotiation agent
        **kwargs: Additional parameters for build_agent
        
    Returns:
        Agent with swarm intelligence capabilities
    """
    return build_agent(
        model=model,
        memory_bank=memory_bank,
        name=name,
        modules=['swarm_intelligence', 'uncertainty_aware'],
        module_configs={
            'swarm_intelligence': {'consensus_threshold': 0.7},
        },
        **kwargs
    )


def build_adaptive_agent(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    name: str = 'AdaptiveNegotiator',
    learning_rate: float = 0.01,
    **kwargs
) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an agent that learns and adapts strategies over time.
    
    Args:
        model: Language model for reasoning
        memory_bank: Memory bank for storing experiences
        name: Name of the negotiation agent
        learning_rate: How quickly to adapt strategies (0-1)
        **kwargs: Additional parameters for build_agent
        
    Returns:
        Agent with strategy evolution capabilities
    """
    return build_agent(
        model=model,
        memory_bank=memory_bank,
        name=name,
        modules=['strategy_evolution', 'uncertainty_aware'],
        module_configs={
            'strategy_evolution': {'learning_rate': learning_rate},
        },
        **kwargs
    )
