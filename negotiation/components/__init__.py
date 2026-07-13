"""Negotiation components for modular agent construction.

This module provides cognitive enhancement components for negotiation agents.

Usage:
    from negotiation.components import (
        TheoryOfMind,
        CulturalAdaptation,
        TemporalStrategy,
    )

    tom = TheoryOfMind(model=model)
    cultural = CulturalAdaptation(model=model, own_culture='western_business')
"""

# Base components
from negotiation.components import negotiation_instructions
from negotiation.components import negotiation_memory
from negotiation.components import negotiation_strategy
from negotiation.components import scoped_act

# Advanced modules
from negotiation.components import cultural_adaptation
from negotiation.components import strategy_evolution
from negotiation.components import swarm_intelligence
from negotiation.components import temporal_strategy
from negotiation.components import theory_of_mind
from negotiation.components import uncertainty_aware
from negotiation.components.contracts import (
    ComponentDiagnosticContract,
    ModelCallDeclaration,
    ModelCallDiagnosticsMixin,
    StatefulComponent,
)
from negotiation.components.config import (
    CulturalAdaptationConfig,
    StrategyEvolutionConfig,
    SwarmIntelligenceConfig,
    TemporalStrategyConfig,
    TheoryOfMindModuleConfig,
    UncertaintyAwareConfig,
)

# Direct class exports for convenience
from negotiation.components.cultural_adaptation import (
    CulturalAdaptation,
)
from negotiation.components.negotiation_instructions import (
    NegotiationInstructions,
)
from negotiation.components.negotiation_memory import (
    NegotiationMemory,
)
from negotiation.components.negotiation_strategy import (
    NegotiationStrategy,
)
from negotiation.components.strategy_evolution import (
    StrategyEvolution,
)
from negotiation.components.swarm_intelligence import (
    SwarmIntelligence,
)
from negotiation.components.temporal_strategy import (
    TemporalStrategy,
)
from negotiation.components.theory_of_mind import (
    TheoryOfMind,
)
from negotiation.components.uncertainty_aware import (
    UncertaintyAware,
)

__all__ = [
    # Module imports
    'negotiation_memory',
    'negotiation_instructions',
    'negotiation_strategy',
    'scoped_act',
    'cultural_adaptation',
    'temporal_strategy',
    'swarm_intelligence',
    'uncertainty_aware',
    'strategy_evolution',
    'theory_of_mind',
    # Direct class exports
    'NegotiationMemory',
    'NegotiationInstructions',
    'NegotiationStrategy',
    'CulturalAdaptation',
    'TemporalStrategy',
    'SwarmIntelligence',
    'UncertaintyAware',
    'StrategyEvolution',
    'TheoryOfMind',
    'ComponentDiagnosticContract',
    'ModelCallDeclaration',
    'ModelCallDiagnosticsMixin',
    'StatefulComponent',
    'CulturalAdaptationConfig',
    'TemporalStrategyConfig',
    'SwarmIntelligenceConfig',
    'UncertaintyAwareConfig',
    'StrategyEvolutionConfig',
    'TheoryOfMindModuleConfig',
]
