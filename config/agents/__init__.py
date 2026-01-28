"""Agent-specific configurations.

This module contains configurations for different agent types.
Currently supports negotiation agents, extensible for future agent types.

To add a new agent type:
1. Create config/agents/your_agent.py
2. Define dataclass configs for your agent
3. Import them here
"""

from config.agents.negotiation import (
    StrategyConfig,
    ModuleDefaults,
    TheoryOfMindConfig,
    DeceptionDetectionConfig,
    RelationshipConfig,
    OutcomeConfig,
    AlgorithmConfig,
    ParsingConfig,
)

__all__ = [
    # Negotiation agent configs
    "StrategyConfig",
    "ModuleDefaults",
    "TheoryOfMindConfig",
    "DeceptionDetectionConfig",
    "RelationshipConfig",
    "OutcomeConfig",
    "AlgorithmConfig",
    "ParsingConfig",
]
