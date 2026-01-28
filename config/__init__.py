"""Central configuration for mechanistic interpretability experiments.

This module provides all configuration classes for running deception detection
experiments on LLM agents. Configure experiments by modifying these settings.

Quick Start - Auto-configured (Recommended):
    from config import ExperimentConfig

    # Everything auto-configures based on model name
    config = ExperimentConfig.for_model("google/gemma-2b-it")
    config = ExperimentConfig.for_model("google/gemma-7b-it")

    # View what was configured
    config.print_config_summary()

Manual Configuration:
    from config import ExperimentConfig, ModelConfig, ProbeConfig

    config = ExperimentConfig(
        model=ModelConfig(name="google/gemma-7b-it"),
        probes=ProbeConfig.for_model("google/gemma-7b-it"),
    )

Supported Models (auto-configure):
    - google/gemma-2b-it (4GB VRAM)
    - google/gemma-7b-it (20GB VRAM)
    - google/gemma-7b-it (54GB VRAM)
    - meta-llama/Llama-3.1-8B-Instruct (16GB VRAM, no SAE)

Available Configurations:
    - ExperimentConfig: Main experiment settings
    - ModelConfig: LLM and interpretability model settings
    - ProbeConfig: Linear probe training settings
    - CausalConfig: Activation patching/ablation settings
    - ScenarioConfig: Deception scenario settings
    - MODEL_PRESETS: Dict of model-specific settings
"""

__version__ = "1.0.0"

from config.experiment import (
    ExperimentConfig,
    ModelConfig,
    EvaluatorConfig,
    ProbeConfig,
    CausalConfig,
    ScenarioConfig,
    # Model presets for auto-configuration
    MODEL_PRESETS,
    get_model_preset,
    # Preset configs
    QUICK_TEST,
    FULL_EXPERIMENT,
    FAST_ITERATION,
)

from config.agents.negotiation import (
    StrategyConfig,
    ModuleDefaults,
    TheoryOfMindConfig,
    DeceptionDetectionConfig,
)

__all__ = [
    # Version
    "__version__",
    # Core experiment configs
    "ExperimentConfig",
    "ModelConfig",
    "EvaluatorConfig",
    "ProbeConfig",
    "CausalConfig",
    "ScenarioConfig",
    # Model presets
    "MODEL_PRESETS",
    "get_model_preset",
    # Preset configs
    "QUICK_TEST",
    "FULL_EXPERIMENT",
    "FAST_ITERATION",
    # Agent configs
    "StrategyConfig",
    "ModuleDefaults",
    "TheoryOfMindConfig",
    "DeceptionDetectionConfig",
]
