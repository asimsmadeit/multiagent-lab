"""Basic example of running a deception detection experiment."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import scenarios directly (avoids torch dependency for this demo)
import importlib.util

# Load scenarios without triggering torch import
spec = importlib.util.spec_from_file_location(
    "emergent_prompts",
    os.path.join(os.path.dirname(__file__), "../interpretability/scenarios/emergent_prompts.py")
)
emergent_prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(emergent_prompts)

EMERGENT_SCENARIOS = emergent_prompts.EMERGENT_SCENARIOS

# Show available scenarios
print("Available deception scenarios:")
print("-" * 40)
for name, scenario in EMERGENT_SCENARIOS.items():
    # Scenarios have value_ranges that describe what's being tested
    value_ranges = scenario.get("value_ranges", {})
    ranges_str = ", ".join(f"{k}: {v}" for k, v in value_ranges.items())
    print(f"  {name}:")
    print(f"    Parameters: {ranges_str}")
print()

# Show config usage
print("Configuration example:")
print("-" * 40)
from config import ExperimentConfig, ModelConfig, ScenarioConfig

# Create experiment config
config = ExperimentConfig(
    model=ModelConfig(
        name="google/gemma-2b-it",
        device="cuda",
    ),
    scenarios=ScenarioConfig(
        mode="emergent",
        scenarios=["ultimatum_bluff", "alliance_betrayal"],
        num_trials=10,
    ),
)

print(f"  Model: {config.model.name}")
print(f"  Device: {config.model.device}")
print(f"  Scenarios: {config.scenarios.scenarios}")
print(f"  Trials per scenario: {config.scenarios.num_trials}")
print()

# How to run
print("To run full experiment:")
print("-" * 40)
print("  python -m interpretability.run_deception_experiment \\")
print("      --mode emergent \\")
print("      --scenarios 6 \\")
print("      --trials 10 \\")
print("      --device cuda")
