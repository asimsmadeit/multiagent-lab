"""Basic import tests to verify package structure."""

import pytest


def test_concordia_mini_imports():
    """Test that concordia_mini core modules import."""
    from concordia_mini.typing import entity_component
    from concordia_mini.language_model import language_model
    assert entity_component is not None
    assert language_model is not None


def test_negotiation_imports():
    """Test that negotiation modules import."""
    from config.agents import negotiation as config
    from negotiation import constants
    from negotiation.components import theory_of_mind
    assert config is not None
    assert constants is not None


def test_negotiation_config():
    """Test that config classes are available."""
    from config.agents.negotiation import (
        StrategyConfig,
        EvaluationConfig,
        DeceptionDetectionConfig,
    )
    assert StrategyConfig is not None
    assert EvaluationConfig is not None


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="PyTorch not available"),
    reason="Requires PyTorch"
)
def test_interpretability_scenarios():
    """Test that scenarios are defined (requires PyTorch)."""
    from interpretability.scenarios.emergent_prompts import EMERGENT_SCENARIOS
    assert len(EMERGENT_SCENARIOS) == 6
    assert "ultimatum_bluff" in EMERGENT_SCENARIOS
    assert "alliance_betrayal" in EMERGENT_SCENARIOS


def test_scenarios_without_torch():
    """Test scenarios can be imported directly."""
    # Import the module directly without going through __init__
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "emergent_prompts",
        os.path.join(os.path.dirname(__file__), "../interpretability/scenarios/emergent_prompts.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "EMERGENT_SCENARIOS")
    assert len(module.EMERGENT_SCENARIOS) == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
