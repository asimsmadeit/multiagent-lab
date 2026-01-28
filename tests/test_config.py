"""Unit tests for configuration classes."""

import pytest
from dataclasses import fields


class TestExperimentConfig:
    """Tests for ExperimentConfig and related classes."""

    def test_experiment_config_defaults(self):
        """Test that ExperimentConfig has sensible defaults."""
        from config.experiment import ExperimentConfig

        config = ExperimentConfig()

        # Check defaults
        assert config.random_seed == 42
        assert config.use_multi_seed is False
        assert config.random_seeds == [42, 123, 456, 789, 1337]
        assert config.output_dir == "./results"
        assert config.verbose is True

    def test_model_config_defaults(self):
        """Test ModelConfig defaults."""
        from config.experiment import ModelConfig

        config = ModelConfig()

        assert config.name == "google/gemma-2-27b-it"
        assert config.device in ["cuda", "cpu", "mps"]
        assert config.use_transformerlens is True
        assert config.sae_layer == 31

    def test_probe_config_defaults(self):
        """Test ProbeConfig defaults."""
        from config.experiment import ProbeConfig

        config = ProbeConfig()

        assert config.train_ratio == 0.8
        assert 0 < config.train_ratio < 1
        assert config.regularization == 1.0
        assert config.token_position == "last"
        assert config.binary_threshold == 0.5
        assert config.run_sanity_checks is True
        assert config.run_cross_scenario_validation is True
        assert config.run_threshold_sensitivity is True

    def test_probe_config_layers_to_probe(self):
        """Test that layers_to_probe is a list of integers."""
        from config.experiment import ProbeConfig

        config = ProbeConfig()

        assert isinstance(config.layers_to_probe, list)
        assert all(isinstance(layer, int) for layer in config.layers_to_probe)
        assert len(config.layers_to_probe) > 0

    def test_causal_config_defaults(self):
        """Test CausalConfig defaults."""
        from config.experiment import CausalConfig

        config = CausalConfig()

        assert config.enabled is True
        assert config.num_samples == 30
        assert config.run_activation_patching is True
        assert config.run_ablation is True
        assert config.run_steering is True

    def test_scenario_config_defaults(self):
        """Test ScenarioConfig defaults."""
        from config.experiment import ScenarioConfig

        config = ScenarioConfig()

        assert config.mode in ["emergent", "instructed", "contest", "both"]
        assert config.num_trials == 50
        assert config.max_rounds == 5
        assert len(config.scenarios) > 0

    def test_quick_test_preset(self):
        """Test QUICK_TEST preset has reduced settings."""
        from config.experiment import QUICK_TEST

        assert QUICK_TEST.scenarios.num_trials == 1
        assert len(QUICK_TEST.scenarios.scenarios) == 1
        assert QUICK_TEST.causal.num_samples == 10

    def test_full_experiment_preset(self):
        """Test FULL_EXPERIMENT preset has full settings."""
        from config.experiment import FULL_EXPERIMENT

        assert FULL_EXPERIMENT.scenarios.num_trials == 50
        assert FULL_EXPERIMENT.causal.num_samples == 30

    def test_fast_iteration_preset(self):
        """Test FAST_ITERATION preset disables slow operations."""
        from config.experiment import FAST_ITERATION

        assert FAST_ITERATION.model.use_sae is False
        assert FAST_ITERATION.causal.enabled is False


class TestVersionExport:
    """Tests for version export."""

    def test_version_exported(self):
        """Test that __version__ is exported from config package."""
        from config import __version__

        assert __version__ == "1.0.0"
        assert isinstance(__version__, str)

    def test_version_in_all(self):
        """Test that __version__ is in __all__."""
        import config

        assert "__version__" in config.__all__


class TestConfigValidation:
    """Tests for config value validation."""

    def test_train_ratio_range(self):
        """Test that train_ratio is in valid range."""
        from config.experiment import ProbeConfig

        config = ProbeConfig()
        assert 0 < config.train_ratio < 1, "train_ratio must be between 0 and 1"

    def test_binary_threshold_range(self):
        """Test that binary_threshold is in valid range."""
        from config.experiment import ProbeConfig

        config = ProbeConfig()
        assert 0 <= config.binary_threshold <= 1, "binary_threshold must be between 0 and 1"

    def test_min_accuracy_range(self):
        """Test that min_accuracy is in valid range."""
        from config.experiment import ProbeConfig

        config = ProbeConfig()
        assert 0 <= config.min_accuracy <= 1, "min_accuracy must be between 0 and 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
