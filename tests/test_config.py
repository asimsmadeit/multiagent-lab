"""Unit tests for configuration classes."""

# Pydantic v2 fields are concrete nested models at runtime; Pylint's static
# inference sees their class-level FieldInfo descriptors instead.
# pylint: disable=no-member

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
        assert config.use_sae is False
        assert config.sae_release is None
        assert config.sae_layer is None
        assert config.sae_id is None

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

        assert config.enabled is False
        assert config.num_samples == 30
        assert config.run_activation_patching is True
        assert config.run_ablation is True
        assert config.run_steering is True

    def test_scenario_config_defaults(self):
        """Test ScenarioConfig defaults."""
        from config.experiment import ScenarioConfig

        config = ScenarioConfig()

        assert config.mode == "emergent"
        assert config.num_trials == 50
        assert config.max_rounds == 5
        assert len(config.scenarios) > 0
        assert config.protocol == "alternating"
        assert config.counterbalance is True
        assert config.probes == "auto"
        assert config.executions_per_family == 64
        assert config.counterpart_policies == [
            "default", "skeptical", "credulous", "informed"
        ]

    def test_scenario_protocol_policy_combinations_fail_closed(self):
        """Absence is a solo scheduler, never a two-party behavior policy."""
        from config.experiment import ScenarioConfig

        solo = ScenarioConfig(
            protocol="solo_no_response",
            counterpart_policies=["absent"],
            counterbalance=False,
            surface_variants=["default"],
        )
        assert solo.protocol == "solo_no_response"
        with pytest.raises(ValueError, match="requires counterpart_policies"):
            ScenarioConfig(protocol="solo_no_response", counterbalance=False)
        with pytest.raises(ValueError, match="does not support"):
            ScenarioConfig(
                protocol="solo_no_response",
                counterpart_policies=["absent"],
            )
        with pytest.raises(ValueError, match="requires protocol"):
            ScenarioConfig(counterpart_policies=["default", "absent"])
        with pytest.raises(ValueError, match="must be unique"):
            ScenarioConfig(counterpart_policies=["default", "default"])
        with pytest.raises(ValueError, match="exactly one counterpart policy"):
            ScenarioConfig(
                counterpart_policies=["default", "skeptical"],
                counterbalance=False,
                surface_variants=["default"],
            )
        with pytest.raises(ValueError, match="probes currently require"):
            ScenarioConfig(protocol="simultaneous", probes="on")
        with pytest.raises(ValueError, match="instructed execution supports"):
            ScenarioConfig(mode="both", protocol="simultaneous")
        with pytest.raises(ValueError, match="custom counterpart policies"):
            ScenarioConfig(
                mode="instructed",
                counterpart_policies=["default"],
            )

    def test_track_protocol_and_module_combinations_fail_closed(self):
        """Config validation must match the executable public runtime."""
        from config.experiment import ExperimentConfig, ScenarioConfig

        with pytest.raises(ValueError, match="bilateral_white_box"):
            ExperimentConfig(
                experiment_track="bilateral_white_box",
                scenarios=ScenarioConfig(
                    protocol="solo_no_response",
                    counterpart_policies=["absent"],
                    counterbalance=False,
                    surface_variants=["default"],
                ),
            )
        with pytest.raises(ValueError, match="emergent mode only"):
            ExperimentConfig(
                experiment_track="text_only",
                scenarios=ScenarioConfig(mode="instructed"),
            )
        with pytest.raises(ValueError, match="strategy_evolution"):
            ExperimentConfig(experiment_track="adaptive")
        adaptive = ExperimentConfig(
            experiment_track="adaptive",
            scenarios=ScenarioConfig(agent_modules=["strategy_evolution"]),
        )
        assert adaptive.scenarios.agent_modules == ["strategy_evolution"]
        with pytest.raises(ValueError, match="does not allow online adaptation"):
            ExperimentConfig(
                experiment_track="single_agent_white_box",
                scenarios=ScenarioConfig(agent_modules=["strategy_evolution"]),
            )

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
        assert FULL_EXPERIMENT.scenarios.mode == "emergent"
        assert FULL_EXPERIMENT.causal.num_samples == 30

    def test_instructed_modes_report_legacy_non_headline_status(self):
        """Explicit compatibility modes cannot look like headline defaults."""
        from config.experiment import ScenarioConfig

        with pytest.warns(UserWarning, match="legacy/non-headline"):
            config = ScenarioConfig(mode="both")

        assert config.mode == "both"

        with pytest.warns(UserWarning, match="legacy/non-headline"):
            instructed = ScenarioConfig(mode="instructed")
        assert instructed.executions_per_family == 1

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

    @pytest.mark.parametrize(
        "payload",
        [
            {"unknown_top_level": True},
            {"model": {"unknown_model_option": True}},
            {"scenarios": {"unknown_scenario_option": True}},
            {"probes": {"unknown_probe_option": True}},
            {"causal": {"unknown_causal_option": True}},
            {"evaluator": {"unknown_evaluator_option": True}},
        ],
    )
    def test_unknown_config_keys_fail_closed_at_every_boundary(self, payload):
        """Typos must never validate as silently ignored experiment controls."""
        from config.experiment import ExperimentConfig

        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            ExperimentConfig.model_validate(payload)

    @pytest.mark.parametrize("value", [-1, True, 1.5, "3"])
    def test_random_seed_requires_an_exact_nonnegative_integer(self, value):
        from config.experiment import ExperimentConfig

        with pytest.raises(ValueError, match="random_seed must be"):
            ExperimentConfig(random_seed=value)

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ([], "non-empty"),
            ([1, 1], "unique"),
            ([1, -1], "non-negative"),
            ([1, True], "non-negative"),
            ([1, "2"], "non-negative"),
        ],
    )
    def test_reserved_random_seed_list_is_valid_even_when_unused(
        self, value, message
    ):
        from config.experiment import ExperimentConfig

        with pytest.raises(ValueError, match=message):
            ExperimentConfig(random_seeds=value)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sae_release": "custom-release"},
            {"sae_layer": 7},
            {"sae_id": "custom-id"},
        ],
    )
    def test_disabled_sae_rejects_inert_release_layer_and_id(self, kwargs):
        from config.experiment import ModelConfig

        with pytest.raises(ValueError, match="must be None when use_sae=False"):
            ModelConfig(use_sae=False, **kwargs)

    @pytest.mark.parametrize(
        "model_name",
        ["offline/unknown-model", "meta-llama/Llama-2-7b-chat-hf"],
    )
    def test_requested_sae_never_silently_downgrades_without_release(
        self, model_name
    ):
        from config.experiment import ModelConfig

        with pytest.raises(ValueError, match="no supported SAE release"):
            ModelConfig(name=model_name, use_sae=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
