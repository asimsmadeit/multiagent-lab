"""Unit tests for probe training functions.

These tests require PyTorch and scikit-learn to be available.
They will be skipped if the dependencies are not installed.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path to allow direct imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check if required dependencies are available
try:
    import torch
    import sklearn
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False

pytestmark = pytest.mark.skipif(
    not HAS_DEPENDENCIES,
    reason="Requires PyTorch and scikit-learn"
)


# Import functions directly to avoid heavy dependencies
def import_train_probes():
    """Import train_probes module with minimal dependencies."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_probes",
        os.path.join(os.path.dirname(__file__), "../interpretability/probes/train_probes.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_sanity_checks():
    """Import sanity_checks module with minimal dependencies."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sanity_checks",
        os.path.join(os.path.dirname(__file__), "../interpretability/probes/sanity_checks.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sample_data():
    """Generate sample activation data for testing."""
    np.random.seed(42)
    n_samples = 100
    d_model = 64

    # Generate random activations
    X = np.random.randn(n_samples, d_model)

    # Generate labels with some signal
    # Positive labels are correlated with first few dimensions
    y = (X[:, :5].mean(axis=1) > 0).astype(float) + 0.1 * np.random.randn(n_samples)
    y = np.clip(y, 0, 1)

    return X, y


@pytest.fixture
def scenario_data(sample_data):
    """Generate sample data with scenario labels."""
    X, y = sample_data
    n_samples = len(y)
    scenarios = np.array(["scenario_a"] * 30 + ["scenario_b"] * 30 +
                        ["scenario_c"] * 20 + ["scenario_d"] * 20)
    return X, y, scenarios.tolist()


@pytest.fixture
def train_probes_module():
    """Fixture to load train_probes module."""
    return import_train_probes()


@pytest.fixture
def sanity_checks_module():
    """Fixture to load sanity_checks module."""
    return import_sanity_checks()


class TestThresholdSensitivityAnalysis:
    """Tests for threshold_sensitivity_analysis function."""

    def test_basic_functionality(self, sample_data, train_probes_module):
        """Test that threshold analysis runs without error."""
        threshold_sensitivity_analysis = train_probes_module.threshold_sensitivity_analysis

        X, y = sample_data
        # Use y as both true and pred for simplicity
        result = threshold_sensitivity_analysis(y, y)

        assert "by_threshold" in result
        assert "best_threshold" in result
        assert "is_robust" in result
        assert "recommendation" in result

    def test_default_thresholds(self, sample_data, train_probes_module):
        """Test that default thresholds are used."""
        threshold_sensitivity_analysis = train_probes_module.threshold_sensitivity_analysis

        X, y = sample_data
        result = threshold_sensitivity_analysis(y, y)

        expected_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        for thresh in expected_thresholds:
            assert thresh in result["by_threshold"]

    def test_custom_thresholds(self, sample_data, train_probes_module):
        """Test custom threshold list."""
        threshold_sensitivity_analysis = train_probes_module.threshold_sensitivity_analysis

        X, y = sample_data
        custom_thresholds = [0.25, 0.5, 0.75]
        result = threshold_sensitivity_analysis(y, y, thresholds=custom_thresholds)

        for thresh in custom_thresholds:
            assert thresh in result["by_threshold"]

    def test_best_threshold_is_float(self, sample_data, train_probes_module):
        """Test that best_threshold is a float."""
        threshold_sensitivity_analysis = train_probes_module.threshold_sensitivity_analysis

        X, y = sample_data
        result = threshold_sensitivity_analysis(y, y)

        assert isinstance(result["best_threshold"], float)
        assert 0 <= result["best_threshold"] <= 1


class TestSanityChecks:
    """Tests for sanity check functions."""

    def test_label_variance_check_passes(self, sample_data, sanity_checks_module):
        """Test that label variance check passes for valid labels."""
        sanity_check_label_variance = sanity_checks_module.sanity_check_label_variance

        X, y = sample_data
        result = sanity_check_label_variance(y)

        assert "passed" in result
        assert "std" in result
        assert "mean" in result
        assert result["std"] > 0  # Should have variance

    def test_label_variance_check_fails_constant(self, sanity_checks_module):
        """Test that label variance check fails for constant labels."""
        sanity_check_label_variance = sanity_checks_module.sanity_check_label_variance

        constant_labels = np.ones(100)
        result = sanity_check_label_variance(constant_labels)

        assert result["passed"] is False
        assert result["std"] == 0


class TestGeneralizationAUC:
    """Tests for cross-scenario generalization."""

    def test_basic_functionality(self, scenario_data, train_probes_module):
        """Test that generalization AUC computes without error."""
        compute_generalization_auc = train_probes_module.compute_generalization_auc

        X, y, scenarios = scenario_data
        result = compute_generalization_auc(X, y, scenarios)

        assert "by_scenario" in result
        assert "average_auc" in result
        assert "average_r2" in result

    def test_all_scenarios_present(self, scenario_data, train_probes_module):
        """Test that all scenarios are in results."""
        compute_generalization_auc = train_probes_module.compute_generalization_auc

        X, y, scenarios = scenario_data
        unique_scenarios = set(scenarios)
        result = compute_generalization_auc(X, y, scenarios)

        for scenario in unique_scenarios:
            assert scenario in result["by_scenario"]

    def test_random_state_reproducibility(self, scenario_data, train_probes_module):
        """Test that random_state produces reproducible results."""
        compute_generalization_auc = train_probes_module.compute_generalization_auc

        X, y, scenarios = scenario_data

        result1 = compute_generalization_auc(X, y, scenarios, random_state=42)
        result2 = compute_generalization_auc(X, y, scenarios, random_state=42)

        # Results should be identical with same seed
        assert result1["average_auc"] == result2["average_auc"]


class TestProbeResult:
    """Tests for ProbeResult dataclass."""

    def test_probe_result_to_dict(self, train_probes_module):
        """Test ProbeResult.to_dict() method."""
        ProbeResult = train_probes_module.ProbeResult

        result = ProbeResult(
            layer=10,
            label_type="gm",
            r2_score=0.5,
            accuracy=0.75,
            auc=0.8,
            train_r2=0.6,
            test_r2=0.5,
            cross_val_scores=[0.4, 0.5, 0.6],
        )

        d = result.to_dict()

        assert d["layer"] == 10
        assert d["label_type"] == "gm"
        assert d["r2_score"] == 0.5
        assert d["accuracy"] == 0.75
        assert d["auc"] == 0.8
        assert "cross_val_mean" in d
        assert "cross_val_std" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
