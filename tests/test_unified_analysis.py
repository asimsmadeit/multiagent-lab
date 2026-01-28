"""Unit tests for unified research question analysis functions.

Tests for RQ1 (cross-mode transfer), RQ2 (implicit encoding),
RQ-MA1 (temporal trajectory), RQ-MA2 (dyadic pairs), RQ-MA3 (outcome prediction).
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
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


def import_train_probes():
    """Import train_probes module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_probes",
        os.path.join(os.path.dirname(__file__), "../interpretability/probes/train_probes.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def train_probes_module():
    """Fixture to load train_probes module."""
    return import_train_probes()


@pytest.fixture
def sample_activations():
    """Generate sample activation data for testing."""
    np.random.seed(42)
    n_samples = 200
    d_model = 64

    # Generate activations with some structure
    X = np.random.randn(n_samples, d_model)

    # Add signal: first 10 dims correlate with deception
    deception_signal = X[:, :10].mean(axis=1)

    # Generate labels based on signal + noise
    y = (deception_signal > 0).astype(float) + 0.1 * np.random.randn(n_samples)
    y = np.clip(y, 0, 1)

    return X, y


@pytest.fixture
def multimode_data(sample_activations):
    """Generate data with mode labels for cross-mode transfer testing."""
    X, y = sample_activations
    n_samples = len(y)

    # Split into instructed (first half) and emergent (second half)
    mode_labels = ["instructed"] * (n_samples // 2) + ["emergent"] * (n_samples - n_samples // 2)

    return X, y, mode_labels


@pytest.fixture
def multiround_data(sample_activations):
    """Generate data with round numbers for trajectory testing."""
    X, y = sample_activations
    n_samples = len(y)

    # 5 rounds, ~40 samples each
    round_nums = []
    for i in range(n_samples):
        round_nums.append((i % 5) + 1)

    return X, y, round_nums


@pytest.fixture
def dyadic_data(sample_activations):
    """Generate data with counterpart indices for dyadic analysis."""
    X, y = sample_activations
    n_samples = len(y)

    # Pair consecutive samples as counterparts
    counterpart_idxs = []
    for i in range(n_samples):
        if i % 2 == 0:
            counterpart_idxs.append(i + 1 if i + 1 < n_samples else -1)
        else:
            counterpart_idxs.append(i - 1)

    return X, y, counterpart_idxs


@pytest.fixture
def outcome_data(sample_activations):
    """Generate data for outcome prediction testing."""
    X, y = sample_activations
    n_samples = len(y)

    # 5 rounds
    round_nums = [(i % 5) + 1 for i in range(n_samples)]

    # Trial IDs (4 samples per trial = 40 trials for 200 samples)
    trial_ids = [f"trial_{i // 5}" for i in range(n_samples)]

    # Outcomes: success if mean deception in trial > 0.5
    trial_outcomes = []
    for i in range(n_samples):
        trial_idx = i // 5
        trial_samples = [j for j in range(n_samples) if j // 5 == trial_idx]
        trial_mean = np.mean([y[j] for j in trial_samples])
        trial_outcomes.append(trial_mean > 0.5)

    return X, y, round_nums, trial_ids, trial_outcomes


class TestGroupByRound:
    """Tests for group_by_round helper function."""

    def test_basic_grouping(self, sample_activations, train_probes_module):
        """Test that samples are correctly grouped by round."""
        group_by_round = train_probes_module.group_by_round

        X, y = sample_activations
        n_samples = len(y)
        round_nums = [(i % 3) + 1 for i in range(n_samples)]

        grouped = group_by_round(X, y, round_nums)

        assert len(grouped) == 3  # 3 unique rounds
        assert 1 in grouped
        assert 2 in grouped
        assert 3 in grouped

    def test_correct_sample_counts(self, sample_activations, train_probes_module):
        """Test that each group has correct number of samples."""
        group_by_round = train_probes_module.group_by_round

        X, y = sample_activations
        n_samples = len(y)
        round_nums = [1] * 50 + [2] * 100 + [3] * 50

        grouped = group_by_round(X, y, round_nums)

        assert len(grouped[1][0]) == 50
        assert len(grouped[2][0]) == 100
        assert len(grouped[3][0]) == 50


class TestCrossModeTransfer:
    """Tests for compute_cross_mode_transfer (RQ1)."""

    def test_basic_functionality(self, multimode_data, train_probes_module):
        """Test that cross-mode transfer runs without error."""
        compute_cross_mode_transfer = train_probes_module.compute_cross_mode_transfer

        X, y, mode_labels = multimode_data
        result = compute_cross_mode_transfer(X, y, mode_labels)

        assert "transfer_auc" in result
        assert "transfer_r2" in result
        assert "interpretation" in result
        assert "n_instructed" in result
        assert "n_emergent" in result

    def test_transfer_auc_range(self, multimode_data, train_probes_module):
        """Test that transfer AUC is in valid range."""
        compute_cross_mode_transfer = train_probes_module.compute_cross_mode_transfer

        X, y, mode_labels = multimode_data
        result = compute_cross_mode_transfer(X, y, mode_labels)

        if result.get("transfer_auc") is not None:
            assert 0 <= result["transfer_auc"] <= 1

    def test_insufficient_samples_handling(self, train_probes_module):
        """Test handling of insufficient samples."""
        compute_cross_mode_transfer = train_probes_module.compute_cross_mode_transfer

        X = np.random.randn(15, 32)
        y = np.random.rand(15)
        mode_labels = ["instructed"] * 5 + ["emergent"] * 10

        result = compute_cross_mode_transfer(X, y, mode_labels)

        assert "error" in result or result["n_instructed"] >= 10

    def test_single_mode_handling(self, sample_activations, train_probes_module):
        """Test handling when only one mode is present."""
        compute_cross_mode_transfer = train_probes_module.compute_cross_mode_transfer

        X, y = sample_activations
        mode_labels = ["instructed"] * len(y)  # All same mode

        result = compute_cross_mode_transfer(X, y, mode_labels)

        # Should return error or indicate insufficient data
        assert "error" in result or result.get("n_emergent", 0) < 10


class TestImplicitEncoding:
    """Tests for analyze_implicit_encoding (RQ2)."""

    def test_basic_functionality(self, sample_activations, train_probes_module):
        """Test that implicit encoding analysis runs without error."""
        analyze_implicit_encoding = train_probes_module.analyze_implicit_encoding

        X, y = sample_activations
        agent_labels = y + 0.1 * np.random.randn(len(y))
        agent_labels = np.clip(agent_labels, 0, 1)

        result = analyze_implicit_encoding(X, y, agent_labels)

        assert "gm_auc" in result
        assert "agent_auc" in result
        assert "auc_gap" in result
        assert "gm_wins" in result
        assert "interpretation" in result

    def test_auc_values_valid(self, sample_activations, train_probes_module):
        """Test that AUC values are in valid range."""
        analyze_implicit_encoding = train_probes_module.analyze_implicit_encoding

        X, y = sample_activations
        agent_labels = y.copy()

        result = analyze_implicit_encoding(X, y, agent_labels)

        assert 0 <= result["gm_auc"] <= 1
        assert 0 <= result["agent_auc"] <= 1

    def test_gm_wins_consistent(self, sample_activations, train_probes_module):
        """Test that gm_wins is consistent with auc_gap."""
        analyze_implicit_encoding = train_probes_module.analyze_implicit_encoding

        X, y = sample_activations
        agent_labels = y.copy()

        result = analyze_implicit_encoding(X, y, agent_labels)

        expected_gm_wins = result["auc_gap"] > 0
        assert result["gm_wins"] == expected_gm_wins


class TestRoundTrajectory:
    """Tests for analyze_round_trajectory (RQ-MA1)."""

    def test_basic_functionality(self, multiround_data, train_probes_module):
        """Test that trajectory analysis runs without error."""
        analyze_round_trajectory = train_probes_module.analyze_round_trajectory

        X, y, round_nums = multiround_data

        # Need activations_by_layer format
        activations_by_layer = {10: X, 15: X, 20: X}

        result = analyze_round_trajectory(activations_by_layer, y, round_nums)

        assert "per_round" in result
        assert "trajectory_slope" in result
        assert "interpretation" in result

    def test_per_round_results(self, multiround_data, train_probes_module):
        """Test that per-round results are computed."""
        analyze_round_trajectory = train_probes_module.analyze_round_trajectory

        X, y, round_nums = multiround_data
        activations_by_layer = {15: X}

        result = analyze_round_trajectory(activations_by_layer, y, round_nums)

        assert len(result["per_round"]) == 5  # 5 rounds

    def test_insufficient_rounds(self, sample_activations, train_probes_module):
        """Test handling of insufficient rounds."""
        analyze_round_trajectory = train_probes_module.analyze_round_trajectory

        X, y = sample_activations
        round_nums = [1] * len(y)  # All same round
        activations_by_layer = {15: X}

        result = analyze_round_trajectory(activations_by_layer, y, round_nums)

        assert "error" in result


class TestDyadicPairs:
    """Tests for analyze_dyadic_pairs (RQ-MA2)."""

    def test_basic_functionality(self, dyadic_data, train_probes_module):
        """Test that dyadic analysis runs without error."""
        analyze_dyadic_pairs = train_probes_module.analyze_dyadic_pairs

        X, y, counterpart_idxs = dyadic_data
        result = analyze_dyadic_pairs(X, y, counterpart_idxs)

        assert "n_pairs" in result
        assert "pair_probe_auc" in result or "error" in result

    def test_d_prime_computed(self, dyadic_data, train_probes_module):
        """Test that d-prime separability measure is computed."""
        analyze_dyadic_pairs = train_probes_module.analyze_dyadic_pairs

        X, y, counterpart_idxs = dyadic_data
        result = analyze_dyadic_pairs(X, y, counterpart_idxs)

        if "error" not in result:
            assert "d_prime" in result

    def test_insufficient_pairs(self, train_probes_module):
        """Test handling of insufficient pairs."""
        analyze_dyadic_pairs = train_probes_module.analyze_dyadic_pairs

        X = np.random.randn(10, 32)
        y = np.random.rand(10)
        counterpart_idxs = [-1] * 10  # No valid pairs

        result = analyze_dyadic_pairs(X, y, counterpart_idxs)

        assert "error" in result


class TestOutcomePrediction:
    """Tests for analyze_outcome_prediction (RQ-MA3)."""

    def test_basic_functionality(self, outcome_data, train_probes_module):
        """Test that outcome prediction runs without error."""
        analyze_outcome_prediction = train_probes_module.analyze_outcome_prediction

        X, y, round_nums, trial_ids, trial_outcomes = outcome_data
        result = analyze_outcome_prediction(X, y, round_nums, trial_ids, trial_outcomes)

        assert "early_rounds_auc" in result or "error" in result

    def test_early_round_filtering(self, outcome_data, train_probes_module):
        """Test that early rounds are correctly filtered."""
        analyze_outcome_prediction = train_probes_module.analyze_outcome_prediction

        X, y, round_nums, trial_ids, trial_outcomes = outcome_data
        result = analyze_outcome_prediction(X, y, round_nums, trial_ids, trial_outcomes)

        if "error" not in result:
            # Check that n_early_samples is reasonable (rounds 1-2)
            expected_early = sum(1 for r in round_nums if r <= 2)
            assert result["n_early_samples"] == expected_early

    def test_insufficient_early_samples(self, train_probes_module):
        """Test handling of insufficient early samples."""
        analyze_outcome_prediction = train_probes_module.analyze_outcome_prediction

        X = np.random.randn(20, 32)
        y = np.random.rand(20)
        round_nums = [5] * 20  # All late rounds
        trial_ids = ["trial_1"] * 20
        trial_outcomes = [True] * 20

        result = analyze_outcome_prediction(X, y, round_nums, trial_ids, trial_outcomes)

        assert "error" in result


class TestInterpretations:
    """Tests for interpretation generation."""

    def test_cross_mode_interpretations(self, multimode_data, train_probes_module):
        """Test that cross-mode transfer generates valid interpretations."""
        compute_cross_mode_transfer = train_probes_module.compute_cross_mode_transfer

        X, y, mode_labels = multimode_data
        result = compute_cross_mode_transfer(X, y, mode_labels)

        if "interpretation" in result:
            assert isinstance(result["interpretation"], str)
            assert len(result["interpretation"]) > 0

    def test_implicit_encoding_interpretations(self, sample_activations, train_probes_module):
        """Test that implicit encoding generates valid interpretations."""
        analyze_implicit_encoding = train_probes_module.analyze_implicit_encoding

        X, y = sample_activations
        result = analyze_implicit_encoding(X, y, y)

        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
