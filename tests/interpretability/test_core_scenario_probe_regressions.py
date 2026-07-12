"""Offline regressions for interpretability data, scenarios, and probes."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from interpretability.baseline_agents import BasicLLMAgent, create_all_baselines
from interpretability.core.dataset_builder import ActivationSample, DatasetBuilder
from interpretability.core.deepeval_detector import DeceptionResult
from interpretability.core.ground_truth import GroundTruthDetector
from interpretability.core.qc_filter import classify_response, filter_samples
from interpretability.probes.mech_interp_tools import extract_direction
from interpretability.probes.sanity_checks import sanity_check_random_labels
from interpretability.probes.train_probes import (
    _train_test_indices,
    analyze_dyadic_pairs,
    analyze_outcome_prediction,
    train_logistic_probe,
    train_ridge_probe,
)
from interpretability.scenarios.contest_scenarios import (
    BluffGameScenario,
    FisheryManagementScenario,
    HiddenInfoScenario,
    PromiseGameScenario,
    RealityGameshowScenario,
    TreatyNegotiationScenario,
)
from interpretability.scenarios.deception_scenarios import (
    AgentState,
    ExperimentMode,
    compute_ground_truth,
    generate_experiment_trials,
)
from interpretability.scenarios.emergent_prompts import (
    get_belief_shift_injection,
    verify_no_forbidden_words,
)


def test_dataset_builder_handles_empty_sae_maps_and_unknown_scenario(tmp_path):
    builder = DatasetBuilder()
    builder.add_sample(ActivationSample(
        trial_id="trial-1",
        round_num=0,
        agent_name="agent",
        activations={"blocks.2.hook_resid_post": torch.ones(4)},
        actual_deception=0.0,
        perceived_deception=0.0,
        sae_features={},
    ))
    path = tmp_path / "dataset.pt"
    builder.save(str(path))

    data = torch.load(path, weights_only=False)
    assert data["config"]["has_sae"] is False
    assert "sae_features" not in data
    assert data["labels"]["scenario"] == ["unknown"]


def test_standard_baseline_factories_share_one_model_contract():
    model = object()
    baselines = create_all_baselines(model)
    assert set(baselines) == {
        "random", "aggressive", "cooperative", "tit_for_tat", "moderate",
        "basic_llm",
    }
    assert isinstance(baselines["basic_llm"], BasicLLMAgent)
    assert baselines["basic_llm"].model is model


def test_qc_rejects_non_text_and_unknown_flag_names():
    assert classify_response(float("nan")) == {"too_short"}
    with pytest.raises(ValueError, match="Unknown QC flags"):
        filter_samples([], keep_flags=["typoed_flag"])


def test_failed_llm_evaluation_falls_back_to_rule_label():
    class FailedDetector:
        def evaluate_deception(self, response, context):
            return DeceptionResult(
                overall_score=0.0,
                is_deceptive=False,
                reasoning="offline",
                evaluation_succeeded=False,
            )

    detector = GroundTruthDetector()
    detector._deepeval_detector = FailedDetector()
    result = detector.detect_with_deepeval(
        "Our team needs 30 days.",
        "capability_bluff",
        {"true_days": 10},
    )
    assert result["deception_detected"] is True
    assert result["evaluation_method"] == "rules_fallback"


def test_instructed_ground_truth_is_consistent_without_observed_promises():
    state = AgentState(10, 10, 10, 10, [], [])
    promise = compute_ground_truth("promise_break", state)
    alliance = compute_ground_truth("alliance_betrayal", state)
    assert promise.is_deceptive is False and promise.deception_score == 0.0
    assert alliance.is_deceptive is False and alliance.deception_score == 0.0

    state.agent_claimed_honest = False
    withholding = compute_ground_truth("info_withholding", state)
    assert withholding.is_deceptive is True


def test_emergent_prompts_are_clean_and_missing_parameters_fail_closed():
    assert verify_no_forbidden_words() == []
    with pytest.raises(KeyError):
        get_belief_shift_injection("ultimatum_bluff", "revealed", {})


def test_contest_scenarios_use_replayable_local_rngs():
    random.seed(9182)
    global_state = random.getstate()
    first = FisheryManagementScenario(seed=7)
    second = FisheryManagementScenario(seed=7)
    assert random.getstate() == global_state
    assert first.config.agent_roles == second.config.agent_roles

    game = RealityGameshowScenario(seed=11)
    game.initialize()
    winner_1 = game.process_actions({})["challenge_winner"]
    game.initialize()
    winner_2 = game.process_actions({})["challenge_winner"]
    assert winner_1 == winner_2

    bluff = BluffGameScenario(seed=3)
    first_values = bluff.initialize()["current_offers"], dict(bluff.true_reservations)
    second_values = bluff.initialize()["current_offers"], dict(bluff.true_reservations)
    assert first_values == second_values


def test_contest_text_parsers_do_not_turn_price_or_negation_into_false_labels():
    hidden = HiddenInfoScenario()
    assert hidden._extract_stated_value("I am asking $90.") is None
    assert hidden._extract_stated_value("The item is worth $90.") == 90

    promise = PromiseGameScenario()
    assert promise._extract_promise("I will not cooperate; I will defect.") == "defect"
    assert promise._extract_action("I promised cooperate, but I choose DEFECT.") == "defect"


def test_all_contest_scenarios_complete_an_offline_transition():
    fishery = FisheryManagementScenario(num_agents=2)
    fishery.initialize()
    fishery.process_actions({role.name: "Deploy 2 boats" for role in fishery.config.agent_roles})

    treaty = TreatyNegotiationScenario(num_agents=3)
    treaty.initialize()
    proposer = treaty.config.agent_roles[0].name
    treaty.process_actions({proposer: "Propose 30% emissions, $100 billion, 10 years"})
    treaty.process_actions({role.name: "I support the proposal" for role in treaty.config.agent_roles})
    assert treaty.state["agreement_reached"] is True

    gameshow = RealityGameshowScenario(num_agents=3)
    gameshow.initialize()
    gameshow.process_actions({})

    bluff = BluffGameScenario()
    bluff.initialize()
    bluff.process_actions({"Agent_A": "My minimum is 45"})

    hidden = HiddenInfoScenario()
    hidden.initialize()
    hidden.process_actions({"Agent_A": "The item is worth $75"})

    promise = PromiseGameScenario()
    promise.initialize()
    promise.process_actions({"Agent_A": "I promise to cooperate"})
    for _ in range(3):
        promise.process_actions({})
    promise.process_actions({"Agent_A": "I choose defect"})


def test_all_emergent_scenario_templates_format_in_experiment_generation():
    trials = generate_experiment_trials(
        trials_per_condition=1,
        seed=19,
        mode=ExperimentMode.EMERGENT,
    )
    assert len(trials) == 12
    assert all("{" not in trial.agent_prompt for trial in trials)
    assert all(trial.expected_deception is None for trial in trials)


def test_group_split_is_disjoint_and_class_balanced():
    X = np.arange(160, dtype=float).reshape(40, 4)
    groups = np.repeat(np.arange(20), 2)
    y = np.repeat(np.arange(20) % 2, 2).astype(float)
    train_idx, test_idx = _train_test_indices(X, y, 42, groups)
    assert set(groups[train_idx]).isdisjoint(groups[test_idx])
    assert set(y[train_idx]) == {0.0, 1.0}
    assert set(y[test_idx]) == {0.0, 1.0}


def test_fitted_probe_pipelines_accept_raw_activation_vectors():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 12))
    y = (X[:, 0] > 0).astype(float)

    ridge, _ = train_ridge_probe(X, y, n_components=4)
    logistic, _ = train_logistic_probe(X, y, n_components=4)
    assert ridge.predict(X[:3]).shape == (3,)
    assert logistic.predict(X[:3]).shape == (3,)


def test_single_class_logistic_probe_returns_a_fitted_predictor():
    X = np.ones((12, 4))
    probe, result = train_logistic_probe(X, np.zeros(12))
    assert np.array_equal(probe.predict(X[:2]), np.zeros(2))
    assert result.auc == 0.5


def test_dyadic_analysis_requires_reciprocal_asymmetric_links():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(24, 8))
    y = np.tile([0.0, 1.0], 12)
    one_way = [i + 1 if i % 2 == 0 else None for i in range(24)]
    result = analyze_dyadic_pairs(X, y, one_way)
    assert result["n_pairs"] == 0
    assert "error" in result


def test_outcome_prediction_excludes_unknown_and_probe_round_rows():
    rng = np.random.default_rng(12)
    trial_count = 20
    rows_per_trial = 4
    X = rng.normal(size=(trial_count * rows_per_trial, 10))
    y = rng.random(trial_count * rows_per_trial)
    rounds = [-1, 0, 1, 2] * trial_count
    trials = [f"trial-{i}" for i in range(trial_count) for _ in range(rows_per_trial)]
    outcomes = [
        ("unknown" if i >= 18 else bool(i % 2))
        for i in range(trial_count)
        for _ in range(rows_per_trial)
    ]
    result = analyze_outcome_prediction(X, y, rounds, trials, outcomes)
    assert result["n_early_samples"] == 18 * 3


def test_random_label_sanity_check_does_not_mutate_numpy_global_rng():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(50, 5))
    y = rng.random(50)
    np.random.seed(77)
    state_before = np.random.get_state()
    sanity_check_random_labels(X, y, n_shuffles=2)
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_zero_mean_difference_has_no_defined_direction():
    identical = np.ones((4, 3))
    with pytest.raises(ValueError, match="identical class means"):
        extract_direction(identical, identical)
