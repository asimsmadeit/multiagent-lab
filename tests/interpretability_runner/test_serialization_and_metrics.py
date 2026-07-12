from types import SimpleNamespace

import pytest
import torch

from interpretability.evaluation import ActivationSample, InterpretabilityRunner
from interpretability.metrics import MetricsCollector


def _sample(*, trial_id=1, layer="blocks.2.hook_resid_post", sae_features=None):
    return ActivationSample(
        trial_id=trial_id,
        round_num=0,
        agent_name="Negotiator",
        activations={layer: torch.tensor([1.0, 2.0, 3.0])},
        prompt="assembled prompt",
        response="I accept these terms.",
        perceived_deception=0.2,
        emotion_intensity=0.1,
        trust_level=0.7,
        cooperation_intent=0.8,
        actual_deception=0.0,
        commitment_violation=0.0,
        manipulation_score=0.0,
        consistency_score=1.0,
        scenario_type="ultimatum_bluff",
        modules_enabled=["theory_of_mind"],
        emergent_ground_truth=False,
        sample_type="negotiation",
        sae_features=sae_features,
    )


def _runner(samples):
    runner = object.__new__(InterpretabilityRunner)
    runner.model = SimpleNamespace(model_name="offline-test-model")
    runner.activation_samples = samples
    runner._pod_id = 0
    runner._trial_id_offset = 0
    return runner


def test_save_dataset_preserves_sae_sample_axis_and_label_semantics(tmp_path):
    runner = _runner([
        _sample(trial_id=1),
        _sample(trial_id=2, sae_features={4: 1.5}),
    ])
    output = tmp_path / "activations.pt"

    runner.save_dataset(str(output))
    data = torch.load(output, weights_only=False)

    assert data["activations"][2].shape == (2, 3)
    assert data["sae_features"].shape == (2, 5)
    assert data["sae_available_mask"] == [False, True]
    assert data["sae_features"][0].count_nonzero() == 0
    assert data["sae_features"][1, 4].item() == 1.5
    assert data["config"]["label_semantics"]["agent_labels"].startswith(
        "acting agent estimate of counterpart"
    )
    assert data["metadata"][0]["activation_position"] == "last_response_token"
    assert output.with_name("activations_transcripts.jsonl").exists()


def test_save_dataset_rejects_misaligned_activation_layers(tmp_path):
    runner = _runner([
        _sample(trial_id=1),
        _sample(trial_id=2, layer="blocks.3.hook_resid_post"),
    ])

    with pytest.raises(ValueError, match="not aligned"):
        runner.save_dataset(str(tmp_path / "bad.pt"))


def test_metrics_collector_finalization_adds_each_trial_once():
    collector = MetricsCollector("test")
    experiment = collector.start_experiment("fishery", [])
    collector.start_negotiation("fishery", 1, ["A", "B"], [], max_rounds=2)

    trial = collector.finalize_negotiation(
        outcome="agreement",
        values={"A": 1.0, "B": 1.0},
        max_values={"A": 1.0, "B": 1.0},
    )

    assert experiment.trials == [trial]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("I accept these terms.", True),
        ("We have a deal.", True),
        ("I disagree with that deal.", False),
        ("I propose a deal if you accept later.", False),
    ],
)
def test_agreement_detection_requires_explicit_acceptance(response, expected):
    assert InterpretabilityRunner._response_signals_agreement(response) is expected
