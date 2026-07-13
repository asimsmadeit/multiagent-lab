from types import SimpleNamespace
from collections import defaultdict

import pytest
import numpy as np
import torch

from interpretability.core.deepeval_detector import DeceptionResult
from interpretability.data import (
    load_activation_dataset,
    load_activation_recovery_checkpoint,
)
from interpretability.evaluation import (
    ActivationSample,
    InterpretabilityRunner,
    _make_intervention_activation_sample,
)
from interpretability.tracks import ExperimentTrack
from interpretability.runtime.trial import TrialState
from interpretability.metrics import MetricsCollector
from interpretability.data.activation_recovery import (
    ACTIVATION_RECOVERY_SCHEMA_VERSION,
    _recovery_hash,
)


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
        trial_family_id=f"family-{trial_id}",
        experiment_track="single_agent_white_box",
        sampling_config={
            "requested": {"temperature": 0.7, "seed": 11},
            "effective": {"temperature": 0.7, "seed": 11},
            "generation_path": "offline_fixture",
        },
    )


def _runner(samples):
    runner = object.__new__(InterpretabilityRunner)
    runner.model = SimpleNamespace(model_name="offline-test-model")
    runner.activation_samples = samples
    runner._pod_id = 0
    runner._trial_id_offset = 0
    runner._trial_id = max(
        (
            int(sample.trial_id)
            for sample in samples
            if isinstance(sample.trial_id, int)
        ),
        default=0,
    )
    runner.experiment_track = ExperimentTrack.SINGLE_AGENT_WHITE_BOX
    runner.captured_actor_ids = ("Negotiator",)
    runner.generation_records = []
    runner.label_records = []
    runner.interaction_events = []
    return runner


def test_save_dataset_preserves_sae_sample_axis_and_label_semantics(tmp_path):
    runner = _runner([
        _sample(trial_id=1),
        _sample(trial_id=2, sae_features={4: 1.5}),
        _sample(trial_id=3),
    ])
    for sample in runner.activation_samples:
        sample.sample_type = "serialization_fixture"
    output = tmp_path / "activations.json"

    manifest_path = runner.save_dataset(str(output))
    data = load_activation_dataset(manifest_path)

    assert data["activations"][2].shape == (3, 3)
    assert data["sae_features"].shape == (3, 5)
    assert data["sae_available_mask"] == [False, True, False]
    assert data["sae_features"][0].count_nonzero() == 0
    assert data["sae_features"][1, 4].item() == 1.5
    assert data["config"]["label_semantics"]["agent_labels"].startswith(
        "acting agent estimate of counterpart"
    )
    assert data["metadata"][0]["activation_position"] == "last_response_token"
    assert manifest_path.with_name("activations_transcripts.jsonl").exists()


def test_save_dataset_rejects_misaligned_activation_layers(tmp_path):
    runner = _runner([
        _sample(trial_id=1),
        _sample(trial_id=2, layer="blocks.3.hook_resid_post"),
    ])

    with pytest.raises(ValueError, match="not aligned"):
        runner.save_dataset(str(tmp_path / "bad.pt"))


class _PublicTom:
    def __init__(self, trust_value=None):
        self._trust_value = trust_value

    def get_counterpart_diagnostics(self):
        return {
            'Bob': {
                'deception_risk': 0.6,
                'emotion_intensity': 0.4,
                'valence': -0.2,
                'dominant_emotion': 'fear',
                'top_goals': ['relationship'],
                'trust': {
                    'value': self._trust_value,
                    'available': self._trust_value is not None,
                    'method': 'test_evidence/v1',
                    'evidence': ['observed cue'] if self._trust_value is not None else [],
                },
                'deception_indicators': {'evasiveness': 0.6},
                'beliefs': [{
                    'level': 1,
                    'available': False,
                    'confidence': None,
                    'proposition': None,
                    'method': 'insufficient_evidence/v1',
                    'evidence': [],
                }],
                'advice': 'Ask for evidence.',
            },
        }

    def get_state(self):
        return {
            'empathy_level': 0.8,
            'recent_emotional_trend': 'insufficient_data',
        }


class _AgentWithPublicTom:
    def __init__(self, tom):
        self._tom = tom

    def get_component(self, name):
        if name == 'TheoryOfMind':
            return self._tom
        if name == 'UncertaintyAware':
            return None
        raise KeyError(name)


def _extraction_runner():
    runner = object.__new__(InterpretabilityRunner)
    runner._component_access_failures = defaultdict(int)
    return runner


def test_runner_uses_public_tom_diagnostics_and_preserves_unknown_trust():
    runner = _extraction_runner()
    agent = _AgentWithPublicTom(_PublicTom())

    labels = runner._extract_agent_labels(agent)
    state = runner._extract_tom_state(agent)

    assert labels['perceived_deception'] == pytest.approx(0.6)
    assert labels['trust_level'] is None
    assert set(state['mental_models']) == {'Bob'}
    assert state['belief_levels'][1]['Bob']['available'] is False
    assert state['deception_indicators']['Bob'] == {'evasiveness': 0.6}
    assert runner._component_access_failures == {}


def test_runner_emits_trust_only_when_public_diagnostic_has_evidence():
    runner = _extraction_runner()
    agent = _AgentWithPublicTom(_PublicTom(trust_value=0.7))

    labels = runner._extract_agent_labels(agent)

    assert labels['trust_level'] == pytest.approx(0.7)


def test_save_dataset_serializes_unknown_trust_as_none(tmp_path):
    samples = [_sample(trial_id=index) for index in (1, 2, 3)]
    samples[0].trust_level = None
    samples[0].perceived_deception = None
    samples[0].ground_truth_evaluation_succeeded = False
    samples[0].ground_truth_evaluation_method = "deepeval_error"
    samples[0].ground_truth_evaluation_error = "TimeoutError: offline timeout"
    for sample in samples:
        sample.sample_type = "serialization_fixture"
    runner = _runner(samples)
    output = tmp_path / 'unknown_trust.json'

    manifest_path = runner.save_dataset(str(output))
    data = load_activation_dataset(manifest_path)

    assert data['metadata'][0]['trust_level'] is None
    assert data['labels']['agent_labels'][0] is None
    assert data['metadata'][0]['ground_truth_evaluation_succeeded'] is False
    assert data['metadata'][0]['ground_truth_evaluation_method'] == 'deepeval_error'
    assert data['metadata'][0]['ground_truth_evaluation_error'].startswith(
        'TimeoutError'
    )


def test_missing_or_nonfinite_tom_risk_remains_unavailable():
    class MissingTomAgent:
        def get_component(self, name):
            raise KeyError(name)

    class MissingRiskTom(_PublicTom):
        def get_counterpart_diagnostics(self):
            return {
                "Bob": {"deception_risk": float("nan"), "trust": {}},
                "Cara": {"trust": {}},
            }

    runner = _extraction_runner()
    assert runner._extract_agent_labels(MissingTomAgent())["perceived_deception"] is None
    assert runner._extract_agent_labels(
        _AgentWithPublicTom(MissingRiskTom())
    )["perceived_deception"] is None


@pytest.mark.parametrize(
    ("sample_type", "round_num"),
    [("pre_verification", -1), ("post_plausibility", -2)],
)
def test_intervention_rows_do_not_encode_honesty_labels(sample_type, round_num):
    sample = _make_intervention_activation_sample(
        sample_type=sample_type,
        round_num=round_num,
        trial_id=1,
        agent_name="Negotiator",
        activations={"blocks.2.hook_resid_post": torch.ones(3)},
        perceived_deception=None,
    )

    assert sample.actual_deception is None
    assert sample.commitment_violation is None
    assert sample.manipulation_score is None
    assert sample.consistency_score is None
    assert sample.emergent_ground_truth is None


def test_deepeval_unavailable_and_failed_results_are_not_honesty():
    runner = _extraction_runner()
    runner._deepeval_detector = None
    unavailable = runner._extract_deepeval_labels(
        "response", "hidden_value", {"true_value": 50}
    )
    assert unavailable["actual_deception"] is None
    assert unavailable["consistency_score"] is None
    assert unavailable["evaluation_succeeded"] is False
    assert unavailable["evaluation_method"] == "deepeval_unavailable"

    class FailedDetector:
        def evaluate_deception(self, action, context):
            del action, context
            return DeceptionResult(
                overall_score=0.0,
                is_deceptive=False,
                reasoning="judge timeout",
                evaluation_succeeded=False,
            )

    runner._deepeval_detector = FailedDetector()
    failed = runner._extract_deepeval_labels(
        "response", "hidden_value", {"true_value": 50}
    )
    assert failed["actual_deception"] is None
    assert failed["evaluation_succeeded"] is False
    assert failed["evaluation_method"] == "deepeval_failed"
    assert failed["evaluation_error"] == "judge timeout"


def test_early_trial_checkpoint_is_safe_resumable_and_not_a_dataset(
    tmp_path,
):
    sample = _sample(trial_id=1)
    sample.sample_type = "serialization_fixture"
    runner = _runner([sample])
    runner.save_dataset = lambda *args, **kwargs: pytest.fail(
        "dataset publication must wait for three independent components"
    )
    checkpoint = tmp_path / "checkpoint_trial001.json"

    written = runner._write_activation_checkpoint(checkpoint)
    outer = __import__("json").loads(written.read_text(encoding="utf-8"))
    payload = outer["manifest"]

    assert payload["checkpoint_schema_version"] == (
        ACTIVATION_RECOVERY_SCHEMA_VERSION
    )
    assert payload["publication_status"] == "non_publishable_recovery"
    assert "three independent" in payload["reason"]
    assert written.with_suffix(".npz").exists()
    with pytest.raises(ValueError, match="activation dataset schema"):
        load_activation_dataset(written)

    restored = load_activation_recovery_checkpoint(written)
    assert len(restored["activation_samples"]) == 1
    assert restored["activation_samples"][0].prompt == sample.prompt
    assert torch.equal(
        restored["activation_samples"][0].activations[
            "blocks.2.hook_resid_post"
        ],
        sample.activations["blocks.2.hook_resid_post"],
    )

    resumed = _runner([])
    resumed.restore_activation_checkpoint(written)
    assert resumed._allocate_trial_id() == 2
    resumed.activation_samples.extend(
        [_sample(trial_id=resumed._trial_id), _sample(trial_id=3)]
    )
    for restored_sample in resumed.activation_samples:
        restored_sample.sample_type = "serialization_fixture"
    published = resumed.save_dataset(str(tmp_path / "resumed.json"))
    assert load_activation_dataset(published)["config"]["n_samples"] == 3


def test_activation_recovery_detects_manifest_tamper_and_missing_arrays(
    tmp_path,
):
    sample = _sample(trial_id=1)
    sample.sample_type = "serialization_fixture"
    runner = _runner([sample])
    written = runner._write_activation_checkpoint(tmp_path / "recovery.json")
    payload = __import__("json").loads(written.read_text(encoding="utf-8"))
    payload["manifest"]["samples"][0]["prompt"] = "tampered prompt"
    written.write_text(
        __import__("json").dumps(payload, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="content hash"):
        load_activation_recovery_checkpoint(written)

    second = runner._write_activation_checkpoint(tmp_path / "missing.json")
    second.with_suffix(".npz").unlink()
    with pytest.raises(FileNotFoundError):
        load_activation_recovery_checkpoint(second)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_manifest",
        "missing_manifest",
        "unknown_runner_state",
        "missing_runner_state",
    ],
)
def test_activation_recovery_rejects_rehashed_schema_extensions(
    tmp_path,
    mutation,
):
    sample = _sample(trial_id=1)
    sample.sample_type = "serialization_fixture"
    runner = _runner([sample])
    written = runner._write_activation_checkpoint(
        tmp_path / f"{mutation}.json"
    )
    payload = __import__("json").loads(written.read_text(encoding="utf-8"))
    manifest = payload["manifest"]
    if mutation == "unknown_manifest":
        manifest["ignored_extension"] = True
    elif mutation == "missing_manifest":
        manifest.pop("reason")
    elif mutation == "unknown_runner_state":
        manifest["runner_state"]["ignored_extension"] = True
    else:
        manifest["runner_state"].pop("current_trial_id")
    with np.load(written.with_suffix(".npz"), allow_pickle=False) as bundle:
        arrays = {name: np.asarray(bundle[name]) for name in bundle.files}
    manifest["recovery_hash"] = _recovery_hash(manifest, arrays)
    written.write_text(
        __import__("json").dumps(payload, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="fields are not exact"):
        load_activation_recovery_checkpoint(written)


def test_activation_recovery_links_runtime_identity_without_model_callbacks(
    tmp_path,
):
    from interpretability.runtime.runner import (
        RUNTIME_EXECUTOR_VERSION,
        CounterbalanceAssignment,
    )
    from interpretability.runtime.trial import TrialRunner
    from negotiation.domain.scenario import RoleView, ScenarioInstance

    public_state = {"rounds": 1, "protocol": "alternating"}
    instance = ScenarioInstance(
        scenario="ultimatum_bluff",
        seed=7,
        trial_id="trial-1",
        trial_family_id="family-1",
        public_state=public_state,
        role_views=(
            RoleView("Negotiator", public_state, {"reservation": 20}),
            RoleView("Counterpart", public_state, {"reservation": 80}),
        ),
        legal_actions=("offer", "accept"),
    )
    assignment = CounterbalanceAssignment(
        role_assignment={
            "actor": "Negotiator",
            "counterpart": "Counterpart",
        },
        first_mover_id="Negotiator",
        counterpart_type="default",
        surface_metadata_variant="default",
    )
    trial_runner = TrialRunner(
        run_id="run-1", trial_id=instance.trial_id, attempt=0
    )
    trial_runner.transition(TrialState.COMPILED, {
        "scenario_instance_id": instance.instance_id,
        "captured_actor_ids": ["Negotiator"],
        "protocol": "alternating",
    })
    runtime_checkpoint = {
        "executor_version": RUNTIME_EXECUTOR_VERSION,
        "scenario_instance": instance.to_dict(),
        "assignment": assignment.to_dict(),
        "trial_runner": trial_runner.get_state(),
        "adjudicator": {},
        "generation_records": [],
        "label_records": [],
        "captured_turns": [],
        "agent_states": {},
        "retry_counts": {},
        "protocol": "alternating",
        "experiment_track": "single_agent_white_box",
        "captured_actor_ids": ["Negotiator"],
        "intervention_schedule": None,
        "intervention_application_log": None,
        "interrupted": False,
    }
    sample = _sample(trial_id=1)
    sample.sample_type = "serialization_fixture"
    sample.scenario_instance_id = instance.instance_id
    runner = _runner([sample])

    written = runner._write_activation_checkpoint(
        tmp_path / "runtime.json",
        runtime_checkpoint=runtime_checkpoint,
        experiment_progress={"completed_trial_number": 1},
    )
    restored = load_activation_recovery_checkpoint(written)

    assert restored["runtime_checkpoint_identity"] == {
        "run_id": "run-1",
        "trial_id": "trial-1",
        "attempt": 0,
        "state": "compiled",
        "scenario_instance_id": instance.instance_id,
        "counterbalance_id": assignment.counterbalance_id,
        "experiment_track": "single_agent_white_box",
        "protocol": "alternating",
        "captured_actor_ids": ["Negotiator"],
        "generation_record_ids": [],
        "label_record_ids": [],
        "interaction_event_ids": [],
        "intervention_design_id": None,
        "intervention_schedule_id": None,
        "intervention_application_log_id": None,
    }
    assert restored["experiment_progress"] == {"completed_trial_number": 1}

    target = _runner([])
    target.model = object()  # No generation/capture method exists to call.
    target.restore_activation_checkpoint(written)
    assert target._restored_runtime_checkpoint == runtime_checkpoint

    mismatched_protocol = dict(runtime_checkpoint)
    mismatched_protocol["protocol"] = "simultaneous"
    with pytest.raises(ValueError, match="protocol disagrees"):
        runner._write_activation_checkpoint(
            tmp_path / "bad-protocol.json",
            runtime_checkpoint=mismatched_protocol,
        )


def test_activation_recovery_binds_intervention_schedule_and_log_identity(
    tmp_path,
):
    from interpretability.runtime.interventions import (
        InterventionApplicationLog,
        InterventionDesign,
        ScriptedObservationKind,
        ScriptedObservationSpec,
    )
    from interpretability.runtime.runner import (
        RUNTIME_EXECUTOR_VERSION,
        CounterbalanceAssignment,
    )
    from interpretability.runtime.trial import TrialRunner
    from negotiation.domain.scenario import RoleView, ScenarioInstance

    public_state = {"rounds": 1, "protocol": "alternating"}
    instance = ScenarioInstance(
        scenario="ultimatum_bluff",
        seed=7,
        trial_id="trial-intervention",
        trial_family_id="family-intervention",
        public_state=public_state,
        role_views=(
            RoleView("Negotiator", public_state, {"reservation": 20}),
            RoleView("Counterpart", public_state, {"reservation": 80}),
        ),
        legal_actions=("offer", "accept"),
    )
    assignment = CounterbalanceAssignment(
        role_assignment={
            "actor": "Negotiator",
            "counterpart": "Counterpart",
        },
        first_mover_id="Negotiator",
        counterpart_type="default",
        surface_metadata_variant="default",
    )
    design = InterventionDesign(specs=(ScriptedObservationSpec(
        kind=ScriptedObservationKind.REGISTERED_TEMPLATE,
        target_actor_id="Negotiator",
        scheduled_round=0,
        committed_action_boundary=0,
        sequence=0,
        enabled=True,
        source="registered:test/v1",
        content="A public bulletin is now available.",
    ),))
    schedule = design.bind(
        run_id="run-1",
        trial_id=instance.trial_id,
        scenario_instance_id=instance.instance_id,
    )
    application_log = InterventionApplicationLog.empty(schedule)
    trial_runner = TrialRunner(run_id="run-1", trial_id=instance.trial_id)
    trial_runner.transition(TrialState.COMPILED, {
        "scenario_instance_id": instance.instance_id,
        "captured_actor_ids": ["Negotiator"],
        "protocol": "alternating",
    })
    runtime_checkpoint = {
        "executor_version": RUNTIME_EXECUTOR_VERSION,
        "scenario_instance": instance.to_dict(),
        "assignment": assignment.to_dict(),
        "trial_runner": trial_runner.get_state(),
        "adjudicator": {},
        "generation_records": [],
        "label_records": [],
        "captured_turns": [],
        "agent_states": {},
        "retry_counts": {},
        "protocol": "alternating",
        "experiment_track": "single_agent_white_box",
        "captured_actor_ids": ["Negotiator"],
        "intervention_schedule": schedule.to_dict(),
        "intervention_application_log": application_log.to_dict(),
        "interrupted": True,
    }
    sample = _sample(trial_id=2)
    sample.sample_type = "serialization_fixture"
    runner = _runner([sample])

    written = runner._write_activation_checkpoint(
        tmp_path / "intervention-runtime.json",
        runtime_checkpoint=runtime_checkpoint,
    )
    identity = load_activation_recovery_checkpoint(written)[
        "runtime_checkpoint_identity"
    ]

    assert identity["intervention_design_id"] == design.design_id
    assert identity["intervention_schedule_id"] == schedule.schedule_id
    assert identity["intervention_application_log_id"] == application_log.log_id

    incompatible_design = InterventionDesign(specs=(ScriptedObservationSpec(
        kind=ScriptedObservationKind.REGISTERED_TEMPLATE,
        target_actor_id="Negotiator",
        scheduled_round=0,
        committed_action_boundary=0,
        sequence=0,
        enabled=True,
        source="registered:test/v1",
        content="A different bulletin is now available.",
    ),))
    incompatible_schedule = incompatible_design.bind(
        run_id="run-1",
        trial_id=instance.trial_id,
        scenario_instance_id=instance.instance_id,
    )
    mismatched = dict(runtime_checkpoint)
    mismatched["intervention_schedule"] = incompatible_schedule.to_dict()
    with pytest.raises(ValueError, match="log disagrees with its schedule"):
        runner._write_activation_checkpoint(
            tmp_path / "mismatched-intervention-runtime.json",
            runtime_checkpoint=mismatched,
        )


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
