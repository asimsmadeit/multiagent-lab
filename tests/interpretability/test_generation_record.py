"""Contract tests for atomic generation-call records."""

from __future__ import annotations

import json
import hashlib

import pytest
import torch

from interpretability.runtime.model_call import (
    ActivationArtifactRef,
    CallPurpose,
    CaptureMode,
    GenerationRecord,
    GenerationRecorder,
    GenerationCallSpec,
    GENERATION_SCHEMA_VERSION,
    SamplingSettings,
    active_generation_recorder,
    generation_call,
    get_active_generation_call_spec,
    get_active_generation_recorder,
    make_call_id,
    make_activation_artifact_refs,
    make_replay_call_id,
    select_final_acting_call,
)


def _settings(*, effective: bool = False) -> SamplingSettings:
    return SamplingSettings(
        max_tokens=256 if effective else 999,
        temperature=0.1 if effective else 0.01,
        top_p=0.8,
        top_k=12,
        seed=7,
        do_sample=True,
        frequency_penalty=1.0 if effective else None,
    )


def _artifact_hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(
    *,
    sequence: int = 2,
    trial_id: str = "trial-4",
    capture_mode: CaptureMode = CaptureMode.GENERATION_PASS,
    replay_call_id: str | None = None,
) -> GenerationRecord:
    call_id = make_call_id(
        run_id="run-1",
        trial_id=trial_id,
        attempt=0,
        sequence=sequence,
        purpose=CallPurpose.ACTOR_ACTION,
        actor_id="seller",
    )
    artifacts = () if capture_mode is CaptureMode.NONE else (
        ActivationArtifactRef(
            artifact_hash=_artifact_hash("artifact"),
            layer="blocks.2.hook_resid_post",
            stage="residual_post",
            token_index=1,
            shape=(4,),
            dtype="float32",
        ),
    )
    return GenerationRecord(
        call_id=call_id,
        run_id="run-1",
        trial_id=trial_id,
        attempt=0,
        sequence=sequence,
        actor_id="seller",
        purpose=CallPurpose.ACTOR_ACTION,
        assembled_prompt="private fact plus public context",
        input_token_ids=(1, 2),
        requested_sampling=_settings(),
        effective_sampling=_settings(effective=True),
        generation_path="transformer_lens_sampling",
        output_token_ids=(3, 4, 0),
        retained_token_ids=(3, 4),
        output_text="offer",
        terminator="<eos>",
        model_revision="model@abc",
        tokenizer_revision="tokenizer@abc",
        concordia_version="2.4.0",
        capture_mode=capture_mode,
        activation_position=(
            None if capture_mode is CaptureMode.NONE
            else "last_retained_response_token"
        ),
        activation_artifacts=artifacts,
        retained_token_index=1,
        replay_call_id=replay_call_id,
    )


def test_generation_record_is_json_safe_and_records_requested_effective_controls() -> None:
    record = _record()
    payload = record.to_dict()

    json.dumps(payload)
    assert payload["prompt_hash"] == record.prompt_hash
    assert payload["requested_sampling"]["max_tokens"] == 999
    assert payload["effective_sampling"]["max_tokens"] == 256
    assert payload["retained_token_index"] == 1
    assert payload["capture_mode"] == "generation_pass"
    assert record.schema_version == GENERATION_SCHEMA_VERSION == "1.4.0"
    assert GenerationRecord.from_dict(json.loads(json.dumps(payload))) == record


def test_probe_call_purposes_are_distinct_from_behavior_actions() -> None:
    assert CallPurpose.BELIEF_VERIFICATION.value == "belief_verification"
    assert CallPurpose.PLAUSIBILITY.value == "plausibility"
    assert CallPurpose.BELIEF_VERIFICATION is not CallPurpose.ACTOR_ACTION
    assert CallPurpose.PLAUSIBILITY is not CallPurpose.ACTOR_ACTION


def test_generation_record_restores_legacy_capture_position() -> None:
    payload = _record().to_dict()
    payload['schema_version'] = '1.0.0'
    payload.pop('activation_position')
    payload.pop('prompt_hash')

    restored = GenerationRecord.from_legacy_dict(payload)

    assert restored.schema_version == '1.0.0'
    assert restored.activation_position == 'last_retained_response_token'


def test_generation_record_rejects_mismatched_call_identity_and_token_alignment() -> None:
    with pytest.raises(ValueError, match="call_id"):
        GenerationRecord(
            **{**_record().__dict__, "call_id": "call_wrong"},
        )
    with pytest.raises(ValueError, match="prefix"):
        GenerationRecord(
            **{**_record().__dict__, "retained_token_ids": (4,)},
        )
    with pytest.raises(ValueError, match="temperature must be finite"):
        SamplingSettings(
            max_tokens=1,
            temperature=float("nan"),
            top_p=0.8,
            top_k=1,
            seed=1,
            do_sample=True,
        )
    with pytest.raises(ValueError, match="temperature must be non-negative"):
        SamplingSettings(
            max_tokens=1,
            temperature=-0.1,
            top_p=0.8,
            top_k=1,
            seed=1,
            do_sample=True,
        )
    with pytest.raises(ValueError, match="seed must be non-negative"):
        SamplingSettings(
            max_tokens=1,
            temperature=0.1,
            top_p=0.8,
            top_k=1,
            seed=-1,
            do_sample=True,
        )
    with pytest.raises(ValueError, match="repetition_penalty must be positive"):
        SamplingSettings(
            max_tokens=1,
            temperature=0.1,
            top_p=0.8,
            top_k=1,
            seed=1,
            do_sample=True,
            repetition_penalty=0.0,
        )


def test_replay_requires_distinct_call_link_and_non_replay_rejects_one() -> None:
    with pytest.raises(ValueError, match="derived from acting"):
        _record(capture_mode=CaptureMode.TEACHER_FORCED_REPLAY)
    with pytest.raises(ValueError, match="only teacher-forced"):
        _record(replay_call_id="call-replay")

    replay = _record(
        capture_mode=CaptureMode.TEACHER_FORCED_REPLAY,
        replay_call_id=make_replay_call_id(_record().call_id),
    )
    assert replay.capture_mode is CaptureMode.TEACHER_FORCED_REPLAY


def test_capture_requires_artifacts_at_the_exact_retained_token() -> None:
    with pytest.raises(ValueError, match="at least one artifact"):
        GenerationRecord(
            **{**_record().__dict__, "activation_artifacts": ()},
        )
    wrong_artifact = ActivationArtifactRef(
        artifact_hash=_artifact_hash("wrong-position"),
        layer="blocks.2.hook_resid_post",
        stage="residual_post",
        token_index=0,
        shape=(4,),
        dtype="float32",
    )
    with pytest.raises(ValueError, match="equal retained_token_index"):
        GenerationRecord(
            **{**_record().__dict__, "activation_artifacts": (wrong_artifact,)},
        )


def test_replay_identity_is_deterministically_bound_to_acting_call() -> None:
    call_id = _record().call_id
    assert make_replay_call_id(call_id) == make_replay_call_id(call_id)
    with pytest.raises(ValueError, match="derived from acting"):
        _record(
            capture_mode=CaptureMode.TEACHER_FORCED_REPLAY,
            replay_call_id=make_replay_call_id("call_from_another_action"),
        )


def test_generation_deserialization_rejects_hostile_types_and_versions() -> None:
    payload = _record().to_dict()
    payload["requested_sampling"]["do_sample"] = "false"
    with pytest.raises(TypeError, match="boolean"):
        GenerationRecord.from_dict(payload)

    payload = _record().to_dict()
    del payload["requested_sampling"]["top_p"]
    with pytest.raises(ValueError, match="sampling settings.*missing fields"):
        GenerationRecord.from_dict(payload)

    payload = _record().to_dict()
    payload["schema_version"] = "999"
    with pytest.raises(ValueError, match="schema version"):
        GenerationRecord.from_dict(payload)

    payload = _record().to_dict()
    payload["model_revision"] = ""
    with pytest.raises(ValueError, match="model_revision"):
        GenerationRecord.from_dict(payload)

    payload = _record().to_dict()
    del payload["prompt_hash"]
    payload["assembled_prompt"] = "tampered prompt"
    with pytest.raises(ValueError, match="prompt_hash is required"):
        GenerationRecord.from_dict(payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_top_level", "missing fields: fallback_reason"),
        ("unknown_top_level", "unknown fields: future_field"),
        ("unknown_sampling", "unknown fields: future_control"),
        ("unknown_artifact", "unknown fields: future_artifact_field"),
    ],
)
def test_current_generation_restore_requires_exact_nested_schema(
    mutation: str,
    match: str,
) -> None:
    payload = _record().to_dict()
    if mutation == "missing_top_level":
        del payload["fallback_reason"]
    elif mutation == "unknown_top_level":
        payload["future_field"] = None
    elif mutation == "unknown_sampling":
        payload["effective_sampling"]["future_control"] = 1
    else:
        payload["activation_artifacts"][0]["future_artifact_field"] = None

    with pytest.raises(ValueError, match=match):
        GenerationRecord.from_dict(payload)


def test_explicit_legacy_generation_restore_retains_legacy_defaults() -> None:
    payload = _record().to_dict()
    payload["schema_version"] = "1.0.0"
    payload["requested_sampling"].pop("repetition_penalty")
    payload["effective_sampling"].pop("frequency_penalty")
    payload["legacy_note"] = "ignored only by the explicit legacy boundary"

    restored = GenerationRecord.from_legacy_dict(payload)

    assert restored.schema_version == "1.0.0"
    assert restored.requested_sampling.repetition_penalty is None
    assert restored.effective_sampling.frequency_penalty is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"artifact_hash": "junk"}, "sha256"),
        ({"shape": ()}, "positive integer"),
        ({"dtype": ""}, "floating dtype"),
    ],
)
def test_activation_artifact_rejects_unbound_content_contracts(
    overrides,
    match,
) -> None:
    values = {
        "artifact_hash": _artifact_hash("artifact"),
        "layer": "blocks.2.hook_resid_post",
        "stage": "residual_post",
        "token_index": 1,
        "shape": (4,),
        "dtype": "float32",
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=match):
        ActivationArtifactRef(**values)


def test_final_acting_call_filters_trial_before_sequence_selection() -> None:
    current = _record(sequence=2, trial_id="trial-current")
    other_trial = _record(sequence=99, trial_id="trial-other")

    selected = select_final_acting_call(
        (current, other_trial),
        trial_id="trial-current",
        actor_id="seller",
    )

    assert selected is current
    with pytest.raises(LookupError, match="no actor action"):
        select_final_acting_call(
            (other_trial,),
            trial_id="trial-current",
            actor_id="seller",
        )


def test_scoped_recorder_publishes_once_and_restores_context_after_error() -> None:
    recorder = GenerationRecorder("run-1")
    record = _record()

    with active_generation_recorder(recorder):
        assert get_active_generation_recorder() is recorder
        recorder.publish(record)
        with pytest.raises(RuntimeError, match="nested"):
            with active_generation_recorder(GenerationRecorder("run-2")):
                pass
        with pytest.raises(ValueError, match="duplicate"):
            recorder.publish(record)

    assert get_active_generation_recorder() is None
    assert recorder.records == (record,)


def test_recorder_binds_and_defensively_copies_activation_snapshot() -> None:
    activation = torch.tensor([1.0, 2.0, 3.0, 4.0])
    artifacts = make_activation_artifact_refs(
        {"blocks.2.hook_resid_post": activation}, 1
    )
    record = GenerationRecord(
        **{**_record().__dict__, "activation_artifacts": artifacts},
    )
    recorder = GenerationRecorder("run-1")

    recorder.publish(
        record,
        activation_snapshot={"blocks.2.hook_resid_post": activation},
    )
    activation.zero_()
    first = recorder.activation_snapshot(record.call_id)
    first["blocks.2.hook_resid_post"].zero_()
    second = recorder.activation_snapshot(record.call_id)

    assert torch.equal(
        second["blocks.2.hook_resid_post"],
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )


def test_recorder_rejects_snapshot_that_does_not_match_artifact_hash() -> None:
    record = _record()
    recorder = GenerationRecorder("run-1")

    with pytest.raises(ValueError, match="hash mismatch"):
        recorder.publish(
            record,
            activation_snapshot={
                "blocks.2.hook_resid_post": torch.ones(4),
            },
        )


def test_call_scope_requires_matching_recorder_and_derives_replay_identity() -> None:
    spec = GenerationCallSpec(
        run_id="run-1",
        trial_id="trial-4",
        attempt=0,
        sequence=2,
        actor_id="seller",
        purpose=CallPurpose.ACTOR_ACTION,
        model_revision="model@abc",
        tokenizer_revision="tokenizer@abc",
        concordia_version="2.4.0",
        capture_mode=CaptureMode.TEACHER_FORCED_REPLAY,
    )
    with pytest.raises(RuntimeError, match="active recorder"):
        with generation_call(spec):
            pass

    recorder = GenerationRecorder("run-1")
    with active_generation_recorder(recorder):
        with generation_call(spec):
            assert get_active_generation_call_spec() is spec
            assert spec.call_id == _record().call_id
            assert spec.replay_call_id.startswith("replay_")
        assert get_active_generation_call_spec() is None
