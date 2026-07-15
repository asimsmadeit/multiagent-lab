"""Permanent contract tests for the event-sourced event-envelope schema."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from interpretability.events.schema import (
    ACTIVATION_CAPTURED_PAYLOAD_SCHEMA_VERSION,
    EVENT_ENVELOPE_SCHEMA_VERSION,
    ActivationCapturedPayload,
    ArtifactReference,
    EventEnvelope,
    EventHashMismatchError,
    EventPayload,
    OpaqueEventPayload,
    PayloadRegistrationError,
    UnknownPayloadTypeError,
    deterministic_dyad_id,
    deterministic_model_call_id,
    deterministic_trial_id,
    parse_payload,
    register_payload,
    registered_payload_types,
)

RUN_ID = "00000000-0000-4000-8000-000000000001"
ROOT_EVENT_ID = "00000000-0000-4000-8000-000000000002"
CHAIN_EVENT_ID = "00000000-0000-4000-8000-000000000003"
PARENT_EVENT_ID = "00000000-0000-4000-8000-000000000004"
RECORDED_AT = datetime(2026, 7, 13, 12, 30, 45, 123456, tzinfo=timezone.utc)

WIRE_FIELDS = {
    "schema_version",
    "event_id",
    "event_type",
    "run_id",
    "pod_id",
    "trial_id",
    "dyad_id",
    "sequence_num",
    "recorded_at",
    "actor_id",
    "actor_role",
    "model_call_id",
    "parent_event_ids",
    "previous_event_hash",
    "payload",
    "payload_schema_version",
    "content_hash",
}


@register_payload("SchemaContractEvent", "1.0.0")
class SchemaContractPayload(EventPayload):
    amount: float
    tags: tuple[str, ...] = ()


@register_payload("SequenceAdvanced", "1.0.0")
class SequenceAdvancedPayload(EventPayload):
    state: tuple[int, ...]


class DeepMutablePayload(EventPayload):
    nested: tuple[list[int], ...]


class NumericPayload(EventPayload):
    values: tuple[float, ...]


def make_event(**overrides: Any) -> EventEnvelope:
    values: dict[str, Any] = {
        "schema_version": EVENT_ENVELOPE_SCHEMA_VERSION,
        "event_id": ROOT_EVENT_ID,
        "event_type": "SchemaContractEvent",
        "run_id": RUN_ID,
        "pod_id": "pod-0",
        "trial_id": "trial-0",
        "dyad_id": "dyad-0",
        "sequence_num": 0,
        "recorded_at": RECORDED_AT,
        "actor_id": "agent-a",
        "actor_role": "seller",
        "model_call_id": "call-0",
        "parent_event_ids": (),
        "previous_event_hash": None,
        "payload": SchemaContractPayload(amount=12.5, tags=("offer",)),
        "payload_schema_version": "1.0.0",
        "content_hash": None,
    }
    values.update(overrides)
    return EventEnvelope(**values)


def make_artifact(**overrides: Any) -> ArtifactReference:
    values: dict[str, Any] = {
        "artifact_hash": "a" * 64,
        "hook_name": "blocks.4.hook_resid_post",
        "layer": 4,
        "token_selection": "last_retained_token",
        "aggregation": "none",
        "shape": (1, 4096),
        "dtype": "float32",
        "tokenizer_id": "tokenizer-v1",
        "model_revision": "model-v1",
        "source_model_call_id": "call-activation-0",
        "external_uri": None,
        "external_checksum": None,
    }
    values.update(overrides)
    return ArtifactReference(**values)


def test_wire_envelope_has_all_and_only_the_17_canonical_fields() -> None:
    wire = make_event().to_dict()

    assert set(wire) == WIRE_FIELDS
    assert len(wire) == 17
    assert wire["schema_version"] == EVENT_ENVELOPE_SCHEMA_VERSION
    assert wire["recorded_at"] == "2026-07-13T12:30:45.123456Z"
    assert wire["content_hash"] is not None


@pytest.mark.parametrize("field", sorted(WIRE_FIELDS))
def test_every_wire_field_is_required(field: str) -> None:
    wire = make_event().to_dict()
    del wire[field]

    with pytest.raises(ValueError, match="fields are not exact"):
        EventEnvelope.from_dict(wire)


def test_unknown_envelope_fields_fail_closed() -> None:
    wire = make_event().to_dict()
    wire["future_field"] = "not-yet-supported"

    with pytest.raises(ValueError, match=r"unknown=\['future_field'\]"):
        EventEnvelope.from_dict(wire)


def test_exact_dict_and_canonical_json_round_trip() -> None:
    event = make_event()
    encoded = event.to_json()

    assert encoded == json.dumps(
        event.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert EventEnvelope.from_dict(event.to_dict()) == event
    assert EventEnvelope.from_json(encoded) == event
    assert EventEnvelope.from_json(encoded.encode("utf-8")) == event


def test_root_and_chained_events_hash_the_declared_material() -> None:
    root = make_event()
    expected_root = hashlib.sha256(
        root.canonical_semantic_json().encode("utf-8")
    ).hexdigest()
    assert root.previous_event_hash is None
    assert root.content_hash == expected_root

    chained = make_event(
        event_id=CHAIN_EVENT_ID,
        event_type="SequenceAdvanced",
        sequence_num=1,
        parent_event_ids=(root.event_id,),
        previous_event_hash=root.content_hash,
        payload=SequenceAdvancedPayload(state=(1, 2)),
    )
    expected_chained = hashlib.sha256(
        (root.content_hash + chained.canonical_semantic_json()).encode("utf-8")
    ).hexdigest()
    assert chained.content_hash == expected_chained
    chained.verify_content_hash()


def test_recorded_at_is_provenance_not_semantic_content() -> None:
    first = make_event(recorded_at=RECORDED_AT)
    later = make_event(recorded_at=RECORDED_AT + timedelta(days=1))

    assert first.recorded_at != later.recorded_at
    assert first.semantic_dict() == later.semantic_dict()
    assert first.canonical_semantic_json() == later.canonical_semantic_json()
    assert first.content_hash == later.content_hash
    assert first.semantically_equal(later)

    wire = first.to_dict()
    wire["recorded_at"] = "2030-01-01T00:00:00.000000Z"
    replayed = EventEnvelope.from_dict(wire)
    assert replayed.content_hash == first.content_hash


def test_models_are_frozen() -> None:
    event = make_event()
    artifact = make_artifact()

    with pytest.raises(ValidationError, match="frozen"):
        event.sequence_num = 2  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        event.payload.amount = 99.0  # type: ignore[attr-defined,misc]
    with pytest.raises(ValidationError, match="frozen"):
        artifact.layer = 9  # type: ignore[misc]


def test_direct_extra_fields_fail_closed() -> None:
    values = make_event().model_dump()
    values["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EventEnvelope(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_id", "not-a-uuid"),
        ("event_id", "A0000000-0000-4000-8000-000000000002"),
        ("run_id", "{00000000-0000-4000-8000-000000000001}"),
        ("pod_id", "pod with spaces"),
        ("trial_id", ""),
        ("dyad_id", "dyad\n1"),
        ("actor_id", "agent a"),
        ("actor_role", "seller role"),
        ("model_call_id", "call#1"),
        ("event_type", "notPascalCase"),
        ("payload_schema_version", "version with spaces"),
        ("schema_version", "99.0.0"),
    ],
)
def test_invalid_envelope_id_and_version_fields_fail(
    field: str, value: Any
) -> None:
    with pytest.raises((ValueError, ValidationError)):
        make_event(**{field: value})


def test_duplicate_invalid_and_self_parent_ids_fail() -> None:
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        make_event(parent_event_ids=(PARENT_EVENT_ID, PARENT_EVENT_ID))
    with pytest.raises(ValidationError, match="canonical UUID"):
        make_event(parent_event_ids=("bad-parent",))
    with pytest.raises(ValidationError, match="cannot be its own parent"):
        make_event(parent_event_ids=(ROOT_EVENT_ID,))


@pytest.mark.parametrize("sequence_num", [-1, True, False, 1.5, "1"])
def test_invalid_sequence_values_fail(sequence_num: Any) -> None:
    with pytest.raises(ValidationError):
        make_event(sequence_num=sequence_num)


def test_hash_chain_root_rules_fail_closed() -> None:
    with pytest.raises(ValidationError, match="sequence 0"):
        make_event(previous_event_hash="a" * 64)
    with pytest.raises(ValidationError, match="requires previous_event_hash"):
        make_event(sequence_num=1, previous_event_hash=None)


@pytest.mark.parametrize(
    "recorded_at",
    [
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=1))),
        "2026-01-01T00:00:00",
        "2026-01-01T01:00:00+01:00",
    ],
)
def test_naive_and_non_utc_timestamps_fail(recorded_at: Any) -> None:
    with pytest.raises(ValidationError, match="recorded_at"):
        make_event(recorded_at=recorded_at)


@pytest.mark.parametrize(
    ("actor_id", "actor_role"),
    [(None, "seller"), ("agent-a", None)],
)
def test_partial_actor_role_identity_fails(
    actor_id: str | None, actor_role: str | None
) -> None:
    with pytest.raises(ValidationError, match="must be supplied together"):
        make_event(actor_id=actor_id, actor_role=actor_role)


def test_actor_and_role_may_both_be_present_or_both_absent() -> None:
    assert make_event(actor_id="agent-a", actor_role="seller")
    assert make_event(actor_id=None, actor_role=None)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_payload_values_fail(value: float) -> None:
    with pytest.raises(ValidationError):
        SchemaContractPayload(amount=value)
    with pytest.raises(ValidationError):
        NumericPayload(values=(0.0, value))


def test_deep_mutable_payload_containers_fail() -> None:
    with pytest.raises((TypeError, ValidationError), match="mutable container"):
        DeepMutablePayload(nested=([1, 2],))


def test_non_string_payload_keys_fail() -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_payload(
            "SchemaContractEvent",
            "1.0.0",
            {"amount": 1.0, 7: "invalid-key"},  # type: ignore[dict-item]
        )


def test_known_payload_missing_and_extra_fields_fail() -> None:
    wire = make_event().to_dict()
    del wire["payload"]["amount"]
    with pytest.raises(ValidationError):
        EventEnvelope.from_dict(wire)

    wire = make_event().to_dict()
    wire["payload"]["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EventEnvelope.from_dict(wire)


def test_registered_payload_type_and_version_must_match_envelope() -> None:
    with pytest.raises(ValidationError, match="event type does not match"):
        make_event(event_type="SequenceAdvanced")
    with pytest.raises(ValidationError, match="schema does not match"):
        make_event(payload_schema_version="2.0.0")
    with pytest.raises((UnknownPayloadTypeError, ValidationError), match="no payload model"):
        parse_payload("NeverRegistered", "1.0.0", {})


def test_registry_is_read_only_and_duplicate_registration_fails() -> None:
    registry = registered_payload_types()
    assert registry[("SchemaContractEvent", "1.0.0")] is SchemaContractPayload
    with pytest.raises(TypeError):
        registry[("Other", "1.0.0")] = SchemaContractPayload  # type: ignore[index]

    class DuplicatePayload(EventPayload):
        value: int

    with pytest.raises(PayloadRegistrationError, match="already registered"):
        register_payload("SchemaContractEvent", "1.0.0")(DuplicatePayload)


@pytest.mark.parametrize(
    "artifact_hash",
    ["a" * 63, "A" * 64, "sha256:" + "a" * 64, "not-a-checksum"],
)
def test_artifact_hash_must_be_lowercase_sha256(artifact_hash: str) -> None:
    with pytest.raises(ValidationError, match="SHA-256"):
        make_artifact(artifact_hash=artifact_hash)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hook_name", ""),
        ("hook_name", "hook with spaces"),
        ("token_selection", "last token"),
        ("aggregation", ""),
        ("dtype", "float 32"),
        ("tokenizer_id", "tokenizer id"),
        ("model_revision", ""),
        ("source_model_call_id", "call#bad"),
    ],
)
def test_artifact_hook_token_dtype_model_and_call_ids_are_stable(
    field: str, value: str
) -> None:
    with pytest.raises(ValidationError):
        make_artifact(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("layer", -1),
        ("layer", True),
        ("shape", ()),
        ("shape", (-1, 4096)),
        ("shape", (True, 4096)),
    ],
)
def test_artifact_layer_and_shape_invariants_fail(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        make_artifact(**{field: value})


def test_external_artifacts_require_a_valid_paired_checksum() -> None:
    checksum = "b" * 64
    assert make_artifact(
        external_uri="s3://research/activation.npz", external_checksum=checksum
    )
    with pytest.raises(ValidationError, match="supplied together"):
        make_artifact(external_uri="s3://research/activation.npz")
    with pytest.raises(ValidationError, match="supplied together"):
        make_artifact(external_checksum=checksum)
    with pytest.raises(ValidationError, match="SHA-256"):
        make_artifact(
            external_uri="s3://research/activation.npz", external_checksum="bad"
        )


def test_activation_payload_exposes_typed_artifact_reference() -> None:
    artifact = make_artifact()
    payload = ActivationCapturedPayload(artifact=artifact)
    event = make_event(
        event_type="ActivationCaptured",
        payload=payload,
        payload_schema_version=ACTIVATION_CAPTURED_PAYLOAD_SCHEMA_VERSION,
    )

    assert event.artifact_refs == (artifact,)
    assert EventEnvelope.from_json(event.to_json()).artifact_refs == (artifact,)


def test_unknown_payload_rejected_by_default_and_preserved_explicitly() -> None:
    opaque = OpaqueEventPayload.from_payload_dict(
        "FutureEvent", "2.0.0", {"nested": {"value": 3}, "items": [1, 2]}
    )
    future = make_event(
        event_type="FutureEvent",
        payload_schema_version="2.0.0",
        payload=opaque,
    )

    with pytest.raises((UnknownPayloadTypeError, ValidationError), match="no payload model"):
        EventEnvelope.from_json(future.to_json())

    preserved = EventEnvelope.from_json(
        future.to_json(), preserve_unknown_payloads=True
    )
    assert isinstance(preserved.payload, OpaqueEventPayload)
    assert preserved.to_json() == future.to_json()
    assert preserved.content_hash == future.content_hash


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("payload", "amount"), 999.0),
        (("actor_role",), "buyer"),
        (("trial_id",), "trial-tampered"),
        (("model_call_id",), "call-tampered"),
    ],
)
def test_semantic_tampering_is_detected(
    path: tuple[str, ...], replacement: Any
) -> None:
    wire = make_event().to_dict()
    target: dict[str, Any] = wire
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement

    with pytest.raises(ValidationError, match="content_hash mismatch"):
        EventEnvelope.from_dict(wire)


def test_supplied_event_hash_mismatch_is_detected() -> None:
    wire = make_event().to_dict()
    wire["content_hash"] = "f" * 64

    with pytest.raises(ValidationError, match="content_hash mismatch"):
        EventEnvelope.from_dict(wire)


def test_tampered_previous_hash_breaks_chained_event_hash() -> None:
    root = make_event()
    chained = make_event(
        event_id=CHAIN_EVENT_ID,
        event_type="SequenceAdvanced",
        sequence_num=1,
        parent_event_ids=(root.event_id,),
        previous_event_hash=root.content_hash,
        payload=SequenceAdvancedPayload(state=(7,)),
    )
    wire = chained.to_dict()
    wire["previous_event_hash"] = "f" * 64

    with pytest.raises(ValidationError, match="content_hash mismatch"):
        EventEnvelope.from_dict(wire)


def test_invalid_hash_encoding_fails_before_hash_comparison() -> None:
    wire = make_event().to_dict()
    wire["previous_event_hash"] = "sha256:not-hex"
    wire["sequence_num"] = 1

    with pytest.raises(ValidationError, match="SHA-256"):
        EventEnvelope.from_dict(wire)


def test_deterministic_identity_helpers_are_stable_and_domain_separated() -> None:
    trial_a = deterministic_trial_id(RUN_ID, "scenario-7", 42)
    trial_b = deterministic_trial_id(RUN_ID, "scenario-7", 42)
    trial_other_seed = deterministic_trial_id(RUN_ID, "scenario-7", 43)
    dyad_ab = deterministic_dyad_id(RUN_ID, "family-1", ("agent-a", "agent-b"))
    dyad_ba = deterministic_dyad_id(RUN_ID, "family-1", ("agent-b", "agent-a"))
    call = deterministic_model_call_id(
        RUN_ID, trial_a, 3, "agent-a", "actor_action"
    )

    assert trial_a == trial_b
    assert trial_a != trial_other_seed
    assert dyad_ab == dyad_ba
    assert len({trial_a, dyad_ab, call}) == 3
    for value in (trial_a, dyad_ab, call):
        assert len(value) == 36


def test_identity_helpers_reject_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="canonical UUID"):
        deterministic_trial_id("bad-run", "scenario-1", 1)
    with pytest.raises(TypeError, match="trial_seed"):
        deterministic_trial_id(RUN_ID, "scenario-1", True)
    with pytest.raises(ValueError, match="two distinct"):
        deterministic_dyad_id(RUN_ID, "family-1", ("agent-a", "agent-a"))
    with pytest.raises(ValueError, match="non-negative"):
        deterministic_model_call_id(
            RUN_ID, "trial-1", -1, "agent-a", "actor_action"
        )
