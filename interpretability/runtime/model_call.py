"""Immutable generation records and a call-scoped publication channel."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import re
from threading import Lock
from typing import Any, Iterator, Mapping


GENERATION_SCHEMA_VERSION = "1.4.0"
SUPPORTED_GENERATION_SCHEMA_VERSIONS = frozenset({
    "1.0.0", "1.1.0", "1.2.0", GENERATION_SCHEMA_VERSION,
})
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ACTIVATION_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})
_SAMPLING_SETTING_FIELDS = frozenset({
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "do_sample",
    "frequency_penalty",
    "repetition_penalty",
})
_ACTIVATION_ARTIFACT_FIELDS = frozenset({
    "artifact_hash",
    "layer",
    "stage",
    "token_index",
    "shape",
    "dtype",
})
_GENERATION_RECORD_FIELDS = frozenset({
    "schema_version",
    "call_id",
    "run_id",
    "trial_id",
    "attempt",
    "sequence",
    "actor_id",
    "purpose",
    "assembled_prompt",
    "prompt_hash",
    "input_token_ids",
    "requested_sampling",
    "effective_sampling",
    "generation_path",
    "output_token_ids",
    "retained_token_ids",
    "output_text",
    "terminator",
    "model_revision",
    "tokenizer_revision",
    "concordia_version",
    "capture_mode",
    "activation_position",
    "activation_artifacts",
    "retained_token_index",
    "replay_call_id",
    "fallback_reason",
})


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    """Reject missing and forward-schema fields at current persistence boundaries."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


class CallPurpose(str, Enum):
    """Why a model call was made."""

    ACTOR_ACTION = "actor_action"
    COUNTERPART_ACTION = "counterpart_action"
    COMPONENT_ANALYSIS = "component_analysis"
    BELIEF_VERIFICATION = "belief_verification"
    PLAUSIBILITY = "plausibility"
    JUDGE = "judge"
    MONITOR = "monitor"


class CaptureMode(str, Enum):
    """Relationship between output generation and captured activations."""

    NONE = "none"
    GENERATION_PASS = "generation_pass"
    TEACHER_FORCED_REPLAY = "teacher_forced_replay"


@dataclass(frozen=True)
class GenerationCallSpec:
    """Persisted identity and provenance supplied before one adapter call."""

    run_id: str
    trial_id: str
    attempt: int
    sequence: int
    actor_id: str
    purpose: CallPurpose
    model_revision: str
    tokenizer_revision: str
    concordia_version: str
    capture_mode: CaptureMode = CaptureMode.NONE

    def __post_init__(self) -> None:
        if not self.run_id or not self.trial_id or not self.actor_id:
            raise ValueError("run, trial, and actor identities must be explicit")
        if type(self.attempt) is not int or type(self.sequence) is not int:
            raise TypeError("attempt and sequence must be integers")
        if self.attempt < 0 or self.sequence < 0:
            raise ValueError("attempt and sequence must be non-negative")
        if not self.model_revision or not self.tokenizer_revision:
            raise ValueError("model and tokenizer revisions must be explicit")
        if not self.concordia_version:
            raise ValueError("concordia_version must be explicit")

    @property
    def call_id(self) -> str:
        return make_call_id(
            run_id=self.run_id,
            trial_id=self.trial_id,
            attempt=self.attempt,
            sequence=self.sequence,
            purpose=self.purpose,
            actor_id=self.actor_id,
        )

    @property
    def replay_call_id(self) -> str | None:
        if self.capture_mode is not CaptureMode.TEACHER_FORCED_REPLAY:
            return None
        return make_replay_call_id(self.call_id)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def make_call_id(
    *,
    run_id: str,
    trial_id: str,
    attempt: int,
    sequence: int,
    purpose: CallPurpose,
    actor_id: str,
) -> str:
    """Derive a deterministic call ID from persisted execution identity."""
    if not run_id or not trial_id or not actor_id:
        raise ValueError("run, trial, and actor identities must be non-empty")
    if type(attempt) is not int or type(sequence) is not int:
        raise TypeError("attempt and sequence must be integers")
    if attempt < 0 or sequence < 0:
        raise ValueError("attempt and sequence must be non-negative")
    payload = {
        "run_id": run_id,
        "trial_id": trial_id,
        "attempt": attempt,
        "sequence": sequence,
        "purpose": purpose.value,
        "actor_id": actor_id,
    }
    return f"call_{_digest(payload)[:24]}"


def make_replay_call_id(acting_call_id: str) -> str:
    """Bind one teacher-forced replay identity to its acting model call."""
    if not acting_call_id:
        raise ValueError("acting_call_id must be non-empty")
    material = f"{acting_call_id}:teacher_forced_replay".encode("utf-8")
    return f"replay_{hashlib.sha256(material).hexdigest()[:24]}"


@dataclass(frozen=True)
class SamplingSettings:
    """Requested or effective generation controls."""

    max_tokens: int | None
    temperature: float | None
    top_p: float | None
    top_k: int | None
    seed: int | None
    do_sample: bool
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_tokens", "top_k", "seed"):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise TypeError(f"{name} must be an integer or None")
        if type(self.do_sample) is not bool:
            raise TypeError("do_sample must be a boolean")
        if self.max_tokens is not None and self.max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        for name, value in (
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("frequency_penalty", self.frequency_penalty),
            ("repetition_penalty", self.repetition_penalty),
        ):
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(f"{name} must be numeric or None")
                if not math.isfinite(value):
                    raise ValueError(f"{name} must be finite")
        if self.top_p is not None and not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0 and 1")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.temperature is not None and self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if (
            self.repetition_penalty is not None
            and self.repetition_penalty <= 0.0
        ):
            raise ValueError("repetition_penalty must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "do_sample": self.do_sample,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SamplingSettings":
        _require_exact_fields(
            value,
            _SAMPLING_SETTING_FIELDS,
            context="serialized sampling settings",
        )
        if type(value.get("do_sample")) is not bool:
            raise TypeError("serialized do_sample must be a boolean")
        return cls(
            max_tokens=value.get("max_tokens"),
            temperature=value.get("temperature"),
            top_p=value.get("top_p"),
            top_k=value.get("top_k"),
            seed=value.get("seed"),
            do_sample=bool(value.get("do_sample", False)),
            frequency_penalty=value.get("frequency_penalty"),
            repetition_penalty=value.get("repetition_penalty"),
        )

    @classmethod
    def from_legacy_dict(cls, value: Mapping[str, Any]) -> "SamplingSettings":
        """Restore an explicitly selected pre-current generation payload."""
        if not isinstance(value, Mapping):
            raise TypeError("legacy sampling settings must be a mapping")
        if "do_sample" not in value:
            raise ValueError("legacy sampling settings require do_sample")
        if type(value["do_sample"]) is not bool:
            raise TypeError("serialized do_sample must be a boolean")
        return cls(
            max_tokens=value.get("max_tokens"),
            temperature=value.get("temperature"),
            top_p=value.get("top_p"),
            top_k=value.get("top_k"),
            seed=value.get("seed"),
            do_sample=value["do_sample"],
            frequency_penalty=value.get("frequency_penalty"),
            repetition_penalty=value.get("repetition_penalty"),
        )


@dataclass(frozen=True)
class ActivationArtifactRef:
    """Content-addressed reference to one captured activation artifact."""

    artifact_hash: str
    layer: str
    stage: str
    token_index: int
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not _SHA256_PATTERN.fullmatch(self.artifact_hash):
            raise ValueError("artifact_hash must be a sha256: digest")
        if not self.layer or not self.stage:
            raise ValueError("layer and stage must be non-empty")
        if type(self.token_index) is not int or self.token_index < 0:
            raise ValueError("artifact token_index must be non-negative")
        shape = tuple(self.shape)
        if not shape or any(type(item) is not int or item <= 0 for item in shape):
            raise ValueError("artifact shape must contain positive integer dimensions")
        object.__setattr__(self, "shape", shape)
        if self.dtype not in _ACTIVATION_DTYPES:
            raise ValueError("artifact dtype must be a supported floating dtype")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_hash": self.artifact_hash,
            "layer": self.layer,
            "stage": self.stage,
            "token_index": self.token_index,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActivationArtifactRef":
        _require_exact_fields(
            value,
            _ACTIVATION_ARTIFACT_FIELDS,
            context="serialized activation artifact",
        )
        shape = value.get("shape")
        if not isinstance(shape, (list, tuple)):
            raise TypeError("serialized artifact shape must be an array")
        return cls(
            artifact_hash=value["artifact_hash"],
            layer=value["layer"],
            stage=value["stage"],
            token_index=value["token_index"],
            shape=tuple(shape),
            dtype=value["dtype"],
        )

    @classmethod
    def from_legacy_dict(cls, value: Mapping[str, Any]) -> "ActivationArtifactRef":
        """Restore an artifact nested in an explicitly selected legacy record."""
        if not isinstance(value, Mapping):
            raise TypeError("legacy activation artifact must be a mapping")
        shape = value.get("shape")
        if not isinstance(shape, (list, tuple)):
            raise TypeError("serialized artifact shape must be an array")
        return cls(
            artifact_hash=value["artifact_hash"],
            layer=value["layer"],
            stage=value["stage"],
            token_index=value["token_index"],
            shape=tuple(shape),
            dtype=value["dtype"],
        )


def make_activation_artifact_refs(
    activations: Mapping[str, Any],
    retained_token_index: int,
) -> tuple[ActivationArtifactRef, ...]:
    """Snapshot, validate, and content-address call-scoped activations."""
    import torch

    if type(retained_token_index) is not int or retained_token_index < 0:
        raise ValueError("retained_token_index must be non-negative")
    artifacts = []
    for layer, activation in sorted(activations.items()):
        if not isinstance(layer, str) or not layer:
            raise ValueError("activation layer names must be non-empty strings")
        if not isinstance(activation, torch.Tensor):
            raise TypeError("activation snapshots must contain torch.Tensor values")
        tensor = activation.detach().cpu().contiguous()
        if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"activation at {layer} must be non-empty and finite")
        dtype = str(tensor.dtype).removeprefix("torch.")
        stage = (
            "residual_post_sequence_mean"
            if layer.endswith(".mean") else "residual_post"
        )
        metadata = json.dumps(
            {
                "layer": layer,
                "stage": stage,
                "token_index": retained_token_index,
                "shape": list(tensor.shape),
                "dtype": dtype,
            },
            sort_keys=True,
        ).encode("utf-8")
        payload = tensor.view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(metadata + b"\0" + payload).hexdigest()
        artifacts.append(ActivationArtifactRef(
            artifact_hash=f"sha256:{digest}",
            layer=layer,
            stage=stage,
            token_index=retained_token_index,
            shape=tuple(tensor.shape),
            dtype=dtype,
        ))
    return tuple(artifacts)


def _validated_activation_snapshot(
    record: "GenerationRecord",
    activations: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    if not isinstance(activations, Mapping):
        raise TypeError("activation_snapshot must be a mapping")
    expected = {artifact.layer: artifact for artifact in record.activation_artifacts}
    if len(expected) != len(record.activation_artifacts):
        raise ValueError("activation artifact layers must be unique")
    if set(activations) != set(expected):
        raise ValueError("activation snapshot layers do not match artifact references")
    snapshot = {}
    for layer, value in activations.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError("activation snapshots must contain torch.Tensor values")
        tensor = value.detach().cpu().contiguous().clone()
        artifact = expected[layer]
        if tuple(tensor.shape) != artifact.shape:
            raise ValueError(f"activation snapshot shape mismatch at {layer}")
        if str(tensor.dtype).removeprefix("torch.") != artifact.dtype:
            raise ValueError(f"activation snapshot dtype mismatch at {layer}")
        regenerated = make_activation_artifact_refs(
            {layer: tensor}, artifact.token_index
        )[0]
        if regenerated != artifact:
            raise ValueError(f"activation snapshot hash mismatch at {layer}")
        snapshot[layer] = tensor
    return snapshot


@dataclass(frozen=True)
class GenerationRecord:
    """One complete model call published only after capture is bound."""

    call_id: str
    run_id: str
    trial_id: str
    attempt: int
    sequence: int
    actor_id: str
    purpose: CallPurpose
    assembled_prompt: str
    input_token_ids: tuple[int, ...]
    requested_sampling: SamplingSettings
    effective_sampling: SamplingSettings
    generation_path: str
    output_token_ids: tuple[int, ...]
    retained_token_ids: tuple[int, ...]
    output_text: str
    terminator: str | None
    model_revision: str
    tokenizer_revision: str
    concordia_version: str
    capture_mode: CaptureMode = CaptureMode.NONE
    activation_position: str | None = None
    activation_artifacts: tuple[ActivationArtifactRef, ...] = ()
    retained_token_index: int | None = None
    replay_call_id: str | None = None
    fallback_reason: str | None = None
    schema_version: str = GENERATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_token_ids", tuple(self.input_token_ids))
        object.__setattr__(self, "output_token_ids", tuple(self.output_token_ids))
        object.__setattr__(self, "retained_token_ids", tuple(self.retained_token_ids))
        object.__setattr__(
            self, "activation_artifacts", tuple(self.activation_artifacts)
        )
        if self.schema_version not in SUPPORTED_GENERATION_SCHEMA_VERSIONS:
            raise ValueError("unsupported generation record schema version")
        for name in (
            "call_id", "run_id", "trial_id", "actor_id", "generation_path",
            "model_revision", "tokenizer_revision", "concordia_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.attempt) is not int or type(self.sequence) is not int:
            raise TypeError("attempt and sequence must be integers")
        for name in ("input_token_ids", "output_token_ids", "retained_token_ids"):
            if any(type(token_id) is not int or token_id < 0 for token_id in getattr(self, name)):
                raise ValueError(f"{name} must contain non-negative integer token IDs")
        expected_id = make_call_id(
            run_id=self.run_id,
            trial_id=self.trial_id,
            attempt=self.attempt,
            sequence=self.sequence,
            purpose=self.purpose,
            actor_id=self.actor_id,
        )
        if self.call_id != expected_id:
            raise ValueError("call_id does not match the persisted call identity")
        if len(self.retained_token_ids) > len(self.output_token_ids):
            raise ValueError("retained tokens cannot exceed generated output tokens")
        if self.retained_token_ids != self.output_token_ids[:len(self.retained_token_ids)]:
            raise ValueError("retained tokens must be a prefix of output tokens")
        if self.retained_token_ids:
            expected_index = len(self.retained_token_ids) - 1
            if self.retained_token_index != expected_index:
                raise ValueError("retained_token_index must identify the final retained token")
        elif self.retained_token_index is not None:
            raise ValueError("an empty retained output has no retained token index")
        if self.capture_mode is CaptureMode.NONE and self.activation_artifacts:
            raise ValueError("capture_mode='none' cannot include activation artifacts")
        if self.capture_mode is CaptureMode.NONE and self.activation_position is not None:
            raise ValueError("capture_mode='none' has no activation position")
        if self.capture_mode is not CaptureMode.NONE:
            if self.activation_position != "last_retained_response_token":
                raise ValueError(
                    "captured activations must identify the last retained response token"
                )
            if not self.retained_token_ids:
                raise ValueError("activation capture requires a retained response token")
            if not self.activation_artifacts:
                raise ValueError("activation capture requires at least one artifact")
            if any(
                artifact.token_index != self.retained_token_index
                for artifact in self.activation_artifacts
            ):
                raise ValueError(
                    "activation artifact token_index must equal retained_token_index"
                )
        if self.capture_mode is CaptureMode.TEACHER_FORCED_REPLAY:
            expected_replay_id = make_replay_call_id(self.call_id)
            if self.replay_call_id != expected_replay_id:
                raise ValueError(
                    "teacher-forced replay_call_id must be derived from acting call_id"
                )
        elif self.replay_call_id is not None:
            raise ValueError("only teacher-forced replay may set replay_call_id")

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.assembled_prompt.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "call_id": self.call_id,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "attempt": self.attempt,
            "sequence": self.sequence,
            "actor_id": self.actor_id,
            "purpose": self.purpose.value,
            "assembled_prompt": self.assembled_prompt,
            "prompt_hash": self.prompt_hash,
            "input_token_ids": list(self.input_token_ids),
            "requested_sampling": self.requested_sampling.to_dict(),
            "effective_sampling": self.effective_sampling.to_dict(),
            "generation_path": self.generation_path,
            "output_token_ids": list(self.output_token_ids),
            "retained_token_ids": list(self.retained_token_ids),
            "output_text": self.output_text,
            "terminator": self.terminator,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "concordia_version": self.concordia_version,
            "capture_mode": self.capture_mode.value,
            "activation_position": self.activation_position,
            "activation_artifacts": [item.to_dict() for item in self.activation_artifacts],
            "retained_token_index": self.retained_token_index,
            "replay_call_id": self.replay_call_id,
            "fallback_reason": self.fallback_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerationRecord":
        """Restore a current record with mandatory schema and prompt identity."""
        if not isinstance(value, Mapping):
            raise TypeError("serialized generation record must be a mapping")
        if value.get("schema_version") != GENERATION_SCHEMA_VERSION:
            raise ValueError("unsupported generation record schema version")
        serialized_prompt_hash = value.get("prompt_hash")
        if not isinstance(serialized_prompt_hash, str) or not serialized_prompt_hash:
            raise ValueError(
                "prompt_hash is required for persisted generation records"
            )
        _require_exact_fields(
            value,
            _GENERATION_RECORD_FIELDS,
            context="serialized generation record",
        )
        return cls._restore_serialized(value)

    @classmethod
    def from_legacy_dict(cls, value: Mapping[str, Any]) -> "GenerationRecord":
        """Explicitly restore a supported pre-1.4 compatibility record."""
        schema_version = value.get("schema_version")
        if (
            schema_version not in SUPPORTED_GENERATION_SCHEMA_VERSIONS
            or schema_version == GENERATION_SCHEMA_VERSION
        ):
            raise ValueError("unsupported generation record schema version")
        return cls._restore_serialized(value)

    @classmethod
    def _restore_serialized(
        cls,
        value: Mapping[str, Any],
    ) -> "GenerationRecord":
        schema_version = value["schema_version"]
        current = schema_version == GENERATION_SCHEMA_VERSION
        settings_loader = (
            SamplingSettings.from_dict
            if current
            else SamplingSettings.from_legacy_dict
        )
        artifact_loader = (
            ActivationArtifactRef.from_dict
            if current
            else ActivationArtifactRef.from_legacy_dict
        )
        capture_mode = CaptureMode(
            value.get("capture_mode", CaptureMode.NONE.value)
        )
        activation_position = value.get("activation_position")
        if (
            activation_position is None
            and capture_mode is not CaptureMode.NONE
            and value.get("schema_version") == "1.0.0"
        ):
            activation_position = "last_retained_response_token"
        record = cls(
            schema_version=schema_version,
            call_id=value["call_id"],
            run_id=value["run_id"],
            trial_id=value["trial_id"],
            attempt=value["attempt"],
            sequence=value["sequence"],
            actor_id=value["actor_id"],
            purpose=CallPurpose(value["purpose"]),
            assembled_prompt=value["assembled_prompt"],
            input_token_ids=tuple(value.get("input_token_ids", ())),
            requested_sampling=settings_loader(value["requested_sampling"]),
            effective_sampling=settings_loader(value["effective_sampling"]),
            generation_path=value["generation_path"],
            output_token_ids=tuple(value.get("output_token_ids", ())),
            retained_token_ids=tuple(value.get("retained_token_ids", ())),
            output_text=value["output_text"],
            terminator=value.get("terminator"),
            model_revision=value["model_revision"],
            tokenizer_revision=value["tokenizer_revision"],
            concordia_version=value["concordia_version"],
            capture_mode=capture_mode,
            activation_position=activation_position,
            activation_artifacts=tuple(
                artifact_loader(item)
                for item in value.get("activation_artifacts", ())
            ),
            retained_token_index=value.get("retained_token_index"),
            replay_call_id=value.get("replay_call_id"),
            fallback_reason=value.get("fallback_reason"),
        )
        serialized_prompt_hash = value.get("prompt_hash")
        if serialized_prompt_hash is not None and serialized_prompt_hash != record.prompt_hash:
            raise ValueError("serialized prompt_hash does not match assembled_prompt")
        return record


class GenerationRecorder:
    """Thread-safe append-only sink for one explicitly scoped run."""

    def __init__(self, run_id: str) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self.run_id = run_id
        self._records: list[GenerationRecord] = []
        self._call_ids: set[str] = set()
        self._activation_snapshots: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def publish(
        self,
        record: GenerationRecord,
        *,
        activation_snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        if record.run_id != self.run_id:
            raise ValueError("record run_id does not match recorder scope")
        snapshot = None
        if activation_snapshot is not None:
            snapshot = _validated_activation_snapshot(record, activation_snapshot)
        elif record.capture_mode is CaptureMode.NONE:
            snapshot = {}
        with self._lock:
            if record.call_id in self._call_ids:
                raise ValueError(f"duplicate GenerationRecord: {record.call_id}")
            self._records.append(record)
            self._call_ids.add(record.call_id)
            if snapshot is not None:
                self._activation_snapshots[record.call_id] = snapshot

    @property
    def records(self) -> tuple[GenerationRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def checkpoint(self) -> int:
        """Return an index that can delimit records from one acting turn."""
        with self._lock:
            return len(self._records)

    def activation_snapshot(self, call_id: str) -> dict[str, Any]:
        """Return a defensive copy of artifacts published with one exact call."""
        with self._lock:
            if call_id not in self._activation_snapshots:
                raise LookupError(
                    f"no call-scoped activation snapshot for {call_id}"
                )
            return {
                layer: value.detach().cpu().clone()
                for layer, value in self._activation_snapshots[call_id].items()
            }


def select_final_acting_call(
    records: tuple[GenerationRecord, ...],
    *,
    trial_id: str,
    actor_id: str,
    start_index: int = 0,
    attempt: int | None = None,
) -> GenerationRecord:
    """Select one final actor action without relying on publication order alone.

    Component and counterpart calls published after the actor are ignored. A
    checkpoint limits selection to the current turn, while attempt/sequence
    identity determines the final matching action.
    """
    if not trial_id or not actor_id:
        raise ValueError("trial_id and actor_id must be non-empty")
    if not 0 <= start_index <= len(records):
        raise ValueError("start_index is outside the recorder")
    candidates = [
        record
        for record in records[start_index:]
        if record.trial_id == trial_id
        and record.actor_id == actor_id
        and record.purpose is CallPurpose.ACTOR_ACTION
        and (attempt is None or record.attempt == attempt)
    ]
    if not candidates:
        raise LookupError("no actor action matched the explicit selection")
    final_key = max((record.attempt, record.sequence) for record in candidates)
    final = [
        record
        for record in candidates
        if (record.attempt, record.sequence) == final_key
    ]
    if len(final) != 1:
        raise RuntimeError("final actor action is not unique")
    return final[0]


_ACTIVE_RECORDER: ContextVar[GenerationRecorder | None] = ContextVar(
    "active_generation_recorder",
    default=None,
)
_ACTIVE_CALL_SPEC: ContextVar[GenerationCallSpec | None] = ContextVar(
    "active_generation_call_spec",
    default=None,
)


def get_active_generation_recorder() -> GenerationRecorder | None:
    """Return the recorder for the current call context, if one is active."""
    return _ACTIVE_RECORDER.get()


def get_active_generation_call_spec() -> GenerationCallSpec | None:
    """Return the identity for the current adapter call, if declared."""
    return _ACTIVE_CALL_SPEC.get()


@contextmanager
def active_generation_recorder(
    recorder: GenerationRecorder,
) -> Iterator[GenerationRecorder]:
    """Activate one unambiguous recorder; nested scopes fail closed."""
    if get_active_generation_recorder() is not None:
        raise RuntimeError("nested generation recorder scopes are ambiguous")
    token: Token[GenerationRecorder | None] = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_RECORDER.reset(token)


@contextmanager
def generation_call(spec: GenerationCallSpec) -> Iterator[GenerationCallSpec]:
    """Declare one unambiguous call within a matching recorder scope."""
    recorder = get_active_generation_recorder()
    if recorder is None:
        raise RuntimeError("generation_call requires an active recorder")
    if recorder.run_id != spec.run_id:
        raise ValueError("call spec run_id does not match active recorder")
    if get_active_generation_call_spec() is not None:
        raise RuntimeError("nested generation call scopes are ambiguous")
    token: Token[GenerationCallSpec | None] = _ACTIVE_CALL_SPEC.set(spec)
    try:
        yield spec
    finally:
        _ACTIVE_CALL_SPEC.reset(token)
