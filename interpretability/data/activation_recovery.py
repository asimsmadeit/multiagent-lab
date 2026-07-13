"""Safe, non-publishable recovery bundles for interrupted activation runs."""

from __future__ import annotations

from dataclasses import fields
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .io import load_array_bundle, save_array_bundle
from .schema import ActivationSample


ACTIVATION_RECOVERY_SCHEMA_VERSION = "activation-recovery/3"

_ARRAY_FIELDS = {"activations", "followup_activations"}
_SAMPLE_FIELDS = tuple(field.name for field in fields(ActivationSample))
_MANIFEST_FIELDS = {
    "checkpoint_schema_version",
    "publication_status",
    "reason",
    "sample_fields",
    "samples",
    "generation_records",
    "label_records",
    "interaction_events",
    "intervention_designs",
    "intervention_schedules",
    "intervention_application_logs",
    "runner_state",
    "runtime_checkpoint",
    "runtime_checkpoint_identity",
    "experiment_progress",
    "schema_registry",
    "schema_registry_checksum",
    "recovery_hash",
}
_RUNNER_STATE_FIELDS = {
    "experiment_track",
    "captured_actor_ids",
    "pod_id",
    "trial_id_offset",
    "current_trial_id",
}
_TYPED_MAPPING_TAG = "__activation_recovery_typed_mapping_v1__"
_RUNTIME_CHECKPOINT_FIELDS = {
    "executor_version",
    "scenario_instance",
    "assignment",
    "trial_runner",
    "adjudicator",
    "generation_records",
    "label_records",
    "captured_turns",
    "agent_states",
    "retry_counts",
    "protocol",
    "experiment_track",
    "captured_actor_ids",
    "intervention_schedule",
    "intervention_application_log",
    "interrupted",
}


def _json_value(value: Any) -> Any:
    """Project supported metadata to deterministic, finite JSON values."""
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                normalized = key
            elif isinstance(key, int) and not isinstance(key, bool):
                normalized = str(key)
            else:
                raise TypeError(
                    "activation recovery JSON keys must be strings or integers"
                )
            if normalized in result:
                raise ValueError(
                    "activation recovery JSON keys collide after encoding"
                )
            result[normalized] = _json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("activation recovery JSON floats must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "activation recovery metadata must be JSON-safe, got "
        f"{type(value).__name__}"
    )


def _safe_array(value: Any, *, name: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"recovery activation {name!r} must be a torch tensor")
    tensor = value.detach().cpu()
    if tensor.dtype is torch.bfloat16:
        tensor = tensor.float()
    array = tensor.numpy()
    if array.dtype.hasobject or array.dtype.kind not in "biuf":
        raise TypeError(f"recovery activation {name!r} must be numeric")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise ValueError(f"recovery activation {name!r} must be finite")
    if array.ndim != 1:
        raise ValueError(f"recovery activation {name!r} must be one-dimensional")
    return np.ascontiguousarray(array)


def _sample_json_value(value: Any) -> Any:
    """Encode sample mappings without losing integer key identities."""
    if isinstance(value, Mapping):
        entries = []
        for key, item in value.items():
            if isinstance(key, str):
                key_kind = "str"
            elif isinstance(key, int) and not isinstance(key, bool):
                key_kind = "int"
            else:
                raise TypeError(
                    "activation recovery sample keys must be strings or integers"
                )
            entries.append([key_kind, key, _sample_json_value(item)])
        return {_TYPED_MAPPING_TAG: entries}
    if isinstance(value, (list, tuple)):
        return [_sample_json_value(item) for item in value]
    return _json_value(value)


def _restore_sample_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) != {_TYPED_MAPPING_TAG}:
            raise ValueError("recovery sample mapping encoding is malformed")
        entries = value[_TYPED_MAPPING_TAG]
        if not isinstance(entries, list):
            raise ValueError("recovery sample mapping entries must be an array")
        restored: dict[str | int, Any] = {}
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 3:
                raise ValueError("recovery sample mapping entry is malformed")
            key_kind, key, item = entry
            if key_kind == "str" and isinstance(key, str):
                restored_key: str | int = key
            elif (
                key_kind == "int"
                and isinstance(key, int)
                and not isinstance(key, bool)
            ):
                restored_key = key
            else:
                raise ValueError("recovery sample mapping key is malformed")
            if restored_key in restored:
                raise ValueError("recovery sample mapping contains duplicate keys")
            restored[restored_key] = _restore_sample_json_value(item)
        return restored
    if isinstance(value, list):
        return [_restore_sample_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("recovery sample value is not JSON-safe")


def _record_payloads(
    records: Sequence[Any],
    *,
    id_key: str,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, Mapping):
            payload = dict(record)
        else:
            serializer = getattr(record, "to_dict", None)
            if not callable(serializer):
                raise TypeError(
                    "recovery canonical records must be mappings or expose to_dict()"
                )
            payload = dict(serializer())
        identity = payload.get(id_key)
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"recovery canonical record is missing {id_key}")
        payloads.append(_json_value(payload))
    identities = [payload[id_key] for payload in payloads]
    if len(set(identities)) != len(identities):
        raise ValueError(f"recovery canonical {id_key} values must be unique")
    return payloads


def _deduplicated_record_payloads(
    records: Sequence[Any],
    *,
    id_key: str,
) -> list[dict[str, Any]]:
    """Serialize shared aggregate records with collision-safe de-duplication."""
    by_id: dict[str, dict[str, Any]] = {}
    encoded_by_id: dict[str, str] = {}
    for record in records:
        if isinstance(record, Mapping):
            raw_payload = dict(record)
        else:
            serializer = getattr(record, "to_dict", None)
            if not callable(serializer):
                raise TypeError(
                    "recovery aggregate records must be mappings or expose "
                    "to_dict()"
                )
            raw_payload = dict(serializer())
        identity = raw_payload.get(id_key)
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"recovery aggregate record is missing {id_key}")
        payload = _json_value(raw_payload)
        identity = payload[id_key]
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        if identity in encoded_by_id and encoded_by_id[identity] != encoded:
            raise ValueError(
                f"conflicting recovery canonical {id_key}: {identity}"
            )
        encoded_by_id[identity] = encoded
        by_id.setdefault(identity, payload)
    return [by_id[identity] for identity in sorted(by_id)]


def _intervention_metadata_rows(
    samples: Sequence[ActivationSample],
) -> list[dict[str, Any]]:
    """Project only fields used by canonical intervention graph validation."""
    return [
        {
            "trial_id": sample.trial_id,
            "scenario_instance_id": sample.scenario_instance_id,
            "intervention_design_id": sample.intervention_design_id,
            "intervention_application_receipt_ids": list(
                sample.intervention_application_receipt_ids
            ),
            "generation_record_id": sample.generation_record_id,
            "sample_type": sample.sample_type,
            "round_num": sample.round_num,
            "interaction_event_id": sample.interaction_event_id,
            "label_record_ids": list(sample.label_record_ids),
            "actual_deception": sample.actual_deception,
            "emergent_ground_truth": sample.emergent_ground_truth,
            "actual_deception_projection": sample.actual_deception_projection,
            "perceived_deception": sample.perceived_deception,
            "is_verification_probe": sample.is_verification_probe,
            "plausibility_response": sample.plausibility_response,
        }
        for sample in samples
    ]


def _validate_aggregate_intervention_graph(
    *,
    samples: Sequence[ActivationSample],
    generation_records: list[dict[str, Any]],
    intervention_designs: list[dict[str, Any]],
    intervention_schedules: list[dict[str, Any]],
    intervention_application_logs: list[dict[str, Any]],
    generation_ids: set[str],
) -> tuple[list[Any], list[Any], list[Any]]:
    """Revalidate the same typed graph used by public dataset publication."""
    from interpretability.data.activation_dataset import (
        _validate_intervention_lineage,
    )
    from interpretability.runtime.interventions import (
        InterventionApplicationLog,
        InterventionDesign,
        InterventionSchedule,
    )

    _validate_intervention_lineage(
        {
            "metadata": _intervention_metadata_rows(samples),
            "generation_records": generation_records,
            "intervention_designs": intervention_designs,
            "intervention_schedules": intervention_schedules,
            "intervention_application_logs": intervention_application_logs,
        },
        generation_ids=generation_ids,
        require_schedule_rows=False,
        allow_unprojected_captures=True,
        require_complete_receipts=False,
    )
    return (
        [InterventionDesign.from_dict(item) for item in intervention_designs],
        [InterventionSchedule.from_dict(item) for item in intervention_schedules],
        [
            InterventionApplicationLog.from_dict(item)
            for item in intervention_application_logs
        ],
    )


def _validate_runtime_intervention_aggregate(
    runtime_identity: Mapping[str, Any] | None,
    *,
    intervention_designs: Sequence[Any],
    intervention_schedules: Sequence[Any],
    intervention_application_logs: Sequence[Any],
) -> None:
    if runtime_identity is None:
        return
    expected_ids = {
        "intervention_design_id": {
            item.design_id for item in intervention_designs
        },
        "intervention_schedule_id": {
            item.schedule_id for item in intervention_schedules
        },
        "intervention_application_log_id": {
            item.log_id for item in intervention_application_logs
        },
    }
    declared = {
        name: runtime_identity.get(name) for name in expected_ids
    }
    if all(value is None for value in declared.values()):
        return
    if any(value is None for value in declared.values()):
        raise ValueError(
            "runtime intervention identity is only partially declared"
        )
    for name, identity in declared.items():
        if identity not in expected_ids[name]:
            raise ValueError(
                "runtime intervention lineage is absent from aggregate recovery "
                f"records: {name}={identity}"
            )


def _validate_record_payloads(
    records: Any,
    *,
    id_key: str,
    record_kind: str,
) -> set[str]:
    if not isinstance(records, list):
        raise TypeError("recovery canonical record collections must be arrays")
    identities: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("recovery canonical records must be mappings")
        identity = record.get(id_key)
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"recovery canonical record requires {id_key}")
        if record_kind == "generation":
            from interpretability.runtime.model_call import GenerationRecord

            restored = GenerationRecord.from_dict(record).call_id
        elif record_kind == "label":
            from interpretability.labels.schema import LabelRecord

            restored = LabelRecord.from_dict(record).label_id
        elif record_kind == "event":
            from negotiation.game_master.adjudication import InteractionEvent

            restored = InteractionEvent.from_dict(record).event_id
        else:  # pragma: no cover - private callers control this argument.
            raise ValueError(f"unsupported recovery record kind: {record_kind}")
        if restored != identity:
            raise ValueError(f"recovery {record_kind} record ID is inconsistent")
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ValueError(f"recovery canonical {id_key} values must be unique")
    return set(identities)


def _recovery_hash(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    payload = dict(manifest)
    payload.pop("recovery_hash", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded)
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _runtime_checkpoint_identity(
    runtime_checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if runtime_checkpoint is None:
        return None
    if not isinstance(runtime_checkpoint, Mapping):
        raise TypeError("runtime_checkpoint must be a mapping or None")
    if set(runtime_checkpoint) != _RUNTIME_CHECKPOINT_FIELDS:
        raise ValueError("runtime checkpoint fields do not match the current schema")
    from interpretability.runtime.runner import (
        RUNTIME_EXECUTOR_VERSION,
        CounterbalanceAssignment,
        EmergentTrialExecutor,
    )
    from interpretability.runtime.trial import TrialRunner
    from interpretability.scenarios.compiled import ScenarioInstance

    if runtime_checkpoint.get("executor_version") != RUNTIME_EXECUTOR_VERSION:
        raise ValueError("unsupported runtime executor checkpoint version")
    trial_runner = TrialRunner.from_state(runtime_checkpoint["trial_runner"])
    instance = ScenarioInstance.from_dict(runtime_checkpoint["scenario_instance"])
    assignment = CounterbalanceAssignment.from_dict(
        runtime_checkpoint["assignment"]
    )
    if trial_runner.trial_id != instance.trial_id:
        raise ValueError(
            "runtime checkpoint trial identity disagrees with scenario instance"
        )
    generation_records = runtime_checkpoint.get("generation_records", [])
    label_records = runtime_checkpoint.get("label_records", [])
    generation_ids = _validate_record_payloads(
        generation_records, id_key="call_id", record_kind="generation"
    )
    label_ids = _validate_record_payloads(
        label_records, id_key="label_id", record_kind="label"
    )
    adjudicator = runtime_checkpoint.get("adjudicator")
    if not isinstance(adjudicator, Mapping):
        raise TypeError("runtime checkpoint adjudicator state must be a mapping")
    interaction_event_ids = _validate_record_payloads(
        adjudicator.get("events", []), id_key="event_id", record_kind="event"
    )
    serialized_schedule = runtime_checkpoint["intervention_schedule"]
    serialized_application_log = runtime_checkpoint[
        "intervention_application_log"
    ]
    if (serialized_schedule is None) != (serialized_application_log is None):
        raise ValueError(
            "runtime intervention schedule and application log must both be null "
            "or both be present"
        )
    intervention_design_id: str | None = None
    intervention_schedule_id: str | None = None
    intervention_application_log_id: str | None = None
    schedule = None
    application_log = None
    if serialized_schedule is not None:
        if not isinstance(serialized_schedule, Mapping):
            raise TypeError("runtime intervention schedule must be a mapping")
        if not isinstance(serialized_application_log, Mapping):
            raise TypeError(
                "runtime intervention application log must be a mapping"
            )
        from interpretability.runtime.interventions import (
            InterventionApplicationLog,
            InterventionSchedule,
            calculate_intervention_progress,
        )

        schedule = InterventionSchedule.from_dict(serialized_schedule)
        application_log = InterventionApplicationLog.from_dict(
            serialized_application_log
        )
        if (
            schedule.run_id != trial_runner.run_id
            or schedule.trial_id != trial_runner.trial_id
            or schedule.scenario_instance_id != instance.instance_id
        ):
            raise ValueError(
                "runtime intervention schedule identity disagrees with the trial"
            )
        if (
            application_log.run_id != schedule.run_id
            or application_log.trial_id != schedule.trial_id
            or application_log.scenario_instance_id
            != schedule.scenario_instance_id
            or application_log.schedule_id != schedule.schedule_id
        ):
            raise ValueError(
                "runtime intervention application log disagrees with its schedule"
            )
        committed_action_boundary = sum(
            event.get("status") == "committed"
            for event in adjudicator.get("events", [])
        )
        calculate_intervention_progress(
            schedule,
            application_log,
            current_round=(
                committed_action_boundary // len(assignment.participants)
            ),
            committed_action_boundary=committed_action_boundary,
        )
        missing_evidence_calls = {
            receipt.evidence_call_id
            for receipt in application_log.receipts
            if receipt.evidence_call_id is not None
        }.difference(generation_ids)
        if missing_evidence_calls:
            raise ValueError(
                "runtime intervention receipts reference missing generation "
                f"records: {sorted(missing_evidence_calls)}"
            )
        intervention_design_id = schedule.intervention_design_id
        intervention_schedule_id = schedule.schedule_id
        intervention_application_log_id = application_log.log_id
    EmergentTrialExecutor._validate_intervention_event_lineage(  # pylint: disable=protected-access
        trial_runner,
        schedule,
        application_log,
    )
    EmergentTrialExecutor._validate_completed_intervention_lineage(  # pylint: disable=protected-access
        trial_runner,
        schedule,
        application_log,
    )
    declared_intervention_design = instance.public_state.get(
        "intervention_design_id"
    )
    if (
        "intervention_design_id" in instance.public_state
        and declared_intervention_design != intervention_design_id
    ):
        raise ValueError(
            "runtime intervention design disagrees with the scenario instance"
        )
    track = runtime_checkpoint.get("experiment_track")
    if not isinstance(track, str) or not track:
        raise ValueError("runtime checkpoint experiment track must be explicit")
    from interpretability.scenarios.compiled import validate_execution_protocol

    protocol = validate_execution_protocol(runtime_checkpoint.get("protocol"))
    if instance.public_state.get("protocol") != protocol.value:
        raise ValueError(
            "runtime checkpoint protocol disagrees with the scenario instance"
        )
    compiled_protocol = trial_runner.events[0].payload.get("protocol")
    if compiled_protocol != protocol.value:
        raise ValueError(
            "runtime checkpoint protocol disagrees with the compiled event"
        )
    captured_actor_ids = runtime_checkpoint.get("captured_actor_ids")
    if not isinstance(captured_actor_ids, list) or any(
        not isinstance(actor_id, str) or not actor_id
        for actor_id in captured_actor_ids
    ):
        raise ValueError(
            "runtime checkpoint captured_actor_ids must be an identifier array"
        )
    if len(set(captured_actor_ids)) != len(captured_actor_ids):
        raise ValueError(
            "runtime checkpoint captured_actor_ids must not contain duplicates"
        )
    if track == "text_only":
        if captured_actor_ids:
            raise ValueError("text_only runtime checkpoint cannot capture actors")
    elif not captured_actor_ids:
        raise ValueError("white-box runtime checkpoint requires captured actors")
    if not set(captured_actor_ids).issubset(assignment.participants):
        raise ValueError(
            "runtime checkpoint captured actors are not trial participants"
        )
    compiled_capture_ids = tuple(
        trial_runner.events[0].payload.get("captured_actor_ids", ())
    )
    if tuple(captured_actor_ids) != compiled_capture_ids:
        raise ValueError(
            "runtime checkpoint captured actors disagree with the compiled event"
        )
    if type(runtime_checkpoint.get("interrupted")) is not bool:
        raise TypeError("runtime checkpoint interrupted must be a boolean")
    return {
        "run_id": trial_runner.run_id,
        "trial_id": trial_runner.trial_id,
        "attempt": trial_runner.attempt,
        "state": trial_runner.state.value,
        "scenario_instance_id": instance.instance_id,
        "counterbalance_id": assignment.counterbalance_id,
        "experiment_track": track,
        "protocol": protocol.value,
        "captured_actor_ids": list(captured_actor_ids),
        "generation_record_ids": sorted(generation_ids),
        "label_record_ids": sorted(label_ids),
        "interaction_event_ids": sorted(interaction_event_ids),
        "intervention_design_id": intervention_design_id,
        "intervention_schedule_id": intervention_schedule_id,
        "intervention_application_log_id": intervention_application_log_id,
    }


def save_activation_recovery_checkpoint(
    path: str | Path,
    *,
    samples: Sequence[ActivationSample],
    generation_records: Sequence[Any] = (),
    label_records: Sequence[Any] = (),
    interaction_events: Sequence[Any] = (),
    intervention_designs: Sequence[Any] = (),
    intervention_schedules: Sequence[Any] = (),
    intervention_application_logs: Sequence[Any] = (),
    experiment_track: str,
    captured_actor_ids: Sequence[str],
    pod_id: int,
    trial_id_offset: int,
    current_trial_id: int,
    reason: str | None,
    runtime_checkpoint: Mapping[str, Any] | None = None,
    experiment_progress: Mapping[str, Any] | None = None,
) -> Path:
    """Persist resumable accumulated state without claiming dataset validity."""
    if not isinstance(experiment_track, str) or not experiment_track:
        raise ValueError("activation recovery requires an experiment track")
    actors = list(captured_actor_ids)
    if any(not isinstance(actor, str) or not actor for actor in actors):
        raise ValueError("activation recovery captured actors must be identifiers")
    if experiment_track == "text_only":
        if actors:
            raise ValueError("text_only recovery cannot declare captured actors")
    elif not actors:
        raise ValueError("white-box recovery requires captured actor identities")
    if len(set(actors)) != len(actors):
        raise ValueError("captured actor identities must be unique")
    if any(
        type(value) is not int
        for value in (pod_id, trial_id_offset, current_trial_id)
    ):
        raise TypeError("recovery pod and trial identities must be integers")
    if trial_id_offset < 0 or current_trial_id < trial_id_offset:
        raise ValueError("recovery current trial must not precede its offset")

    arrays: dict[str, np.ndarray] = {
        "__checkpoint__": np.asarray([1], dtype=np.uint8),
    }
    serialized_samples: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, ActivationSample):
            raise TypeError("activation recovery accepts ActivationSample rows")
        row = {
            name: _sample_json_value(getattr(sample, name))
            for name in _SAMPLE_FIELDS
            if name not in _ARRAY_FIELDS
        }
        for field_name in _ARRAY_FIELDS:
            layer_values = getattr(sample, field_name)
            if layer_values is None:
                row[field_name] = None
                continue
            if not isinstance(layer_values, Mapping):
                raise TypeError(f"ActivationSample.{field_name} must be a mapping")
            layer_refs: dict[str, str] = {}
            for layer_index, (layer_name, value) in enumerate(layer_values.items()):
                if not isinstance(layer_name, str) or not layer_name:
                    raise ValueError("recovery activation layer names must be non-empty")
                array_name = (
                    f"sample_{sample_index:08d}_{field_name}_{layer_index:04d}"
                )
                arrays[array_name] = _safe_array(value, name=array_name)
                layer_refs[layer_name] = array_name
            if not layer_refs and field_name == "activations":
                raise ValueError("activation recovery samples require activations")
            row[field_name] = layer_refs
        serialized_samples.append(row)

    runtime_payload = (
        None if runtime_checkpoint is None else _json_value(runtime_checkpoint)
    )
    aggregate_designs = list(intervention_designs)
    aggregate_schedules = list(intervention_schedules)
    aggregate_application_logs = list(intervention_application_logs)
    if runtime_payload is not None and runtime_payload.get(
        "intervention_schedule"
    ) is not None:
        from interpretability.runtime.interventions import (
            InterventionApplicationLog,
            InterventionDesign,
            InterventionSchedule,
        )

        runtime_schedule = InterventionSchedule.from_dict(
            runtime_payload["intervention_schedule"]
        )
        runtime_application_log = InterventionApplicationLog.from_dict(
            runtime_payload["intervention_application_log"]
        )
        runtime_design = InterventionDesign(
            specs=tuple(plan.to_spec() for plan in runtime_schedule.plans)
        )
        aggregate_designs.append(runtime_design)
        aggregate_schedules.append(runtime_schedule)
        aggregate_application_logs.append(runtime_application_log)

    canonical_records = {
        "generation_records": _record_payloads(
            generation_records, id_key="call_id"
        ),
        "label_records": _record_payloads(label_records, id_key="label_id"),
        "interaction_events": _record_payloads(
            interaction_events, id_key="event_id"
        ),
        "intervention_designs": _deduplicated_record_payloads(
            aggregate_designs, id_key="design_id"
        ),
        "intervention_schedules": _deduplicated_record_payloads(
            aggregate_schedules, id_key="schedule_id"
        ),
        "intervention_application_logs": _deduplicated_record_payloads(
            aggregate_application_logs, id_key="log_id"
        ),
    }
    runtime_identity = _runtime_checkpoint_identity(runtime_payload)
    generation_ids = {
        record["call_id"] for record in canonical_records["generation_records"]
    }
    label_ids = {
        record["label_id"] for record in canonical_records["label_records"]
    }
    event_ids = {
        record["event_id"] for record in canonical_records["interaction_events"]
    }
    (
        restored_intervention_designs,
        restored_intervention_schedules,
        restored_intervention_application_logs,
    ) = _validate_aggregate_intervention_graph(
        samples=samples,
        generation_records=canonical_records["generation_records"],
        intervention_designs=canonical_records["intervention_designs"],
        intervention_schedules=canonical_records["intervention_schedules"],
        intervention_application_logs=canonical_records[
            "intervention_application_logs"
        ],
        generation_ids=generation_ids,
    )
    _validate_runtime_intervention_aggregate(
        runtime_identity,
        intervention_designs=restored_intervention_designs,
        intervention_schedules=restored_intervention_schedules,
        intervention_application_logs=restored_intervention_application_logs,
    )
    if runtime_identity is not None:
        if runtime_identity["experiment_track"] != experiment_track:
            raise ValueError(
                "runtime checkpoint track disagrees with recovery runner state"
            )
        if runtime_identity["captured_actor_ids"] != actors:
            raise ValueError(
                "runtime checkpoint captured actors disagree with recovery state"
            )
        if not set(runtime_identity["generation_record_ids"]).issubset(generation_ids):
            raise ValueError(
                "runtime checkpoint generation records are absent from recovery state"
            )
        if not set(runtime_identity["label_record_ids"]).issubset(label_ids):
            raise ValueError(
                "runtime checkpoint label records are absent from recovery state"
            )
        if not set(runtime_identity["interaction_event_ids"]).issubset(event_ids):
            raise ValueError(
                "runtime checkpoint interaction events are absent from recovery state"
            )

    from interpretability.schema_registry import (
        schema_registry,
        schema_registry_checksum,
    )

    manifest: dict[str, Any] = {
        "checkpoint_schema_version": ACTIVATION_RECOVERY_SCHEMA_VERSION,
        "publication_status": "non_publishable_recovery",
        "reason": reason,
        "sample_fields": list(_SAMPLE_FIELDS),
        "samples": serialized_samples,
        **canonical_records,
        "runner_state": {
            "experiment_track": experiment_track,
            "captured_actor_ids": actors,
            "pod_id": pod_id,
            "trial_id_offset": trial_id_offset,
            "current_trial_id": current_trial_id,
        },
        "runtime_checkpoint": runtime_payload,
        "runtime_checkpoint_identity": runtime_identity,
        "experiment_progress": _json_value(experiment_progress or {}),
        "schema_registry": schema_registry(),
        "schema_registry_checksum": schema_registry_checksum(),
    }
    manifest["recovery_hash"] = _recovery_hash(manifest, arrays)
    _, manifest_path = save_array_bundle(path, arrays, manifest)
    return manifest_path


def load_activation_recovery_checkpoint(path: str | Path) -> dict[str, Any]:
    """Restore accumulated activation state without invoking a model callback."""
    arrays, manifest = load_array_bundle(path)
    if set(manifest) != _MANIFEST_FIELDS:
        missing = sorted(_MANIFEST_FIELDS.difference(manifest))
        unknown = sorted(set(manifest).difference(_MANIFEST_FIELDS))
        raise ValueError(
            "activation recovery manifest fields are not exact; "
            f"missing={missing}, unknown={unknown}"
        )
    if manifest.get("checkpoint_schema_version") != ACTIVATION_RECOVERY_SCHEMA_VERSION:
        raise ValueError("unsupported activation recovery schema version")
    if manifest.get("publication_status") != "non_publishable_recovery":
        raise ValueError("activation recovery cannot be treated as a dataset")
    if manifest.get("sample_fields") != list(_SAMPLE_FIELDS):
        raise ValueError("activation recovery sample schema does not match runtime")
    if manifest.get("recovery_hash") != _recovery_hash(manifest, arrays):
        raise ValueError("activation recovery content hash mismatch")

    from interpretability.schema_registry import (
        schema_registry,
        schema_registry_checksum,
    )

    if manifest.get("schema_registry") != schema_registry():
        raise ValueError("activation recovery schema registry is not current")
    if manifest.get("schema_registry_checksum") != schema_registry_checksum():
        raise ValueError("activation recovery schema registry checksum mismatch")

    generation_ids = _validate_record_payloads(
        manifest.get("generation_records"),
        id_key="call_id",
        record_kind="generation",
    )
    label_ids = _validate_record_payloads(
        manifest.get("label_records"),
        id_key="label_id",
        record_kind="label",
    )
    event_ids = _validate_record_payloads(
        manifest.get("interaction_events"),
        id_key="event_id",
        record_kind="event",
    )
    runtime_payload = manifest.get("runtime_checkpoint")
    runtime_identity = _runtime_checkpoint_identity(runtime_payload)
    if runtime_identity != manifest.get("runtime_checkpoint_identity"):
        raise ValueError("activation recovery runtime checkpoint identity mismatch")
    runner_state = manifest.get("runner_state")
    if not isinstance(runner_state, Mapping):
        raise ValueError("activation recovery runner state is malformed")
    if runtime_identity is not None:
        if runtime_identity["experiment_track"] != runner_state.get(
            "experiment_track"
        ):
            raise ValueError("runtime track is inconsistent with recovery state")
        if not set(runtime_identity["generation_record_ids"]).issubset(generation_ids):
            raise ValueError("runtime generation lineage is absent from recovery")
        if not set(runtime_identity["label_record_ids"]).issubset(label_ids):
            raise ValueError("runtime label lineage is absent from recovery")
        if not set(runtime_identity["interaction_event_ids"]).issubset(event_ids):
            raise ValueError("runtime event lineage is absent from recovery")

    rows = manifest.get("samples")
    if not isinstance(rows, list):
        raise ValueError("activation recovery samples must be an array")
    marker = arrays.get("__checkpoint__")
    if (
        marker is None
        or marker.dtype != np.dtype("uint8")
        or marker.shape != (1,)
        or int(marker[0]) != 1
    ):
        raise ValueError("activation recovery checkpoint marker is malformed")
    used_arrays: set[str] = set()
    samples: list[ActivationSample] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != set(_SAMPLE_FIELDS):
            raise ValueError("activation recovery sample fields are malformed")
        restored = {
            name: (
                value
                if name in _ARRAY_FIELDS
                else _restore_sample_json_value(value)
            )
            for name, value in row.items()
        }
        for field_name in _ARRAY_FIELDS:
            refs = restored[field_name]
            if refs is None:
                if field_name == "activations":
                    raise ValueError("recovery sample is missing activations")
                continue
            if not isinstance(refs, Mapping) or not refs:
                raise ValueError(f"recovery {field_name} references are malformed")
            layer_values: dict[str, torch.Tensor] = {}
            for layer_name, array_name in refs.items():
                if (
                    not isinstance(layer_name, str)
                    or not layer_name
                    or not isinstance(array_name, str)
                    or array_name not in arrays
                ):
                    raise ValueError("recovery activation reference is invalid")
                array = arrays[array_name]
                if array.dtype.hasobject or array.dtype.kind not in "biuf":
                    raise TypeError("recovery activations must be numeric")
                if array.ndim != 1:
                    raise ValueError("recovery activations must be one-dimensional")
                if array.dtype.kind == "f" and not np.isfinite(array).all():
                    raise ValueError("recovery activations must be finite")
                if array_name in used_arrays:
                    raise ValueError("recovery activation arrays cannot be reused")
                used_arrays.add(array_name)
                layer_values[layer_name] = torch.from_numpy(array.copy())
            restored[field_name] = layer_values
        sample = ActivationSample(**restored)
        if sample.generation_record_id is not None and (
            sample.generation_record_id not in generation_ids
        ):
            raise ValueError("recovery sample references a missing GenerationRecord")
        if sample.interaction_event_id is not None and (
            sample.interaction_event_id not in event_ids
        ):
            raise ValueError("recovery sample references a missing InteractionEvent")
        if not set(sample.label_record_ids).issubset(label_ids):
            raise ValueError("recovery sample references missing LabelRecords")
        samples.append(sample)
    if used_arrays | {"__checkpoint__"} != set(arrays):
        raise ValueError("activation recovery contains unreferenced arrays")

    (
        intervention_designs,
        intervention_schedules,
        intervention_application_logs,
    ) = _validate_aggregate_intervention_graph(
        samples=samples,
        generation_records=manifest["generation_records"],
        intervention_designs=manifest["intervention_designs"],
        intervention_schedules=manifest["intervention_schedules"],
        intervention_application_logs=manifest[
            "intervention_application_logs"
        ],
        generation_ids=generation_ids,
    )
    _validate_runtime_intervention_aggregate(
        runtime_identity,
        intervention_designs=intervention_designs,
        intervention_schedules=intervention_schedules,
        intervention_application_logs=intervention_application_logs,
    )

    track = runner_state.get("experiment_track")
    actors = runner_state.get("captured_actor_ids")
    if not isinstance(track, str) or not track:
        raise ValueError("activation recovery experiment track is missing")
    if (
        not isinstance(actors, list)
        or any(not isinstance(actor, str) or not actor for actor in actors)
        or len(set(actors)) != len(actors)
    ):
        raise ValueError("activation recovery captured actors are malformed")
    if track == "text_only":
        if actors:
            raise ValueError("text_only recovery cannot contain captured actors")
    elif not actors:
        raise ValueError("white-box recovery requires captured actors")
    if set(runner_state) != _RUNNER_STATE_FIELDS:
        missing = sorted(_RUNNER_STATE_FIELDS.difference(runner_state))
        unknown = sorted(set(runner_state).difference(_RUNNER_STATE_FIELDS))
        raise ValueError(
            "activation recovery runner-state fields are not exact; "
            f"missing={missing}, unknown={unknown}"
        )
    for name in ("pod_id", "trial_id_offset", "current_trial_id"):
        if type(runner_state.get(name)) is not int:
            raise TypeError(f"activation recovery {name} must be an integer")
    if (
        runner_state["trial_id_offset"] < 0
        or runner_state["current_trial_id"] < runner_state["trial_id_offset"]
    ):
        raise ValueError("activation recovery trial identity is invalid")

    from interpretability.labels.schema import LabelRecord
    from interpretability.runtime.model_call import GenerationRecord
    from negotiation.game_master.adjudication import InteractionEvent

    return {
        "activation_samples": samples,
        "generation_records": [
            GenerationRecord.from_dict(record)
            for record in manifest["generation_records"]
        ],
        "label_records": [
            LabelRecord.from_dict(record) for record in manifest["label_records"]
        ],
        "interaction_events": [
            InteractionEvent.from_dict(record)
            for record in manifest["interaction_events"]
        ],
        "intervention_designs": intervention_designs,
        "intervention_schedules": intervention_schedules,
        "intervention_application_logs": intervention_application_logs,
        "runner_state": dict(runner_state),
        "runtime_checkpoint": runtime_payload,
        "runtime_checkpoint_identity": runtime_identity,
        "experiment_progress": dict(manifest.get("experiment_progress", {})),
        "reason": manifest.get("reason"),
        "recovery_hash": manifest["recovery_hash"],
    }


__all__ = [
    "ACTIVATION_RECOVERY_SCHEMA_VERSION",
    "load_activation_recovery_checkpoint",
    "save_activation_recovery_checkpoint",
]
