"""Safe, versioned JSON+NPZ activation-dataset persistence."""

from __future__ import annotations

from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .io import load_array_bundle, load_trusted_legacy_torch, save_array_bundle


ACTIVATION_DATASET_SCHEMA_VERSION = "4.1.0"
_KNOWN_TRACKS = {
    "text_only",
    "single_agent_white_box",
    "bilateral_white_box",
    "theory_of_mind",
    "adaptive",
}
_PROVENANCE_SENTINELS = frozenset({
    "none",
    "not-installed",
    "unknown",
    "unresolved",
})
_CAPTURE_MODES = frozenset({"generation_pass", "teacher_forced_replay"})
_ACTIVATION_MANIFEST_FIELDS = frozenset({
    "activation_dataset_schema_version",
    "schema_registry",
    "schema_registry_checksum",
    "layer_arrays",
    "labels",
    "metadata",
    "config",
    "sae_top_features",
    "generation_records",
    "interaction_events",
    "label_records",
    "intervention_designs",
    "intervention_schedules",
    "intervention_application_logs",
    "split_manifest",
    "pod_info",
    "merge_info",
    "extras",
    "dataset_hash",
})
_LAYER_SPECIFICATION_FIELDS = frozenset({
    "array",
    "key_type",
    "key",
    "source_dtype",
    "stored_dtype",
})


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if isinstance(key, str):
                normalized_key = key
            elif isinstance(key, int) and not isinstance(key, bool):
                normalized_key = str(key)
            else:
                raise TypeError(
                    "activation dataset JSON keys must be strings or integers"
                )
            if normalized_key in result:
                raise ValueError("activation dataset JSON keys collide after encoding")
            result[normalized_key] = _json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("activation dataset JSON floats must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "activation dataset metadata must be JSON-safe, got "
        f"{type(value).__name__}"
    )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _safe_array(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype is torch.bfloat16:
            tensor = tensor.float()
        array = tensor.numpy()
    else:
        array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"array {name!r} has executable object dtype")
    if array.dtype.kind not in "biuf":
        raise TypeError(f"array {name!r} must be numeric or boolean")
    if array.dtype.kind in "f" and not np.isfinite(array).all():
        raise ValueError(f"array {name!r} must contain only finite values")
    if array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and not np.little_endian
    ):
        array = array.astype(array.dtype.newbyteorder("<"))
    return np.ascontiguousarray(array)


def _dataset_hash(
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> str:
    payload = dict(manifest)
    payload.pop("dataset_hash", None)
    digest = hashlib.sha256(_canonical_json(payload))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        descriptor = {
            "name": name,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        }
        digest.update(_canonical_json(descriptor))
        digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def _base_path(path: str | Path) -> Path:
    value = Path(path)
    if value.suffix in {".json", ".npz"}:
        return value.with_suffix("")
    if value.suffix:
        raise ValueError("safe activation datasets use a .json manifest and .npz arrays")
    return value


def _layer_key(value: Any) -> tuple[str, int | str]:
    if isinstance(value, bool):
        raise TypeError("activation layer keys must be integers or strings")
    if isinstance(value, (int, np.integer)):
        return "int", int(value)
    if isinstance(value, str) and value:
        return "str", value
    raise TypeError("activation layer keys must be non-empty integers or strings")


def _validate_provenance(config: Mapping[str, Any]) -> None:
    track = config.get("experiment_track")
    if track not in _KNOWN_TRACKS:
        raise ValueError("config.experiment_track must name a supported access track")
    if track == "text_only":
        raise ValueError("text_only tracks cannot contain activation datasets")
    captured = config.get("captured_actor_ids")
    if not isinstance(captured, list) or not captured:
        raise ValueError("config.captured_actor_ids must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in captured):
        raise ValueError("captured actor IDs must be non-empty strings")
    if len(set(captured)) != len(captured):
        raise ValueError("captured actor IDs must not contain duplicates")
    provenance = config.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("config.provenance is required")
    for kind in ("model", "tokenizer"):
        identity = provenance.get(kind)
        if not isinstance(identity, Mapping):
            raise ValueError(f"config.provenance.{kind} is required")
        for field_name in ("name", "revision"):
            value = identity.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"config.provenance.{kind}.{field_name} must be non-empty"
                )
            if value.strip().lower() in _PROVENANCE_SENTINELS:
                raise ValueError(
                    f"config.provenance.{kind}.{field_name} is unresolved"
                )
    sampling_configs = provenance.get("sampling_configs")
    if not isinstance(sampling_configs, list) or not sampling_configs:
        raise ValueError(
            "config.provenance.sampling_configs must be a non-empty list"
        )
    for index, settings in enumerate(sampling_configs):
        if not isinstance(settings, Mapping):
            raise TypeError(f"sampling config {index} must be a mapping")
        for stage in ("requested", "effective"):
            if not isinstance(settings.get(stage), Mapping):
                raise ValueError(
                    f"sampling config {index} must retain {stage} settings"
                )
        if not isinstance(settings.get("generation_path"), str) or not settings[
            "generation_path"
        ]:
            raise ValueError(
                f"sampling config {index} must retain generation_path"
            )
    capture = provenance.get("capture")
    if not isinstance(capture, Mapping):
        raise ValueError("config.provenance.capture is required")
    if not isinstance(capture.get("layers"), list) or not capture["layers"]:
        raise ValueError("capture provenance must declare non-empty layers")
    if not isinstance(capture.get("activation_positions"), list) or not capture[
        "activation_positions"
    ]:
        raise ValueError("capture provenance must declare activation positions")
    if any(
        not isinstance(position, str)
        or not position.strip()
        or position.strip().lower() in _PROVENANCE_SENTINELS
        for position in capture["activation_positions"]
    ):
        raise ValueError("capture provenance contains unresolved activation positions")
    capture_modes = capture.get("capture_modes")
    if not isinstance(capture_modes, list) or not capture_modes:
        raise ValueError("capture provenance must declare capture modes")
    invalid_modes = {
        str(mode) for mode in capture_modes if mode not in _CAPTURE_MODES
    }
    if invalid_modes:
        raise ValueError(
            "capture provenance contains unresolved or unsupported modes: "
            f"{sorted(invalid_modes)}"
        )


def _record_ids(
    records: Sequence[Mapping[str, Any]],
    *,
    id_key: str,
    record_kind: str,
) -> set[str]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("canonical record collections must be sequences")
    identities = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("canonical record collections must contain mappings")
        identity = record.get(id_key)
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"canonical records require {id_key}")
        identities.append(identity)
        if record_kind == "generation":
            from interpretability.runtime.model_call import GenerationRecord

            restored_identity = GenerationRecord.from_dict(record).call_id
        elif record_kind == "label":
            from interpretability.labels.schema import LabelRecord

            restored_identity = LabelRecord.from_dict(record).label_id
        elif record_kind == "event":
            from negotiation.game_master.adjudication import InteractionEvent

            restored_identity = InteractionEvent.from_dict(record).event_id
        else:  # pragma: no cover - private caller controls this value.
            raise ValueError(f"unknown canonical record kind: {record_kind}")
        if restored_identity != identity:
            raise ValueError(f"canonical {record_kind} record ID is inconsistent")
    if len(set(identities)) != len(identities):
        raise ValueError(f"canonical record {id_key} values must be unique")
    return set(identities)


def _validate_lineage(dataset: Mapping[str, Any]) -> None:
    generation_ids = _record_ids(
        dataset.get("generation_records", ()),
        id_key="call_id",
        record_kind="generation",
    )
    event_ids = _record_ids(
        dataset.get("interaction_events", ()),
        id_key="event_id",
        record_kind="event",
    )
    label_ids = _record_ids(
        dataset.get("label_records", ()),
        id_key="label_id",
        record_kind="label",
    )
    for row in dataset.get("metadata", ()):
        if not isinstance(row, Mapping):
            raise TypeError("activation metadata rows must be mappings")
        generation_id = row.get("generation_record_id")
        if generation_id is not None and generation_id not in generation_ids:
            raise ValueError(
                f"metadata references missing GenerationRecord: {generation_id}"
            )
        event_id = row.get("interaction_event_id")
        if event_id is not None and event_id not in event_ids:
            raise ValueError(
                f"metadata references missing InteractionEvent: {event_id}"
            )
        row_label_ids = row.get("label_record_ids", ())
        if not isinstance(row_label_ids, list) or any(
            not isinstance(item, str) or not item for item in row_label_ids
        ):
            raise ValueError("metadata label_record_ids must be a list of IDs")
        missing_labels = set(row_label_ids).difference(label_ids)
        if missing_labels:
            raise ValueError(
                "metadata references missing LabelRecords: "
                f"{sorted(missing_labels)}"
            )
        if row.get("sample_type") == "negotiation":
            if not isinstance(generation_id, str) or not generation_id:
                raise ValueError(
                    "negotiation rows require a GenerationRecord reference"
                )
            if not isinstance(event_id, str) or not event_id:
                raise ValueError(
                    "negotiation rows require an InteractionEvent reference"
                )
            if not row_label_ids:
                raise ValueError(
                    "negotiation rows require at least one LabelRecord reference"
                )
            generation_record = next(
                record
                for record in dataset.get("generation_records", ())
                if record["call_id"] == generation_id
            )
            if generation_record.get("capture_mode") == "none":
                raise ValueError(
                    "negotiation-row GenerationRecord must contain capture provenance"
                )
            interaction_event = next(
                record
                for record in dataset.get("interaction_events", ())
                if record["event_id"] == event_id
            )
            if interaction_event.get("status") != "committed":
                raise ValueError(
                    "negotiation rows must reference a committed InteractionEvent"
                )
    _validate_intervention_lineage(
        dataset,
        generation_ids=generation_ids,
    )


def _validate_intervention_lineage(
    dataset: Mapping[str, Any],
    *,
    generation_ids: set[str],
    require_schedule_rows: bool = True,
    allow_unprojected_captures: bool = False,
    require_complete_receipts: bool = True,
) -> None:
    """Restore and cross-check every published intervention record.

    Intervention records are content addressed individually, but their IDs are
    only meaningful when the unbound design, trial-bound schedule, application
    log, activation rows, and probe generation calls all agree.  Publication
    therefore treats the complete graph as one fail-closed contract.
    """
    from interpretability.runtime.interventions import (
        InterventionApplicationLog,
        InterventionApplicationReceipt,
        InterventionApplicationStatus,
        InterventionDesign,
        InterventionFamily,
        InterventionSchedule,
        ProbeInterventionPlan,
        ProbeKind,
        ProbeLabelStatus,
    )
    from interpretability.runtime.model_call import (
        CallPurpose,
        CaptureMode,
        GenerationRecord,
    )

    def restore_collection(
        collection_name: str,
        *,
        record_type: type[Any],
        id_name: str,
    ) -> dict[str, Any]:
        raw_records = dataset.get(collection_name)
        if not isinstance(raw_records, list):
            raise TypeError(f"{collection_name} must be a list")
        restored: dict[str, Any] = {}
        encoded_by_id: dict[str, bytes] = {}
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise TypeError(f"{collection_name} entries must be mappings")
            record = record_type.from_dict(raw_record)
            identity = getattr(record, id_name)
            encoded = _canonical_json(dict(raw_record))
            if identity in encoded_by_id:
                if encoded_by_id[identity] != encoded:
                    raise ValueError(
                        f"conflicting duplicate {id_name} in {collection_name}"
                    )
                raise ValueError(
                    f"duplicate {id_name} in {collection_name}: {identity}"
                )
            encoded_by_id[identity] = encoded
            restored[identity] = record
        return restored

    designs = restore_collection(
        "intervention_designs",
        record_type=InterventionDesign,
        id_name="design_id",
    )
    schedules = restore_collection(
        "intervention_schedules",
        record_type=InterventionSchedule,
        id_name="schedule_id",
    )
    logs = restore_collection(
        "intervention_application_logs",
        record_type=InterventionApplicationLog,
        id_name="log_id",
    )
    generation_by_id = {
        str(raw_record["call_id"]): GenerationRecord.from_dict(raw_record)
        for raw_record in dataset.get("generation_records", ())
    }

    schedules_by_trial: dict[tuple[str, str], InterventionSchedule] = {}
    for schedule in schedules.values():
        design = designs.get(schedule.intervention_design_id)
        if design is None:
            raise ValueError(
                "InterventionSchedule references a missing InterventionDesign"
            )
        expected_schedule = design.bind(
            run_id=schedule.run_id,
            trial_id=schedule.trial_id,
            scenario_instance_id=schedule.scenario_instance_id,
        )
        if schedule != expected_schedule:
            raise ValueError(
                "InterventionSchedule does not exactly bind its InterventionDesign"
            )
        trial_key = (schedule.trial_id, schedule.scenario_instance_id)
        if trial_key in schedules_by_trial:
            raise ValueError("one trial instance cannot have multiple schedules")
        schedules_by_trial[trial_key] = schedule

    if set(designs) != {
        schedule.intervention_design_id for schedule in schedules.values()
    }:
        raise ValueError(
            "published InterventionDesign records must be referenced by a schedule"
        )

    logs_by_schedule: dict[str, InterventionApplicationLog] = {}
    receipt_by_id: dict[str, Any] = {}
    for application_log in logs.values():
        schedule = schedules.get(application_log.schedule_id)
        if schedule is None:
            raise ValueError(
                "InterventionApplicationLog references a missing schedule"
            )
        if (
            application_log.run_id != schedule.run_id
            or application_log.trial_id != schedule.trial_id
            or application_log.scenario_instance_id
            != schedule.scenario_instance_id
        ):
            raise ValueError(
                "InterventionApplicationLog identity disagrees with its schedule"
            )
        if schedule.schedule_id in logs_by_schedule:
            raise ValueError("one intervention schedule cannot have multiple logs")
        logs_by_schedule[schedule.schedule_id] = application_log
        plans = {plan.design_id: plan for plan in schedule.plans}
        for receipt in application_log.receipts:
            plan = plans.get(receipt.design_id)
            if plan is None:
                raise ValueError(
                    "intervention receipt references a plan absent from its schedule"
                )
            expected_receipt = InterventionApplicationReceipt.for_plan(
                schedule,
                plan,
                status=receipt.status,
                evidence_call_id=receipt.evidence_call_id,
                label_status=receipt.label_status,
            )
            if receipt != expected_receipt:
                raise ValueError(
                    "intervention receipt does not exactly attest its scheduled plan"
                )
            if receipt.receipt_id in receipt_by_id:
                raise ValueError(
                    "intervention receipt IDs must be globally unique in a dataset"
                )
            receipt_by_id[receipt.receipt_id] = receipt
        if require_complete_receipts and {
            receipt.design_id for receipt in application_log.receipts
        } != set(plans):
            raise ValueError(
                "published intervention logs require one terminal receipt per plan"
            )

    if set(logs_by_schedule) != set(schedules):
        raise ValueError(
            "every published InterventionSchedule requires exactly one "
            "InterventionApplicationLog"
        )

    metadata = dataset.get("metadata", ())
    rows_by_generation_id: dict[str, list[Mapping[str, Any]]] = {}
    published_trial_keys: set[tuple[str, str]] = set()
    for row in metadata:
        trial_id = row.get("trial_id")
        scenario_instance_id = row.get("scenario_instance_id")
        if not isinstance(trial_id, (str, int)) or isinstance(trial_id, bool):
            raise ValueError("activation metadata trial_id is malformed")
        raw_receipt_ids = row.get("intervention_application_receipt_ids")
        if not isinstance(raw_receipt_ids, list) or any(
            not isinstance(receipt_id, str) or not receipt_id
            for receipt_id in raw_receipt_ids
        ):
            raise ValueError(
                "metadata intervention receipt references must be a list of IDs"
            )
        if len(set(raw_receipt_ids)) != len(raw_receipt_ids):
            raise ValueError("metadata intervention receipt IDs must be unique")
        has_scenario_instance = (
            isinstance(scenario_instance_id, str) and bool(scenario_instance_id)
        )
        if not has_scenario_instance:
            if row.get("intervention_design_id") is not None or raw_receipt_ids:
                raise ValueError(
                    "metadata intervention lineage requires a scenario_instance_id"
                )
            generation_id = row.get("generation_record_id")
            if isinstance(generation_id, str) and generation_id:
                rows_by_generation_id.setdefault(generation_id, []).append(row)
            continue
        trial_key = (str(trial_id), scenario_instance_id)
        published_trial_keys.add(trial_key)
        schedule = schedules_by_trial.get(trial_key)
        if schedule is None:
            if row.get("intervention_design_id") is not None or raw_receipt_ids:
                raise ValueError(
                    "metadata intervention lineage exists without a trial schedule"
                )
        else:
            if row.get("intervention_design_id") != schedule.intervention_design_id:
                raise ValueError(
                    "metadata intervention design disagrees with its trial schedule"
                )
            application_log = logs_by_schedule[schedule.schedule_id]
            expected_receipt_ids = [
                receipt.receipt_id for receipt in application_log.receipts
            ]
            if raw_receipt_ids != expected_receipt_ids:
                raise ValueError(
                    "metadata intervention receipts disagree with the trial log"
                )
        generation_id = row.get("generation_record_id")
        if isinstance(generation_id, str) and generation_id:
            rows_by_generation_id.setdefault(generation_id, []).append(row)

    if (
        require_schedule_rows
        and set(schedules_by_trial).difference(published_trial_keys)
    ):
        raise ValueError(
            "intervention schedules cannot describe trials absent from dataset rows"
        )

    expected_probe_evidence: dict[str, tuple[Any, Any]] = {}
    for schedule in schedules.values():
        plan_by_id = {plan.design_id: plan for plan in schedule.plans}
        application_log = logs_by_schedule[schedule.schedule_id]
        for receipt in application_log.receipts:
            plan = plan_by_id[receipt.design_id]
            if isinstance(plan, ProbeInterventionPlan):
                if receipt.status is not InterventionApplicationStatus.APPLIED:
                    continue
                if receipt.label_status is not ProbeLabelStatus.UNKNOWN:
                    raise ValueError(
                        "applied probe receipts must retain an unknown label status"
                    )
                evidence_call_id = receipt.evidence_call_id
                if evidence_call_id in expected_probe_evidence:
                    raise ValueError(
                        "one probe GenerationRecord cannot evidence multiple receipts"
                    )
                expected_probe_evidence[str(evidence_call_id)] = (plan, receipt)
            elif (
                receipt.family is not InterventionFamily.SCRIPTED_OBSERVATION
                or receipt.evidence_call_id is not None
            ):
                raise ValueError(
                    "scripted observation receipts cannot carry generation evidence"
                )

    probe_purposes = {
        CallPurpose.BELIEF_VERIFICATION,
        CallPurpose.PLAUSIBILITY,
    }
    actual_probe_evidence = {
        call_id
        for call_id, record in generation_by_id.items()
        if record.purpose in probe_purposes
    }
    if actual_probe_evidence != set(expected_probe_evidence):
        raise ValueError(
            "probe GenerationRecords do not exactly match intervention receipts"
        )
    if not set(expected_probe_evidence).issubset(generation_ids):
        raise ValueError("probe receipt references a missing GenerationRecord")

    for call_id, (plan, receipt) in expected_probe_evidence.items():
        generation = generation_by_id[call_id]
        expected_purpose = (
            CallPurpose.BELIEF_VERIFICATION
            if plan.kind is ProbeKind.BELIEF_VERIFICATION
            else CallPurpose.PLAUSIBILITY
        )
        if (
            generation.run_id != plan.run_id
            or generation.trial_id != plan.trial_id
            or generation.actor_id != plan.target_actor_id
            or generation.purpose is not expected_purpose
            or generation.attempt != 0
        ):
            raise ValueError(
                "probe GenerationRecord does not exactly match its scheduled plan"
            )
        rows = rows_by_generation_id.get(call_id, [])
        if generation.capture_mode is CaptureMode.NONE:
            if rows:
                raise ValueError(
                    "uncaptured probe GenerationRecord cannot have an activation row"
                )
            continue
        if allow_unprojected_captures and not rows:
            # Interrupted recovery can retain a captured call before terminal
            # ActivationSample projection; the runtime checkpoint owns its
            # activation snapshot until the trial is completed.
            continue
        if len(rows) != 1:
            raise ValueError(
                "captured probe GenerationRecord requires exactly one activation row"
            )
        row = rows[0]
        expected_sample_type = (
            "pre_verification"
            if plan.kind is ProbeKind.BELIEF_VERIFICATION
            else "post_plausibility"
        )
        expected_round = -1 if plan.kind is ProbeKind.BELIEF_VERIFICATION else -2
        if (
            row.get("sample_type") != expected_sample_type
            or row.get("round_num") != expected_round
            or row.get("interaction_event_id") is not None
            or row.get("label_record_ids") != []
            or row.get("actual_deception") is not None
            or row.get("emergent_ground_truth") is not None
            or row.get("actual_deception_projection") is not None
            or row.get("perceived_deception") is not None
            or row.get("is_verification_probe")
            is not (plan.kind is ProbeKind.BELIEF_VERIFICATION)
            or row.get("plausibility_response")
            != (
                generation.output_text
                if plan.kind is ProbeKind.PLAUSIBILITY
                else None
            )
            or receipt.receipt_id
            not in row.get("intervention_application_receipt_ids", ())
        ):
            raise ValueError(
                "probe activation row does not exactly match its receipt evidence"
            )


def _validate_split_projection(dataset: Mapping[str, Any]) -> None:
    from interpretability.data.splits import SplitManifest

    raw_manifest = dataset.get("split_manifest")
    if not isinstance(raw_manifest, Mapping):
        raise ValueError("activation dataset requires a SplitManifest")
    manifest = SplitManifest.from_json(
        json.dumps(raw_manifest, sort_keys=True, allow_nan=False)
    )
    manifest.validate()
    if not manifest.locked:
        raise ValueError("activation dataset SplitManifest must be locked")
    by_trial = {
        assignment.trial_id: assignment for assignment in manifest.assignments
    }
    assignments = list(manifest.assignments)
    parents = list(range(len(assignments)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    family_owner: dict[str, int] = {}
    dyad_owner: dict[str, int] = {}
    for index, assignment in enumerate(assignments):
        for identity, owners in (
            (assignment.trial_family_id, family_owner),
            (assignment.dyad_id, dyad_owner),
        ):
            if identity in owners:
                union(index, owners[identity])
            else:
                owners[identity] = index
    component_trials: dict[int, list[str]] = {}
    for index, assignment in enumerate(assignments):
        component_trials.setdefault(find(index), []).append(assignment.trial_id)
    component_ids = {
        root: "connected_"
        + hashlib.sha256(
            "|".join(sorted(trial_ids)).encode("utf-8")
        ).hexdigest()[:20]
        for root, trial_ids in component_trials.items()
    }
    expected_groups_by_trial = {
        assignment.trial_id: component_ids[find(index)]
        for index, assignment in enumerate(assignments)
    }
    metadata = dataset.get("metadata", ())
    labels = dataset.get("labels", {})
    expected_partitions = []
    expected_groups = []
    group_by_trial: dict[str, str] = {}
    for row in metadata:
        trial_id = str(row.get("trial_id", ""))
        if trial_id not in by_trial:
            raise ValueError("metadata trial is absent from SplitManifest")
        assignment = by_trial[trial_id]
        if row.get("trial_family_id") != assignment.trial_family_id:
            raise ValueError("metadata family disagrees with SplitManifest")
        if row.get("dyad_id") != assignment.dyad_id:
            raise ValueError("metadata dyad disagrees with SplitManifest")
        partition = row.get("split_partition")
        group_id = row.get("split_group_id")
        if partition != assignment.partition:
            raise ValueError("metadata partition disagrees with SplitManifest")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("metadata connected split group is missing")
        if group_id != expected_groups_by_trial[trial_id]:
            raise ValueError("metadata connected split group is inconsistent")
        if row.get("connected_group_id") != group_id:
            raise ValueError("metadata connected-group aliases disagree")
        previous_group = group_by_trial.setdefault(trial_id, group_id)
        if previous_group != group_id:
            raise ValueError("one trial cannot have multiple connected groups")
        expected_partitions.append(partition)
        expected_groups.append(group_id)
    if set(by_trial) != {str(row.get("trial_id", "")) for row in metadata}:
        raise ValueError("SplitManifest contains trials absent from dataset rows")
    if labels.get("split_partitions") != expected_partitions:
        raise ValueError("split_partitions labels are not row-aligned")
    if labels.get("connected_group_ids") != expected_groups:
        raise ValueError("connected_group_ids labels are not row-aligned")
    config = dataset.get("config", {})
    if config.get("split_seed") != manifest.seed:
        raise ValueError("config.split_seed disagrees with SplitManifest")
    if config.get("split_manifest_id") != manifest.manifest_id:
        raise ValueError("config.split_manifest_id is inconsistent")


def _validate_loaded_structure(
    *,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> None:
    """Validate semantic alignment independently of bundle/hash integrity."""
    labels = dataset.get("labels")
    metadata = dataset.get("metadata")
    config = dataset.get("config")
    if not isinstance(labels, Mapping) or not isinstance(
        labels.get("gm_labels"), list
    ):
        raise ValueError("activation dataset requires labels.gm_labels")
    if not isinstance(metadata, list) or not isinstance(config, Mapping):
        raise ValueError("activation dataset metadata and config are required")
    n_samples = len(labels["gm_labels"])
    if n_samples < 1 or config.get("n_samples") != n_samples:
        raise ValueError("activation dataset sample count is inconsistent")
    if len(metadata) != n_samples:
        raise ValueError("activation metadata is not row-aligned")
    for name, values in labels.items():
        if not isinstance(values, list) or len(values) != n_samples:
            raise ValueError(f"label {name!r} is not row-aligned")

    specifications = manifest.get("layer_arrays")
    if not isinstance(specifications, list) or not specifications:
        raise ValueError("activation layer manifest is missing")
    expected_keys: list[int | str] = []
    seen_array_names: set[str] = set()
    for specification in specifications:
        if not isinstance(specification, Mapping):
            raise TypeError("activation layer specifications must be mappings")
        _require_exact_fields(
            specification,
            _LAYER_SPECIFICATION_FIELDS,
            context="activation layer specification",
        )
        array_name = specification.get("array")
        if not isinstance(array_name, str) or array_name in seen_array_names:
            raise ValueError("activation array names must be unique")
        seen_array_names.add(array_name)
        if array_name not in arrays:
            raise ValueError("activation layer array is missing")
        array = arrays[array_name]
        if array.dtype.hasobject or array.dtype.kind not in "biuf":
            raise TypeError("activation arrays must be non-executable numeric arrays")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError("activation arrays must contain finite values")
        if array.ndim != 2 or array.shape[0] != n_samples or array.shape[1] < 1:
            raise ValueError("activation arrays must have shape [n_samples, d_model]")
        if specification.get("stored_dtype") != str(array.dtype):
            raise ValueError("activation stored dtype provenance is inconsistent")
        source_dtype = specification.get("source_dtype")
        if not isinstance(source_dtype, str) or not source_dtype:
            raise ValueError("activation source dtype provenance is missing")
        key_type = specification.get("key_type")
        raw_key = specification.get("key")
        if key_type == "int" and type(raw_key) is int:
            expected_keys.append(raw_key)
        elif key_type == "str" and isinstance(raw_key, str) and raw_key:
            expected_keys.append(raw_key)
        else:
            raise ValueError("activation layer key encoding is invalid")
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("activation layer keys must be unique")
    if config.get("layers") != expected_keys:
        raise ValueError("config.layers does not match the activation manifest")

    has_sae = config.get("has_sae")
    if type(has_sae) is not bool:
        raise TypeError("config.has_sae must be a boolean")
    sae_arrays = {"sae_features", "sae_available_mask"}
    if has_sae:
        if not sae_arrays.issubset(arrays):
            raise ValueError("SAE manifest is missing required arrays")
        sae = arrays["sae_features"]
        mask = arrays["sae_available_mask"]
        if (
            sae.dtype.hasobject
            or sae.dtype.kind not in "biuf"
            or sae.ndim != 2
            or sae.shape[0] != n_samples
            or sae.shape[1] < 1
            or (sae.dtype.kind == "f" and not np.isfinite(sae).all())
        ):
            raise ValueError("SAE features are malformed")
        if mask.dtype.kind != "b" or mask.shape != (n_samples,):
            raise ValueError("SAE availability mask is malformed")
        if config.get("sae_dim") != sae.shape[1]:
            raise ValueError("config.sae_dim does not match SAE features")
        top_rows = manifest.get("sae_top_features")
        if not isinstance(top_rows, list) or len(top_rows) != n_samples:
            raise ValueError("SAE top-feature rows are not aligned")
        for row_index, feature_ids in enumerate(top_rows):
            if not isinstance(feature_ids, list) or any(
                type(feature_id) is not int
                or not 0 <= feature_id < sae.shape[1]
                for feature_id in feature_ids
            ):
                raise ValueError(
                    f"invalid SAE top-feature IDs at row {row_index}"
                )
            if not bool(mask[row_index]) and feature_ids:
                raise ValueError("unavailable SAE rows cannot declare top features")
    elif sae_arrays.intersection(arrays) or manifest.get("sae_top_features"):
        raise ValueError("undeclared SAE payload is present")

    captured = set(config.get("captured_actor_ids", ()))
    for row in metadata:
        if not isinstance(row, Mapping):
            raise TypeError("activation metadata rows must be mappings")
        actor_id = row.get("agent_name")
        if not isinstance(actor_id, str) or actor_id not in captured:
            raise ValueError("metadata actor is not declared in captured_actor_ids")


def _prepare_dataset(
    dataset: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    activations = dataset.get("activations")
    labels = dataset.get("labels")
    metadata = dataset.get("metadata")
    config = dataset.get("config")
    if not isinstance(activations, Mapping) or not activations:
        raise ValueError("activation dataset requires at least one activation layer")
    if not isinstance(labels, Mapping) or "gm_labels" not in labels:
        raise ValueError("activation dataset requires labels.gm_labels")
    if not isinstance(metadata, Sequence) or isinstance(metadata, (str, bytes)):
        raise ValueError("activation dataset metadata must be a row sequence")
    if not isinstance(config, Mapping):
        raise ValueError("activation dataset config is required")

    labels_json = _json_value(labels)
    metadata_json = _json_value(metadata)
    config_json = _json_value(config)
    config_json.pop("dataset_hash", None)
    n_samples = len(labels_json["gm_labels"])
    if n_samples < 1:
        raise ValueError("activation dataset must contain at least one sample")
    if len(metadata_json) != n_samples:
        raise ValueError("metadata rows must align with labels")
    for name, values in labels_json.items():
        if not isinstance(values, list) or len(values) != n_samples:
            raise ValueError(f"label {name!r} must align with the sample axis")

    arrays: dict[str, np.ndarray] = {}
    layer_specs = []
    seen_layer_keys: set[tuple[str, int | str]] = set()
    prepared_layers = []
    for raw_key, value in activations.items():
        key_type, key = _layer_key(raw_key)
        prepared_layers.append((key_type, key, value))
    prepared_layers.sort(
        key=lambda item: (
            item[0],
            item[1] if item[0] == "int" else str(item[1]),
        )
    )
    for index, (key_type, key, value) in enumerate(prepared_layers):
        typed_key = (key_type, key)
        if typed_key in seen_layer_keys:
            raise ValueError("activation layer keys must be unique")
        seen_layer_keys.add(typed_key)
        array_name = f"activation_{index:04d}"
        array = _safe_array(value, name=array_name)
        if array.ndim != 2 or array.shape[0] != n_samples:
            raise ValueError(
                f"activation layer {key!r} must have shape [n_samples, d_model]"
            )
        arrays[array_name] = array
        source_dtype = (
            str(value.dtype)
            if isinstance(value, torch.Tensor)
            else str(np.asarray(value).dtype)
        )
        layer_specs.append(
            {
                "array": array_name,
                "key_type": key_type,
                "key": key,
                "source_dtype": source_dtype,
                "stored_dtype": str(array.dtype),
            }
        )

    raw_has_sae = config_json.get("has_sae", False)
    if not isinstance(raw_has_sae, bool):
        raise TypeError("config.has_sae must be a boolean")
    has_sae = raw_has_sae
    sae_present = dataset.get("sae_features") is not None
    if has_sae != sae_present:
        raise ValueError("config.has_sae must exactly match the SAE array payload")
    sae_top_features = _json_value(dataset.get("sae_top_features", []))
    if sae_present:
        sae = _safe_array(dataset["sae_features"], name="sae_features")
        mask = _safe_array(
            dataset.get("sae_available_mask"), name="sae_available_mask"
        )
        if sae.ndim != 2 or sae.shape[0] != n_samples or sae.shape[1] < 1:
            raise ValueError("SAE features must have shape [n_samples, sae_dim]")
        if mask.dtype.kind != "b" or mask.shape != (n_samples,):
            raise ValueError("SAE availability mask must be boolean and row-aligned")
        if len(sae_top_features) != n_samples:
            raise ValueError("SAE top-feature rows must align with samples")
        for row_index, feature_ids in enumerate(sae_top_features):
            if not isinstance(feature_ids, list) or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or not 0 <= item < sae.shape[1]
                for item in feature_ids
            ):
                raise ValueError(
                    f"invalid SAE top-feature IDs at row {row_index}"
                )
            if not mask[row_index] and feature_ids:
                raise ValueError("unavailable SAE rows cannot declare top features")
        arrays["sae_features"] = sae
        arrays["sae_available_mask"] = mask
        config_json["sae_dim"] = int(sae.shape[1])
    elif sae_top_features:
        raise ValueError("SAE top features require an SAE feature array")

    from interpretability.schema_registry import (
        schema_registry,
        schema_registry_checksum,
    )

    registry = schema_registry()
    config_json["dataset_schema_version"] = ACTIVATION_DATASET_SCHEMA_VERSION
    config_json["schema_registry_checksum"] = schema_registry_checksum()
    config_json["n_samples"] = n_samples
    config_json["layers"] = [spec["key"] for spec in layer_specs]
    _validate_provenance(config_json)
    for row in metadata_json:
        row_track = row.get("experiment_track")
        if row_track is not None and row_track != config_json["experiment_track"]:
            raise ValueError("metadata experiment track does not match config")
        if row.get("agent_name") not in config_json["captured_actor_ids"]:
            raise ValueError(
                "metadata actor is not declared in captured_actor_ids"
            )

    known = {
        "activations",
        "labels",
        "config",
        "metadata",
        "sae_features",
        "sae_top_features",
        "sae_available_mask",
        "generation_records",
        "interaction_events",
        "label_records",
        "intervention_designs",
        "intervention_schedules",
        "intervention_application_logs",
        "split_manifest",
        "pod_info",
        "merge_info",
    }
    manifest = {
        "activation_dataset_schema_version": ACTIVATION_DATASET_SCHEMA_VERSION,
        "schema_registry": registry,
        "schema_registry_checksum": schema_registry_checksum(),
        "layer_arrays": layer_specs,
        "labels": labels_json,
        "metadata": metadata_json,
        "config": config_json,
        "sae_top_features": sae_top_features,
        "generation_records": _json_value(dataset.get("generation_records", [])),
        "interaction_events": _json_value(dataset.get("interaction_events", [])),
        "label_records": _json_value(dataset.get("label_records", [])),
        "intervention_designs": _json_value(
            dataset.get("intervention_designs", [])
        ),
        "intervention_schedules": _json_value(
            dataset.get("intervention_schedules", [])
        ),
        "intervention_application_logs": _json_value(
            dataset.get("intervention_application_logs", [])
        ),
        "split_manifest": _json_value(dataset.get("split_manifest", {})),
        "pod_info": _json_value(dataset.get("pod_info", {})),
        "merge_info": _json_value(dataset.get("merge_info", {})),
        "extras": _json_value(
            {key: value for key, value in dataset.items() if key not in known}
        ),
    }
    lineage_view = {
        "metadata": manifest["metadata"],
        "generation_records": manifest["generation_records"],
        "interaction_events": manifest["interaction_events"],
        "label_records": manifest["label_records"],
        "intervention_designs": manifest["intervention_designs"],
        "intervention_schedules": manifest["intervention_schedules"],
        "intervention_application_logs": manifest[
            "intervention_application_logs"
        ],
    }
    _validate_lineage(lineage_view)
    _validate_split_projection({
        "metadata": manifest["metadata"],
        "labels": manifest["labels"],
        "config": manifest["config"],
        "split_manifest": manifest["split_manifest"],
    })
    manifest["dataset_hash"] = _dataset_hash(arrays, manifest)
    return arrays, manifest


def save_activation_dataset(
    path: str | Path,
    dataset: Mapping[str, Any],
    *,
    trusted_legacy: bool = False,
) -> tuple[Path, Path] | Path:
    """Save JSON+NPZ by default; pickle-capable .pt needs explicit opt-in."""
    destination = Path(path)
    if destination.suffix == ".pt":
        if not trusted_legacy:
            raise PermissionError(
                "Writing legacy .pt activation datasets requires "
                "trusted_legacy=True; use a .json path for safe JSON+NPZ."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(dataset), destination)
        return destination
    arrays, manifest = _prepare_dataset(dataset)
    return save_array_bundle(_base_path(destination), arrays, manifest)


def load_activation_dataset(
    path: str | Path,
    *,
    trusted_legacy: bool = False,
) -> dict[str, Any]:
    """Load a safe activation dataset, or reviewed legacy .pt with opt-in."""
    source = Path(path)
    if source.suffix == ".pt":
        data = load_trusted_legacy_torch(source, trusted=trusted_legacy)
        if not isinstance(data, dict):
            raise TypeError("legacy activation dataset must contain a dictionary")
        return data
    manifest_path = _base_path(source).with_suffix(".json")
    arrays, manifest = load_array_bundle(manifest_path)
    if (
        manifest.get("activation_dataset_schema_version")
        != ACTIVATION_DATASET_SCHEMA_VERSION
    ):
        raise ValueError("unsupported activation dataset schema version")
    _require_exact_fields(
        manifest,
        _ACTIVATION_MANIFEST_FIELDS,
        context="activation dataset manifest",
    )
    stored_registry = manifest.get("schema_registry")
    if not isinstance(stored_registry, Mapping):
        raise ValueError("activation dataset schema registry is missing")
    stored_checksum = hashlib.sha256(_canonical_json(stored_registry)).hexdigest()
    if manifest.get("schema_registry_checksum") != stored_checksum:
        raise ValueError("activation dataset schema-registry checksum mismatch")
    from interpretability.schema_registry import (
        schema_registry,
        schema_registry_checksum,
    )

    current_registry = schema_registry()
    if dict(stored_registry) != current_registry:
        raise ValueError(
            "activation dataset schema registry does not match the current registry"
        )
    if stored_checksum != schema_registry_checksum():
        raise ValueError("activation dataset schema registry is incompatible")
    expected_hash = manifest.get("dataset_hash")
    if not isinstance(expected_hash, str) or expected_hash != _dataset_hash(
        arrays, manifest
    ):
        raise ValueError("activation dataset hash mismatch")

    activations: dict[int | str, torch.Tensor] = {}
    expected_arrays = set()
    for specification in manifest.get("layer_arrays", ()):
        array_name = specification.get("array")
        if not isinstance(array_name, str) or array_name not in arrays:
            raise ValueError("activation layer array is missing")
        key_type = specification.get("key_type")
        raw_key = specification.get("key")
        if key_type == "int" and isinstance(raw_key, int) and not isinstance(
            raw_key, bool
        ):
            key: int | str = raw_key
        elif key_type == "str" and isinstance(raw_key, str) and raw_key:
            key = raw_key
        else:
            raise ValueError("activation layer key encoding is invalid")
        if key in activations:
            raise ValueError("activation layer keys must be unique")
        array = arrays[array_name]
        activations[key] = torch.from_numpy(np.array(array, copy=True))
        expected_arrays.add(array_name)

    if not isinstance(manifest["config"], Mapping):
        raise TypeError("activation dataset config must be a mapping")
    if not isinstance(manifest["extras"], Mapping):
        raise TypeError("activation dataset extras must be a mapping")
    config = dict(manifest["config"])
    config["dataset_hash"] = expected_hash
    extras = dict(manifest["extras"])
    reserved = {
        "activations",
        "labels",
        "config",
        "metadata",
        "generation_records",
        "interaction_events",
        "label_records",
        "intervention_designs",
        "intervention_schedules",
        "intervention_application_logs",
        "split_manifest",
        "pod_info",
        "merge_info",
        "sae_features",
        "sae_available_mask",
        "sae_top_features",
    }
    if reserved.intersection(extras):
        raise ValueError("activation dataset extras contain reserved keys")
    dataset: dict[str, Any] = {
        "activations": activations,
        "labels": dict(manifest.get("labels", {})),
        "config": config,
        "metadata": list(manifest.get("metadata", [])),
        "generation_records": list(manifest.get("generation_records", [])),
        "interaction_events": list(manifest.get("interaction_events", [])),
        "label_records": list(manifest.get("label_records", [])),
        "intervention_designs": list(
            manifest.get("intervention_designs", [])
        ),
        "intervention_schedules": list(
            manifest.get("intervention_schedules", [])
        ),
        "intervention_application_logs": list(
            manifest.get("intervention_application_logs", [])
        ),
        "split_manifest": dict(manifest.get("split_manifest", {})),
        "pod_info": dict(manifest.get("pod_info", {})),
        "merge_info": dict(manifest.get("merge_info", {})),
        **extras,
    }
    if config.get("has_sae"):
        required = {"sae_features", "sae_available_mask"}
        if not required.issubset(arrays):
            raise ValueError("SAE manifest is missing required arrays")
        expected_arrays.update(required)
        dataset["sae_features"] = torch.from_numpy(
            np.array(arrays["sae_features"], copy=True)
        )
        dataset["sae_available_mask"] = arrays[
            "sae_available_mask"
        ].astype(bool).tolist()
        dataset["sae_top_features"] = list(
            manifest.get("sae_top_features", [])
        )
    elif manifest.get("sae_top_features"):
        raise ValueError("SAE top features exist without an SAE payload")
    if set(arrays) != expected_arrays:
        raise ValueError("activation dataset contains undeclared arrays")
    _validate_provenance(config)
    if config.get("dataset_schema_version") != ACTIVATION_DATASET_SCHEMA_VERSION:
        raise ValueError("activation dataset config schema version mismatch")
    if config.get("schema_registry_checksum") != stored_checksum:
        raise ValueError("activation dataset config registry checksum mismatch")
    _validate_loaded_structure(
        arrays=arrays,
        manifest=manifest,
        dataset=dataset,
    )
    _validate_split_projection(dataset)
    _validate_lineage(dataset)
    return dataset


__all__ = [
    "ACTIVATION_DATASET_SCHEMA_VERSION",
    "load_activation_dataset",
    "save_activation_dataset",
]
