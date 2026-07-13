"""Activation-dataset projection and safe JSON+NPZ serialization."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
import hashlib
import json
import logging
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

from ..data import (
    ACTIVATION_DATASET_SCHEMA_VERSION,
    ActivationSample,
    load_activation_dataset,
    save_activation_dataset,
)
from ..data.splits import SplitManifest
from .qc_filter import QC_VERSION, classify_sample_response

logger = logging.getLogger(__name__)


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def _record_mapping(record: Any, *, id_key: str) -> dict[str, Any]:
    if isinstance(record, Mapping):
        payload = dict(record)
    else:
        serializer = getattr(record, "to_dict", None)
        if not callable(serializer):
            raise TypeError("canonical records must be mappings or expose to_dict()")
        payload = dict(serializer())
    identity = payload.get(id_key)
    if not isinstance(identity, str) or not identity:
        raise ValueError(f"canonical record is missing {id_key}")
    return payload


def _serialize_records(records: Iterable[Any], *, id_key: str) -> list[dict[str, Any]]:
    payloads = [_record_mapping(record, id_key=id_key) for record in records]
    identities = [payload[id_key] for payload in payloads]
    if len(set(identities)) != len(identities):
        raise ValueError(f"canonical record {id_key} values must be unique")
    return payloads


def _serialize_deduplicated_records(
    records: Iterable[Any],
    *,
    id_key: str,
) -> list[dict[str, Any]]:
    """Canonicalize shared records while rejecting identity collisions."""
    by_id: dict[str, dict[str, Any]] = {}
    encoded_by_id: dict[str, str] = {}
    for record in records:
        payload = _record_mapping(record, id_key=id_key)
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
                f"conflicting canonical record identity {identity!r} for {id_key}"
            )
        encoded_by_id[identity] = encoded
        by_id.setdefault(identity, payload)
    return [by_id[identity] for identity in sorted(by_id)]


def _activation_layer_key(layer_name: str) -> int | str:
    is_mean = layer_name.endswith(".mean")
    base_name = layer_name.removesuffix(".mean")
    try:
        layer_number = int(base_name.split(".")[1])
    except (IndexError, ValueError):
        return layer_name
    return f"{layer_number}_mean" if is_mean else layer_number


def _connected_split_groups(manifest: SplitManifest) -> dict[str, str]:
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
        for value, owners in (
            (assignment.trial_family_id, family_owner),
            (assignment.dyad_id, dyad_owner),
        ):
            if value in owners:
                union(index, owners[value])
            else:
                owners[value] = index
    components: dict[int, list[str]] = {}
    for index, assignment in enumerate(assignments):
        components.setdefault(find(index), []).append(assignment.trial_id)
    component_ids = {
        root: "connected_"
        + hashlib.sha256(
            "|".join(sorted(trial_ids)).encode("utf-8")
        ).hexdigest()[:20]
        for root, trial_ids in components.items()
    }
    return {
        assignment.trial_id: component_ids[find(index)]
        for index, assignment in enumerate(assignments)
    }


def _build_split_projection(
    metadata: Sequence[Mapping[str, Any]],
    *,
    split_seed: int,
    supplied_manifest: SplitManifest | Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    if type(split_seed) is not int or split_seed < 0:
        raise ValueError("split_seed must be a non-negative integer")
    by_trial: dict[str, dict[str, str]] = {}
    for row in metadata:
        trial_id = str(row.get("trial_id", ""))
        family_id = row.get("trial_family_id")
        if not trial_id or not isinstance(family_id, str) or not family_id:
            raise ValueError(
                "every activation row requires trial_id and trial_family_id "
                "before split publication"
            )
        explicit_dyad = row.get("dyad_id")
        if explicit_dyad is not None and (
            not isinstance(explicit_dyad, str) or not explicit_dyad
        ):
            raise ValueError("dyad_id must be null or a non-empty string")
        if explicit_dyad is None:
            counterpart = (
                row.get("counterpart_assignment_id")
                or row.get("counterpart_name")
                or "unpaired"
            )
            explicit_dyad = (
                f"trial:{trial_id}:dyad:{row.get('agent_name')}:{counterpart}"
            )
        record = {
            "trial_id": trial_id,
            "trial_family_id": family_id,
            "dyad_id": explicit_dyad,
        }
        previous = by_trial.setdefault(trial_id, record)
        if previous != record:
            raise ValueError(
                f"trial {trial_id!r} has inconsistent family/dyad identities"
            )

    if supplied_manifest is None:
        manifest = SplitManifest.build(by_trial.values(), seed=split_seed)
    elif isinstance(supplied_manifest, SplitManifest):
        manifest = supplied_manifest
    elif isinstance(supplied_manifest, Mapping):
        manifest = SplitManifest.from_json(
            json.dumps(supplied_manifest, sort_keys=True, allow_nan=False)
        )
    else:
        raise TypeError("split_manifest must be SplitManifest, mapping, or None")
    manifest.validate()
    if not manifest.locked:
        raise ValueError("activation datasets require a locked SplitManifest")
    assigned_trials = {assignment.trial_id for assignment in manifest.assignments}
    if assigned_trials != set(by_trial):
        raise ValueError("SplitManifest trial membership does not match the dataset")
    expected_identity = {
        assignment.trial_id: (
            assignment.trial_family_id,
            assignment.dyad_id,
        )
        for assignment in manifest.assignments
    }
    for trial_id, record in by_trial.items():
        if expected_identity[trial_id] != (
            record["trial_family_id"],
            record["dyad_id"],
        ):
            raise ValueError("SplitManifest family/dyad identity mismatch")
    partition_by_trial = {
        assignment.trial_id: assignment.partition
        for assignment in manifest.assignments
    }
    group_by_trial = _connected_split_groups(manifest)
    partitions = [partition_by_trial[str(row["trial_id"])] for row in metadata]
    connected_groups = [group_by_trial[str(row["trial_id"])] for row in metadata]
    for row, partition, group_id in zip(
        metadata, partitions, connected_groups
    ):
        row["dyad_id"] = by_trial[str(row["trial_id"])]["dyad_id"]
        row["split_partition"] = partition
        row["split_group_id"] = group_id
        row["connected_group_id"] = group_id
    return manifest.to_dict(), partitions, connected_groups


class DatasetBuilder:
    """Collect canonical activation rows and publish a safe dataset bundle."""

    def __init__(
        self,
        *,
        generation_records: Sequence[Any] = (),
        label_records: Sequence[Any] = (),
        interaction_events: Sequence[Any] = (),
        intervention_designs: Sequence[Any] = (),
        intervention_schedules: Sequence[Any] = (),
        intervention_application_logs: Sequence[Any] = (),
        experiment_track: str | None = None,
        captured_actor_ids: Sequence[str] = (),
        capture_modes: Sequence[str] = (),
        split_manifest: SplitManifest | Mapping[str, Any] | None = None,
        split_seed: int = 42,
        pod_info: Mapping[str, Any] | None = None,
    ) -> None:
        self.samples: List[ActivationSample] = []
        self.generation_records = list(generation_records)
        self.label_records = list(label_records)
        self.interaction_events = list(interaction_events)
        self.intervention_designs = list(intervention_designs)
        self.intervention_schedules = list(intervention_schedules)
        self.intervention_application_logs = list(
            intervention_application_logs
        )
        self.experiment_track = experiment_track
        self.captured_actor_ids = tuple(captured_actor_ids)
        self.capture_modes = tuple(capture_modes)
        self.split_manifest = split_manifest
        self.split_seed = split_seed
        self.pod_info = dict(pod_info or {})

    def add_sample(self, sample: ActivationSample) -> None:
        if not isinstance(sample, ActivationSample):
            raise TypeError("DatasetBuilder accepts canonical ActivationSample rows")
        self.samples.append(sample)

    def clear(self) -> None:
        self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def _canonical_records(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        generation_records = _serialize_records(
            self.generation_records, id_key="call_id"
        )
        label_records = _serialize_records(self.label_records, id_key="label_id")
        interaction_events = _serialize_records(
            self.interaction_events, id_key="event_id"
        )
        generation_ids = {item["call_id"] for item in generation_records}
        label_ids = {item["label_id"] for item in label_records}
        event_ids = {item["event_id"] for item in interaction_events}
        generation_by_id = {
            item["call_id"]: item for item in generation_records
        }
        for sample in self.samples:
            if (
                sample.generation_record_id is not None
                and sample.generation_record_id not in generation_ids
            ):
                raise ValueError(
                    "ActivationSample references a missing GenerationRecord: "
                    f"{sample.generation_record_id}"
                )
            missing_labels = set(sample.label_record_ids).difference(label_ids)
            if missing_labels:
                raise ValueError(
                    "ActivationSample references missing LabelRecords: "
                    f"{sorted(missing_labels)}"
                )
            if (
                sample.interaction_event_id is not None
                and sample.interaction_event_id not in event_ids
            ):
                raise ValueError(
                    "ActivationSample references a missing InteractionEvent: "
                    f"{sample.interaction_event_id}"
                )
            if sample.generation_record_id is not None:
                from interpretability.runtime.model_call import (
                    GenerationRecord,
                    make_activation_artifact_refs,
                )

                record = GenerationRecord.from_dict(
                    generation_by_id[sample.generation_record_id]
                )
                regenerated_artifacts = make_activation_artifact_refs(
                    sample.activations,
                    record.retained_token_index,
                )
                if regenerated_artifacts != record.activation_artifacts:
                    raise ValueError(
                        "ActivationSample payload does not match its "
                        "GenerationRecord artifacts"
                    )
        return generation_records, label_records, interaction_events

    def _canonical_intervention_records(
        self,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        """Serialize the design -> schedule -> receipt-log provenance graph."""
        designs = _serialize_deduplicated_records(
            self.intervention_designs,
            id_key="design_id",
        )
        schedules = _serialize_deduplicated_records(
            self.intervention_schedules,
            id_key="schedule_id",
        )
        application_logs = _serialize_deduplicated_records(
            self.intervention_application_logs,
            id_key="log_id",
        )
        return designs, schedules, application_logs

    def _stack_activations(self) -> Dict[int | str, torch.Tensor]:
        expected_names: set[str] | None = None
        by_layer: dict[int | str, list[torch.Tensor]] = {}
        mapped_names: dict[int | str, str] = {}
        for sample_index, sample in enumerate(self.samples):
            layer_names = set(sample.activations)
            if not layer_names:
                raise ValueError(
                    f"Sample {sample_index} ({sample.trial_id}) has no activations"
                )
            if expected_names is None:
                expected_names = layer_names
            elif layer_names != expected_names:
                raise ValueError(
                    "Activation layers are not aligned across samples: "
                    f"sample {sample_index} has {sorted(layer_names)}, expected "
                    f"{sorted(expected_names)}"
                )
            for layer_name, activation in sample.activations.items():
                if not isinstance(layer_name, str) or not layer_name:
                    raise ValueError("activation hook names must be non-empty strings")
                if not isinstance(activation, torch.Tensor):
                    raise TypeError("ActivationSample activations must be torch tensors")
                if activation.ndim != 1:
                    raise ValueError("each activation row must be one-dimensional")
                if not torch.isfinite(activation.float()).all():
                    raise ValueError("activation rows must contain finite values")
                key = _activation_layer_key(layer_name)
                previous_name = mapped_names.setdefault(key, layer_name)
                if previous_name != layer_name:
                    raise ValueError(
                        "activation hook names collide after layer projection: "
                        f"{previous_name!r} and {layer_name!r}"
                    )
                by_layer.setdefault(key, []).append(activation.detach().cpu())
        stacked = {key: torch.stack(rows) for key, rows in by_layer.items()}
        dimensions = {value.shape[1] for value in stacked.values()}
        if len(dimensions) != 1:
            raise ValueError("activation layers must share one hidden dimension")
        return stacked

    def _sae_payload(self) -> dict[str, Any]:
        sparse_rows = [sample.sae_features for sample in self.samples]
        top_rows = [list(sample.sae_top_features or []) for sample in self.samples]
        populated = [features for features in sparse_rows if features]
        if not populated:
            if any(top_rows):
                raise ValueError("SAE top features exist without populated SAE rows")
            return {"has_sae": False}
        feature_ids = []
        for row_index, features in enumerate(sparse_rows):
            if features is None:
                if top_rows[row_index]:
                    raise ValueError("unavailable SAE rows cannot declare top features")
                continue
            if not isinstance(features, Mapping):
                raise TypeError("SAE features must be sparse index/value mappings")
            for feature_id, value in features.items():
                if (
                    isinstance(feature_id, bool)
                    or not isinstance(feature_id, Integral)
                    or feature_id < 0
                ):
                    raise ValueError("SAE feature indices must be non-negative integers")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                ):
                    raise ValueError("SAE feature values must be finite numbers")
                feature_ids.append(int(feature_id))
        sae_dim = max(feature_ids) + 1
        for row_index, top_features in enumerate(top_rows):
            if any(
                isinstance(feature_id, bool)
                or not isinstance(feature_id, Integral)
                or not 0 <= feature_id < sae_dim
                for feature_id in top_features
            ):
                raise ValueError(f"invalid SAE top-feature IDs at row {row_index}")
        dense = torch.zeros(len(sparse_rows), sae_dim, dtype=torch.float32)
        for row_index, features in enumerate(sparse_rows):
            for feature_id, value in (features or {}).items():
                dense[row_index, feature_id] = float(value)
        return {
            "has_sae": True,
            "sae_dim": sae_dim,
            "sae_features": dense,
            "sae_top_features": top_rows,
            "sae_available_mask": [features is not None for features in sparse_rows],
        }

    @staticmethod
    def _sample_metadata(sample: ActivationSample, scenario: str) -> dict[str, Any]:
        qc_flags = sorted(
            classify_sample_response(
                sample.response,
                scenario=scenario,
                semantic_phase=sample.semantic_phase,
            )
        )
        return {
            "trial_id": sample.trial_id,
            "round_num": sample.round_num,
            "sample_type": sample.sample_type,
            "semantic_phase": sample.semantic_phase,
            "agent_name": sample.agent_name,
            "scenario": scenario,
            "incentive_condition": sample.incentive_condition,
            "scenario_params": dict(sample.scenario_params),
            "emergent_ground_truth": sample.emergent_ground_truth,
            "actual_deception": sample.actual_deception,
            "perceived_deception": sample.perceived_deception,
            "modules_enabled": list(sample.modules_enabled),
            "gm_modules_enabled": list(sample.gm_modules_enabled),
            "activation_position": sample.activation_position,
            "sampling_config": dict(sample.sampling_config),
            "generation_record_id": sample.generation_record_id,
            "interaction_event_id": sample.interaction_event_id,
            "label_record_ids": list(sample.label_record_ids),
            "actual_deception_projection": sample.actual_deception_projection,
            "trial_family_id": sample.trial_family_id,
            "scenario_instance_id": sample.scenario_instance_id,
            "role_assignment_id": sample.role_assignment_id,
            "order_assignment_id": sample.order_assignment_id,
            "counterpart_assignment_id": sample.counterpart_assignment_id,
            "surface_assignment_id": sample.surface_assignment_id,
            "counterbalance_id": sample.counterbalance_id,
            "first_mover_id": sample.first_mover_id,
            "role_assignment": dict(sample.role_assignment),
            "surface_assignment": dict(sample.surface_assignment),
            "actor_profile": sample.actor_profile,
            "counterpart_profile": sample.counterpart_profile,
            "experiment_mode": sample.experiment_mode,
            "experiment_track": sample.experiment_track,
            "execution_protocol": sample.execution_protocol,
            "intervention_design_id": sample.intervention_design_id,
            "intervention_application_receipt_ids": list(
                sample.intervention_application_receipt_ids
            ),
            "counterpart_idx": sample.counterpart_idx,
            "counterpart_name": sample.counterpart_name,
            "counterpart_type": sample.counterpart_type,
            "trial_outcome": sample.trial_outcome,
            "joint_value": sample.joint_value,
            "agent_utility": sample.agent_utility,
            "condition_id": sample.condition_id,
            "pod_id": sample.pod_id,
            "emotion_intensity": sample.emotion_intensity,
            "trust_level": sample.trust_level,
            "cooperation_intent": sample.cooperation_intent,
            "commitment_violation": sample.commitment_violation,
            "manipulation_score": sample.manipulation_score,
            "consistency_score": sample.consistency_score,
            "tom_state": sample.tom_state,
            "deepeval_false_claims": sample.deepeval_false_claims,
            "deepeval_omission": sample.deepeval_omission,
            "deepeval_framing": sample.deepeval_framing,
            "deepeval_commitment": sample.deepeval_commitment,
            "deepeval_confidence": sample.deepeval_confidence,
            "deepeval_reasoning": sample.deepeval_reasoning,
            "ground_truth_evaluation_succeeded": (
                sample.ground_truth_evaluation_succeeded
            ),
            "ground_truth_evaluation_method": sample.ground_truth_evaluation_method,
            "ground_truth_evaluation_error": sample.ground_truth_evaluation_error,
            "component_reasoning": sample.component_reasoning,
            "dialogue_history": sample.dialogue_history,
            "full_prompt": sample.prompt,
            "full_response": sample.response,
            "is_verification_probe": sample.is_verification_probe,
            "plausibility_response": sample.plausibility_response,
            "gt_regex": sample.gt_regex,
            "gt_llm_rules": sample.gt_llm_rules,
            "gt_deepeval": sample.gt_deepeval,
            "belief_shift_injected": sample.belief_shift_injected,
            "belief_shift_type": sample.belief_shift_type,
            "belief_shift_round": sample.belief_shift_round,
            "framing_variant": sample.framing_variant,
            "qc_flags": qc_flags,
            "qc_status": "passed" if not qc_flags else "rejected",
            "qc_version": QC_VERSION,
        }

    def build(
        self,
        *,
        model_name: str = "unknown",
        model_revision: str | None = None,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        experiment_track: str | None = None,
        captured_actor_ids: Sequence[str] | None = None,
        capture_modes: Sequence[str] | None = None,
        split_manifest: SplitManifest | Mapping[str, Any] | None = None,
        split_seed: int | None = None,
    ) -> dict[str, Any]:
        if not self.samples:
            raise ValueError("No activation samples to save")
        activations = self._stack_activations()
        generation_records, label_records, interaction_events = (
            self._canonical_records()
        )
        (
            intervention_designs,
            intervention_schedules,
            intervention_application_logs,
        ) = self._canonical_intervention_records()
        scenarios = [
            sample.emergent_scenario or sample.scenario_type or "unknown"
            for sample in self.samples
        ]
        gm_labels = [
            (1.0 if sample.emergent_ground_truth else 0.0)
            if sample.emergent_ground_truth is not None
            else sample.actual_deception
            for sample in self.samples
        ]
        metadata = [
            self._sample_metadata(sample, scenario)
            for sample, scenario in zip(self.samples, scenarios)
        ]
        labels = {
            "gm_labels": gm_labels,
            "agent_labels": [sample.perceived_deception for sample in self.samples],
            "scenario": scenarios,
            "mode_labels": [
                sample.experiment_mode or "unknown" for sample in self.samples
            ],
            "round_nums": [sample.round_num for sample in self.samples],
            "trial_ids": [sample.trial_id for sample in self.samples],
            "counterpart_idxs": [sample.counterpart_idx for sample in self.samples],
            "trial_outcomes": [sample.trial_outcome for sample in self.samples],
            "pod_ids": [sample.pod_id for sample in self.samples],
        }
        tracks = {
            sample.experiment_track
            for sample in self.samples
            if sample.experiment_track
        }
        selected_track = experiment_track or self.experiment_track
        if selected_track is None:
            if len(tracks) > 1:
                raise ValueError("activation rows contain multiple experiment tracks")
            if not tracks:
                raise ValueError(
                    "experiment_track must be declared before dataset publication"
                )
            selected_track = next(iter(tracks))
        elif tracks and tracks != {selected_track}:
            raise ValueError("row experiment tracks do not match the dataset track")
        actors = tuple(captured_actor_ids or self.captured_actor_ids)
        if not actors:
            actors = tuple(sorted({sample.agent_name for sample in self.samples}))
        if len(set(actors)) != len(actors) or any(not actor for actor in actors):
            raise ValueError("captured_actor_ids must be unique and non-empty")
        undeclared_actors = {
            sample.agent_name for sample in self.samples
        }.difference(actors)
        if undeclared_actors:
            raise ValueError(
                "activation rows contain actors absent from captured_actor_ids: "
                f"{sorted(undeclared_actors)}"
            )

        generation_by_id = {
            record["call_id"]: record for record in generation_records
        }
        referenced_generation_records = [
            generation_by_id[sample.generation_record_id]
            for sample in self.samples
            if sample.generation_record_id is not None
        ]
        for sample, row in zip(self.samples, metadata):
            row["experiment_track"] = selected_track
            if sample.generation_record_id is None:
                continue
            record = generation_by_id[sample.generation_record_id]
            row["activation_position"] = record["activation_position"]
            row["sampling_config"] = {
                "requested": record["requested_sampling"],
                "effective": record["effective_sampling"],
                "generation_path": record["generation_path"],
                "fallback_reason": record.get("fallback_reason"),
            }

        record_model_revisions = sorted(
            {
                str(record["model_revision"])
                for record in referenced_generation_records
                if record.get("model_revision")
            }
        )
        record_tokenizer_revisions = sorted(
            {
                str(record["tokenizer_revision"])
                for record in referenced_generation_records
                if record.get("tokenizer_revision")
            }
        )
        sampling_configs = []
        seen_sampling = set()
        for row in metadata:
            sampling_config = row.get("sampling_config")
            if not sampling_config:
                continue
            encoded = json.dumps(sampling_config, sort_keys=True, allow_nan=False)
            if encoded not in seen_sampling:
                seen_sampling.add(encoded)
                sampling_configs.append(json.loads(encoded))
        record_capture_modes = sorted(
            {
                str(record["capture_mode"])
                for record in referenced_generation_records
                if record.get("capture_mode") and record.get("capture_mode") != "none"
            }
        )
        if len(record_model_revisions) > 1:
            raise ValueError("captured rows use multiple model revisions")
        if model_revision and record_model_revisions and (
            model_revision != record_model_revisions[0]
        ):
            raise ValueError("declared model revision disagrees with call records")
        resolved_model_revision = model_revision or (
            record_model_revisions[0] if record_model_revisions else "unresolved"
        )
        resolved_tokenizer_name = tokenizer_name or model_name
        if len(record_tokenizer_revisions) > 1:
            raise ValueError("captured rows use multiple tokenizer revisions")
        if tokenizer_revision and record_tokenizer_revisions and (
            tokenizer_revision != record_tokenizer_revisions[0]
        ):
            raise ValueError(
                "declared tokenizer revision disagrees with call records"
            )
        resolved_tokenizer_revision = tokenizer_revision or (
            record_tokenizer_revisions[0]
            if record_tokenizer_revisions
            else "unresolved"
        )
        declared_capture_modes = tuple(capture_modes or self.capture_modes)
        if declared_capture_modes and record_capture_modes and (
            set(declared_capture_modes) != set(record_capture_modes)
        ):
            raise ValueError("declared capture modes disagree with call records")
        resolved_capture_modes = sorted(
            set(record_capture_modes or declared_capture_modes)
        )
        split_manifest_payload, split_partitions, connected_group_ids = (
            _build_split_projection(
                metadata,
                split_seed=(self.split_seed if split_seed is None else split_seed),
                supplied_manifest=(
                    self.split_manifest
                    if split_manifest is None
                    else split_manifest
                ),
            )
        )
        labels["split_partitions"] = split_partitions
        labels["connected_group_ids"] = connected_group_ids
        sae = self._sae_payload()
        config = {
            "dataset_schema_version": ACTIVATION_DATASET_SCHEMA_VERSION,
            "model": model_name,
            "layers": list(activations),
            "n_samples": len(self.samples),
            "has_sae": sae["has_sae"],
            "label_semantics": {
                "gm_labels": "acting-agent deception ground truth",
                "agent_labels": "acting agent estimate of counterpart deception",
            },
            "experiment_track": selected_track,
            "captured_actor_ids": list(actors),
            "split_seed": split_manifest_payload["seed"],
            "split_manifest_id": split_manifest_payload["manifest_id"],
            "provenance": {
                "model": {
                    "name": model_name,
                    "revision": resolved_model_revision,
                    "record_revisions": record_model_revisions,
                },
                "tokenizer": {
                    "name": resolved_tokenizer_name,
                    "revision": resolved_tokenizer_revision,
                    "record_revisions": record_tokenizer_revisions,
                },
                "sampling_configs": sampling_configs,
                "capture": {
                    "layers": list(activations),
                    "activation_positions": sorted(
                        {row["activation_position"] for row in metadata}
                    ),
                    "capture_modes": resolved_capture_modes or ["unresolved"],
                },
                "runtime_versions": {
                    "gdm-concordia": _package_version("gdm-concordia"),
                    "transformer-lens": _package_version("transformer-lens"),
                    "torch": _package_version("torch"),
                },
            },
        }
        if sae["has_sae"]:
            config["sae_dim"] = sae["sae_dim"]
        pod_info = dict(self.pod_info)
        pod_info.setdefault("pod_id", self.samples[0].pod_id)
        pod_info.setdefault("n_samples", len(self.samples))
        dataset = {
            "activations": activations,
            "labels": labels,
            "config": config,
            "metadata": metadata,
            "generation_records": generation_records,
            "label_records": label_records,
            "interaction_events": interaction_events,
            "intervention_designs": intervention_designs,
            "intervention_schedules": intervention_schedules,
            "intervention_application_logs": intervention_application_logs,
            "split_manifest": split_manifest_payload,
            "pod_info": pod_info,
        }
        if sae["has_sae"]:
            dataset.update(
                {
                    "sae_features": sae["sae_features"],
                    "sae_top_features": sae["sae_top_features"],
                    "sae_available_mask": sae["sae_available_mask"],
                }
            )
        return dataset

    def save(
        self,
        filepath: str | Path,
        model_name: str = "unknown",
        *,
        model_revision: str | None = None,
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        experiment_track: str | None = None,
        captured_actor_ids: Sequence[str] | None = None,
        capture_modes: Sequence[str] | None = None,
        split_manifest: SplitManifest | Mapping[str, Any] | None = None,
        split_seed: int | None = None,
        trusted_legacy: bool = False,
    ) -> Path:
        dataset = self.build(
            model_name=model_name,
            model_revision=model_revision,
            tokenizer_name=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            experiment_track=experiment_track,
            captured_actor_ids=captured_actor_ids,
            capture_modes=capture_modes,
            split_manifest=split_manifest,
            split_seed=split_seed,
        )
        saved = save_activation_dataset(
            filepath,
            dataset,
            trusted_legacy=trusted_legacy,
        )
        output = saved if isinstance(saved, Path) else saved[1]
        transcript_path = output.with_name(f"{output.stem}_transcripts.jsonl")
        with transcript_path.open("w", encoding="utf-8") as stream:
            for row in dataset["metadata"]:
                record = {
                    "trial_id": row.get("trial_id"),
                    "round_num": row.get("round_num"),
                    "agent_name": row.get("agent_name"),
                    "scenario": row.get("scenario"),
                    "actual_deception": row.get("actual_deception"),
                    "prompt": row.get("full_prompt", "")[:2000],
                    "response": row.get("full_response", ""),
                    "tom_state": row.get("tom_state"),
                }
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._log_summary(output, dataset)
        return output

    @staticmethod
    def _log_summary(filepath: Path, dataset: Mapping[str, Any]) -> None:
        labels = dataset["labels"]
        layers = list(dataset["activations"])
        gm_labels = [value for value in labels["gm_labels"] if value is not None]
        logger.info("Saved %d samples to %s", len(labels["gm_labels"]), filepath)
        logger.info("Layers: %s", layers)
        if gm_labels:
            logger.info("GM deception rate: %.1f%%", np.mean(gm_labels) * 100)
        logger.info(
            "SAE features: %s",
            "captured" if dataset["config"]["has_sae"] else "not captured",
        )

    @staticmethod
    def load(
        filepath: str | Path,
        *,
        trusted: bool = False,
    ) -> Dict[str, Any]:
        """Load safe JSON+NPZ, or legacy .pt only with explicit trust."""
        return load_activation_dataset(filepath, trusted_legacy=trusted)


__all__ = ["ActivationSample", "DatasetBuilder"]
