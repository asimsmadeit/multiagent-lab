#!/usr/bin/env python3
"""Merge aligned activation pods into one safe JSON+NPZ dataset."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from interpretability.core.dataset_builder import _build_split_projection
from interpretability.data import load_activation_dataset, save_activation_dataset


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _validate_alignment(data: Mapping[str, Any], source: str) -> int:
    activations = data.get("activations")
    labels = data.get("labels")
    metadata = data.get("metadata")
    config = data.get("config")
    if not isinstance(activations, Mapping) or not activations:
        raise ValueError(f"{source}: activation layers are missing")
    if not isinstance(labels, Mapping) or not isinstance(
        labels.get("gm_labels"), list
    ):
        raise ValueError(f"{source}: labels.gm_labels is missing")
    if not isinstance(metadata, list) or not isinstance(config, Mapping):
        raise ValueError(f"{source}: metadata/config is missing")
    n_samples = len(labels["gm_labels"])
    if n_samples < 1 or len(metadata) != n_samples:
        raise ValueError(f"{source}: metadata is not sample-aligned")
    for name, values in labels.items():
        if not isinstance(values, list) or len(values) != n_samples:
            raise ValueError(f"{source}: label {name!r} is not sample-aligned")
    for layer, tensor in activations.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{source}: layer {layer!r} is not a tensor")
        if tensor.ndim != 2 or tensor.shape[0] != n_samples:
            raise ValueError(f"{source}: layer {layer!r} is not sample-aligned")
        if not bool(torch.isfinite(tensor.float()).all()):
            raise ValueError(f"{source}: layer {layer!r} contains non-finite values")
    return n_samples


def _config_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    identity = copy.deepcopy(dict(config))
    for key in (
        "dataset_hash",
        "n_samples",
        "schema_registry_checksum",
        "split_manifest_id",
    ):
        identity.pop(key, None)
    provenance = identity.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("sampling_configs", None)
    return identity


def _merge_canonical_records(
    datasets: Sequence[Mapping[str, Any]],
    *,
    collection: str,
    id_key: str,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    encoded_by_id: dict[str, str] = {}
    for data in datasets:
        records = data.get(collection, [])
        if not isinstance(records, list):
            raise TypeError(f"{collection} must be a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError(f"{collection} entries must be mappings")
            identity = record.get(id_key)
            if not isinstance(identity, str) or not identity:
                raise ValueError(f"{collection} entries require {id_key}")
            payload = copy.deepcopy(dict(record))
            encoded = _canonical_json(payload)
            if identity in encoded_by_id and encoded_by_id[identity] != encoded:
                raise ValueError(
                    f"conflicting duplicate canonical ID {identity!r} in {collection}"
                )
            encoded_by_id[identity] = encoded
            by_id.setdefault(identity, payload)
    return [by_id[identity] for identity in sorted(by_id)]


def _merge_sampling_configs(datasets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_encoding: dict[str, dict[str, Any]] = {}
    for data in datasets:
        configs = data["config"]["provenance"]["sampling_configs"]
        for config in configs:
            payload = copy.deepcopy(dict(config))
            by_encoding.setdefault(_canonical_json(payload), payload)
    return [by_encoding[key] for key in sorted(by_encoding)]


def merge_parallel_activations(
    pod_files: Sequence[str],
    output_dir: str | None = None,
    timestamp: str | None = None,
    verbose: bool = True,
    trusted_legacy: bool = False,
) -> str:
    """Merge pods without weakening provenance, lineage, or SAE validation."""
    if not pod_files:
        raise ValueError("No pod files provided")
    sources = sorted(map(str, pod_files))
    datasets = [
        load_activation_dataset(path, trusted_legacy=trusted_legacy)
        for path in sources
    ]
    sample_counts = [
        _validate_alignment(data, source)
        for data, source in zip(datasets, sources)
    ]
    first = datasets[0]
    expected_layers = list(first["activations"])
    expected_labels = set(first["labels"])
    expected_config = _config_identity(first["config"])
    expected_sae = bool(first["config"].get("has_sae"))
    for data, source in zip(datasets[1:], sources[1:]):
        if list(data["activations"]) != expected_layers:
            raise ValueError(f"{source}: activation layer order/identity differs")
        if set(data["labels"]) != expected_labels:
            raise ValueError(f"{source}: label schema differs")
        if _config_identity(data["config"]) != expected_config:
            raise ValueError(f"{source}: dataset provenance/config is incompatible")
        if bool(data["config"].get("has_sae")) != expected_sae:
            raise ValueError(f"{source}: SAE presence differs")

    activations: dict[Any, torch.Tensor] = {}
    for layer in expected_layers:
        tensors = [data["activations"][layer] for data in datasets]
        reference = tensors[0]
        if any(
            tensor.dtype != reference.dtype
            or tensor.shape[1:] != reference.shape[1:]
            for tensor in tensors[1:]
        ):
            raise ValueError(f"activation dtype/shape differs at layer {layer!r}")
        activations[layer] = torch.cat(tensors, dim=0)

    labels = {
        name: [item for data in datasets for item in data["labels"][name]]
        for name in first["labels"]
        if name not in {"split_partitions", "connected_group_ids"}
    }
    metadata = [
        copy.deepcopy(row)
        for data in datasets
        for row in data["metadata"]
    ]

    offsets = np.cumsum([0, *sample_counts[:-1]]).tolist()
    if "counterpart_idxs" in labels:
        remapped: list[int | None] = []
        cursor = 0
        for source_index, (count, offset) in enumerate(zip(sample_counts, offsets)):
            source_values = datasets[source_index]["labels"]["counterpart_idxs"]
            for local_index in source_values:
                if local_index is None:
                    remapped.append(None)
                elif type(local_index) is not int or not 0 <= local_index < count:
                    raise ValueError("counterpart index is outside its source pod")
                else:
                    remapped.append(offset + local_index)
                cursor += 1
        labels["counterpart_idxs"] = remapped
        for row, counterpart_idx in zip(metadata, remapped):
            row["counterpart_idx"] = counterpart_idx

    split_seed = first["config"].get("split_seed")
    split_manifest, partitions, connected_groups = _build_split_projection(
        metadata,
        split_seed=split_seed,
        supplied_manifest=None,
    )
    labels["split_partitions"] = partitions
    labels["connected_group_ids"] = connected_groups

    config = copy.deepcopy(dict(first["config"]))
    config.pop("dataset_hash", None)
    config.pop("schema_registry_checksum", None)
    config["n_samples"] = sum(sample_counts)
    config["split_manifest_id"] = split_manifest["manifest_id"]
    config["provenance"]["sampling_configs"] = _merge_sampling_configs(datasets)
    merged: dict[str, Any] = {
        "activations": activations,
        "labels": labels,
        "config": config,
        "metadata": metadata,
        "generation_records": _merge_canonical_records(
            datasets, collection="generation_records", id_key="call_id"
        ),
        "interaction_events": _merge_canonical_records(
            datasets, collection="interaction_events", id_key="event_id"
        ),
        "label_records": _merge_canonical_records(
            datasets, collection="label_records", id_key="label_id"
        ),
        "intervention_designs": _merge_canonical_records(
            datasets, collection="intervention_designs", id_key="design_id"
        ),
        "intervention_schedules": _merge_canonical_records(
            datasets, collection="intervention_schedules", id_key="schedule_id"
        ),
        "intervention_application_logs": _merge_canonical_records(
            datasets,
            collection="intervention_application_logs",
            id_key="log_id",
        ),
        "split_manifest": split_manifest,
        "pod_info": {
            "pod_id": "merged",
            "n_samples": sum(sample_counts),
        },
        "merge_info": {
            "source_files": sources,
            "source_dataset_hashes": [
                data.get("config", {}).get("dataset_hash") for data in datasets
            ],
            "n_pods": len(datasets),
            "pod_infos": [copy.deepcopy(data.get("pod_info", {})) for data in datasets],
            "total_samples": sum(sample_counts),
            "merged_at": timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"),
        },
    }

    if expected_sae:
        sae_tensors = [data.get("sae_features") for data in datasets]
        if any(not isinstance(tensor, torch.Tensor) for tensor in sae_tensors):
            raise ValueError("every SAE-enabled pod must contain SAE features")
        reference = sae_tensors[0]
        if any(
            tensor.dtype != reference.dtype
            or tensor.shape[1:] != reference.shape[1:]
            for tensor in sae_tensors[1:]
        ):
            raise ValueError("SAE feature dtype/shape differs across pods")
        merged["sae_features"] = torch.cat(sae_tensors, dim=0)
        merged["sae_top_features"] = [
            list(row)
            for data in datasets
            for row in data.get("sae_top_features", [])
        ]
        merged["sae_available_mask"] = [
            value
            for data in datasets
            for value in data.get("sae_available_mask", [])
        ]
        if len(merged["sae_top_features"]) != config["n_samples"] or len(
            merged["sae_available_mask"]
        ) != config["n_samples"]:
            raise ValueError("SAE metadata is not sample-aligned after merge")

    destination = Path(output_dir) if output_dir else Path(sources[0]).parent
    destination.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = save_activation_dataset(destination / f"activations_merged_{ts}.json", merged)
    output_path = saved[1]
    if verbose:
        print(
            f"Merged {len(datasets)} pods / {config['n_samples']} samples "
            f"to {output_path}"
        )
    return str(output_path)


def validate_merge(
    merged_path: str,
    verbose: bool = True,
    *,
    trusted_legacy: bool = False,
) -> dict[str, Any]:
    """Load and validate an existing merged artifact."""
    data = load_activation_dataset(
        merged_path,
        trusted_legacy=trusted_legacy,
    )
    n_samples = _validate_alignment(data, merged_path)
    finite_labels = [
        float(value)
        for value in data["labels"]["gm_labels"]
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
    ]
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {
            "n_samples": n_samples,
            "n_pods": data.get("merge_info", {}).get("n_pods", 1),
            "layers": list(data["activations"]),
            "deception_rate": (
                float(np.mean(np.asarray(finite_labels) > 0.5))
                if finite_labels
                else None
            ),
        },
    }
    if verbose:
        print(f"Validated {merged_path}: {n_samples} aligned samples")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge safe activation datasets from parallel execution"
    )
    parser.add_argument("pod_files", nargs="+", help="Safe .json or legacy .pt files")
    parser.add_argument("-o", "--output-dir", default=None)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument(
        "--trust-legacy-pt",
        action="store_true",
        help="Allow pickle-capable .pt loading only for reviewed inputs",
    )
    args = parser.parse_args()
    if args.validate_only:
        validate_merge(
            args.pod_files[0],
            verbose=not args.quiet,
            trusted_legacy=args.trust_legacy_pt,
        )
    else:
        output_path = merge_parallel_activations(
            args.pod_files,
            output_dir=args.output_dir,
            timestamp=args.timestamp,
            verbose=not args.quiet,
            trusted_legacy=args.trust_legacy_pt,
        )
        print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
