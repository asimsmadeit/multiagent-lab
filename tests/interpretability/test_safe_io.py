"""Safe artifact and explicit legacy-trust contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from interpretability.core.dataset_builder import DatasetBuilder
from interpretability.data.io import (
    load_array_bundle,
    load_trusted_legacy_torch,
    save_array_bundle,
)


def test_array_bundle_round_trip_disables_objects_and_checks_manifest(tmp_path) -> None:
    base = tmp_path / "probe_data"
    array_path, manifest_path = save_array_bundle(
        base,
        {"activations": torch.arange(12).reshape(3, 4), "labels": np.array([0, 1, 0])},
        {"dataset_hash": "source-hash", "split_manifest_id": "split-1"},
    )

    arrays, manifest = load_array_bundle(manifest_path)

    assert np.array_equal(arrays["activations"], np.arange(12).reshape(3, 4))
    assert arrays["labels"].tolist() == [0, 1, 0]
    assert manifest["split_manifest_id"] == "split-1"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["array_file"] == array_path.name
    assert len(payload["array_sha256"]) == 64


def test_array_bundle_rejects_object_dtype_and_tampering(tmp_path) -> None:
    with pytest.raises(TypeError, match="object dtype"):
        save_array_bundle(
            tmp_path / "unsafe",
            {"objects": np.array([{"code": "payload"}], dtype=object)},
            {},
        )

    invalid_base = tmp_path / "invalid_manifest"
    with pytest.raises(ValueError, match="Out of range float"):
        save_array_bundle(
            invalid_base,
            {"values": np.array([1, 2])},
            {"invalid": float("nan")},
        )
    assert not invalid_base.with_suffix(".npz").exists()

    array_path, manifest_path = save_array_bundle(
        tmp_path / "safe",
        {"values": np.array([1, 2, 3])},
        {},
    )
    array_path.write_bytes(array_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_array_bundle(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_root", "missing fields: manifest"),
        ("unknown_root", "unknown fields: future_field"),
        ("missing_spec", "missing fields: dtype"),
        ("unknown_spec", "unknown fields: future_field"),
    ],
)
def test_array_bundle_restore_requires_exact_current_schema(
    tmp_path,
    mutation,
    match,
) -> None:
    _, manifest_path = save_array_bundle(
        tmp_path / mutation,
        {"values": np.array([1, 2, 3])},
        {"kind": "fixture"},
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing_root":
        del payload["manifest"]
    elif mutation == "unknown_root":
        payload["future_field"] = None
    elif mutation == "missing_spec":
        del payload["arrays"]["values"]["dtype"]
    else:
        payload["arrays"]["values"]["future_field"] = None
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_array_bundle(manifest_path)


def test_legacy_torch_loading_requires_explicit_trust(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"labels": [0, 1]}, path)

    with pytest.raises(PermissionError, match="trusted=True"):
        load_trusted_legacy_torch(path)
    with pytest.raises(PermissionError, match="trusted=True"):
        DatasetBuilder.load(str(path))

    assert load_trusted_legacy_torch(path, trusted=True) == {"labels": [0, 1]}
    assert DatasetBuilder.load(str(path), trusted=True) == {"labels": [0, 1]}
