"""Integrity and filesystem-safety tests for activation artifact storage."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import interpretability.events.artifacts as artifacts_module
from interpretability.events.artifacts import (
    ARTIFACT_FORMAT_VERSION,
    ArtifactCollisionError,
    ArtifactCorruptionError,
    ArtifactError,
    ArtifactNotFoundError,
    ArtifactStore,
    ArtifactValidationError,
    UnsafeArtifactPathError,
)

_MANIFEST_MEMBER = "__manifest__.npy"


def sample_arrays() -> dict[str, np.ndarray]:
    return {
        "blocks.2.hook_resid_post": np.arange(12, dtype=np.float32).reshape(3, 4),
        "blocks.8.hook_resid_post": np.array([[True, False, True]], dtype=np.bool_),
    }


def put_sample(root: Path, **kwargs: Any):
    store = ArtifactStore(root)
    stored = store.put(
        sample_arrays(),
        metadata={"model": "model-v1", "seed": 17},
        **kwargs,
    )
    return store, stored


def npy_bytes(array: np.ndarray, *, allow_pickle: bool = False) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=allow_pickle)
    return buffer.getvalue()


def read_members(path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(path, "r") as archive:
        return [(info.filename, archive.read(info)) for info in archive.infolist()]


def write_members(path: Path, members: list[tuple[str, bytes]]) -> None:
    temporary = path.with_suffix(".rewrite.tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for name, value in members:
                archive.writestr(name, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(path: Path) -> dict[str, Any]:
    member = next(value for name, value in read_members(path) if name == _MANIFEST_MEMBER)
    manifest_array = np.load(io.BytesIO(member), allow_pickle=False)
    return json.loads(manifest_array.tobytes().decode("utf-8"))


def canonical_manifest_member(manifest: dict[str, Any]) -> bytes:
    encoded = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return npy_bytes(np.frombuffer(encoded, dtype=np.uint8))


def mutate_manifest(path: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    manifest = read_manifest(path)
    mutation(manifest)
    members = read_members(path)
    write_members(
        path,
        [
            (name, canonical_manifest_member(manifest) if name == _MANIFEST_MEMBER else value)
            for name, value in members
        ],
    )


def replace_member(path: Path, name: str, value: bytes) -> None:
    members = read_members(path)
    assert any(member_name == name for member_name, _ in members)
    write_members(
        path,
        [
            (member_name, value if member_name == name else member_value)
            for member_name, member_value in members
        ],
    )


def test_hash_and_npz_bytes_are_deterministic_across_roots_and_input_order(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    arrays = sample_arrays()
    reversed_arrays = dict(reversed(tuple(arrays.items())))

    first = ArtifactStore(first_root).put(
        arrays, metadata={"seed": 17, "model": "model-v1"}
    )
    second = ArtifactStore(second_root).put(
        reversed_arrays, metadata={"model": "model-v1", "seed": 17}
    )

    assert first.artifact_hash == second.artifact_hash
    assert first.file_sha256 == second.file_sha256
    assert first.path.read_bytes() == second.path.read_bytes()
    assert tuple(item.hook_name for item in first.arrays) == (
        "blocks.2.hook_resid_post",
        "blocks.8.hook_resid_post",
    )


def test_endian_contiguity_and_views_have_one_logical_identity(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
    contiguous_little = np.ascontiguousarray(values.astype("<f4"))
    fortran_big = np.asfortranarray(values.astype(">f4"))
    assert not values.flags.c_contiguous
    assert fortran_big.flags.f_contiguous

    hashes = {
        ArtifactStore(tmp_path / name)
        .put({"blocks.0.hook": array})
        .artifact_hash
        for name, array in (
            ("view", values),
            ("little", contiguous_little),
            ("big", fortran_big),
        )
    }

    assert len(hashes) == 1
    stored = ArtifactStore(tmp_path / "big").load(next(iter(hashes)))
    assert stored.arrays["blocks.0.hook"].dtype.str == "<f4"
    assert stored.arrays["blocks.0.hook"].flags.c_contiguous


def test_optional_torch_tensor_conversion_is_detached_and_copied(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.arange(6, dtype=torch.float32, requires_grad=True).reshape(2, 3)

    store = ArtifactStore(tmp_path)
    stored = store.put({"blocks.0.hook": tensor})
    tensor.detach().add_(100)
    loaded = store.load(stored.artifact_hash)

    assert np.array_equal(
        loaded.arrays["blocks.0.hook"],
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )


def test_multi_hook_order_is_canonical_and_case_collisions_fail(tmp_path: Path) -> None:
    arrays = {
        "z_hook": np.ones((1, 2), dtype=np.float32),
        "A_hook": np.zeros((1, 2), dtype=np.float32),
        "m_hook": np.full((1, 2), 2, dtype=np.float32),
    }
    stored = ArtifactStore(tmp_path).put(arrays)

    assert tuple(item.hook_name for item in stored.arrays) == (
        "A_hook",
        "m_hook",
        "z_hook",
    )
    with pytest.raises(ArtifactValidationError, match="collide"):
        ArtifactStore(tmp_path / "collision").put(
            {"Hook": np.ones(1), "hook": np.ones(1)}
        )


def test_metadata_is_canonical_and_input_aliases_are_isolated(tmp_path: Path) -> None:
    metadata = {"z": [1, {"value": True}], "a": {"name": "model"}}
    source = np.arange(4, dtype=np.float32)
    store = ArtifactStore(tmp_path)
    stored = store.put({"hook": source}, metadata=metadata)

    metadata["z"][1]["value"] = False
    metadata["a"]["name"] = "mutated"
    source[:] = 99
    loaded = store.load(stored.artifact_hash)

    assert dict(loaded.metadata["a"]) == {"name": "model"}
    assert loaded.metadata["z"][1]["value"] is True
    assert np.array_equal(loaded.arrays["hook"], np.arange(4, dtype=np.float32))
    assert stored.metadata_json == '{"a":{"name":"model"},"z":[1,{"value":true}]}'


@pytest.mark.parametrize(
    "metadata",
    [
        {1: "non-string"},
        {"unsupported": object()},
        {"unsupported": b"bytes"},
        {"unsupported": {1, 2}},
        {"number": float("nan")},
        {"number": float("inf")},
        {"nested": [0, -float("inf")]},
        ["not", "a", "mapping"],
    ],
)
def test_unsafe_metadata_is_rejected(tmp_path: Path, metadata: Any) -> None:
    with pytest.raises(ArtifactValidationError):
        ArtifactStore(tmp_path).put({"hook": np.ones(1)}, metadata=metadata)


@pytest.mark.parametrize(
    "array",
    [
        np.array([object()], dtype=object),
        np.array(["text"], dtype="U4"),
        np.array([b"text"], dtype="S4"),
        np.array(["2026-01-01"], dtype="datetime64[D]"),
        np.array([1], dtype="timedelta64[D]"),
        np.ma.array([1.0, 2.0], mask=[False, True]),
        np.array(1.0, dtype=np.float32),
        np.array([], dtype=np.float32),
        np.empty((1, 0), dtype=np.float32),
        np.array([float("nan")], dtype=np.float32),
        np.array([float("inf")], dtype=np.float64),
        np.array([complex(float("nan"), 0.0)], dtype=np.complex64),
    ],
)
def test_unsafe_empty_scalar_and_nonfinite_arrays_are_rejected(
    tmp_path: Path, array: np.ndarray
) -> None:
    with pytest.raises(ArtifactValidationError):
        ArtifactStore(tmp_path).put({"hook": array})


@pytest.mark.parametrize(
    "hook_name",
    [
        "",
        ".hidden",
        "../escape",
        "two..dots",
        "nested/hook",
        "nested\\hook",
        "hook with space",
        "Kelvin_hook",
        "x" * 257,
        7,
    ],
)
def test_unsafe_hook_names_are_rejected(tmp_path: Path, hook_name: Any) -> None:
    with pytest.raises(ArtifactValidationError):
        ArtifactStore(tmp_path).put({hook_name: np.ones(1)})


def test_put_load_path_read_only_arrays_and_idempotence(tmp_path: Path) -> None:
    store, first = put_sample(tmp_path)
    expected = (
        tmp_path.resolve()
        / "artifacts"
        / "sha256"
        / first.artifact_hash[:2]
        / f"{first.artifact_hash}.npz"
    )
    before = first.path.stat()
    second = store.put(
        sample_arrays(), metadata={"seed": 17, "model": "model-v1"}
    )
    loaded = store.load(first.artifact_hash)

    assert first.path == expected
    assert first.relative_path == expected.relative_to(tmp_path).as_posix()
    assert first == second == loaded.stored
    assert second.path.stat().st_ino == before.st_ino
    assert second.path.stat().st_mtime_ns == before.st_mtime_ns
    assert set(loaded.arrays) == set(sample_arrays())
    assert all(not value.flags.writeable for value in loaded.arrays.values())
    with pytest.raises(ValueError):
        loaded.arrays["blocks.2.hook_resid_post"][0, 0] = 100
    with pytest.raises(TypeError):
        loaded.arrays["new"] = np.ones(1)  # type: ignore[index]


def test_temporary_files_are_removed_after_success_and_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    successful_root = tmp_path / "successful"
    ArtifactStore(successful_root).put({"hook": np.ones(2, dtype=np.float32)})
    assert not list(successful_root.rglob(".artifact-*"))

    failing_root = tmp_path / "failing"
    store = ArtifactStore(failing_root)

    def fail_replace(source: Any, destination: Any) -> None:
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr(artifacts_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        store.put({"hook": np.ones(2, dtype=np.float32)})
    assert not list(failing_root.rglob(".artifact-*"))
    assert not list(failing_root.rglob("*.npz"))


def test_occupied_corrupt_destination_is_a_collision_not_overwritten(
    tmp_path: Path,
) -> None:
    arrays = {"hook": np.arange(3, dtype=np.float32)}
    probe = ArtifactStore(tmp_path / "probe").put(arrays)
    store = ArtifactStore(tmp_path / "target")
    occupied = store.path_for(probe.artifact_hash, create_parent=True)
    occupied.write_bytes(b"occupied-corrupt-data")
    before = occupied.read_bytes()

    with pytest.raises(ArtifactCollisionError, match="refusing to overwrite"):
        store.put(arrays)
    assert occupied.read_bytes() == before


def test_modified_array_member_is_detected(tmp_path: Path) -> None:
    store, stored = put_sample(tmp_path)
    manifest = read_manifest(stored.path)
    descriptor = manifest["arrays"][0]
    replacement = np.full(descriptor["shape"], 999, dtype=descriptor["dtype_str"])
    replace_member(
        stored.path,
        f'{descriptor["storage_key"]}.npy',
        npy_bytes(replacement),
    )

    with pytest.raises(ArtifactCorruptionError, match="logical hash mismatch"):
        store.load(stored.artifact_hash)


@pytest.mark.parametrize(
    ("label", "mutation"),
    [
        (
            "metadata",
            lambda manifest: manifest["metadata"].update({"tampered": True}),
        ),
        (
            "dtype",
            lambda manifest: manifest["arrays"][0].update({"dtype": "float64"}),
        ),
        (
            "shape",
            lambda manifest: manifest["arrays"][0].update({"shape": [999, 4]}),
        ),
        (
            "nbytes",
            lambda manifest: manifest["arrays"][0].update({"nbytes": 1}),
        ),
        (
            "version",
            lambda manifest: manifest.update({"format_version": "future/9.0"}),
        ),
        (
            "extra manifest field",
            lambda manifest: manifest.update({"extra": "forbidden"}),
        ),
        (
            "missing manifest field",
            lambda manifest: manifest.pop("metadata"),
        ),
        (
            "extra descriptor field",
            lambda manifest: manifest["arrays"][0].update({"extra": True}),
        ),
        (
            "missing descriptor field",
            lambda manifest: manifest["arrays"][0].pop("shape"),
        ),
    ],
)
def test_modified_manifest_contract_is_detected(
    tmp_path: Path,
    label: str,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    del label
    store, stored = put_sample(tmp_path)
    mutate_manifest(stored.path, mutation)

    with pytest.raises(ArtifactError):
        store.load(stored.artifact_hash)


def test_noncanonical_manifest_json_is_detected(tmp_path: Path) -> None:
    store, stored = put_sample(tmp_path)
    manifest = read_manifest(stored.path)
    indented = json.dumps(manifest, indent=2, sort_keys=False).encode("utf-8")
    replace_member(
        stored.path,
        _MANIFEST_MEMBER,
        npy_bytes(np.frombuffer(indented, dtype=np.uint8)),
    )

    with pytest.raises(ArtifactCorruptionError, match="not canonical"):
        store.load(stored.artifact_hash)


def test_archive_copied_to_wrong_hash_path_fails_logical_hash_check(
    tmp_path: Path,
) -> None:
    store, stored = put_sample(tmp_path)
    wrong_hash = "f" * 64
    assert wrong_hash != stored.artifact_hash
    wrong_path = store.path_for(wrong_hash, create_parent=True)
    shutil.copyfile(stored.path, wrong_path)

    with pytest.raises(ArtifactCorruptionError, match="logical hash mismatch"):
        store.load(wrong_hash)


def test_missing_and_extra_archive_members_are_detected(tmp_path: Path) -> None:
    store, stored = put_sample(tmp_path)
    members = read_members(stored.path)
    array_member = next(name for name, _ in members if name.startswith("array_"))
    write_members(
        stored.path, [(name, value) for name, value in members if name != array_member]
    )
    with pytest.raises(ArtifactCorruptionError, match="missing"):
        store.load(stored.artifact_hash)

    store, stored = put_sample(tmp_path / "extra")
    members = read_members(stored.path)
    members.append(("undeclared.npy", npy_bytes(np.ones(1, dtype=np.float32))))
    write_members(stored.path, members)
    with pytest.raises(ArtifactCorruptionError, match="not declared"):
        store.load(stored.artifact_hash)


def test_duplicate_zip_members_are_detected(tmp_path: Path) -> None:
    store, stored = put_sample(tmp_path)
    members = read_members(stored.path)
    members.append(members[0])
    with pytest.warns(UserWarning, match="Duplicate name"):
        write_members(stored.path, members)

    with pytest.raises(ArtifactCorruptionError, match="duplicate ZIP members"):
        store.load(stored.artifact_hash)


@pytest.mark.parametrize(
    "unsafe_member", ["../escape.npy", "nested/escape.npy", "nested\\escape.npy"]
)
def test_unsafe_zip_member_paths_are_detected(
    tmp_path: Path, unsafe_member: str
) -> None:
    store, stored = put_sample(tmp_path)
    members = read_members(stored.path)
    members.append((unsafe_member, npy_bytes(np.ones(1, dtype=np.float32))))
    write_members(stored.path, members)

    with pytest.raises(ArtifactCorruptionError, match="unsafe NPZ member"):
        store.load(stored.artifact_hash)


def test_allow_pickle_false_is_enforced_and_object_npy_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, stored = put_sample(tmp_path)
    calls: list[bool | None] = []
    original_load = artifacts_module.np.load

    def guarded_load(*args: Any, **kwargs: Any):
        calls.append(kwargs.get("allow_pickle"))
        assert kwargs.get("allow_pickle") is False
        return original_load(*args, **kwargs)

    monkeypatch.setattr(artifacts_module.np, "load", guarded_load)
    store.load(stored.artifact_hash)
    assert calls and set(calls) == {False}

    monkeypatch.setattr(artifacts_module.np, "load", original_load)
    manifest = read_manifest(stored.path)
    array_member = f'{manifest["arrays"][0]["storage_key"]}.npy'
    replace_member(
        stored.path,
        array_member,
        npy_bytes(np.array([{"unsafe": True}], dtype=object), allow_pickle=True),
    )
    with pytest.raises(ArtifactCorruptionError):
        store.load(stored.artifact_hash)


def test_missing_and_corrupt_dependency_status(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    missing_hash = "a" * 64
    missing = store.dependency_status(missing_hash)
    assert missing.state == "missing"
    assert not missing.available
    with pytest.raises(ArtifactNotFoundError):
        store.load(missing_hash)

    stored = store.put({"hook": np.ones(2, dtype=np.float32)})
    assert store.dependency_status(stored.artifact_hash).state == "available"
    stored.path.write_bytes(b"corrupt")
    corrupt = store.dependency_status(stored.artifact_hash)
    assert corrupt.state == "corrupt"
    assert not corrupt.available
    assert corrupt.detail


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "a" * 63,
        "A" * 64,
        "g" * 64,
        "sha256:" + "a" * 64,
        "../" + "a" * 64,
        "a" * 64 + "/escape",
    ],
)
def test_digest_validation_prevents_path_traversal(
    tmp_path: Path, digest: str
) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactValidationError):
        store.path_for(digest)
    with pytest.raises(ArtifactValidationError):
        store.load(digest)


def test_root_symlink_is_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(UnsafeArtifactPathError, match="root must not be a symlink"):
        ArtifactStore(linked_root)


def test_prefix_symlink_escape_is_rejected(tmp_path: Path) -> None:
    arrays = {"hook": np.arange(3, dtype=np.float32)}
    probe = ArtifactStore(tmp_path / "probe").put(arrays)
    store = ArtifactStore(tmp_path / "store")
    outside = tmp_path / "outside"
    outside.mkdir()
    prefix = store.content_root / probe.artifact_hash[:2]
    prefix.symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafeArtifactPathError):
        store.put(arrays)


def test_file_symlink_escape_is_rejected(tmp_path: Path) -> None:
    arrays = {"hook": np.arange(3, dtype=np.float32)}
    probe = ArtifactStore(tmp_path / "probe").put(arrays)
    store = ArtifactStore(tmp_path / "store")
    target = store.path_for(probe.artifact_hash, create_parent=True)
    outside = tmp_path / "outside.npz"
    outside.write_bytes(probe.path.read_bytes())
    target.symlink_to(outside)

    with pytest.raises(UnsafeArtifactPathError, match="symlink"):
        store.load(probe.artifact_hash)
    with pytest.raises(UnsafeArtifactPathError, match="symlink"):
        store.put(arrays)


def test_stored_metadata_and_reference_fields_are_immutable(tmp_path: Path) -> None:
    store, stored = put_sample(tmp_path)
    array = stored.arrays[0]
    fields = array.reference_fields(stored.artifact_hash)

    with pytest.raises(FrozenInstanceError):
        stored.artifact_hash = "f" * 64  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        array.dtype = "float64"  # type: ignore[misc]
    with pytest.raises(TypeError):
        stored.metadata["seed"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        fields["dtype"] = "float64"  # type: ignore[index]
    assert fields == {
        "artifact_hash": stored.artifact_hash,
        "hook_name": array.hook_name,
        "shape": array.shape,
        "dtype": array.dtype,
    }


def test_file_and_logical_checksums_are_reported_and_reverified(tmp_path: Path) -> None:
    store, stored = put_sample(tmp_path)
    physical = hashlib.sha256(stored.path.read_bytes()).hexdigest()

    assert stored.file_sha256 == physical
    assert len(stored.artifact_hash) == 64
    assert stored.size_bytes == len(stored.path.read_bytes())
    assert stored.manifest_json == json.dumps(
        read_manifest(stored.path),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert read_manifest(stored.path)["format_version"] == ARTIFACT_FORMAT_VERSION
    assert store.verify(stored.artifact_hash) == stored


def test_atomic_concurrent_same_content_writers_are_idempotent(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    arrays = sample_arrays()

    def write_once(_: int):
        return store.put(arrays, metadata={"model": "model-v1", "seed": 17})

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(write_once, range(24)))

    assert len({result.artifact_hash for result in results}) == 1
    assert len({result.file_sha256 for result in results}) == 1
    assert len(list(store.content_root.rglob("*.npz"))) == 1
    assert not list(store.content_root.rglob(".artifact-*"))
    store.verify(results[0].artifact_hash)

