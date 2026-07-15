"""Safe content-addressed storage for activation-array artifacts.

Artifact identity is the SHA-256 digest of canonical logical metadata and array
bytes, not of the surrounding ZIP container.  Files are nevertheless written as
deterministic compressed NPZ archives so they remain inspectable with NumPy.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import struct
import tempfile
import threading
import unicodedata
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

ARTIFACT_FORMAT_VERSION = "activation-artifact/1.0.0"
_MANIFEST_KEY = "__manifest__"
_MANIFEST_MEMBER = f"{_MANIFEST_KEY}.npy"
_MANIFEST_LIMIT_BYTES = 1_048_576
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HOOK_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_ARRAY_DESCRIPTOR_FIELDS = frozenset(
    {"dtype", "dtype_str", "hook_name", "nbytes", "shape", "storage_key"}
)
_MANIFEST_FIELDS = frozenset({"arrays", "format_version", "metadata"})


class ArtifactError(RuntimeError):
    """Base class for artifact-store failures."""


class ArtifactValidationError(ArtifactError, ValueError):
    """Raised when data cannot be represented as a safe activation artifact."""


class ArtifactNotFoundError(ArtifactError, FileNotFoundError):
    """Raised when a referenced artifact dependency is unavailable."""


class ArtifactCorruptionError(ArtifactError):
    """Raised when stored bytes do not reproduce their declared logical hash."""


class ArtifactCollisionError(ArtifactError):
    """Raised rather than overwriting an occupied content-addressed path."""


class UnsafeArtifactPathError(ArtifactError):
    """Raised when a store path is a symlink or escapes the configured root."""


@dataclass(frozen=True, slots=True)
class StoredArrayMetadata:
    """Immutable logical metadata for one array in an artifact."""

    hook_name: str
    storage_key: str
    dtype: str
    dtype_str: str
    shape: tuple[int, ...]
    nbytes: int

    def reference_fields(self, artifact_hash: str) -> Mapping[str, Any]:
        """Return stored fields used when constructing ``ArtifactReference``."""

        return MappingProxyType(
            {
                "artifact_hash": artifact_hash,
                "hook_name": self.hook_name,
                "shape": self.shape,
                "dtype": self.dtype,
            }
        )


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """Immutable identity and integrity metadata for a persisted archive."""

    artifact_hash: str
    path: Path
    relative_path: str
    arrays: tuple[StoredArrayMetadata, ...]
    metadata_json: str
    manifest_json: str
    file_sha256: str
    size_bytes: int

    @property
    def metadata(self) -> Mapping[str, Any]:
        parsed = json.loads(self.metadata_json)
        return _freeze_json(parsed)

    def array(self, hook_name: str) -> StoredArrayMetadata:
        for array in self.arrays:
            if array.hook_name == hook_name:
                return array
        raise KeyError(hook_name)


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    """A verified stored artifact and its read-only activation arrays."""

    stored: StoredArtifact
    arrays: Mapping[str, np.ndarray]

    @property
    def artifact_hash(self) -> str:
        return self.stored.artifact_hash

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.stored.metadata


@dataclass(frozen=True, slots=True)
class ArtifactDependencyStatus:
    """Non-throwing availability result for a projected artifact dependency."""

    artifact_hash: str
    path: Path
    state: Literal["available", "missing", "corrupt"]
    detail: str | None = None

    @property
    def available(self) -> bool:
        return self.state == "available"


@dataclass(frozen=True, slots=True)
class _PreparedArtifact:
    artifact_hash: str
    arrays: tuple[tuple[StoredArrayMetadata, np.ndarray], ...]
    metadata_json: str
    manifest_json: str


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _normalize_json(value: Any, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactValidationError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactValidationError(f"{path} keys must be strings")
            normalized[key] = _normalize_json(item, f"{path}.{key}")
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [
            _normalize_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ArtifactValidationError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_hash(artifact_hash: str) -> str:
    if not isinstance(artifact_hash, str) or not _HASH_PATTERN.fullmatch(artifact_hash):
        raise ArtifactValidationError(
            "artifact_hash must be a lowercase SHA-256 hexadecimal digest"
        )
    return artifact_hash


def _normalize_hook_name(hook_name: str) -> tuple[str, str]:
    if not isinstance(hook_name, str):
        raise ArtifactValidationError("activation hook names must be strings")
    normalized_unicode = unicodedata.normalize("NFKC", hook_name)
    if normalized_unicode != hook_name:
        raise ArtifactValidationError(
            f"hook name {hook_name!r} is not in canonical Unicode form"
        )
    if (
        not _HOOK_PATTERN.fullmatch(hook_name)
        or ".." in hook_name
        or hook_name in {".", ".."}
    ):
        raise ArtifactValidationError(f"unsafe activation hook name {hook_name!r}")
    return hook_name, hook_name.casefold()


def _to_numpy(value: Any, hook_name: str) -> np.ndarray:
    if isinstance(value, np.ma.MaskedArray):
        raise ArtifactValidationError(
            f"activation {hook_name!r} must not be a masked array"
        )
    if isinstance(value, np.ndarray):
        array = value
    elif type(value).__module__.split(".", 1)[0] == "torch":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise ArtifactValidationError(
                "a torch tensor was supplied but torch is unavailable"
            ) from exc
        if not isinstance(value, torch.Tensor):
            raise ArtifactValidationError(
                f"activation {hook_name!r} must be a NumPy array or torch.Tensor"
            )
        try:
            array = value.detach().cpu().numpy()
        except (RuntimeError, TypeError) as exc:
            raise ArtifactValidationError(
                f"activation {hook_name!r} cannot be converted safely to NumPy"
            ) from exc
    else:
        raise ArtifactValidationError(
            f"activation {hook_name!r} must be a NumPy array or torch.Tensor"
        )

    dtype = array.dtype
    if dtype.hasobject or dtype.kind not in "biufc":
        raise ArtifactValidationError(
            f"activation {hook_name!r} has unsafe dtype {dtype}; only finite "
            "numeric and boolean arrays are supported"
        )
    if array.ndim == 0 or array.size == 0:
        raise ArtifactValidationError(
            f"activation {hook_name!r} must have a non-empty shape"
        )
    try:
        finite = bool(np.all(np.isfinite(array)))
    except TypeError as exc:
        raise ArtifactValidationError(
            f"activation {hook_name!r} cannot be checked for finite values"
        ) from exc
    if not finite:
        raise ArtifactValidationError(
            f"activation {hook_name!r} contains NaN or infinite values"
        )

    canonical_dtype = dtype.newbyteorder("<") if dtype.itemsize > 1 else dtype
    canonical = np.array(array, dtype=canonical_dtype, order="C", copy=True)
    canonical.setflags(write=False)
    return canonical


def _logical_hash(manifest_json: str, arrays: tuple[np.ndarray, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"multiagent-lab:activation-artifact\x00")
    manifest_bytes = manifest_json.encode("utf-8")
    digest.update(struct.pack(">Q", len(manifest_bytes)))
    digest.update(manifest_bytes)
    for array in arrays:
        raw = array.tobytes(order="C")
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _write_deterministic_npz(
    stream: Any,
    manifest_json: str,
    arrays: tuple[tuple[StoredArrayMetadata, np.ndarray], ...],
) -> None:
    entries = [
        (_MANIFEST_MEMBER, np.frombuffer(manifest_json.encode("utf-8"), dtype=np.uint8))
    ]
    entries.extend((f"{metadata.storage_key}.npy", array) for metadata, array in arrays)
    with zipfile.ZipFile(
        stream, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for member_name, array in entries:
            info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                _npy_bytes(array),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )


def _prepare(
    activations: Mapping[str, Any], metadata: Mapping[str, Any] | None
) -> _PreparedArtifact:
    if not isinstance(activations, Mapping) or not activations:
        raise ArtifactValidationError("activations must be a non-empty mapping")
    normalized_metadata = _normalize_json({} if metadata is None else metadata)
    if not isinstance(normalized_metadata, dict):
        raise ArtifactValidationError("metadata must be a mapping")

    named_arrays: list[tuple[str, str, np.ndarray]] = []
    normalized_hooks: dict[str, str] = {}
    for hook_name, value in activations.items():
        canonical_name, normalized_name = _normalize_hook_name(hook_name)
        previous = normalized_hooks.get(normalized_name)
        if previous is not None:
            raise ArtifactValidationError(
                f"hook names {previous!r} and {canonical_name!r} collide after "
                "normalization"
            )
        normalized_hooks[normalized_name] = canonical_name
        named_arrays.append(
            (canonical_name, normalized_name, _to_numpy(value, canonical_name))
        )
    named_arrays.sort(key=lambda item: (item[1], item[0]))

    prepared_arrays: list[tuple[StoredArrayMetadata, np.ndarray]] = []
    descriptors: list[dict[str, Any]] = []
    for index, (hook_name, _, array) in enumerate(named_arrays):
        storage_key = f"array_{index:06d}"
        array_metadata = StoredArrayMetadata(
            hook_name=hook_name,
            storage_key=storage_key,
            dtype=array.dtype.name,
            dtype_str=array.dtype.str,
            shape=tuple(int(size) for size in array.shape),
            nbytes=int(array.nbytes),
        )
        prepared_arrays.append((array_metadata, array))
        descriptors.append(
            {
                "dtype": array_metadata.dtype,
                "dtype_str": array_metadata.dtype_str,
                "hook_name": array_metadata.hook_name,
                "nbytes": array_metadata.nbytes,
                "shape": list(array_metadata.shape),
                "storage_key": array_metadata.storage_key,
            }
        )

    manifest = {
        "arrays": descriptors,
        "format_version": ARTIFACT_FORMAT_VERSION,
        "metadata": normalized_metadata,
    }
    manifest_json = _canonical_json(manifest)
    artifact_hash = _logical_hash(
        manifest_json, tuple(array for _, array in prepared_arrays)
    )
    return _PreparedArtifact(
        artifact_hash=artifact_hash,
        arrays=tuple(prepared_arrays),
        metadata_json=_canonical_json(normalized_metadata),
        manifest_json=manifest_json,
    )


class ArtifactStore:
    """Content-addressed activation store rooted at an explicit directory."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        if root is None or str(root) == "":
            raise ArtifactValidationError("ArtifactStore requires an explicit root")
        requested_root = Path(root).expanduser()
        if requested_root.is_symlink():
            raise UnsafeArtifactPathError("artifact-store root must not be a symlink")
        try:
            requested_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise UnsafeArtifactPathError(
                f"cannot create artifact-store root {requested_root}"
            ) from exc
        if not requested_root.is_dir():
            raise UnsafeArtifactPathError(
                f"artifact-store root is not a directory: {requested_root}"
            )
        self._root = requested_root.resolve(strict=True)
        self._lock = threading.RLock()
        artifacts_dir = self._safe_directory(self._root / "artifacts")
        self._content_root = self._safe_directory(artifacts_dir / "sha256")

    @property
    def root(self) -> Path:
        return self._root

    @property
    def content_root(self) -> Path:
        return self._content_root

    def _assert_contained(self, path: Path) -> None:
        try:
            resolved = path.resolve(strict=False)
        except OSError as exc:
            raise UnsafeArtifactPathError(f"cannot resolve artifact path {path}") from exc
        if not resolved.is_relative_to(self._root):
            raise UnsafeArtifactPathError(
                f"artifact path escapes configured root: {path}"
            )

    def _safe_directory(self, path: Path) -> Path:
        self._assert_contained(path)
        if path.is_symlink():
            raise UnsafeArtifactPathError(f"artifact directory is a symlink: {path}")
        try:
            path.mkdir(exist_ok=True)
        except OSError as exc:
            raise UnsafeArtifactPathError(
                f"cannot create artifact directory {path}"
            ) from exc
        if path.is_symlink() or not path.is_dir():
            raise UnsafeArtifactPathError(f"unsafe artifact directory: {path}")
        self._assert_contained(path)
        return path

    def path_for(self, artifact_hash: str, *, create_parent: bool = False) -> Path:
        digest = _validate_hash(artifact_hash)
        prefix_dir = self._content_root / digest[:2]
        if create_parent:
            prefix_dir = self._safe_directory(prefix_dir)
        else:
            self._assert_contained(prefix_dir)
            if prefix_dir.is_symlink():
                raise UnsafeArtifactPathError(
                    f"artifact prefix directory is a symlink: {prefix_dir}"
                )
        path = prefix_dir / f"{digest}.npz"
        self._assert_contained(path)
        if path.is_symlink():
            raise UnsafeArtifactPathError(f"artifact file is a symlink: {path}")
        return path

    def exists(self, artifact_hash: str) -> bool:
        path = self.path_for(artifact_hash)
        return path.is_file()

    def dependency_status(self, artifact_hash: str) -> ArtifactDependencyStatus:
        path = self.path_for(artifact_hash)
        try:
            self.verify(artifact_hash)
        except ArtifactNotFoundError as exc:
            return ArtifactDependencyStatus(artifact_hash, path, "missing", str(exc))
        except (ArtifactCorruptionError, UnsafeArtifactPathError) as exc:
            return ArtifactDependencyStatus(artifact_hash, path, "corrupt", str(exc))
        return ArtifactDependencyStatus(artifact_hash, path, "available")

    def put(
        self,
        activations: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> StoredArtifact:
        """Validate and atomically persist one logical activation artifact."""

        prepared = _prepare(activations, metadata)
        with self._lock:
            destination = self.path_for(prepared.artifact_hash, create_parent=True)
            if os.path.lexists(destination):
                return self._accept_existing_or_raise(destination, prepared)

            descriptor: int | None = None
            temporary_path: Path | None = None
            try:
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".artifact-", suffix=".npz.tmp", dir=destination.parent
                )
                temporary_path = Path(temporary_name)
                with os.fdopen(descriptor, "w+b") as stream:
                    descriptor = None
                    _write_deterministic_npz(
                        stream, prepared.manifest_json, prepared.arrays
                    )
                    stream.flush()
                    os.fsync(stream.fileno())

                validated = self._read_archive(
                    temporary_path, expected_hash=prepared.artifact_hash
                )
                if not self._matches_prepared(validated, prepared):
                    raise ArtifactCorruptionError(
                        "temporary artifact differs from the logical data supplied"
                    )
                if os.path.lexists(destination):
                    return self._accept_existing_or_raise(destination, prepared)

                os.replace(temporary_path, destination)
                temporary_path = None
                self._fsync_directory(destination.parent)
                return self.verify(prepared.artifact_hash)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    def load(self, artifact_hash: str) -> LoadedArtifact:
        """Load arrays with ``allow_pickle=False`` and verify all logical metadata."""

        digest = _validate_hash(artifact_hash)
        path = self.path_for(digest)
        if not path.is_file():
            raise ArtifactNotFoundError(f"artifact {digest} is missing at {path}")
        return self._read_archive(path, expected_hash=digest)

    def verify(self, artifact_hash: str) -> StoredArtifact:
        """Verify a dependency and return immutable stored metadata."""

        return self.load(artifact_hash).stored

    def _accept_existing_or_raise(
        self, path: Path, prepared: _PreparedArtifact
    ) -> StoredArtifact:
        if path.is_symlink():
            raise UnsafeArtifactPathError(f"artifact file is a symlink: {path}")
        try:
            existing = self._read_archive(path, expected_hash=prepared.artifact_hash)
        except (ArtifactCorruptionError, ArtifactNotFoundError) as exc:
            raise ArtifactCollisionError(
                f"refusing to overwrite occupied artifact path {path}"
            ) from exc
        if not self._matches_prepared(existing, prepared):
            raise ArtifactCollisionError(
                f"SHA-256 collision at {path}; stored logical content differs"
            )
        return existing.stored

    @staticmethod
    def _matches_prepared(
        loaded: LoadedArtifact, prepared: _PreparedArtifact
    ) -> bool:
        if loaded.stored.manifest_json != prepared.manifest_json:
            return False
        if tuple(loaded.arrays) != tuple(
            metadata.hook_name for metadata, _ in prepared.arrays
        ):
            return False
        return all(
            np.array_equal(loaded.arrays[metadata.hook_name], array)
            for metadata, array in prepared.arrays
        )

    def _read_archive(
        self, path: Path, *, expected_hash: str
    ) -> LoadedArtifact:
        self._assert_contained(path)
        if path.is_symlink():
            raise UnsafeArtifactPathError(f"artifact file is a symlink: {path}")
        if not path.is_file():
            raise ArtifactNotFoundError(
                f"artifact {expected_hash} is missing at {path}"
            )

        try:
            with zipfile.ZipFile(path, mode="r") as zip_archive:
                members = [info.filename for info in zip_archive.infolist()]
                if len(members) != len(set(members)):
                    raise ArtifactCorruptionError("NPZ contains duplicate ZIP members")
                for member in members:
                    pure = PurePosixPath(member)
                    if (
                        pure.is_absolute()
                        or len(pure.parts) != 1
                        or "\\" in member
                        or not member.endswith(".npy")
                    ):
                        raise ArtifactCorruptionError(
                            f"unsafe NPZ member name {member!r}"
                        )

            with np.load(path, allow_pickle=False) as archive:
                archive_keys = list(archive.files)
                if len(archive_keys) != len(set(archive_keys)):
                    raise ArtifactCorruptionError("NPZ contains duplicate array keys")
                if _MANIFEST_KEY not in archive_keys:
                    raise ArtifactCorruptionError("NPZ manifest is missing")
                manifest_array = archive[_MANIFEST_KEY]
                if manifest_array.dtype != np.dtype("uint8") or manifest_array.ndim != 1:
                    raise ArtifactCorruptionError(
                        "NPZ manifest must be a one-dimensional uint8 array"
                    )
                if manifest_array.nbytes > _MANIFEST_LIMIT_BYTES:
                    raise ArtifactCorruptionError("NPZ manifest exceeds the size limit")
                try:
                    manifest_json = manifest_array.tobytes().decode("utf-8")
                    manifest = json.loads(
                        manifest_json,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f"non-finite JSON constant {value}")
                        ),
                    )
                except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                    raise ArtifactCorruptionError("NPZ manifest is invalid JSON") from exc
                if not isinstance(manifest, dict) or frozenset(manifest) != _MANIFEST_FIELDS:
                    raise ArtifactCorruptionError("NPZ manifest fields are not exact")
                if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
                    raise ArtifactCorruptionError("unsupported artifact format version")
                normalized_metadata = _normalize_json(manifest.get("metadata"))
                if not isinstance(normalized_metadata, dict):
                    raise ArtifactCorruptionError("artifact metadata must be an object")
                if manifest_json != _canonical_json(manifest):
                    raise ArtifactCorruptionError("NPZ manifest is not canonical JSON")

                descriptors = manifest.get("arrays")
                if not isinstance(descriptors, list) or not descriptors:
                    raise ArtifactCorruptionError("NPZ must describe at least one array")
                expected_keys = {_MANIFEST_KEY}
                array_metadata: list[StoredArrayMetadata] = []
                loaded_arrays: dict[str, np.ndarray] = {}
                normalized_hooks: set[str] = set()
                logical_arrays: list[np.ndarray] = []

                for index, descriptor in enumerate(descriptors):
                    if (
                        not isinstance(descriptor, dict)
                        or frozenset(descriptor) != _ARRAY_DESCRIPTOR_FIELDS
                    ):
                        raise ArtifactCorruptionError(
                            f"array descriptor {index} fields are not exact"
                        )
                    hook_name, normalized_hook = _normalize_hook_name(
                        descriptor.get("hook_name")
                    )
                    if normalized_hook in normalized_hooks:
                        raise ArtifactCorruptionError(
                            "array descriptors contain colliding hook names"
                        )
                    normalized_hooks.add(normalized_hook)
                    storage_key = descriptor.get("storage_key")
                    if storage_key != f"array_{index:06d}":
                        raise ArtifactCorruptionError(
                            f"array descriptor {index} has an invalid storage key"
                        )
                    expected_keys.add(storage_key)
                    if storage_key not in archive_keys:
                        raise ArtifactCorruptionError(
                            f"NPZ array {storage_key!r} is missing"
                        )
                    shape_value = descriptor.get("shape")
                    if (
                        not isinstance(shape_value, list)
                        or not shape_value
                        or any(
                            isinstance(size, bool)
                            or not isinstance(size, int)
                            or size <= 0
                            for size in shape_value
                        )
                    ):
                        raise ArtifactCorruptionError(
                            f"array descriptor {index} has an invalid shape"
                        )
                    shape = tuple(shape_value)
                    dtype_name = descriptor.get("dtype")
                    dtype_str = descriptor.get("dtype_str")
                    if not isinstance(dtype_name, str) or not isinstance(dtype_str, str):
                        raise ArtifactCorruptionError(
                            f"array descriptor {index} has an invalid dtype"
                        )
                    try:
                        declared_dtype = np.dtype(dtype_str)
                    except TypeError as exc:
                        raise ArtifactCorruptionError(
                            f"array descriptor {index} has an unknown dtype"
                        ) from exc
                    if (
                        declared_dtype.hasobject
                        or declared_dtype.kind not in "biufc"
                        or declared_dtype.name != dtype_name
                    ):
                        raise ArtifactCorruptionError(
                            f"array descriptor {index} has an unsafe dtype"
                        )
                    nbytes = descriptor.get("nbytes")
                    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0:
                        raise ArtifactCorruptionError(
                            f"array descriptor {index} has invalid nbytes"
                        )

                    array = archive[storage_key]
                    if (
                        array.dtype.str != dtype_str
                        or array.dtype.name != dtype_name
                        or tuple(array.shape) != shape
                        or int(array.nbytes) != nbytes
                    ):
                        raise ArtifactCorruptionError(
                            f"array {hook_name!r} does not match its descriptor"
                        )
                    if not bool(np.all(np.isfinite(array))):
                        raise ArtifactCorruptionError(
                            f"array {hook_name!r} contains non-finite values"
                        )
                    array.setflags(write=False)
                    metadata = StoredArrayMetadata(
                        hook_name=hook_name,
                        storage_key=storage_key,
                        dtype=dtype_name,
                        dtype_str=dtype_str,
                        shape=shape,
                        nbytes=nbytes,
                    )
                    array_metadata.append(metadata)
                    loaded_arrays[hook_name] = array
                    logical_arrays.append(array)

                if set(archive_keys) != expected_keys:
                    raise ArtifactCorruptionError(
                        "NPZ contains arrays not declared by the manifest"
                    )

            logical_hash = _logical_hash(manifest_json, tuple(logical_arrays))
            if logical_hash != expected_hash:
                raise ArtifactCorruptionError(
                    f"artifact logical hash mismatch: expected {expected_hash}, "
                    f"computed {logical_hash}"
                )
            relative_path = path.relative_to(self._root).as_posix()
            stored = StoredArtifact(
                artifact_hash=logical_hash,
                path=path,
                relative_path=relative_path,
                arrays=tuple(array_metadata),
                metadata_json=_canonical_json(normalized_metadata),
                manifest_json=manifest_json,
                file_sha256=_file_sha256(path),
                size_bytes=path.stat().st_size,
            )
            return LoadedArtifact(stored, MappingProxyType(loaded_arrays))
        except (ArtifactError, UnsafeArtifactPathError):
            raise
        except (OSError, EOFError, KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
            raise ArtifactCorruptionError(f"cannot safely read artifact {path}") from exc

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "ArtifactCollisionError",
    "ArtifactCorruptionError",
    "ArtifactDependencyStatus",
    "ArtifactError",
    "ArtifactNotFoundError",
    "ArtifactStore",
    "ArtifactValidationError",
    "LoadedArtifact",
    "StoredArrayMetadata",
    "StoredArtifact",
    "UnsafeArtifactPathError",
]
