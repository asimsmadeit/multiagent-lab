"""Safe public array bundles and explicit trusted legacy loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


ARRAY_BUNDLE_SCHEMA_VERSION = "1.0.0"
_ARRAY_BUNDLE_FIELDS = frozenset({
    "schema_version",
    "array_file",
    "array_sha256",
    "arrays",
    "manifest",
})
_ARRAY_SPECIFICATION_FIELDS = frozenset({"shape", "dtype"})


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_array_bundle(
    base_path: str | Path,
    arrays: Mapping[str, np.ndarray | torch.Tensor],
    manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write non-executable arrays plus a checksummed JSON manifest."""
    base = Path(base_path)
    array_path = base.with_suffix(".npz")
    manifest_path = base.with_suffix(".json")
    if not arrays:
        raise ValueError("array bundle must contain at least one array")
    normalized: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not name or not isinstance(name, str):
            raise ValueError("array names must be non-empty strings")
        array = (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )
        if array.dtype.hasobject:
            raise TypeError(f"array {name!r} has executable object dtype")
        normalized[name] = array
    manifest_payload = dict(manifest)
    json.dumps(manifest_payload, sort_keys=True, allow_nan=False)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(array_path, **normalized)
    payload = {
        "schema_version": ARRAY_BUNDLE_SCHEMA_VERSION,
        "array_file": array_path.name,
        "array_sha256": _sha256(array_path),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in normalized.items()
        },
        "manifest": manifest_payload,
    }
    # Fail before publishing metadata if the caller supplied non-JSON content.
    serialized = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    manifest_path.write_text(serialized + "\n", encoding="utf-8")
    return array_path, manifest_path


def load_array_bundle(
    manifest_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a checksummed bundle with NumPy pickle support disabled."""
    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("array bundle root must be a mapping")
    _require_exact_fields(
        payload,
        _ARRAY_BUNDLE_FIELDS,
        context="array bundle",
    )
    if payload.get("schema_version") != ARRAY_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported array bundle schema version")
    array_file = payload["array_file"]
    if not isinstance(array_file, str) or not array_file:
        raise ValueError("array_file must be a non-empty string")
    if Path(array_file).name != array_file:
        raise ValueError("array_file must name a sibling file")
    if (
        not isinstance(payload["array_sha256"], str)
        or len(payload["array_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in payload["array_sha256"])
    ):
        raise ValueError("array_sha256 must be a lowercase SHA-256 digest")
    specifications = payload["arrays"]
    if not isinstance(specifications, Mapping) or not specifications:
        raise ValueError("array bundle arrays must be a non-empty mapping")
    for name, specification in specifications.items():
        if not isinstance(name, str) or not name:
            raise ValueError("array bundle names must be non-empty strings")
        _require_exact_fields(
            specification,
            _ARRAY_SPECIFICATION_FIELDS,
            context=f"array bundle specification {name!r}",
        )
        if not isinstance(specification["shape"], list) or any(
            type(dimension) is not int or dimension < 0
            for dimension in specification["shape"]
        ):
            raise ValueError("array bundle shapes must be arrays of dimensions")
        if not isinstance(specification["dtype"], str) or not specification["dtype"]:
            raise ValueError("array bundle dtypes must be non-empty strings")
    if not isinstance(payload["manifest"], Mapping):
        raise TypeError("array bundle manifest must be a mapping")
    array_path = manifest_file.with_name(array_file)
    if _sha256(array_path) != payload.get("array_sha256"):
        raise ValueError("array bundle checksum mismatch")
    arrays: dict[str, np.ndarray] = {}
    with np.load(array_path, allow_pickle=False) as bundle:
        if set(bundle.files) != set(specifications):
            raise ValueError("array bundle members do not match manifest")
        for name in bundle.files:
            array = np.asarray(bundle[name])
            specification = specifications[name]
            if list(array.shape) != specification.get("shape"):
                raise ValueError(f"array shape mismatch for {name}")
            if str(array.dtype) != specification.get("dtype"):
                raise ValueError(f"array dtype mismatch for {name}")
            arrays[name] = array
    return arrays, dict(payload["manifest"])


def load_trusted_legacy_torch(
    path: str | Path,
    *,
    trusted: bool = False,
) -> Any:
    """Load a pickle-capable legacy artifact only after explicit trust opt-in."""
    if not trusted:
        raise PermissionError(
            "Legacy .pt files can execute pickle payloads. Pass trusted=True "
            "(or use --trust-legacy-pt) only for a reviewed artifact, or migrate "
            "it to an array bundle."
        )
    return torch.load(Path(path), weights_only=False)
