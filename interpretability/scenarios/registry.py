"""Explicit, versioned registry for released scenario specifications.

There is intentionally no module-global registry.  Callers own a
``ScenarioRegistry`` instance, register exact content-addressed specs, and
request exact semantic versions (or an explicitly configured default).
Filesystem and package-resource loaders validate the complete persisted wire
record before mutating registry state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
import json
import math
from pathlib import Path, PurePath, PurePosixPath
import re
import stat
from threading import RLock
from typing import Any, BinaryIO
from urllib.parse import quote

from interpretability.scenarios.compiler import compile_scenario
from interpretability.scenarios.schema import (
    SCENARIO_DSL_SCHEMA_VERSION,
    IncentiveCondition,
    ScenarioInstance,
    ScenarioSpec,
    canonical_json,
    canonical_sha256,
)


REGISTRY_MANIFEST_VERSION = "scenario-registry/1.0.0"
MAX_SCENARIO_SPEC_BYTES = 2 * 1024 * 1024
RELEASE_FILE_SUFFIX = ".scenario.json"

_SEMVER = re.compile(
    r"^(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_SCENARIO_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]*$")

RegistryKey = tuple[str, str]


class ScenarioRegistryError(ValueError):
    """Base class for deterministic registry failures."""


class DuplicateScenarioError(ScenarioRegistryError):
    """The exact scenario identity is already registered."""


class ScenarioCollisionError(ScenarioRegistryError):
    """A key or content hash collides with different registered content."""


class UnknownScenarioError(ScenarioRegistryError):
    """No specification is registered for a scenario identity."""


class UnknownScenarioVersionError(ScenarioRegistryError):
    """The requested exact semantic version is not registered."""


class ExactVersionRequiredError(ScenarioRegistryError):
    """Lookup omitted a version and no explicit default exists."""


class ScenarioLoadError(ScenarioRegistryError):
    """A released JSON file or package resource is unsafe or invalid."""


class UnsafeScenarioPathError(ScenarioLoadError):
    """A release path is non-regular, traversing, or symlinked."""


@dataclass(frozen=True, order=True, slots=True)
class RegistryEntry:
    """One immutable manifest entry sorted by scenario and semantic version."""

    scenario_id: str
    spec_version: str
    spec_hash: str
    is_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-domain manifest entry."""
        return {
            "scenario_id": self.scenario_id,
            "spec_version": self.spec_version,
            "spec_hash": self.spec_hash,
            "is_default": self.is_default,
        }


@dataclass(frozen=True, slots=True)
class RegistrySnapshot:
    """Canonical point-in-time registry manifest."""

    entries: tuple[RegistryEntry, ...]
    manifest_hash: str

    def canonical_json(self) -> str:
        """Return deterministic JSON for this complete snapshot."""
        return canonical_json(
            {
                "manifest_version": REGISTRY_MANIFEST_VERSION,
                "entries": [entry.to_dict() for entry in self.entries],
                "manifest_hash": self.manifest_hash,
            }
        )


def release_filename_for(scenario_id: str, spec_version: str) -> str:
    """Return the canonical filename for one released scenario version."""
    _validate_scenario_id(scenario_id)
    _validate_semantic_version(spec_version)
    encoded_id = quote(scenario_id, safe="._-")
    encoded_version = quote(spec_version, safe="._-")
    return f"{encoded_id}--{encoded_version}{RELEASE_FILE_SUFFIX}"


def _validate_scenario_id(scenario_id: str) -> None:
    if not isinstance(scenario_id, str) or not _SCENARIO_ID.fullmatch(scenario_id):
        raise ScenarioRegistryError("scenario_id is not a stable identifier")


def _validate_semantic_version(spec_version: str) -> None:
    if not isinstance(spec_version, str) or not _SEMVER.fullmatch(spec_version):
        raise UnknownScenarioVersionError(
            "spec_version must be an exact semantic version"
        )


def _validated_spec(spec: ScenarioSpec) -> ScenarioSpec:
    if not isinstance(spec, ScenarioSpec):
        raise TypeError("registry entries must be ScenarioSpec objects")
    if spec.schema_version != SCENARIO_DSL_SCHEMA_VERSION:
        raise ScenarioRegistryError("unsupported scenario schema version")
    validated = ScenarioSpec.from_persisted_json(spec.canonical_json())
    _validate_scenario_id(validated.metadata.scenario_id)
    _validate_semantic_version(validated.spec_version)
    return validated


def _manifest_hash(entries: tuple[RegistryEntry, ...]) -> str:
    return canonical_sha256(
        {
            "manifest_version": REGISTRY_MANIFEST_VERSION,
            "entries": [entry.to_dict() for entry in entries],
        }
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ScenarioLoadError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ScenarioLoadError(f"nonfinite JSON constant {value!r} is forbidden")


def _reject_nonfinite(value: Any, path: str = "spec") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ScenarioLoadError(f"{path} contains a nonfinite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, f"{path}[{index}]")


def _read_bounded(stream: BinaryIO, source: str) -> bytes:
    data = stream.read(MAX_SCENARIO_SPEC_BYTES + 1)
    if len(data) > MAX_SCENARIO_SPEC_BYTES:
        raise ScenarioLoadError(
            f"scenario spec {source!r} exceeds {MAX_SCENARIO_SPEC_BYTES} bytes"
        )
    return data


def _decode_released_spec(
    data: bytes,
    *,
    source: str,
    filename: str,
    require_canonical_json: bool,
) -> ScenarioSpec:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ScenarioLoadError(
            f"scenario spec {source!r} is not strict UTF-8"
        ) from exc
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except ScenarioLoadError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ScenarioLoadError(
            f"scenario spec {source!r} is not valid JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise ScenarioLoadError("released scenario JSON must contain one object")
    _reject_nonfinite(decoded)
    try:
        normalized = canonical_json(decoded)
    except (TypeError, ValueError) as exc:
        raise ScenarioLoadError(
            f"scenario spec {source!r} is not canonicalizable JSON"
        ) from exc
    if require_canonical_json and text != normalized:
        raise ScenarioLoadError(
            f"scenario spec {source!r} is not canonical JSON"
        )
    try:
        spec = ScenarioSpec.from_persisted(decoded)
    except (TypeError, ValueError) as exc:
        raise ScenarioLoadError(
            f"scenario spec {source!r} failed persisted identity validation"
        ) from exc
    expected_filename = release_filename_for(
        spec.metadata.scenario_id,
        spec.spec_version,
    )
    if filename != expected_filename:
        raise ScenarioLoadError(
            f"released filename must be {expected_filename!r}, got {filename!r}"
        )
    return spec


def _has_parent_traversal(path: PurePath) -> bool:
    """Return whether a supplied relative path contains unsafe segments."""
    return any(part in {"", ".", ".."} for part in path.parts)


# Path validation is deliberately explicit for each attack class.
# pylint: disable-next=too-many-branches
def _safe_release_path(
    path: str | Path,
    *,
    root: str | Path | None,
) -> Path:
    """Resolve one regular release while enforcing the declared trust boundary.

    A caller-supplied root is itself the trust anchor: its resolved target is
    accepted even when an ancestor is a platform-level symlink (for example,
    macOS ``/var``).  The root cannot itself be a symlink, every descendant
    named by the untrusted relative path must be non-symlinked, and the final
    target must remain beneath that one resolved root.
    """
    supplied = Path(path)
    if _has_parent_traversal(PurePath(supplied)):
        raise UnsafeScenarioPathError("release path cannot traverse directories")

    if root is not None:
        release_root = Path(root)
        if release_root.is_symlink():
            raise UnsafeScenarioPathError("release root cannot be a symlink")
        try:
            resolved_root = release_root.resolve(strict=True)
        except OSError as exc:
            raise UnsafeScenarioPathError("release root does not exist") from exc
        if not resolved_root.is_dir():
            raise UnsafeScenarioPathError("release root must be a directory")
        if supplied.is_absolute():
            raise UnsafeScenarioPathError(
                "paths under a release root must be relative"
            )
        current = release_root
        for part in supplied.parts:
            current = current / part
            if current.is_symlink():
                raise UnsafeScenarioPathError(
                    "release path cannot contain symlink components"
                )
        candidate = current
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            raise UnsafeScenarioPathError(
                "release path escapes its configured root or does not exist"
            ) from exc
    else:
        candidate = supplied
        if candidate.is_symlink():
            raise UnsafeScenarioPathError("release file cannot be a symlink")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise UnsafeScenarioPathError("release file does not exist") from exc

    try:
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise UnsafeScenarioPathError("release file cannot be inspected") from exc
    if not stat.S_ISREG(mode):
        raise UnsafeScenarioPathError("release path must be a regular file")
    return resolved


def _validate_resource_name(resource: str) -> PurePosixPath:
    if not isinstance(resource, str) or not resource or "\\" in resource:
        raise UnsafeScenarioPathError("package resource name is invalid")
    path = PurePosixPath(resource)
    if path.is_absolute() or _has_parent_traversal(path):
        raise UnsafeScenarioPathError(
            "package resource must be a non-traversing relative path"
        )
    return path


class ScenarioRegistry:
    """Thread-safe explicit owner of exact released scenario versions."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._specs: dict[RegistryKey, ScenarioSpec] = {}
        self._sources: dict[RegistryKey, str] = {}
        self._defaults: dict[str, str] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._specs)

    # Atomic staging keeps every collision/default check ahead of mutation.
    # pylint: disable-next=too-many-locals
    def _register_records(
        self,
        records: Iterable[tuple[ScenarioSpec, str]],
        *,
        defaults: Iterable[RegistryKey] = (),
    ) -> tuple[ScenarioSpec, ...]:
        validated_records = tuple(
            (_validated_spec(spec), source) for spec, source in records
        )
        requested_defaults = tuple(defaults)
        with self._lock:
            staged_specs = dict(self._specs)
            staged_sources = dict(self._sources)
            staged_defaults = dict(self._defaults)
            hash_to_key = {
                spec.spec_hash: key for key, spec in staged_specs.items()
            }

            for spec, source in validated_records:
                key = (spec.metadata.scenario_id, spec.spec_version)
                existing = staged_specs.get(key)
                if existing is not None:
                    if existing.spec_hash == spec.spec_hash:
                        raise DuplicateScenarioError(
                            f"scenario {key[0]}@{key[1]} is already registered"
                        )
                    raise ScenarioCollisionError(
                        f"scenario identity collision for {key[0]}@{key[1]}"
                    )
                hash_owner = hash_to_key.get(spec.spec_hash)
                if hash_owner is not None and hash_owner != key:
                    raise ScenarioCollisionError(
                        "scenario content hash is already registered under "
                        f"{hash_owner[0]}@{hash_owner[1]}"
                    )
                staged_specs[key] = spec
                staged_sources[key] = source
                hash_to_key[spec.spec_hash] = key

            for scenario_id, spec_version in requested_defaults:
                key = (scenario_id, spec_version)
                if key not in staged_specs:
                    raise UnknownScenarioVersionError(
                        f"cannot default unknown scenario {scenario_id}@{spec_version}"
                    )
                existing_default = staged_defaults.get(scenario_id)
                if existing_default is not None and existing_default != spec_version:
                    raise ScenarioRegistryError(
                        f"scenario {scenario_id!r} already has an explicit default"
                    )
                staged_defaults[scenario_id] = spec_version

            self._specs = staged_specs
            self._sources = staged_sources
            self._defaults = staged_defaults
        return tuple(spec for spec, _ in validated_records)

    def register(
        self,
        spec: ScenarioSpec,
        *,
        source: str = "<memory>",
        default: bool = False,
    ) -> ScenarioSpec:
        """Register one exact spec; duplicate registration is always an error."""
        if not isinstance(source, str) or not source:
            raise TypeError("registration source must be nonempty text")
        validated = _validated_spec(spec)
        key = (validated.metadata.scenario_id, validated.spec_version)
        defaults = (key,) if default else ()
        return self._register_records(
            ((validated, source),),
            defaults=defaults,
        )[0]

    def register_many(
        self,
        specs: Iterable[ScenarioSpec],
        *,
        defaults: Iterable[RegistryKey] = (),
    ) -> tuple[ScenarioSpec, ...]:
        """Atomically register multiple in-memory specs."""
        return self._register_records(
            ((spec, "<memory>") for spec in specs),
            defaults=defaults,
        )

    def set_default(self, scenario_id: str, spec_version: str) -> None:
        """Explicitly select one already registered default version."""
        _validate_scenario_id(scenario_id)
        _validate_semantic_version(spec_version)
        key = (scenario_id, spec_version)
        with self._lock:
            if key not in self._specs:
                raise UnknownScenarioVersionError(
                    f"scenario {scenario_id}@{spec_version} is not registered"
                )
            self._defaults = {**self._defaults, scenario_id: spec_version}

    def get(
        self,
        scenario_id: str,
        spec_version: str | None = None,
    ) -> ScenarioSpec:
        """Return one exact version, using only an explicitly declared default."""
        _validate_scenario_id(scenario_id)
        with self._lock:
            versions = tuple(
                sorted(version for name, version in self._specs if name == scenario_id)
            )
            if not versions:
                raise UnknownScenarioError(
                    f"scenario {scenario_id!r} is not registered"
                )
            selected = spec_version
            if selected is None:
                selected = self._defaults.get(scenario_id)
                if selected is None:
                    raise ExactVersionRequiredError(
                        f"scenario {scenario_id!r} requires an exact spec_version"
                    )
            _validate_semantic_version(selected)
            spec = self._specs.get((scenario_id, selected))
            if spec is None:
                raise UnknownScenarioVersionError(
                    f"scenario {scenario_id}@{selected} is not registered; "
                    f"available versions: {versions}"
                )
            return spec

    def list(self, scenario_id: str | None = None) -> tuple[RegistryEntry, ...]:
        """List immutable entries in deterministic identity order."""
        if scenario_id is not None:
            _validate_scenario_id(scenario_id)
        with self._lock:
            return tuple(
                sorted(
                    RegistryEntry(
                        scenario_id=key[0],
                        spec_version=key[1],
                        spec_hash=spec.spec_hash,
                        is_default=self._defaults.get(key[0]) == key[1],
                    )
                    for key, spec in self._specs.items()
                    if scenario_id is None or key[0] == scenario_id
                )
            )

    def list_specs(
        self,
        scenario_id: str | None = None,
    ) -> tuple[RegistryEntry, ...]:
        """Named alias for deterministic registry listing."""
        return self.list(scenario_id)

    def list_versions(self, scenario_id: str) -> tuple[str, ...]:
        """List exact registered semantic versions for one scenario."""
        entries = self.list(scenario_id)
        if not entries:
            raise UnknownScenarioError(
                f"scenario {scenario_id!r} is not registered"
            )
        return tuple(entry.spec_version for entry in entries)

    def snapshot(self) -> RegistrySnapshot:
        """Return a deterministic immutable manifest snapshot."""
        entries = self.list()
        return RegistrySnapshot(
            entries=entries,
            manifest_hash=_manifest_hash(entries),
        )

    def load_file(
        self,
        path: str | Path,
        *,
        root: str | Path | None = None,
        require_canonical_json: bool = True,
        default: bool = False,
    ) -> ScenarioSpec:
        """Strictly load and register one regular released JSON file."""
        resolved = _safe_release_path(path, root=root)
        try:
            with resolved.open("rb") as stream:
                data = _read_bounded(stream, str(resolved))
        except OSError as exc:
            raise ScenarioLoadError(
                f"cannot read scenario release {str(resolved)!r}"
            ) from exc
        spec = _decode_released_spec(
            data,
            source=str(resolved),
            filename=resolved.name,
            require_canonical_json=require_canonical_json,
        )
        return self.register(spec, source=str(resolved), default=default)

    def load_files(
        self,
        paths: Iterable[str | Path],
        *,
        root: str | Path | None = None,
        require_canonical_json: bool = True,
        defaults: Iterable[RegistryKey] = (),
    ) -> tuple[ScenarioSpec, ...]:
        """Strictly decode many files and register them atomically."""
        records: list[tuple[ScenarioSpec, str]] = []
        for path in paths:
            resolved = _safe_release_path(path, root=root)
            try:
                with resolved.open("rb") as stream:
                    data = _read_bounded(stream, str(resolved))
            except OSError as exc:
                raise ScenarioLoadError(
                    f"cannot read scenario release {str(resolved)!r}"
                ) from exc
            spec = _decode_released_spec(
                data,
                source=str(resolved),
                filename=resolved.name,
                require_canonical_json=require_canonical_json,
            )
            records.append((spec, str(resolved)))
        return self._register_records(records, defaults=defaults)

    def _decode_package_resource(
        self,
        package: str,
        resource: str,
        *,
        require_canonical_json: bool,
    ) -> tuple[ScenarioSpec, str]:
        if not isinstance(package, str) or not package:
            raise ScenarioLoadError("package name must be nonempty text")
        resource_path = _validate_resource_name(resource)
        try:
            root = resources.files(package)
        except (ModuleNotFoundError, TypeError) as exc:
            raise ScenarioLoadError(
                f"scenario resource package {package!r} is not available"
            ) from exc
        target = root.joinpath(*resource_path.parts)
        try:
            is_file = target.is_file()
        except OSError as exc:
            raise ScenarioLoadError(
                f"scenario resource {package}:{resource} cannot be inspected"
            ) from exc
        if not is_file:
            raise ScenarioLoadError(
                f"scenario resource {package}:{resource} is not a file"
            )
        try:
            with target.open("rb") as stream:
                data = _read_bounded(stream, f"{package}:{resource}")
        except (FileNotFoundError, IsADirectoryError, OSError) as exc:
            raise ScenarioLoadError(
                f"scenario resource {package}:{resource} cannot be read"
            ) from exc
        spec = _decode_released_spec(
            data,
            source=f"{package}:{resource}",
            filename=resource_path.name,
            require_canonical_json=require_canonical_json,
        )
        return spec, f"{package}:{resource}"

    def load_package_resource(
        self,
        package: str,
        resource: str,
        *,
        require_canonical_json: bool = True,
        default: bool = False,
    ) -> ScenarioSpec:
        """Load one released spec through ``importlib.resources``."""
        spec, source = self._decode_package_resource(
            package,
            resource,
            require_canonical_json=require_canonical_json,
        )
        return self.register(spec, source=source, default=default)

    def load_package_resources(
        self,
        package: str,
        resource_names: Iterable[str],
        *,
        require_canonical_json: bool = True,
        defaults: Iterable[RegistryKey] = (),
    ) -> tuple[ScenarioSpec, ...]:
        """Load several package resources and register them atomically."""
        records = tuple(
            self._decode_package_resource(
                package,
                resource,
                require_canonical_json=require_canonical_json,
            )
            for resource in resource_names
        )
        return self._register_records(records, defaults=defaults)


# Exact registry identity plus compilation identity form this boundary.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def compile_registered(
    registry: ScenarioRegistry,
    scenario_id: str,
    spec_version: str,
    trial_id: str,
    run_seed: int,
    condition: IncentiveCondition,
) -> ScenarioInstance:
    """Compile one exact registered scenario version."""
    if not isinstance(registry, ScenarioRegistry):
        raise TypeError("registry must be a ScenarioRegistry")
    spec = registry.get(scenario_id, spec_version)
    return compile_scenario(spec, trial_id, run_seed, condition)


__all__ = [
    "DuplicateScenarioError",
    "ExactVersionRequiredError",
    "MAX_SCENARIO_SPEC_BYTES",
    "REGISTRY_MANIFEST_VERSION",
    "RELEASE_FILE_SUFFIX",
    "RegistryEntry",
    "RegistrySnapshot",
    "ScenarioCollisionError",
    "ScenarioLoadError",
    "ScenarioRegistry",
    "ScenarioRegistryError",
    "UnknownScenarioError",
    "UnknownScenarioVersionError",
    "UnsafeScenarioPathError",
    "compile_registered",
    "release_filename_for",
]
