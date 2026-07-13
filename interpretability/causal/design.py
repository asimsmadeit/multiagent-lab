"""Frozen causal intervention designs defined before confirmatory execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from numbers import Integral, Real
import re
from typing import Any, Mapping, Sequence

import numpy as np


CAUSAL_DESIGN_SCHEMA_VERSION = "1.3.0"
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_ROW_LEVEL_UNITS = {"row", "rows", "sample", "samples", "turn", "turns"}


class InterventionKind(str, Enum):
    PATCH = "patch"
    ABLATE = "ablate"
    STEER = "steer"
    CLAMP = "clamp"


class ControlKind(str, Enum):
    ZERO_HOOK = "zero_hook"
    SHAM_HOOK = "sham_hook"
    NORM_MATCHED_RANDOM = "norm_matched_random"
    LABEL_SHUFFLED = "label_shuffled"
    NUISANCE_DIRECTION = "nuisance_direction"
    POSITIVE_HOOK = "positive_hook"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _current_schema_registry_checksum() -> str:
    from interpretability.schema_registry import schema_registry_checksum

    return schema_registry_checksum()


def canonical_vector_sha256(value: Any) -> tuple[str, int]:
    """Hash one finite 1-D vector using canonical little-endian float64 bytes."""
    vector = np.asarray(value, dtype="<f8")
    if vector.ndim != 1 or vector.size < 1:
        raise ValueError("causal directions must be non-empty one-dimensional vectors")
    if not np.isfinite(vector).all():
        raise ValueError("causal directions must contain only finite values")
    canonical = np.ascontiguousarray(vector, dtype="<f8")
    digest = hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
    return f"sha256:{digest}", int(canonical.size)


@dataclass(frozen=True)
class DirectionVectorIdentity:
    """Tamper-evident identity for a named canonical float64 direction vector."""

    source: str
    sha256: str
    dimension: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("direction identity source must be non-empty")
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.sha256
        ):
            raise ValueError("direction identity requires a canonical SHA256")
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, Integral)
            or self.dimension < 1
        ):
            raise ValueError("direction identity dimension must be positive")
        object.__setattr__(self, "dimension", int(self.dimension))

    @classmethod
    def from_vector(cls, source: str, vector: Any) -> "DirectionVectorIdentity":
        digest, dimension = canonical_vector_sha256(vector)
        return cls(source=source, sha256=digest, dimension=dimension)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sha256": self.sha256,
            "dimension": self.dimension,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DirectionVectorIdentity":
        return cls(
            source=value["source"],
            sha256=value["sha256"],
            dimension=value["dimension"],
        )


def _identity(value: DirectionVectorIdentity | Mapping[str, Any]) -> DirectionVectorIdentity:
    if isinstance(value, DirectionVectorIdentity):
        return value
    if isinstance(value, Mapping):
        return DirectionVectorIdentity.from_dict(value)
    raise TypeError("direction identity must be a DirectionVectorIdentity or mapping")


def _finite_coefficients(values: Sequence[float]) -> tuple[float, ...]:
    raw = tuple(values)
    if not raw:
        raise ValueError("at least one intervention coefficient is required")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw):
        raise ValueError("intervention coefficients must be numeric")
    coefficients = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError("intervention coefficients must be finite")
    if len(set(coefficients)) != len(coefficients):
        raise ValueError("intervention coefficients must not contain duplicates")
    return coefficients


def _string_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class CausalDesignManifest:
    """Pre-execution identity and estimand for a causal intervention family."""

    study_id: str
    dataset_hash: str
    intervention: InterventionKind
    intervention_adapter_version: str
    direction_source: str
    primary_direction_identity: DirectionVectorIdentity
    layer: int
    token_stage: str
    coefficients: tuple[float, ...]
    outcomes: tuple[str, ...]
    independent_unit: str
    group_key: str
    controls: tuple[ControlKind, ...]
    repetitions: int
    random_seed: int
    schema_registry_checksum: str = field(
        default_factory=_current_schema_registry_checksum
    )
    n_bootstrap: int = 2000
    n_permutations: int = 10000
    alpha: float = 0.05
    scorer_version: str | None = None
    prompt_ids: tuple[str, ...] = ()
    nuisance_direction_ids: tuple[str, ...] = ()
    nuisance_direction_identities: tuple[DirectionVectorIdentity, ...] = ()
    positive_control_source: str | None = None
    positive_direction_identity: DirectionVectorIdentity | None = None
    locked: bool = True
    schema_version: str = CAUSAL_DESIGN_SCHEMA_VERSION
    design_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != CAUSAL_DESIGN_SCHEMA_VERSION:
            raise ValueError("unsupported causal design schema_version")
        if (
            not isinstance(self.schema_registry_checksum, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.schema_registry_checksum)
        ):
            raise ValueError("causal design requires a schema-registry checksum")
        if self.schema_registry_checksum != _current_schema_registry_checksum():
            raise ValueError("causal design schema-registry checksum mismatch")
        if not isinstance(self.locked, bool):
            raise TypeError("locked must be a boolean")
        if not self.locked:
            raise ValueError("confirmatory manifests must be locked before execution")
        try:
            intervention = InterventionKind(self.intervention)
            controls = tuple(ControlKind(value) for value in self.controls)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid causal intervention or control") from exc
        object.__setattr__(self, "intervention", intervention)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(
            self,
            "primary_direction_identity",
            _identity(self.primary_direction_identity),
        )
        object.__setattr__(
            self,
            "nuisance_direction_identities",
            tuple(_identity(value) for value in self.nuisance_direction_identities),
        )
        if self.positive_direction_identity is not None:
            object.__setattr__(
                self,
                "positive_direction_identity",
                _identity(self.positive_direction_identity),
            )
        object.__setattr__(
            self, "coefficients", _finite_coefficients(self.coefficients)
        )
        object.__setattr__(
            self, "outcomes", _string_tuple(self.outcomes, name="outcomes")
        )
        object.__setattr__(
            self, "prompt_ids", _string_tuple(self.prompt_ids, name="prompt_ids")
        )
        object.__setattr__(
            self,
            "nuisance_direction_ids",
            _string_tuple(
                self.nuisance_direction_ids,
                name="nuisance_direction_ids",
            ),
        )
        object.__setattr__(
            self, "repetitions", _positive_int(self.repetitions, name="repetitions")
        )
        object.__setattr__(
            self, "n_bootstrap", _positive_int(self.n_bootstrap, name="n_bootstrap")
        )
        object.__setattr__(
            self,
            "n_permutations",
            _positive_int(self.n_permutations, name="n_permutations"),
        )
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, Integral)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer")
        object.__setattr__(self, "random_seed", int(self.random_seed))
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, Integral)
            or self.layer < 0
        ):
            raise ValueError("layer must be a non-negative integer")
        object.__setattr__(self, "layer", int(self.layer))
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, Real)
            or not math.isfinite(float(self.alpha))
            or not 0.0 < float(self.alpha) < 1.0
        ):
            raise ValueError("alpha must be finite and between 0 and 1")
        object.__setattr__(self, "alpha", float(self.alpha))

        for name in (
            "study_id",
            "dataset_hash",
            "intervention_adapter_version",
            "direction_source",
            "token_stage",
            "independent_unit",
            "group_key",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
        if not _SHA256_PATTERN.fullmatch(self.dataset_hash):
            raise ValueError("dataset_hash must be a canonical SHA256 identity")
        if self.independent_unit.strip().lower() in _ROW_LEVEL_UNITS:
            raise ValueError(
                "causal independent_unit must be a family, dyad, trial, or "
                "other clustered unit, never an individual row"
            )
        if len(set(self.controls)) != len(self.controls):
            raise ValueError("controls must not contain duplicates")
        if self.primary_direction_identity.source != self.direction_source:
            raise ValueError("primary direction identity source does not match")

        nuisance_identities = self.nuisance_direction_identities
        nuisance_sources = tuple(item.source for item in nuisance_identities)
        if ControlKind.NUISANCE_DIRECTION in self.controls:
            if not self.nuisance_direction_ids:
                raise ValueError(
                    "nuisance_direction control requires declared direction IDs"
                )
            if nuisance_sources != self.nuisance_direction_ids:
                raise ValueError(
                    "nuisance direction identities must match declared direction IDs"
                )
        elif self.nuisance_direction_ids or nuisance_identities:
            raise ValueError(
                "nuisance direction identities require the nuisance_direction control"
            )

        if ControlKind.POSITIVE_HOOK in self.controls:
            if not self.positive_control_source:
                raise ValueError(
                    "positive_hook control requires positive_control_source"
                )
            if self.positive_direction_identity is None:
                raise ValueError(
                    "positive_hook control requires a positive direction identity"
                )
            if (
                self.positive_direction_identity.source
                != self.positive_control_source
            ):
                raise ValueError("positive direction identity source does not match")
        elif (
            self.positive_control_source is not None
            or self.positive_direction_identity is not None
        ):
            raise ValueError(
                "positive direction identity requires the positive_hook control"
            )

        dimensions = {
            self.primary_direction_identity.dimension,
            *(item.dimension for item in nuisance_identities),
        }
        if self.positive_direction_identity is not None:
            dimensions.add(self.positive_direction_identity.dimension)
        if len(dimensions) != 1:
            raise ValueError("all locked causal directions must have one dimension")

        behavioral = any(
            outcome
            in {
                "behavior",
                "behavioral",
                "deception_behavior",
                "counterpart",
                "counterpart_outcome",
            }
            for outcome in self.outcomes
        )
        if behavioral:
            required = {
                ControlKind.ZERO_HOOK,
                ControlKind.NORM_MATCHED_RANDOM,
                ControlKind.LABEL_SHUFFLED,
            }
            missing = required.difference(self.controls)
            if missing:
                raise ValueError(
                    "behavioral designs require controls: "
                    + ", ".join(sorted(item.value for item in missing))
                )
            if not self.scorer_version:
                raise ValueError("behavioral designs require scorer_version")

        digest = hashlib.sha256(
            _canonical_json(self.to_dict(include_id=False)).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "design_id", f"causal_design_{digest[:24]}")

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "schema_registry_checksum": self.schema_registry_checksum,
            "study_id": self.study_id,
            "dataset_hash": self.dataset_hash,
            "intervention": self.intervention.value,
            "intervention_adapter_version": self.intervention_adapter_version,
            "direction_source": self.direction_source,
            "primary_direction_identity": self.primary_direction_identity.to_dict(),
            "layer": self.layer,
            "token_stage": self.token_stage,
            "coefficients": list(self.coefficients),
            "outcomes": list(self.outcomes),
            "independent_unit": self.independent_unit,
            "group_key": self.group_key,
            "controls": [item.value for item in self.controls],
            "repetitions": self.repetitions,
            "random_seed": self.random_seed,
            "n_bootstrap": self.n_bootstrap,
            "n_permutations": self.n_permutations,
            "alpha": self.alpha,
            "scorer_version": self.scorer_version,
            "prompt_ids": list(self.prompt_ids),
            "nuisance_direction_ids": list(self.nuisance_direction_ids),
            "nuisance_direction_identities": [
                item.to_dict() for item in self.nuisance_direction_identities
            ],
            "positive_control_source": self.positive_control_source,
            "positive_direction_identity": (
                None
                if self.positive_direction_identity is None
                else self.positive_direction_identity.to_dict()
            ),
            "locked": self.locked,
        }
        if include_id:
            result["design_id"] = self.design_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CausalDesignManifest":
        if value.get("schema_version") != CAUSAL_DESIGN_SCHEMA_VERSION:
            raise ValueError("unsupported causal design schema_version")
        if "schema_registry_checksum" not in value:
            raise ValueError("causal design requires a schema-registry checksum")
        manifest = cls(
            schema_version=value["schema_version"],
            schema_registry_checksum=value["schema_registry_checksum"],
            study_id=value["study_id"],
            dataset_hash=value["dataset_hash"],
            intervention=value["intervention"],
            intervention_adapter_version=value["intervention_adapter_version"],
            direction_source=value["direction_source"],
            primary_direction_identity=value["primary_direction_identity"],
            layer=value["layer"],
            token_stage=value["token_stage"],
            coefficients=tuple(value["coefficients"]),
            outcomes=tuple(value["outcomes"]),
            independent_unit=value["independent_unit"],
            group_key=value["group_key"],
            controls=tuple(value["controls"]),
            repetitions=value["repetitions"],
            random_seed=value["random_seed"],
            n_bootstrap=value["n_bootstrap"],
            n_permutations=value["n_permutations"],
            alpha=value["alpha"],
            scorer_version=value.get("scorer_version"),
            prompt_ids=tuple(value.get("prompt_ids", ())),
            nuisance_direction_ids=tuple(
                value.get("nuisance_direction_ids", ())
            ),
            nuisance_direction_identities=tuple(
                value.get("nuisance_direction_identities", ())
            ),
            positive_control_source=value.get("positive_control_source"),
            positive_direction_identity=value.get("positive_direction_identity"),
            locked=value.get("locked"),
        )
        serialized_id = value.get("design_id")
        if not isinstance(serialized_id, str) or not serialized_id:
            raise ValueError("current causal designs require design_id")
        if serialized_id != manifest.design_id:
            raise ValueError("serialized design_id does not match manifest content")
        return manifest


__all__ = [
    "CAUSAL_DESIGN_SCHEMA_VERSION",
    "CausalDesignManifest",
    "ControlKind",
    "DirectionVectorIdentity",
    "InterventionKind",
    "canonical_vector_sha256",
]
