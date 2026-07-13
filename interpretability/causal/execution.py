"""Execution and reporting for controls declared by a causal design manifest."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from interpretability.causal.design import (
    CausalDesignManifest,
    ControlKind,
    DirectionVectorIdentity,
    canonical_vector_sha256,
)
from interpretability.causal.statistics import paired_clustered_estimate


MeasurementFunction = Callable[..., Mapping[str, Any]]


_ROW_LEVEL_UNITS = {"row", "rows", "sample", "samples", "turn", "turns"}
_BEHAVIOR_OUTCOMES = {"behavior", "behavioral", "deception_behavior"}
CAUSAL_APPLICATION_RECEIPT_SCHEMA_VERSION = "1.0.0"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _readonly_array(value: Any, *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float).copy()
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected a {ndim}-dimensional array, got {array.ndim}")
    if not np.isfinite(array).all():
        raise ValueError("causal vectors and activations must contain finite values")
    array.setflags(write=False)
    return array


def _freeze_direction_mapping(
    values: Mapping[str, Any] | None,
) -> Mapping[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for name, value in (values or {}).items():
        if not name:
            raise ValueError("Nuisance direction IDs cannot be empty")
        frozen[str(name)] = _readonly_array(value, ndim=1)
    return MappingProxyType(frozen)


def _freeze_provenance(
    values: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Mapping[str, Any]]:
    frozen = {
        str(outcome): MappingProxyType(dict(provenance))
        for outcome, provenance in (values or {}).items()
    }
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class InterventionApplicationReceipt:
    """Tamper-evident adapter attestation for one measured intervention row."""

    design_id: str
    intervention_adapter_version: str
    applied: bool
    direction_id: str | None
    coefficient: float
    layer: int
    token_stage: str
    prompt_id: str
    repeat: int
    seed: int
    condition_id: str
    hook_mode: str
    schema_version: str = CAUSAL_APPLICATION_RECEIPT_SCHEMA_VERSION
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != CAUSAL_APPLICATION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("unsupported causal application receipt schema_version")
        for name in (
            "design_id",
            "intervention_adapter_version",
            "token_stage",
            "prompt_id",
            "condition_id",
            "hook_mode",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"application receipt {name} must be non-empty")
        if not isinstance(self.applied, bool):
            raise TypeError("application receipt applied must be a boolean")
        if self.direction_id is not None and (
            not isinstance(self.direction_id, str) or not self.direction_id
        ):
            raise ValueError("application receipt direction_id must be named")
        if (
            isinstance(self.coefficient, bool)
            or not isinstance(self.coefficient, Real)
            or not math.isfinite(float(self.coefficient))
        ):
            raise ValueError("application receipt coefficient must be finite")
        object.__setattr__(self, "coefficient", float(self.coefficient))
        for name, minimum in (("layer", 0), ("repeat", 0), ("seed", 0)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(
                    f"application receipt {name} must be a non-negative integer"
                )
            object.__setattr__(self, name, int(value))
        digest = hashlib.sha256(
            _canonical_json(self.to_dict(include_id=False)).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "receipt_id", f"causal_application_{digest[:24]}")

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "design_id": self.design_id,
            "intervention_adapter_version": self.intervention_adapter_version,
            "applied": self.applied,
            "direction_id": self.direction_id,
            "coefficient": self.coefficient,
            "layer": self.layer,
            "token_stage": self.token_stage,
            "prompt_id": self.prompt_id,
            "repeat": self.repeat,
            "seed": self.seed,
            "condition_id": self.condition_id,
            "hook_mode": self.hook_mode,
        }
        if include_id:
            result["receipt_id"] = self.receipt_id
        return result

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "InterventionApplicationReceipt":
        if value.get("schema_version") != CAUSAL_APPLICATION_RECEIPT_SCHEMA_VERSION:
            raise ValueError("unsupported causal application receipt schema_version")
        receipt = cls(
            schema_version=value["schema_version"],
            design_id=value["design_id"],
            intervention_adapter_version=value["intervention_adapter_version"],
            applied=value["applied"],
            direction_id=value.get("direction_id"),
            coefficient=value["coefficient"],
            layer=value["layer"],
            token_stage=value["token_stage"],
            prompt_id=value["prompt_id"],
            repeat=value["repeat"],
            seed=value["seed"],
            condition_id=value["condition_id"],
            hook_mode=value["hook_mode"],
        )
        if value.get("receipt_id") != receipt.receipt_id:
            raise ValueError("application receipt ID does not match its content")
        return receipt


@dataclass(frozen=True)
class CausalExecutionInputs:
    """Runtime inputs whose identities must match one locked manifest."""

    design_id: str | None = None
    dataset_hash: str | None = None
    intervention: str | None = None
    intervention_adapter_version: str | None = None
    direction_source: str | None = None
    layer: int | None = None
    token_stage: str | None = None
    coefficients: tuple[float, ...] | None = None
    repetitions: int | None = None
    random_seed: int | None = None
    independent_unit: str | None = None
    group_key: str | None = None
    prompts: tuple[str, ...] = ()
    prompt_ids: tuple[str, ...] = ()
    group_ids: tuple[str, ...] = ()
    primary_direction: np.ndarray | None = field(default=None, repr=False, compare=False)
    nuisance_directions: Mapping[str, np.ndarray] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    positive_direction: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    positive_direction_source: str | None = None
    direction_activations: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    direction_labels: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    direction_group_ids: tuple[str, ...] = ()
    outcome_provenance: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict,
        compare=False,
    )
    measurement_fn: MeasurementFunction | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.intervention is not None and hasattr(self.intervention, "value"):
            object.__setattr__(self, "intervention", self.intervention.value)
        object.__setattr__(self, "prompts", tuple(self.prompts))
        object.__setattr__(self, "prompt_ids", tuple(self.prompt_ids))
        object.__setattr__(self, "group_ids", tuple(self.group_ids))
        object.__setattr__(
            self,
            "direction_group_ids",
            tuple(self.direction_group_ids),
        )
        if self.coefficients is not None:
            raw_coefficients = tuple(self.coefficients)
            if any(
                isinstance(value, bool) or not isinstance(value, Real)
                for value in raw_coefficients
            ):
                raise ValueError("execution coefficients must be numeric")
            coefficients = tuple(float(value) for value in raw_coefficients)
            if not all(math.isfinite(value) for value in coefficients):
                raise ValueError("execution coefficients must be finite")
            if len(set(coefficients)) != len(coefficients):
                raise ValueError("execution coefficients must not contain duplicates")
            object.__setattr__(
                self,
                "coefficients",
                coefficients,
            )
        for name in ("primary_direction", "positive_direction"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _readonly_array(value, ndim=1))
        if self.direction_activations is not None:
            object.__setattr__(
                self,
                "direction_activations",
                _readonly_array(self.direction_activations, ndim=2),
            )
        if self.direction_labels is not None:
            object.__setattr__(
                self,
                "direction_labels",
                _readonly_array(self.direction_labels, ndim=1),
            )
            if not np.isin(self.direction_labels, (0.0, 1.0)).all():
                raise ValueError("direction_labels must be binary 0/1 values")
        object.__setattr__(
            self,
            "nuisance_directions",
            _freeze_direction_mapping(self.nuisance_directions),
        )
        object.__setattr__(
            self,
            "outcome_provenance",
            _freeze_provenance(self.outcome_provenance),
        )
        if self.random_seed is not None and (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, Integral)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a non-negative integer")
        if self.random_seed is not None:
            object.__setattr__(self, "random_seed", int(self.random_seed))
        for name, values in (
            ("prompts", self.prompts),
            ("prompt_ids", self.prompt_ids),
            ("group_ids", self.group_ids),
            ("direction_group_ids", self.direction_group_ids),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
        if len(set(self.prompt_ids)) != len(self.prompt_ids):
            raise ValueError("prompt_ids must be unique")


@dataclass(frozen=True)
class _Condition:
    condition_id: str
    control_kind: ControlKind | None
    source: str
    vectors_by_repeat: tuple[np.ndarray | None, ...] = ()
    hook_mode: str = "direction"
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None


def _seed(*parts: int) -> int:
    sequence = np.random.SeedSequence(parts)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _vector_id(source: str, vector: np.ndarray) -> str:
    digest, dimension = canonical_vector_sha256(vector)
    return f"{source}:{digest}:d{dimension}"


def _norm_matched_random_directions(
    primary: np.ndarray,
    *,
    repetitions: int,
    random_seed: int,
) -> tuple[np.ndarray, ...]:
    target_norm = float(np.linalg.norm(primary))
    if target_norm <= 0:
        raise ValueError("Primary direction must have non-zero norm")
    directions = []
    for repeat in range(repetitions):
        rng = np.random.default_rng(_seed(random_seed, repeat, 17))
        vector = rng.standard_normal(len(primary))
        norm = float(np.linalg.norm(vector))
        if norm <= 0:  # pragma: no cover - RNG defensive branch
            raise ValueError("Random control direction has zero norm")
        vector = vector * (target_norm / norm)
        vector.setflags(write=False)
        directions.append(vector)
    return tuple(directions)


def _group_block_permutation(
    labels: np.ndarray,
    groups: Sequence[str],
    *,
    random_seed: int,
) -> np.ndarray:
    group_array = np.asarray(groups)
    if len(labels) != len(group_array):
        raise ValueError("Direction labels and direction groups must align")
    unique_groups = list(dict.fromkeys(group_array.tolist()))
    indices = {
        group: np.flatnonzero(group_array == group)
        for group in unique_groups
    }
    by_size: dict[int, list[str]] = {}
    for group, rows in indices.items():
        by_size.setdefault(len(rows), []).append(group)
    if any(len(group_names) < 2 for group_names in by_size.values()):
        raise ValueError(
            "Label-shuffled control requires at least two exchangeable groups "
            "for every group size"
        )

    shuffled = np.empty_like(labels)
    rng = np.random.default_rng(random_seed)
    for group_names in by_size.values():
        offset = int(rng.integers(1, len(group_names)))
        sources = group_names[offset:] + group_names[:offset]
        for target, source in zip(group_names, sources):
            shuffled[indices[target]] = labels[indices[source]]
    return shuffled


def _direction_from_labels(
    activations: np.ndarray,
    labels: np.ndarray,
    *,
    target_norm: float,
) -> np.ndarray:
    positive = labels > 0.5
    if not positive.any() or positive.all():
        raise ValueError("Label-shuffled direction requires both label classes")
    direction = activations[positive].mean(axis=0) - activations[~positive].mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("Label-shuffled direction has zero norm")
    direction = direction * (target_norm / norm)
    direction.setflags(write=False)
    return direction


def _label_shuffled_directions(
    inputs: CausalExecutionInputs,
    manifest: CausalDesignManifest,
) -> tuple[np.ndarray, ...]:
    if inputs.primary_direction is None:
        raise ValueError("Primary direction is required")
    if inputs.direction_activations is None or inputs.direction_labels is None:
        raise ValueError("Direction activations and labels are required")
    if not inputs.direction_group_ids:
        raise ValueError("Direction group IDs are required for block shuffling")
    if len(inputs.direction_activations) != len(inputs.direction_labels):
        raise ValueError("Direction activations and labels must align")
    target_norm = float(np.linalg.norm(inputs.primary_direction))
    directions = []
    for repeat in range(manifest.repetitions):
        shuffled = _group_block_permutation(
            inputs.direction_labels,
            inputs.direction_group_ids,
            random_seed=_seed(manifest.random_seed, repeat, 29),
        )
        directions.append(
            _direction_from_labels(
                inputs.direction_activations,
                shuffled,
                target_norm=target_norm,
            )
        )
    return tuple(directions)


def _unavailable_condition(
    condition_id: str,
    control_kind: ControlKind,
    reason: str,
) -> _Condition:
    return _Condition(
        condition_id=condition_id,
        control_kind=control_kind,
        source=control_kind.value,
        unavailable_reason=reason,
    )


def _build_conditions(
    manifest: CausalDesignManifest,
    inputs: CausalExecutionInputs,
) -> list[_Condition]:
    primary = inputs.primary_direction
    conditions = [
        _Condition(
            condition_id="baseline_no_hook",
            control_kind=None,
            source="baseline",
            vectors_by_repeat=(None,) * manifest.repetitions,
            hook_mode="none",
        )
    ]
    if primary is None:
        conditions.append(
            _Condition(
                condition_id="target",
                control_kind=None,
                source=manifest.direction_source,
                unavailable_reason="primary_direction_missing",
            )
        )
    else:
        conditions.append(
            _Condition(
                condition_id="target",
                control_kind=None,
                source=manifest.direction_source,
                vectors_by_repeat=(primary,) * manifest.repetitions,
            )
        )

    for control in manifest.controls:
        if control is ControlKind.ZERO_HOOK:
            if primary is None:
                conditions.append(
                    _unavailable_condition(
                        control.value,
                        control,
                        "primary_direction_missing_for_dimension",
                    )
                )
            else:
                zero = np.zeros_like(primary)
                zero.setflags(write=False)
                conditions.append(
                    _Condition(
                        condition_id=control.value,
                        control_kind=control,
                        source="zero_vector",
                        vectors_by_repeat=(zero,) * manifest.repetitions,
                        hook_mode="zero",
                    )
                )
        elif control is ControlKind.SHAM_HOOK:
            conditions.append(
                _Condition(
                    condition_id=control.value,
                    control_kind=control,
                    source="identity_hook",
                    vectors_by_repeat=(None,) * manifest.repetitions,
                    hook_mode="sham",
                )
            )
        elif control is ControlKind.NORM_MATCHED_RANDOM:
            try:
                if primary is None:
                    raise ValueError("Primary direction is required")
                vectors = _norm_matched_random_directions(
                    primary,
                    repetitions=manifest.repetitions,
                    random_seed=manifest.random_seed,
                )
                conditions.append(
                    _Condition(
                        condition_id=control.value,
                        control_kind=control,
                        source="seeded_norm_matched_random",
                        vectors_by_repeat=vectors,
                    )
                )
            except ValueError as exc:
                conditions.append(
                    _unavailable_condition(control.value, control, str(exc))
                )
        elif control is ControlKind.LABEL_SHUFFLED:
            try:
                vectors = _label_shuffled_directions(inputs, manifest)
                conditions.append(
                    _Condition(
                        condition_id=control.value,
                        control_kind=control,
                        source="group_block_label_shuffle",
                        vectors_by_repeat=vectors,
                    )
                )
            except ValueError as exc:
                conditions.append(
                    _unavailable_condition(control.value, control, str(exc))
                )
        elif control is ControlKind.NUISANCE_DIRECTION:
            undeclared = set(inputs.nuisance_directions).difference(
                manifest.nuisance_direction_ids
            )
            if undeclared:
                raise ValueError(
                    "nuisance direction IDs do not match the locked causal design"
                )
            for name in manifest.nuisance_direction_ids:
                vector = inputs.nuisance_directions.get(name)
                if vector is None:
                    conditions.append(
                        _unavailable_condition(
                            f"{control.value}:{name}",
                            control,
                            f"declared nuisance direction was not supplied: {name}",
                        )
                    )
                else:
                    conditions.append(
                        _Condition(
                            condition_id=f"{control.value}:{name}",
                            control_kind=control,
                            source=name,
                            vectors_by_repeat=(vector,) * manifest.repetitions,
                        )
                    )
        elif control is ControlKind.POSITIVE_HOOK:
            if inputs.positive_direction is None:
                conditions.append(
                    _unavailable_condition(
                        control.value,
                        control,
                        "positive hook direction was not supplied",
                    )
                )
            elif not inputs.positive_direction_source:
                conditions.append(
                    _unavailable_condition(
                        control.value,
                        control,
                        "positive hook source identity was not supplied",
                    )
                )
            elif (
                inputs.positive_direction_source
                != manifest.positive_control_source
            ):
                raise ValueError(
                    "positive hook source does not match the locked causal design"
                )
            else:
                conditions.append(
                    _Condition(
                        condition_id=control.value,
                        control_kind=control,
                        source=inputs.positive_direction_source,
                        vectors_by_repeat=(inputs.positive_direction,)
                        * manifest.repetitions,
                    )
                )
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError(f"Unsupported declared causal control: {control.value}")
    return conditions


def _identity_missing(
    manifest: CausalDesignManifest,
    inputs: CausalExecutionInputs,
) -> list[str]:
    missing = []
    for field_name in (
        "design_id",
        "dataset_hash",
        "intervention",
        "intervention_adapter_version",
        "direction_source",
        "layer",
        "token_stage",
        "coefficients",
        "repetitions",
        "random_seed",
        "independent_unit",
        "group_key",
    ):
        if getattr(inputs, field_name) is None:
            missing.append(field_name)
    if not manifest.prompt_ids:
        missing.append("manifest.prompt_ids")
    if not inputs.prompts:
        missing.append("prompts")
    if not inputs.prompt_ids:
        missing.append("prompt_ids")
    if not inputs.group_ids:
        missing.append(manifest.group_key)
    if inputs.primary_direction is None:
        missing.append("primary_direction")
    return missing


def _validate_identity(
    manifest: CausalDesignManifest,
    inputs: CausalExecutionInputs,
) -> None:
    comparisons = {
        "design_id": (inputs.design_id, manifest.design_id),
        "dataset_hash": (inputs.dataset_hash, manifest.dataset_hash),
        "intervention": (inputs.intervention, manifest.intervention.value),
        "intervention_adapter_version": (
            inputs.intervention_adapter_version,
            manifest.intervention_adapter_version,
        ),
        "direction_source": (inputs.direction_source, manifest.direction_source),
        "layer": (inputs.layer, manifest.layer),
        "token_stage": (inputs.token_stage, manifest.token_stage),
        "coefficients": (inputs.coefficients, manifest.coefficients),
        "repetitions": (inputs.repetitions, manifest.repetitions),
        "random_seed": (inputs.random_seed, manifest.random_seed),
        "independent_unit": (
            inputs.independent_unit,
            manifest.independent_unit,
        ),
        "group_key": (inputs.group_key, manifest.group_key),
    }
    for field_name, (actual, expected) in comparisons.items():
        if actual is not None and actual != expected:
            raise ValueError(f"{field_name} does not match the locked causal design")
    if inputs.prompt_ids and manifest.prompt_ids:
        if inputs.prompt_ids != manifest.prompt_ids:
            raise ValueError("prompt_ids do not match the locked causal design")
    if manifest.independent_unit.strip().lower() in _ROW_LEVEL_UNITS:
        raise ValueError("row-level independent units are invalid for causal inference")
    if inputs.primary_direction is not None:
        primary_identity = DirectionVectorIdentity.from_vector(
            manifest.direction_source,
            inputs.primary_direction,
        )
        if primary_identity != manifest.primary_direction_identity:
            raise ValueError(
                "primary direction does not match the locked causal design"
            )
    locked_nuisance = {
        identity.source: identity
        for identity in manifest.nuisance_direction_identities
    }
    for source, vector in inputs.nuisance_directions.items():
        expected = locked_nuisance.get(source)
        if expected is None:
            raise ValueError(
                "nuisance direction IDs do not match the locked causal design"
            )
        if DirectionVectorIdentity.from_vector(source, vector) != expected:
            raise ValueError(
                f"nuisance direction {source} does not match the locked causal design"
            )
    if inputs.positive_direction is not None:
        if not inputs.positive_direction_source:
            raise ValueError("positive hook source identity was not supplied")
        supplied_positive = DirectionVectorIdentity.from_vector(
            inputs.positive_direction_source,
            inputs.positive_direction,
        )
        if supplied_positive != manifest.positive_direction_identity:
            raise ValueError(
                "positive direction does not match the locked causal design"
            )
    if inputs.direction_activations is not None and (
        inputs.direction_activations.shape[1]
        != manifest.primary_direction_identity.dimension
    ):
        raise ValueError(
            "direction activations do not match the locked direction dimension"
        )


def _outcome_provenance(
    manifest: CausalDesignManifest,
    inputs: CausalExecutionInputs,
    outcome: str,
) -> dict[str, Any]:
    supplied = dict(inputs.outcome_provenance.get(outcome, {}))
    if outcome in _BEHAVIOR_OUTCOMES:
        supplied_version = supplied.get("scorer_version")
        if supplied_version is not None and supplied_version != manifest.scorer_version:
            raise ValueError(
                f"outcome scorer version for {outcome} does not match the manifest"
            )
        supplied.setdefault("scorer_version", manifest.scorer_version)
    supplied.setdefault("source", "measurement_function")
    return supplied


def _empty_outcomes(
    manifest: CausalDesignManifest,
    inputs: CausalExecutionInputs,
    reason: str,
) -> dict[str, Any]:
    return {
        outcome: {
            "available": False,
            "reason": reason,
            "scorer_provenance": _outcome_provenance(manifest, inputs, outcome),
            "by_coefficient": {},
        }
        for outcome in manifest.outcomes
    }


def _unavailable_report(
    manifest: CausalDesignManifest,
    inputs: CausalExecutionInputs,
    conditions: Sequence[_Condition],
    reason: str,
) -> dict[str, Any]:
    expected_rows = (
        len(inputs.prompts) * manifest.repetitions * len(manifest.coefficients)
    )
    condition_reports = {}
    for condition in conditions:
        condition_reason = condition.unavailable_reason or reason
        condition_reports[condition.condition_id] = {
            "available": False,
            "reason": condition_reason,
            "control_kind": (
                condition.control_kind.value if condition.control_kind else None
            ),
            "source": condition.source,
            "expected_rows": expected_rows,
            "rows": [],
            "application_receipts_verified": False,
            "outcomes": _empty_outcomes(
                manifest,
                inputs,
                condition_reason,
            ),
        }
    return {
        "status": "unavailable",
        "available": False,
        "reason": reason,
        "design_id": manifest.design_id,
        "declared_controls": [control.value for control in manifest.controls],
        "conditions": condition_reports,
        "control_status": _control_status(manifest, condition_reports),
        "application_receipts_verified": False,
        "causal_claim_ready": False,
        "causal_evidence_strength": None,
        "interpretation_limits": _interpretation_limits(),
    }


def unavailable_control_report(
    manifest: CausalDesignManifest,
    reason: str,
) -> dict[str, Any]:
    """Return a complete unavailable entry for every declared control."""
    inputs = CausalExecutionInputs()
    return _unavailable_report(
        manifest,
        inputs,
        _build_conditions(manifest, inputs),
        reason,
    )


def _control_status(
    manifest: CausalDesignManifest,
    condition_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    statuses = {}
    for control in manifest.controls:
        matching = [
            (condition_id, report)
            for condition_id, report in condition_reports.items()
            if report.get("control_kind") == control.value
        ]
        statuses[control.value] = {
            "available": bool(matching) and all(
                bool(report.get("available")) for _, report in matching
            ),
            "condition_ids": [condition_id for condition_id, _ in matching],
            "reasons": [
                report.get("reason")
                for _, report in matching
                if report.get("reason")
            ],
        }
    return statuses


def _interpretation_limits() -> list[str]:
    return [
        "Each outcome is scorer- and intervention-site-specific.",
        "Logit sensitivity is not behavioral control or deceptive intent.",
        "Behavioral scores depend on the declared scorer and prompt population.",
        "Fluency changes can indicate a nonspecific intervention side effect.",
        "Utility and counterpart outcomes are downstream game-specific effects.",
        "Repeated prompt rows are paired and clustered; they are not independent units.",
        "No aggregate evidence adjective or causal-claim readiness is calibrated.",
    ]


def _run_condition(
    condition: _Condition,
    manifest: CausalDesignManifest,
    inputs: CausalExecutionInputs,
    measurement_fn: MeasurementFunction,
    model: Any,
) -> list[dict[str, Any]]:
    rows = []
    for coefficient_index, coefficient in enumerate(manifest.coefficients):
        for repeat in range(manifest.repetitions):
            vector = condition.vectors_by_repeat[repeat]
            vector_identity = (
                None if vector is None else _vector_id(condition.source, vector)
            )
            for prompt_index, (prompt, prompt_id, group_id) in enumerate(
                zip(inputs.prompts, inputs.prompt_ids, inputs.group_ids)
            ):
                pair_id = f"{prompt_id}|r{repeat}|c{coefficient_index}"
                row_seed = _seed(
                    manifest.random_seed,
                    repeat,
                    coefficient_index,
                    prompt_index,
                )
                error = None
                measured: Mapping[str, Any] = {}
                application_receipt = None
                expected_receipt = InterventionApplicationReceipt(
                    design_id=manifest.design_id,
                    intervention_adapter_version=(
                        manifest.intervention_adapter_version
                    ),
                    applied=condition.hook_mode != "none",
                    direction_id=vector_identity,
                    coefficient=float(coefficient),
                    layer=manifest.layer,
                    token_stage=manifest.token_stage,
                    prompt_id=prompt_id,
                    repeat=repeat,
                    seed=row_seed,
                    condition_id=condition.condition_id,
                    hook_mode=condition.hook_mode,
                )
                outcome_values: dict[str, float | None] = {
                    outcome: None for outcome in manifest.outcomes
                }
                try:
                    measured = measurement_fn(
                        model=model,
                        design_id=manifest.design_id,
                        intervention=manifest.intervention.value,
                        intervention_adapter_version=(
                            manifest.intervention_adapter_version
                        ),
                        prompt=prompt,
                        prompt_id=prompt_id,
                        prompt_index=prompt_index,
                        group_id=group_id,
                        repeat=repeat,
                        seed=row_seed,
                        condition_id=condition.condition_id,
                        control_kind=(
                            condition.control_kind.value
                            if condition.control_kind else None
                        ),
                        hook_mode=condition.hook_mode,
                        direction=vector,
                        direction_id=vector_identity,
                        coefficient=float(coefficient),
                        layer=manifest.layer,
                        token_stage=manifest.token_stage,
                        outcomes=manifest.outcomes,
                        scorer_provenance={
                            outcome: _outcome_provenance(manifest, inputs, outcome)
                            for outcome in manifest.outcomes
                        },
                    )
                    if not isinstance(measured, Mapping):
                        raise TypeError("measurement_fn must return a mapping")
                    receipt_payload = measured.get("application_receipt")
                    if not isinstance(receipt_payload, Mapping):
                        raise ValueError(
                            "measurement_fn must return an application_receipt"
                        )
                    receipt = InterventionApplicationReceipt.from_dict(
                        receipt_payload
                    )
                    if receipt != expected_receipt:
                        raise ValueError(
                            "application receipt does not match the requested "
                            "intervention"
                        )
                    application_receipt = receipt.to_dict()
                    for outcome in manifest.outcomes:
                        raw_value = measured.get(outcome)
                        if raw_value is None:
                            continue
                        if isinstance(raw_value, bool):
                            value = float(raw_value)
                        elif isinstance(raw_value, Real):
                            value = float(raw_value)
                        else:
                            raise ValueError(
                                f"measurement outcome {outcome} must be numeric"
                            )
                        if not math.isfinite(value):
                            raise ValueError(
                                f"measurement outcome {outcome} must be finite"
                            )
                        outcome_values[outcome] = value
                except Exception as exc:  # retain every failed paired row
                    error = f"{type(exc).__name__}: {exc}"
                    outcome_values = {
                        outcome: None for outcome in manifest.outcomes
                    }
                rows.append(
                    {
                        "pair_id": pair_id,
                        "prompt_id": prompt_id,
                        "prompt_index": prompt_index,
                        "group_id": group_id,
                        "repeat": repeat,
                        "coefficient": float(coefficient),
                        "seed": row_seed,
                        "direction_id": vector_identity,
                        "application_receipt": application_receipt,
                        "outcomes": outcome_values,
                        "measurement_error": error,
                    }
                )
    return rows


def _paired_outcomes(
    manifest: CausalDesignManifest,
    inputs: CausalExecutionInputs,
    baseline_rows: Sequence[Mapping[str, Any]],
    condition_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline_by_pair = {row["pair_id"]: row for row in baseline_rows}
    condition_by_pair = {row["pair_id"]: row for row in condition_rows}
    outcomes: dict[str, Any] = {}
    for outcome_index, outcome in enumerate(manifest.outcomes):
        coefficient_reports = {}
        for coefficient_index, coefficient in enumerate(manifest.coefficients):
            paired_rows = []
            baseline_values = []
            intervention_values = []
            clusters = []
            for pair_id, baseline in baseline_by_pair.items():
                intervention = condition_by_pair.get(pair_id)
                if intervention is None or baseline["coefficient"] != float(coefficient):
                    continue
                baseline_value = baseline["outcomes"].get(outcome)
                intervention_value = intervention["outcomes"].get(outcome)
                available = baseline_value is not None and intervention_value is not None
                paired_rows.append(
                    {
                        "pair_id": pair_id,
                        "prompt_id": baseline["prompt_id"],
                        "group_id": baseline["group_id"],
                        "repeat": baseline["repeat"],
                        "coefficient": float(coefficient),
                        "seed": baseline["seed"],
                        "baseline": baseline_value,
                        "intervention": intervention_value,
                        "difference": (
                            intervention_value - baseline_value if available else None
                        ),
                        "available": available,
                        "baseline_error": baseline["measurement_error"],
                        "intervention_error": intervention["measurement_error"],
                    }
                )
                if available:
                    baseline_values.append(baseline_value)
                    intervention_values.append(intervention_value)
                    clusters.append(baseline["group_id"])
            estimate = None
            reason = None
            if not paired_rows:
                reason = "no paired rows were produced"
            elif len(baseline_values) != len(paired_rows):
                reason = "one or more paired outcome values are unavailable"
            else:
                try:
                    estimate = paired_clustered_estimate(
                        baseline_values,
                        intervention_values,
                        clusters,
                        cluster_unit=manifest.independent_unit,
                        n_bootstrap=manifest.n_bootstrap,
                        n_permutations=manifest.n_permutations,
                        random_seed=_seed(
                            manifest.random_seed,
                            coefficient_index,
                            outcome_index,
                            43,
                        ),
                        alpha=manifest.alpha,
                    ).to_dict()
                except ValueError as exc:
                    reason = str(exc)
            coefficient_reports[str(float(coefficient))] = {
                "available": estimate is not None,
                "reason": reason,
                "estimate": estimate,
                "paired_rows": paired_rows,
            }
        outcomes[outcome] = {
            "available": all(
                report["available"] for report in coefficient_reports.values()
            ),
            "scorer_provenance": _outcome_provenance(manifest, inputs, outcome),
            "by_coefficient": coefficient_reports,
            "interpretation": _outcome_interpretation(outcome),
        }
    return outcomes


def _outcome_interpretation(outcome: str) -> str:
    if outcome == "logit":
        return "Paired change in the declared logit estimand; not behavior or intent."
    if outcome in _BEHAVIOR_OUTCOMES:
        return "Paired scorer-defined generated behavior; not latent intent."
    if outcome == "fluency":
        return "Paired fluency change used to detect nonspecific degradation."
    if outcome == "utility":
        return "Paired actor utility under the declared game and outcome scorer."
    if outcome in {"counterpart", "counterpart_outcome"}:
        return "Paired counterpart outcome under the declared interaction policy."
    return "Paired change in the manifest-declared measured outcome."


def execute_manifest_controls(
    manifest: CausalDesignManifest,
    inputs: CausalExecutionInputs,
    *,
    model: Any = None,
) -> dict[str, Any]:
    """Execute the target and every declared control on one paired grid."""
    _validate_identity(manifest, inputs)
    conditions = _build_conditions(manifest, inputs)
    missing = _identity_missing(manifest, inputs)
    if missing:
        return _unavailable_report(
            manifest,
            inputs,
            conditions,
            "missing required execution inputs: " + ", ".join(missing),
        )
    if not (
        len(inputs.prompts) == len(inputs.prompt_ids) == len(inputs.group_ids)
    ):
        raise ValueError("prompts, prompt_ids, and group_ids must align")
    if len(set(inputs.prompt_ids)) != len(inputs.prompt_ids):
        raise ValueError("prompt_ids must be unique")
    if len(set(inputs.group_ids)) < 3:
        return _unavailable_report(
            manifest,
            inputs,
            conditions,
            "at least three family/dyad clusters are required",
        )
    primary_dim = len(inputs.primary_direction)
    for condition_index, condition in enumerate(conditions):
        for vector in condition.vectors_by_repeat:
            if vector is not None and len(vector) != primary_dim:
                if condition.control_kind in {
                    ControlKind.NUISANCE_DIRECTION,
                    ControlKind.POSITIVE_HOOK,
                }:
                    conditions[condition_index] = _Condition(
                        condition_id=condition.condition_id,
                        control_kind=condition.control_kind,
                        source=condition.source,
                        unavailable_reason=(
                            "control direction dimension does not match primary direction"
                        ),
                    )
                    break
                raise ValueError("control direction dimension does not match primary direction")

    measurement_fn = inputs.measurement_fn
    if measurement_fn is None and model is not None:
        measurement_fn = getattr(model, "measure_causal_outcomes", None)
    if not callable(measurement_fn):
        return _unavailable_report(
            manifest,
            inputs,
            conditions,
            "model does not provide measure_causal_outcomes and no measurement_fn was supplied",
        )

    condition_reports: dict[str, Any] = {}
    baseline_rows: list[dict[str, Any]] = []
    expected_rows = len(inputs.prompts) * manifest.repetitions * len(
        manifest.coefficients
    )
    for condition in conditions:
        if condition.unavailable_reason is not None:
            condition_reports[condition.condition_id] = {
                "available": False,
                "reason": condition.unavailable_reason,
                "control_kind": (
                    condition.control_kind.value if condition.control_kind else None
                ),
                "source": condition.source,
                "expected_rows": expected_rows,
                "rows": [],
                "application_receipts_verified": False,
                "direction_by_repeat": [],
                "outcomes": _empty_outcomes(
                    manifest,
                    inputs,
                    condition.unavailable_reason,
                ),
            }
            continue
        rows = _run_condition(condition, manifest, inputs, measurement_fn, model)
        if condition.condition_id == "baseline_no_hook":
            baseline_rows = rows
        condition_reports[condition.condition_id] = {
            "available": True,
            "reason": None,
            "control_kind": (
                condition.control_kind.value if condition.control_kind else None
            ),
            "source": condition.source,
            "expected_rows": expected_rows,
            "rows": rows,
            "application_receipts_verified": all(
                row["application_receipt"] is not None for row in rows
            ),
            "direction_by_repeat": [
                {
                    "repeat": repeat,
                    "direction_id": (
                        None if vector is None else _vector_id(condition.source, vector)
                    ),
                    "norm": (
                        None if vector is None else float(np.linalg.norm(vector))
                    ),
                }
                for repeat, vector in enumerate(condition.vectors_by_repeat)
            ],
            "outcomes": {},
        }

    for report in condition_reports.values():
        if not report["available"]:
            continue
        report["outcomes"] = _paired_outcomes(
            manifest,
            inputs,
            baseline_rows,
            report["rows"],
        )

    control_status = _control_status(manifest, condition_reports)
    available_pair_sets = [
        {row["pair_id"] for row in report["rows"]}
        for report in condition_reports.values()
        if report["available"]
    ]
    paired_grid_verified = bool(available_pair_sets) and all(
        pair_set == available_pair_sets[0]
        for pair_set in available_pair_sets[1:]
    )
    if not paired_grid_verified:  # pragma: no cover - internal invariant
        raise RuntimeError("Causal conditions did not retain one shared paired grid")
    complete = all(status["available"] for status in control_status.values())
    measured = all(
        outcome_report["available"]
        for report in condition_reports.values()
        if report["available"]
        for outcome_report in report["outcomes"].values()
    )
    receipts_verified = all(
        report["application_receipts_verified"]
        for report in condition_reports.values()
        if report["available"]
    )
    return {
        "status": (
            "complete" if complete and measured and receipts_verified else "partial"
        ),
        "available": True,
        "reason": None,
        "design_id": manifest.design_id,
        "manifest_identity_verified": True,
        "random_seed": manifest.random_seed,
        "repetitions": manifest.repetitions,
        "coefficients": list(manifest.coefficients),
        "prompt_ids": list(inputs.prompt_ids),
        "group_key": manifest.group_key,
        "independent_unit": manifest.independent_unit,
        "paired_grid_verified": paired_grid_verified,
        "application_receipts_verified": receipts_verified,
        "declared_controls": [control.value for control in manifest.controls],
        "conditions": condition_reports,
        "control_status": control_status,
        "causal_claim_ready": False,
        "causal_evidence_strength": None,
        "interpretation_limits": _interpretation_limits(),
    }
