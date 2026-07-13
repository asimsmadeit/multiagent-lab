"""Immutable runtime contracts for probes and scripted observations.

These records keep non-negotiation model calls and externally supplied
observations out of the ordinary negotiation-turn namespace.  Every persisted
object is current-schema-only and content-addressed; callers that need to
resume execution can therefore distinguish work that is pending from work
that already has an exactly-once receipt without invoking a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping, Sequence, TypeAlias


PROBE_INTERVENTION_SCHEMA_VERSION = "1.0.0"
SCRIPTED_OBSERVATION_SCHEMA_VERSION = "1.0.0"
INTERVENTION_DESIGN_SCHEMA_VERSION = "1.0.0"
INTERVENTION_SCHEDULE_SCHEMA_VERSION = "1.0.0"
INTERVENTION_APPLICATION_SCHEMA_VERSION = "1.1.0"
INTERVENTION_APPLICATION_LOG_SCHEMA_VERSION = "1.1.0"

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class InterventionFamily(str, Enum):
    """Runtime intervention namespaces."""

    PROBE = "probe"
    SCRIPTED_OBSERVATION = "scripted_observation"


class ProbeKind(str, Enum):
    """Probe calls that are never ordinary negotiation samples."""

    BELIEF_VERIFICATION = "belief_verification"
    PLAUSIBILITY = "plausibility"


class ScriptedObservationKind(str, Enum):
    """Provenance class for an externally supplied observation."""

    REGISTERED_TEMPLATE = "registered_template"
    CUSTOM = "custom"


class InterventionApplicationStatus(str, Enum):
    """Terminal outcomes for one scheduled application."""

    APPLIED = "applied"
    SKIPPED_DISABLED = "skipped_disabled"
    SKIPPED_TERMINAL = "skipped_terminal"


class ProbeLabelStatus(str, Enum):
    """Non-boolean label states allowed for intervention rows."""

    INAPPLICABLE = "inapplicable"
    UNKNOWN = "unknown"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def content_sha256(content: str) -> str:
    """Return the canonical UTF-8 identity for intervention content."""
    if not isinstance(content, str):
        raise TypeError("intervention content must be a string")
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def _require_non_empty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_non_negative_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    record_name: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"serialized {record_name} must be a mapping")
    names = set(value)
    missing = expected.difference(names)
    unknown = names.difference(expected)
    if missing:
        raise ValueError(
            f"serialized {record_name} is missing fields: "
            + ", ".join(sorted(missing))
        )
    if unknown:
        raise ValueError(
            f"serialized {record_name} has unknown fields: "
            + ", ".join(sorted(unknown))
        )


def _require_serialized_enum(
    value: Any,
    enum_type: type[Enum],
    *,
    name: str,
) -> Enum:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"unsupported {name}: {value!r}") from exc


def _validate_common_plan(
    *,
    run_id: Any,
    trial_id: Any,
    scenario_instance_id: Any,
    target_actor_id: Any,
    scheduled_round: Any,
    committed_action_boundary: Any,
    sequence: Any,
    enabled: Any,
    source: Any,
    content: Any,
) -> None:
    for name, value in (
        ("run_id", run_id),
        ("trial_id", trial_id),
        ("scenario_instance_id", scenario_instance_id),
        ("target_actor_id", target_actor_id),
        ("source", source),
        ("content", content),
    ):
        _require_non_empty_string(value, name=name)
    _require_non_negative_int(scheduled_round, name="scheduled_round")
    _require_non_negative_int(
        committed_action_boundary,
        name="committed_action_boundary",
    )
    _require_non_negative_int(sequence, name="sequence")
    if type(enabled) is not bool:
        raise TypeError("enabled must be a boolean")


@dataclass(frozen=True)
class ProbeInterventionSpec:
    """Unbound scientific design for a verification or plausibility call."""

    kind: ProbeKind
    target_actor_id: str
    scheduled_round: int
    committed_action_boundary: int
    sequence: int
    enabled: bool
    source: str
    content: str
    schema_version: str = PROBE_INTERVENTION_SCHEMA_VERSION
    content_hash: str = field(init=False)
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != PROBE_INTERVENTION_SCHEMA_VERSION:
            raise ValueError("unsupported probe intervention schema_version")
        if not isinstance(self.kind, ProbeKind):
            raise TypeError("kind must be a ProbeKind")
        _validate_common_plan(
            run_id="unbound",
            trial_id="unbound",
            scenario_instance_id="unbound",
            target_actor_id=self.target_actor_id,
            scheduled_round=self.scheduled_round,
            committed_action_boundary=self.committed_action_boundary,
            sequence=self.sequence,
            enabled=self.enabled,
            source=self.source,
            content=self.content,
        )
        object.__setattr__(self, "content_hash", content_sha256(self.content))
        object.__setattr__(
            self,
            "spec_id",
            _stable_id("probe_spec", self.to_dict(include_id=False)),
        )

    @property
    def family(self) -> InterventionFamily:
        return InterventionFamily.PROBE

    def bind(
        self,
        *,
        run_id: str,
        trial_id: str,
        scenario_instance_id: str,
    ) -> "ProbeInterventionPlan":
        """Bind the design to execution identities after scenario compilation."""
        return ProbeInterventionPlan(
            run_id=run_id,
            trial_id=trial_id,
            scenario_instance_id=scenario_instance_id,
            kind=self.kind,
            target_actor_id=self.target_actor_id,
            scheduled_round=self.scheduled_round,
            committed_action_boundary=self.committed_action_boundary,
            sequence=self.sequence,
            enabled=self.enabled,
            source=self.source,
            content=self.content,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "family": self.family.value,
            "kind": self.kind.value,
            "target_actor_id": self.target_actor_id,
            "scheduled_round": self.scheduled_round,
            "committed_action_boundary": self.committed_action_boundary,
            "sequence": self.sequence,
            "enabled": self.enabled,
            "source": self.source,
            "content": self.content,
            "content_hash": self.content_hash,
        }
        if include_id:
            result["spec_id"] = self.spec_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProbeInterventionSpec":
        _require_exact_fields(
            value,
            expected=frozenset({
                "schema_version",
                "family",
                "kind",
                "target_actor_id",
                "scheduled_round",
                "committed_action_boundary",
                "sequence",
                "enabled",
                "source",
                "content",
                "content_hash",
                "spec_id",
            }),
            record_name="probe intervention spec",
        )
        if value["schema_version"] != PROBE_INTERVENTION_SCHEMA_VERSION:
            raise ValueError("unsupported probe intervention schema_version")
        if value["family"] != InterventionFamily.PROBE.value:
            raise ValueError("probe spec family must be 'probe'")
        spec = cls(
            schema_version=value["schema_version"],
            kind=_require_serialized_enum(
                value["kind"], ProbeKind, name="probe kind"
            ),
            target_actor_id=value["target_actor_id"],
            scheduled_round=value["scheduled_round"],
            committed_action_boundary=value["committed_action_boundary"],
            sequence=value["sequence"],
            enabled=value["enabled"],
            source=value["source"],
            content=value["content"],
        )
        _validate_persisted_content_identity(
            value,
            expected_hash=spec.content_hash,
            expected_id=spec.spec_id,
            id_name="spec_id",
            record_name="probe spec",
        )
        return spec


@dataclass(frozen=True)
class ScriptedObservationSpec:
    """Unbound scientific design for an externally supplied observation."""

    kind: ScriptedObservationKind
    target_actor_id: str
    scheduled_round: int
    committed_action_boundary: int
    sequence: int
    enabled: bool
    source: str
    content: str
    schema_version: str = SCRIPTED_OBSERVATION_SCHEMA_VERSION
    content_hash: str = field(init=False)
    spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != SCRIPTED_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unsupported scripted observation schema_version")
        if not isinstance(self.kind, ScriptedObservationKind):
            raise TypeError("kind must be a ScriptedObservationKind")
        _validate_common_plan(
            run_id="unbound",
            trial_id="unbound",
            scenario_instance_id="unbound",
            target_actor_id=self.target_actor_id,
            scheduled_round=self.scheduled_round,
            committed_action_boundary=self.committed_action_boundary,
            sequence=self.sequence,
            enabled=self.enabled,
            source=self.source,
            content=self.content,
        )
        object.__setattr__(self, "content_hash", content_sha256(self.content))
        object.__setattr__(
            self,
            "spec_id",
            _stable_id("observation_spec", self.to_dict(include_id=False)),
        )

    @property
    def family(self) -> InterventionFamily:
        return InterventionFamily.SCRIPTED_OBSERVATION

    def bind(
        self,
        *,
        run_id: str,
        trial_id: str,
        scenario_instance_id: str,
    ) -> "ScriptedObservationPlan":
        """Bind the design to execution identities after scenario compilation."""
        return ScriptedObservationPlan(
            run_id=run_id,
            trial_id=trial_id,
            scenario_instance_id=scenario_instance_id,
            kind=self.kind,
            target_actor_id=self.target_actor_id,
            scheduled_round=self.scheduled_round,
            committed_action_boundary=self.committed_action_boundary,
            sequence=self.sequence,
            enabled=self.enabled,
            source=self.source,
            content=self.content,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "family": self.family.value,
            "kind": self.kind.value,
            "target_actor_id": self.target_actor_id,
            "scheduled_round": self.scheduled_round,
            "committed_action_boundary": self.committed_action_boundary,
            "sequence": self.sequence,
            "enabled": self.enabled,
            "source": self.source,
            "content": self.content,
            "content_hash": self.content_hash,
        }
        if include_id:
            result["spec_id"] = self.spec_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScriptedObservationSpec":
        _require_exact_fields(
            value,
            expected=frozenset({
                "schema_version",
                "family",
                "kind",
                "target_actor_id",
                "scheduled_round",
                "committed_action_boundary",
                "sequence",
                "enabled",
                "source",
                "content",
                "content_hash",
                "spec_id",
            }),
            record_name="scripted observation spec",
        )
        if value["schema_version"] != SCRIPTED_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unsupported scripted observation schema_version")
        if value["family"] != InterventionFamily.SCRIPTED_OBSERVATION.value:
            raise ValueError(
                "scripted observation spec family must be 'scripted_observation'"
            )
        spec = cls(
            schema_version=value["schema_version"],
            kind=_require_serialized_enum(
                value["kind"],
                ScriptedObservationKind,
                name="scripted observation kind",
            ),
            target_actor_id=value["target_actor_id"],
            scheduled_round=value["scheduled_round"],
            committed_action_boundary=value["committed_action_boundary"],
            sequence=value["sequence"],
            enabled=value["enabled"],
            source=value["source"],
            content=value["content"],
        )
        _validate_persisted_content_identity(
            value,
            expected_hash=spec.content_hash,
            expected_id=spec.spec_id,
            id_name="spec_id",
            record_name="scripted observation spec",
        )
        return spec


RuntimeInterventionSpec: TypeAlias = (
    ProbeInterventionSpec | ScriptedObservationSpec
)


def _validate_persisted_content_identity(
    value: Mapping[str, Any],
    *,
    expected_hash: str,
    expected_id: str,
    id_name: str,
    record_name: str,
) -> None:
    serialized_hash = value["content_hash"]
    if not isinstance(serialized_hash, str) or not _SHA256_PATTERN.fullmatch(
        serialized_hash
    ):
        raise ValueError(f"{record_name} requires a canonical content hash")
    if serialized_hash != expected_hash:
        raise ValueError(f"{record_name} content hash does not match content")
    serialized_id = value[id_name]
    if not isinstance(serialized_id, str) or not serialized_id:
        raise ValueError(f"persisted {record_name} requires {id_name}")
    if serialized_id != expected_id:
        raise ValueError(f"{record_name} {id_name} does not match content")


@dataclass(frozen=True)
class ProbeInterventionPlan:
    """Locked design for one verification or plausibility model call."""

    run_id: str
    trial_id: str
    scenario_instance_id: str
    kind: ProbeKind
    target_actor_id: str
    scheduled_round: int
    committed_action_boundary: int
    sequence: int
    enabled: bool
    source: str
    content: str
    schema_version: str = PROBE_INTERVENTION_SCHEMA_VERSION
    content_hash: str = field(init=False)
    spec_id: str = field(init=False)
    design_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != PROBE_INTERVENTION_SCHEMA_VERSION:
            raise ValueError("unsupported probe intervention schema_version")
        if not isinstance(self.kind, ProbeKind):
            raise TypeError("kind must be a ProbeKind")
        _validate_common_plan(
            run_id=self.run_id,
            trial_id=self.trial_id,
            scenario_instance_id=self.scenario_instance_id,
            target_actor_id=self.target_actor_id,
            scheduled_round=self.scheduled_round,
            committed_action_boundary=self.committed_action_boundary,
            sequence=self.sequence,
            enabled=self.enabled,
            source=self.source,
            content=self.content,
        )
        spec = self.to_spec()
        object.__setattr__(self, "content_hash", spec.content_hash)
        object.__setattr__(self, "spec_id", spec.spec_id)
        object.__setattr__(
            self,
            "design_id",
            _stable_id("probe_design", self.to_dict(include_id=False)),
        )

    @property
    def family(self) -> InterventionFamily:
        return InterventionFamily.PROBE

    def to_spec(self) -> ProbeInterventionSpec:
        """Project the bound plan back to its unchanged scientific spec."""
        return ProbeInterventionSpec(
            kind=self.kind,
            target_actor_id=self.target_actor_id,
            scheduled_round=self.scheduled_round,
            committed_action_boundary=self.committed_action_boundary,
            sequence=self.sequence,
            enabled=self.enabled,
            source=self.source,
            content=self.content,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "family": self.family.value,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "scenario_instance_id": self.scenario_instance_id,
            "kind": self.kind.value,
            "target_actor_id": self.target_actor_id,
            "scheduled_round": self.scheduled_round,
            "committed_action_boundary": self.committed_action_boundary,
            "sequence": self.sequence,
            "enabled": self.enabled,
            "source": self.source,
            "content": self.content,
            "content_hash": self.content_hash,
            "spec_id": self.spec_id,
        }
        if include_id:
            result["design_id"] = self.design_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProbeInterventionPlan":
        _require_exact_fields(
            value,
            expected=frozenset({
                "schema_version",
                "family",
                "run_id",
                "trial_id",
                "scenario_instance_id",
                "kind",
                "target_actor_id",
                "scheduled_round",
                "committed_action_boundary",
                "sequence",
                "enabled",
                "source",
                "content",
                "content_hash",
                "spec_id",
                "design_id",
            }),
            record_name="probe intervention plan",
        )
        if value["schema_version"] != PROBE_INTERVENTION_SCHEMA_VERSION:
            raise ValueError("unsupported probe intervention schema_version")
        if value["family"] != InterventionFamily.PROBE.value:
            raise ValueError("probe plan family must be 'probe'")
        kind = _require_serialized_enum(
            value["kind"], ProbeKind, name="probe kind"
        )
        plan = cls(
            schema_version=value["schema_version"],
            run_id=value["run_id"],
            trial_id=value["trial_id"],
            scenario_instance_id=value["scenario_instance_id"],
            kind=kind,
            target_actor_id=value["target_actor_id"],
            scheduled_round=value["scheduled_round"],
            committed_action_boundary=value["committed_action_boundary"],
            sequence=value["sequence"],
            enabled=value["enabled"],
            source=value["source"],
            content=value["content"],
        )
        _validate_persisted_content_identity(
            value,
            expected_hash=plan.content_hash,
            expected_id=plan.spec_id,
            id_name="spec_id",
            record_name="probe plan",
        )
        if not isinstance(value["design_id"], str) or not value["design_id"]:
            raise ValueError("persisted probe plan requires design_id")
        if value["design_id"] != plan.design_id:
            raise ValueError("probe design_id does not match plan content")
        return plan


@dataclass(frozen=True)
class ScriptedObservationPlan:
    """Locked external observation scheduled at one committed boundary."""

    run_id: str
    trial_id: str
    scenario_instance_id: str
    kind: ScriptedObservationKind
    target_actor_id: str
    scheduled_round: int
    committed_action_boundary: int
    sequence: int
    enabled: bool
    source: str
    content: str
    schema_version: str = SCRIPTED_OBSERVATION_SCHEMA_VERSION
    content_hash: str = field(init=False)
    spec_id: str = field(init=False)
    design_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != SCRIPTED_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unsupported scripted observation schema_version")
        if not isinstance(self.kind, ScriptedObservationKind):
            raise TypeError("kind must be a ScriptedObservationKind")
        _validate_common_plan(
            run_id=self.run_id,
            trial_id=self.trial_id,
            scenario_instance_id=self.scenario_instance_id,
            target_actor_id=self.target_actor_id,
            scheduled_round=self.scheduled_round,
            committed_action_boundary=self.committed_action_boundary,
            sequence=self.sequence,
            enabled=self.enabled,
            source=self.source,
            content=self.content,
        )
        spec = self.to_spec()
        object.__setattr__(self, "content_hash", spec.content_hash)
        object.__setattr__(self, "spec_id", spec.spec_id)
        object.__setattr__(
            self,
            "design_id",
            _stable_id("observation_design", self.to_dict(include_id=False)),
        )

    @property
    def family(self) -> InterventionFamily:
        return InterventionFamily.SCRIPTED_OBSERVATION

    def to_spec(self) -> ScriptedObservationSpec:
        """Project the bound plan back to its unchanged scientific spec."""
        return ScriptedObservationSpec(
            kind=self.kind,
            target_actor_id=self.target_actor_id,
            scheduled_round=self.scheduled_round,
            committed_action_boundary=self.committed_action_boundary,
            sequence=self.sequence,
            enabled=self.enabled,
            source=self.source,
            content=self.content,
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "family": self.family.value,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "scenario_instance_id": self.scenario_instance_id,
            "kind": self.kind.value,
            "target_actor_id": self.target_actor_id,
            "scheduled_round": self.scheduled_round,
            "committed_action_boundary": self.committed_action_boundary,
            "sequence": self.sequence,
            "enabled": self.enabled,
            "source": self.source,
            "content": self.content,
            "content_hash": self.content_hash,
            "spec_id": self.spec_id,
        }
        if include_id:
            result["design_id"] = self.design_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScriptedObservationPlan":
        _require_exact_fields(
            value,
            expected=frozenset({
                "schema_version",
                "family",
                "run_id",
                "trial_id",
                "scenario_instance_id",
                "kind",
                "target_actor_id",
                "scheduled_round",
                "committed_action_boundary",
                "sequence",
                "enabled",
                "source",
                "content",
                "content_hash",
                "spec_id",
                "design_id",
            }),
            record_name="scripted observation plan",
        )
        if value["schema_version"] != SCRIPTED_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unsupported scripted observation schema_version")
        if value["family"] != InterventionFamily.SCRIPTED_OBSERVATION.value:
            raise ValueError(
                "scripted observation family must be 'scripted_observation'"
            )
        kind = _require_serialized_enum(
            value["kind"],
            ScriptedObservationKind,
            name="scripted observation kind",
        )
        plan = cls(
            schema_version=value["schema_version"],
            run_id=value["run_id"],
            trial_id=value["trial_id"],
            scenario_instance_id=value["scenario_instance_id"],
            kind=kind,
            target_actor_id=value["target_actor_id"],
            scheduled_round=value["scheduled_round"],
            committed_action_boundary=value["committed_action_boundary"],
            sequence=value["sequence"],
            enabled=value["enabled"],
            source=value["source"],
            content=value["content"],
        )
        _validate_persisted_content_identity(
            value,
            expected_hash=plan.content_hash,
            expected_id=plan.spec_id,
            id_name="spec_id",
            record_name="scripted observation plan",
        )
        if not isinstance(value["design_id"], str) or not value["design_id"]:
            raise ValueError("persisted scripted observation requires design_id")
        if value["design_id"] != plan.design_id:
            raise ValueError("observation design_id does not match plan content")
        return plan


RuntimeInterventionPlan: TypeAlias = (
    ProbeInterventionPlan | ScriptedObservationPlan
)


def _spec_order(spec: RuntimeInterventionSpec) -> tuple[int, int, int, str]:
    return (
        spec.scheduled_round,
        spec.committed_action_boundary,
        spec.sequence,
        spec.spec_id,
    )


def _validate_unique_interventions(
    interventions: Sequence[RuntimeInterventionSpec | RuntimeInterventionPlan],
    *,
    identity_name: str,
) -> None:
    identities = [getattr(item, identity_name) for item in interventions]
    if len(set(identities)) != len(identities):
        raise ValueError(f"intervention collection contains duplicate {identity_name}s")
    slots = [
        (
            item.target_actor_id,
            item.scheduled_round,
            item.committed_action_boundary,
            item.sequence,
        )
        for item in interventions
    ]
    if len(set(slots)) != len(slots):
        raise ValueError(
            "interventions for one actor and boundary require unique sequence"
        )


@dataclass(frozen=True)
class InterventionDesign:
    """Unbound, deterministic scientific intervention design for a trial."""

    specs: tuple[RuntimeInterventionSpec, ...]
    schema_version: str = INTERVENTION_DESIGN_SCHEMA_VERSION
    design_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != INTERVENTION_DESIGN_SCHEMA_VERSION:
            raise ValueError("unsupported intervention design schema_version")
        if not isinstance(self.specs, Sequence) or isinstance(
            self.specs, (str, bytes)
        ):
            raise TypeError("specs must be a sequence of intervention specs")
        specs = tuple(self.specs)
        if any(
            not isinstance(item, (ProbeInterventionSpec, ScriptedObservationSpec))
            for item in specs
        ):
            raise TypeError("design contains an unsupported intervention spec")
        _validate_unique_interventions(specs, identity_name="spec_id")
        object.__setattr__(self, "specs", tuple(sorted(specs, key=_spec_order)))
        object.__setattr__(
            self,
            "design_id",
            _stable_id("intervention_design", self.to_dict(include_id=False)),
        )

    def bind(
        self,
        *,
        run_id: str,
        trial_id: str,
        scenario_instance_id: str,
    ) -> "InterventionSchedule":
        """Create bound plans while preserving this scientific design ID."""
        schedule = InterventionSchedule(
            run_id=run_id,
            trial_id=trial_id,
            scenario_instance_id=scenario_instance_id,
            plans=tuple(
                item.bind(
                    run_id=run_id,
                    trial_id=trial_id,
                    scenario_instance_id=scenario_instance_id,
                )
                for item in self.specs
            ),
        )
        if schedule.intervention_design_id != self.design_id:
            raise RuntimeError("bound schedule changed intervention design identity")
        return schedule

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "specs": [item.to_dict() for item in self.specs],
        }
        if include_id:
            result["design_id"] = self.design_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InterventionDesign":
        _require_exact_fields(
            value,
            expected=frozenset({"schema_version", "specs", "design_id"}),
            record_name="intervention design",
        )
        if value["schema_version"] != INTERVENTION_DESIGN_SCHEMA_VERSION:
            raise ValueError("unsupported intervention design schema_version")
        if not isinstance(value["specs"], list):
            raise TypeError("serialized intervention specs must be an array")
        specs: list[RuntimeInterventionSpec] = []
        for serialized in value["specs"]:
            if not isinstance(serialized, Mapping):
                raise TypeError("serialized intervention spec must be a mapping")
            family = _require_serialized_enum(
                serialized.get("family"),
                InterventionFamily,
                name="intervention family",
            )
            if family is InterventionFamily.PROBE:
                specs.append(ProbeInterventionSpec.from_dict(serialized))
            else:
                specs.append(ScriptedObservationSpec.from_dict(serialized))
        design = cls(
            schema_version=value["schema_version"],
            specs=tuple(specs),
        )
        if not isinstance(value["design_id"], str) or not value["design_id"]:
            raise ValueError("persisted intervention design requires design_id")
        if value["design_id"] != design.design_id:
            raise ValueError("design_id does not match intervention design content")
        return design


def _plan_order(plan: RuntimeInterventionPlan) -> tuple[int, int, int, str]:
    return (
        plan.scheduled_round,
        plan.committed_action_boundary,
        plan.sequence,
        plan.design_id,
    )


@dataclass(frozen=True)
class InterventionSchedule:
    """Deterministic, duplicate-free intervention plans for one trial."""

    run_id: str
    trial_id: str
    scenario_instance_id: str
    plans: tuple[RuntimeInterventionPlan, ...]
    schema_version: str = INTERVENTION_SCHEDULE_SCHEMA_VERSION
    intervention_design_id: str = field(init=False)
    schedule_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != INTERVENTION_SCHEDULE_SCHEMA_VERSION:
            raise ValueError("unsupported intervention schedule schema_version")
        for name in ("run_id", "trial_id", "scenario_instance_id"):
            _require_non_empty_string(getattr(self, name), name=name)
        if not isinstance(self.plans, Sequence) or isinstance(
            self.plans, (str, bytes)
        ):
            raise TypeError("plans must be a sequence of intervention plans")
        plans = tuple(self.plans)
        if any(
            not isinstance(item, (ProbeInterventionPlan, ScriptedObservationPlan))
            for item in plans
        ):
            raise TypeError("schedule contains an unsupported intervention plan")
        for plan in plans:
            if (
                plan.run_id != self.run_id
                or plan.trial_id != self.trial_id
                or plan.scenario_instance_id != self.scenario_instance_id
            ):
                raise ValueError("plan identity does not match intervention schedule")
        _validate_unique_interventions(plans, identity_name="design_id")
        object.__setattr__(self, "plans", tuple(sorted(plans, key=_plan_order)))
        unbound_design = InterventionDesign(
            specs=tuple(item.to_spec() for item in self.plans)
        )
        object.__setattr__(
            self,
            "intervention_design_id",
            unbound_design.design_id,
        )
        object.__setattr__(
            self,
            "schedule_id",
            _stable_id("intervention_schedule", self.to_dict(include_id=False)),
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "scenario_instance_id": self.scenario_instance_id,
            "intervention_design_id": self.intervention_design_id,
            "plans": [item.to_dict() for item in self.plans],
        }
        if include_id:
            result["schedule_id"] = self.schedule_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InterventionSchedule":
        _require_exact_fields(
            value,
            expected=frozenset({
                "schema_version",
                "run_id",
                "trial_id",
                "scenario_instance_id",
                "intervention_design_id",
                "plans",
                "schedule_id",
            }),
            record_name="intervention schedule",
        )
        if value["schema_version"] != INTERVENTION_SCHEDULE_SCHEMA_VERSION:
            raise ValueError("unsupported intervention schedule schema_version")
        if not isinstance(value["plans"], list):
            raise TypeError("serialized intervention plans must be an array")
        plans: list[RuntimeInterventionPlan] = []
        for serialized in value["plans"]:
            if not isinstance(serialized, Mapping):
                raise TypeError("serialized intervention plan must be a mapping")
            family = _require_serialized_enum(
                serialized.get("family"),
                InterventionFamily,
                name="intervention family",
            )
            if family is InterventionFamily.PROBE:
                plans.append(ProbeInterventionPlan.from_dict(serialized))
            else:
                plans.append(ScriptedObservationPlan.from_dict(serialized))
        schedule = cls(
            schema_version=value["schema_version"],
            run_id=value["run_id"],
            trial_id=value["trial_id"],
            scenario_instance_id=value["scenario_instance_id"],
            plans=tuple(plans),
        )
        if not isinstance(value["schedule_id"], str) or not value["schedule_id"]:
            raise ValueError("persisted intervention schedule requires schedule_id")
        if value["intervention_design_id"] != schedule.intervention_design_id:
            raise ValueError(
                "intervention_design_id does not match bound plan specs"
            )
        if value["schedule_id"] != schedule.schedule_id:
            raise ValueError("schedule_id does not match intervention schedule")
        return schedule


@dataclass(frozen=True)
class InterventionApplicationReceipt:
    """Exactly-once attestation for one scheduled runtime intervention."""

    run_id: str
    trial_id: str
    scenario_instance_id: str
    schedule_id: str
    design_id: str
    family: InterventionFamily
    status: InterventionApplicationStatus
    applied_round: int
    committed_action_boundary: int
    evidence_call_id: str | None
    label_status: ProbeLabelStatus
    schema_version: str = INTERVENTION_APPLICATION_SCHEMA_VERSION
    application_id: str = field(init=False)
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != INTERVENTION_APPLICATION_SCHEMA_VERSION:
            raise ValueError("unsupported intervention application schema_version")
        for name in (
            "run_id",
            "trial_id",
            "scenario_instance_id",
            "schedule_id",
            "design_id",
        ):
            _require_non_empty_string(getattr(self, name), name=name)
        if not isinstance(self.family, InterventionFamily):
            raise TypeError("family must be an InterventionFamily")
        if not isinstance(self.status, InterventionApplicationStatus):
            raise TypeError("status must be an InterventionApplicationStatus")
        if not isinstance(self.label_status, ProbeLabelStatus):
            raise TypeError(
                "label_status must be explicitly inapplicable or unknown"
            )
        _require_non_negative_int(self.applied_round, name="applied_round")
        _require_non_negative_int(
            self.committed_action_boundary,
            name="committed_action_boundary",
        )
        if self.evidence_call_id is not None:
            _require_non_empty_string(
                self.evidence_call_id,
                name="evidence_call_id",
            )
        if self.status in {
            InterventionApplicationStatus.SKIPPED_DISABLED,
            InterventionApplicationStatus.SKIPPED_TERMINAL,
        }:
            if self.evidence_call_id is not None:
                raise ValueError("skipped applications cannot have evidence calls")
            if self.label_status is not ProbeLabelStatus.INAPPLICABLE:
                raise ValueError("skipped applications have inapplicable labels")
        if (
            self.family is InterventionFamily.PROBE
            and self.status is InterventionApplicationStatus.APPLIED
            and self.evidence_call_id is None
        ):
            raise ValueError("applied probes require an evidence_call_id")
        if (
            self.family is InterventionFamily.SCRIPTED_OBSERVATION
            and self.label_status is not ProbeLabelStatus.INAPPLICABLE
        ):
            raise ValueError("scripted observations have inapplicable probe labels")
        object.__setattr__(
            self,
            "application_id",
            _stable_id("intervention_application", {
                "schedule_id": self.schedule_id,
                "design_id": self.design_id,
            }),
        )
        object.__setattr__(
            self,
            "receipt_id",
            _stable_id("intervention_receipt", self.to_dict(include_ids=False)),
        )

    @classmethod
    def for_plan(
        cls,
        schedule: InterventionSchedule,
        plan: RuntimeInterventionPlan,
        *,
        status: InterventionApplicationStatus,
        evidence_call_id: str | None,
        label_status: ProbeLabelStatus,
    ) -> "InterventionApplicationReceipt":
        """Create a receipt after validating plan membership and disposition."""
        matching = {
            candidate.design_id: candidate for candidate in schedule.plans
        }.get(plan.design_id)
        if matching != plan:
            raise ValueError("receipt plan is not present in the schedule")
        if plan.enabled and status not in {
            InterventionApplicationStatus.APPLIED,
            InterventionApplicationStatus.SKIPPED_TERMINAL,
        }:
            raise ValueError(
                "enabled intervention must be applied or skipped_terminal"
            )
        if (
            not plan.enabled
            and status is not InterventionApplicationStatus.SKIPPED_DISABLED
        ):
            raise ValueError("disabled intervention must have skipped_disabled status")
        return cls(
            run_id=plan.run_id,
            trial_id=plan.trial_id,
            scenario_instance_id=plan.scenario_instance_id,
            schedule_id=schedule.schedule_id,
            design_id=plan.design_id,
            family=plan.family,
            status=status,
            applied_round=plan.scheduled_round,
            committed_action_boundary=plan.committed_action_boundary,
            evidence_call_id=evidence_call_id,
            label_status=label_status,
        )

    def to_dict(self, *, include_ids: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "scenario_instance_id": self.scenario_instance_id,
            "schedule_id": self.schedule_id,
            "design_id": self.design_id,
            "family": self.family.value,
            "status": self.status.value,
            "applied_round": self.applied_round,
            "committed_action_boundary": self.committed_action_boundary,
            "evidence_call_id": self.evidence_call_id,
            "label_status": self.label_status.value,
        }
        if include_ids:
            result["application_id"] = self.application_id
            result["receipt_id"] = self.receipt_id
        return result

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "InterventionApplicationReceipt":
        _require_exact_fields(
            value,
            expected=frozenset({
                "schema_version",
                "run_id",
                "trial_id",
                "scenario_instance_id",
                "schedule_id",
                "design_id",
                "family",
                "status",
                "applied_round",
                "committed_action_boundary",
                "evidence_call_id",
                "label_status",
                "application_id",
                "receipt_id",
            }),
            record_name="intervention application receipt",
        )
        if value["schema_version"] != INTERVENTION_APPLICATION_SCHEMA_VERSION:
            raise ValueError("unsupported intervention application schema_version")
        family = _require_serialized_enum(
            value["family"], InterventionFamily, name="intervention family"
        )
        status = _require_serialized_enum(
            value["status"],
            InterventionApplicationStatus,
            name="application status",
        )
        label_status = _require_serialized_enum(
            value["label_status"], ProbeLabelStatus, name="probe label status"
        )
        receipt = cls(
            schema_version=value["schema_version"],
            run_id=value["run_id"],
            trial_id=value["trial_id"],
            scenario_instance_id=value["scenario_instance_id"],
            schedule_id=value["schedule_id"],
            design_id=value["design_id"],
            family=family,
            status=status,
            applied_round=value["applied_round"],
            committed_action_boundary=value["committed_action_boundary"],
            evidence_call_id=value["evidence_call_id"],
            label_status=label_status,
        )
        for name in ("application_id", "receipt_id"):
            if not isinstance(value[name], str) or not value[name]:
                raise ValueError(f"persisted receipt requires {name}")
            if value[name] != getattr(receipt, name):
                raise ValueError(f"{name} does not match receipt content")
        return receipt


@dataclass(frozen=True)
class InterventionApplicationLog:
    """Immutable, duplicate-free prefix of exactly-once applications."""

    run_id: str
    trial_id: str
    scenario_instance_id: str
    schedule_id: str
    receipts: tuple[InterventionApplicationReceipt, ...] = ()
    schema_version: str = INTERVENTION_APPLICATION_LOG_SCHEMA_VERSION
    log_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != INTERVENTION_APPLICATION_LOG_SCHEMA_VERSION:
            raise ValueError("unsupported intervention application log schema_version")
        for name in (
            "run_id",
            "trial_id",
            "scenario_instance_id",
            "schedule_id",
        ):
            _require_non_empty_string(getattr(self, name), name=name)
        if not isinstance(self.receipts, Sequence) or isinstance(
            self.receipts, (str, bytes)
        ):
            raise TypeError("receipts must be a sequence")
        receipts = tuple(self.receipts)
        if any(not isinstance(item, InterventionApplicationReceipt) for item in receipts):
            raise TypeError("application log contains an unsupported receipt")
        for receipt in receipts:
            if (
                receipt.run_id != self.run_id
                or receipt.trial_id != self.trial_id
                or receipt.scenario_instance_id != self.scenario_instance_id
                or receipt.schedule_id != self.schedule_id
            ):
                raise ValueError("receipt identity does not match application log")
        for name, identities in (
            ("receipt", [item.receipt_id for item in receipts]),
            ("application", [item.application_id for item in receipts]),
            ("plan", [item.design_id for item in receipts]),
        ):
            if len(set(identities)) != len(identities):
                raise ValueError(
                    f"application log contains duplicate {name} receipts"
                )
        object.__setattr__(
            self,
            "receipts",
            tuple(sorted(receipts, key=lambda item: item.application_id)),
        )
        object.__setattr__(
            self,
            "log_id",
            _stable_id("intervention_log", self.to_dict(include_id=False)),
        )

    @classmethod
    def empty(cls, schedule: InterventionSchedule) -> "InterventionApplicationLog":
        return cls(
            run_id=schedule.run_id,
            trial_id=schedule.trial_id,
            scenario_instance_id=schedule.scenario_instance_id,
            schedule_id=schedule.schedule_id,
        )

    def append(
        self, receipt: InterventionApplicationReceipt
    ) -> "InterventionApplicationLog":
        """Return a new log, rejecting a second receipt for one application."""
        return InterventionApplicationLog(
            run_id=self.run_id,
            trial_id=self.trial_id,
            scenario_instance_id=self.scenario_instance_id,
            schedule_id=self.schedule_id,
            receipts=(*self.receipts, receipt),
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "scenario_instance_id": self.scenario_instance_id,
            "schedule_id": self.schedule_id,
            "receipts": [item.to_dict() for item in self.receipts],
        }
        if include_id:
            result["log_id"] = self.log_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InterventionApplicationLog":
        _require_exact_fields(
            value,
            expected=frozenset({
                "schema_version",
                "run_id",
                "trial_id",
                "scenario_instance_id",
                "schedule_id",
                "receipts",
                "log_id",
            }),
            record_name="intervention application log",
        )
        if value["schema_version"] != INTERVENTION_APPLICATION_LOG_SCHEMA_VERSION:
            raise ValueError("unsupported intervention application log schema_version")
        if not isinstance(value["receipts"], list):
            raise TypeError("serialized receipts must be an array")
        application_log = cls(
            schema_version=value["schema_version"],
            run_id=value["run_id"],
            trial_id=value["trial_id"],
            scenario_instance_id=value["scenario_instance_id"],
            schedule_id=value["schedule_id"],
            receipts=tuple(
                InterventionApplicationReceipt.from_dict(item)
                for item in value["receipts"]
            ),
        )
        if not isinstance(value["log_id"], str) or not value["log_id"]:
            raise ValueError("persisted intervention application log requires log_id")
        if value["log_id"] != application_log.log_id:
            raise ValueError("log_id does not match application log content")
        return application_log


@dataclass(frozen=True)
class InterventionProgress:
    """Pure execution partition for one persisted trial boundary."""

    pending: tuple[RuntimeInterventionPlan, ...]
    applied: tuple[RuntimeInterventionPlan, ...]
    skipped: tuple[RuntimeInterventionPlan, ...]
    terminal_skipped: tuple[RuntimeInterventionPlan, ...]
    future: tuple[RuntimeInterventionPlan, ...]
    overdue: tuple[RuntimeInterventionPlan, ...]
    disabled: tuple[RuntimeInterventionPlan, ...]


def calculate_intervention_progress(
    schedule: InterventionSchedule,
    application_log: InterventionApplicationLog,
    *,
    current_round: int,
    committed_action_boundary: int,
) -> InterventionProgress:
    """Partition plans without callbacks, mutation, time, or random state.

    A due enabled plan is ``pending`` only at its exact declared boundary.  If
    both monotonic coordinates have advanced without a receipt it is
    ``overdue`` so a resume path can fail closed instead of applying an
    intervention at a scientifically different point.
    """
    _require_non_negative_int(current_round, name="current_round")
    _require_non_negative_int(
        committed_action_boundary,
        name="committed_action_boundary",
    )
    if (
        application_log.run_id != schedule.run_id
        or application_log.trial_id != schedule.trial_id
        or application_log.scenario_instance_id != schedule.scenario_instance_id
        or application_log.schedule_id != schedule.schedule_id
    ):
        raise ValueError("application log does not belong to intervention schedule")

    plans_by_id = {item.design_id: item for item in schedule.plans}
    receipts_by_id = {item.design_id: item for item in application_log.receipts}
    for design_id, receipt in receipts_by_id.items():
        plan = plans_by_id.get(design_id)
        if plan is None:
            raise ValueError("application receipt references an unknown plan")
        if receipt.family is not plan.family:
            raise ValueError("application receipt family does not match plan")
        if (
            receipt.applied_round != plan.scheduled_round
            or receipt.committed_action_boundary
            != plan.committed_action_boundary
        ):
            raise ValueError("application receipt boundary does not match plan")
        expected_statuses = (
            {
                InterventionApplicationStatus.APPLIED,
                InterventionApplicationStatus.SKIPPED_TERMINAL,
            }
            if plan.enabled else {
                InterventionApplicationStatus.SKIPPED_DISABLED
            }
        )
        if receipt.status not in expected_statuses:
            raise ValueError("application receipt status does not match plan")

    buckets: dict[str, list[RuntimeInterventionPlan]] = {
        "pending": [],
        "applied": [],
        "skipped": [],
        "terminal_skipped": [],
        "future": [],
        "overdue": [],
        "disabled": [],
    }
    for plan in schedule.plans:
        receipt = receipts_by_id.get(plan.design_id)
        if receipt is not None:
            if receipt.status is InterventionApplicationStatus.APPLIED:
                name = "applied"
            elif (
                receipt.status
                is InterventionApplicationStatus.SKIPPED_TERMINAL
            ):
                name = "terminal_skipped"
            else:
                name = "skipped"
            buckets[name].append(plan)
            continue
        if not plan.enabled:
            buckets["disabled"].append(plan)
            continue
        round_reached = plan.scheduled_round <= current_round
        boundary_reached = (
            plan.committed_action_boundary <= committed_action_boundary
        )
        if (
            plan.scheduled_round == current_round
            and plan.committed_action_boundary == committed_action_boundary
        ):
            buckets["pending"].append(plan)
        elif round_reached and boundary_reached:
            buckets["overdue"].append(plan)
        else:
            buckets["future"].append(plan)

    return InterventionProgress(
        pending=tuple(buckets["pending"]),
        applied=tuple(buckets["applied"]),
        skipped=tuple(buckets["skipped"]),
        terminal_skipped=tuple(buckets["terminal_skipped"]),
        future=tuple(buckets["future"]),
        overdue=tuple(buckets["overdue"]),
        disabled=tuple(buckets["disabled"]),
    )


__all__ = [
    "INTERVENTION_APPLICATION_LOG_SCHEMA_VERSION",
    "INTERVENTION_APPLICATION_SCHEMA_VERSION",
    "INTERVENTION_DESIGN_SCHEMA_VERSION",
    "INTERVENTION_SCHEDULE_SCHEMA_VERSION",
    "PROBE_INTERVENTION_SCHEMA_VERSION",
    "SCRIPTED_OBSERVATION_SCHEMA_VERSION",
    "InterventionApplicationLog",
    "InterventionApplicationReceipt",
    "InterventionApplicationStatus",
    "InterventionDesign",
    "InterventionFamily",
    "InterventionProgress",
    "InterventionSchedule",
    "ProbeInterventionPlan",
    "ProbeInterventionSpec",
    "ProbeKind",
    "ProbeLabelStatus",
    "RuntimeInterventionPlan",
    "RuntimeInterventionSpec",
    "ScriptedObservationKind",
    "ScriptedObservationPlan",
    "ScriptedObservationSpec",
    "calculate_intervention_progress",
    "content_sha256",
]
