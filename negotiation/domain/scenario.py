"""Immutable compiled scenario and explicit role-view contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from negotiation.domain.schema import (
    SCHEMA_VERSION,
    _verify_serialized_id,
    freeze_mapping,
    stable_id,
    thaw_json,
)


_ROLE_VIEW_FIELDS = frozenset({"role_id", "public_state", "private_state"})
_SCENARIO_INSTANCE_FIELDS = frozenset({
    "spec_version",
    "scenario",
    "seed",
    "trial_id",
    "trial_family_id",
    "public_state",
    "role_views",
    "legal_actions",
    "rule_config",
    "instance_id",
})


def _require_exact_fields(
    data: Mapping[str, Any],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    if not isinstance(data, Mapping):
        raise TypeError(f"{context} must be a mapping")
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class RoleView:
    """The public and private facts visible to exactly one role."""

    role_id: str
    public_state: Mapping[str, Any]
    private_state: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.role_id, str) or not self.role_id:
            raise ValueError("role_id cannot be empty")
        object.__setattr__(self, "public_state", freeze_mapping(self.public_state))
        object.__setattr__(self, "private_state", freeze_mapping(self.private_state))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_id": self.role_id,
            "public_state": thaw_json(self.public_state),
            "private_state": thaw_json(self.private_state),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoleView:
        _require_exact_fields(
            data,
            _ROLE_VIEW_FIELDS,
            context="serialized role view",
        )
        return cls(
            role_id=data["role_id"],
            public_state=data["public_state"],
            private_state=data["private_state"],
        )


@dataclass(frozen=True)
class ScenarioInstance:
    """A deterministic scenario record used by prompts, rules, and metadata."""

    scenario: str
    seed: int
    trial_id: str
    trial_family_id: str
    public_state: Mapping[str, Any]
    role_views: tuple[RoleView, ...]
    legal_actions: tuple[str, ...]
    rule_config: Mapping[str, Any] = field(default_factory=dict)
    spec_version: str = SCHEMA_VERSION
    instance_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("spec_version", "scenario", "trial_id", "trial_family_id"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            not self.spec_version
            or not self.scenario
            or not self.trial_id
            or not self.trial_family_id
        ):
            raise ValueError("Scenario and trial identifiers cannot be empty")
        if self.spec_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported scenario spec_version: {self.spec_version!r}; "
                f"expected {SCHEMA_VERSION!r}"
            )
        public_state = freeze_mapping(self.public_state)
        role_views = tuple(self.role_views)
        legal_actions = tuple(self.legal_actions)
        if not legal_actions or any(
            not isinstance(item, str) or not item for item in legal_actions
        ):
            raise ValueError("legal_actions must contain named actions")
        role_ids = [view.role_id for view in role_views]
        if not role_ids or len(role_ids) != len(set(role_ids)):
            raise ValueError("role_views must contain unique role IDs")
        if any(dict(view.public_state) != dict(public_state) for view in role_views):
            raise ValueError("Every role view must share the scenario public state")
        object.__setattr__(self, "public_state", public_state)
        object.__setattr__(self, "role_views", role_views)
        object.__setattr__(self, "legal_actions", legal_actions)
        object.__setattr__(self, "rule_config", freeze_mapping(self.rule_config))
        object.__setattr__(self, "instance_id", stable_id("scenario", self._content_dict()))

    def view_for(self, role_id: str) -> RoleView:
        for view in self.role_views:
            if view.role_id == role_id:
                return view
        raise KeyError(f"Unknown role: {role_id}")

    def _content_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "scenario": self.scenario,
            "seed": self.seed,
            "trial_id": self.trial_id,
            "trial_family_id": self.trial_family_id,
            "public_state": thaw_json(self.public_state),
            "role_views": [view.to_dict() for view in self.role_views],
            "legal_actions": list(self.legal_actions),
            "rule_config": thaw_json(self.rule_config),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "instance_id": self.instance_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScenarioInstance:
        if not isinstance(data, Mapping):
            raise TypeError("serialized scenario instance must be a mapping")
        if "spec_version" not in data:
            raise ValueError("spec_version is required for persisted scenarios")
        if "instance_id" not in data:
            raise ValueError(
                "instance_id is required for persisted domain records"
            )
        _require_exact_fields(
            data,
            _SCENARIO_INSTANCE_FIELDS,
            context="serialized scenario instance",
        )
        instance = cls(
            spec_version=data["spec_version"],
            scenario=data["scenario"],
            seed=data["seed"],
            trial_id=data["trial_id"],
            trial_family_id=data["trial_family_id"],
            public_state=data["public_state"],
            role_views=tuple(RoleView.from_dict(item) for item in data["role_views"]),
            legal_actions=tuple(data["legal_actions"]),
            rule_config=data["rule_config"],
        )
        _verify_serialized_id(data, "instance_id", instance.instance_id)
        return instance
