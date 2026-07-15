"""Deterministic compiler and authorization-safe renderer for scenario specs.

Compilation is a single pure boundary: a validated, content-addressed
``ScenarioSpec`` plus run identity resolves facts exactly once through a local
RNG.  Prompt rendering never samples or recompiles; it reads only values in the
compiled instance view authorized for the requested role.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random
import re
from string import Formatter
from typing import Any

from interpretability.scenarios.schema import (
    AdjudicatorView,
    FactDefinition,
    FactRef,
    FactSamplingKind,
    FactValueType,
    IncentiveCondition,
    PrivateView,
    PromptKind,
    PromptTemplate,
    PublicState,
    RoleKind,
    SCENARIO_DSL_SCHEMA_VERSION,
    ScenarioInstance,
    ScenarioSpec,
    ScheduledIntervention,
    Visibility,
    canonical_json,
    canonical_sha256,
)


COMPILER_VERSION = "scenario-compiler/1.0.0"

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]*$")
_DIRECT_PLACEHOLDER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_CONTEXT_FIELDS = frozenset({"scenario_id", "trial_id", "condition"})

# Exact concrete types are part of the persisted scientific contract.
# pylint: disable=unidiomatic-typecheck


@dataclass(frozen=True, slots=True)
# Provenance is intentionally explicit rather than hidden in an opaque mapping.
# pylint: disable-next=too-many-instance-attributes
class RenderedPrompt:
    """Rendered text plus immutable template and visibility provenance."""

    text: str
    template_id: str
    template_version: str
    template_hash: str
    instance_hash: str
    role_id: str
    kind: PromptKind
    condition: IncentiveCondition
    visible_fact_ids: tuple[str, ...]
    used_fact_ids: tuple[str, ...]
    render_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name, values in (
            ("visible_fact_ids", self.visible_fact_ids),
            ("used_fact_ids", self.used_fact_ids),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be canonically ordered and unique")
        if not set(self.used_fact_ids).issubset(self.visible_fact_ids):
            raise ValueError("used facts must be present in the authorized view")
        object.__setattr__(
            self,
            "render_hash",
            canonical_sha256(self._content_payload()),
        )

    @property
    def prompt(self) -> str:
        """Compatibility name for the rendered text."""
        return self.text

    def _content_payload(self) -> dict[str, Any]:
        return {
            "compiler_version": COMPILER_VERSION,
            "text": self.text,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "template_hash": self.template_hash,
            "instance_hash": self.instance_hash,
            "role_id": self.role_id,
            "kind": self.kind.value,
            "condition": self.condition.value,
            "visible_fact_ids": list(self.visible_fact_ids),
            "used_fact_ids": list(self.used_fact_ids),
        }

    def canonical_json(self) -> str:
        """Return deterministic JSON for the complete provenance record."""
        return canonical_json(
            {**self._content_payload(), "render_hash": self.render_hash}
        )


def _validated_spec(spec: ScenarioSpec) -> ScenarioSpec:
    if not isinstance(spec, ScenarioSpec):
        raise TypeError("spec must be a ScenarioSpec")
    if spec.schema_version != SCENARIO_DSL_SCHEMA_VERSION:
        raise ValueError("unsupported scenario specification schema version")
    return ScenarioSpec.from_persisted_json(spec.canonical_json())


def _validate_run_identity(
    trial_id: str,
    run_seed: int,
    condition: IncentiveCondition,
) -> None:
    if not isinstance(trial_id, str) or not _IDENTIFIER.fullmatch(trial_id):
        raise ValueError("trial_id must be a stable scenario identifier")
    if type(run_seed) is not int or run_seed < 0:
        raise ValueError("run_seed must be a nonnegative integer, not a boolean")
    if not isinstance(condition, IncentiveCondition):
        raise TypeError("condition must be an IncentiveCondition")


def _compilation_seed(
    spec: ScenarioSpec,
    trial_id: str,
    run_seed: int,
    condition: IncentiveCondition,
) -> int:
    material = canonical_json(
        {
            "domain": COMPILER_VERSION,
            "scenario_id": spec.metadata.scenario_id,
            "spec_version": spec.spec_version,
            "spec_hash": spec.spec_hash,
            "trial_id": trial_id,
            "run_seed": run_seed,
            "condition": condition.value,
        }
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def derive_compilation_seed(
    spec: ScenarioSpec,
    trial_id: str,
    run_seed: int,
    condition: IncentiveCondition,
) -> int:
    """Return the domain-separated local seed for one compilation identity."""
    validated = _validated_spec(spec)
    _validate_run_identity(trial_id, run_seed, condition)
    supported = {item.condition for item in validated.conditions}
    if condition not in supported:
        raise ValueError("condition is not declared by this scenario")
    return _compilation_seed(validated, trial_id, run_seed, condition)


def _matches_fact_type(value: Any, value_type: FactValueType) -> bool:
    expected_types: dict[FactValueType, type[Any]] = {
        FactValueType.INTEGER: int,
        FactValueType.FLOAT: float,
        FactValueType.STRING: str,
        FactValueType.BOOLEAN: bool,
    }
    expected = expected_types[value_type]
    if type(value) is not expected:
        return False
    return not (expected is float and not math.isfinite(value))


def _require_exact_fact_value(
    value: Any,
    value_type: FactValueType,
    *,
    fact_id: str,
) -> None:
    if not _matches_fact_type(value, value_type):
        raise ValueError(
            f"fact {fact_id!r} sampling value does not match declared "
            f"{value_type.value} type"
        )


def _resolve_fact(definition: FactDefinition, rng: random.Random) -> FactRef:
    sampling = definition.sampling
    if sampling.kind is FactSamplingKind.FIXED:
        value = sampling.fixed_value
        _require_exact_fact_value(
            value,
            definition.value_type,
            fact_id=definition.fact_id,
        )
    elif sampling.kind is FactSamplingKind.INTEGER_RANGE:
        if definition.value_type is not FactValueType.INTEGER:
            raise ValueError("integer_range requires an integer fact declaration")
        if type(sampling.minimum) is not int or type(sampling.maximum) is not int:
            raise ValueError("integer_range bounds must be exact integers")
        value = rng.randint(sampling.minimum, sampling.maximum)
    elif sampling.kind is FactSamplingKind.FLOAT_RANGE:
        if definition.value_type is not FactValueType.FLOAT:
            raise ValueError("float_range requires a float fact declaration")
        if type(sampling.minimum) is not float or type(sampling.maximum) is not float:
            raise ValueError("float_range bounds must be exact floats")
        if not math.isfinite(sampling.minimum) or not math.isfinite(
            sampling.maximum
        ):
            raise ValueError("float_range bounds must be finite")
        value = rng.uniform(sampling.minimum, sampling.maximum)
        if not math.isfinite(value):
            raise ValueError("float_range produced a nonfinite value")
    elif sampling.kind is FactSamplingKind.CHOICE:
        for choice in sampling.choices:
            _require_exact_fact_value(
                choice,
                definition.value_type,
                fact_id=definition.fact_id,
            )
        value = sampling.choices[rng.randrange(len(sampling.choices))]
    else:  # pragma: no cover - strict enum and exhaustive schema are defensive
        raise ValueError(f"unsupported fact sampling kind: {sampling.kind!r}")

    _require_exact_fact_value(
        value,
        definition.value_type,
        fact_id=definition.fact_id,
    )
    return FactRef(
        fact_id=definition.fact_id,
        fact_version=definition.fact_version,
        value=value,
        visibility=definition.visibility,
        visible_to=definition.visible_to,
    )


def validate_template_placeholders(template: str) -> tuple[str, ...]:
    """Return direct placeholders while rejecting Python format traversal."""
    if type(template) is not str or not template:
        raise ValueError("prompt template must be nonempty text")
    placeholders: list[str] = []
    try:
        parsed = tuple(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError("prompt template has invalid brace syntax") from exc
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if not field_name or not _DIRECT_PLACEHOLDER.fullmatch(field_name):
            raise ValueError(
                "prompt placeholders must be direct identifiers without "
                "attribute, index, or positional traversal"
            )
        if conversion is not None or format_spec:
            raise ValueError(
                "prompt placeholders cannot use conversions or format specs"
            )
        placeholders.append(field_name)
    return tuple(placeholders)


def _role_fact_ids(spec: ScenarioSpec, role_id: str) -> frozenset[str]:
    roles = {role.role_id: role for role in spec.roles}
    role = roles.get(role_id)
    if role is None:
        raise ValueError("role is not declared by this scenario")
    if role.kind is RoleKind.ADJUDICATOR:
        return frozenset(fact.fact_id for fact in spec.facts)
    return frozenset(
        fact.fact_id
        for fact in spec.facts
        if fact.visibility is Visibility.PUBLIC
        or (
            fact.visibility is Visibility.ROLE_PRIVATE
            and role_id in fact.visible_to
        )
    )


def _validate_prompt_templates(spec: ScenarioSpec) -> None:
    fact_ids = {fact.fact_id for fact in spec.facts}
    collisions = fact_ids.intersection(_CONTEXT_FIELDS)
    if collisions:
        raise ValueError("fact IDs cannot shadow prompt context fields")

    match_keys: set[tuple[str, IncentiveCondition, PromptKind]] = set()
    for template in spec.prompt_templates:
        key = (template.role_id, template.condition, template.kind)
        if key in match_keys:
            raise ValueError(
                "scenario contains duplicate templates for role, condition, and kind"
            )
        match_keys.add(key)
        authorized = _role_fact_ids(spec, template.role_id)
        for placeholder in validate_template_placeholders(template.template):
            if placeholder in _CONTEXT_FIELDS:
                continue
            if placeholder not in fact_ids:
                raise ValueError("prompt template references an unknown fact")
            if placeholder not in authorized:
                raise PermissionError(
                    "prompt template references a fact unavailable to its role"
                )


def _participant_role_ids(spec: ScenarioSpec) -> tuple[str, ...]:
    role_ids = tuple(
        role.role_id for role in spec.roles if role.kind is not RoleKind.ADJUDICATOR
    )
    if not role_ids:
        raise ValueError("scenario must declare at least one participant role")
    adjudicator_roles = {
        role.role_id for role in spec.roles if role.kind is RoleKind.ADJUDICATOR
    }
    for fact in spec.facts:
        if fact.visibility is Visibility.ROLE_PRIVATE and adjudicator_roles.intersection(
            fact.visible_to
        ):
            raise ValueError(
                "role-private facts cannot target the adjudicator role; use "
                "adjudicator_only visibility"
            )
    return role_ids


def compile_scenario(
    spec: ScenarioSpec,
    trial_id: str,
    run_seed: int,
    condition: IncentiveCondition,
) -> ScenarioInstance:
    """Compile one spec/run identity without mutating process-global RNG state."""
    validated = _validated_spec(spec)
    _validate_run_identity(trial_id, run_seed, condition)
    if condition not in {item.condition for item in validated.conditions}:
        raise ValueError("condition is not declared by this scenario")
    _validate_prompt_templates(validated)
    participant_role_ids = _participant_role_ids(validated)

    rng = random.Random(
        _compilation_seed(validated, trial_id, run_seed, condition)
    )
    resolved_facts = tuple(
        _resolve_fact(definition, rng) for definition in validated.facts
    )
    public_facts = tuple(
        fact for fact in resolved_facts if fact.visibility is Visibility.PUBLIC
    )
    private_views = tuple(
        PrivateView(
            role_id=role_id,
            facts=tuple(
                fact
                for fact in resolved_facts
                if fact.visibility is Visibility.PUBLIC
                or (
                    fact.visibility is Visibility.ROLE_PRIVATE
                    and role_id in fact.visible_to
                )
            ),
        )
        for role_id in participant_role_ids
    )
    interventions = tuple(
        ScheduledIntervention(
            intervention_id=item.intervention_id,
            round_index=item.round_index,
            recipient_role_id=item.recipient_role_id,
            template_id=item.template_id,
        )
        for item in validated.intervention_points
    )
    return ScenarioInstance(
        scenario_id=validated.metadata.scenario_id,
        spec_version=validated.spec_version,
        spec_hash=validated.spec_hash,
        run_seed=run_seed,
        trial_id=trial_id,
        condition=condition,
        resolved_facts=resolved_facts,
        public_state=PublicState(facts=public_facts),
        private_views=private_views,
        adjudicator_view=AdjudicatorView(facts=resolved_facts),
        legal_action_ids=tuple(
            action.action_id for action in validated.action_space
        ),
        interventions=interventions,
    )


def _validated_instance(instance: ScenarioInstance) -> ScenarioInstance:
    if not isinstance(instance, ScenarioInstance):
        raise TypeError("instance must be a ScenarioInstance")
    return ScenarioInstance.from_persisted_json(instance.canonical_json())


def _value_obeys_sampling(
    value: Any,
    definition: FactDefinition,
) -> bool:
    if not _matches_fact_type(value, definition.value_type):
        return False
    sampling = definition.sampling
    if sampling.kind is FactSamplingKind.FIXED:
        return value == sampling.fixed_value
    if sampling.kind in {
        FactSamplingKind.INTEGER_RANGE,
        FactSamplingKind.FLOAT_RANGE,
    }:
        return sampling.minimum <= value <= sampling.maximum
    if sampling.kind is FactSamplingKind.CHOICE:
        return value in sampling.choices
    return False


def _expected_interventions(
    spec: ScenarioSpec,
) -> tuple[ScheduledIntervention, ...]:
    return tuple(
        ScheduledIntervention(
            intervention_id=item.intervention_id,
            round_index=item.round_index,
            recipient_role_id=item.recipient_role_id,
            template_id=item.template_id,
        )
        for item in spec.intervention_points
    )


def _validate_instance_against_spec(
    spec: ScenarioSpec,
    instance: ScenarioInstance,
) -> None:
    if (
        instance.scenario_id != spec.metadata.scenario_id
        or instance.spec_version != spec.spec_version
        or instance.spec_hash != spec.spec_hash
    ):
        raise ValueError("compiled instance does not match its scenario spec")
    if instance.condition not in {item.condition for item in spec.conditions}:
        raise ValueError("compiled instance uses an undeclared condition")

    definitions = tuple(spec.facts)
    if tuple(fact.fact_id for fact in instance.resolved_facts) != tuple(
        definition.fact_id for definition in definitions
    ):
        raise ValueError("compiled instance facts do not match spec order and IDs")
    for definition, fact in zip(definitions, instance.resolved_facts, strict=True):
        if (
            fact.fact_version != definition.fact_version
            or fact.visibility is not definition.visibility
            or fact.visible_to != definition.visible_to
            or not _value_obeys_sampling(fact.value, definition)
        ):
            raise ValueError("compiled fact does not match its fact definition")

    participant_roles = _participant_role_ids(spec)
    if tuple(view.role_id for view in instance.private_views) != participant_roles:
        raise ValueError("compiled private views do not match participant roles")
    if instance.legal_action_ids != tuple(
        action.action_id for action in spec.action_space
    ):
        raise ValueError("compiled legal actions do not match the scenario spec")
    if instance.interventions != _expected_interventions(spec):
        raise ValueError("compiled interventions do not match the scenario spec")


def _authorized_instance_facts(
    spec: ScenarioSpec,
    instance: ScenarioInstance,
    role_id: str,
) -> tuple[FactRef, ...]:
    roles = {role.role_id: role for role in spec.roles}
    role = roles.get(role_id)
    if role is None:
        raise ValueError("role is not declared by this scenario")
    if role.kind is RoleKind.ADJUDICATOR:
        return instance.adjudicator_view.facts
    matching = tuple(
        view for view in instance.private_views if view.role_id == role_id
    )
    if len(matching) != 1:
        raise ValueError("compiled instance lacks one complete role view")
    return matching[0].facts


def _select_template(
    spec: ScenarioSpec,
    instance: ScenarioInstance,
    role_id: str,
    kind: PromptKind,
) -> PromptTemplate:
    if not isinstance(kind, PromptKind):
        raise TypeError("kind must be a PromptKind")
    if role_id not in {role.role_id for role in spec.roles}:
        raise ValueError("role is not declared by this scenario")
    matches = tuple(
        template
        for template in spec.prompt_templates
        if template.role_id == role_id
        and template.condition is instance.condition
        and template.kind is kind
    )
    if not matches:
        raise ValueError("no prompt template matches role, condition, and kind")
    if len(matches) != 1:
        raise ValueError("multiple prompt templates match role, condition, and kind")
    return matches[0]


def _render_scalar(value: Any) -> str:
    if type(value) is str:
        return value
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return json.dumps(value, allow_nan=False, ensure_ascii=False)
    raise ValueError("prompt facts must be finite declared scalar values")


def _render_validated_template(
    template: str,
    values: dict[str, Any],
) -> str:
    parts: list[str] = []
    for literal_text, field_name, _, _ in Formatter().parse(template):
        parts.append(literal_text)
        if field_name is not None:
            parts.append(_render_scalar(values[field_name]))
    return "".join(parts)


def render_prompt(
    spec: ScenarioSpec,
    instance: ScenarioInstance,
    role_id: str,
    kind: PromptKind,
) -> RenderedPrompt:
    """Render one authorized template strictly from a compiled instance."""
    validated_spec = _validated_spec(spec)
    validated_instance = _validated_instance(instance)
    _validate_prompt_templates(validated_spec)
    _validate_instance_against_spec(validated_spec, validated_instance)
    template = _select_template(
        validated_spec,
        validated_instance,
        role_id,
        kind,
    )
    facts = _authorized_instance_facts(
        validated_spec,
        validated_instance,
        role_id,
    )
    fact_values = {fact.fact_id: fact.value for fact in facts}
    placeholders = validate_template_placeholders(template.template)
    values: dict[str, Any] = {
        **fact_values,
        "scenario_id": validated_instance.scenario_id,
        "trial_id": validated_instance.trial_id,
        "condition": validated_instance.condition.value,
    }
    missing = tuple(name for name in placeholders if name not in values)
    if missing:
        raise PermissionError("prompt requires a missing or unauthorized fact")

    visible_fact_ids = tuple(sorted(fact_values))
    used_fact_ids = tuple(
        sorted(set(placeholders).intersection(fact_values))
    )
    return RenderedPrompt(
        text=_render_validated_template(template.template, values),
        template_id=template.template_id,
        template_version=template.template_version,
        template_hash=template.prompt_template_hash,
        instance_hash=validated_instance.instance_hash,
        role_id=role_id,
        kind=kind,
        condition=validated_instance.condition,
        visible_fact_ids=visible_fact_ids,
        used_fact_ids=used_fact_ids,
    )


__all__ = [
    "COMPILER_VERSION",
    "RenderedPrompt",
    "compile_scenario",
    "derive_compilation_seed",
    "render_prompt",
    "validate_template_placeholders",
]
