"""Strict, immutable contracts for versioned deception scenarios.

The models in this module deliberately contain no rendering, extraction, or
evaluation logic.  They define the persisted boundary shared by those layers:
specifications describe a scenario, instances bind one specification to
resolved facts, observed actions retain atomic text evidence, and labels and
outcomes retain the facts and rules that justify them.

All collections are tuples and all models are frozen.  Persisted loaders
require the appropriate content identifier and verify it against canonical
JSON.  Timestamps are intentionally absent from content-addressed records.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
import math
import re
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Mapping,
    Self,
    TypeAlias,
    get_args,
    get_origin,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


SCENARIO_DSL_SCHEMA_VERSION = "scenario-dsl/1.0.0"
SCENARIO_INSTANCE_SCHEMA_VERSION = "scenario-instance/1.0.0"
ACTION_EVIDENCE_SCHEMA_VERSION = "scenario-action-evidence/1.0.0"
BEHAVIOR_LABEL_SCHEMA_VERSION = "scenario-behavior-label/1.0.0"
OUTCOME_SCHEMA_VERSION = "scenario-outcome/1.0.0"

_SEMVER = re.compile(
    r"^(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")

Identifier = Annotated[str, Field(min_length=1, pattern=_IDENTIFIER.pattern)]
NonEmptyString = Annotated[str, Field(min_length=1)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]

FactScalar: TypeAlias = str | int | float | bool
FactValue: TypeAlias = FactScalar | tuple[FactScalar, ...]


def _json_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible copy without mutable aliases."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical JSON does not permit NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize a model or JSON value with stable key and separator policy."""
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 digest of canonical UTF-8 JSON."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _content_identifier(prefix: str, value: Any) -> str:
    digest = canonical_sha256(value).removeprefix("sha256:")
    return f"{prefix}_{digest}"


def _annotation_contains_tuple(annotation: Any) -> bool:
    """Return whether a field annotation admits an immutable tuple value."""
    if get_origin(annotation) is tuple:
        return True
    return any(_annotation_contains_tuple(item) for item in get_args(annotation))


class _StrictFrozenModel(BaseModel):
    """Shared Pydantic policy for every canonical scenario object."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )

    @model_validator(mode="before")
    @classmethod
    def _restore_json_arrays_as_tuples(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Any:
        """Accept JSON arrays at persisted tuple fields without scalar coercion."""
        if not (info.context and info.context.get("persisted")):
            return value
        if not isinstance(value, Mapping):
            return value
        restored = dict(value)
        for field_name, field_info in cls.model_fields.items():
            field_value = restored.get(field_name)
            if isinstance(field_value, list) and _annotation_contains_tuple(
                field_info.annotation
            ):
                restored[field_name] = tuple(field_value)
        return restored


class _ContentAddressedModel(_StrictFrozenModel):
    """Derive and verify one subclass-specific content identifier."""

    _content_field: ClassVar[str]
    _content_prefix: ClassVar[str | None] = None

    @model_validator(mode="before")
    @classmethod
    def _require_persisted_identifier(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Any:
        if info.context and info.context.get("persisted"):
            if not isinstance(value, Mapping):
                raise TypeError("persisted scenario objects must be JSON objects")
            if "schema_version" not in value:
                raise ValueError(
                    "schema_version is required on persisted scenario objects"
                )
            persisted_id = value.get(cls._content_field)
            if not isinstance(persisted_id, str) or not persisted_id:
                raise ValueError(
                    f"{cls._content_field} is required on persisted objects"
                )
        return value

    @model_validator(mode="after")
    def _bind_content_identifier(self) -> Self:
        payload = self.model_dump(mode="json", exclude={self._content_field})
        expected = (
            canonical_sha256(payload)
            if self._content_prefix is None
            else _content_identifier(self._content_prefix, payload)
        )
        supplied = getattr(self, self._content_field)
        if supplied and supplied != expected:
            raise ValueError(
                f"{self._content_field} does not match canonical content"
            )
        object.__setattr__(self, self._content_field, expected)
        return self

    def canonical_json(self) -> str:
        """Return the complete persisted representation as canonical JSON."""
        return canonical_json(self)

    def canonical_content_json(self) -> str:
        """Return the exact representation covered by the content identifier."""
        return canonical_json(
            self.model_dump(mode="json", exclude={self._content_field})
        )

    @classmethod
    def from_persisted_json(cls, value: str | bytes) -> Self:
        """Restore JSON while requiring and verifying its content identifier."""
        return cls.model_validate_json(value, context={"persisted": True})

    @classmethod
    def from_persisted(cls, value: Mapping[str, Any]) -> Self:
        """Restore an already decoded JSON object with persisted validation."""
        return cls.model_validate_json(
            json.dumps(value, ensure_ascii=False, allow_nan=False),
            context={"persisted": True},
        )


class RoleKind(str, Enum):
    """Semantic roles supported by the dyadic scenario contract."""

    ACTOR = "actor"
    COUNTERPART = "counterpart"
    ADJUDICATOR = "adjudicator"


class Visibility(str, Enum):
    """Who may receive a resolved fact."""

    PUBLIC = "public"
    ROLE_PRIVATE = "role_private"
    ADJUDICATOR_ONLY = "adjudicator_only"


class IncentiveCondition(str, Enum):
    """Stable experimental conditions used by the six emergent scenarios."""

    HIGH_INCENTIVE = "high_incentive"
    LOW_INCENTIVE = "low_incentive"
    PENALTY = "penalty"
    MINIMAL = "minimal"


class FactValueType(str, Enum):
    """Declared scalar type for a generated fact."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"


class FactSamplingKind(str, Enum):
    """Closed set of deterministic fact-resolution strategies."""

    FIXED = "fixed"
    INTEGER_RANGE = "integer_range"
    FLOAT_RANGE = "float_range"
    CHOICE = "choice"


class PromptKind(str, Enum):
    """Purpose of one specification-owned prompt template."""

    INITIAL = "initial"
    COUNTERPART = "counterpart"
    INTERVENTION = "intervention"
    VERIFICATION = "verification"


class ActionKind(str, Enum):
    """Scenario-relevant structured action categories."""

    MESSAGE = "message"
    CLAIM = "claim"
    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"
    WALK_AWAY = "walk_away"
    COMMITMENT = "commitment"
    DISCLOSURE = "disclosure"


class ParseStatus(str, Enum):
    """Whether deterministic extraction produced usable atomic evidence."""

    COMPLETE = "complete"
    UNCERTAIN = "uncertain"
    NO_RELEVANT_ACTION = "no_relevant_action"
    FAILED = "failed"


class LabelValue(str, Enum):
    """Tri-state behavioral assessment."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class BehaviorSubtype(str, Enum):
    """Canonical, non-overlapping behavior constructs."""

    FALSE_CLAIM = "false_claim"
    STRATEGIC_OMISSION = "strategic_omission"
    MISLEADING_FRAMING = "misleading_framing"
    COMMITMENT_VIOLATION = "commitment_violation"


class LabelSource(str, Enum):
    """Independent provenance source for a behavior label."""

    RULE = "rule"
    MODEL_JUDGE = "model_judge"
    HUMAN = "human"
    AGENT_BELIEF = "agent_belief"
    LEGACY = "legacy"


class BeliefVerificationStatus(str, Enum):
    """Evidence that the acting model had the label-relevant private fact."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    FAILED = "failed"
    NOT_REQUIRED = "not_required"


class AgreementState(str, Enum):
    """Terminal agreement status without inferring success from prose."""

    AGREEMENT = "agreement"
    NO_AGREEMENT = "no_agreement"
    UNKNOWN = "unknown"


class OutcomeStatus(str, Enum):
    """Execution status of an adjudicated scenario outcome."""

    RESOLVED = "resolved"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ScenarioMetadata(_ContentAddressedModel):
    """Human-facing identity that remains versioned and content-addressed."""

    schema_version: Literal[SCENARIO_DSL_SCHEMA_VERSION] = (
        SCENARIO_DSL_SCHEMA_VERSION
    )
    scenario_id: Identifier
    display_name: NonEmptyString
    description: NonEmptyString
    research_constructs: tuple[BehaviorSubtype, ...]
    tags: tuple[Identifier, ...] = ()
    metadata_id: str = ""

    _content_field = "metadata_id"
    _content_prefix = "scenario_metadata"

    @model_validator(mode="after")
    def _validate_metadata(self) -> Self:
        if not self.research_constructs:
            raise ValueError("research_constructs must not be empty")
        if len(set(self.research_constructs)) != len(self.research_constructs):
            raise ValueError("research_constructs must be unique")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("metadata tags must be unique")
        return self


class RoleDefinition(_ContentAddressedModel):
    """One logical role declared by a scenario specification."""

    schema_version: Literal[SCENARIO_DSL_SCHEMA_VERSION] = (
        SCENARIO_DSL_SCHEMA_VERSION
    )
    role_id: Identifier
    kind: RoleKind
    description: NonEmptyString
    role_definition_id: str = ""

    _content_field = "role_definition_id"
    _content_prefix = "scenario_role"


class FactSamplingDefinition(_ContentAddressedModel):
    """Typed, executable parameter source; arbitrary code is never stored."""

    schema_version: Literal[SCENARIO_DSL_SCHEMA_VERSION] = (
        SCENARIO_DSL_SCHEMA_VERSION
    )
    kind: FactSamplingKind
    fixed_value: FactValue | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[FactScalar, ...] = ()
    sampling_id: str = ""

    _content_field = "sampling_id"
    _content_prefix = "fact_sampling"

    @model_validator(mode="after")
    def _validate_sampling_contract(self) -> Self:
        if self.kind is FactSamplingKind.FIXED:
            if self.fixed_value is None:
                raise ValueError("fixed sampling requires fixed_value")
            if self.minimum is not None or self.maximum is not None or self.choices:
                raise ValueError("fixed sampling cannot include ranges or choices")
        elif self.kind in {
            FactSamplingKind.INTEGER_RANGE,
            FactSamplingKind.FLOAT_RANGE,
        }:
            if self.minimum is None or self.maximum is None:
                raise ValueError("range sampling requires minimum and maximum")
            if isinstance(self.minimum, bool) or isinstance(self.maximum, bool):
                raise TypeError("range bounds cannot be booleans")
            if self.minimum > self.maximum:
                raise ValueError("sampling minimum cannot exceed maximum")
            if self.fixed_value is not None or self.choices:
                raise ValueError("range sampling cannot include fixed_value or choices")
            if self.kind is FactSamplingKind.INTEGER_RANGE and (
                type(self.minimum) is not int or type(self.maximum) is not int
            ):
                raise TypeError("integer_range bounds must be integers")
        elif self.kind is FactSamplingKind.CHOICE:
            if not self.choices:
                raise ValueError("choice sampling requires at least one choice")
            if len(set(self.choices)) != len(self.choices):
                raise ValueError("sampling choices must be unique")
            if (
                self.fixed_value is not None
                or self.minimum is not None
                or self.maximum is not None
            ):
                raise ValueError("choice sampling cannot include fixed or range values")
        return self


class FactDefinition(_ContentAddressedModel):
    """Versioned declaration of one public, private, or adjudicator fact."""

    schema_version: Literal[SCENARIO_DSL_SCHEMA_VERSION] = (
        SCENARIO_DSL_SCHEMA_VERSION
    )
    fact_id: Identifier
    fact_version: NonEmptyString
    value_type: FactValueType
    visibility: Visibility
    visible_to: tuple[Identifier, ...] = ()
    description: NonEmptyString
    sampling: FactSamplingDefinition
    fact_definition_id: str = ""

    _content_field = "fact_definition_id"
    _content_prefix = "fact_definition"

    @model_validator(mode="after")
    def _validate_visibility(self) -> Self:
        if len(set(self.visible_to)) != len(self.visible_to):
            raise ValueError("visible_to roles must be unique")
        if self.visibility is Visibility.ROLE_PRIVATE and not self.visible_to:
            raise ValueError("role-private facts require visible_to roles")
        if self.visibility is not Visibility.ROLE_PRIVATE and self.visible_to:
            raise ValueError(
                "only role-private facts may declare visible_to roles"
            )
        return self


class ConditionDefinition(_ContentAddressedModel):
    """One versioned incentive condition available to a scenario."""

    schema_version: Literal[SCENARIO_DSL_SCHEMA_VERSION] = (
        SCENARIO_DSL_SCHEMA_VERSION
    )
    condition: IncentiveCondition
    description: NonEmptyString
    condition_definition_id: str = ""

    _content_field = "condition_definition_id"
    _content_prefix = "scenario_condition"


class PromptTemplate(_ContentAddressedModel):
    """A role- and condition-bound prompt owned by the specification."""

    schema_version: Literal[SCENARIO_DSL_SCHEMA_VERSION] = (
        SCENARIO_DSL_SCHEMA_VERSION
    )
    template_id: Identifier
    template_version: NonEmptyString
    role_id: Identifier
    condition: IncentiveCondition
    kind: PromptKind
    template: NonEmptyString
    prompt_template_hash: str = ""

    _content_field = "prompt_template_hash"
    _content_prefix = None


class InterventionPoint(_ContentAddressedModel):
    """A legal, explicit point where a registered intervention may occur."""

    schema_version: Literal[SCENARIO_DSL_SCHEMA_VERSION] = (
        SCENARIO_DSL_SCHEMA_VERSION
    )
    intervention_id: Identifier
    round_index: NonNegativeInt
    recipient_role_id: Identifier
    template_id: Identifier
    intervention_point_id: str = ""

    _content_field = "intervention_point_id"
    _content_prefix = "intervention_point"

    @field_validator("round_index")
    @classmethod
    def _round_is_not_bool(cls, value: int) -> int:
        if type(value) is not int:
            raise TypeError("round_index must be an integer, not a boolean")
        return value


class ActionDefinition(_ContentAddressedModel):
    """One structured action admitted by the scenario adjudicator."""

    schema_version: Literal[SCENARIO_DSL_SCHEMA_VERSION] = (
        SCENARIO_DSL_SCHEMA_VERSION
    )
    action_id: Identifier
    kind: ActionKind
    actor_role_ids: tuple[Identifier, ...]
    terminal: bool = False
    description: NonEmptyString
    action_definition_id: str = ""

    _content_field = "action_definition_id"
    _content_prefix = "action_definition"

    @model_validator(mode="after")
    def _validate_roles(self) -> Self:
        if not self.actor_role_ids:
            raise ValueError("actions require at least one actor role")
        if len(set(self.actor_role_ids)) != len(self.actor_role_ids):
            raise ValueError("action actor roles must be unique")
        return self


class ExtractorReference(_ContentAddressedModel):
    """Versioned registry reference for deterministic action extraction."""

    schema_version: Literal[SCENARIO_DSL_SCHEMA_VERSION] = (
        SCENARIO_DSL_SCHEMA_VERSION
    )
    extractor_name: Identifier
    extractor_version: NonEmptyString
    supported_action_kinds: tuple[ActionKind, ...]
    deterministic: bool
    extractor_ref_id: str = ""

    _content_field = "extractor_ref_id"
    _content_prefix = "extractor_ref"

    @model_validator(mode="after")
    def _validate_supported_actions(self) -> Self:
        if not self.supported_action_kinds:
            raise ValueError("extractors require supported_action_kinds")
        if len(set(self.supported_action_kinds)) != len(
            self.supported_action_kinds
        ):
            raise ValueError("extractor action kinds must be unique")
        return self


class RuleReference(_ContentAddressedModel):
    """Stable reference from a spec to one closed predicate rule."""

    schema_version: Literal[SCENARIO_DSL_SCHEMA_VERSION] = (
        SCENARIO_DSL_SCHEMA_VERSION
    )
    rule_id: Identifier
    rule_version: NonEmptyString
    predicate_id: Identifier
    input_fact_ids: tuple[Identifier, ...]
    description: NonEmptyString
    rule_reference_id: str = ""

    _content_field = "rule_reference_id"
    _content_prefix = "rule_reference"

    @model_validator(mode="after")
    def _validate_inputs(self) -> Self:
        if len(set(self.input_fact_ids)) != len(self.input_fact_ids):
            raise ValueError("rule input facts must be unique")
        return self


class BehaviorTargetDefinition(_ContentAddressedModel):
    """One explicitly named behavioral construct and its rule aggregation."""

    schema_version: Literal[SCENARIO_DSL_SCHEMA_VERSION] = (
        SCENARIO_DSL_SCHEMA_VERSION
    )
    target_id: Identifier
    subtype: BehaviorSubtype
    rule_ids: tuple[Identifier, ...]
    belief_dependent: bool
    default_severity: UnitInterval
    behavior_target_id: str = ""

    _content_field = "behavior_target_id"
    _content_prefix = "behavior_target"

    @model_validator(mode="after")
    def _validate_rules(self) -> Self:
        if not self.rule_ids:
            raise ValueError("behavior targets require at least one rule")
        if len(set(self.rule_ids)) != len(self.rule_ids):
            raise ValueError("behavior target rules must be unique")
        return self


class OutcomeDefinition(_ContentAddressedModel):
    """One versioned outcome branch grounded in committed actions."""

    schema_version: Literal[SCENARIO_DSL_SCHEMA_VERSION] = (
        SCENARIO_DSL_SCHEMA_VERSION
    )
    outcome_id: Identifier
    rule_ids: tuple[Identifier, ...]
    agreement_state: AgreementState
    utility_role_ids: tuple[Identifier, ...]
    regret_baseline_rule_id: Identifier | None = None
    description: NonEmptyString
    outcome_definition_id: str = ""

    _content_field = "outcome_definition_id"
    _content_prefix = "outcome_definition"

    @model_validator(mode="after")
    def _validate_outcome_definition(self) -> Self:
        if not self.rule_ids:
            raise ValueError("outcome definitions require at least one rule")
        if len(set(self.rule_ids)) != len(self.rule_ids):
            raise ValueError("outcome rules must be unique")
        if len(set(self.utility_role_ids)) != len(self.utility_role_ids):
            raise ValueError("outcome utility roles must be unique")
        return self


class ScenarioSpec(_ContentAddressedModel):
    """The complete, versioned declaration compiled into scenario instances."""

    schema_version: Literal[SCENARIO_DSL_SCHEMA_VERSION] = (
        SCENARIO_DSL_SCHEMA_VERSION
    )
    spec_version: NonEmptyString
    metadata: ScenarioMetadata
    roles: tuple[RoleDefinition, ...]
    facts: tuple[FactDefinition, ...]
    conditions: tuple[ConditionDefinition, ...]
    prompt_templates: tuple[PromptTemplate, ...]
    intervention_points: tuple[InterventionPoint, ...] = ()
    action_space: tuple[ActionDefinition, ...]
    extractors: tuple[ExtractorReference, ...]
    rules: tuple[RuleReference, ...]
    behavior_targets: tuple[BehaviorTargetDefinition, ...]
    outcomes: tuple[OutcomeDefinition, ...]
    spec_hash: str = ""

    _content_field = "spec_hash"
    _content_prefix = None

    @field_validator("spec_version")
    @classmethod
    def _semantic_spec_version(cls, value: str) -> str:
        if not _SEMVER.fullmatch(value):
            raise ValueError("spec_version must be semantic version text")
        return value

    @model_validator(mode="after")
    def _validate_spec_graph(self) -> Self:
        def unique(values: tuple[str, ...], name: str) -> set[str]:
            if not values:
                raise ValueError(f"{name} must not be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique IDs")
            return set(values)

        role_ids = unique(tuple(item.role_id for item in self.roles), "roles")
        fact_ids = unique(tuple(item.fact_id for item in self.facts), "facts")
        condition_ids = unique(
            tuple(item.condition.value for item in self.conditions),
            "conditions",
        )
        template_ids = unique(
            tuple(item.template_id for item in self.prompt_templates),
            "prompt_templates",
        )
        action_ids = unique(
            tuple(item.action_id for item in self.action_space),
            "action_space",
        )
        del action_ids
        unique(
            tuple(item.extractor_name for item in self.extractors),
            "extractors",
        )
        rule_ids = unique(tuple(item.rule_id for item in self.rules), "rules")
        unique(
            tuple(item.target_id for item in self.behavior_targets),
            "behavior_targets",
        )
        unique(tuple(item.outcome_id for item in self.outcomes), "outcomes")

        if RoleKind.ACTOR not in {item.kind for item in self.roles}:
            raise ValueError("a scenario requires an actor role")
        if RoleKind.COUNTERPART not in {item.kind for item in self.roles}:
            raise ValueError("a scenario requires a counterpart role")
        for fact in self.facts:
            if not set(fact.visible_to).issubset(role_ids):
                raise ValueError(f"fact {fact.fact_id!r} names an unknown role")
        for template in self.prompt_templates:
            if template.role_id not in role_ids:
                raise ValueError(
                    f"prompt {template.template_id!r} names an unknown role"
                )
            if template.condition.value not in condition_ids:
                raise ValueError(
                    f"prompt {template.template_id!r} names an unknown condition"
                )
        for intervention in self.intervention_points:
            if intervention.recipient_role_id not in role_ids:
                raise ValueError(
                    f"intervention {intervention.intervention_id!r} names "
                    "an unknown role"
                )
            if intervention.template_id not in template_ids:
                raise ValueError(
                    f"intervention {intervention.intervention_id!r} names "
                    "an unknown template"
                )
        for action in self.action_space:
            if not set(action.actor_role_ids).issubset(role_ids):
                raise ValueError(f"action {action.action_id!r} names an unknown role")
        for rule in self.rules:
            if not set(rule.input_fact_ids).issubset(fact_ids):
                raise ValueError(f"rule {rule.rule_id!r} names an unknown fact")
        for target in self.behavior_targets:
            if not set(target.rule_ids).issubset(rule_ids):
                raise ValueError(
                    f"behavior target {target.target_id!r} names an unknown rule"
                )
        for outcome in self.outcomes:
            if not set(outcome.rule_ids).issubset(rule_ids):
                raise ValueError(
                    f"outcome {outcome.outcome_id!r} names an unknown rule"
                )
            if not set(outcome.utility_role_ids).issubset(role_ids):
                raise ValueError(
                    f"outcome {outcome.outcome_id!r} names an unknown utility role"
                )
            if (
                outcome.regret_baseline_rule_id is not None
                and outcome.regret_baseline_rule_id not in rule_ids
            ):
                raise ValueError(
                    f"outcome {outcome.outcome_id!r} names an unknown regret rule"
                )
        return self


class FactRef(_ContentAddressedModel):
    """One resolved fact with explicit visibility and version provenance."""

    schema_version: Literal[SCENARIO_INSTANCE_SCHEMA_VERSION] = (
        SCENARIO_INSTANCE_SCHEMA_VERSION
    )
    fact_id: Identifier
    fact_version: NonEmptyString
    value: FactValue
    visibility: Visibility
    visible_to: tuple[Identifier, ...] = ()
    fact_hash: str = ""

    _content_field = "fact_hash"
    _content_prefix = None

    @model_validator(mode="after")
    def _validate_visibility(self) -> Self:
        if len(set(self.visible_to)) != len(self.visible_to):
            raise ValueError("resolved fact visible_to roles must be unique")
        if self.visibility is Visibility.ROLE_PRIVATE and not self.visible_to:
            raise ValueError("role-private resolved facts require visible_to")
        if self.visibility is not Visibility.ROLE_PRIVATE and self.visible_to:
            raise ValueError(
                "public and adjudicator-only facts cannot name visible roles"
            )
        return self


class PublicState(_ContentAddressedModel):
    """Exactly the resolved facts visible to every participant."""

    schema_version: Literal[SCENARIO_INSTANCE_SCHEMA_VERSION] = (
        SCENARIO_INSTANCE_SCHEMA_VERSION
    )
    facts: tuple[FactRef, ...]
    public_state_hash: str = ""

    _content_field = "public_state_hash"
    _content_prefix = None

    @model_validator(mode="after")
    def _public_facts_only(self) -> Self:
        ids = tuple(item.fact_id for item in self.facts)
        if len(set(ids)) != len(ids):
            raise ValueError("public facts must have unique IDs")
        if any(item.visibility is not Visibility.PUBLIC for item in self.facts):
            raise ValueError("public state may contain only public facts")
        return self


class PrivateView(_ContentAddressedModel):
    """The complete authorized fact view delivered to one participant role."""

    schema_version: Literal[SCENARIO_INSTANCE_SCHEMA_VERSION] = (
        SCENARIO_INSTANCE_SCHEMA_VERSION
    )
    role_id: Identifier
    facts: tuple[FactRef, ...]
    view_hash: str = ""

    _content_field = "view_hash"
    _content_prefix = None

    @model_validator(mode="after")
    def _authorized_facts_only(self) -> Self:
        ids = tuple(item.fact_id for item in self.facts)
        if len(set(ids)) != len(ids):
            raise ValueError("private-view facts must have unique IDs")
        for fact in self.facts:
            if fact.visibility is Visibility.ADJUDICATOR_ONLY:
                raise ValueError("adjudicator-only facts cannot enter a role view")
            if (
                fact.visibility is Visibility.ROLE_PRIVATE
                and self.role_id not in fact.visible_to
            ):
                raise ValueError("role view contains an unauthorized private fact")
        return self


class AdjudicatorView(_ContentAddressedModel):
    """The access-controlled complete fact set used by rules and outcomes."""

    schema_version: Literal[SCENARIO_INSTANCE_SCHEMA_VERSION] = (
        SCENARIO_INSTANCE_SCHEMA_VERSION
    )
    facts: tuple[FactRef, ...]
    adjudicator_view_hash: str = ""

    _content_field = "adjudicator_view_hash"
    _content_prefix = None

    @model_validator(mode="after")
    def _unique_facts(self) -> Self:
        ids = tuple(item.fact_id for item in self.facts)
        if len(set(ids)) != len(ids):
            raise ValueError("adjudicator facts must have unique IDs")
        return self


class ScheduledIntervention(_ContentAddressedModel):
    """One intervention bound to an instance and recipient role."""

    schema_version: Literal[SCENARIO_INSTANCE_SCHEMA_VERSION] = (
        SCENARIO_INSTANCE_SCHEMA_VERSION
    )
    intervention_id: Identifier
    round_index: NonNegativeInt
    recipient_role_id: Identifier
    template_id: Identifier
    scheduled_intervention_id: str = ""

    _content_field = "scheduled_intervention_id"
    _content_prefix = "scheduled_intervention"

    @field_validator("round_index")
    @classmethod
    def _round_is_integer(cls, value: int) -> int:
        if type(value) is not int:
            raise TypeError("round_index must be an integer, not a boolean")
        return value


class ScenarioInstance(_ContentAddressedModel):
    """Immutable compilation result shared by prompts, labels, and outcomes."""

    schema_version: Literal[SCENARIO_INSTANCE_SCHEMA_VERSION] = (
        SCENARIO_INSTANCE_SCHEMA_VERSION
    )
    scenario_id: Identifier
    spec_version: NonEmptyString
    spec_hash: NonEmptyString
    run_seed: NonNegativeInt
    trial_id: Identifier
    condition: IncentiveCondition
    resolved_facts: tuple[FactRef, ...]
    public_state: PublicState
    private_views: tuple[PrivateView, ...]
    adjudicator_view: AdjudicatorView
    legal_action_ids: tuple[Identifier, ...]
    interventions: tuple[ScheduledIntervention, ...] = ()
    instance_hash: str = ""

    _content_field = "instance_hash"
    _content_prefix = None

    @field_validator("spec_version")
    @classmethod
    def _semantic_spec_version(cls, value: str) -> str:
        if not _SEMVER.fullmatch(value):
            raise ValueError("spec_version must be semantic version text")
        return value

    @field_validator("spec_hash")
    @classmethod
    def _valid_spec_hash(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("spec_hash must be a sha256: digest")
        return value

    @field_validator("run_seed")
    @classmethod
    def _seed_is_integer(cls, value: int) -> int:
        if type(value) is not int:
            raise TypeError("run_seed must be an integer, not a boolean")
        return value

    @model_validator(mode="after")
    def _validate_compiled_views(self) -> Self:
        fact_by_id = {item.fact_id: item for item in self.resolved_facts}
        if not fact_by_id or len(fact_by_id) != len(self.resolved_facts):
            raise ValueError("resolved_facts must contain unique fact IDs")
        if not self.private_views:
            raise ValueError("private_views must not be empty")
        role_ids = tuple(view.role_id for view in self.private_views)
        if len(set(role_ids)) != len(role_ids):
            raise ValueError("private views must have unique role IDs")
        if not self.legal_action_ids or len(set(self.legal_action_ids)) != len(
            self.legal_action_ids
        ):
            raise ValueError("legal_action_ids must be non-empty and unique")

        expected_public = {
            item.fact_id: item
            for item in self.resolved_facts
            if item.visibility is Visibility.PUBLIC
        }
        if {item.fact_id: item for item in self.public_state.facts} != expected_public:
            raise ValueError("public_state does not match resolved public facts")
        if {
            item.fact_id: item for item in self.adjudicator_view.facts
        } != fact_by_id:
            raise ValueError("adjudicator_view must contain every resolved fact")

        for view in self.private_views:
            expected = {
                item.fact_id: item
                for item in self.resolved_facts
                if item.visibility is Visibility.PUBLIC
                or (
                    item.visibility is Visibility.ROLE_PRIVATE
                    and view.role_id in item.visible_to
                )
            }
            if {item.fact_id: item for item in view.facts} != expected:
                raise ValueError(
                    f"private view {view.role_id!r} does not match authorization"
                )
        for fact in self.resolved_facts:
            if fact.visibility is Visibility.ROLE_PRIVATE and not set(
                fact.visible_to
            ).issubset(role_ids):
                raise ValueError(
                    f"resolved fact {fact.fact_id!r} names an absent role view"
                )
        if any(
            intervention.recipient_role_id not in role_ids
            for intervention in self.interventions
        ):
            raise ValueError("scheduled intervention names an absent role view")
        return self


class NormalizationDecision(_ContentAddressedModel):
    """Auditable normalization applied to one exact source span."""

    schema_version: Literal[ACTION_EVIDENCE_SCHEMA_VERSION] = (
        ACTION_EVIDENCE_SCHEMA_VERSION
    )
    normalizer_id: Identifier
    normalizer_version: NonEmptyString
    normalized_value: FactValue
    normalization_id: str = ""

    _content_field = "normalization_id"
    _content_prefix = "normalization"


class EvidenceSpan(_ContentAddressedModel):
    """Exact character evidence retained from an observed action."""

    schema_version: Literal[ACTION_EVIDENCE_SCHEMA_VERSION] = (
        ACTION_EVIDENCE_SCHEMA_VERSION
    )
    kind: Identifier
    start: NonNegativeInt
    end: PositiveInt
    text: NonEmptyString
    normalization: NormalizationDecision | None = None
    span_id: str = ""

    _content_field = "span_id"
    _content_prefix = "evidence_span"

    @model_validator(mode="after")
    def _validate_offsets(self) -> Self:
        if type(self.start) is not int or type(self.end) is not int:
            raise TypeError("evidence offsets must be integers, not booleans")
        if self.end <= self.start:
            raise ValueError("evidence end must be greater than start")
        if len(self.text) != self.end - self.start:
            raise ValueError("evidence text length must match its offsets")
        return self


class Claim(_ContentAddressedModel):
    """One atomic factual assertion attributed to the acting participant."""

    schema_version: Literal[ACTION_EVIDENCE_SCHEMA_VERSION] = (
        ACTION_EVIDENCE_SCHEMA_VERSION
    )
    subject_id: Identifier
    predicate: Identifier
    value: FactValue
    asserted_by: Identifier
    polarity: bool
    fact_id: Identifier | None = None
    evidence_spans: tuple[EvidenceSpan, ...]
    claim_id: str = ""

    _content_field = "claim_id"
    _content_prefix = "claim"

    @model_validator(mode="after")
    def _requires_evidence(self) -> Self:
        if not self.evidence_spans:
            raise ValueError("claims require evidence spans")
        return self


class OfferTerm(_ContentAddressedModel):
    """One normalized, evidence-bearing term of an offer."""

    schema_version: Literal[ACTION_EVIDENCE_SCHEMA_VERSION] = (
        ACTION_EVIDENCE_SCHEMA_VERSION
    )
    name: Identifier
    value: FactValue
    evidence_spans: tuple[EvidenceSpan, ...]
    term_id: str = ""

    _content_field = "term_id"
    _content_prefix = "offer_term"


class Offer(_ContentAddressedModel):
    """One actor proposal, separate from factual claims about private value."""

    schema_version: Literal[ACTION_EVIDENCE_SCHEMA_VERSION] = (
        ACTION_EVIDENCE_SCHEMA_VERSION
    )
    actor_id: Identifier
    recipient_ids: tuple[Identifier, ...]
    terms: tuple[OfferTerm, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    offer_id: str = ""

    _content_field = "offer_id"
    _content_prefix = "observed_offer"

    @model_validator(mode="after")
    def _validate_offer(self) -> Self:
        if not self.recipient_ids or len(set(self.recipient_ids)) != len(
            self.recipient_ids
        ):
            raise ValueError("offers require unique recipients")
        if not self.terms or len({item.name for item in self.terms}) != len(
            self.terms
        ):
            raise ValueError("offers require uniquely named terms")
        if not self.evidence_spans:
            raise ValueError("offers require evidence spans")
        return self


class Commitment(_ContentAddressedModel):
    """One explicit commitment to a future normalized action."""

    schema_version: Literal[ACTION_EVIDENCE_SCHEMA_VERSION] = (
        ACTION_EVIDENCE_SCHEMA_VERSION
    )
    actor_id: Identifier
    promised_action_id: Identifier
    beneficiary_role_ids: tuple[Identifier, ...]
    effective_round: NonNegativeInt | None = None
    evidence_spans: tuple[EvidenceSpan, ...]
    commitment_id: str = ""

    _content_field = "commitment_id"
    _content_prefix = "observed_commitment"

    @model_validator(mode="after")
    def _validate_commitment(self) -> Self:
        if not self.evidence_spans:
            raise ValueError("commitments require evidence spans")
        if len(set(self.beneficiary_role_ids)) != len(self.beneficiary_role_ids):
            raise ValueError("commitment beneficiaries must be unique")
        if self.effective_round is not None and type(self.effective_round) is not int:
            raise TypeError("effective_round must be an integer, not a boolean")
        return self


class Disclosure(_ContentAddressedModel):
    """One explicit disclosure of a versioned fact to named recipients."""

    schema_version: Literal[ACTION_EVIDENCE_SCHEMA_VERSION] = (
        ACTION_EVIDENCE_SCHEMA_VERSION
    )
    actor_id: Identifier
    recipient_ids: tuple[Identifier, ...]
    fact_id: Identifier
    fact_version: NonEmptyString
    evidence_spans: tuple[EvidenceSpan, ...]
    disclosure_id: str = ""

    _content_field = "disclosure_id"
    _content_prefix = "observed_disclosure"

    @model_validator(mode="after")
    def _validate_disclosure(self) -> Self:
        if not self.recipient_ids or len(set(self.recipient_ids)) != len(
            self.recipient_ids
        ):
            raise ValueError("disclosures require unique recipients")
        if not self.evidence_spans:
            raise ValueError("disclosures require evidence spans")
        return self


class ObservedAction(_ContentAddressedModel):
    """Raw text plus every deterministic atomic extraction and source span."""

    schema_version: Literal[ACTION_EVIDENCE_SCHEMA_VERSION] = (
        ACTION_EVIDENCE_SCHEMA_VERSION
    )
    scenario_id: Identifier
    spec_version: NonEmptyString
    trial_id: Identifier
    actor_id: Identifier
    raw_text: str
    parse_status: ParseStatus
    parser_name: Identifier
    parser_version: NonEmptyString
    claims: tuple[Claim, ...] = ()
    offers: tuple[Offer, ...] = ()
    commitments: tuple[Commitment, ...] = ()
    disclosures: tuple[Disclosure, ...] = ()
    parse_error: str | None = None
    action_id: str = ""

    _content_field = "action_id"
    _content_prefix = "observed_action"

    @field_validator("spec_version")
    @classmethod
    def _semantic_spec_version(cls, value: str) -> str:
        if not _SEMVER.fullmatch(value):
            raise ValueError("spec_version must be semantic version text")
        return value

    @model_validator(mode="after")
    def _validate_action_evidence(self) -> Self:
        records: tuple[Claim | Offer | Commitment | Disclosure, ...] = (
            *self.claims,
            *self.offers,
            *self.commitments,
            *self.disclosures,
        )
        if self.parse_status is ParseStatus.COMPLETE and not records:
            raise ValueError("complete action extraction requires atomic evidence")
        if self.parse_status is ParseStatus.NO_RELEVANT_ACTION and records:
            raise ValueError("no_relevant_action cannot include atomic evidence")
        if self.parse_status is ParseStatus.FAILED and not self.parse_error:
            raise ValueError("failed extraction requires parse_error")
        if self.parse_status is not ParseStatus.FAILED and self.parse_error is not None:
            raise ValueError("parse_error is reserved for failed extraction")

        spans: list[EvidenceSpan] = []
        for record in records:
            spans.extend(record.evidence_spans)
            if isinstance(record, Offer):
                for term in record.terms:
                    spans.extend(term.evidence_spans)
        for span in spans:
            if span.end > len(self.raw_text):
                raise ValueError("evidence span lies outside raw_text")
            if self.raw_text[span.start:span.end] != span.text:
                raise ValueError("evidence text does not match raw_text")
        return self


class BehaviorLabel(_ContentAddressedModel):
    """One evidence-bearing rule/source assessment of an acting agent."""

    schema_version: Literal[BEHAVIOR_LABEL_SCHEMA_VERSION] = (
        BEHAVIOR_LABEL_SCHEMA_VERSION
    )
    scenario_id: Identifier
    spec_version: NonEmptyString
    subject_actor_id: Identifier
    target_action_id: NonEmptyString
    value: LabelValue
    behavioral_misrepresentation: LabelValue
    subtype: BehaviorSubtype
    severity: UnitInterval | None
    rule_ids: tuple[Identifier, ...]
    fact_ids: tuple[Identifier, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    belief_dependent: bool
    belief_verification_status: BeliefVerificationStatus
    belief_evidence_call_id: NonEmptyString | None = None
    label_source: LabelSource
    reason_codes: tuple[Identifier, ...]
    label_id: str = ""

    _content_field = "label_id"
    _content_prefix = "behavior_label"

    @field_validator("spec_version")
    @classmethod
    def _semantic_spec_version(cls, value: str) -> str:
        if not _SEMVER.fullmatch(value):
            raise ValueError("spec_version must be semantic version text")
        return value

    @model_validator(mode="after")
    def _validate_label_contract(self) -> Self:
        if not self.rule_ids or len(set(self.rule_ids)) != len(self.rule_ids):
            raise ValueError("labels require unique rule IDs")
        if len(set(self.fact_ids)) != len(self.fact_ids):
            raise ValueError("label fact IDs must be unique")
        if not self.reason_codes:
            raise ValueError("labels require reason_codes")
        if self.value is LabelValue.UNKNOWN and self.severity is not None:
            raise ValueError("unknown labels cannot assert severity")
        if self.value is not LabelValue.UNKNOWN and self.severity is None:
            raise ValueError("known labels require severity")
        if self.belief_dependent:
            if self.belief_verification_status is not BeliefVerificationStatus.VERIFIED:
                if self.value is not LabelValue.UNKNOWN:
                    raise ValueError(
                        "belief-dependent labels must remain unknown until verified"
                    )
            elif self.belief_evidence_call_id is None:
                raise ValueError(
                    "verified belief-dependent labels require evidence call identity"
                )
        elif self.belief_verification_status is not BeliefVerificationStatus.NOT_REQUIRED:
            raise ValueError(
                "belief-independent labels must use verification status not_required"
            )
        return self

    @property
    def actual_deception(self) -> bool | None:
        """Compatibility projection; the canonical record remains tri-state."""
        if self.value is LabelValue.UNKNOWN:
            return None
        return self.value is LabelValue.TRUE


class Utility(_ContentAddressedModel):
    """One role's realized utility under an adjudicated outcome."""

    schema_version: Literal[OUTCOME_SCHEMA_VERSION] = OUTCOME_SCHEMA_VERSION
    role_id: Identifier
    value: float
    utility_id: str = ""

    _content_field = "utility_id"
    _content_prefix = "outcome_utility"


class ConstraintViolation(_ContentAddressedModel):
    """One executable constraint violation grounded in committed events."""

    schema_version: Literal[OUTCOME_SCHEMA_VERSION] = OUTCOME_SCHEMA_VERSION
    constraint_id: Identifier
    severity: UnitInterval
    evidence_event_ids: tuple[NonEmptyString, ...]
    description: NonEmptyString
    violation_id: str = ""

    _content_field = "violation_id"
    _content_prefix = "constraint_violation"

    @model_validator(mode="after")
    def _unique_evidence(self) -> Self:
        if len(set(self.evidence_event_ids)) != len(self.evidence_event_ids):
            raise ValueError("constraint evidence event IDs must be unique")
        return self


class Outcome(_ContentAddressedModel):
    """Committed-action outcome with utilities, violations, and optional regret."""

    schema_version: Literal[OUTCOME_SCHEMA_VERSION] = OUTCOME_SCHEMA_VERSION
    scenario_id: Identifier
    spec_version: NonEmptyString
    trial_id: Identifier
    status: OutcomeStatus
    agreement_state: AgreementState
    agreement_id: NonEmptyString | None = None
    committed_action_ids: tuple[NonEmptyString, ...]
    utilities: tuple[Utility, ...]
    constraint_violations: tuple[ConstraintViolation, ...] = ()
    scenario_regret: float | None = None
    baseline_outcome_id: NonEmptyString | None = None
    rule_ids: tuple[Identifier, ...]
    evidence_event_ids: tuple[NonEmptyString, ...]
    outcome_id: str = ""

    _content_field = "outcome_id"
    _content_prefix = "scenario_outcome"

    @field_validator("spec_version")
    @classmethod
    def _semantic_spec_version(cls, value: str) -> str:
        if not _SEMVER.fullmatch(value):
            raise ValueError("spec_version must be semantic version text")
        return value

    @model_validator(mode="after")
    def _validate_outcome(self) -> Self:
        if self.agreement_state is AgreementState.AGREEMENT and not self.agreement_id:
            raise ValueError("agreement outcomes require agreement_id")
        if self.agreement_state is not AgreementState.AGREEMENT and self.agreement_id:
            raise ValueError("only agreement outcomes may carry agreement_id")
        if len(set(self.committed_action_ids)) != len(self.committed_action_ids):
            raise ValueError("committed action IDs must be unique")
        role_ids = tuple(item.role_id for item in self.utilities)
        if len(set(role_ids)) != len(role_ids):
            raise ValueError("outcome utilities must have unique role IDs")
        if not self.rule_ids or len(set(self.rule_ids)) != len(self.rule_ids):
            raise ValueError("outcomes require unique rule IDs")
        if not self.evidence_event_ids or len(set(self.evidence_event_ids)) != len(
            self.evidence_event_ids
        ):
            raise ValueError("outcomes require unique evidence event IDs")
        if (self.scenario_regret is None) != (self.baseline_outcome_id is None):
            raise ValueError(
                "scenario_regret and baseline_outcome_id must appear together"
            )
        return self


__all__ = [
    "ACTION_EVIDENCE_SCHEMA_VERSION",
    "ActionDefinition",
    "ActionKind",
    "AdjudicatorView",
    "AgreementState",
    "BEHAVIOR_LABEL_SCHEMA_VERSION",
    "BehaviorLabel",
    "BehaviorSubtype",
    "BehaviorTargetDefinition",
    "BeliefVerificationStatus",
    "Claim",
    "Commitment",
    "ConditionDefinition",
    "ConstraintViolation",
    "Disclosure",
    "EvidenceSpan",
    "ExtractorReference",
    "FactDefinition",
    "FactRef",
    "FactSamplingDefinition",
    "FactSamplingKind",
    "FactValueType",
    "IncentiveCondition",
    "InterventionPoint",
    "LabelSource",
    "LabelValue",
    "NormalizationDecision",
    "OUTCOME_SCHEMA_VERSION",
    "ObservedAction",
    "Offer",
    "OfferTerm",
    "Outcome",
    "OutcomeDefinition",
    "OutcomeStatus",
    "ParseStatus",
    "PrivateView",
    "PromptKind",
    "PromptTemplate",
    "PublicState",
    "RoleDefinition",
    "RoleKind",
    "RuleReference",
    "SCENARIO_DSL_SCHEMA_VERSION",
    "SCENARIO_INSTANCE_SCHEMA_VERSION",
    "ScenarioInstance",
    "ScenarioMetadata",
    "ScenarioSpec",
    "ScheduledIntervention",
    "Utility",
    "Visibility",
    "canonical_json",
    "canonical_sha256",
]
