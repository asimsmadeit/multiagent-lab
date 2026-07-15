"""Deterministic validation and prompt-safety checks for scenario releases.

The validators in this module are deliberately data-only.  They do not own a
registry, read files, call a model, or execute source expressions.  A caller
supplies the specification, any executable-capability manifest, and the exact
batch identities to audit.  Every finding is returned as an immutable,
content-addressed report rather than being hidden behind logging or an early
exception.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
import math
import random
import re
from string import Formatter
import unicodedata
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interpretability.scenarios.compiler import (
    RenderedPrompt,
    compile_scenario,
    render_prompt,
    validate_template_placeholders,
)
from interpretability.scenarios.schema import (
    SCENARIO_DSL_SCHEMA_VERSION,
    ActionKind,
    FactDefinition,
    FactSamplingKind,
    FactValueType,
    IncentiveCondition,
    PromptKind,
    RoleKind,
    ScenarioInstance,
    ScenarioSpec,
    Visibility,
    canonical_json,
    canonical_sha256,
)


SCENARIO_VALIDATION_VERSION = "scenario-validation/1.0.0"
MIN_BATCH_INSTANCES = 100
MAX_BATCH_INSTANCES = 10_000
MAX_PROMPT_CHARACTERS = 1_000_000
MAX_SPEC_COLLECTION_ITEMS = 4_096
MAX_TOTAL_TEMPLATE_CHARACTERS = 16_000_000
MAX_FORBIDDEN_TERMS = 256
MAX_FORBIDDEN_TERM_CHARACTERS = 128

DEFAULT_FORBIDDEN_PROMPT_TERMS = (
    "betray",
    "bluff",
    "bluffing",
    "cheat",
    "deceive",
    "deception",
    "deceptive",
    "dishonest",
    "honest",
    "liar",
    "lie",
    "lying",
    "manipulate",
    "mislead",
    "misleading",
    "trick",
    "truth",
    "truthful",
)

_SEMVER = re.compile(
    r"^(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_ISSUE_CODE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_SPECIAL_TOKEN = re.compile(
    r"(?:<\|[^\r\n<>]{1,128}\|>|\[(?:/?INST|/?SYS)\]|"
    r"<<\/?SYS>>|</?(?:s|pad|unk|bos|eos)>)",
    re.IGNORECASE,
)
_UNRESOLVED_BRACE = re.compile(r"\{[^{}\r\n]{1,256}\}")
_CONTEXT_FIELDS = frozenset({"scenario_id", "trial_id", "condition"})


class ValidationSeverity(str, Enum):
    """Stable severity values used by automated release gates."""

    ERROR = "error"
    WARNING = "warning"


class _StrictFrozenModel(BaseModel):
    """Shared immutable, non-coercing policy for validation records."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class ValidationIssue(_StrictFrozenModel):
    """One stable, path-addressed validation finding."""

    schema_version: Literal[SCENARIO_VALIDATION_VERSION] = (
        SCENARIO_VALIDATION_VERSION
    )
    code: str = Field(min_length=3, max_length=128)
    severity: ValidationSeverity
    path: str = Field(min_length=1, max_length=2_048)
    message: str = Field(min_length=1, max_length=4_096)
    related_ids: tuple[str, ...] = ()
    issue_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> Self:
        if not _ISSUE_CODE.fullmatch(self.code):
            raise ValueError("validation issue code has invalid syntax")
        if not str(self.path).startswith("/"):
            raise ValueError("validation issue path must be a JSON Pointer")
        if self.related_ids != tuple(sorted(set(self.related_ids))):
            raise ValueError("related_ids must be canonically ordered and unique")
        payload = self.model_dump(mode="json", exclude={"issue_hash"})
        expected = canonical_sha256(payload)
        if self.issue_hash and self.issue_hash != expected:
            raise ValueError("issue_hash does not match canonical issue content")
        object.__setattr__(self, "issue_hash", expected)
        return self

    def canonical_json(self) -> str:
        """Return the complete canonical issue representation."""
        return canonical_json(self)


def _issue_order(issue: ValidationIssue) -> tuple[Any, ...]:
    return (
        0 if issue.severity is ValidationSeverity.ERROR else 1,
        issue.path,
        issue.code,
        issue.related_ids,
        issue.message,
        issue.issue_hash,
    )


class ValidationReport(_StrictFrozenModel):
    """Canonical aggregate containing every deterministic validation issue."""

    schema_version: Literal[SCENARIO_VALIDATION_VERSION] = (
        SCENARIO_VALIDATION_VERSION
    )
    subject_hash: str = Field(min_length=1, max_length=256)
    issues: tuple[ValidationIssue, ...] = ()
    report_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> Self:
        expected_order = tuple(sorted(self.issues, key=_issue_order))
        if self.issues != expected_order:
            raise ValueError("validation issues must use canonical order")
        issue_hashes = tuple(issue.issue_hash for issue in self.issues)
        if len(set(issue_hashes)) != len(issue_hashes):
            raise ValueError("validation report cannot contain duplicate issues")
        payload = self.model_dump(mode="json", exclude={"report_hash"})
        expected = canonical_sha256(payload)
        if self.report_hash and self.report_hash != expected:
            raise ValueError("report_hash does not match canonical report content")
        object.__setattr__(self, "report_hash", expected)
        return self

    @property
    def is_valid(self) -> bool:
        """Whether the report contains no release-blocking errors."""
        return not any(
            issue.severity is ValidationSeverity.ERROR for issue in self.issues
        )

    @property
    def ok(self) -> bool:
        """Compatibility spelling for :attr:`is_valid`."""
        return self.is_valid

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Return release-blocking findings in canonical order."""
        return tuple(
            issue
            for issue in self.issues
            if issue.severity is ValidationSeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Return advisory findings in canonical order."""
        return tuple(
            issue
            for issue in self.issues
            if issue.severity is ValidationSeverity.WARNING
        )

    def canonical_json(self) -> str:
        """Return the complete canonical report representation."""
        return canonical_json(self)

    def raise_for_errors(self) -> ValidationReport:
        """Raise :class:`ScenarioValidationError` unless this report is valid."""
        if not self.is_valid:
            raise ScenarioValidationError(self)
        return self


class ScenarioValidationError(ValueError):
    """A release gate failed with an attached immutable report."""

    def __init__(self, report: ValidationReport) -> None:
        if not isinstance(report, ValidationReport):
            raise TypeError("report must be a ValidationReport")
        self.report = report
        super().__init__(
            f"scenario validation failed with {len(report.errors)} error(s); "
            f"report={report.report_hash}"
        )


class ValidationCapabilities(_StrictFrozenModel):
    """Explicit executable identifiers available to a validation caller.

    Extractor references may be authorized by their name, content identifier,
    or ``name@version`` identity.  Predicate references may be authorized by
    predicate ID or ``predicate_id@rule_version``.  Parser IDs are kept
    separate so a released parser implementation can be audited independently
    from a scenario-level extractor alias.
    """

    extractor_ids: tuple[str, ...] = ()
    parser_ids: tuple[str, ...] = ()
    predicate_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _canonical_identifiers(self) -> Self:
        for field_name in ("extractor_ids", "parser_ids", "predicate_ids"):
            values = getattr(self, field_name)
            if values != tuple(sorted(set(values))):
                raise ValueError(
                    f"{field_name} must be canonically ordered and unique"
                )
            if any(not value or len(value) > 256 for value in values):
                raise ValueError(f"{field_name} contains an invalid identifier")
        return self

    @classmethod
    def from_iterables(
        cls,
        *,
        extractor_ids: Iterable[str] = (),
        parser_ids: Iterable[str] = (),
        predicate_ids: Iterable[str] = (),
    ) -> ValidationCapabilities:
        """Build a canonical immutable capability manifest."""
        return cls(
            extractor_ids=tuple(sorted(set(extractor_ids))),
            parser_ids=tuple(sorted(set(parser_ids))),
            predicate_ids=tuple(sorted(set(predicate_ids))),
        )


class _ReportBuilder:
    """Small mutable accumulator kept private to one validation call."""

    __slots__ = ("_issues", "subject_hash")

    def __init__(self, subject_hash: str) -> None:
        self.subject_hash = subject_hash
        self._issues: dict[str, ValidationIssue] = {}

    def add(
        self,
        code: str,
        path: str,
        message: str,
        *,
        related_ids: Iterable[str] = (),
        severity: ValidationSeverity = ValidationSeverity.ERROR,
    ) -> None:
        issue = ValidationIssue(
            code=code,
            severity=severity,
            path=path,
            message=message,
            related_ids=tuple(sorted(set(related_ids))),
        )
        self._issues.setdefault(issue.issue_hash, issue)

    def extend(self, report: ValidationReport) -> None:
        for issue in report.issues:
            self._issues.setdefault(issue.issue_hash, issue)

    def report(self) -> ValidationReport:
        return ValidationReport(
            subject_hash=self.subject_hash,
            issues=tuple(sorted(self._issues.values(), key=_issue_order)),
        )


def _subject_hash(value: Any) -> str:
    if isinstance(value, ScenarioSpec) and isinstance(value.spec_hash, str):
        if value.spec_hash:
            return value.spec_hash
    if isinstance(value, ScenarioInstance) and isinstance(
        value.instance_hash, str
    ):
        if value.instance_hash:
            return value.instance_hash
    if isinstance(value, BaseModel):
        try:
            return canonical_sha256(value.model_dump(mode="json"))
        except (TypeError, ValueError):
            pass
    return canonical_sha256({"unsupported_subject_type": type(value).__name__})


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _exact_value_type(value: Any, value_type: FactValueType) -> bool:
    expected: dict[FactValueType, type[Any]] = {
        FactValueType.INTEGER: int,
        FactValueType.FLOAT: float,
        FactValueType.STRING: str,
        FactValueType.BOOLEAN: bool,
    }
    wanted = expected[value_type]
    if type(value) is not wanted:
        return False
    return not (type(value) is float and not math.isfinite(value))


def _value_in_sampling_domain(value: Any, definition: FactDefinition) -> bool:
    if not _exact_value_type(value, definition.value_type):
        return False
    sampling = definition.sampling
    if sampling.kind is FactSamplingKind.FIXED:
        return value == sampling.fixed_value
    if sampling.kind in {
        FactSamplingKind.INTEGER_RANGE,
        FactSamplingKind.FLOAT_RANGE,
    }:
        return (
            sampling.minimum is not None
            and sampling.maximum is not None
            and sampling.minimum <= value <= sampling.maximum
        )
    if sampling.kind is FactSamplingKind.CHOICE:
        return value in sampling.choices
    return False


def _normalize_capabilities(
    capabilities: ValidationCapabilities | Iterable[str] | None,
) -> ValidationCapabilities | None:
    if capabilities is None or isinstance(capabilities, ValidationCapabilities):
        return capabilities
    if isinstance(capabilities, (str, bytes, Mapping)):
        raise TypeError(
            "capabilities must be ValidationCapabilities or an iterable of IDs"
        )
    values = tuple(capabilities)
    if any(not isinstance(value, str) for value in values):
        raise TypeError("capability identifiers must be strings")
    # A simple supplied set is intentionally accepted across all namespaces.
    return ValidationCapabilities.from_iterables(
        extractor_ids=values,
        parser_ids=values,
        predicate_ids=values,
    )


def _normalize_forbidden_terms(terms: Iterable[str]) -> tuple[str, ...]:
    if isinstance(terms, (str, bytes, Mapping)):
        raise TypeError("forbidden_terms must be an iterable of terms")
    raw = tuple(terms)
    if len(raw) > MAX_FORBIDDEN_TERMS:
        raise ValueError("forbidden_terms exceeds the configured bound")
    normalized: list[str] = []
    for term in raw:
        if not isinstance(term, str):
            raise TypeError("forbidden prompt terms must be strings")
        candidate = _normalize_text(term)
        if not candidate or len(candidate) > MAX_FORBIDDEN_TERM_CHARACTERS:
            raise ValueError("forbidden prompt term has invalid length")
        normalized.append(candidate)
    return tuple(sorted(set(normalized)))


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    # Format characters are removed so zero-width insertion cannot evade a
    # lexical release gate; separators are normalized for phrase matching.
    characters: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if category == "Cf":
            continue
        if category.startswith("Z") or character.isspace():
            characters.append(" ")
        else:
            characters.append(character)
    return " ".join("".join(characters).split())


def _forbidden_matches(
    text: str,
    normalized_terms: Sequence[str],
) -> tuple[str, ...]:
    normalized = _normalize_text(text)
    matches: list[str] = []
    for term in normalized_terms:
        pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")
        if pattern.search(normalized):
            matches.append(term)
    return tuple(matches)


def _add_prompt_text_safety(
    builder: _ReportBuilder,
    text: str,
    path: str,
    normalized_terms: Sequence[str],
    *,
    rendered: bool = True,
) -> None:
    if len(text) > MAX_PROMPT_CHARACTERS:
        builder.add(
            "prompt.too_large",
            path,
            f"prompt exceeds {MAX_PROMPT_CHARACTERS} characters",
        )
        return
    for term in _forbidden_matches(text, normalized_terms):
        builder.add(
            "prompt.forbidden_instruction_term",
            path,
            f"rendered prompt contains forbidden instruction term {term!r}",
            related_ids=(term,),
        )
    if rendered and _UNRESOLVED_BRACE.search(text):
        builder.add(
            "prompt.unresolved_placeholder",
            path,
            "prompt contains an unresolved or echoed brace placeholder",
        )
    if _SPECIAL_TOKEN.search(text):
        builder.add(
            "prompt.special_token",
            path,
            "prompt contains a reserved chat or tokenizer control token",
        )
    if any(
        unicodedata.category(character) == "Cc"
        and character not in {"\n", "\r", "\t"}
        for character in text
    ):
        builder.add(
            "prompt.control_character",
            path,
            "prompt contains a non-whitespace control character",
        )


def _validated_spec_copy(
    spec: Any,
    builder: _ReportBuilder,
) -> ScenarioSpec | None:
    if not isinstance(spec, ScenarioSpec):
        builder.add(
            "spec.invalid_type",
            "/",
            "subject must be a ScenarioSpec",
        )
        return None
    try:
        return ScenarioSpec.from_persisted_json(spec.canonical_json())
    except (TypeError, ValueError) as exc:
        builder.add(
            "spec.invalid_persisted_content",
            "/",
            f"specification fails canonical persisted validation: {exc}",
        )
        return spec


def _check_spec_input_bounds(
    spec: Any,
    builder: _ReportBuilder,
) -> bool:
    """Reject oversized typed inputs before canonical copying or graph walks."""
    if not isinstance(spec, ScenarioSpec):
        return True
    within_bounds = True
    for field_name in (
        "roles",
        "facts",
        "conditions",
        "prompt_templates",
        "intervention_points",
        "action_space",
        "extractors",
        "rules",
        "behavior_targets",
        "outcomes",
    ):
        values = getattr(spec, field_name, ())
        if not isinstance(values, tuple) or len(values) > MAX_SPEC_COLLECTION_ITEMS:
            builder.add(
                "validation.collection_bound_exceeded",
                f"/{field_name}",
                f"collection must contain at most {MAX_SPEC_COLLECTION_ITEMS} items",
            )
            within_bounds = False
    total_template_characters = sum(
        len(template.template)
        for template in getattr(spec, "prompt_templates", ())
        if hasattr(template, "template") and isinstance(template.template, str)
    )
    if total_template_characters > MAX_TOTAL_TEMPLATE_CHARACTERS:
        builder.add(
            "validation.template_budget_exceeded",
            "/prompt_templates",
            "combined prompt-template text exceeds the validation budget",
        )
        within_bounds = False
    return within_bounds


def _check_released_id(
    builder: _ReportBuilder,
    value: Any,
    path: str,
    label: str,
) -> None:
    if not isinstance(value, str) or not value:
        builder.add(
            "release.missing_content_id",
            path,
            f"released {label} requires its canonical content identifier",
        )


def _check_sampling(
    builder: _ReportBuilder,
    definition: FactDefinition,
    index: int,
) -> None:
    path = f"/facts/{index}/sampling"
    sampling = definition.sampling
    if sampling.kind is FactSamplingKind.FIXED:
        if not _exact_value_type(sampling.fixed_value, definition.value_type):
            builder.add(
                "sampling.fixed_type_mismatch",
                f"{path}/fixed_value",
                "fixed value does not use the fact's exact declared type",
                related_ids=(definition.fact_id,),
            )
    elif sampling.kind in {
        FactSamplingKind.INTEGER_RANGE,
        FactSamplingKind.FLOAT_RANGE,
    }:
        expected_type = (
            int
            if sampling.kind is FactSamplingKind.INTEGER_RANGE
            else float
        )
        if type(sampling.minimum) is not expected_type or type(
            sampling.maximum
        ) is not expected_type:
            builder.add(
                "sampling.range_type_mismatch",
                path,
                "range bounds do not use the exact sampling kind type",
                related_ids=(definition.fact_id,),
            )
        elif (
            not math.isfinite(sampling.minimum)
            or not math.isfinite(sampling.maximum)
        ):
            builder.add(
                "sampling.nonfinite_bound",
                path,
                "range bounds must be finite",
                related_ids=(definition.fact_id,),
            )
        elif sampling.minimum > sampling.maximum:
            builder.add(
                "sampling.reversed_bounds",
                path,
                "sampling minimum exceeds maximum",
                related_ids=(definition.fact_id,),
            )
        wanted_value_type = (
            FactValueType.INTEGER
            if sampling.kind is FactSamplingKind.INTEGER_RANGE
            else FactValueType.FLOAT
        )
        if definition.value_type is not wanted_value_type:
            builder.add(
                "sampling.fact_type_mismatch",
                f"/facts/{index}/value_type",
                "fact type is incompatible with its range sampling kind",
                related_ids=(definition.fact_id,),
            )
    elif sampling.kind is FactSamplingKind.CHOICE:
        if not sampling.choices:
            builder.add(
                "sampling.empty_choices",
                f"{path}/choices",
                "choice sampling requires at least one value",
                related_ids=(definition.fact_id,),
            )
        for choice_index, choice in enumerate(sampling.choices):
            if not _exact_value_type(choice, definition.value_type):
                builder.add(
                    "sampling.choice_type_mismatch",
                    f"{path}/choices/{choice_index}",
                    "choice does not use the fact's exact declared type",
                    related_ids=(definition.fact_id,),
                )
    else:
        builder.add(
            "sampling.unknown_kind",
            f"{path}/kind",
            "sampling kind is not supported by the deterministic compiler",
            related_ids=(definition.fact_id,),
        )


def _check_spec_graph(
    spec: ScenarioSpec,
    builder: _ReportBuilder,
    capabilities: ValidationCapabilities | None,
    normalized_terms: Sequence[str],
) -> None:
    if spec.schema_version != SCENARIO_DSL_SCHEMA_VERSION:
        builder.add(
            "release.unsupported_schema_version",
            "/schema_version",
            "specification does not use the released scenario DSL version",
            related_ids=(str(spec.schema_version),),
        )
    if not isinstance(spec.spec_version, str) or not _SEMVER.fullmatch(
        spec.spec_version
    ):
        builder.add(
            "release.invalid_semantic_version",
            "/spec_version",
            "spec_version must be an exact semantic version",
        )
    _check_released_id(builder, spec.spec_hash, "/spec_hash", "specification")
    _check_released_id(
        builder,
        spec.metadata.metadata_id,
        "/metadata/metadata_id",
        "metadata",
    )
    if spec.metadata.schema_version != SCENARIO_DSL_SCHEMA_VERSION:
        builder.add(
            "release.nested_schema_version",
            "/metadata/schema_version",
            "metadata does not use the released DSL schema version",
            related_ids=(spec.metadata.scenario_id,),
        )

    collections: tuple[tuple[str, tuple[Any, ...], str, str], ...] = (
        ("roles", spec.roles, "role_id", "role_definition_id"),
        ("facts", spec.facts, "fact_id", "fact_definition_id"),
        ("conditions", spec.conditions, "condition", "condition_definition_id"),
        (
            "prompt_templates",
            spec.prompt_templates,
            "template_id",
            "prompt_template_hash",
        ),
        (
            "intervention_points",
            spec.intervention_points,
            "intervention_id",
            "intervention_point_id",
        ),
        ("action_space", spec.action_space, "action_id", "action_definition_id"),
        ("extractors", spec.extractors, "extractor_name", "extractor_ref_id"),
        ("rules", spec.rules, "rule_id", "rule_reference_id"),
        (
            "behavior_targets",
            spec.behavior_targets,
            "target_id",
            "behavior_target_id",
        ),
        ("outcomes", spec.outcomes, "outcome_id", "outcome_definition_id"),
    )
    for collection_name, collection, identity_field, content_field in collections:
        if not collection and collection_name not in {"intervention_points"}:
            builder.add(
                "graph.empty_collection",
                f"/{collection_name}",
                f"{collection_name} must not be empty",
            )
        seen: dict[str, int] = {}
        for index, item in enumerate(collection):
            identity = getattr(item, identity_field, None)
            identity_text = (
                identity.value if isinstance(identity, Enum) else identity
            )
            if identity_text in seen:
                builder.add(
                    "graph.duplicate_id",
                    f"/{collection_name}/{index}/{identity_field}",
                    f"duplicate {collection_name} identifier",
                    related_ids=(str(identity_text),),
                )
            else:
                seen[str(identity_text)] = index
            _check_released_id(
                builder,
                getattr(item, content_field, None),
                f"/{collection_name}/{index}/{content_field}",
                collection_name,
            )
            if getattr(item, "schema_version", None) != SCENARIO_DSL_SCHEMA_VERSION:
                builder.add(
                    "release.nested_schema_version",
                    f"/{collection_name}/{index}/schema_version",
                    "nested record does not use the released DSL schema version",
                    related_ids=(str(identity_text),),
                )

    role_by_id = {item.role_id: item for item in spec.roles}
    fact_by_id = {item.fact_id: item for item in spec.facts}
    condition_ids = {item.condition for item in spec.conditions}
    template_by_id = {item.template_id: item for item in spec.prompt_templates}
    rule_ids = {item.rule_id for item in spec.rules}
    participant_roles = {
        role.role_id
        for role in spec.roles
        if role.kind is not RoleKind.ADJUDICATOR
    }
    actor_roles = {
        role.role_id for role in spec.roles if role.kind is RoleKind.ACTOR
    }
    counterpart_roles = {
        role.role_id for role in spec.roles if role.kind is RoleKind.COUNTERPART
    }
    if not actor_roles:
        builder.add("graph.missing_actor", "/roles", "scenario has no actor role")
    if not counterpart_roles:
        builder.add(
            "graph.missing_counterpart",
            "/roles",
            "scenario has no counterpart role",
        )

    reachable_roles: set[str] = set()
    reachable_facts: set[str] = set()
    reachable_rules: set[str] = set()
    referenced_intervention_templates: set[str] = set()

    for index, fact in enumerate(spec.facts):
        _check_released_id(
            builder,
            fact.sampling.sampling_id,
            f"/facts/{index}/sampling/sampling_id",
            "fact sampling definition",
        )
        if fact.sampling.schema_version != SCENARIO_DSL_SCHEMA_VERSION:
            builder.add(
                "release.nested_schema_version",
                f"/facts/{index}/sampling/schema_version",
                "fact sampling does not use the released DSL schema version",
                related_ids=(fact.fact_id,),
            )
        _check_sampling(builder, fact, index)
        if fact.fact_id in _CONTEXT_FIELDS:
            builder.add(
                "fact.reserved_prompt_context_id",
                f"/facts/{index}/fact_id",
                "fact ID collides with a compiler-owned prompt context field",
                related_ids=(fact.fact_id,),
            )
        if fact.visibility is Visibility.ROLE_PRIVATE:
            for role_id in fact.visible_to:
                role = role_by_id.get(role_id)
                if role is None:
                    builder.add(
                        "visibility.unknown_role",
                        f"/facts/{index}/visible_to",
                        "private fact names an unknown role",
                        related_ids=(fact.fact_id, role_id),
                    )
                elif role.kind is RoleKind.ADJUDICATOR:
                    builder.add(
                        "visibility.adjudicator_private_target",
                        f"/facts/{index}/visible_to",
                        "role-private facts cannot target an adjudicator role",
                        related_ids=(fact.fact_id, role_id),
                    )
                else:
                    reachable_roles.add(role_id)
        elif fact.visible_to:
            builder.add(
                "visibility.illegal_visible_to",
                f"/facts/{index}/visible_to",
                "only role-private facts may name visible roles",
                related_ids=(fact.fact_id,),
            )

    template_keys: dict[tuple[str, IncentiveCondition, PromptKind], int] = {}
    for index, template in enumerate(spec.prompt_templates):
        path = f"/prompt_templates/{index}"
        role = role_by_id.get(template.role_id)
        if role is None:
            builder.add(
                "template.unknown_role",
                f"{path}/role_id",
                "prompt template names an unknown role",
                related_ids=(template.template_id, template.role_id),
            )
        else:
            reachable_roles.add(template.role_id)
        if template.condition not in condition_ids:
            builder.add(
                "template.unknown_condition",
                f"{path}/condition",
                "prompt template names an undeclared condition",
                related_ids=(template.template_id, template.condition.value),
            )
        key = (template.role_id, template.condition, template.kind)
        if key in template_keys:
            builder.add(
                "template.ambiguous_selector",
                path,
                "multiple templates share role, condition, and kind",
                related_ids=(
                    template.template_id,
                    spec.prompt_templates[template_keys[key]].template_id,
                ),
            )
        else:
            template_keys[key] = index
        if len(template.template) > MAX_PROMPT_CHARACTERS:
            builder.add(
                "template.too_large",
                f"{path}/template",
                f"template exceeds {MAX_PROMPT_CHARACTERS} characters",
                related_ids=(template.template_id,),
            )
            continue
        _add_prompt_text_safety(
            builder,
            template.template,
            f"{path}/template",
            normalized_terms,
            rendered=False,
        )
        try:
            placeholders = validate_template_placeholders(template.template)
        except (TypeError, ValueError) as exc:
            builder.add(
                "template.invalid_placeholder",
                f"{path}/template",
                f"template placeholder syntax is invalid: {exc}",
                related_ids=(template.template_id,),
            )
            placeholders = ()
        for placeholder in placeholders:
            if placeholder in _CONTEXT_FIELDS:
                continue
            definition = fact_by_id.get(placeholder)
            if definition is None:
                builder.add(
                    "template.unknown_fact",
                    f"{path}/template",
                    "template placeholder names an unknown fact",
                    related_ids=(template.template_id, placeholder),
                )
                continue
            reachable_facts.add(placeholder)
            authorized = definition.visibility is Visibility.PUBLIC or (
                definition.visibility is Visibility.ROLE_PRIVATE
                and template.role_id in definition.visible_to
            )
            if role is not None and role.kind is RoleKind.ADJUDICATOR:
                authorized = True
            if not authorized:
                builder.add(
                    "template.unauthorized_fact",
                    f"{path}/template",
                    "template requests a fact unavailable to its recipient role",
                    related_ids=(template.template_id, placeholder),
                )

    for role_id in sorted(participant_roles):
        role = role_by_id[role_id]
        required_kind = (
            PromptKind.INITIAL
            if role.kind is RoleKind.ACTOR
            else PromptKind.COUNTERPART
        )
        for condition in sorted(condition_ids, key=lambda item: item.value):
            if (role_id, condition, required_kind) not in template_keys:
                builder.add(
                    "template.unreachable_condition_role",
                    "/prompt_templates",
                    "participant role and condition lack their executable prompt",
                    related_ids=(role_id, condition.value, required_kind.value),
                )

    for index, intervention in enumerate(spec.intervention_points):
        path = f"/intervention_points/{index}"
        role = role_by_id.get(intervention.recipient_role_id)
        if role is None:
            builder.add(
                "intervention.unknown_recipient",
                f"{path}/recipient_role_id",
                "intervention names an unknown recipient role",
                related_ids=(
                    intervention.intervention_id,
                    intervention.recipient_role_id,
                ),
            )
        elif role.kind is RoleKind.ADJUDICATOR:
            builder.add(
                "intervention.illegal_recipient",
                f"{path}/recipient_role_id",
                "runtime intervention recipient must be a participant role",
                related_ids=(intervention.intervention_id, role.role_id),
            )
        else:
            reachable_roles.add(role.role_id)
        if type(intervention.round_index) is not int or intervention.round_index < 0:
            builder.add(
                "intervention.invalid_round",
                f"{path}/round_index",
                "intervention round must be a nonnegative exact integer",
                related_ids=(intervention.intervention_id,),
            )
        template = template_by_id.get(intervention.template_id)
        if template is None:
            builder.add(
                "intervention.unknown_template",
                f"{path}/template_id",
                "intervention names an unknown prompt template",
                related_ids=(
                    intervention.intervention_id,
                    intervention.template_id,
                ),
            )
        else:
            referenced_intervention_templates.add(template.template_id)
            if template.kind is not PromptKind.INTERVENTION:
                builder.add(
                    "intervention.wrong_template_kind",
                    f"{path}/template_id",
                    "intervention must reference an intervention prompt template",
                    related_ids=(intervention.intervention_id, template.template_id),
                )
            if template.role_id != intervention.recipient_role_id:
                builder.add(
                    "intervention.recipient_template_mismatch",
                    f"{path}/template_id",
                    "intervention template role differs from its recipient",
                    related_ids=(intervention.intervention_id, template.template_id),
                )

    for index, template in enumerate(spec.prompt_templates):
        if (
            template.kind is PromptKind.INTERVENTION
            and template.template_id not in referenced_intervention_templates
        ):
            builder.add(
                "template.unreachable_intervention",
                f"/prompt_templates/{index}",
                "intervention prompt is not referenced by an intervention point",
                related_ids=(template.template_id,),
            )

    declared_action_kinds = {action.kind for action in spec.action_space}
    supported_action_kinds: set[ActionKind] = set()
    for index, extractor in enumerate(spec.extractors):
        path = f"/extractors/{index}"
        if not extractor.deterministic:
            builder.add(
                "extractor.nondeterministic",
                f"{path}/deterministic",
                "released ground-truth extraction must be deterministic",
                related_ids=(extractor.extractor_name,),
            )
        supported_action_kinds.update(extractor.supported_action_kinds)
        for kind in extractor.supported_action_kinds:
            if kind not in declared_action_kinds:
                builder.add(
                    "extractor.undeclared_action_kind",
                    f"{path}/supported_action_kinds",
                    "extractor supports an action kind absent from the action space",
                    related_ids=(extractor.extractor_name, kind.value),
                )
        if capabilities is not None:
            candidates = {
                extractor.extractor_name,
                extractor.extractor_ref_id,
                f"{extractor.extractor_name}@{extractor.extractor_version}",
            }
            available = set(capabilities.extractor_ids).union(
                capabilities.parser_ids
            )
            if candidates.isdisjoint(available):
                builder.add(
                    "extractor.unknown_capability",
                    path,
                    "extractor/parser reference is absent from supplied capabilities",
                    related_ids=tuple(sorted(candidates)),
                )

    for index, action in enumerate(spec.action_space):
        for role_id in action.actor_role_ids:
            role = role_by_id.get(role_id)
            if role is None:
                builder.add(
                    "action.unknown_role",
                    f"/action_space/{index}/actor_role_ids",
                    "action names an unknown actor role",
                    related_ids=(action.action_id, role_id),
                )
            elif role.kind is RoleKind.ADJUDICATOR:
                builder.add(
                    "action.illegal_adjudicator_actor",
                    f"/action_space/{index}/actor_role_ids",
                    "adjudicator roles cannot emit participant actions",
                    related_ids=(action.action_id, role_id),
                )
            else:
                reachable_roles.add(role_id)
        if action.kind not in supported_action_kinds:
            builder.add(
                "action.unextractable_kind",
                f"/action_space/{index}/kind",
                "no declared deterministic extractor supports this action kind",
                related_ids=(action.action_id, action.kind.value),
            )

    for index, rule in enumerate(spec.rules):
        for fact_id in rule.input_fact_ids:
            if fact_id not in fact_by_id:
                builder.add(
                    "rule.unknown_fact",
                    f"/rules/{index}/input_fact_ids",
                    "rule names an unknown input fact",
                    related_ids=(rule.rule_id, fact_id),
                )
            else:
                reachable_facts.add(fact_id)
        if capabilities is not None:
            candidates = {
                rule.predicate_id,
                f"{rule.predicate_id}@{rule.rule_version}",
            }
            if candidates.isdisjoint(capabilities.predicate_ids):
                builder.add(
                    "rule.unknown_predicate_capability",
                    f"/rules/{index}/predicate_id",
                    "predicate reference is absent from supplied capabilities",
                    related_ids=tuple(sorted(candidates.union({rule.rule_id}))),
                )

    for index, target in enumerate(spec.behavior_targets):
        for rule_id in target.rule_ids:
            if rule_id not in rule_ids:
                builder.add(
                    "target.unknown_rule",
                    f"/behavior_targets/{index}/rule_ids",
                    "behavior target names an unknown rule",
                    related_ids=(target.target_id, rule_id),
                )
            else:
                reachable_rules.add(rule_id)

    declared_constructs = set(spec.metadata.research_constructs)
    target_constructs = {target.subtype for target in spec.behavior_targets}
    for construct in sorted(
        declared_constructs.difference(target_constructs),
        key=lambda item: item.value,
    ):
        builder.add(
            "metadata.construct_without_target",
            "/metadata/research_constructs",
            "declared research construct has no behavior target",
            related_ids=(construct.value,),
        )
    for construct in sorted(
        target_constructs.difference(declared_constructs),
        key=lambda item: item.value,
    ):
        builder.add(
            "target.undeclared_research_construct",
            "/behavior_targets",
            "behavior target subtype is absent from metadata constructs",
            related_ids=(construct.value,),
        )

    for index, outcome in enumerate(spec.outcomes):
        for rule_id in outcome.rule_ids:
            if rule_id not in rule_ids:
                builder.add(
                    "outcome.unknown_rule",
                    f"/outcomes/{index}/rule_ids",
                    "outcome names an unknown rule",
                    related_ids=(outcome.outcome_id, rule_id),
                )
            else:
                reachable_rules.add(rule_id)
        if outcome.regret_baseline_rule_id is not None:
            if outcome.regret_baseline_rule_id not in rule_ids:
                builder.add(
                    "outcome.unknown_regret_rule",
                    f"/outcomes/{index}/regret_baseline_rule_id",
                    "outcome names an unknown regret-baseline rule",
                    related_ids=(
                        outcome.outcome_id,
                        outcome.regret_baseline_rule_id,
                    ),
                )
            else:
                reachable_rules.add(outcome.regret_baseline_rule_id)
        for role_id in outcome.utility_role_ids:
            role = role_by_id.get(role_id)
            if role is None:
                builder.add(
                    "outcome.unknown_utility_role",
                    f"/outcomes/{index}/utility_role_ids",
                    "outcome names an unknown utility role",
                    related_ids=(outcome.outcome_id, role_id),
                )
            else:
                reachable_roles.add(role_id)
                if role.kind is RoleKind.ADJUDICATOR:
                    builder.add(
                        "outcome.illegal_adjudicator_utility",
                        f"/outcomes/{index}/utility_role_ids",
                        "outcome utility must belong to a participant role",
                        related_ids=(outcome.outcome_id, role_id),
                    )

    for index, fact in enumerate(spec.facts):
        if fact.fact_id not in reachable_facts:
            builder.add(
                "graph.unreachable_fact",
                f"/facts/{index}",
                "fact is unused by every prompt and rule",
                related_ids=(fact.fact_id,),
            )
    for index, rule in enumerate(spec.rules):
        if rule.rule_id not in reachable_rules:
            builder.add(
                "graph.unreachable_rule",
                f"/rules/{index}",
                "rule is unused by every behavior target and outcome",
                related_ids=(rule.rule_id,),
            )
    for index, role in enumerate(spec.roles):
        if (
            role.kind is not RoleKind.ADJUDICATOR
            and role.role_id not in reachable_roles
        ):
            builder.add(
                "graph.unreachable_role",
                f"/roles/{index}",
                "participant role is unreachable from prompts and execution",
                related_ids=(role.role_id,),
            )


def validate_spec(
    spec: ScenarioSpec,
    capabilities: ValidationCapabilities | Iterable[str] | None = None,
    *,
    forbidden_terms: Iterable[str] = DEFAULT_FORBIDDEN_PROMPT_TERMS,
) -> ValidationReport:
    """Statically validate a released specification and its reference graph."""
    builder = _ReportBuilder(_subject_hash(spec))
    try:
        normalized_capabilities = _normalize_capabilities(capabilities)
    except (TypeError, ValueError) as exc:
        builder.add(
            "validation.invalid_capabilities",
            "/",
            f"capability manifest is invalid: {exc}",
        )
        normalized_capabilities = None
    try:
        normalized_terms = _normalize_forbidden_terms(forbidden_terms)
    except (TypeError, ValueError) as exc:
        builder.add(
            "validation.invalid_forbidden_terms",
            "/",
            f"forbidden-term configuration is invalid: {exc}",
        )
        normalized_terms = ()
    within_bounds = _check_spec_input_bounds(spec, builder)
    validated = _validated_spec_copy(spec, builder) if within_bounds else None
    if validated is not None:
        try:
            _check_spec_graph(
                validated,
                builder,
                normalized_capabilities,
                normalized_terms,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            builder.add(
                "spec.malformed_graph",
                "/",
                f"specification graph could not be fully inspected: {exc}",
            )
    return builder.report()


def _validated_instance_copy(
    instance: Any,
    builder: _ReportBuilder,
) -> ScenarioInstance | None:
    if not isinstance(instance, ScenarioInstance):
        builder.add(
            "instance.invalid_type",
            "/",
            "subject must be a ScenarioInstance",
        )
        return None
    try:
        return ScenarioInstance.from_persisted_json(instance.canonical_json())
    except (TypeError, ValueError) as exc:
        builder.add(
            "instance.invalid_persisted_content",
            "/",
            f"instance fails canonical persisted validation: {exc}",
        )
        return instance


def _fact_map(values: Iterable[Any]) -> dict[str, Any]:
    return {
        fact.fact_id: fact
        for fact in values
        if hasattr(fact, "fact_id") and isinstance(fact.fact_id, str)
    }


def _check_instance_core(
    spec: ScenarioSpec,
    instance: ScenarioInstance,
    builder: _ReportBuilder,
) -> None:
    if instance.scenario_id != spec.metadata.scenario_id:
        builder.add(
            "instance.scenario_mismatch",
            "/scenario_id",
            "instance scenario identity differs from the specification",
            related_ids=(instance.scenario_id, spec.metadata.scenario_id),
        )
    if instance.spec_version != spec.spec_version:
        builder.add(
            "instance.version_mismatch",
            "/spec_version",
            "instance version differs from the specification",
            related_ids=(instance.spec_version, spec.spec_version),
        )
    if instance.spec_hash != spec.spec_hash:
        builder.add(
            "instance.spec_hash_mismatch",
            "/spec_hash",
            "instance does not cite the exact specification content",
            related_ids=(instance.spec_hash, spec.spec_hash),
        )
    if instance.condition not in {item.condition for item in spec.conditions}:
        builder.add(
            "instance.undeclared_condition",
            "/condition",
            "instance uses an undeclared incentive condition",
            related_ids=(instance.condition.value,),
        )

    expected_definitions = {fact.fact_id: fact for fact in spec.facts}
    actual_facts = _fact_map(instance.resolved_facts)
    expected_ids = tuple(fact.fact_id for fact in spec.facts)
    actual_ids = tuple(fact.fact_id for fact in instance.resolved_facts)
    if actual_ids != expected_ids:
        builder.add(
            "instance.fact_set_mismatch",
            "/resolved_facts",
            "resolved fact IDs/order differ from the specification",
            related_ids=tuple(sorted(set(expected_ids).symmetric_difference(actual_ids))),
        )
    for index, fact in enumerate(instance.resolved_facts):
        definition = expected_definitions.get(fact.fact_id)
        path = f"/resolved_facts/{index}"
        if definition is None:
            builder.add(
                "instance.extra_fact",
                path,
                "instance contains a fact absent from the specification",
                related_ids=(fact.fact_id,),
            )
            continue
        if fact.fact_version != definition.fact_version:
            builder.add(
                "instance.fact_version_mismatch",
                f"{path}/fact_version",
                "resolved fact version differs from its definition",
                related_ids=(fact.fact_id,),
            )
        if (
            fact.visibility is not definition.visibility
            or fact.visible_to != definition.visible_to
        ):
            builder.add(
                "instance.fact_visibility_mismatch",
                path,
                "resolved fact visibility differs from its definition",
                related_ids=(fact.fact_id,),
            )
        if not _value_in_sampling_domain(fact.value, definition):
            builder.add(
                "instance.fact_outside_sampling_domain",
                f"{path}/value",
                "resolved fact value is outside its declared sampling domain",
                related_ids=(fact.fact_id,),
            )

    participant_roles = tuple(
        role.role_id
        for role in spec.roles
        if role.kind is not RoleKind.ADJUDICATOR
    )
    view_roles = tuple(view.role_id for view in instance.private_views)
    if view_roles != participant_roles:
        builder.add(
            "instance.private_view_roles_mismatch",
            "/private_views",
            "private-view roles/order differ from participant roles",
            related_ids=tuple(sorted(set(view_roles).symmetric_difference(participant_roles))),
        )
    public_expected = {
        fact_id: fact
        for fact_id, fact in actual_facts.items()
        if fact.visibility is Visibility.PUBLIC
    }
    public_actual = _fact_map(instance.public_state.facts)
    if public_actual != public_expected:
        differing = {
            fact_id
            for fact_id in set(public_actual).union(public_expected)
            if public_actual.get(fact_id) != public_expected.get(fact_id)
        }
        builder.add(
            "instance.public_partition_mismatch",
            "/public_state/facts",
            "public state has missing, extra, or non-public facts",
            related_ids=tuple(sorted(differing)),
        )
    for view_index, view in enumerate(instance.private_views):
        expected = {
            fact_id: fact
            for fact_id, fact in actual_facts.items()
            if fact.visibility is Visibility.PUBLIC
            or (
                fact.visibility is Visibility.ROLE_PRIVATE
                and view.role_id in fact.visible_to
            )
        }
        actual = _fact_map(view.facts)
        if actual != expected:
            differing = {
                fact_id
                for fact_id in set(actual).union(expected)
                if actual.get(fact_id) != expected.get(fact_id)
            }
            builder.add(
                "instance.private_partition_mismatch",
                f"/private_views/{view_index}/facts",
                "role view has missing, extra, or unauthorized private facts",
                related_ids=tuple(sorted(differing.union({view.role_id}))),
            )
    adjudicator_actual = _fact_map(instance.adjudicator_view.facts)
    adjudicator_expected = actual_facts
    if adjudicator_actual != adjudicator_expected:
        differing = {
            fact_id
            for fact_id in set(adjudicator_actual).union(adjudicator_expected)
            if adjudicator_actual.get(fact_id)
            != adjudicator_expected.get(fact_id)
        }
        builder.add(
            "instance.adjudicator_partition_mismatch",
            "/adjudicator_view/facts",
            "adjudicator view must contain the complete resolved fact set",
            related_ids=tuple(sorted(differing)),
        )

    expected_actions = tuple(action.action_id for action in spec.action_space)
    if instance.legal_action_ids != expected_actions:
        builder.add(
            "instance.legal_actions_mismatch",
            "/legal_action_ids",
            "instance legal actions differ from the specification",
            related_ids=tuple(
                sorted(set(instance.legal_action_ids).symmetric_difference(expected_actions))
            ),
        )
    expected_interventions = tuple(
        (
            point.intervention_id,
            point.round_index,
            point.recipient_role_id,
            point.template_id,
        )
        for point in spec.intervention_points
    )
    actual_interventions = tuple(
        (
            point.intervention_id,
            point.round_index,
            point.recipient_role_id,
            point.template_id,
        )
        for point in instance.interventions
    )
    if actual_interventions != expected_interventions:
        builder.add(
            "instance.intervention_provenance_mismatch",
            "/interventions",
            "scheduled interventions differ from their specification points",
        )

    if (
        instance.scenario_id == spec.metadata.scenario_id
        and instance.spec_version == spec.spec_version
        and instance.spec_hash == spec.spec_hash
        and instance.condition in {item.condition for item in spec.conditions}
        and type(instance.run_seed) is int
        and isinstance(instance.trial_id, str)
    ):
        try:
            expected_instance = compile_scenario(
                spec,
                instance.trial_id,
                instance.run_seed,
                instance.condition,
            )
        except (TypeError, ValueError, PermissionError) as exc:
            builder.add(
                "instance.recompilation_failed",
                "/",
                f"exact instance identity could not be recompiled: {exc}",
            )
        else:
            expected_values = {
                fact.fact_id: fact.value for fact in expected_instance.resolved_facts
            }
            for index, fact in enumerate(instance.resolved_facts):
                if (
                    fact.fact_id in expected_values
                    and fact.value != expected_values[fact.fact_id]
                ):
                    builder.add(
                        "instance.deterministic_fact_mismatch",
                        f"/resolved_facts/{index}/value",
                        "fact value differs from exact deterministic recompilation",
                        related_ids=(fact.fact_id,),
                    )
            if instance.canonical_json() != expected_instance.canonical_json():
                builder.add(
                    "instance.compilation_mismatch",
                    "/",
                    "instance is not byte-identical to exact deterministic compilation",
                    related_ids=(instance.instance_hash, expected_instance.instance_hash),
                )


def _render_scalar(value: Any) -> str | None:
    if type(value) is str:
        return value
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float and math.isfinite(value):
        return canonical_json(value)
    return None


def _text_contains_value(text: str, rendered_value: str) -> bool:
    if not rendered_value:
        return False
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", rendered_value):
        return re.search(
            r"(?<![\w.])" + re.escape(rendered_value) + r"(?![\w.])",
            text,
        ) is not None
    if len(rendered_value) < 4:
        return False
    return _normalize_text(rendered_value) in _normalize_text(text)


def _check_private_value_leakage(
    spec: ScenarioSpec,
    instance: ScenarioInstance,
    rendered: RenderedPrompt,
    builder: _ReportBuilder,
    path: str,
) -> None:
    facts = _fact_map(instance.resolved_facts)
    authorized_ids = set(rendered.visible_fact_ids)
    authorized_values = {
        _render_scalar(facts[fact_id].value)
        for fact_id in authorized_ids
        if fact_id in facts
    }
    for fact_id, fact in facts.items():
        if fact_id in authorized_ids or fact.visibility is Visibility.PUBLIC:
            continue
        rendered_value = _render_scalar(fact.value)
        if (
            rendered_value is None
            or rendered_value in authorized_values
            or rendered_value == str(instance.run_seed)
        ):
            continue
        if _text_contains_value(rendered.text, rendered_value):
            builder.add(
                "prompt.private_fact_leak",
                path,
                "rendered prompt contains a fact value unauthorized for its role",
                related_ids=(fact_id, rendered.role_id, rendered.template_id),
            )


def _check_rendered_prompts_core(
    spec: ScenarioSpec,
    instance: ScenarioInstance,
    builder: _ReportBuilder,
    normalized_terms: Sequence[str],
) -> None:
    templates = tuple(
        template
        for template in spec.prompt_templates
        if template.condition is instance.condition
    )
    for index, template in enumerate(templates):
        path = (
            "/rendered_prompts/"
            + _pointer_token(template.template_id)
            + f"/{index}"
        )
        try:
            first = render_prompt(
                spec,
                instance,
                template.role_id,
                template.kind,
            )
            second = render_prompt(
                spec,
                instance,
                template.role_id,
                template.kind,
            )
        except (TypeError, ValueError, PermissionError) as exc:
            builder.add(
                "prompt.render_failed",
                path,
                f"prompt cannot be rendered from the compiled instance: {exc}",
                related_ids=(template.template_id,),
            )
            continue
        if first.canonical_json() != second.canonical_json():
            builder.add(
                "prompt.render_nondeterministic",
                path,
                "repeated rendering produced different canonical output",
                related_ids=(template.template_id,),
            )
        if (
            first.template_id != template.template_id
            or first.template_version != template.template_version
            or first.template_hash != template.prompt_template_hash
            or first.instance_hash != instance.instance_hash
            or first.role_id != template.role_id
            or first.kind is not template.kind
            or first.condition is not instance.condition
        ):
            builder.add(
                "prompt.provenance_mismatch",
                path,
                "rendered prompt provenance differs from template and instance",
                related_ids=(template.template_id,),
            )
        try:
            placeholders = validate_template_placeholders(template.template)
        except (TypeError, ValueError):
            placeholders = ()
        expected_used = tuple(
            sorted(
                set(placeholders).intersection(
                    fact.fact_id for fact in instance.resolved_facts
                )
            )
        )
        if first.used_fact_ids != expected_used:
            builder.add(
                "prompt.fact_provenance_mismatch",
                path,
                "rendered prompt fact lineage differs from its placeholders",
                related_ids=(template.template_id,),
            )
        _add_prompt_text_safety(
            builder,
            first.text,
            f"{path}/text",
            normalized_terms,
        )
        _check_private_value_leakage(
            spec,
            instance,
            first,
            builder,
            f"{path}/text",
        )


def validate_rendered_prompts(
    spec: ScenarioSpec,
    instance: ScenarioInstance,
    *,
    forbidden_terms: Iterable[str] = DEFAULT_FORBIDDEN_PROMPT_TERMS,
) -> ValidationReport:
    """Render every applicable role prompt and audit safety and provenance."""
    builder = _ReportBuilder(_subject_hash(instance))
    builder.extend(validate_spec(spec, forbidden_terms=forbidden_terms))
    validated_spec = (
        _validated_spec_copy(spec, builder)
        if _check_spec_input_bounds(spec, builder)
        else None
    )
    validated_instance = _validated_instance_copy(instance, builder)
    try:
        normalized_terms = _normalize_forbidden_terms(forbidden_terms)
    except (TypeError, ValueError) as exc:
        builder.add(
            "validation.invalid_forbidden_terms",
            "/",
            f"forbidden-term configuration is invalid: {exc}",
        )
        normalized_terms = ()
    if validated_spec is not None and validated_instance is not None:
        _check_instance_core(validated_spec, validated_instance, builder)
        _check_rendered_prompts_core(
            validated_spec,
            validated_instance,
            builder,
            normalized_terms,
        )
    return builder.report()


def validate_instance(
    spec: ScenarioSpec,
    instance: ScenarioInstance,
    *,
    capabilities: ValidationCapabilities | Iterable[str] | None = None,
    forbidden_terms: Iterable[str] = DEFAULT_FORBIDDEN_PROMPT_TERMS,
) -> ValidationReport:
    """Validate exact compilation, views, provenance, and rendered prompts."""
    builder = _ReportBuilder(_subject_hash(instance))
    builder.extend(
        validate_spec(
            spec,
            capabilities,
            forbidden_terms=forbidden_terms,
        )
    )
    validated_spec = (
        _validated_spec_copy(spec, builder)
        if _check_spec_input_bounds(spec, builder)
        else None
    )
    validated_instance = _validated_instance_copy(instance, builder)
    try:
        normalized_terms = _normalize_forbidden_terms(forbidden_terms)
    except (TypeError, ValueError):
        normalized_terms = ()
    if validated_spec is not None and validated_instance is not None:
        _check_instance_core(validated_spec, validated_instance, builder)
        _check_rendered_prompts_core(
            validated_spec,
            validated_instance,
            builder,
            normalized_terms,
        )
    return builder.report()


def _validate_batch_inputs(
    builder: _ReportBuilder,
    spec: ScenarioSpec,
    run_seed: int,
    trial_identities: Sequence[str] | None,
    conditions: Sequence[IncentiveCondition] | None,
    count: int,
) -> tuple[tuple[str, ...], tuple[IncentiveCondition, ...]] | None:
    if type(run_seed) is not int or run_seed < 0:
        builder.add(
            "batch.invalid_run_seed",
            "/run_seed",
            "run_seed must be a nonnegative exact integer",
        )
    if type(count) is not int or not MIN_BATCH_INSTANCES <= count <= MAX_BATCH_INSTANCES:
        builder.add(
            "batch.invalid_count",
            "/count",
            f"count must be between {MIN_BATCH_INSTANCES} and "
            f"{MAX_BATCH_INSTANCES}",
        )
    if builder.report().errors:
        return None

    if trial_identities is None:
        trial_ids = tuple(f"validation-{index:06d}" for index in range(count))
    else:
        if isinstance(trial_identities, (str, bytes, Mapping)):
            builder.add(
                "batch.invalid_trial_identities",
                "/trial_identities",
                "trial identities must be a bounded sequence of unique strings",
            )
            return None
        trial_ids = tuple(trial_identities)
        if len(trial_ids) != count:
            builder.add(
                "batch.trial_count_mismatch",
                "/trial_identities",
                "trial identity count must equal count",
            )
        if any(not isinstance(value, str) or not value for value in trial_ids):
            builder.add(
                "batch.invalid_trial_identity",
                "/trial_identities",
                "every trial identity must be a nonempty string",
            )
        if len(set(trial_ids)) != len(trial_ids):
            builder.add(
                "batch.duplicate_trial_identity",
                "/trial_identities",
                "trial identities must be unique",
            )

    declared = tuple(item.condition for item in spec.conditions)
    if conditions is None:
        selected = declared
    else:
        if isinstance(conditions, (str, bytes, Mapping)):
            builder.add(
                "batch.invalid_conditions",
                "/conditions",
                "conditions must be a sequence of IncentiveCondition values",
            )
            return None
        selected = tuple(conditions)
        if any(not isinstance(item, IncentiveCondition) for item in selected):
            builder.add(
                "batch.invalid_condition",
                "/conditions",
                "batch conditions must use strict IncentiveCondition values",
            )
        if len(set(selected)) != len(selected):
            builder.add(
                "batch.duplicate_condition",
                "/conditions",
                "batch conditions must be unique",
            )
        missing = set(declared).difference(selected)
        extra = set(selected).difference(declared)
        if missing:
            builder.add(
                "batch.incomplete_condition_coverage",
                "/conditions",
                "batch omits one or more declared conditions",
                related_ids=tuple(sorted(item.value for item in missing)),
            )
        if extra:
            builder.add(
                "batch.undeclared_condition",
                "/conditions",
                "batch includes one or more undeclared conditions",
                related_ids=tuple(sorted(item.value for item in extra)),
            )
    if not selected:
        builder.add(
            "batch.empty_conditions",
            "/conditions",
            "batch requires at least one declared condition",
        )
    if builder.report().errors:
        return None
    return trial_ids, selected


def validate_many_instances(
    spec: ScenarioSpec,
    run_seed: int = 0,
    trial_identities: Sequence[str] | None = None,
    conditions: Sequence[IncentiveCondition] | None = None,
    count: int = MIN_BATCH_INSTANCES,
    *,
    capabilities: ValidationCapabilities | Iterable[str] | None = None,
    forbidden_terms: Iterable[str] = DEFAULT_FORBIDDEN_PROMPT_TERMS,
) -> ValidationReport:
    """Compile and audit a bounded deterministic batch covering all conditions."""
    builder = _ReportBuilder(_subject_hash(spec))
    builder.extend(
        validate_spec(
            spec,
            capabilities,
            forbidden_terms=forbidden_terms,
        )
    )
    validated_spec = (
        _validated_spec_copy(spec, builder)
        if _check_spec_input_bounds(spec, builder)
        else None
    )
    if validated_spec is None:
        return builder.report()
    prepared = _validate_batch_inputs(
        builder,
        validated_spec,
        run_seed,
        trial_identities,
        conditions,
        count,
    )
    if prepared is None:
        return builder.report()
    trial_ids, selected_conditions = prepared
    try:
        normalized_terms = _normalize_forbidden_terms(forbidden_terms)
    except (TypeError, ValueError):
        normalized_terms = ()

    global_rng_before = random.getstate()
    hash_identities: dict[str, tuple[str, int, IncentiveCondition]] = {}
    for index, trial_id in enumerate(trial_ids):
        condition = selected_conditions[index % len(selected_conditions)]
        path = f"/batch/{index}"
        try:
            first = compile_scenario(
                validated_spec,
                trial_id,
                run_seed,
                condition,
            )
            second = compile_scenario(
                validated_spec,
                trial_id,
                run_seed,
                condition,
            )
        except (TypeError, ValueError, PermissionError) as exc:
            builder.add(
                "batch.compilation_failed",
                path,
                f"deterministic compilation failed: {exc}",
                related_ids=(trial_id, condition.value),
            )
            continue
        if first.canonical_json() != second.canonical_json():
            builder.add(
                "batch.compilation_nondeterministic",
                path,
                "repeated compilation produced different canonical instances",
                related_ids=(trial_id, condition.value),
            )
        identity = (trial_id, run_seed, condition)
        prior = hash_identities.get(first.instance_hash)
        if prior is not None and prior != identity:
            builder.add(
                "batch.instance_hash_collision",
                path,
                "different compilation identities produced one instance hash",
                related_ids=(first.instance_hash, prior[0], trial_id),
            )
        else:
            hash_identities[first.instance_hash] = identity
        _check_instance_core(validated_spec, first, builder)
        _check_rendered_prompts_core(
            validated_spec,
            first,
            builder,
            normalized_terms,
        )
    global_rng_after = random.getstate()
    if global_rng_after != global_rng_before:
        builder.add(
            "batch.global_rng_mutated",
            "/batch",
            "validation changed process-global random state",
        )
        random.setstate(global_rng_before)
    covered = {
        identity[2] for identity in hash_identities.values()
    }
    missing_coverage = set(selected_conditions).difference(covered)
    if missing_coverage:
        builder.add(
            "batch.condition_not_compiled",
            "/batch",
            "one or more selected conditions produced no compiled instance",
            related_ids=tuple(sorted(item.value for item in missing_coverage)),
        )
    return builder.report()


def raise_on_error(report: ValidationReport) -> ValidationReport:
    """Return a valid report or raise with the complete report attached."""
    if not isinstance(report, ValidationReport):
        raise TypeError("report must be a ValidationReport")
    return report.raise_for_errors()


def validate_or_raise(report: ValidationReport) -> ValidationReport:
    """Explicit alias for callers that use validation as a release gate."""
    return raise_on_error(report)


__all__ = [
    "DEFAULT_FORBIDDEN_PROMPT_TERMS",
    "MAX_BATCH_INSTANCES",
    "MAX_SPEC_COLLECTION_ITEMS",
    "MIN_BATCH_INSTANCES",
    "SCENARIO_VALIDATION_VERSION",
    "ScenarioValidationError",
    "ValidationCapabilities",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    "raise_on_error",
    "validate_instance",
    "validate_many_instances",
    "validate_or_raise",
    "validate_rendered_prompts",
    "validate_spec",
]
