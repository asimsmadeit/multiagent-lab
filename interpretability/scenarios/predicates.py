"""Closed, data-only predicate language for scenario adjudication.

The module deliberately accepts only a small discriminated-union AST.  It
contains no expression source, dynamic imports, callables, or executable text.
Predicates resolve typed operands through an immutable, authorization-checked
view of scenario facts and produce content-addressed tri-state results.
"""

# Exact concrete-type identity is required to reject bool-as-int and subclasses.
# pylint: disable=unidiomatic-typecheck

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import json
import math
import re
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Self,
    TypeAlias,
    get_args,
    get_origin,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from interpretability.scenarios.schema import (
    AdjudicatorView,
    FactRef,
    PrivateView,
    PublicState,
    Visibility,
    canonical_json as _canonical_json,
    canonical_sha256,
)


PREDICATE_DSL_SCHEMA_VERSION = "predicate-dsl/1.0.0"
PREDICATE_ENVIRONMENT_SCHEMA_VERSION = "predicate-environment/1.0.0"
PREDICATE_RESULT_SCHEMA_VERSION = "predicate-result/1.0.0"

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]*$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

Identifier = Annotated[
    str,
    Field(strict=True, min_length=1, pattern=_IDENTIFIER_PATTERN.pattern),
]

LiteralScalar: TypeAlias = StrictStr | StrictInt | StrictFloat | StrictBool
LiteralValue: TypeAlias = LiteralScalar | tuple[LiteralScalar, ...]


def _annotation_contains_tuple(annotation: Any) -> bool:
    if get_origin(annotation) is tuple:
        return True
    return any(_annotation_contains_tuple(item) for item in get_args(annotation))


def _content_identifier(prefix: str, payload: Any) -> str:
    digest = canonical_sha256(payload).removeprefix("sha256:")
    return f"{prefix}_{digest}"


class _StrictFrozenModel(BaseModel):
    """Strict immutable Pydantic policy with explicit persisted restoration."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _restore_persisted_tuples(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Any:
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
    """Compute and verify a subclass-specific identity over canonical content."""

    _identity_field: ClassVar[str]
    _identity_prefix: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def _require_persisted_identity(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Any:
        if not (info.context and info.context.get("persisted")):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("persisted predicate records must be JSON objects")
        if "schema_version" not in value:
            raise ValueError("schema_version is required on persisted records")
        supplied = value.get(cls._identity_field)
        if not isinstance(supplied, str) or not supplied:
            raise ValueError(
                f"{cls._identity_field} is required on persisted records"
            )
        return value

    @model_validator(mode="after")
    def _bind_content_identity(self) -> Self:
        payload = self.model_dump(mode="json", exclude={self._identity_field})
        expected = _content_identifier(self._identity_prefix, payload)
        supplied = getattr(self, self._identity_field)
        if supplied and supplied != expected:
            raise ValueError(
                f"{self._identity_field} does not match canonical content"
            )
        object.__setattr__(self, self._identity_field, expected)
        return self

    def canonical_json(self) -> str:
        """Return the complete canonical persisted representation."""
        return _canonical_json(self)

    def canonical_content_json(self) -> str:
        """Return the canonical payload covered by this record's identity."""
        return _canonical_json(
            self.model_dump(mode="json", exclude={self._identity_field})
        )

    @classmethod
    def from_persisted_json(cls, value: str | bytes) -> Self:
        """Restore canonical data while requiring and verifying its identity."""
        return cls.model_validate_json(value, context={"persisted": True})

    @classmethod
    def from_persisted(cls, value: Mapping[str, Any]) -> Self:
        """Restore a decoded JSON mapping with persisted-record validation."""
        return cls.model_validate_json(
            json.dumps(value, ensure_ascii=False, allow_nan=False),
            context={"persisted": True},
        )


class LiteralValueType(str, Enum):
    """Exact value types permitted in predicate operands."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING_TUPLE = "string_tuple"
    INTEGER_TUPLE = "integer_tuple"
    FLOAT_TUPLE = "float_tuple"
    BOOLEAN_TUPLE = "boolean_tuple"


_TUPLE_MEMBER_TYPES: dict[LiteralValueType, LiteralValueType] = {
    LiteralValueType.STRING_TUPLE: LiteralValueType.STRING,
    LiteralValueType.INTEGER_TUPLE: LiteralValueType.INTEGER,
    LiteralValueType.FLOAT_TUPLE: LiteralValueType.FLOAT,
    LiteralValueType.BOOLEAN_TUPLE: LiteralValueType.BOOLEAN,
}


def _value_matches_type(value: Any, value_type: LiteralValueType) -> bool:
    scalar_types: dict[LiteralValueType, type[Any]] = {
        LiteralValueType.STRING: str,
        LiteralValueType.INTEGER: int,
        LiteralValueType.FLOAT: float,
        LiteralValueType.BOOLEAN: bool,
    }
    expected_scalar = scalar_types.get(value_type)
    if expected_scalar is not None:
        if type(value) is not expected_scalar:
            return False
        return not (type(value) is float and not math.isfinite(value))

    member_type = _TUPLE_MEMBER_TYPES.get(value_type)
    if member_type is None or type(value) is not tuple:
        return False
    return all(_value_matches_type(item, member_type) for item in value)


class LiteralOperand(_StrictFrozenModel):
    """An explicitly typed inert literal value."""

    kind: Literal["literal"] = "literal"
    value_type: LiteralValueType
    value: LiteralValue

    @field_validator("value", mode="before")
    @classmethod
    def _reject_subclass_coercion(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Any:
        """Validate the raw value before Pydantic normalizes subclasses."""
        value_type = info.data.get("value_type")
        if not isinstance(value_type, LiteralValueType):
            return value
        candidate = value
        if (
            info.context
            and info.context.get("persisted")
            and isinstance(candidate, list)
            and value_type in _TUPLE_MEMBER_TYPES
        ):
            candidate = tuple(candidate)
        if not _value_matches_type(candidate, value_type):
            raise ValueError(
                "literal value must use the exact declared concrete type"
            )
        return candidate

    @model_validator(mode="after")
    def _value_has_declared_type(self) -> Self:
        if not _value_matches_type(self.value, self.value_type):
            raise ValueError("literal value does not match its declared value_type")
        return self

    @classmethod
    def from_value(cls, value: LiteralValue) -> LiteralOperand:
        """Create a typed literal without applying numeric or text coercion."""
        if type(value) is str:
            value_type = LiteralValueType.STRING
        elif type(value) is int:
            value_type = LiteralValueType.INTEGER
        elif type(value) is float:
            value_type = LiteralValueType.FLOAT
        elif type(value) is bool:
            value_type = LiteralValueType.BOOLEAN
        elif type(value) is tuple:
            if not value:
                raise ValueError(
                    "empty tuple literals require an explicit value_type"
                )
            first_type = type(value[0])
            tuple_types = {
                str: LiteralValueType.STRING_TUPLE,
                int: LiteralValueType.INTEGER_TUPLE,
                float: LiteralValueType.FLOAT_TUPLE,
                bool: LiteralValueType.BOOLEAN_TUPLE,
            }
            value_type = tuple_types.get(first_type)
            if value_type is None or not _value_matches_type(value, value_type):
                raise ValueError("tuple literals must contain one supported type")
        else:
            raise TypeError(f"unsupported literal type: {type(value).__name__}")
        return cls(value_type=value_type, value=value)


class FactRefOperand(_StrictFrozenModel):
    """A typed reference resolved only through an authorized fact environment."""

    kind: Literal["fact_ref"] = "fact_ref"
    fact_id: Identifier
    value_type: LiteralValueType


FactOperand = FactRefOperand
Operand: TypeAlias = Annotated[
    LiteralOperand | FactRefOperand,
    Field(discriminator="kind"),
]


def _require_matching_operand_types(left: Operand, right: Operand) -> None:
    if left.value_type is not right.value_type:
        raise ValueError("predicate operands must declare the same value_type")


def _require_numeric_operand_types(left: Operand, right: Operand) -> None:
    _require_matching_operand_types(left, right)
    if left.value_type not in {
        LiteralValueType.INTEGER,
        LiteralValueType.FLOAT,
    }:
        raise ValueError("ordered comparisons require same-type numeric operands")


class _PredicateNode(_ContentAddressedModel):
    """Content-addressed base shared by every closed predicate node."""

    schema_version: Literal[PREDICATE_DSL_SCHEMA_VERSION] = (
        PREDICATE_DSL_SCHEMA_VERSION
    )
    predicate_id: str = ""

    _identity_field = "predicate_id"
    _identity_prefix = "predicate"


class _EqualityPredicate(_PredicateNode):
    left: Operand
    right: Operand

    @model_validator(mode="after")
    def _matching_types(self) -> Self:
        _require_matching_operand_types(self.left, self.right)
        return self


class EqPredicate(_EqualityPredicate):
    """Exact, non-coercing equality."""

    op: Literal["eq"] = "eq"


class NePredicate(_EqualityPredicate):
    """Exact, non-coercing inequality."""

    op: Literal["ne"] = "ne"


class _OrderingPredicate(_PredicateNode):
    left: Operand
    right: Operand

    @model_validator(mode="after")
    def _numeric_types(self) -> Self:
        _require_numeric_operand_types(self.left, self.right)
        return self


class LtPredicate(_OrderingPredicate):
    """Strict numeric less-than comparison."""

    op: Literal["lt"] = "lt"


class LtePredicate(_OrderingPredicate):
    """Strict numeric less-than-or-equal comparison."""

    op: Literal["lte"] = "lte"


class GtPredicate(_OrderingPredicate):
    """Strict numeric greater-than comparison."""

    op: Literal["gt"] = "gt"


class GtePredicate(_OrderingPredicate):
    """Strict numeric greater-than-or-equal comparison."""

    op: Literal["gte"] = "gte"


class ContainsPredicate(_PredicateNode):
    """Safe substring or homogeneous-tuple membership comparison."""

    op: Literal["contains"] = "contains"
    container: Operand
    item: Operand

    @model_validator(mode="after")
    def _safe_declared_container(self) -> Self:
        if self.container.value_type is LiteralValueType.STRING:
            member_type = LiteralValueType.STRING
        else:
            member_type = _TUPLE_MEMBER_TYPES.get(self.container.value_type)
        if member_type is None:
            raise ValueError("contains requires a string or homogeneous tuple")
        if self.item.value_type is not member_type:
            raise ValueError("contains item type must match the container member type")
        return self


class AllPredicate(_PredicateNode):
    """Kleene conjunction: false dominates, then unknown, then true."""

    op: Literal["all"] = "all"
    predicates: Annotated[tuple["Predicate", ...], Field(min_length=1)]


class AnyPredicate(_PredicateNode):
    """Kleene disjunction: true dominates, then unknown, then false."""

    op: Literal["any"] = "any"
    predicates: Annotated[tuple["Predicate", ...], Field(min_length=1)]


class NotPredicate(_PredicateNode):
    """Tri-state logical negation that preserves unknown."""

    op: Literal["not"] = "not"
    predicate: "Predicate"


Predicate: TypeAlias = Annotated[
    EqPredicate
    | NePredicate
    | LtPredicate
    | LtePredicate
    | GtPredicate
    | GtePredicate
    | ContainsPredicate
    | AllPredicate
    | AnyPredicate
    | NotPredicate,
    Field(discriminator="op"),
]

_PREDICATE_TYPES = (AllPredicate, AnyPredicate, NotPredicate)
for _predicate_type in _PREDICATE_TYPES:
    _predicate_type.model_rebuild(_types_namespace={"Predicate": Predicate})

PREDICATE_ADAPTER = TypeAdapter(Predicate)


def predicate_canonical_json(predicate: Predicate) -> str:
    """Serialize a predicate AST using the repository's canonical JSON policy."""
    return _canonical_json(predicate)


def predicate_from_persisted_json(value: str | bytes) -> Predicate:
    """Restore a complete predicate AST and verify every nested identity."""
    return PREDICATE_ADAPTER.validate_json(value, context={"persisted": True})


def predicate_from_persisted(value: Mapping[str, Any]) -> Predicate:
    """Restore a decoded predicate mapping with tamper validation."""
    return predicate_from_persisted_json(
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    )


class PredicateScope(str, Enum):
    """Authorization boundary represented by a predicate environment."""

    PUBLIC = "public"
    ROLE_PRIVATE = "role_private"
    ADJUDICATOR = "adjudicator"


class PredicateEnvironment(_ContentAddressedModel):
    """Canonical facts available at one authorization boundary."""

    schema_version: Literal[PREDICATE_ENVIRONMENT_SCHEMA_VERSION] = (
        PREDICATE_ENVIRONMENT_SCHEMA_VERSION
    )
    scope: PredicateScope
    role_id: Identifier | None = None
    source_view_hash: str
    facts: tuple[FactRef, ...]
    environment_id: str = ""

    _identity_field = "environment_id"
    _identity_prefix = "predicate_environment"

    @field_validator("source_view_hash")
    @classmethod
    def _valid_view_hash(cls, value: str) -> str:
        if not _SHA256_PATTERN.fullmatch(value):
            raise ValueError("source_view_hash must be a canonical sha256 digest")
        return value

    @model_validator(mode="after")
    def _authorized_canonical_facts(self) -> Self:
        fact_ids = tuple(fact.fact_id for fact in self.facts)
        if fact_ids != tuple(sorted(set(fact_ids))):
            raise ValueError("environment facts must have ordered unique fact IDs")

        if self.scope is PredicateScope.PUBLIC:
            if self.role_id is not None:
                raise ValueError("public environments cannot name a role")
            if any(fact.visibility is not Visibility.PUBLIC for fact in self.facts):
                raise ValueError("public environments may contain only public facts")
        elif self.scope is PredicateScope.ROLE_PRIVATE:
            if self.role_id is None:
                raise ValueError("role-private environments require role_id")
            for fact in self.facts:
                if fact.visibility is Visibility.ADJUDICATOR_ONLY:
                    raise ValueError(
                        "role-private environments cannot contain adjudicator facts"
                    )
                if (
                    fact.visibility is Visibility.ROLE_PRIVATE
                    and self.role_id not in fact.visible_to
                ):
                    raise ValueError(
                        "role-private environment contains an unauthorized fact"
                    )
        elif self.scope is PredicateScope.ADJUDICATOR:
            if self.role_id is not None:
                raise ValueError("adjudicator environments cannot name a role")
        else:  # pragma: no cover - strict enum makes this defensive only
            raise ValueError("unsupported predicate environment scope")
        return self

    @classmethod
    def from_public_state(cls, view: PublicState) -> PredicateEnvironment:
        """Derive an environment containing only universally visible facts."""
        if not isinstance(view, PublicState):
            raise TypeError("view must be a PublicState")
        return cls(
            scope=PredicateScope.PUBLIC,
            source_view_hash=view.public_state_hash,
            facts=tuple(sorted(view.facts, key=lambda fact: fact.fact_id)),
        )

    @classmethod
    def from_private_view(cls, view: PrivateView) -> PredicateEnvironment:
        """Derive an environment for one role's authorized public/private view."""
        if not isinstance(view, PrivateView):
            raise TypeError("view must be a PrivateView")
        return cls(
            scope=PredicateScope.ROLE_PRIVATE,
            role_id=view.role_id,
            source_view_hash=view.view_hash,
            facts=tuple(sorted(view.facts, key=lambda fact: fact.fact_id)),
        )

    @classmethod
    def from_adjudicator_view(
        cls,
        view: AdjudicatorView,
    ) -> PredicateEnvironment:
        """Derive an environment authorized to resolve every supplied fact."""
        if not isinstance(view, AdjudicatorView):
            raise TypeError("view must be an AdjudicatorView")
        return cls(
            scope=PredicateScope.ADJUDICATOR,
            source_view_hash=view.adjudicator_view_hash,
            facts=tuple(sorted(view.facts, key=lambda fact: fact.fact_id)),
        )


class PredicateTruth(str, Enum):
    """Three-valued result domain."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


TruthValue = PredicateTruth


class PredicateReasonCode(str, Enum):
    """Stable explanation for a predicate's top-level result."""

    EVALUATED_TRUE = "evaluated_true"
    EVALUATED_FALSE = "evaluated_false"
    FACT_UNAVAILABLE = "fact_missing_or_unauthorized"
    TYPE_INCOMPATIBLE = "type_incompatible"
    NONFINITE_NUMERIC = "nonfinite_numeric"
    UNSAFE_CONTAINER = "unsafe_container"


class PredicateResult(_ContentAddressedModel):
    """Immutable tri-state evaluation with complete fact-reference lineage."""

    schema_version: Literal[PREDICATE_RESULT_SCHEMA_VERSION] = (
        PREDICATE_RESULT_SCHEMA_VERSION
    )
    predicate_id: str
    environment_id: str
    truth: PredicateTruth
    reason_code: PredicateReasonCode
    input_fact_ids: tuple[Identifier, ...]
    result_id: str = ""

    _identity_field = "result_id"
    _identity_prefix = "predicate_result"

    @model_validator(mode="after")
    def _canonical_input_fact_ids(self) -> Self:
        if self.input_fact_ids != tuple(sorted(set(self.input_fact_ids))):
            raise ValueError("input_fact_ids must be ordered and unique")
        if self.truth is PredicateTruth.TRUE and (
            self.reason_code is not PredicateReasonCode.EVALUATED_TRUE
        ):
            raise ValueError("true results require evaluated_true reason_code")
        if self.truth is PredicateTruth.FALSE and (
            self.reason_code is not PredicateReasonCode.EVALUATED_FALSE
        ):
            raise ValueError("false results require evaluated_false reason_code")
        if self.truth is PredicateTruth.UNKNOWN and self.reason_code in {
            PredicateReasonCode.EVALUATED_TRUE,
            PredicateReasonCode.EVALUATED_FALSE,
        }:
            raise ValueError("unknown results require an unknown reason_code")
        return self


class _ResolvedOperand(_StrictFrozenModel):
    value: LiteralValue
    value_type: LiteralValueType


class _NodeEvaluation(_StrictFrozenModel):
    truth: PredicateTruth
    reason_code: PredicateReasonCode


_TRUE = _NodeEvaluation(
    truth=PredicateTruth.TRUE,
    reason_code=PredicateReasonCode.EVALUATED_TRUE,
)
_FALSE = _NodeEvaluation(
    truth=PredicateTruth.FALSE,
    reason_code=PredicateReasonCode.EVALUATED_FALSE,
)


def _unknown(reason_code: PredicateReasonCode) -> _NodeEvaluation:
    return _NodeEvaluation(
        truth=PredicateTruth.UNKNOWN,
        reason_code=reason_code,
    )


def _fact_is_authorized(
    fact: FactRef,
    environment: PredicateEnvironment,
) -> bool:
    if environment.scope is PredicateScope.ADJUDICATOR:
        return True
    if fact.visibility is Visibility.ADJUDICATOR_ONLY:
        return False
    if environment.scope is PredicateScope.PUBLIC:
        return fact.visibility is Visibility.PUBLIC
    return fact.visibility is Visibility.PUBLIC or (
        fact.visibility is Visibility.ROLE_PRIVATE
        and environment.role_id in fact.visible_to
    )


def _resolve_operand(
    operand: Operand,
    environment: PredicateEnvironment,
) -> _ResolvedOperand | _NodeEvaluation:
    if isinstance(operand, LiteralOperand):
        if not _value_matches_type(operand.value, operand.value_type):
            return _unknown(PredicateReasonCode.TYPE_INCOMPATIBLE)
        return _ResolvedOperand(value=operand.value, value_type=operand.value_type)

    if isinstance(operand, FactRefOperand):
        fact = next(
            (item for item in environment.facts if item.fact_id == operand.fact_id),
            None,
        )
        if fact is None or not _fact_is_authorized(fact, environment):
            return _unknown(PredicateReasonCode.FACT_UNAVAILABLE)
        if not _value_matches_type(fact.value, operand.value_type):
            return _unknown(PredicateReasonCode.TYPE_INCOMPATIBLE)
        return _ResolvedOperand(value=fact.value, value_type=operand.value_type)

    raise TypeError(f"unsupported operand node: {type(operand).__name__}")


def _resolve_binary(
    left: Operand,
    right: Operand,
    environment: PredicateEnvironment,
) -> tuple[_ResolvedOperand, _ResolvedOperand] | _NodeEvaluation:
    resolved_left = _resolve_operand(left, environment)
    if isinstance(resolved_left, _NodeEvaluation):
        return resolved_left
    resolved_right = _resolve_operand(right, environment)
    if isinstance(resolved_right, _NodeEvaluation):
        return resolved_right
    if resolved_left.value_type is not resolved_right.value_type:
        return _unknown(PredicateReasonCode.TYPE_INCOMPATIBLE)
    return resolved_left, resolved_right


def _evaluate_equality(
    left: Operand,
    right: Operand,
    environment: PredicateEnvironment,
    *,
    negate: bool,
) -> _NodeEvaluation:
    resolved = _resolve_binary(left, right, environment)
    if isinstance(resolved, _NodeEvaluation):
        return resolved
    resolved_left, resolved_right = resolved
    equal = resolved_left.value == resolved_right.value
    return _TRUE if equal is not negate else _FALSE


def _finite_numeric(value: Any) -> bool:
    if type(value) is int:
        return True
    return type(value) is float and math.isfinite(value)


def _evaluate_ordering(
    predicate: LtPredicate | LtePredicate | GtPredicate | GtePredicate,
    environment: PredicateEnvironment,
) -> _NodeEvaluation:
    resolved = _resolve_binary(predicate.left, predicate.right, environment)
    if isinstance(resolved, _NodeEvaluation):
        return resolved
    left, right = resolved
    if left.value_type not in {
        LiteralValueType.INTEGER,
        LiteralValueType.FLOAT,
    }:
        return _unknown(PredicateReasonCode.TYPE_INCOMPATIBLE)
    if not _finite_numeric(left.value) or not _finite_numeric(right.value):
        return _unknown(PredicateReasonCode.NONFINITE_NUMERIC)

    if isinstance(predicate, LtPredicate):
        outcome = left.value < right.value
    elif isinstance(predicate, LtePredicate):
        outcome = left.value <= right.value
    elif isinstance(predicate, GtPredicate):
        outcome = left.value > right.value
    elif isinstance(predicate, GtePredicate):
        outcome = left.value >= right.value
    else:  # pragma: no cover - closed caller union is exhaustive
        raise TypeError(f"unsupported ordering node: {type(predicate).__name__}")
    return _TRUE if outcome else _FALSE


def _evaluate_contains(
    predicate: ContainsPredicate,
    environment: PredicateEnvironment,
) -> _NodeEvaluation:
    container = _resolve_operand(predicate.container, environment)
    if isinstance(container, _NodeEvaluation):
        return container
    item = _resolve_operand(predicate.item, environment)
    if isinstance(item, _NodeEvaluation):
        return item

    if container.value_type is LiteralValueType.STRING:
        if item.value_type is not LiteralValueType.STRING:
            return _unknown(PredicateReasonCode.TYPE_INCOMPATIBLE)
        outcome = item.value in container.value
    else:
        member_type = _TUPLE_MEMBER_TYPES.get(container.value_type)
        if member_type is None:
            return _unknown(PredicateReasonCode.UNSAFE_CONTAINER)
        if item.value_type is not member_type or type(container.value) is not tuple:
            return _unknown(PredicateReasonCode.TYPE_INCOMPATIBLE)
        outcome = item.value in container.value
    return _TRUE if outcome else _FALSE


def _evaluate_all(
    predicate: AllPredicate,
    environment: PredicateEnvironment,
) -> _NodeEvaluation:
    first_unknown: _NodeEvaluation | None = None
    for child in predicate.predicates:
        result = _evaluate_node(child, environment)
        if result.truth is PredicateTruth.FALSE:
            return _FALSE
        if result.truth is PredicateTruth.UNKNOWN and first_unknown is None:
            first_unknown = result
    return first_unknown or _TRUE


def _evaluate_any(
    predicate: AnyPredicate,
    environment: PredicateEnvironment,
) -> _NodeEvaluation:
    first_unknown: _NodeEvaluation | None = None
    for child in predicate.predicates:
        result = _evaluate_node(child, environment)
        if result.truth is PredicateTruth.TRUE:
            return _TRUE
        if result.truth is PredicateTruth.UNKNOWN and first_unknown is None:
            first_unknown = result
    return first_unknown or _FALSE


def _evaluate_not(
    predicate: NotPredicate,
    environment: PredicateEnvironment,
) -> _NodeEvaluation:
    result = _evaluate_node(predicate.predicate, environment)
    if result.truth is PredicateTruth.TRUE:
        return _FALSE
    if result.truth is PredicateTruth.FALSE:
        return _TRUE
    return result


# Explicit dispatch over every closed AST variant is intentional.
# pylint: disable-next=too-many-return-statements
def _evaluate_node(
    predicate: Predicate,
    environment: PredicateEnvironment,
) -> _NodeEvaluation:
    if isinstance(predicate, EqPredicate):
        return _evaluate_equality(
            predicate.left,
            predicate.right,
            environment,
            negate=False,
        )
    if isinstance(predicate, NePredicate):
        return _evaluate_equality(
            predicate.left,
            predicate.right,
            environment,
            negate=True,
        )
    if isinstance(predicate, LtPredicate):
        return _evaluate_ordering(predicate, environment)
    if isinstance(predicate, LtePredicate):
        return _evaluate_ordering(predicate, environment)
    if isinstance(predicate, GtPredicate):
        return _evaluate_ordering(predicate, environment)
    if isinstance(predicate, GtePredicate):
        return _evaluate_ordering(predicate, environment)
    if isinstance(predicate, ContainsPredicate):
        return _evaluate_contains(predicate, environment)
    if isinstance(predicate, AllPredicate):
        return _evaluate_all(predicate, environment)
    if isinstance(predicate, AnyPredicate):
        return _evaluate_any(predicate, environment)
    if isinstance(predicate, NotPredicate):
        return _evaluate_not(predicate, environment)
    raise TypeError(f"unsupported predicate node: {type(predicate).__name__}")


def _operand_fact_ids(operand: Operand) -> tuple[str, ...]:
    if isinstance(operand, LiteralOperand):
        return ()
    if isinstance(operand, FactRefOperand):
        return (operand.fact_id,)
    raise TypeError(f"unsupported operand node: {type(operand).__name__}")


def _predicate_fact_ids(predicate: Predicate) -> tuple[str, ...]:
    if isinstance(
        predicate,
        (EqPredicate, NePredicate, LtPredicate, LtePredicate, GtPredicate, GtePredicate),
    ):
        return (
            *_operand_fact_ids(predicate.left),
            *_operand_fact_ids(predicate.right),
        )
    if isinstance(predicate, ContainsPredicate):
        return (
            *_operand_fact_ids(predicate.container),
            *_operand_fact_ids(predicate.item),
        )
    if isinstance(predicate, AllPredicate):
        return tuple(
            fact_id
            for child in predicate.predicates
            for fact_id in _predicate_fact_ids(child)
        )
    if isinstance(predicate, AnyPredicate):
        return tuple(
            fact_id
            for child in predicate.predicates
            for fact_id in _predicate_fact_ids(child)
        )
    if isinstance(predicate, NotPredicate):
        return _predicate_fact_ids(predicate.predicate)
    raise TypeError(f"unsupported predicate node: {type(predicate).__name__}")


def evaluate_predicate(
    predicate: Predicate,
    environment: PredicateEnvironment,
) -> PredicateResult:
    """Evaluate a validated AST without code execution or value coercion."""
    if not isinstance(environment, PredicateEnvironment):
        raise TypeError("environment must be a PredicateEnvironment")
    validated = PREDICATE_ADAPTER.validate_python(predicate)
    evaluation = _evaluate_node(validated, environment)
    input_fact_ids = tuple(sorted(set(_predicate_fact_ids(validated))))
    return PredicateResult(
        predicate_id=validated.predicate_id,
        environment_id=environment.environment_id,
        truth=evaluation.truth,
        reason_code=evaluation.reason_code,
        input_fact_ids=input_fact_ids,
    )


__all__ = [
    "AllPredicate",
    "AnyPredicate",
    "ContainsPredicate",
    "EqPredicate",
    "FactOperand",
    "FactRefOperand",
    "GtPredicate",
    "GtePredicate",
    "LiteralOperand",
    "LiteralValueType",
    "LtPredicate",
    "LtePredicate",
    "NePredicate",
    "NotPredicate",
    "Operand",
    "PREDICATE_ADAPTER",
    "PREDICATE_DSL_SCHEMA_VERSION",
    "PREDICATE_ENVIRONMENT_SCHEMA_VERSION",
    "PREDICATE_RESULT_SCHEMA_VERSION",
    "Predicate",
    "PredicateEnvironment",
    "PredicateReasonCode",
    "PredicateResult",
    "PredicateScope",
    "PredicateTruth",
    "TruthValue",
    "evaluate_predicate",
    "predicate_canonical_json",
    "predicate_from_persisted",
    "predicate_from_persisted_json",
]
