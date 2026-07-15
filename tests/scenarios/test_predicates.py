"""Permanent contracts for the closed scenario predicate language."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from interpretability.scenarios.predicates import (
    AllPredicate,
    AnyPredicate,
    ContainsPredicate,
    EqPredicate,
    FactRefOperand,
    GtPredicate,
    GtePredicate,
    LiteralOperand,
    LiteralValueType,
    LtPredicate,
    LtePredicate,
    NePredicate,
    NotPredicate,
    PREDICATE_ADAPTER,
    Predicate,
    PredicateEnvironment,
    PredicateReasonCode,
    PredicateResult,
    PredicateScope,
    PredicateTruth,
    evaluate_predicate,
    predicate_canonical_json,
    predicate_from_persisted,
    predicate_from_persisted_json,
)
from interpretability.scenarios.schema import (
    AdjudicatorView,
    FactRef,
    PrivateView,
    PublicState,
    Visibility,
)


OPERATORS = (
    "eq",
    "ne",
    "lt",
    "lte",
    "gt",
    "gte",
    "contains",
    "all",
    "any",
    "not",
)


def _literal(value: Any) -> LiteralOperand:
    return LiteralOperand.from_value(value)


def _fact(fact_id: str, value_type: LiteralValueType) -> FactRefOperand:
    return FactRefOperand(fact_id=fact_id, value_type=value_type)


@dataclass(frozen=True)
class PredicateViews:
    facts: dict[str, FactRef]
    public: PublicState
    actor: PrivateView
    adjudicator: AdjudicatorView


@pytest.fixture(scope="module")
def predicate_views() -> PredicateViews:
    facts = {
        "actor.secret": FactRef(
            fact_id="actor.secret",
            fact_version="v1",
            value=7,
            visibility=Visibility.ROLE_PRIVATE,
            visible_to=("actor",),
        ),
        "judge.secret": FactRef(
            fact_id="judge.secret",
            fact_version="v1",
            value="sealed",
            visibility=Visibility.ADJUDICATOR_ONLY,
        ),
        "public.boolean": FactRef(
            fact_id="public.boolean",
            fact_version="v1",
            value=True,
            visibility=Visibility.PUBLIC,
        ),
        "public.empty": FactRef(
            fact_id="public.empty",
            fact_version="v1",
            value=(),
            visibility=Visibility.PUBLIC,
        ),
        "public.float": FactRef(
            fact_id="public.float",
            fact_version="v1",
            value=2.5,
            visibility=Visibility.PUBLIC,
        ),
        "public.integer": FactRef(
            fact_id="public.integer",
            fact_version="v1",
            value=10,
            visibility=Visibility.PUBLIC,
        ),
        "public.integers": FactRef(
            fact_id="public.integers",
            fact_version="v1",
            value=(1, 2, 3),
            visibility=Visibility.PUBLIC,
        ),
        "public.mixed": FactRef(
            fact_id="public.mixed",
            fact_version="v1",
            value=(1, "red"),
            visibility=Visibility.PUBLIC,
        ),
        "public.strings": FactRef(
            fact_id="public.strings",
            fact_version="v1",
            value=("red", "blue"),
            visibility=Visibility.PUBLIC,
        ),
        "public.text": FactRef(
            fact_id="public.text",
            fact_version="v1",
            value="alpha beta",
            visibility=Visibility.PUBLIC,
        ),
    }
    public_facts = tuple(
        fact
        for fact in reversed(tuple(facts.values()))
        if fact.visibility is Visibility.PUBLIC
    )
    return PredicateViews(
        facts=facts,
        public=PublicState(facts=public_facts),
        actor=PrivateView(
            role_id="actor",
            facts=(facts["actor.secret"], *public_facts),
        ),
        adjudicator=AdjudicatorView(facts=tuple(reversed(tuple(facts.values())))),
    )


@pytest.fixture(scope="module")
def public_environment(predicate_views: PredicateViews) -> PredicateEnvironment:
    return PredicateEnvironment.from_public_state(predicate_views.public)


@pytest.fixture(scope="module")
def actor_environment(predicate_views: PredicateViews) -> PredicateEnvironment:
    return PredicateEnvironment.from_private_view(predicate_views.actor)


@pytest.fixture(scope="module")
def adjudicator_environment(
    predicate_views: PredicateViews,
) -> PredicateEnvironment:
    return PredicateEnvironment.from_adjudicator_view(predicate_views.adjudicator)


def _unknown_leaf() -> EqPredicate:
    return EqPredicate(
        left=_fact("missing.fact", LiteralValueType.INTEGER),
        right=_literal(1),
    )


def _true_leaf() -> EqPredicate:
    return EqPredicate(
        left=_fact("public.integer", LiteralValueType.INTEGER),
        right=_literal(10),
    )


def _false_leaf() -> EqPredicate:
    return EqPredicate(
        left=_fact("public.integer", LiteralValueType.INTEGER),
        right=_literal(11),
    )


def _operator_case(operator: str, truth: PredicateTruth) -> Predicate:
    integer = _fact("public.integer", LiteralValueType.INTEGER)
    if operator == "eq":
        right = _literal(10 if truth is PredicateTruth.TRUE else 11)
        return _unknown_leaf() if truth is PredicateTruth.UNKNOWN else EqPredicate(
            left=integer,
            right=right,
        )
    if operator == "ne":
        right = _literal(11 if truth is PredicateTruth.TRUE else 10)
        return NePredicate(
            left=(
                _fact("missing.fact", LiteralValueType.INTEGER)
                if truth is PredicateTruth.UNKNOWN
                else integer
            ),
            right=right,
        )
    if operator == "lt":
        right = _literal(11 if truth is PredicateTruth.TRUE else 9)
        return LtPredicate(
            left=(
                _fact("missing.fact", LiteralValueType.INTEGER)
                if truth is PredicateTruth.UNKNOWN
                else integer
            ),
            right=right,
        )
    if operator == "lte":
        right = _literal(10 if truth is PredicateTruth.TRUE else 9)
        return LtePredicate(
            left=(
                _fact("missing.fact", LiteralValueType.INTEGER)
                if truth is PredicateTruth.UNKNOWN
                else integer
            ),
            right=right,
        )
    if operator == "gt":
        right = _literal(9 if truth is PredicateTruth.TRUE else 11)
        return GtPredicate(
            left=(
                _fact("missing.fact", LiteralValueType.INTEGER)
                if truth is PredicateTruth.UNKNOWN
                else integer
            ),
            right=right,
        )
    if operator == "gte":
        right = _literal(10 if truth is PredicateTruth.TRUE else 11)
        return GtePredicate(
            left=(
                _fact("missing.fact", LiteralValueType.INTEGER)
                if truth is PredicateTruth.UNKNOWN
                else integer
            ),
            right=right,
        )
    if operator == "contains":
        return ContainsPredicate(
            container=(
                _fact("missing.fact", LiteralValueType.STRING)
                if truth is PredicateTruth.UNKNOWN
                else _fact("public.text", LiteralValueType.STRING)
            ),
            item=_literal("beta" if truth is PredicateTruth.TRUE else "gamma"),
        )
    if operator == "all":
        if truth is PredicateTruth.TRUE:
            children = (_true_leaf(), _true_leaf())
        elif truth is PredicateTruth.FALSE:
            children = (_true_leaf(), _false_leaf())
        else:
            children = (_true_leaf(), _unknown_leaf())
        return AllPredicate(predicates=children)
    if operator == "any":
        if truth is PredicateTruth.TRUE:
            children = (_false_leaf(), _true_leaf())
        elif truth is PredicateTruth.FALSE:
            children = (_false_leaf(), _false_leaf())
        else:
            children = (_false_leaf(), _unknown_leaf())
        return AnyPredicate(predicates=children)
    if operator == "not":
        child = {
            PredicateTruth.TRUE: _false_leaf(),
            PredicateTruth.FALSE: _true_leaf(),
            PredicateTruth.UNKNOWN: _unknown_leaf(),
        }[truth]
        return NotPredicate(predicate=child)
    raise AssertionError(f"test case missing for operator {operator!r}")


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize(
    "truth",
    (PredicateTruth.TRUE, PredicateTruth.FALSE, PredicateTruth.UNKNOWN),
)
def test_every_operator_has_a_complete_truth_table(
    operator: str,
    truth: PredicateTruth,
    public_environment: PredicateEnvironment,
) -> None:
    predicate = _operator_case(operator, truth)
    result = evaluate_predicate(predicate, public_environment)

    assert result.truth is truth
    if truth is PredicateTruth.TRUE:
        assert result.reason_code is PredicateReasonCode.EVALUATED_TRUE
    elif truth is PredicateTruth.FALSE:
        assert result.reason_code is PredicateReasonCode.EVALUATED_FALSE
    else:
        assert result.reason_code is PredicateReasonCode.FACT_UNAVAILABLE


@pytest.mark.parametrize(
    ("child", "expected"),
    [
        (PredicateTruth.TRUE, PredicateTruth.TRUE),
        (PredicateTruth.FALSE, PredicateTruth.FALSE),
        (PredicateTruth.UNKNOWN, PredicateTruth.UNKNOWN),
    ],
)
def test_double_negation_identity(
    child: PredicateTruth,
    expected: PredicateTruth,
    public_environment: PredicateEnvironment,
) -> None:
    predicate = NotPredicate(predicate=NotPredicate(predicate=_operator_case("eq", child)))
    assert evaluate_predicate(predicate, public_environment).truth is expected


@pytest.mark.parametrize("operator", ("all", "any"))
@pytest.mark.parametrize(
    "truth",
    (PredicateTruth.TRUE, PredicateTruth.FALSE, PredicateTruth.UNKNOWN),
)
def test_single_child_all_and_any_are_identities(
    operator: str,
    truth: PredicateTruth,
    public_environment: PredicateEnvironment,
) -> None:
    child = _operator_case("eq", truth)
    predicate: Predicate = (
        AllPredicate(predicates=(child,))
        if operator == "all"
        else AnyPredicate(predicates=(child,))
    )
    assert evaluate_predicate(predicate, public_environment).truth is truth


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        (
            AllPredicate(predicates=(_unknown_leaf(), _false_leaf(), _true_leaf())),
            PredicateTruth.FALSE,
        ),
        (
            AllPredicate(predicates=(_unknown_leaf(), _true_leaf())),
            PredicateTruth.UNKNOWN,
        ),
        (
            AnyPredicate(predicates=(_unknown_leaf(), _true_leaf(), _false_leaf())),
            PredicateTruth.TRUE,
        ),
        (
            AnyPredicate(predicates=(_unknown_leaf(), _false_leaf())),
            PredicateTruth.UNKNOWN,
        ),
    ],
    ids=("false-dominates-all", "unknown-all", "true-dominates-any", "unknown-any"),
)
def test_all_any_three_valued_dominance(
    predicate: Predicate,
    expected: PredicateTruth,
    public_environment: PredicateEnvironment,
) -> None:
    assert evaluate_predicate(predicate, public_environment).truth is expected


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        (
            AllPredicate(
                predicates=(
                    _false_leaf(),
                    _unknown_leaf(),
                    EqPredicate(
                        left=_fact("actor.secret", LiteralValueType.INTEGER),
                        right=_literal(7),
                    ),
                    _true_leaf(),
                )
            ),
            PredicateTruth.FALSE,
        ),
        (
            AnyPredicate(
                predicates=(
                    _true_leaf(),
                    _unknown_leaf(),
                    EqPredicate(
                        left=_fact("actor.secret", LiteralValueType.INTEGER),
                        right=_literal(7),
                    ),
                    _false_leaf(),
                )
            ),
            PredicateTruth.TRUE,
        ),
    ],
    ids=("all-short-circuits-false", "any-short-circuits-true"),
)
def test_nested_short_circuit_retains_complete_static_fact_lineage(
    predicate: Predicate,
    expected: PredicateTruth,
    public_environment: PredicateEnvironment,
) -> None:
    result = evaluate_predicate(predicate, public_environment)

    assert result.truth is expected
    assert result.input_fact_ids == (
        "actor.secret",
        "missing.fact",
        "public.integer",
    )
    assert len(result.input_fact_ids) == len(set(result.input_fact_ids))


def test_deeply_nested_lineage_is_ordered_unique(
    actor_environment: PredicateEnvironment,
) -> None:
    predicate = NotPredicate(
        predicate=AnyPredicate(
            predicates=(
                AllPredicate(predicates=(_true_leaf(), _unknown_leaf())),
                EqPredicate(
                    left=_fact("actor.secret", LiteralValueType.INTEGER),
                    right=_literal(7),
                ),
                _true_leaf(),
            )
        )
    )

    result = evaluate_predicate(predicate, actor_environment)

    assert result.truth is PredicateTruth.FALSE
    assert result.input_fact_ids == (
        "actor.secret",
        "missing.fact",
        "public.integer",
    )


def test_environment_factories_preserve_real_view_authorization(
    predicate_views: PredicateViews,
    public_environment: PredicateEnvironment,
    actor_environment: PredicateEnvironment,
    adjudicator_environment: PredicateEnvironment,
) -> None:
    assert public_environment.scope is PredicateScope.PUBLIC
    assert public_environment.role_id is None
    assert public_environment.source_view_hash == predicate_views.public.public_state_hash
    assert actor_environment.scope is PredicateScope.ROLE_PRIVATE
    assert actor_environment.role_id == "actor"
    assert actor_environment.source_view_hash == predicate_views.actor.view_hash
    assert adjudicator_environment.scope is PredicateScope.ADJUDICATOR
    assert adjudicator_environment.role_id is None
    assert (
        adjudicator_environment.source_view_hash
        == predicate_views.adjudicator.adjudicator_view_hash
    )
    for environment in (
        public_environment,
        actor_environment,
        adjudicator_environment,
    ):
        ids = tuple(fact.fact_id for fact in environment.facts)
        assert ids == tuple(sorted(ids))
        assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    ("fact_id", "value_type", "expected"),
    [
        ("actor.secret", LiteralValueType.INTEGER, PredicateTruth.UNKNOWN),
        ("judge.secret", LiteralValueType.STRING, PredicateTruth.UNKNOWN),
        ("missing.fact", LiteralValueType.INTEGER, PredicateTruth.UNKNOWN),
    ],
)
def test_public_environment_fails_closed_without_revealing_absence_reason(
    fact_id: str,
    value_type: LiteralValueType,
    expected: PredicateTruth,
    public_environment: PredicateEnvironment,
) -> None:
    right = _literal(7 if value_type is LiteralValueType.INTEGER else "sealed")
    result = evaluate_predicate(
        EqPredicate(left=_fact(fact_id, value_type), right=right),
        public_environment,
    )

    assert result.truth is expected
    assert result.reason_code is PredicateReasonCode.FACT_UNAVAILABLE


def test_private_and_adjudicator_resolution_boundaries(
    actor_environment: PredicateEnvironment,
    adjudicator_environment: PredicateEnvironment,
) -> None:
    private_predicate = EqPredicate(
        left=_fact("actor.secret", LiteralValueType.INTEGER),
        right=_literal(7),
    )
    adjudicator_predicate = EqPredicate(
        left=_fact("judge.secret", LiteralValueType.STRING),
        right=_literal("sealed"),
    )

    assert (
        evaluate_predicate(private_predicate, actor_environment).truth
        is PredicateTruth.TRUE
    )
    private_judge_result = evaluate_predicate(
        adjudicator_predicate,
        actor_environment,
    )
    assert private_judge_result.truth is PredicateTruth.UNKNOWN
    assert private_judge_result.reason_code is PredicateReasonCode.FACT_UNAVAILABLE
    assert (
        evaluate_predicate(private_predicate, adjudicator_environment).truth
        is PredicateTruth.TRUE
    )
    assert (
        evaluate_predicate(adjudicator_predicate, adjudicator_environment).truth
        is PredicateTruth.TRUE
    )


def test_environment_factories_require_the_declared_view_type(
    predicate_views: PredicateViews,
) -> None:
    with pytest.raises(TypeError):
        PredicateEnvironment.from_public_state(predicate_views.actor)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PredicateEnvironment.from_private_view(predicate_views.public)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PredicateEnvironment.from_adjudicator_view(predicate_views.actor)  # type: ignore[arg-type]


def test_environment_constructor_rejects_unauthorized_fact_sets(
    predicate_views: PredicateViews,
) -> None:
    actor_private = predicate_views.facts["actor.secret"]
    judge_only = predicate_views.facts["judge.secret"]
    view_hash = predicate_views.actor.view_hash

    with pytest.raises(ValidationError):
        PredicateEnvironment(
            scope=PredicateScope.PUBLIC,
            source_view_hash=view_hash,
            facts=(actor_private,),
        )
    with pytest.raises(ValidationError):
        PredicateEnvironment(
            scope=PredicateScope.ROLE_PRIVATE,
            role_id="counterpart",
            source_view_hash=view_hash,
            facts=(actor_private,),
        )
    with pytest.raises(ValidationError):
        PredicateEnvironment(
            scope=PredicateScope.ROLE_PRIVATE,
            role_id="actor",
            source_view_hash=view_hash,
            facts=(judge_only,),
        )
    with pytest.raises(ValidationError):
        PredicateEnvironment(
            scope=PredicateScope.ROLE_PRIVATE,
            source_view_hash=view_hash,
            facts=(),
        )
    with pytest.raises(ValidationError):
        PredicateEnvironment(
            scope=PredicateScope.ADJUDICATOR,
            role_id="actor",
            source_view_hash=view_hash,
            facts=(),
        )


@pytest.mark.parametrize(
    ("fact_id", "declared_type", "literal", "reason"),
    [
        (
            "public.integer",
            LiteralValueType.STRING,
            "10",
            PredicateReasonCode.TYPE_INCOMPATIBLE,
        ),
        (
            "public.integer",
            LiteralValueType.FLOAT,
            10.0,
            PredicateReasonCode.TYPE_INCOMPATIBLE,
        ),
        (
            "public.boolean",
            LiteralValueType.INTEGER,
            1,
            PredicateReasonCode.TYPE_INCOMPATIBLE,
        ),
    ],
)
def test_resolved_fact_type_mismatch_is_unknown(
    fact_id: str,
    declared_type: LiteralValueType,
    literal: Any,
    reason: PredicateReasonCode,
    public_environment: PredicateEnvironment,
) -> None:
    result = evaluate_predicate(
        EqPredicate(
            left=_fact(fact_id, declared_type),
            right=LiteralOperand(value_type=declared_type, value=literal),
        ),
        public_environment,
    )

    assert result.truth is PredicateTruth.UNKNOWN
    assert result.reason_code is reason


def test_declared_operand_mismatch_and_non_numeric_ordering_are_invalid() -> None:
    with pytest.raises(ValidationError):
        EqPredicate(left=_literal(1), right=_literal(1.0))
    with pytest.raises(ValidationError):
        LtPredicate(left=_literal(True), right=_literal(False))
    with pytest.raises(ValidationError):
        GtPredicate(left=_literal("2"), right=_literal("1"))


@pytest.mark.parametrize(
    ("value_type", "value"),
    [
        (LiteralValueType.INTEGER, True),
        (LiteralValueType.INTEGER, 1.0),
        (LiteralValueType.INTEGER, "1"),
        (LiteralValueType.FLOAT, 1),
        (LiteralValueType.FLOAT, "1.0"),
        (LiteralValueType.BOOLEAN, 1),
        (LiteralValueType.STRING, 1),
        (LiteralValueType.INTEGER_TUPLE, [1, 2]),
    ],
)
def test_literal_construction_never_coerces(
    value_type: LiteralValueType,
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        LiteralOperand(value_type=value_type, value=value)


class _IntSubclass(int):
    pass


class _FloatSubclass(float):
    pass


@pytest.mark.parametrize(
    ("value_type", "value"),
    [
        (LiteralValueType.INTEGER, _IntSubclass(1)),
        (LiteralValueType.FLOAT, _FloatSubclass(1.0)),
    ],
)
def test_direct_literal_constructor_rejects_numeric_subclasses(
    value_type: LiteralValueType,
    value: Any,
) -> None:
    with pytest.raises(ValidationError):
        LiteralOperand(value_type=value_type, value=value)
    with pytest.raises(TypeError):
        LiteralOperand.from_value(value)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_literals_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        LiteralOperand(value_type=LiteralValueType.FLOAT, value=value)
    with pytest.raises(ValidationError):
        LiteralOperand.from_value(value)


@pytest.mark.parametrize(
    ("container", "item", "expected"),
    [
        ("alpha beta", "beta", PredicateTruth.TRUE),
        ("alpha beta", "gamma", PredicateTruth.FALSE),
        (("red", "blue"), "blue", PredicateTruth.TRUE),
        (("red", "blue"), "green", PredicateTruth.FALSE),
        ((1, 2, 3), 2, PredicateTruth.TRUE),
        ((1.5, 2.5), 3.5, PredicateTruth.FALSE),
        ((True, False), False, PredicateTruth.TRUE),
    ],
)
def test_contains_supports_only_declared_safe_container_shapes(
    container: Any,
    item: Any,
    expected: PredicateTruth,
    public_environment: PredicateEnvironment,
) -> None:
    predicate = ContainsPredicate(
        container=_literal(container),
        item=_literal(item),
    )
    assert evaluate_predicate(predicate, public_environment).truth is expected


def test_explicitly_typed_empty_tuple_is_safe_and_contains_nothing(
    public_environment: PredicateEnvironment,
) -> None:
    empty = LiteralOperand(
        value_type=LiteralValueType.STRING_TUPLE,
        value=(),
    )
    result = evaluate_predicate(
        ContainsPredicate(container=empty, item=_literal("anything")),
        public_environment,
    )

    assert result.truth is PredicateTruth.FALSE
    with pytest.raises(ValueError):
        LiteralOperand.from_value(())


def test_mixed_fact_container_fails_closed(
    public_environment: PredicateEnvironment,
) -> None:
    result = evaluate_predicate(
        ContainsPredicate(
            container=_fact("public.mixed", LiteralValueType.INTEGER_TUPLE),
            item=_literal(1),
        ),
        public_environment,
    )

    assert result.truth is PredicateTruth.UNKNOWN
    assert result.reason_code is PredicateReasonCode.TYPE_INCOMPATIBLE


@pytest.mark.parametrize(
    ("container", "item"),
    [
        (_literal(10), _literal(1)),
        (_literal(True), _literal(True)),
        (_literal((1, 2)), _literal(1.0)),
        (_literal("abc"), _literal(1)),
    ],
)
def test_contains_rejects_unsafe_or_mismatched_declared_shapes(
    container: LiteralOperand,
    item: LiteralOperand,
) -> None:
    with pytest.raises(ValidationError):
        ContainsPredicate(container=container, item=item)


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        ("text", LiteralValueType.STRING),
        (1, LiteralValueType.INTEGER),
        (1.5, LiteralValueType.FLOAT),
        (True, LiteralValueType.BOOLEAN),
        (("a", "b"), LiteralValueType.STRING_TUPLE),
        ((1, 2), LiteralValueType.INTEGER_TUPLE),
        ((1.0, 2.0), LiteralValueType.FLOAT_TUPLE),
        ((True, False), LiteralValueType.BOOLEAN_TUPLE),
    ],
)
def test_literal_factory_infers_exact_supported_type(
    value: Any,
    expected_type: LiteralValueType,
) -> None:
    literal = LiteralOperand.from_value(value)

    assert literal.value_type is expected_type
    assert literal.value == value
    assert type(literal.value) is type(value)


@pytest.mark.parametrize("value", ((), (1, "mixed"), [], {}, {1, 2}, object()))
def test_literal_factory_rejects_ambiguous_or_mutable_values(value: Any) -> None:
    with pytest.raises((TypeError, ValueError, ValidationError)):
        LiteralOperand.from_value(value)


def _nested_predicate() -> AllPredicate:
    return AllPredicate(
        predicates=(
            EqPredicate(
                left=_fact("public.integer", LiteralValueType.INTEGER),
                right=_literal(10),
            ),
            NotPredicate(
                predicate=ContainsPredicate(
                    container=_fact("public.text", LiteralValueType.STRING),
                    item=_literal("gamma"),
                )
            ),
        )
    )


def test_predicate_identity_and_canonical_json_are_stable_and_content_sensitive() -> None:
    first = _nested_predicate()
    second = _nested_predicate()
    changed_child = EqPredicate(
        left=_fact("public.integer", LiteralValueType.INTEGER),
        right=_literal(11),
    )
    changed = AllPredicate(predicates=(changed_child, first.predicates[1]))

    assert first.predicate_id == second.predicate_id
    assert first.predicate_id.startswith("predicate_")
    assert len(first.predicate_id) == len("predicate_") + 64
    assert first.predicate_id != changed.predicate_id
    assert predicate_canonical_json(first) == predicate_canonical_json(second)
    raw = predicate_canonical_json(first)
    assert raw == json.dumps(
        json.loads(raw),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def test_nested_predicate_environment_and_result_round_trip(
    actor_environment: PredicateEnvironment,
) -> None:
    predicate = _nested_predicate()
    result = evaluate_predicate(predicate, actor_environment)

    restored_predicate = predicate_from_persisted_json(
        predicate_canonical_json(predicate)
    )
    restored_environment = PredicateEnvironment.from_persisted_json(
        actor_environment.canonical_json()
    )
    restored_result = PredicateResult.from_persisted_json(result.canonical_json())

    assert restored_predicate == predicate
    assert predicate_from_persisted(
        json.loads(predicate_canonical_json(predicate))
    ) == predicate
    assert restored_environment == actor_environment
    assert restored_result == result
    assert evaluate_predicate(restored_predicate, restored_environment) == result


def _tamper_nested_predicate(payload: dict[str, Any], case: str) -> None:
    first_child = payload["predicates"][0]
    second_child = payload["predicates"][1]
    grandchild = second_child["predicate"]
    if case == "top-content":
        payload["op"] = "any"
    elif case == "child-content":
        first_child["right"]["value"] = 999
    elif case == "grandchild-content":
        grandchild["item"]["value"] = "beta"
    elif case == "top-id":
        payload["predicate_id"] = "predicate_" + "0" * 64
    elif case == "child-id":
        first_child["predicate_id"] = "predicate_" + "0" * 64
    elif case == "grandchild-id":
        grandchild["predicate_id"] = "predicate_" + "0" * 64
    elif case == "top-missing-version":
        del payload["schema_version"]
    elif case == "child-missing-version":
        del first_child["schema_version"]
    elif case == "grandchild-missing-version":
        del grandchild["schema_version"]
    elif case == "top-missing-id":
        del payload["predicate_id"]
    elif case == "child-missing-id":
        del first_child["predicate_id"]
    elif case == "grandchild-missing-id":
        del grandchild["predicate_id"]
    elif case == "top-unknown-operator":
        payload["op"] = "execute"
    elif case == "child-unknown-operator":
        first_child["op"] = "execute"
    elif case == "grandchild-unknown-operator":
        grandchild["op"] = "execute"
    elif case == "top-extra":
        payload["source_text"] = "1 == 1"
    elif case == "child-extra":
        first_child["callable"] = "builtins.eval"
    elif case == "grandchild-extra":
        grandchild["unexpected"] = True
    elif case == "operand-extra":
        first_child["left"]["unexpected"] = True
    elif case == "operand-unknown-kind":
        first_child["left"]["kind"] = "python_source"
    else:
        raise AssertionError(f"unknown tamper case {case!r}")


@pytest.mark.parametrize(
    "case",
    (
        "top-content",
        "child-content",
        "grandchild-content",
        "top-id",
        "child-id",
        "grandchild-id",
        "top-missing-version",
        "child-missing-version",
        "grandchild-missing-version",
        "top-missing-id",
        "child-missing-id",
        "grandchild-missing-id",
        "top-unknown-operator",
        "child-unknown-operator",
        "grandchild-unknown-operator",
        "top-extra",
        "child-extra",
        "grandchild-extra",
        "operand-extra",
        "operand-unknown-kind",
    ),
)
def test_every_nested_persisted_predicate_tamper_is_rejected(case: str) -> None:
    payload = json.loads(predicate_canonical_json(_nested_predicate()))
    _tamper_nested_predicate(payload, case)

    with pytest.raises(ValidationError):
        predicate_from_persisted_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("record_kind", "case"),
    [
        ("environment", "content"),
        ("environment", "id"),
        ("environment", "missing-version"),
        ("environment", "missing-id"),
        ("environment", "extra"),
        ("result", "content"),
        ("result", "id"),
        ("result", "missing-version"),
        ("result", "missing-id"),
        ("result", "extra"),
    ],
)
def test_environment_and_result_persisted_tamper_is_rejected(
    record_kind: str,
    case: str,
    actor_environment: PredicateEnvironment,
) -> None:
    result = evaluate_predicate(_nested_predicate(), actor_environment)
    record = actor_environment if record_kind == "environment" else result
    payload = json.loads(record.canonical_json())
    identity_field = "environment_id" if record_kind == "environment" else "result_id"

    if case == "content":
        if record_kind == "environment":
            payload["role_id"] = "counterpart"
        else:
            payload["input_fact_ids"] = ["different.fact"]
    elif case == "id":
        payload[identity_field] = identity_field + "_tampered"
    elif case == "missing-version":
        del payload["schema_version"]
    elif case == "missing-id":
        del payload[identity_field]
    elif case == "extra":
        payload["unexpected"] = True
    else:
        raise AssertionError(case)

    with pytest.raises(ValidationError):
        if record_kind == "environment":
            PredicateEnvironment.from_persisted(payload)
        else:
            PredicateResult.from_persisted(payload)


def test_supplied_identity_must_match_content(
    actor_environment: PredicateEnvironment,
) -> None:
    with pytest.raises(ValidationError):
        EqPredicate(
            left=_literal(1),
            right=_literal(1),
            predicate_id="predicate_" + "0" * 64,
        )
    result = evaluate_predicate(_nested_predicate(), actor_environment)
    with pytest.raises(ValidationError):
        PredicateResult(
            predicate_id=result.predicate_id,
            environment_id=result.environment_id,
            truth=result.truth,
            reason_code=result.reason_code,
            input_fact_ids=result.input_fact_ids,
            result_id="predicate_result_" + "0" * 64,
        )


def test_models_are_frozen_and_persisted_input_has_no_mutable_aliases(
    actor_environment: PredicateEnvironment,
) -> None:
    predicate = _nested_predicate()
    result = evaluate_predicate(predicate, actor_environment)
    payload = json.loads(predicate_canonical_json(predicate))
    restored = predicate_from_persisted(payload)

    with pytest.raises(ValidationError):
        predicate.predicate_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        actor_environment.facts = ()  # type: ignore[misc]
    with pytest.raises(ValidationError):
        result.input_fact_ids = ()  # type: ignore[misc]

    payload["predicates"][0]["right"]["value"] = 999
    assert restored == predicate
    assert restored.predicates[0].right.value == 10


def test_mutable_literal_aliases_are_rejected() -> None:
    mutable_values = ([1, 2], {"one": 1}, {1, 2})
    for value in mutable_values:
        with pytest.raises(ValidationError):
            LiteralOperand(
                value_type=LiteralValueType.INTEGER_TUPLE,
                value=value,
            )


def test_inert_source_like_strings_cannot_execute_or_import(
    tmp_path: Path,
    public_environment: PredicateEnvironment,
) -> None:
    sentinel = tmp_path / "predicate-source-executed"
    source_like_text = (
        "__import__('pathlib').Path(" + repr(str(sentinel)) + ").write_text('owned')"
    )
    predicate = EqPredicate(
        left=_literal(source_like_text),
        right=_literal(source_like_text),
    )

    assert evaluate_predicate(predicate, public_environment).truth is PredicateTruth.TRUE
    assert not sentinel.exists()
    with pytest.raises((TypeError, ValidationError)):
        LiteralOperand.from_value(lambda: sentinel.touch())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        LiteralOperand(
            value_type=LiteralValueType.STRING,
            value=lambda: sentinel.touch(),  # type: ignore[arg-type]
        )
    assert not sentinel.exists()


def test_adapter_schema_and_dispatch_are_closed_and_exhaustive(
    public_environment: PredicateEnvironment,
) -> None:
    schema = PREDICATE_ADAPTER.json_schema()
    mapping = schema["discriminator"]["mapping"]

    assert tuple(sorted(mapping)) == tuple(sorted(OPERATORS))
    for operator in OPERATORS:
        predicate = _operator_case(operator, PredicateTruth.TRUE)
        restored = predicate_from_persisted_json(predicate_canonical_json(predicate))
        assert restored.op == operator
        assert evaluate_predicate(restored, public_environment).truth is PredicateTruth.TRUE


@pytest.mark.parametrize(
    "payload",
    [
        {"op": "execute"},
        {"op": "eq", "source_text": "1 == 1"},
        {"op": "all", "predicates": []},
        {"op": "not", "predicate": {"op": "execute"}},
    ],
)
def test_adapter_rejects_unknown_extra_and_incomplete_ast(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        PREDICATE_ADAPTER.validate_json(json.dumps(payload))
