"""Permanent contracts for deterministic scenario compilation and rendering."""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
from itertools import product
import json
import random
from typing import Any

import pytest
from pydantic import ValidationError

import interpretability.scenarios.compiler as compiler_module
from interpretability.scenarios.compiler import (
    RenderedPrompt,
    compile_scenario,
    derive_compilation_seed,
    render_prompt,
    validate_template_placeholders,
)
from interpretability.scenarios.schema import (
    ActionDefinition,
    ActionKind,
    AgreementState,
    BehaviorSubtype,
    BehaviorTargetDefinition,
    ConditionDefinition,
    ExtractorReference,
    FactDefinition,
    FactSamplingDefinition,
    FactSamplingKind,
    FactValueType,
    IncentiveCondition,
    InterventionPoint,
    OutcomeDefinition,
    PromptKind,
    PromptTemplate,
    RoleDefinition,
    RoleKind,
    RuleReference,
    ScenarioInstance,
    ScenarioMetadata,
    ScenarioSpec,
    Visibility,
    canonical_sha256,
)


SCHEMA_ERRORS = (ValidationError, TypeError, ValueError)


def _compiler_spec() -> ScenarioSpec:
    roles = (
        RoleDefinition(
            role_id="counterpart",
            kind=RoleKind.COUNTERPART,
            description="Negotiating counterpart.",
        ),
        RoleDefinition(
            role_id="adjudicator",
            kind=RoleKind.ADJUDICATOR,
            description="Access-controlled evaluator.",
        ),
        RoleDefinition(
            role_id="actor",
            kind=RoleKind.ACTOR,
            description="Acting negotiator.",
        ),
    )
    facts = (
        FactDefinition(
            fact_id="judge_only",
            fact_version="judge/1",
            value_type=FactValueType.STRING,
            visibility=Visibility.ADJUDICATOR_ONLY,
            description="Adjudicator control value.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.FIXED,
                fixed_value="audit-A",
            ),
        ),
        FactDefinition(
            fact_id="fixed_public",
            fact_version="fixed/1",
            value_type=FactValueType.INTEGER,
            visibility=Visibility.PUBLIC,
            description="Public fixed pool.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.FIXED,
                fixed_value=100,
            ),
        ),
        FactDefinition(
            fact_id="int_private",
            fact_version="integer-range/1",
            value_type=FactValueType.INTEGER,
            visibility=Visibility.ROLE_PRIVATE,
            visible_to=("actor",),
            description="Actor private integer.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.INTEGER_RANGE,
                minimum=20,
                maximum=50,
            ),
        ),
        FactDefinition(
            fact_id="float_public",
            fact_version="float-range/1",
            value_type=FactValueType.FLOAT,
            visibility=Visibility.PUBLIC,
            description="Public floating rate.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.FLOAT_RANGE,
                minimum=0.25,
                maximum=0.75,
            ),
        ),
        FactDefinition(
            fact_id="choice_private",
            fact_version="choice/1",
            value_type=FactValueType.STRING,
            visibility=Visibility.ROLE_PRIVATE,
            visible_to=("counterpart",),
            description="Counterpart private category.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.CHOICE,
                choices=("low", "mid", "high"),
            ),
        ),
        FactDefinition(
            fact_id="bool_public",
            fact_version="boolean/1",
            value_type=FactValueType.BOOLEAN,
            visibility=Visibility.PUBLIC,
            description="Public Boolean flag.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.FIXED,
                fixed_value=True,
            ),
        ),
        FactDefinition(
            fact_id="text_public",
            fact_version="text/1",
            value_type=FactValueType.STRING,
            visibility=Visibility.PUBLIC,
            description="Public text label.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.FIXED,
                fixed_value="alpha",
            ),
        ),
    )
    prompts = (
        PromptTemplate(
            template_id="actor.minimal.initial",
            template_version="prompt/1",
            role_id="actor",
            condition=IncentiveCondition.MINIMAL,
            kind=PromptKind.INITIAL,
            template=(
                "Actor {scenario_id}/{trial_id}/{condition}: {{contract}}; "
                "pool {fixed_public}/{fixed_public}; reserve {int_private}; "
                "rate {float_public}; flag {bool_public}; label {text_public}."
            ),
        ),
        PromptTemplate(
            template_id="counterpart.minimal",
            template_version="prompt/1",
            role_id="counterpart",
            condition=IncentiveCondition.MINIMAL,
            kind=PromptKind.COUNTERPART,
            template=(
                "Pool {fixed_public}; category {choice_private}; "
                "rate {float_public}; flag {bool_public}."
            ),
        ),
        PromptTemplate(
            template_id="actor.high.initial",
            template_version="prompt/1",
            role_id="actor",
            condition=IncentiveCondition.HIGH_INCENTIVE,
            kind=PromptKind.INITIAL,
            template="High condition {condition}: {fixed_public}/{int_private}.",
        ),
        PromptTemplate(
            template_id="counterpart.high",
            template_version="prompt/1",
            role_id="counterpart",
            condition=IncentiveCondition.HIGH_INCENTIVE,
            kind=PromptKind.COUNTERPART,
            template="High condition: {fixed_public}/{choice_private}.",
        ),
        PromptTemplate(
            template_id="actor.intervention",
            template_version="prompt/1",
            role_id="actor",
            condition=IncentiveCondition.HIGH_INCENTIVE,
            kind=PromptKind.INTERVENTION,
            template="Intervention for {scenario_id} at {trial_id}.",
        ),
        PromptTemplate(
            template_id="adjudicator.verification",
            template_version="prompt/1",
            role_id="adjudicator",
            condition=IncentiveCondition.MINIMAL,
            kind=PromptKind.VERIFICATION,
            template="Verify {judge_only}, {int_private}, and {choice_private}.",
        ),
    )
    rules = (
        RuleReference(
            rule_id="feasible",
            rule_version="rule/1",
            predicate_id="predicate.feasible",
            input_fact_ids=("fixed_public", "int_private", "choice_private"),
            description="Fixture feasibility rule.",
        ),
    )
    return ScenarioSpec(
        spec_version="1.0.0",
        metadata=ScenarioMetadata(
            scenario_id="compiler_fixture",
            display_name="Compiler fixture",
            description="Strict compiler contract fixture.",
            research_constructs=(BehaviorSubtype.FALSE_CLAIM,),
            tags=("compiler",),
        ),
        roles=roles,
        facts=facts,
        conditions=(
            ConditionDefinition(
                condition=IncentiveCondition.MINIMAL,
                description="Minimal condition.",
            ),
            ConditionDefinition(
                condition=IncentiveCondition.HIGH_INCENTIVE,
                description="High incentive condition.",
            ),
        ),
        prompt_templates=prompts,
        intervention_points=(
            InterventionPoint(
                intervention_id="late_notice",
                round_index=3,
                recipient_role_id="actor",
                template_id="actor.intervention",
            ),
            InterventionPoint(
                intervention_id="early_notice",
                round_index=1,
                recipient_role_id="actor",
                template_id="actor.intervention",
            ),
        ),
        action_space=(
            ActionDefinition(
                action_id="offer",
                kind=ActionKind.OFFER,
                actor_role_ids=("actor", "counterpart"),
                description="Make an offer.",
            ),
            ActionDefinition(
                action_id="message",
                kind=ActionKind.MESSAGE,
                actor_role_ids=("counterpart", "actor"),
                description="Send a message.",
            ),
            ActionDefinition(
                action_id="walk_away",
                kind=ActionKind.WALK_AWAY,
                actor_role_ids=("actor", "counterpart"),
                terminal=True,
                description="End negotiation.",
            ),
        ),
        extractors=(
            ExtractorReference(
                extractor_name="compiler_action",
                extractor_version="extractor/1",
                supported_action_kinds=(ActionKind.MESSAGE, ActionKind.OFFER),
                deterministic=True,
            ),
        ),
        rules=rules,
        behavior_targets=(
            BehaviorTargetDefinition(
                target_id="false_claim",
                subtype=BehaviorSubtype.FALSE_CLAIM,
                rule_ids=("feasible",),
                belief_dependent=True,
                default_severity=1.0,
            ),
        ),
        outcomes=(
            OutcomeDefinition(
                outcome_id="complete",
                rule_ids=("feasible",),
                agreement_state=AgreementState.AGREEMENT,
                utility_role_ids=("actor", "counterpart"),
                description="Compiled outcome.",
            ),
        ),
    )


@pytest.fixture(scope="module")
def compiler_spec() -> ScenarioSpec:
    return _compiler_spec()


@pytest.fixture(scope="module")
def compiled_instance(compiler_spec: ScenarioSpec) -> ScenarioInstance:
    return compile_scenario(
        compiler_spec,
        "trial-7",
        7,
        IncentiveCondition.MINIMAL,
    )


def _copy_spec(spec: ScenarioSpec, **changes: Any) -> ScenarioSpec:
    fields: dict[str, Any] = {
        "spec_version": spec.spec_version,
        "metadata": spec.metadata,
        "roles": spec.roles,
        "facts": spec.facts,
        "conditions": spec.conditions,
        "prompt_templates": spec.prompt_templates,
        "intervention_points": spec.intervention_points,
        "action_space": spec.action_space,
        "extractors": spec.extractors,
        "rules": spec.rules,
        "behavior_targets": spec.behavior_targets,
        "outcomes": spec.outcomes,
    }
    fields.update(changes)
    return ScenarioSpec(**fields)


def _replace_fact(spec: ScenarioSpec, replacement: FactDefinition) -> ScenarioSpec:
    return _copy_spec(
        spec,
        facts=tuple(
            replacement if fact.fact_id == replacement.fact_id else fact
            for fact in spec.facts
        ),
    )


def _replace_prompt(
    spec: ScenarioSpec,
    template_id: str,
    text: str,
) -> ScenarioSpec:
    templates = []
    for prompt in spec.prompt_templates:
        if prompt.template_id == template_id:
            prompt = PromptTemplate(
                template_id=prompt.template_id,
                template_version=prompt.template_version,
                role_id=prompt.role_id,
                condition=prompt.condition,
                kind=prompt.kind,
                template=text,
            )
        templates.append(prompt)
    return _copy_spec(spec, prompt_templates=tuple(templates))


def _fact_values(instance: ScenarioInstance) -> dict[str, Any]:
    return {fact.fact_id: fact.value for fact in instance.resolved_facts}


def test_all_sampling_kinds_resolve_once_with_exact_declared_types(
    compiler_spec: ScenarioSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = compiler_module._resolve_fact  # pylint: disable=protected-access

    def tracked(definition: FactDefinition, rng: random.Random) -> Any:
        calls.append(definition.fact_id)
        return original(definition, rng)

    monkeypatch.setattr(compiler_module, "_resolve_fact", tracked)
    instance = compile_scenario(
        compiler_spec,
        "trial-once",
        91,
        IncentiveCondition.MINIMAL,
    )
    values = _fact_values(instance)

    assert Counter(calls) == Counter({fact.fact_id: 1 for fact in compiler_spec.facts})
    assert values["fixed_public"] == 100
    assert type(values["fixed_public"]) is int
    assert 20 <= values["int_private"] <= 50
    assert type(values["int_private"]) is int
    assert 0.25 <= values["float_public"] <= 0.75
    assert type(values["float_public"]) is float
    assert values["choice_private"] in {"low", "mid", "high"}
    assert type(values["choice_private"]) is str
    assert values["bool_public"] is True
    assert values["text_public"] == "alpha"


def test_range_and_choice_endpoints_are_inclusive(compiler_spec: ScenarioSpec) -> None:
    integer_endpoint = FactDefinition(
        fact_id="int_private",
        fact_version="integer-range/endpoint",
        value_type=FactValueType.INTEGER,
        visibility=Visibility.ROLE_PRIVATE,
        visible_to=("actor",),
        description="Fixed integer range endpoint.",
        sampling=FactSamplingDefinition(
            kind=FactSamplingKind.INTEGER_RANGE,
            minimum=37,
            maximum=37,
        ),
    )
    float_endpoint = FactDefinition(
        fact_id="float_public",
        fact_version="float-range/endpoint",
        value_type=FactValueType.FLOAT,
        visibility=Visibility.PUBLIC,
        description="Fixed float range endpoint.",
        sampling=FactSamplingDefinition(
            kind=FactSamplingKind.FLOAT_RANGE,
            minimum=0.5,
            maximum=0.5,
        ),
    )
    choice_endpoint = FactDefinition(
        fact_id="choice_private",
        fact_version="choice/endpoint",
        value_type=FactValueType.STRING,
        visibility=Visibility.ROLE_PRIVATE,
        visible_to=("counterpart",),
        description="Single choice endpoint.",
        sampling=FactSamplingDefinition(
            kind=FactSamplingKind.CHOICE,
            choices=("only",),
        ),
    )
    endpoint_spec = _replace_fact(
        _replace_fact(
            _replace_fact(compiler_spec, integer_endpoint),
            float_endpoint,
        ),
        choice_endpoint,
    )

    values = _fact_values(
        compile_scenario(
            endpoint_spec,
            "trial-endpoints",
            1,
            IncentiveCondition.MINIMAL,
        )
    )

    assert values["int_private"] == 37
    assert values["float_public"] == 0.5
    assert values["choice_private"] == "only"


@pytest.mark.parametrize(
    "case",
    (
        "fixed-int-for-float",
        "integer-range-for-float",
        "float-range-for-integer",
        "float-range-int-bounds",
        "choice-mixed-types",
        "choice-bool-for-integer",
    ),
)
def test_sampling_type_mismatch_fails_closed(
    case: str,
    compiler_spec: ScenarioSpec,
) -> None:
    if case == "fixed-int-for-float":
        fact = FactDefinition(
            fact_id="float_public",
            fact_version="bad/1",
            value_type=FactValueType.FLOAT,
            visibility=Visibility.PUBLIC,
            description="Invalid fixed type.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.FIXED,
                fixed_value=1,
            ),
        )
    elif case == "integer-range-for-float":
        fact = FactDefinition(
            fact_id="float_public",
            fact_version="bad/1",
            value_type=FactValueType.FLOAT,
            visibility=Visibility.PUBLIC,
            description="Invalid integer range declaration.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.INTEGER_RANGE,
                minimum=1,
                maximum=2,
            ),
        )
    elif case == "float-range-for-integer":
        fact = FactDefinition(
            fact_id="float_public",
            fact_version="bad/1",
            value_type=FactValueType.INTEGER,
            visibility=Visibility.PUBLIC,
            description="Invalid float range declaration.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.FLOAT_RANGE,
                minimum=1.0,
                maximum=2.0,
            ),
        )
    elif case == "float-range-int-bounds":
        fact = FactDefinition(
            fact_id="float_public",
            fact_version="bad/1",
            value_type=FactValueType.FLOAT,
            visibility=Visibility.PUBLIC,
            description="Invalid float range bounds.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.FLOAT_RANGE,
                minimum=1,
                maximum=2,
            ),
        )
    elif case == "choice-mixed-types":
        fact = FactDefinition(
            fact_id="choice_private",
            fact_version="bad/1",
            value_type=FactValueType.STRING,
            visibility=Visibility.ROLE_PRIVATE,
            visible_to=("counterpart",),
            description="Invalid mixed choices.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.CHOICE,
                choices=("one", 2),
            ),
        )
    elif case == "choice-bool-for-integer":
        fact = FactDefinition(
            fact_id="int_private",
            fact_version="bad/1",
            value_type=FactValueType.INTEGER,
            visibility=Visibility.ROLE_PRIVATE,
            visible_to=("actor",),
            description="Invalid Boolean choice.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.CHOICE,
                choices=(True, False),
            ),
        )
    else:
        raise AssertionError(case)

    with pytest.raises(ValueError):
        compile_scenario(
            _replace_fact(compiler_spec, fact),
            "trial-bad-type",
            1,
            IncentiveCondition.MINIMAL,
        )


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_float_range_is_rejected_by_strict_schema(value: float) -> None:
    with pytest.raises(ValidationError):
        FactSamplingDefinition(
            kind=FactSamplingKind.FLOAT_RANGE,
            minimum=value,
            maximum=1.0,
        )


def test_same_inputs_have_identical_canonical_instance_and_hash(
    compiler_spec: ScenarioSpec,
) -> None:
    first = compile_scenario(
        compiler_spec,
        "trial-stable",
        1234,
        IncentiveCondition.MINIMAL,
    )
    second = compile_scenario(
        compiler_spec,
        "trial-stable",
        1234,
        IncentiveCondition.MINIMAL,
    )

    assert first == second
    assert first.instance_hash == second.instance_hash
    assert first.canonical_json() == second.canonical_json()
    assert first.spec_hash == compiler_spec.spec_hash
    assert first.spec_version == compiler_spec.spec_version
    assert first.scenario_id == compiler_spec.metadata.scenario_id


def test_trial_seed_condition_and_spec_are_domain_separated(
    compiler_spec: ScenarioSpec,
) -> None:
    changed_fixed = FactDefinition(
        fact_id="fixed_public",
        fact_version="fixed/2",
        value_type=FactValueType.INTEGER,
        visibility=Visibility.PUBLIC,
        description="Changed public fixed pool.",
        sampling=FactSamplingDefinition(
            kind=FactSamplingKind.FIXED,
            fixed_value=101,
        ),
    )
    changed_spec = _replace_fact(compiler_spec, changed_fixed)
    identities = (
        (compiler_spec, "trial-a", 1, IncentiveCondition.MINIMAL),
        (compiler_spec, "trial-b", 1, IncentiveCondition.MINIMAL),
        (compiler_spec, "trial-a", 2, IncentiveCondition.MINIMAL),
        (compiler_spec, "trial-a", 1, IncentiveCondition.HIGH_INCENTIVE),
        (changed_spec, "trial-a", 1, IncentiveCondition.MINIMAL),
    )

    seeds = {derive_compilation_seed(*identity) for identity in identities}
    hashes = {compile_scenario(*identity).instance_hash for identity in identities}

    assert len(seeds) == len(identities)
    assert len(hashes) == len(identities)


def test_compilation_never_mutates_process_global_rng(
    compiler_spec: ScenarioSpec,
) -> None:
    random.seed(812_881)
    before = random.getstate()

    for index in range(20):
        compile_scenario(
            compiler_spec,
            f"trial-global-{index}",
            index,
            IncentiveCondition.MINIMAL,
        )

    assert random.getstate() == before


def test_views_are_complete_ordered_and_do_not_leak(
    compiler_spec: ScenarioSpec,
    compiled_instance: ScenarioInstance,
) -> None:
    assert tuple(fact.fact_id for fact in compiled_instance.resolved_facts) == tuple(
        fact.fact_id for fact in compiler_spec.facts
    )
    assert tuple(fact.fact_id for fact in compiled_instance.public_state.facts) == (
        "fixed_public",
        "float_public",
        "bool_public",
        "text_public",
    )
    assert tuple(view.role_id for view in compiled_instance.private_views) == (
        "counterpart",
        "actor",
    )
    counterpart, actor = compiled_instance.private_views
    counterpart_ids = tuple(fact.fact_id for fact in counterpart.facts)
    actor_ids = tuple(fact.fact_id for fact in actor.facts)

    assert counterpart_ids == (
        "fixed_public",
        "float_public",
        "choice_private",
        "bool_public",
        "text_public",
    )
    assert actor_ids == (
        "fixed_public",
        "int_private",
        "float_public",
        "bool_public",
        "text_public",
    )
    assert "int_private" not in counterpart_ids
    assert "choice_private" not in actor_ids
    assert "judge_only" not in counterpart_ids
    assert "judge_only" not in actor_ids
    assert compiled_instance.adjudicator_view.facts == compiled_instance.resolved_facts


def test_role_action_and_intervention_order_and_identity_are_preserved(
    compiler_spec: ScenarioSpec,
    compiled_instance: ScenarioInstance,
) -> None:
    assert tuple(view.role_id for view in compiled_instance.private_views) == (
        "counterpart",
        "actor",
    )
    assert compiled_instance.legal_action_ids == (
        "offer",
        "message",
        "walk_away",
    )
    assert tuple(
        intervention.intervention_id
        for intervention in compiled_instance.interventions
    ) == ("late_notice", "early_notice")
    assert tuple(
        intervention.round_index for intervention in compiled_instance.interventions
    ) == (3, 1)
    assert tuple(
        intervention.template_id for intervention in compiled_instance.interventions
    ) == ("actor.intervention", "actor.intervention")

    recompiled = compile_scenario(
        compiler_spec,
        compiled_instance.trial_id,
        compiled_instance.run_seed,
        compiled_instance.condition,
    )
    assert tuple(
        item.scheduled_intervention_id for item in compiled_instance.interventions
    ) == tuple(item.scheduled_intervention_id for item in recompiled.interventions)


def test_rendering_never_constructs_rng_or_resamples(
    compiler_spec: ScenarioSpec,
    compiled_instance: ScenarioInstance,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_rng(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("rendering attempted to construct an RNG")

    monkeypatch.setattr(compiler_module.random, "Random", fail_rng)
    first = render_prompt(
        compiler_spec,
        compiled_instance,
        "actor",
        PromptKind.INITIAL,
    )
    second = render_prompt(
        compiler_spec,
        compiled_instance,
        "actor",
        PromptKind.INITIAL,
    )

    assert first == second


def test_spec_and_instance_linkage_and_tampering_fail_closed(
    compiler_spec: ScenarioSpec,
    compiled_instance: ScenarioInstance,
) -> None:
    stale_spec = compiler_spec.model_copy(
        update={"spec_hash": "sha256:" + "0" * 64}
    )
    stale_instance = compiled_instance.model_copy(update={"trial_id": "trial-stale"})
    wrong_actions = ScenarioInstance(
        scenario_id=compiled_instance.scenario_id,
        spec_version=compiled_instance.spec_version,
        spec_hash=compiled_instance.spec_hash,
        run_seed=compiled_instance.run_seed,
        trial_id=compiled_instance.trial_id,
        condition=compiled_instance.condition,
        resolved_facts=compiled_instance.resolved_facts,
        public_state=compiled_instance.public_state,
        private_views=compiled_instance.private_views,
        adjudicator_view=compiled_instance.adjudicator_view,
        legal_action_ids=("message",),
        interventions=compiled_instance.interventions,
    )
    changed_fixed = FactDefinition(
        fact_id="fixed_public",
        fact_version="fixed/2",
        value_type=FactValueType.INTEGER,
        visibility=Visibility.PUBLIC,
        description="Changed linkage fact.",
        sampling=FactSamplingDefinition(
            kind=FactSamplingKind.FIXED,
            fixed_value=101,
        ),
    )
    different_spec = _replace_fact(compiler_spec, changed_fixed)

    with pytest.raises(ValidationError):
        compile_scenario(
            stale_spec,
            "trial-7",
            7,
            IncentiveCondition.MINIMAL,
        )
    with pytest.raises(ValidationError):
        render_prompt(
            compiler_spec,
            stale_instance,
            "actor",
            PromptKind.INITIAL,
        )
    with pytest.raises(ValueError, match="legal actions"):
        render_prompt(
            compiler_spec,
            wrong_actions,
            "actor",
            PromptKind.INITIAL,
        )
    with pytest.raises(ValueError, match="does not match"):
        render_prompt(
            different_spec,
            compiled_instance,
            "actor",
            PromptKind.INITIAL,
        )


@pytest.mark.parametrize(
    ("trial_id", "run_seed", "condition"),
    [
        ("", 1, IncentiveCondition.MINIMAL),
        ("7", 1, IncentiveCondition.MINIMAL),
        ("trial", -1, IncentiveCondition.MINIMAL),
        ("trial", True, IncentiveCondition.MINIMAL),
        ("trial", 1, "minimal"),
        ("trial", 1, IncentiveCondition.LOW_INCENTIVE),
    ],
)
def test_invalid_run_identity_or_condition_fails_closed(
    trial_id: Any,
    run_seed: Any,
    condition: Any,
    compiler_spec: ScenarioSpec,
) -> None:
    with pytest.raises(SCHEMA_ERRORS):
        compile_scenario(compiler_spec, trial_id, run_seed, condition)


def test_placeholder_parser_supports_escaped_repeated_and_context_fields() -> None:
    template = (
        "{{literal}} {fixed_public} {fixed_public} "
        "{scenario_id}/{trial_id}/{condition}"
    )

    assert validate_template_placeholders(template) == (
        "fixed_public",
        "fixed_public",
        "scenario_id",
        "trial_id",
        "condition",
    )


@pytest.mark.parametrize(
    "template",
    (
        "{fact.value}",
        "{fact[0]}",
        "{0}",
        "{}",
        "{fact!r}",
        "{fact!s}",
        "{fact:>10}",
        "{fact:.2f}",
        "{fact:{width}}",
        "{fact",
        "fact}",
    ),
)
def test_placeholder_parser_rejects_traversal_conversion_and_malformed_braces(
    template: str,
) -> None:
    with pytest.raises(ValueError):
        validate_template_placeholders(template)


@pytest.mark.parametrize("case", ("unknown", "unauthorized", "dotted"))
def test_compilation_rejects_unknown_unauthorized_and_dotted_placeholders(
    case: str,
    compiler_spec: ScenarioSpec,
) -> None:
    if case == "unknown":
        spec = _replace_prompt(
            compiler_spec,
            "actor.minimal.initial",
            "Unknown {missing_fact}.",
        )
    elif case == "unauthorized":
        spec = _replace_prompt(
            compiler_spec,
            "actor.minimal.initial",
            "Counterpart secret {choice_private}.",
        )
    elif case == "dotted":
        dotted = FactDefinition(
            fact_id="dotted.fact",
            fact_version="dotted/1",
            value_type=FactValueType.STRING,
            visibility=Visibility.PUBLIC,
            description="Valid fact ID unsafe as a format placeholder.",
            sampling=FactSamplingDefinition(
                kind=FactSamplingKind.FIXED,
                fixed_value="value",
            ),
        )
        spec = _copy_spec(compiler_spec, facts=(*compiler_spec.facts, dotted))
        spec = _replace_prompt(
            spec,
            "actor.minimal.initial",
            "Dotted {dotted.fact}.",
        )
    else:
        raise AssertionError(case)

    with pytest.raises((ValueError, PermissionError)):
        compile_scenario(
            spec,
            "trial-placeholder",
            1,
            IncentiveCondition.MINIMAL,
        )


def test_dotted_fact_id_is_valid_when_not_used_as_format_traversal(
    compiler_spec: ScenarioSpec,
) -> None:
    dotted = FactDefinition(
        fact_id="dotted.fact",
        fact_version="dotted/1",
        value_type=FactValueType.STRING,
        visibility=Visibility.PUBLIC,
        description="Unused dotted fact.",
        sampling=FactSamplingDefinition(
            kind=FactSamplingKind.FIXED,
            fixed_value="safe-data",
        ),
    )
    spec = _copy_spec(compiler_spec, facts=(*compiler_spec.facts, dotted))

    instance = compile_scenario(
        spec,
        "trial-dotted",
        1,
        IncentiveCondition.MINIMAL,
    )

    assert _fact_values(instance)["dotted.fact"] == "safe-data"


def test_fact_id_cannot_shadow_prompt_context_field(
    compiler_spec: ScenarioSpec,
) -> None:
    collision = FactDefinition(
        fact_id="condition",
        fact_version="collision/1",
        value_type=FactValueType.STRING,
        visibility=Visibility.PUBLIC,
        description="Context collision.",
        sampling=FactSamplingDefinition(
            kind=FactSamplingKind.FIXED,
            fixed_value="shadow",
        ),
    )
    collision_spec = _copy_spec(
        compiler_spec,
        facts=(*compiler_spec.facts, collision),
    )

    with pytest.raises(ValueError, match="shadow"):
        compile_scenario(
            collision_spec,
            "trial-collision",
            1,
            IncentiveCondition.MINIMAL,
        )


def test_duplicate_matching_template_is_rejected(compiler_spec: ScenarioSpec) -> None:
    duplicate = PromptTemplate(
        template_id="actor.minimal.duplicate",
        template_version="prompt/1",
        role_id="actor",
        condition=IncentiveCondition.MINIMAL,
        kind=PromptKind.INITIAL,
        template="Duplicate {fixed_public}.",
    )
    duplicate_spec = _copy_spec(
        compiler_spec,
        prompt_templates=(*compiler_spec.prompt_templates, duplicate),
    )

    with pytest.raises(ValueError, match="duplicate"):
        compile_scenario(
            duplicate_spec,
            "trial-duplicate",
            1,
            IncentiveCondition.MINIMAL,
        )


def test_missing_template_fails_only_when_requested(compiler_spec: ScenarioSpec) -> None:
    no_actor_initial = _copy_spec(
        compiler_spec,
        prompt_templates=tuple(
            prompt
            for prompt in compiler_spec.prompt_templates
            if prompt.template_id != "actor.minimal.initial"
        ),
    )
    instance = compile_scenario(
        no_actor_initial,
        "trial-no-template",
        1,
        IncentiveCondition.MINIMAL,
    )

    with pytest.raises(ValueError, match="no prompt template"):
        render_prompt(
            no_actor_initial,
            instance,
            "actor",
            PromptKind.INITIAL,
        )


def test_unknown_role_kind_and_unmatched_kind_fail_closed(
    compiler_spec: ScenarioSpec,
    compiled_instance: ScenarioInstance,
) -> None:
    with pytest.raises(ValueError, match="role"):
        render_prompt(
            compiler_spec,
            compiled_instance,
            "ghost",
            PromptKind.INITIAL,
        )
    with pytest.raises(TypeError, match="PromptKind"):
        render_prompt(
            compiler_spec,
            compiled_instance,
            "actor",
            "initial",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="no prompt template"):
        render_prompt(
            compiler_spec,
            compiled_instance,
            "counterpart",
            PromptKind.INITIAL,
        )


def test_rendered_prompt_content_hash_freezing_and_fact_lineage(
    compiler_spec: ScenarioSpec,
    compiled_instance: ScenarioInstance,
) -> None:
    values = _fact_values(compiled_instance)
    rendered = render_prompt(
        compiler_spec,
        compiled_instance,
        "actor",
        PromptKind.INITIAL,
    )
    expected = (
        "Actor compiler_fixture/trial-7/minimal: {contract}; "
        f"pool 100/100; reserve {values['int_private']}; "
        f"rate {json.dumps(values['float_public'])}; flag true; label alpha."
    )

    assert rendered.text == expected
    assert rendered.prompt == expected
    assert rendered.template_id == "actor.minimal.initial"
    assert rendered.template_version == "prompt/1"
    template = next(
        item
        for item in compiler_spec.prompt_templates
        if item.template_id == rendered.template_id
    )
    assert rendered.template_hash == template.prompt_template_hash
    assert rendered.instance_hash == compiled_instance.instance_hash
    assert rendered.condition is IncentiveCondition.MINIMAL
    assert rendered.visible_fact_ids == (
        "bool_public",
        "fixed_public",
        "float_public",
        "int_private",
        "text_public",
    )
    assert rendered.used_fact_ids == rendered.visible_fact_ids
    assert rendered.render_hash.startswith("sha256:")
    payload = json.loads(rendered.canonical_json())
    assert payload["render_hash"] == rendered.render_hash
    assert rendered.render_hash == canonical_sha256(
        {key: value for key, value in payload.items() if key != "render_hash"}
    )
    with pytest.raises(FrozenInstanceError):
        rendered.text = "changed"  # type: ignore[misc]


def test_counterpart_and_adjudicator_rendering_use_only_their_views(
    compiler_spec: ScenarioSpec,
    compiled_instance: ScenarioInstance,
) -> None:
    counterpart = render_prompt(
        compiler_spec,
        compiled_instance,
        "counterpart",
        PromptKind.COUNTERPART,
    )
    adjudicator = render_prompt(
        compiler_spec,
        compiled_instance,
        "adjudicator",
        PromptKind.VERIFICATION,
    )

    assert counterpart.used_fact_ids == (
        "bool_public",
        "choice_private",
        "fixed_public",
        "float_public",
    )
    assert "int_private" not in counterpart.visible_fact_ids
    assert "judge_only" not in counterpart.visible_fact_ids
    assert adjudicator.used_fact_ids == (
        "choice_private",
        "int_private",
        "judge_only",
    )
    assert adjudicator.visible_fact_ids == tuple(
        sorted(fact.fact_id for fact in compiled_instance.resolved_facts)
    )


def test_rendered_prompt_rejects_invalid_lineage() -> None:
    with pytest.raises(ValueError, match="ordered"):
        RenderedPrompt(
            text="text",
            template_id="template",
            template_version="v1",
            template_hash="sha256:" + "0" * 64,
            instance_hash="sha256:" + "1" * 64,
            role_id="actor",
            kind=PromptKind.INITIAL,
            condition=IncentiveCondition.MINIMAL,
            visible_fact_ids=("z", "a"),
            used_fact_ids=("a",),
        )
    with pytest.raises(ValueError, match="authorized"):
        RenderedPrompt(
            text="text",
            template_id="template",
            template_version="v1",
            template_hash="sha256:" + "0" * 64,
            instance_hash="sha256:" + "1" * 64,
            role_id="actor",
            kind=PromptKind.INITIAL,
            condition=IncentiveCondition.MINIMAL,
            visible_fact_ids=("a",),
            used_fact_ids=("b",),
        )


def test_property_style_seed_matrix_is_unique_and_deterministic(
    compiler_spec: ScenarioSpec,
) -> None:
    identities = tuple(
        product(
            ("trial-a", "trial-b", "trial-c", "trial-d", "trial-e"),
            (0, 1, 7, 19, 101),
            (IncentiveCondition.MINIMAL, IncentiveCondition.HIGH_INCENTIVE),
        )
    )
    derived = [
        derive_compilation_seed(compiler_spec, trial_id, seed, condition)
        for trial_id, seed, condition in identities
    ]

    assert len(derived) == 50
    assert len(set(derived)) == len(derived)
    for trial_id, seed, condition in identities:
        first = compile_scenario(compiler_spec, trial_id, seed, condition)
        second = compile_scenario(compiler_spec, trial_id, seed, condition)
        assert first.canonical_json() == second.canonical_json()


def test_one_hundred_compilations_preserve_bounds_and_view_consistency(
    compiler_spec: ScenarioSpec,
) -> None:
    choice_values: set[str] = set()
    instance_hashes: set[str] = set()
    public_ids = {
        fact.fact_id
        for fact in compiler_spec.facts
        if fact.visibility is Visibility.PUBLIC
    }

    for index in range(100):
        condition = (
            IncentiveCondition.MINIMAL
            if index % 2 == 0
            else IncentiveCondition.HIGH_INCENTIVE
        )
        instance = compile_scenario(
            compiler_spec,
            f"trial-matrix-{index}",
            index,
            condition,
        )
        values = _fact_values(instance)
        choice_values.add(values["choice_private"])
        instance_hashes.add(instance.instance_hash)

        assert 20 <= values["int_private"] <= 50
        assert type(values["int_private"]) is int
        assert 0.25 <= values["float_public"] <= 0.75
        assert type(values["float_public"]) is float
        assert values["choice_private"] in {"low", "mid", "high"}
        assert {
            fact.fact_id for fact in instance.public_state.facts
        } == public_ids
        assert {
            fact.fact_id for fact in instance.adjudicator_view.facts
        } == {fact.fact_id for fact in instance.resolved_facts}
        assert ScenarioInstance.from_persisted_json(instance.canonical_json()) == instance

    assert choice_values == {"low", "mid", "high"}
    assert len(instance_hashes) == 100
