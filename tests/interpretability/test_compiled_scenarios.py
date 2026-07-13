"""One-source scenario prompt, privacy, rule, and identity contracts."""

from __future__ import annotations

import json

import pytest

from negotiation.domain import (
    CommittedDisclosure,
    DisclosureDecision,
    DisclosureObligation,
    Fact,
)
from negotiation.domain.scenario import RoleView, ScenarioInstance
from negotiation.domain.schema import SCHEMA_VERSION, thaw_json
from interpretability.scenarios.compiled import (
    COUNTERPART_KNOWLEDGE_GRANT_VERSION,
    EMERGENT_SCENARIO_SPEC_VERSION,
    SUPPORTED_COUNTERPART_POLICIES,
    SUPPORTED_SURFACE_VARIANTS,
    CounterpartPolicy,
    ExecutionProtocol,
    compile_emergent_scenario,
    evaluate_actor_response,
    parameters_for_role,
    render_actor_prompt,
    render_counterpart_prompt,
)
from interpretability.scenarios.emergent_prompts import (
    IncentiveCondition,
    get_all_scenarios,
    get_counterpart_prompt,
)
from interpretability.runtime.runner import build_counterbalance_schedule


@pytest.mark.parametrize("scenario", get_all_scenarios())
def test_compiled_scenario_round_trip_renders_authorized_views(scenario: str) -> None:
    instance = compile_emergent_scenario(
        scenario,
        family_seed=17,
        trial_seed=3,
        condition=IncentiveCondition.HIGH_INCENTIVE,
        role_assignment={"actor": "Alice", "counterpart": "Bob"},
    )
    restored = ScenarioInstance.from_dict(json.loads(json.dumps(instance.to_dict())))
    actor_prompt = render_actor_prompt(restored, "Alice")
    counterpart_prompt = render_counterpart_prompt(restored, "Bob")

    assert restored == instance
    assert restored.spec_version == SCHEMA_VERSION
    assert (
        restored.public_state["scenario_spec_version"]
        == EMERGENT_SCENARIO_SPEC_VERSION
    )
    assert "{" not in actor_prompt
    assert "{" not in counterpart_prompt
    actor_private_parameters = dict(
        restored.view_for("Alice").private_state["parameters"]
    )
    counterpart_parameters = parameters_for_role(restored, "Bob")
    for key in actor_private_parameters:
        assert key not in counterpart_parameters


def test_counterbalanced_variants_share_family_but_not_trial_identity() -> None:
    first = compile_emergent_scenario(
        "hidden_value",
        family_seed=12,
        trial_seed=9,
        condition="high_incentive",
        role_assignment={"actor": "Alice", "counterpart": "Bob"},
    )
    mirrored = compile_emergent_scenario(
        "hidden_value",
        family_seed=12,
        trial_seed=9,
        condition="high_incentive",
        role_assignment={"actor": "Bob", "counterpart": "Alice"},
    )

    assert first.trial_family_id == mirrored.trial_family_id
    assert first.trial_id != mirrored.trial_id
    assert first.rule_config == mirrored.rule_config


def test_all_counterbalance_axes_share_family_and_have_distinct_identities() -> None:
    common = {
        "scenario": "hidden_value",
        "family_seed": 12,
        "trial_seed": 9,
        "condition": "high_incentive",
    }
    variants = (
        compile_emergent_scenario(
            **common,
            role_assignment={"actor": "Alice", "counterpart": "Bob"},
        ),
        compile_emergent_scenario(
            **common,
            role_assignment={"actor": "Bob", "counterpart": "Alice"},
        ),
        compile_emergent_scenario(
            **common,
            role_assignment={"actor": "Alice", "counterpart": "Bob"},
            first_mover="counterpart",
        ),
        compile_emergent_scenario(
            **common,
            role_assignment={"actor": "Alice", "counterpart": "Bob"},
            counterpart_type="skeptical",
        ),
        compile_emergent_scenario(
            **common,
            role_assignment={"actor": "Alice", "counterpart": "Bob"},
            surface_variant="formal-brief",
        ),
        compile_emergent_scenario(
            **common,
            role_assignment={"actor": "Alice", "counterpart": "Bob"},
            actor_profile="ultrafast_minimal/1",
        ),
        compile_emergent_scenario(
            **common,
            role_assignment={"actor": "Alice", "counterpart": "Bob"},
            intervention_design_id="intervention_design_test",
        ),
    )

    assert len({item.trial_family_id for item in variants}) == 1
    assert len({item.trial_id for item in variants}) == len(variants)
    assert len({item.instance_id for item in variants}) == len(variants)
    assert all(item.rule_config == variants[0].rule_config for item in variants)


def test_complete_cross_uses_one_family_and_distinct_physical_trial_seeds() -> None:
    schedule = build_counterbalance_schedule(
        participant_ids=('Alice', 'Bob'),
        counterpart_types=('default',),
        surface_variants=('default',),
        schedule_seed=9,
    )
    variants = tuple(
        compile_emergent_scenario(
            'hidden_value',
            family_seed=23,
            trial_seed=index + 100,
            condition='high_incentive',
            role_assignment=assignment.role_assignment,
            first_mover=assignment.first_mover_id,
            counterpart_type=assignment.counterpart_type,
            surface_variant=assignment.surface_metadata_variant,
        )
        for index, assignment in enumerate(schedule)
    )

    assert len(schedule) == 4
    assert len({item.trial_family_id for item in variants}) == 1
    assert len({item.trial_id for item in variants}) == len(schedule)
    assert len({item.instance_id for item in variants}) == len(schedule)
    assert {
        item.public_state['family_seed'] for item in variants
    } == {23}


def test_compiler_records_profiles_and_rejects_unversioned_profile_aliases() -> None:
    instance = compile_emergent_scenario(
        "hidden_value",
        family_seed=12,
        trial_seed=9,
        condition="minimal",
        actor_profile="ultrafast_minimal/1",
        counterpart_profile="advanced_negotiator/1",
    )

    assert thaw_json(instance.public_state)["agent_profiles"] == {
        "actor": "ultrafast_minimal/1",
        "counterpart": "advanced_negotiator/1",
    }
    with pytest.raises(ValueError, match="unsupported agent profile"):
        compile_emergent_scenario(
            "hidden_value",
            family_seed=12,
            trial_seed=9,
            condition="minimal",
            actor_profile="ultrafast",
        )


def test_intervention_design_is_explicit_and_contributes_trial_identity() -> None:
    common = {
        "scenario": "hidden_value",
        "family_seed": 12,
        "trial_seed": 9,
        "condition": "minimal",
    }
    control = compile_emergent_scenario(**common)
    treated = compile_emergent_scenario(
        **common,
        intervention_design_id="intervention_design_example",
    )

    assert control.trial_family_id == treated.trial_family_id
    assert control.trial_id != treated.trial_id
    assert control.public_state["intervention_design_id"] is None
    assert (
        treated.public_state["intervention_design_id"]
        == "intervention_design_example"
    )
    with pytest.raises(ValueError, match="intervention_design_id"):
        compile_emergent_scenario(**common, intervention_design_id="")


def test_protocol_is_explicit_and_contributes_trial_identity() -> None:
    common = {
        "scenario": "hidden_value",
        "family_seed": 12,
        "trial_seed": 9,
        "condition": "minimal",
    }
    alternating = compile_emergent_scenario(**common)
    simultaneous = compile_emergent_scenario(
        **common,
        protocol=ExecutionProtocol.SIMULTANEOUS,
    )

    assert alternating.trial_family_id == simultaneous.trial_family_id
    assert alternating.trial_id != simultaneous.trial_id
    assert alternating.public_state["protocol"] == "alternating"
    assert simultaneous.public_state["protocol"] == "simultaneous"
    with pytest.raises(ValueError, match="Unsupported execution protocol"):
        compile_emergent_scenario(**common, protocol="parallel-ish")


def test_solo_protocol_requires_absence_and_never_renders_counterpart() -> None:
    common = {
        "scenario": "hidden_value",
        "family_seed": 12,
        "trial_seed": 9,
        "condition": "minimal",
        "protocol": "solo_no_response",
    }
    instance = compile_emergent_scenario(
        **common,
        counterpart_type="absent",
    )

    true_value = instance.view_for("actor").private_state["parameters"][
        "true_value"
    ]
    assert str(true_value) in render_actor_prompt(instance, "actor")
    assert instance.public_state["counterbalance"]["counterpart_type"] == "absent"
    with pytest.raises(ValueError, match="no counterpart prompt"):
        render_counterpart_prompt(instance, "counterpart")
    with pytest.raises(ValueError, match="requires counterpart_type='absent'"):
        compile_emergent_scenario(**common)
    with pytest.raises(ValueError, match="separate solo protocol"):
        compile_emergent_scenario(
            "hidden_value",
            family_seed=12,
            trial_seed=9,
            condition="minimal",
            counterpart_type="absent",
        )
def test_surface_variants_render_distinct_forms_without_changing_authorization() -> None:
    instances = [
        compile_emergent_scenario(
            "hidden_value",
            family_seed=12,
            trial_seed=9,
            condition="minimal",
            role_assignment={"actor": "Alice", "counterpart": "Bob"},
            surface_variant=surface_variant,
        )
        for surface_variant in SUPPORTED_SURFACE_VARIANTS
    ]

    assert len({render_actor_prompt(item, "Alice") for item in instances}) == (
        len(SUPPORTED_SURFACE_VARIANTS)
    )
    assert len({render_counterpart_prompt(item, "Bob") for item in instances}) == (
        len(SUPPORTED_SURFACE_VARIANTS)
    )
    assert len({item.trial_id for item in instances}) == len(instances)
    assert all(
        item.view_for("Alice").private_state
        == instances[0].view_for("Alice").private_state
        for item in instances
    )
    assert all(
        item.view_for("Bob").private_state
        == instances[0].view_for("Bob").private_state
        for item in instances
    )
    assert all(
        item.public_state["parameters"] == instances[0].public_state["parameters"]
        for item in instances
    )


@pytest.mark.parametrize("scenario", get_all_scenarios())
def test_counterpart_policies_render_four_meaningfully_distinct_prompts(
    scenario: str,
) -> None:
    instances = {
        policy: compile_emergent_scenario(
            scenario,
            family_seed=21,
            trial_seed=4,
            condition="minimal",
            role_assignment={"actor": "Alice", "counterpart": "Bob"},
            counterpart_type=policy,
        )
        for policy in CounterpartPolicy
    }
    prompts = {
        policy: render_counterpart_prompt(instance, "Bob")
        for policy, instance in instances.items()
    }

    assert tuple(policy.value for policy in CounterpartPolicy) == (
        SUPPORTED_COUNTERPART_POLICIES
    )
    assert len(set(prompts.values())) == 4
    assert "SKEPTICAL" in prompts[CounterpartPolicy.SKEPTICAL]
    assert "CREDULOUS" in prompts[CounterpartPolicy.CREDULOUS]
    assert "INFORMED" in prompts[CounterpartPolicy.INFORMED]
    default = instances[CounterpartPolicy.DEFAULT]
    assert prompts[CounterpartPolicy.DEFAULT] == get_counterpart_prompt(
        scenario,
        parameters_for_role(default, "Bob"),
    )
    assert len({item.trial_family_id for item in instances.values()}) == 1
    assert len({item.trial_id for item in instances.values()}) == 4


@pytest.mark.parametrize(
    ("scenario", "authorized_keys"),
    (
        ("ultimatum_bluff", {"true_walkaway"}),
        ("capability_bluff", {"true_days"}),
        ("hidden_value", {"true_value"}),
        ("info_withholding", {"defect"}),
        ("promise_break", {"temptation"}),
        ("alliance_betrayal", {"betrayal"}),
    ),
)
def test_informed_policy_grants_only_allowlisted_actor_knowledge(
    scenario: str,
    authorized_keys: set[str],
) -> None:
    instance = compile_emergent_scenario(
        scenario,
        family_seed=31,
        trial_seed=6,
        condition="minimal",
        role_assignment={"actor": "Alice", "counterpart": "Bob"},
        counterpart_type=CounterpartPolicy.INFORMED,
    )
    actor_private = dict(instance.view_for("Alice").private_state["parameters"])
    counterpart_private = thaw_json(instance.view_for("Bob").private_state)
    grant = counterpart_private["policy_knowledge_grant"]

    assert grant["schema_version"] == COUNTERPART_KNOWLEDGE_GRANT_VERSION
    assert grant["policy"] == CounterpartPolicy.INFORMED.value
    assert grant["recipient_role_id"] == "Bob"
    assert set(grant["parameters"]) == authorized_keys
    assert set(grant["parameters"]).issubset(actor_private)
    assert parameters_for_role(instance, "Bob").keys().isdisjoint(
        authorized_keys
    )
    informed_prompt = render_counterpart_prompt(instance, "Bob")
    assert all(str(value) in informed_prompt for value in grant["parameters"].values())


@pytest.mark.parametrize(
    "policy",
    (
        CounterpartPolicy.DEFAULT,
        CounterpartPolicy.SKEPTICAL,
        CounterpartPolicy.CREDULOUS,
    ),
)
def test_non_informed_counterpart_views_never_gain_actor_private_facts(
    policy: CounterpartPolicy,
) -> None:
    instance = compile_emergent_scenario(
        "hidden_value",
        family_seed=31,
        trial_seed=6,
        condition="minimal",
        role_assignment={"actor": "Alice", "counterpart": "Bob"},
        counterpart_type=policy,
    )
    counterpart_private = thaw_json(instance.view_for("Bob").private_state)

    assert "policy_knowledge_grant" not in counterpart_private
    assert "true_value" not in counterpart_private["parameters"]
    assert str(instance.view_for("Alice").private_state["parameters"]["true_value"]) not in (
        render_counterpart_prompt(instance, "Bob")
    )


@pytest.mark.parametrize(
    ("counterpart_type", "error"),
    (
        ("absent", "solo protocol is required"),
        ("aggressive", "Unsupported counterpart policy"),
    ),
)
def test_compiler_rejects_non_executable_counterpart_policy(
    counterpart_type: str,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        compile_emergent_scenario(
            "hidden_value",
            family_seed=12,
            trial_seed=9,
            condition="minimal",
            counterpart_type=counterpart_type,
        )


def test_renderer_rejects_rehashed_unauthorized_informed_grant() -> None:
    instance = compile_emergent_scenario(
        "hidden_value",
        family_seed=12,
        trial_seed=9,
        condition="minimal",
        role_assignment={"actor": "Alice", "counterpart": "Bob"},
        counterpart_type="informed",
    )
    altered_views = []
    for view in instance.role_views:
        private = thaw_json(view.private_state)
        if view.role_id == "Bob":
            private["policy_knowledge_grant"]["parameters"][
                "unauthorized_actor_fact"
            ] = "secret"
        altered_views.append(
            RoleView(
                role_id=view.role_id,
                public_state=instance.public_state,
                private_state=private,
            )
        )
    altered = ScenarioInstance(
        spec_version=instance.spec_version,
        scenario=instance.scenario,
        seed=instance.seed,
        trial_id=instance.trial_id,
        trial_family_id=instance.trial_family_id,
        public_state=instance.public_state,
        role_views=tuple(altered_views),
        legal_actions=instance.legal_actions,
        rule_config=instance.rule_config,
    )

    with pytest.raises(ValueError, match="knowledge grant"):
        render_counterpart_prompt(altered, "Bob")


def test_compiler_rejects_unsupported_surface_before_rendering() -> None:
    with pytest.raises(ValueError, match="Unsupported surface variant"):
        compile_emergent_scenario(
            "hidden_value",
            family_seed=12,
            trial_seed=9,
            condition="minimal",
            surface_variant="unregistered-paraphrase",
        )


def test_compiler_rejects_unknown_condition_before_rendering() -> None:
    with pytest.raises(ValueError, match="Unknown incentive condition"):
        compile_emergent_scenario(
            "hidden_value",
            family_seed=12,
            trial_seed=9,
            condition="typoed_condition",
        )


def test_prompt_and_hidden_value_rule_consume_one_instance() -> None:
    instance = compile_emergent_scenario(
        "hidden_value",
        family_seed=17,
        trial_seed=1,
        condition="minimal",
    )
    actor_role = instance.view_for("actor")
    true_value = actor_role.private_state["parameters"]["true_value"]

    assert str(true_value) in render_actor_prompt(instance, "actor")
    assert str(true_value) not in render_counterpart_prompt(instance, "counterpart")
    assert evaluate_actor_response(instance, "My asking price is $1000.") is False
    assert evaluate_actor_response(
        instance,
        f"This item is worth ${int(true_value) + 1}.",
    ) is True
    with pytest.raises(ValueError, match="canonical fields"):
        evaluate_actor_response(
            instance,
            f"This item is worth ${int(true_value)}.",
            rule_evidence={"true_value": int(true_value) + 1},
        )


def test_compiled_withholding_rule_keeps_private_fact_and_requires_history() -> None:
    instance = compile_emergent_scenario(
        "info_withholding",
        family_seed=17,
        trial_seed=1,
        condition="minimal",
        role_assignment={"actor": "seller", "counterpart": "buyer"},
    )
    fact = Fact.from_dict(instance.rule_config["disclosure_fact"])
    obligation = DisclosureObligation.from_dict(
        instance.rule_config["disclosure_obligation"]
    )
    decision = DisclosureDecision(
        actor_id="seller",
        recipient_id="buyer",
        decision_event_id="event-offer",
        decision_at_boundary=2,
        history_complete=True,
    )
    disclosure = CommittedDisclosure(
        actor_id="seller",
        recipient_ids=("buyer",),
        fact_id=fact.fact_id,
        fact_version=fact.fact_version,
        committed_event_id="event-disclosure",
        committed_at_boundary=1,
    )

    assert fact.value not in render_counterpart_prompt(instance, "buyer")
    assert evaluate_actor_response(instance, "A terse offer.") is False
    assert evaluate_actor_response(
        instance,
        "A terse offer.",
        rule_evidence={
            "disclosure_decision": decision.to_dict(),
            "disclosure_history": [disclosure.to_dict()],
        },
    ) is False
    assert evaluate_actor_response(
        instance,
        "A terse offer.",
        rule_evidence={
            "disclosure_decision": decision.to_dict(),
            "disclosure_history": [],
        },
    ) is True
