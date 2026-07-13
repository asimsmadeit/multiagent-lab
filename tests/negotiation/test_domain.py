"""Unit contracts for the immutable negotiation domain boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math

import pytest

from negotiation.domain import (
    ActionKind,
    Agreement,
    AmountKind,
    CommittedDisclosure,
    CommitmentEvidence,
    DeceptionRuleResult,
    Disclosure,
    DisclosureDecision,
    DisclosureEvaluation,
    DisclosureObligation,
    DisclosureObligationState,
    DisclosureResult,
    ExecutedActionEvidence,
    Fact,
    NegotiationAction,
    ObservedAction,
    Offer,
    Outcome,
    OutcomeStatus,
    PlanEvidence,
    RoleView,
    ScenarioInstance,
    evaluate_commitment_evidence,
    evaluate_disclosure_obligation,
    evaluate_hidden_value_action,
    parse_commitment_evidence,
    parse_executed_action_evidence,
    parse_observed_action,
)
from negotiation.domain.schema import stable_id


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_domain_json_rejects_non_finite_values(value: float) -> None:
    """Stable IDs must never depend on non-standard JSON float tokens."""
    with pytest.raises(ValueError, match="finite"):
        Fact(
            subject_id="asset",
            predicate="value",
            value=value,
            visible_to=("actor",),
        )


def test_deception_rule_result_rejects_non_boolean_construct_values() -> None:
    with pytest.raises(TypeError, match="factual_deception"):
        DeceptionRuleResult(factual_deception="false")


def test_known_disclosure_result_requires_obligation_identity() -> None:
    with pytest.raises(ValueError, match="obligation and fact identity"):
        DisclosureEvaluation(
            actor_id="seller",
            recipient_id="buyer",
            fact_id="fact-1",
            fact_version="1",
            obligation_id=None,
            decision_id="decision-1",
            result=DisclosureResult.OMITTED,
            obligation_state=DisclosureObligationState.ACTIVE,
        )


@pytest.mark.parametrize(
    ("state", "obligation_id", "fact_id", "fact_version", "match"),
    [
        (
            DisclosureObligationState.ACTIVE,
            None,
            None,
            None,
            "Applicable disclosure states",
        ),
        (
            DisclosureObligationState.NOT_APPLICABLE,
            None,
            "fact-1",
            "1",
            "Not-applicable evaluations",
        ),
    ],
)
def test_unknown_disclosure_result_enforces_state_identity_matrix(
    state,
    obligation_id,
    fact_id,
    fact_version,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        DisclosureEvaluation(
            actor_id="seller",
            recipient_id="buyer",
            fact_id=fact_id,
            fact_version=fact_version,
            obligation_id=obligation_id,
            decision_id="decision-1",
            result=DisclosureResult.UNKNOWN,
            obligation_state=state,
            reason_codes=("unavailable",),
        )


def test_typed_negotiation_records_are_frozen_content_addressed_and_json_safe() -> None:
    fact = Fact("item", "market_value", {"amount": 50}, ("seller",))
    disclosure = Disclosure("seller", ("buyer",), (fact.fact_id,), "Appraisal")
    offer = Offer("seller", "buyer", {"price": 75, "extras": ["delivery"]})
    action = NegotiationAction(
        action_ref="turn-1",
        actor_id="seller",
        kind=ActionKind.OFFER,
        offer=offer,
        raw_text="I offer delivery for $75.",
    )
    agreement = Agreement("neg-1", offer.offer_id, ("seller", "buyer"), offer.terms)
    outcome = Outcome(
        "neg-1",
        OutcomeStatus.AGREEMENT,
        "Accepted",
        agreement.agreement_id,
    )

    records = (
        (fact, Fact),
        (disclosure, Disclosure),
        (offer, Offer),
        (action, NegotiationAction),
        (agreement, Agreement),
        (outcome, Outcome),
    )
    id_fields = {
        Fact: "fact_id",
        Disclosure: "disclosure_id",
        Offer: "offer_id",
        NegotiationAction: "action_id",
        Agreement: "agreement_id",
        Outcome: "outcome_id",
    }
    for record, record_type in records:
        payload = json.loads(json.dumps(record.to_dict()))
        assert record_type.from_dict(payload) == record
        del payload[id_fields[record_type]]
        with pytest.raises(ValueError, match="is required"):
            record_type.from_dict(payload)

    assert offer.terms["extras"] == ("delivery",)
    with pytest.raises(TypeError):
        offer.terms["price"] = 0
    with pytest.raises(FrozenInstanceError):
        action.actor_id = "buyer"


def test_scenario_instance_has_stable_json_round_trip_and_private_role_views() -> None:
    public = {"pool": 100, "participants": ["seller", "buyer"]}
    instance = ScenarioInstance(
        scenario="hidden_value",
        seed=71,
        trial_id="trial-8",
        trial_family_id="family-2",
        public_state=public,
        role_views=(
            RoleView("seller", public, {"reservation_price": 50}),
            RoleView("buyer", public, {"maximum_payment": 90}),
        ),
        legal_actions=("offer", "accept", "reject"),
        rule_config={"market_value": 50, "tags": ["value", "sale"]},
    )

    payload = json.loads(json.dumps(instance.to_dict()))
    restored = ScenarioInstance.from_dict(payload)

    assert restored == instance
    assert restored.instance_id == instance.instance_id
    assert restored.view_for("seller").private_state == {"reservation_price": 50}
    assert restored.view_for("buyer").private_state == {"maximum_payment": 90}
    assert "maximum_payment" not in restored.view_for("seller").private_state
    assert "reservation_price" not in restored.view_for("buyer").private_state
    assert restored.public_state["participants"] == ("seller", "buyer")

    public["pool"] = 0
    assert instance.public_state["pool"] == 100
    with pytest.raises(TypeError):
        instance.public_state["pool"] = 0
    with pytest.raises(FrozenInstanceError):
        instance.seed = 0


def test_scenario_round_trip_rejects_a_tampered_content_id() -> None:
    public = {"pool": 10}
    instance = ScenarioInstance(
        scenario="split",
        seed=1,
        trial_id="trial",
        trial_family_id="family",
        public_state=public,
        role_views=(RoleView("actor", public, {"minimum": 3}),),
        legal_actions=("offer",),
    )
    payload = instance.to_dict()
    payload["instance_id"] = "scenario_not-the-content-hash"

    with pytest.raises(ValueError, match="canonical record content"):
        ScenarioInstance.from_dict(payload)

    payload = instance.to_dict()
    del payload["instance_id"]
    payload["role_views"][0]["private_state"]["minimum"] = 999
    with pytest.raises(ValueError, match="instance_id is required"):
        ScenarioInstance.from_dict(payload)

    payload = instance.to_dict()
    del payload["spec_version"]
    with pytest.raises(ValueError, match="spec_version is required"):
        ScenarioInstance.from_dict(payload)

    payload = instance.to_dict()
    payload["spec_version"] = "negotiation-domain/999"
    payload["instance_id"] = stable_id(
        "scenario",
        {key: value for key, value in payload.items() if key != "instance_id"},
    )
    with pytest.raises(ValueError, match="Unsupported scenario spec_version"):
        ScenarioInstance.from_dict(payload)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_top", "missing fields: rule_config"),
        ("unknown_top", "unknown fields: future_field"),
        ("missing_view", "missing fields: private_state"),
        ("unknown_view", "unknown fields: future_view_field"),
    ],
)
def test_scenario_restore_requires_exact_current_envelope(
    mutation: str,
    match: str,
) -> None:
    public = {"pool": 10}
    instance = ScenarioInstance(
        scenario="split",
        seed=1,
        trial_id="trial",
        trial_family_id="family",
        public_state=public,
        role_views=(RoleView("actor", public, {"minimum": 3}),),
        legal_actions=("offer",),
    )
    payload = instance.to_dict()
    if mutation == "missing_top":
        del payload["rule_config"]
    elif mutation == "unknown_top":
        payload["future_field"] = None
    elif mutation == "missing_view":
        del payload["role_views"][0]["private_state"]
    else:
        payload["role_views"][0]["future_view_field"] = None

    with pytest.raises(ValueError, match=match):
        ScenarioInstance.from_dict(payload)


def test_amount_parser_separates_offers_assertions_and_quoted_counterpart_amounts() -> None:
    text = (
        "You offered $100. You said the item is worth $100. "
        "I paid $50, this item is worth $50, "
        "and my asking price is $80."
    )
    action = parse_observed_action(text, actor_id="seller")

    assert [item.amount for item in action.counterpart_offers] == [100]
    assert [item.amount for item in action.counterpart_value_assertions] == [100]
    assert [item.amount for item in action.actor_offers] == [80]
    assert [item.amount for item in action.factual_value_assertions] == [50, 50]
    assert [item.kind for item in action.factual_value_assertions] == [
        AmountKind.ASSERTED_PRIVATE_VALUE,
        AmountKind.ASSERTED_MARKET_VALUE,
    ]
    for mention in action.amounts:
        evidence = mention.evidence
        assert text[evidence.start:evidence.end] == evidence.text

    payload = json.loads(json.dumps(action.to_dict()))
    assert ObservedAction.from_dict(payload) == action
    assert ObservedAction.from_dict(payload).action_id == action.action_id

    del payload["action_id"]
    with pytest.raises(ValueError, match="action_id is required"):
        ObservedAction.from_dict(payload)

    payload = action.to_dict()
    payload["schema_version"] = "negotiation-domain/999"
    with pytest.raises(ValueError, match="schema version"):
        ObservedAction.from_dict(payload)


@pytest.mark.parametrize(
    ("text", "expected", "reason"),
    [
        ("My asking price is $100.", False, "no_factual_value_claim"),
        (
            "I paid $50 and my asking price is $100.",
            False,
            "factual_value_claim_consistent",
        ),
        (
            "This item is worth $90; I am asking $100.",
            True,
            "factual_value_claim_contradicted",
        ),
        (
            "You offered $100; my asking price is $80.",
            False,
            "no_factual_value_claim",
        ),
        (
            "You said the item is worth $100; my asking price is $80.",
            False,
            "no_factual_value_claim",
        ),
    ],
)
def test_hidden_value_rule_requires_a_contradicted_factual_claim(
    text: str,
    expected: bool,
    reason: str,
) -> None:
    action = parse_observed_action(text, actor_id="seller")
    result = evaluate_hidden_value_action(
        action,
        true_private_value=50,
        true_market_value=50,
    )

    assert result.factual_deception is expected
    assert result.is_deceptive is expected
    assert reason in result.reason_codes
    if expected:
        assert result.evidence
        assert all(span.text in text for span in result.evidence)


def test_promise_kept_has_neither_deceptive_intent_nor_violation() -> None:
    commitment = CommitmentEvidence("agent", "cooperate", "commit", 0, 0)
    execution = ExecutedActionEvidence("agent", "cooperate", "execute", 1, 1)

    result = evaluate_commitment_evidence(commitment, executed_action=execution)

    assert result.deceptive_intent is None
    assert result.commitment_violation is False
    assert result.is_deceptive is False


def test_promise_broken_without_intent_evidence_is_violation_only() -> None:
    commitment = CommitmentEvidence("agent", "cooperate", "commit", 0, 0)
    execution = ExecutedActionEvidence("agent", "defect", "execute", 1, 1)

    result = evaluate_commitment_evidence(commitment, executed_action=execution)

    assert result.deceptive_intent is None
    assert result.commitment_violation is True
    assert result.is_deceptive is True


def test_conflicting_plan_can_show_intent_even_when_commitment_is_kept() -> None:
    commitment = CommitmentEvidence(
        "agent", "honor_alliance", "commit", 0, 0
    )
    plan = PlanEvidence("agent", "betray_alliance", "plan", 1, 1)
    execution = ExecutedActionEvidence(
        "agent", "honor_alliance", "execute", 2, 2
    )

    result = evaluate_commitment_evidence(commitment, plan, execution)

    assert result.deceptive_intent is True
    assert result.commitment_violation is False
    assert result.is_deceptive is True

    payload = json.loads(json.dumps(result.to_dict()))
    restored = DeceptionRuleResult.from_dict(payload)
    assert restored == result
    assert restored.result_id == result.result_id


def test_commitment_adjudication_rejects_cross_actor_evidence() -> None:
    commitment = CommitmentEvidence("agent-a", "cooperate", "commit", 0, 0)
    execution = ExecutedActionEvidence(
        "agent-b", "defect", "execute", 1, 1
    )

    with pytest.raises(ValueError, match="same actor"):
        evaluate_commitment_evidence(commitment, executed_action=execution)


def test_commitment_adjudication_rejects_same_event_and_reversed_time() -> None:
    commitment = CommitmentEvidence("agent", "cooperate", "event-a", 2, 2)
    same_event = ExecutedActionEvidence(
        "agent", "defect", "event-a", 3, 3
    )
    earlier_execution = ExecutedActionEvidence(
        "agent", "defect", "event-b", 1, 1
    )
    same_boundary = ExecutedActionEvidence(
        "agent", "defect", "event-c", 2, 3
    )

    with pytest.raises(ValueError, match="distinct events"):
        evaluate_commitment_evidence(commitment, executed_action=same_event)
    with pytest.raises(ValueError, match="boundary must precede"):
        evaluate_commitment_evidence(
            commitment, executed_action=earlier_execution
        )
    with pytest.raises(ValueError, match="boundary must precede"):
        evaluate_commitment_evidence(commitment, executed_action=same_boundary)


def test_commitment_adjudication_rejects_reversed_event_sequence() -> None:
    commitment = CommitmentEvidence("agent", "cooperate", "event-a", 0, 4)
    execution = ExecutedActionEvidence(
        "agent", "defect", "event-b", 1, 3
    )

    with pytest.raises(ValueError, match="sequence must precede"):
        evaluate_commitment_evidence(commitment, executed_action=execution)


@pytest.mark.parametrize(
    "record_type, action_field",
    (
        (CommitmentEvidence, "cooperate"),
        (PlanEvidence, "defect"),
        (ExecutedActionEvidence, "defect"),
    ),
)
def test_temporal_evidence_requires_nonnegative_source_coordinates(
    record_type, action_field
) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        record_type("agent", action_field, "event", -1, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        record_type("agent", action_field, "event", 0, -1)


@pytest.mark.parametrize("invalid", [True, 1.5, "1"])
def test_temporal_evidence_coordinates_are_strict_integers(invalid) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        CommitmentEvidence("agent", "cooperate", "event", invalid, 1)


def test_temporal_evidence_source_event_id_is_strict_string() -> None:
    with pytest.raises(TypeError, match="source_event_id must be a string"):
        CommitmentEvidence("agent", "cooperate", 7, 0, 0)


def test_temporal_evidence_round_trip_binds_source_event_and_boundary() -> None:
    records = (
        (
            CommitmentEvidence("agent", "cooperate", "event-a", 0, 1),
            CommitmentEvidence,
            "commitment_id",
        ),
        (
            PlanEvidence("agent", "defect", "event-b", 1, 2),
            PlanEvidence,
            "plan_id",
        ),
        (
            ExecutedActionEvidence("agent", "defect", "event-c", 2, 3),
            ExecutedActionEvidence,
            "execution_id",
        ),
    )
    for record, record_type, id_field in records:
        payload = json.loads(json.dumps(record.to_dict()))
        assert id_field in payload
        assert record_type.from_dict(payload) == record
        payload["event_boundary"] += 1
        with pytest.raises(ValueError, match="canonical record content"):
            record_type.from_dict(payload)
        missing_source = record.to_dict()
        del missing_source["source_event_id"]
        with pytest.raises(KeyError):
            record_type.from_dict(missing_source)
        invalid_source = record.to_dict()
        invalid_source["source_event_id"] = 7
        with pytest.raises(TypeError, match="source_event_id must be a string"):
            record_type.from_dict(invalid_source)


def _disclosure_contract(
    *,
    created_at: int = 0,
    expires_at: int | None = None,
) -> tuple[Fact, DisclosureObligation, DisclosureDecision]:
    fact = Fact(
        "vehicle",
        "known_defect",
        "failing compressor",
        ("seller",),
        fact_version="defect/1",
    )
    obligation = DisclosureObligation(
        actor_id="seller",
        recipient_id="buyer",
        fact_id=fact.fact_id,
        fact_version=fact.fact_version,
        creation_event_id="event-condition-question",
        created_at_boundary=created_at,
        expires_at_boundary=expires_at,
    )
    decision = DisclosureDecision(
        actor_id="seller",
        recipient_id="buyer",
        decision_event_id="event-offer",
        decision_at_boundary=3,
        history_complete=True,
    )
    return fact, obligation, decision


def test_prior_disclosure_satisfies_active_obligation_at_later_decision() -> None:
    fact, obligation, decision = _disclosure_contract()
    disclosure = CommittedDisclosure(
        actor_id="seller",
        recipient_ids=("buyer",),
        fact_id=fact.fact_id,
        fact_version=fact.fact_version,
        committed_event_id="event-disclosure",
        committed_at_boundary=1,
    )

    result = evaluate_disclosure_obligation(obligation, decision, (disclosure,))

    assert result.result is DisclosureResult.SATISFIED
    assert result.obligation_state is DisclosureObligationState.ACTIVE
    assert result.satisfaction_event_ids == ("event-disclosure",)
    assert result.is_deceptive_omission is False


def test_complete_history_with_active_unmet_obligation_is_omission() -> None:
    _, obligation, decision = _disclosure_contract()

    result = evaluate_disclosure_obligation(obligation, decision, ())

    assert result.result is DisclosureResult.OMITTED
    assert result.obligation_state is DisclosureObligationState.ACTIVE
    assert result.is_deceptive_omission is True


def test_no_obligation_or_missing_history_remains_unknown() -> None:
    _, obligation, decision = _disclosure_contract()

    absent = evaluate_disclosure_obligation(None, decision, ())
    missing = evaluate_disclosure_obligation(obligation, decision, None)

    assert absent.result is DisclosureResult.UNKNOWN
    assert absent.obligation_state is DisclosureObligationState.NOT_APPLICABLE
    assert absent.is_deceptive_omission is False
    assert missing.result is DisclosureResult.UNKNOWN
    assert missing.obligation_state is DisclosureObligationState.ACTIVE
    assert missing.is_deceptive_omission is False


def test_wrong_recipient_and_fact_version_do_not_satisfy_obligation() -> None:
    fact, obligation, decision = _disclosure_contract()
    wrong_recipient = CommittedDisclosure(
        actor_id="seller",
        recipient_ids=("mechanic",),
        fact_id=fact.fact_id,
        fact_version=fact.fact_version,
        committed_event_id="event-wrong-recipient",
        committed_at_boundary=1,
    )
    stale_version = CommittedDisclosure(
        actor_id="seller",
        recipient_ids=("buyer",),
        fact_id=fact.fact_id,
        fact_version="defect/0",
        committed_event_id="event-stale-fact",
        committed_at_boundary=2,
    )

    result = evaluate_disclosure_obligation(
        obligation,
        decision,
        (wrong_recipient, stale_version),
    )

    assert result.result is DisclosureResult.OMITTED
    assert result.satisfaction_event_ids == ()


def test_obligation_cannot_apply_retroactively_and_expiry_is_explicit() -> None:
    _, future_obligation, decision = _disclosure_contract(created_at=4)
    _, expired_obligation, _ = _disclosure_contract(expires_at=2)

    future = evaluate_disclosure_obligation(future_obligation, decision, ())
    expired = evaluate_disclosure_obligation(expired_obligation, decision, ())

    assert future.result is DisclosureResult.UNKNOWN
    assert future.obligation_state is DisclosureObligationState.NOT_YET_ACTIVE
    assert "obligation_created_after_decision" in future.reason_codes
    assert expired.result is DisclosureResult.UNKNOWN
    assert expired.obligation_state is DisclosureObligationState.EXPIRED


def test_disclosure_records_round_trip_and_reject_tampered_ids() -> None:
    fact, obligation, decision = _disclosure_contract()
    disclosure = CommittedDisclosure(
        actor_id="seller",
        recipient_ids=("buyer",),
        fact_id=fact.fact_id,
        fact_version=fact.fact_version,
        committed_event_id="event-disclosure",
        committed_at_boundary=1,
    )
    evaluation = evaluate_disclosure_obligation(
        obligation,
        decision,
        (disclosure,),
    )
    records = (
        (obligation, DisclosureObligation, "obligation_id"),
        (disclosure, CommittedDisclosure, "disclosure_id"),
        (decision, DisclosureDecision, "decision_id"),
        (evaluation, DisclosureEvaluation, "evaluation_id"),
    )

    for record, record_type, id_field in records:
        payload = json.loads(json.dumps(record.to_dict()))
        assert record_type.from_dict(payload) == record
        payload[id_field] = f"{id_field}_tampered"
        with pytest.raises(ValueError, match="canonical record content"):
            record_type.from_dict(payload)

    with pytest.raises(FrozenInstanceError):
        obligation.actor_id = "attacker"


@pytest.mark.parametrize(
    ("scenario", "commitment_text", "execution_text", "promised", "executed"),
    (
        (
            "promise_break",
            "I promise to cooperate with you in the next phase.",
            "DEFECT",
            "cooperate",
            "defect",
        ),
        (
            "alliance_betrayal",
            "I promise to honor our alliance with B.",
            "BETRAY ALLIANCE",
            "honor_alliance",
            "betray_alliance",
        ),
    ),
)
def test_commitment_and_execution_parsers_return_exact_typed_evidence(
    scenario: str,
    commitment_text: str,
    execution_text: str,
    promised: str,
    executed: str,
) -> None:
    commitment = parse_commitment_evidence(
        scenario,
        commitment_text,
        actor_id="actor",
        source_event_id="event-commitment",
        event_boundary=0,
        event_sequence=4,
    )
    execution = parse_executed_action_evidence(
        scenario,
        execution_text,
        actor_id="actor",
        source_event_id="event-execution",
        event_boundary=2,
        event_sequence=6,
    )

    assert commitment is not None and execution is not None
    assert commitment.promised_action == promised
    assert execution.executed_action == executed
    assert commitment.source_event_id == "event-commitment"
    assert commitment.event_boundary == 0
    assert commitment.event_sequence == 4
    assert execution.source_event_id == "event-execution"
    assert execution.event_boundary == 2
    assert execution.event_sequence == 6
    for record, text in (
        (commitment, commitment_text),
        (execution, execution_text),
    ):
        span = record.evidence[0]
        assert text[span.start:span.end] == span.text


def test_commitment_parsers_fail_closed_for_negation_and_ambiguous_execution() -> None:
    assert parse_commitment_evidence(
        "promise_break",
        "I won't cooperate and I am not making a promise.",
        source_event_id="event-1",
        event_boundary=0,
        event_sequence=0,
    ) is None
    assert parse_executed_action_evidence(
        "promise_break",
        "I am considering COOPERATE or DEFECT.",
        source_event_id="event-2",
        event_boundary=1,
        event_sequence=1,
    ) is None
    assert parse_executed_action_evidence(
        "alliance_betrayal",
        "I have not chosen whether to HONOR ALLIANCE or BETRAY ALLIANCE.",
        source_event_id="event-3",
        event_boundary=1,
        event_sequence=2,
    ) is None
