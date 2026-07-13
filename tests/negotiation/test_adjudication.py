"""Transaction and replay tests for the public negotiation adjudicator."""

from __future__ import annotations

import copy
import datetime
import json
from typing import Any

import pytest

from negotiation.domain import ActionKind, NegotiationAction, Offer, OutcomeStatus
from negotiation.game_master.adjudication import (
    ActionResolution,
    DispatchReceipt,
    EventStatus,
    InteractionEvent,
    ModuleDispatchError,
    NegotiationAdjudicator,
    RejectionPolicy,
    ValidationDecision,
    ValidationMode,
    ValidatorBinding,
)
from negotiation.game_master.components import gm_modules, gm_state, gm_validation


class RecordingModule(gm_modules.NegotiationGMModule):
    """Stateful module whose injected failure is outside its scientific state."""

    def __init__(
        self,
        name: str,
        priority: int = 50,
        failures_remaining: int = 0,
    ) -> None:
        super().__init__(name=name, priority=priority)
        self.events: list[tuple[str, str]] = []
        self.failures_remaining = failures_remaining
        self.update_attempts = 0
        self.validation_context_ids: list[int] = []
        self.update_context_ids: list[int] = []

    def get_supported_agent_modules(self) -> set[str]:
        return set()

    def validate_action(
        self,
        actor: str,
        action: str,
        context: gm_modules.ModuleContext,
    ) -> tuple[bool, str | None]:
        self.validation_context_ids.append(id(context))
        # Deliberately impure: the adjudicator must restore validation state.
        self.events.append((actor, f"VALIDATE:{action}"))
        return True, None

    def update_state(
        self,
        event: str,
        actor: str,
        context: gm_modules.ModuleContext,
    ) -> None:
        self.update_attempts += 1
        self.update_context_ids.append(id(context))
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("injected dispatch failure")
        self.events.append((actor, event))

    def get_observation_context(
        self,
        observer: str,
        context: gm_modules.ModuleContext,
    ) -> str:
        del observer, context
        return ""

    def get_module_report(self) -> str:
        return "recording module"

    def get_state(self) -> dict[str, Any]:
        return {"events": [list(item) for item in self.events]}

    def set_state(self, state: dict[str, Any]) -> None:
        self.events = [tuple(item) for item in state.get("events", ())]


def _tracker() -> gm_state.NegotiationStateTracker:
    timestamp = datetime.datetime(2035, 4, 3, 2, 1)
    return gm_state.NegotiationStateTracker(
        max_rounds=10,
        enable_deadlines=False,
        clock=lambda: timestamp,
    )


def _adjudicator(
    *,
    modules: dict[str, RecordingModule] | None = None,
    validators: tuple[ValidatorBinding, ...] = (),
    validate_prices: bool = False,
    after_event_append: Any = None,
    before_resolution_store: Any = None,
) -> tuple[NegotiationAdjudicator, gm_state.NegotiationStateTracker]:
    tracker = _tracker()
    service = NegotiationAdjudicator(
        negotiation_id="neg-1",
        participants=("Alice", "Bob"),
        state_tracker=tracker,
        validator=(
            gm_validation.NegotiationValidator(domain_type="price")
            if validate_prices
            else None
        ),
        modules=modules,
        validators=validators,
        after_event_append=after_event_append,
        before_resolution_store=before_resolution_store,
    )
    return service, tracker


def test_validation_decision_restore_rejects_truthy_non_boolean() -> None:
    decision = ValidationDecision(
        validator_id="validator",
        priority=1,
        mode=ValidationMode.HARD,
        allowed=False,
    )
    payload = decision.to_dict()
    payload["allowed"] = "false"

    with pytest.raises(TypeError, match="allowed must be a boolean"):
        ValidationDecision.from_dict(payload)


def _offer_action(action_ref: str = "action-1", price: int = 75) -> NegotiationAction:
    return NegotiationAction(
        action_ref=action_ref,
        actor_id="Alice",
        kind=ActionKind.OFFER,
        offer=Offer("Alice", "Bob", {"price": price}),
        raw_text=f"I offer ${price}.",
    )


def test_hard_validation_blocks_but_advisory_failure_is_retained_in_order() -> None:
    module = RecordingModule("audit")
    bindings = (
        ValidatorBinding(
            "advisory_market_check",
            20,
            ValidationMode.ADVISORY,
            lambda action, context: (False, "far from market"),
        ),
        ValidatorBinding(
            "hard_authorization",
            10,
            ValidationMode.HARD,
            lambda action, context: (False, "not authorized"),
        ),
    )
    service, tracker = _adjudicator(
        modules={"audit": module},
        validators=bindings,
    )
    before = copy.deepcopy(tracker.get_state())

    resolution = service.submit(_offer_action())

    assert resolution.accepted is False
    assert resolution.event.status is EventStatus.REJECTED
    assert resolution.event.rejection_policy is RejectionPolicy.RETRY_SAME_ACTOR
    assert resolution.event.rejection_policy_version == "rejection-policy/1"
    custom = [
        decision
        for decision in resolution.decisions
        if decision.validator_id in {"hard_authorization", "advisory_market_check"}
    ]
    assert [decision.validator_id for decision in custom] == [
        "hard_authorization",
        "advisory_market_check",
    ]
    assert custom[0].mode is ValidationMode.HARD
    assert custom[1].mode is ValidationMode.ADVISORY
    assert tracker.get_state() == before
    assert module.events == []
    assert service.get_receipts() == ()
    assert service.next_actor == "Alice"

    event_payload = json.loads(json.dumps(resolution.event.to_dict()))
    resolution_payload = json.loads(json.dumps(resolution.to_dict()))
    assert InteractionEvent.from_dict(event_payload) == resolution.event
    assert ActionResolution.from_dict(resolution_payload) == resolution
    tampered = copy.deepcopy(event_payload)
    tampered["committed_turn_index"] = 4
    with pytest.raises(ValueError, match="canonical record content"):
        InteractionEvent.from_dict(tampered)
    missing_id = copy.deepcopy(event_payload)
    del missing_id["event_id"]
    with pytest.raises(ValueError, match="event_id is required"):
        InteractionEvent.from_dict(missing_id)
    missing_version = copy.deepcopy(resolution_payload)
    del missing_version["resolution_version"]
    with pytest.raises(ValueError, match="resolution version"):
        ActionResolution.from_dict(missing_version)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_event", "missing fields: module_context"),
        ("unknown_event", "unknown fields: future_field"),
        ("unknown_action", "unknown fields: future_action_field"),
        ("unknown_offer", "unknown fields: future_offer_field"),
        ("unknown_decision", "unknown fields: future_decision_field"),
    ],
)
def test_interaction_event_restore_requires_exact_current_envelope(
    mutation: str,
    match: str,
) -> None:
    decision = ValidationDecision(
        validator_id="validator",
        priority=1,
        mode=ValidationMode.ADVISORY,
        allowed=True,
    )
    event = InteractionEvent(
        negotiation_id="negotiation",
        action=_offer_action(),
        status=EventStatus.COMMITTED,
        round_index=0,
        action_sequence=0,
        committed_turn_index=0,
        decisions=(decision,),
        module_context={"scenario": "split"},
    )
    payload = event.to_dict()
    if mutation == "missing_event":
        del payload["module_context"]
    elif mutation == "unknown_event":
        payload["future_field"] = None
    elif mutation == "unknown_action":
        payload["action"]["future_action_field"] = None
    elif mutation == "unknown_offer":
        payload["action"]["offer"]["future_offer_field"] = None
    else:
        payload["decisions"][0]["future_decision_field"] = None

    with pytest.raises(ValueError, match=match):
        InteractionEvent.from_dict(payload)


def test_rejection_sequences_every_proposal_but_retries_the_same_committed_turn() -> None:
    service, tracker = _adjudicator(validate_prices=True)

    rejected = service.submit(_offer_action(action_ref="invalid", price=0))
    accepted = service.submit(_offer_action(action_ref="corrected", price=75))

    assert [event.action_sequence for event in service.get_event_log()] == [0, 1]
    assert [event.committed_turn_index for event in service.get_event_log()] == [0, 0]
    assert [event.round_index for event in service.get_event_log()] == [0, 0]
    assert rejected.accepted is False
    assert accepted.accepted is True
    assert service.next_actor == "Bob"
    assert len(tracker.get_negotiation("neg-1").offers_history) == 1
    state = service.get_state()
    assert state["action_sequence"] == 2
    assert state["committed_turn_index"] == 1
    assert state["rejection_policy"] == "retry_same_actor"
    assert state["rejection_policy_version"] == "rejection-policy/1"


def test_builtin_offer_validator_is_hard_and_rejection_isolation_is_complete() -> None:
    module = RecordingModule("audit")
    service, tracker = _adjudicator(
        modules={"audit": module},
        validate_prices=True,
    )

    result = service.submit(_offer_action(price=0))

    assert result.accepted is False
    decision = next(
        item for item in result.decisions if item.validator_id == "negotiation_validator"
    )
    assert decision.allowed is False
    assert decision.mode is ValidationMode.HARD
    assert "positive_price" in str(decision.message)
    assert tracker.get_negotiation("neg-1").offers_history == []
    assert module.events == []


def test_advisory_failure_is_evidence_but_does_not_block_commit() -> None:
    advisory = ValidatorBinding(
        "market_warning",
        5,
        ValidationMode.ADVISORY,
        lambda action, context: (False, "unusual terms"),
    )
    service, tracker = _adjudicator(validators=(advisory,))

    result = service.submit(_offer_action())

    assert result.accepted is True
    warning = next(
        item for item in result.decisions if item.validator_id == "market_warning"
    )
    assert warning.allowed is False
    assert warning.mode is ValidationMode.ADVISORY
    assert len(tracker.get_negotiation("neg-1").offers_history) == 1


def test_offer_and_rejection_transition_once_with_alternating_round_semantics() -> None:
    module = RecordingModule("audit", priority=5)
    service, tracker = _adjudicator(modules={"audit": module})
    offer_action = _offer_action()

    offered = service.submit(offer_action)
    duplicate = service.submit(offer_action)

    assert duplicate is offered
    assert offered.event.round_index == 0
    assert offered.event.action_sequence == 0
    assert offered.event.committed_turn_index == 0
    assert len(tracker.get_negotiation("neg-1").offers_history) == 1
    assert service.next_actor == "Bob"
    assert len(module.events) == 1
    assert len(service.get_receipts()) == 1
    assert module.validation_context_ids[0] == module.update_context_ids[0]
    service.dispatch_event(offered.event.event_id)
    service.dispatch_event(offered.event.event_id)
    assert len(module.events) == 1
    assert len(service.get_receipts()) == 1

    rejected = service.submit(
        NegotiationAction(
            action_ref="action-2",
            actor_id="Bob",
            kind=ActionKind.REJECT,
            referenced_offer_id=offer_action.offer.offer_id,
            reason="Price is too high",
        )
    )
    state = tracker.get_negotiation("neg-1")
    assert rejected.accepted is True
    assert rejected.event.round_index == 0
    assert rejected.event.action_sequence == 1
    assert rejected.event.committed_turn_index == 1
    assert state.offers_history[0].is_rejected is True
    assert state.current_round == 1
    assert service.next_actor == "Alice"
    assert len(module.events) == 2
    assert len(service.get_event_log()) == 2
    assert len(service.get_receipts()) == 2


def test_accept_references_stable_offer_and_creates_agreement_and_outcome() -> None:
    service, tracker = _adjudicator()
    offer_action = _offer_action()
    service.submit(offer_action)
    accepted = service.submit(
        NegotiationAction(
            action_ref="action-2",
            actor_id="Bob",
            kind=ActionKind.ACCEPT,
            referenced_offer_id=offer_action.offer.offer_id,
        )
    )

    assert accepted.agreement is not None
    assert accepted.agreement.offer_id == offer_action.offer.offer_id
    assert accepted.outcome is not None
    assert accepted.outcome.status is OutcomeStatus.AGREEMENT
    assert accepted.outcome.agreement_id == accepted.agreement.agreement_id
    assert tracker.get_negotiation("neg-1").phase == "completed"
    assert service.next_actor is None


def test_walk_away_commits_terminal_failed_outcome_without_an_offer() -> None:
    service, tracker = _adjudicator()

    result = service.submit(
        NegotiationAction(
            action_ref="action-1",
            actor_id="Alice",
            kind=ActionKind.WALK_AWAY,
            reason="BATNA is preferable",
        )
    )

    assert result.accepted is True
    assert result.outcome is not None
    assert result.outcome.status is OutcomeStatus.WALK_AWAY
    assert result.outcome.reason == "BATNA is preferable"
    assert tracker.get_negotiation("neg-1").phase == "failed"
    assert tracker.get_negotiation("neg-1").offers_history == []
    assert service.next_actor is None


def test_committed_dispatch_retry_skips_receipts_and_rolls_back_failed_module() -> None:
    first = RecordingModule("first", priority=10)
    flaky = RecordingModule("flaky", priority=20, failures_remaining=1)
    service, tracker = _adjudicator(modules={"first": first, "flaky": flaky})
    action = _offer_action()

    with pytest.raises(ModuleDispatchError, match="flaky"):
        service.submit(action)

    assert len(service.get_event_log()) == 1
    assert len(tracker.get_negotiation("neg-1").offers_history) == 1
    assert len(first.events) == 1
    assert flaky.events == []
    assert [receipt.module_id for receipt in service.get_receipts()] == ["first"]

    resolution = service.submit(action)
    assert resolution.accepted is True
    assert len(first.events) == 1
    assert first.update_attempts == 1
    assert len(flaky.events) == 1
    assert flaky.update_attempts == 2
    assert [receipt.module_id for receipt in service.get_receipts()] == [
        "first",
        "flaky",
    ]


def test_missing_receipt_retry_survives_serialization_and_process_restart() -> None:
    first = RecordingModule("first", priority=10)
    flaky = RecordingModule("flaky", priority=20, failures_remaining=1)
    service, _ = _adjudicator(modules={"first": first, "flaky": flaky})
    action = _offer_action()
    with pytest.raises(ModuleDispatchError):
        service.submit(action)
    snapshot = json.loads(json.dumps(service.get_state()))

    restored_first = RecordingModule("first", priority=10)
    restored_flaky = RecordingModule("flaky", priority=20)
    restored, tracker = _adjudicator(
        modules={"first": restored_first, "flaky": restored_flaky}
    )
    restored.set_state(snapshot)
    restored.submit(action)

    assert len(restored.get_event_log()) == 1
    assert len(tracker.get_negotiation("neg-1").offers_history) == 1
    assert len(restored_first.events) == 1
    assert restored_first.update_attempts == 0
    assert len(restored_flaky.events) == 1
    assert restored_flaky.update_attempts == 1
    assert [receipt.module_id for receipt in restored.get_receipts()] == [
        "first",
        "flaky",
    ]


def test_pending_committed_event_resumes_after_append_crash_without_duplication() -> None:
    def fail_after_append(event: InteractionEvent) -> None:
        raise RuntimeError(f"crash after {event.event_id}")

    service, tracker = _adjudicator(after_event_append=fail_after_append)
    action = _offer_action()
    with pytest.raises(RuntimeError, match="crash after event_"):
        service.submit(action)

    assert len(service.get_event_log()) == 1
    pending_event = service.get_event_log()[0]
    assert pending_event.committed is True
    assert pending_event.action_sequence == 0
    assert pending_event.committed_turn_index == 0
    assert tracker.get_negotiation("neg-1").offers_history == []
    assert service.next_actor == "Alice"
    snapshot = json.loads(json.dumps(service.get_state()))
    assert snapshot["pending_action_ids"] == [action.action_id]
    assert snapshot["action_sequence"] == 1
    assert snapshot["committed_turn_index"] == 0

    restored, restored_tracker = _adjudicator()
    restored.set_state(snapshot)
    result = restored.submit(action)

    assert result.event.event_id == pending_event.event_id
    assert len(restored.get_event_log()) == 1
    assert len(restored_tracker.get_negotiation("neg-1").offers_history) == 1
    assert restored.next_actor == "Bob"
    restored_state = restored.get_state()
    assert restored_state["pending_action_ids"] == []
    assert restored_state["action_sequence"] == 1
    assert restored_state["committed_turn_index"] == 1


def test_applied_pending_event_reconstructs_resolution_without_reapplying_tracker() -> None:
    def fail_before_resolution(event: InteractionEvent) -> None:
        raise RuntimeError(f"crash before resolution for {event.event_id}")

    service, tracker = _adjudicator(before_resolution_store=fail_before_resolution)
    action = _offer_action()
    with pytest.raises(RuntimeError, match="crash before resolution"):
        service.submit(action)

    assert len(service.get_event_log()) == 1
    assert len(tracker.get_negotiation("neg-1").offers_history) == 1
    assert service.get_resolution(action.action_id) is None
    snapshot = json.loads(json.dumps(service.get_state()))
    assert snapshot["pending_action_ids"] == [action.action_id]
    assert snapshot["committed_turn_index"] == 1

    restored, restored_tracker = _adjudicator()
    restored.set_state(snapshot)
    result = restored.submit(action)

    assert result.accepted is True
    assert result.event.event_id == service.get_event_log()[0].event_id
    assert len(restored.get_event_log()) == 1
    assert len(restored_tracker.get_negotiation("neg-1").offers_history) == 1
    assert restored.get_state()["pending_action_ids"] == []


def test_adjudicator_state_round_trip_replays_without_duplicate_delivery() -> None:
    module = RecordingModule("audit")
    service, tracker = _adjudicator(modules={"audit": module})
    offer_action = _offer_action()
    service.submit(offer_action)
    rejection = NegotiationAction(
        action_ref="action-2",
        actor_id="Bob",
        kind=ActionKind.REJECT,
        referenced_offer_id=offer_action.offer.offer_id,
        reason="No agreement",
    )
    service.submit(rejection)
    snapshot = json.loads(json.dumps(service.get_state()))

    restored_module = RecordingModule("audit")
    restored, restored_tracker = _adjudicator(modules={"audit": restored_module})
    restored.set_state(copy.deepcopy(snapshot))

    assert restored.get_state() == snapshot
    assert restored.get_event_log() == service.get_event_log()
    assert restored.get_receipts() == service.get_receipts()
    assert restored_tracker.get_state() == tracker.get_state()
    restored.submit(rejection)
    assert len(restored_module.events) == 2
    assert len(restored.get_event_log()) == 2
    assert len(restored.get_receipts()) == 2

    tampered = copy.deepcopy(snapshot)
    tampered["rejection_policy_version"] = "rejection-policy/tampered"
    incompatible, _ = _adjudicator(modules={"audit": RecordingModule("audit")})
    with pytest.raises(ValueError, match="rejection-policy version"):
        incompatible.set_state(tampered)

    bad_sequence = copy.deepcopy(snapshot)
    bad_sequence["action_sequence"] += 1
    incompatible_sequence, _ = _adjudicator(
        modules={"audit": RecordingModule("audit")}
    )
    before_failed_restore = copy.deepcopy(incompatible_sequence.get_state())
    with pytest.raises(ValueError, match="proposal sequence"):
        incompatible_sequence.set_state(bad_sequence)
    assert incompatible_sequence.get_state() == before_failed_restore

    bad_turn = copy.deepcopy(snapshot)
    bad_turn["committed_turn_index"] += 1
    incompatible_turn, _ = _adjudicator(
        modules={"audit": RecordingModule("audit")}
    )
    with pytest.raises(ValueError, match="committed-turn index"):
        incompatible_turn.set_state(bad_turn)


def test_restore_binds_module_versions_receipts_and_rolls_back_every_failure() -> None:
    source, _ = _adjudicator(modules={"audit": RecordingModule("audit")})
    source.submit(_offer_action())
    snapshot = json.loads(json.dumps(source.get_state()))
    module_version = snapshot["module_versions"]["audit"]
    unknown_event_receipt = DispatchReceipt(
        event_id="event-not-in-log",
        module_id="audit",
        module_version=module_version,
    ).to_dict()

    hostile_states = []
    missing_module = copy.deepcopy(snapshot)
    del missing_module["module_versions"]["audit"]
    hostile_states.append((missing_module, "module-version IDs"))
    wrong_version = copy.deepcopy(snapshot)
    wrong_version["module_versions"]["audit"] = "audit/tampered"
    hostile_states.append((wrong_version, "module version"))
    unknown_receipt = copy.deepcopy(snapshot)
    unknown_receipt["receipts"].append(unknown_event_receipt)
    hostile_states.append((unknown_receipt, "unknown event"))
    duplicate_receipt = copy.deepcopy(snapshot)
    duplicate_receipt["receipts"].append(
        copy.deepcopy(duplicate_receipt["receipts"][0])
    )
    hostile_states.append((duplicate_receipt, "receipts contain duplicates"))

    target, _ = _adjudicator(modules={"audit": RecordingModule("audit")})
    target.submit(_offer_action(action_ref="target-action", price=70))
    original = copy.deepcopy(target.get_state())
    for hostile, match in hostile_states:
        with pytest.raises(ValueError, match=match):
            target.set_state(hostile)
        assert target.get_state() == original
