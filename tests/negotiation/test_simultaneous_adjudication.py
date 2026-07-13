"""Atomicity, visibility, and replay contracts for simultaneous adjudication."""

from __future__ import annotations

import copy
import datetime
import json
from typing import Any

import pytest

from negotiation.domain import ActionKind, NegotiationAction, Offer, OutcomeStatus
from negotiation.game_master.adjudication import (
    EventStatus,
    InteractionEvent,
    ModuleDispatchError,
    NegotiationAdjudicator,
    RejectionPolicy,
    ValidationMode,
    ValidatorBinding,
)
from negotiation.game_master.components import gm_modules, gm_state


class RecordingModule(gm_modules.NegotiationGMModule):
    """Exactly-once test module with restorable scientific state."""

    def __init__(self, failures_remaining: int = 0) -> None:
        super().__init__(name="recording", priority=10)
        self.events: list[tuple[str, str, int]] = []
        self.failures_remaining = failures_remaining

    def get_supported_agent_modules(self) -> set[str]:
        return set()

    def validate_action(
        self,
        actor: str,
        action: str,
        context: gm_modules.ModuleContext,
    ) -> tuple[bool, str | None]:
        del actor, action, context
        return True, None

    def update_state(
        self,
        event: str,
        actor: str,
        context: gm_modules.ModuleContext,
    ) -> None:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("injected simultaneous dispatch failure")
        self.events.append((actor, event, context.current_round))

    def get_observation_context(
        self,
        observer: str,
        context: gm_modules.ModuleContext,
    ) -> str:
        del observer, context
        return ""

    def get_module_report(self) -> str:
        return "recording"

    def get_state(self) -> dict[str, Any]:
        return {"events": [list(item) for item in self.events]}

    def set_state(self, state: dict[str, Any]) -> None:
        self.events = [tuple(item) for item in state.get("events", ())]


def _tracker() -> gm_state.NegotiationStateTracker:
    now = datetime.datetime(2040, 1, 2, 3, 4)
    return gm_state.NegotiationStateTracker(
        max_rounds=10,
        enable_deadlines=False,
        clock=lambda: now,
    )


def _service(
    *,
    validators: tuple[ValidatorBinding, ...] = (),
    module: RecordingModule | None = None,
    after_event_append: Any = None,
    before_resolution_store: Any = None,
    protocol: str = "simultaneous",
) -> tuple[NegotiationAdjudicator, gm_state.NegotiationStateTracker]:
    tracker = _tracker()
    service = NegotiationAdjudicator(
        negotiation_id="sim-1",
        participants=("Alice", "Bob"),
        state_tracker=tracker,
        protocol=protocol,
        validators=validators,
        modules=({"recording": module} if module else None),
        after_event_append=after_event_append,
        before_resolution_store=before_resolution_store,
    )
    return service, tracker


def _offer(actor: str, recipient: str, ref: str, price: int) -> NegotiationAction:
    return NegotiationAction(
        action_ref=ref,
        actor_id=actor,
        kind=ActionKind.OFFER,
        offer=Offer(actor, recipient, {"price": price}),
        raw_text=f"{actor} offers {price}",
    )


def _round_actions(suffix: str = "0") -> tuple[NegotiationAction, ...]:
    return (
        _offer("Alice", "Bob", f"alice-{suffix}", 70),
        _offer("Bob", "Alice", f"bob-{suffix}", 60),
    )


def test_batch_validators_share_one_pre_round_state_and_hide_peer_actions() -> None:
    seen: list[tuple[str, int, int, tuple[str, ...]]] = []
    service: NegotiationAdjudicator

    def inspect_pre_round(action, context):
        state = service._state_tracker.get_negotiation("sim-1")  # pylint: disable=protected-access
        seen.append((
            action.actor_id,
            context.current_round,
            len(state.offers_history),
            tuple(sorted(context.shared_data)),
        ))
        # Deliberately impure validator: each peer must still see round zero.
        service._state_tracker.advance_round("sim-1")  # pylint: disable=protected-access
        return True, None

    binding = ValidatorBinding(
        "pre_round_probe",
        5,
        ValidationMode.HARD,
        inspect_pre_round,
    )
    service, tracker = _service(validators=(binding,))

    results = service.submit_batch(reversed(_round_actions()))

    assert [result.event.action.actor_id for result in results] == ["Alice", "Bob"]
    assert [(actor, round_index, offers) for actor, round_index, offers, _ in seen] == [
        ("Alice", 0, 0),
        ("Bob", 0, 0),
    ]
    assert all("peer_actions" not in keys for *_, keys in seen)
    assert len(tracker.get_negotiation("sim-1").offers_history) == 2
    assert tracker.get_negotiation("sim-1").current_round == 1


def test_same_batch_offer_cannot_be_referenced_during_validation() -> None:
    service, tracker = _service()
    alice_offer = _offer("Alice", "Bob", "alice-offer", 75)
    bob_accept = NegotiationAction(
        action_ref="bob-accept",
        actor_id="Bob",
        kind=ActionKind.ACCEPT,
        referenced_offer_id=alice_offer.offer.offer_id,
    )

    results = service.submit_batch((alice_offer, bob_accept))

    assert all(not result.accepted for result in results)
    bob_core = next(
        decision
        for decision in results[1].decisions
        if decision.validator_id == "core_protocol"
    )
    assert bob_core.allowed is False
    assert "does not exist" in str(bob_core.message)
    assert tracker.get_negotiation("sim-1").offers_history == []
    assert tracker.get_negotiation("sim-1").current_round == 0


def test_batch_input_order_does_not_change_events_or_state() -> None:
    actions = _round_actions()
    left, left_tracker = _service()
    right, right_tracker = _service()

    left_results = left.submit_batch(actions)
    right_results = right.submit_batch(tuple(reversed(actions)))

    assert [item.event.event_id for item in left_results] == [
        item.event.event_id for item in right_results
    ]
    assert left.get_state() == right.get_state()
    assert left_tracker.get_state() == right_tracker.get_state()


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        ((_round_actions()[0],), "Missing simultaneous batch actors: Bob"),
        (
            (
                _round_actions()[0],
                _offer("Alice", "Bob", "alice-duplicate", 65),
            ),
            "Duplicate simultaneous batch actors",
        ),
        (
            (
                _round_actions()[0],
                _offer("Mallory", "Alice", "unknown", 65),
            ),
            "Unknown simultaneous batch actors",
        ),
    ],
)
def test_batch_requires_complete_unique_known_actor_set(actions, message) -> None:
    service, tracker = _service()
    before = copy.deepcopy(service.get_state())

    with pytest.raises(ValueError, match=message):
        service.submit_batch(actions)

    assert service.get_state() == before
    assert tracker.get_negotiation("sim-1").offers_history == []


def test_one_hard_failure_rejects_whole_batch_under_versioned_retry_policy() -> None:
    reject_bob = ValidatorBinding(
        "reject_bob",
        2,
        ValidationMode.HARD,
        lambda action, context: (
            (False, "Bob failed") if action.actor_id == "Bob" else (True, None)
        ),
    )
    service, tracker = _service(validators=(reject_bob,))

    rejected = service.submit_batch(_round_actions("invalid"))

    assert [item.event.status for item in rejected] == [
        EventStatus.REJECTED,
        EventStatus.REJECTED,
    ]
    assert all(
        item.event.rejection_policy is RejectionPolicy.RETRY_FULL_BATCH
        for item in rejected
    )
    assert all(
        item.event.rejection_policy_version == "simultaneous-batch-rejection/1"
        for item in rejected
    )
    assert all(
        any(
            decision.validator_id == "simultaneous_batch_atomicity"
            for decision in item.decisions
        )
        for item in rejected
    )
    assert [item.event.action_sequence for item in rejected] == [0, 1]
    assert [item.event.committed_turn_index for item in rejected] == [0, 0]
    assert tracker.get_negotiation("sim-1").offers_history == []
    assert tracker.get_negotiation("sim-1").current_round == 0


def test_conflicting_terminal_batch_rejects_atomically() -> None:
    service, tracker = _service()
    actions = (
        NegotiationAction("alice-walk", "Alice", ActionKind.WALK_AWAY),
        NegotiationAction("bob-walk", "Bob", ActionKind.WALK_AWAY),
    )

    results = service.submit_batch(actions)

    assert all(not result.accepted for result in results)
    assert all(
        any(
            decision.validator_id == "simultaneous_terminal_conflict"
            for decision in result.decisions
        )
        for result in results
    )
    assert tracker.get_negotiation("sim-1").phase == "opening"


def test_single_terminal_action_is_deferred_until_full_batch_applies() -> None:
    service, tracker = _service()
    actions = (
        NegotiationAction(
            "alice-walk",
            "Alice",
            ActionKind.WALK_AWAY,
            reason="outside option",
        ),
        _offer("Bob", "Alice", "bob-final", 55),
    )

    results = service.submit_batch(actions)
    state = tracker.get_negotiation("sim-1")

    assert all(result.accepted for result in results)
    assert results[0].outcome is not None
    assert results[0].outcome.status is OutcomeStatus.WALK_AWAY
    assert results[1].outcome is None
    assert len(state.offers_history) == 1
    assert state.phase == "failed"
    assert state.current_round == 0
    assert [item.event.committed_turn_index for item in results] == [0, 1]


def test_duplicate_batch_retry_is_idempotent_and_dispatches_once() -> None:
    module = RecordingModule()
    service, tracker = _service(module=module)
    actions = _round_actions()

    first = service.submit_batch(actions)
    second = service.submit_batch(tuple(reversed(actions)))

    assert second == first
    assert len(service.get_event_log()) == 2
    assert len(service.get_receipts()) == 2
    assert len(module.events) == 2
    assert len(tracker.get_negotiation("sim-1").offers_history) == 2
    assert tracker.get_negotiation("sim-1").current_round == 1


def test_batch_append_crash_restarts_and_applies_entire_round_once() -> None:
    def fail_after_append(event: InteractionEvent) -> None:
        raise RuntimeError(f"crash after batch append {event.event_id}")

    service, tracker = _service(after_event_append=fail_after_append)
    actions = _round_actions()
    with pytest.raises(RuntimeError, match="crash after batch append"):
        service.submit_batch(actions)

    assert len(service.get_event_log()) == 2
    assert tracker.get_negotiation("sim-1").offers_history == []
    snapshot = json.loads(json.dumps(service.get_state()))
    assert len(snapshot["pending_batches"]) == 1

    restored, restored_tracker = _service()
    restored.set_state(snapshot)
    results = restored.submit_batch(tuple(reversed(actions)))

    assert all(result.accepted for result in results)
    assert len(restored.get_event_log()) == 2
    assert len(restored_tracker.get_negotiation("sim-1").offers_history) == 2
    assert restored_tracker.get_negotiation("sim-1").current_round == 1
    assert restored.get_state()["pending_batches"] == {}


def test_apply_before_resolution_crash_replays_without_round_duplication() -> None:
    def fail_before_resolution(event: InteractionEvent) -> None:
        raise RuntimeError(f"crash before batch resolution {event.event_id}")

    service, tracker = _service(before_resolution_store=fail_before_resolution)
    actions = _round_actions()
    with pytest.raises(RuntimeError, match="crash before batch resolution"):
        service.submit_batch(actions)

    assert len(tracker.get_negotiation("sim-1").offers_history) == 2
    assert tracker.get_negotiation("sim-1").current_round == 1
    snapshot = json.loads(json.dumps(service.get_state()))

    restored, restored_tracker = _service()
    restored.set_state(snapshot)
    restored.submit_batch(actions)

    assert len(restored_tracker.get_negotiation("sim-1").offers_history) == 2
    assert restored_tracker.get_negotiation("sim-1").current_round == 1
    assert restored.get_state()["pending_batches"] == {}


def test_restart_after_partial_batch_apply_finishes_remaining_prefix_once() -> None:
    def stop_after_append(event: InteractionEvent) -> None:
        raise RuntimeError(event.event_id)

    service, tracker = _service(after_event_append=stop_after_append)
    actions = _round_actions()
    with pytest.raises(RuntimeError):
        service.submit_batch(actions)
    first_event = service.get_event_log()[0]
    service._apply_committed_action(  # pylint: disable=protected-access
        first_event,
        defer_terminal=True,
        advance_round=False,
    )
    assert len(tracker.get_negotiation("sim-1").offers_history) == 1
    snapshot = json.loads(json.dumps(service.get_state()))

    restored, restored_tracker = _service()
    restored.set_state(snapshot)
    restored.submit_batch(actions)

    state = restored_tracker.get_negotiation("sim-1")
    assert len(state.offers_history) == 2
    assert state.current_round == 1
    assert len(restored.get_event_log()) == 2
    assert restored.get_state()["committed_turn_index"] == 2


def test_restart_after_last_apply_but_before_batch_finalize_advances_once() -> None:
    def stop_after_append(event: InteractionEvent) -> None:
        raise RuntimeError(event.event_id)

    service, tracker = _service(after_event_append=stop_after_append)
    actions = _round_actions()
    with pytest.raises(RuntimeError):
        service.submit_batch(actions)
    for event in service.get_event_log():
        service._apply_committed_action(  # pylint: disable=protected-access
            event,
            defer_terminal=True,
            advance_round=False,
        )
    assert len(tracker.get_negotiation("sim-1").offers_history) == 2
    assert tracker.get_negotiation("sim-1").current_round == 0
    snapshot = json.loads(json.dumps(service.get_state()))
    assert snapshot["finalized_batch_ids"] == []

    restored, restored_tracker = _service()
    restored.set_state(snapshot)
    restored.submit_batch(actions)

    state = restored_tracker.get_negotiation("sim-1")
    assert len(state.offers_history) == 2
    assert state.current_round == 1
    assert len(restored.get_state()["finalized_batch_ids"]) == 1


def test_unexpected_apply_failure_rolls_back_whole_batch_before_retry(
    monkeypatch,
) -> None:
    service, tracker = _service()
    original_record_offer = tracker.record_offer

    def fail_second_offer(
        negotiation_id,
        offerer,
        recipient,
        offer_type,
        terms,
    ):
        if offerer == "Bob":
            raise RuntimeError("injected apply failure")
        return original_record_offer(
            negotiation_id,
            offerer,
            recipient,
            offer_type,
            terms,
        )

    monkeypatch.setattr(tracker, "record_offer", fail_second_offer)
    actions = _round_actions()
    with pytest.raises(RuntimeError, match="injected apply failure"):
        service.submit_batch(actions)

    state = tracker.get_negotiation("sim-1")
    assert state.offers_history == []
    assert state.current_round == 0
    assert service.get_state()["committed_turn_index"] == 0
    assert service.get_state()["applied_event_ids"] == []
    assert len(service.get_state()["pending_batches"]) == 1

    monkeypatch.setattr(tracker, "record_offer", original_record_offer)
    results = service.submit_batch(tuple(reversed(actions)))
    assert all(result.accepted for result in results)
    assert len(tracker.get_negotiation("sim-1").offers_history) == 2
    assert tracker.get_negotiation("sim-1").current_round == 1


def test_dispatch_failure_retry_and_restart_preserve_exactly_once_receipts() -> None:
    module = RecordingModule(failures_remaining=1)
    service, _ = _service(module=module)
    actions = _round_actions()
    with pytest.raises(ModuleDispatchError):
        service.submit_batch(actions)

    snapshot = json.loads(json.dumps(service.get_state()))
    restored_module = RecordingModule()
    restored, tracker = _service(module=restored_module)
    restored.set_state(snapshot)
    restored.submit_batch(actions)

    assert len(restored_module.events) == 2
    assert len(restored.get_receipts()) == 2
    assert len(tracker.get_negotiation("sim-1").offers_history) == 2


def test_round_and_indices_advance_only_after_each_complete_batch() -> None:
    service, tracker = _service()

    first = service.submit_batch(_round_actions("first"))
    second = service.submit_batch((
        _offer("Alice", "Bob", "alice-second", 71),
        _offer("Bob", "Alice", "bob-second", 61),
    ))
    events = [*first, *second]

    assert [item.event.round_index for item in events] == [0, 0, 1, 1]
    assert [item.event.action_sequence for item in events] == [0, 1, 2, 3]
    assert [item.event.committed_turn_index for item in events] == [0, 1, 2, 3]
    assert tracker.get_negotiation("sim-1").current_round == 2


def test_protocol_entrypoints_fail_closed_without_changing_alternating() -> None:
    simultaneous, _ = _service()
    with pytest.raises(RuntimeError, match="requires submit_batch"):
        simultaneous.submit(_round_actions()[0])

    alternating, tracker = _service(protocol="alternating")
    accepted = alternating.submit(_round_actions()[0])
    assert accepted.accepted is True
    assert alternating.next_actor == "Bob"
    assert tracker.get_negotiation("sim-1").current_round == 0
    with pytest.raises(RuntimeError, match="only valid for simultaneous"):
        alternating.submit_batch(_round_actions())
