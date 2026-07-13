"""Transactional public adjudication boundary for negotiation game masters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any

from concordia.typing import entity_component

from negotiation.domain.schema import (
    ActionKind,
    Agreement,
    NegotiationAction,
    Offer,
    Outcome,
    OutcomeStatus,
    _verify_serialized_id,
    freeze_mapping,
    stable_id,
    thaw_json,
)
from negotiation.game_master.components import gm_modules
from negotiation.game_master.components import gm_state
from negotiation.game_master.components import gm_validation


ADJUDICATION_VERSION = "negotiation-adjudication/2"
SIMULTANEOUS_BATCH_VERSION = "simultaneous-batch/1"
_VALIDATION_DECISION_FIELDS = frozenset({
    "validator_id",
    "priority",
    "mode",
    "allowed",
    "message",
    "decision_id",
})
_INTERACTION_EVENT_FIELDS = frozenset({
    "event_version",
    "negotiation_id",
    "action",
    "status",
    "round_index",
    "action_sequence",
    "committed_turn_index",
    "decisions",
    "module_context",
    "rejection_policy",
    "rejection_policy_version",
    "event_id",
    "idempotency_key",
})
_NEGOTIATION_ACTION_FIELDS = frozenset({
    "action_version",
    "action_ref",
    "actor_id",
    "kind",
    "offer",
    "referenced_offer_id",
    "reason",
    "raw_text",
    "action_id",
})
_OFFER_FIELDS = frozenset({
    "offer_version",
    "actor_id",
    "recipient_id",
    "offer_type",
    "terms",
    "offer_id",
})


def _require_exact_fields(
    data: Mapping[str, Any],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    if not isinstance(data, Mapping):
        raise TypeError(f"{context} must be a mapping")
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


class ValidationMode(str, Enum):
    """Whether a failed validator blocks commit or remains advisory evidence."""

    HARD = "hard"
    ADVISORY = "advisory"


class EventStatus(str, Enum):
    """Persistence status of an adjudicated action."""

    COMMITTED = "committed"
    REJECTED = "rejected"


class RejectionPolicy(str, Enum):
    """Versioned protocol response to a hard validation rejection."""

    RETRY_SAME_ACTOR = "retry_same_actor"
    RETRY_FULL_BATCH = "retry_full_batch"


@dataclass(frozen=True)
class ValidationDecision:
    """One ordered validator result retained on the audit event."""

    validator_id: str
    priority: int
    mode: ValidationMode
    allowed: bool
    message: str | None = None
    decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.validator_id, str) or not self.validator_id:
            raise ValueError("validator_id cannot be empty")
        if type(self.priority) is not int:
            raise TypeError("priority must be an integer")
        if type(self.allowed) is not bool:
            raise TypeError("allowed must be a boolean")
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError("message must be a string or None")
        object.__setattr__(self, "decision_id", stable_id("validation", self._content_dict()))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "validator_id": self.validator_id,
            "priority": self.priority,
            "mode": self.mode.value,
            "allowed": self.allowed,
            "message": self.message,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "decision_id": self.decision_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ValidationDecision:
        _require_exact_fields(
            data,
            _VALIDATION_DECISION_FIELDS,
            context="serialized validation decision",
        )
        decision = cls(
            validator_id=data["validator_id"],
            priority=data["priority"],
            mode=ValidationMode(data["mode"]),
            allowed=data["allowed"],
            message=data.get("message"),
        )
        _verify_serialized_id(data, "decision_id", decision.decision_id)
        return decision


ValidatorCallable = Callable[
    [NegotiationAction, gm_modules.ModuleContext],
    bool | tuple[bool, str | None],
]


@dataclass(frozen=True)
class ValidatorBinding:
    """Runtime binding for an explicitly prioritized validator."""

    validator_id: str
    priority: int
    mode: ValidationMode
    validator: ValidatorCallable = field(compare=False, repr=False)

    def evaluate(
        self,
        action: NegotiationAction,
        context: gm_modules.ModuleContext,
    ) -> ValidationDecision:
        raw = self.validator(action, context)
        if isinstance(raw, tuple):
            allowed, message = raw
        else:
            allowed, message = raw, None
        return ValidationDecision(
            validator_id=self.validator_id,
            priority=self.priority,
            mode=self.mode,
            allowed=allowed,
            message=message,
        )


@dataclass(frozen=True)
class InteractionEvent:
    """An immutable committed or rejected negotiation action event."""

    negotiation_id: str
    action: NegotiationAction
    status: EventStatus
    round_index: int
    action_sequence: int
    committed_turn_index: int
    decisions: tuple[ValidationDecision, ...]
    module_context: Mapping[str, Any]
    rejection_policy: RejectionPolicy = RejectionPolicy.RETRY_SAME_ACTOR
    rejection_policy_version: str = "rejection-policy/1"
    event_version: str = ADJUDICATION_VERSION
    event_id: str = field(init=False)
    idempotency_key: str = field(init=False)

    def __post_init__(self) -> None:
        if self.event_version != ADJUDICATION_VERSION:
            raise ValueError("unsupported adjudication event version")
        if (
            not isinstance(self.negotiation_id, str)
            or not self.negotiation_id
            or type(self.round_index) is not int
            or type(self.action_sequence) is not int
            or type(self.committed_turn_index) is not int
            or self.round_index < 0
            or self.action_sequence < 0
            or self.committed_turn_index < 0
        ):
            raise ValueError("Event negotiation ID and non-negative indices are required")
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "module_context", freeze_mapping(self.module_context))
        content = self._content_dict()
        object.__setattr__(self, "event_id", stable_id("event", content))
        object.__setattr__(
            self,
            "idempotency_key",
            stable_id(
                "idempotency",
                {
                    "negotiation_id": self.negotiation_id,
                    "action_id": self.action.action_id,
                },
            ),
        )

    @property
    def committed(self) -> bool:
        return self.status is EventStatus.COMMITTED

    def _content_dict(self) -> dict[str, Any]:
        return {
            "event_version": self.event_version,
            "negotiation_id": self.negotiation_id,
            "action": self.action.to_dict(),
            "status": self.status.value,
            "round_index": self.round_index,
            "action_sequence": self.action_sequence,
            "committed_turn_index": self.committed_turn_index,
            "decisions": [item.to_dict() for item in self.decisions],
            "module_context": thaw_json(self.module_context),
            "rejection_policy": self.rejection_policy.value,
            "rejection_policy_version": self.rejection_policy_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content_dict(),
            "event_id": self.event_id,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InteractionEvent:
        if not isinstance(data, Mapping):
            raise TypeError("serialized interaction event must be a mapping")
        if data.get("event_version") != ADJUDICATION_VERSION:
            raise ValueError("unsupported adjudication event version")
        if "event_id" not in data:
            raise ValueError("event_id is required for persisted domain records")
        if "idempotency_key" not in data:
            raise ValueError(
                "idempotency_key is required for persisted domain records"
            )
        _require_exact_fields(
            data,
            _INTERACTION_EVENT_FIELDS,
            context="serialized interaction event",
        )
        action_payload = data["action"]
        _require_exact_fields(
            action_payload,
            _NEGOTIATION_ACTION_FIELDS,
            context="serialized interaction action",
        )
        offer_payload = action_payload["offer"]
        if offer_payload is not None:
            _require_exact_fields(
                offer_payload,
                _OFFER_FIELDS,
                context="serialized interaction offer",
            )
        decisions_payload = data["decisions"]
        if not isinstance(decisions_payload, list):
            raise TypeError("serialized interaction decisions must be an array")
        if not isinstance(data["module_context"], Mapping):
            raise TypeError("serialized interaction module_context must be a mapping")
        if (
            not isinstance(data["rejection_policy_version"], str)
            or not data["rejection_policy_version"]
        ):
            raise ValueError("rejection_policy_version must be a non-empty string")
        event = cls(
            event_version=data["event_version"],
            negotiation_id=data["negotiation_id"],
            action=NegotiationAction.from_dict(data["action"]),
            status=EventStatus(data["status"]),
            round_index=data["round_index"],
            action_sequence=data["action_sequence"],
            committed_turn_index=data["committed_turn_index"],
            decisions=tuple(
                ValidationDecision.from_dict(item) for item in decisions_payload
            ),
            module_context=data["module_context"],
            rejection_policy=RejectionPolicy(data["rejection_policy"]),
            rejection_policy_version=data["rejection_policy_version"],
        )
        _verify_serialized_id(data, "event_id", event.event_id)
        _verify_serialized_id(data, "idempotency_key", event.idempotency_key)
        return event


@dataclass(frozen=True)
class DispatchReceipt:
    """Proof that one module version applied one committed event."""

    event_id: str
    module_id: str
    module_version: str
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.event_id, self.module_id, self.module_version)
        ):
            raise ValueError("Receipt event, module, and version are required")
        object.__setattr__(self, "receipt_id", stable_id("receipt", self._content_dict()))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "module_id": self.module_id,
            "module_version": self.module_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DispatchReceipt:
        receipt = cls(
            event_id=data["event_id"],
            module_id=data["module_id"],
            module_version=data["module_version"],
        )
        _verify_serialized_id(data, "receipt_id", receipt.receipt_id)
        return receipt


@dataclass(frozen=True)
class ActionResolution:
    """Immutable outcome returned by the public structured submission API."""

    event: InteractionEvent
    agreement: Agreement | None = None
    outcome: Outcome | None = None
    resolution_version: str = ADJUDICATION_VERSION
    resolution_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.resolution_version != ADJUDICATION_VERSION:
            raise ValueError("unsupported adjudication resolution version")
        if not self.event.committed and (self.agreement or self.outcome):
            raise ValueError("Rejected events cannot create agreements or outcomes")
        object.__setattr__(self, "resolution_id", stable_id("resolution", self._content_dict()))

    @property
    def accepted(self) -> bool:
        return self.event.committed

    @property
    def decisions(self) -> tuple[ValidationDecision, ...]:
        return self.event.decisions

    def _content_dict(self) -> dict[str, Any]:
        return {
            "resolution_version": self.resolution_version,
            "event": self.event.to_dict(),
            "agreement": self.agreement.to_dict() if self.agreement else None,
            "outcome": self.outcome.to_dict() if self.outcome else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "resolution_id": self.resolution_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActionResolution:
        if data.get("resolution_version") != ADJUDICATION_VERSION:
            raise ValueError("unsupported adjudication resolution version")
        agreement = data.get("agreement")
        outcome = data.get("outcome")
        resolution = cls(
            resolution_version=data["resolution_version"],
            event=InteractionEvent.from_dict(data["event"]),
            agreement=Agreement.from_dict(agreement) if agreement else None,
            outcome=Outcome.from_dict(outcome) if outcome else None,
        )
        _verify_serialized_id(data, "resolution_id", resolution.resolution_id)
        return resolution


class ModuleDispatchError(RuntimeError):
    """A committed event has one or more module receipts still missing."""

    def __init__(self, event_id: str, module_id: str, cause: Exception):
        super().__init__(
            f"Committed event {event_id} could not update module {module_id}: {cause}"
        )
        self.event_id = event_id
        self.module_id = module_id


class NegotiationAdjudicator(entity_component.ContextComponent):
    """Validate, commit, apply, and exactly-once dispatch structured actions."""

    def __init__(
        self,
        negotiation_id: str,
        participants: Sequence[str],
        state_tracker: gm_state.NegotiationStateTracker,
        validator: gm_validation.NegotiationValidator | None = None,
        modules: Mapping[str, gm_modules.NegotiationGMModule] | None = None,
        validators: Sequence[ValidatorBinding] = (),
        active_modules: Mapping[str, set[str]] | None = None,
        module_validation_modes: Mapping[str, ValidationMode | str] | None = None,
        protocol: str = "alternating",
        rejection_policy: RejectionPolicy | str | None = None,
        after_event_append: Callable[[InteractionEvent], None] | None = None,
        before_resolution_store: Callable[[InteractionEvent], None] | None = None,
    ) -> None:
        if not negotiation_id:
            raise ValueError("negotiation_id cannot be empty")
        participant_tuple = tuple(str(item) for item in participants)
        if len(participant_tuple) < 2 or len(set(participant_tuple)) != len(
            participant_tuple
        ):
            raise ValueError("Adjudication requires at least two unique participants")
        if protocol not in {"alternating", "simultaneous"}:
            raise ValueError("protocol must be 'alternating' or 'simultaneous'")
        self.negotiation_id = negotiation_id
        self.participants = participant_tuple
        self.protocol = protocol
        default_policy = (
            RejectionPolicy.RETRY_SAME_ACTOR
            if protocol == "alternating"
            else RejectionPolicy.RETRY_FULL_BATCH
        )
        self.rejection_policy = RejectionPolicy(rejection_policy or default_policy)
        if protocol == "alternating" and (
            self.rejection_policy is not RejectionPolicy.RETRY_SAME_ACTOR
        ):
            raise ValueError("alternating protocol requires retry_same_actor")
        if protocol == "simultaneous" and (
            self.rejection_policy is not RejectionPolicy.RETRY_FULL_BATCH
        ):
            raise ValueError("simultaneous protocol requires retry_full_batch")
        self.rejection_policy_version = (
            "rejection-policy/1"
            if protocol == "alternating"
            else "simultaneous-batch-rejection/1"
        )
        self._after_event_append = after_event_append
        self._before_resolution_store = before_resolution_store
        self._state_tracker = state_tracker
        self._validator = validator
        self._modules = dict(modules or {})
        self._active_modules = {
            participant: set((active_modules or {}).get(participant, set()))
            for participant in participant_tuple
        }
        self._module_validation_modes = {
            key: ValidationMode(value)
            for key, value in (module_validation_modes or {}).items()
        }
        self._custom_validators = tuple(validators)
        self._event_log: list[InteractionEvent] = []
        self._events_by_action_id: dict[str, InteractionEvent] = {}
        self._pending_events: dict[str, InteractionEvent] = {}
        self._pending_batches: dict[str, tuple[str, ...]] = {}
        self._finalized_batch_ids: set[str] = set()
        self._resolutions: dict[str, ActionResolution] = {}
        self._receipts: list[DispatchReceipt] = []
        self._receipt_keys: set[tuple[str, str, str]] = set()
        self._applied_event_ids: set[str] = set()
        self._offers: dict[str, Offer] = {}
        self._legacy_offers: dict[str, gm_state.NegotiationOffer] = {}
        self._offer_legacy_indices: dict[str, int] = {}
        self._proposal_sequence = 0
        self._committed_turn_index = 0
        self._outcome: Outcome | None = None

        for module_id, module in self._modules.items():
            if not callable(getattr(module, "get_state", None)) or not callable(
                getattr(module, "set_state", None)
            ):
                raise TypeError(
                    f"Module {module_id} must support get_state/set_state for atomic dispatch"
                )
        self._state_tracker.start_negotiation(
            self.negotiation_id,
            list(self.participants),
        )

    @property
    def next_actor(self) -> str | None:
        if self._outcome is not None or self.protocol == "simultaneous":
            return None
        return self.participants[
            self._committed_turn_index % len(self.participants)
        ]

    def submit(self, action: NegotiationAction) -> ActionResolution:
        """Submit one explicit action; duplicate action IDs resume missing dispatch only."""
        if self.protocol == "simultaneous":
            raise RuntimeError(
                "simultaneous protocol requires submit_batch() with one action "
                "per participant"
            )
        existing = self._resolutions.get(action.action_id)
        if existing is not None:
            if existing.accepted:
                self._dispatch_missing(existing.event)
            return existing
        pending = self._pending_events.get(action.action_id)
        if pending is not None:
            return self._resume_pending_event(pending)
        if self._pending_events:
            raise RuntimeError(
                "A committed event is pending; retry that action before submitting another"
            )

        context = self._make_module_context(action)
        decisions = tuple(
            binding.evaluate(action, context) for binding in self._validator_bindings()
        )
        rejected = any(
            not decision.allowed and decision.mode is ValidationMode.HARD
            for decision in decisions
        )
        event = InteractionEvent(
            negotiation_id=self.negotiation_id,
            action=action,
            status=EventStatus.REJECTED if rejected else EventStatus.COMMITTED,
            round_index=self._committed_turn_index // len(self.participants),
            action_sequence=self._proposal_sequence,
            committed_turn_index=self._committed_turn_index,
            decisions=decisions,
            module_context=self._context_to_dict(context),
            rejection_policy=self.rejection_policy,
            rejection_policy_version=self.rejection_policy_version,
        )
        self._event_log.append(event)
        self._events_by_action_id[action.action_id] = event
        self._proposal_sequence += 1

        if rejected:
            resolution = ActionResolution(event=event)
            self._resolutions[action.action_id] = resolution
            return resolution

        self._pending_events[action.action_id] = event
        if self._after_event_append is not None:
            self._after_event_append(event)
        return self._resume_pending_event(event, context)

    def submit_batch(
        self,
        actions: Sequence[NegotiationAction],
    ) -> tuple[ActionResolution, ...]:
        """Atomically adjudicate one simultaneous round from one frozen state.

        The versioned policy rejects the entire batch when any hard validator
        rejects or when multiple terminal actions conflict. A sole terminal
        action is committed in participant order, but tracker termination is
        deferred until every action in the accepted batch has been applied.
        """
        if self.protocol != "simultaneous":
            raise RuntimeError("submit_batch() is only valid for simultaneous protocol")
        ordered_actions = self._normalize_simultaneous_batch(actions)
        batch_id = self._batch_id(ordered_actions)
        action_ids = tuple(action.action_id for action in ordered_actions)

        existing = tuple(self._resolutions.get(action_id) for action_id in action_ids)
        if all(resolution is not None for resolution in existing):
            self._verify_existing_batch(batch_id, ordered_actions)
            resolved = tuple(resolution for resolution in existing if resolution)
            for resolution in resolved:
                if resolution.accepted:
                    self._dispatch_missing(resolution.event)
            return resolved
        if any(resolution is not None for resolution in existing):
            raise RuntimeError("Simultaneous batch overlaps a previously resolved batch")

        if batch_id in self._pending_batches:
            if self._pending_batches[batch_id] != action_ids:
                raise ValueError("Pending simultaneous batch action IDs do not match")
            return self._resume_pending_batch(batch_id)
        if self._pending_batches or self._pending_events:
            raise RuntimeError(
                "A committed simultaneous batch is pending; retry that full batch"
            )
        overlapping = [
            action.action_id
            for action in ordered_actions
            if action.action_id in self._events_by_action_id
        ]
        if overlapping:
            raise RuntimeError(
                "Simultaneous batch reuses action IDs from another batch: "
                + ", ".join(overlapping)
            )

        contexts = tuple(
            self._make_module_context(action, batch_id=batch_id)
            for action in ordered_actions
        )
        decision_rows = self._evaluate_simultaneous_validators(
            ordered_actions,
            contexts,
        )
        terminal_count = sum(
            action.kind in {ActionKind.ACCEPT, ActionKind.WALK_AWAY}
            for action in ordered_actions
        )
        if terminal_count > 1:
            terminal_decision = ValidationDecision(
                validator_id="simultaneous_terminal_conflict",
                priority=-200,
                mode=ValidationMode.HARD,
                allowed=False,
                message="A simultaneous batch may contain at most one terminal action",
            )
            decision_rows = [
                (terminal_decision, *decisions) for decisions in decision_rows
            ]
        hard_rejected = any(
            not decision.allowed and decision.mode is ValidationMode.HARD
            for decisions in decision_rows
            for decision in decisions
        )
        if hard_rejected:
            atomic_decision = ValidationDecision(
                validator_id="simultaneous_batch_atomicity",
                priority=-201,
                mode=ValidationMode.HARD,
                allowed=False,
                message=(
                    "The full simultaneous batch is rejected when any proposal "
                    "fails hard validation"
                ),
            )
            decision_rows = [
                (atomic_decision, *decisions) for decisions in decision_rows
            ]

        round_index = self._committed_turn_index // len(self.participants)
        turn_base = self._committed_turn_index
        events = []
        for offset, (action, context, decisions) in enumerate(
            zip(ordered_actions, contexts, decision_rows)
        ):
            event = InteractionEvent(
                negotiation_id=self.negotiation_id,
                action=action,
                status=(
                    EventStatus.REJECTED
                    if hard_rejected
                    else EventStatus.COMMITTED
                ),
                round_index=round_index,
                action_sequence=self._proposal_sequence + offset,
                committed_turn_index=(turn_base if hard_rejected else turn_base + offset),
                decisions=decisions,
                module_context=self._context_to_dict(context),
                rejection_policy=self.rejection_policy,
                rejection_policy_version=self.rejection_policy_version,
            )
            events.append(event)
            self._event_log.append(event)
            self._events_by_action_id[action.action_id] = event
        self._proposal_sequence += len(events)

        if hard_rejected:
            resolutions = tuple(ActionResolution(event=event) for event in events)
            self._resolutions.update({
                resolution.event.action.action_id: resolution
                for resolution in resolutions
            })
            return resolutions

        self._pending_batches[batch_id] = action_ids
        self._pending_events.update({
            event.action.action_id: event for event in events
        })
        if self._after_event_append is not None:
            for event in events:
                self._after_event_append(event)
        return self._resume_pending_batch(batch_id, contexts)

    def get_event_log(self) -> tuple[InteractionEvent, ...]:
        return tuple(self._event_log)

    def get_receipts(self) -> tuple[DispatchReceipt, ...]:
        return tuple(self._receipts)

    def get_resolution(self, action_id: str) -> ActionResolution | None:
        return self._resolutions.get(action_id)

    def _resume_pending_event(
        self,
        event: InteractionEvent,
        context: gm_modules.ModuleContext | None = None,
    ) -> ActionResolution:
        tracker_snapshot = copy.deepcopy(self._state_tracker.get_state())
        offers_snapshot = dict(self._offers)
        legacy_snapshot = dict(self._legacy_offers)
        indices_snapshot = dict(self._offer_legacy_indices)
        applied_snapshot = set(self._applied_event_ids)
        outcome_snapshot = self._outcome
        committed_turn_snapshot = self._committed_turn_index
        was_applied = event.event_id in self._applied_event_ids
        try:
            agreement, outcome = self._apply_committed_action(event)
        except Exception:
            self._state_tracker.set_state(tracker_snapshot)
            self._offers = offers_snapshot
            self._legacy_offers = legacy_snapshot
            self._offer_legacy_indices = indices_snapshot
            self._applied_event_ids = applied_snapshot
            self._outcome = outcome_snapshot
            self._committed_turn_index = committed_turn_snapshot
            raise
        if not was_applied and self._before_resolution_store is not None:
            self._before_resolution_store(event)
        resolution = ActionResolution(
            event=event,
            agreement=agreement,
            outcome=outcome,
        )
        self._resolutions[event.action.action_id] = resolution
        self._pending_events.pop(event.action.action_id, None)
        self._dispatch_missing(event, context)
        return resolution

    def _resume_pending_batch(
        self,
        batch_id: str,
        contexts: Sequence[gm_modules.ModuleContext] | None = None,
    ) -> tuple[ActionResolution, ...]:
        action_ids = self._pending_batches[batch_id]
        events = tuple(self._pending_events[action_id] for action_id in action_ids)
        applied_before = tuple(
            event.event_id in self._applied_event_ids for event in events
        )

        tracker_snapshot = copy.deepcopy(self._state_tracker.get_state())
        offers_snapshot = dict(self._offers)
        legacy_snapshot = dict(self._legacy_offers)
        indices_snapshot = dict(self._offer_legacy_indices)
        applied_snapshot = set(self._applied_event_ids)
        finalized_snapshot = set(self._finalized_batch_ids)
        outcome_snapshot = self._outcome
        committed_turn_snapshot = self._committed_turn_index
        event_results: list[tuple[Agreement | None, Outcome | None]] = []
        try:
            event_results = [
                (
                    self._derive_event_result(event)
                    if was_applied
                    else self._apply_committed_action(
                        event,
                        defer_terminal=True,
                        advance_round=False,
                    )
                )
                for event, was_applied in zip(events, applied_before)
            ]
            if batch_id not in self._finalized_batch_ids:
                terminal = next(
                    (outcome for _, outcome in event_results if outcome is not None),
                    None,
                )
                if terminal is None:
                    self._state_tracker.advance_round(self.negotiation_id)
                else:
                    self._state_tracker.terminate_negotiation(
                        self.negotiation_id,
                        reason=terminal.reason,
                        success=terminal.status is OutcomeStatus.AGREEMENT,
                    )
                    self._outcome = terminal
                self._finalized_batch_ids.add(batch_id)
        except Exception:
            self._state_tracker.set_state(tracker_snapshot)
            self._offers = offers_snapshot
            self._legacy_offers = legacy_snapshot
            self._offer_legacy_indices = indices_snapshot
            self._applied_event_ids = applied_snapshot
            self._finalized_batch_ids = finalized_snapshot
            self._outcome = outcome_snapshot
            self._committed_turn_index = committed_turn_snapshot
            raise

        if not all(applied_before) and self._before_resolution_store is not None:
            for event in events:
                self._before_resolution_store(event)
        resolutions = tuple(
            ActionResolution(event=event, agreement=agreement, outcome=outcome)
            for event, (agreement, outcome) in zip(events, event_results)
        )
        self._resolutions.update({
            resolution.event.action.action_id: resolution
            for resolution in resolutions
        })
        for action_id in action_ids:
            self._pending_events.pop(action_id, None)
        self._pending_batches.pop(batch_id, None)
        for resolution, context in zip(resolutions, contexts or (None,) * len(events)):
            self._dispatch_missing(resolution.event, context)
        return resolutions

    def _normalize_simultaneous_batch(
        self,
        actions: Sequence[NegotiationAction],
    ) -> tuple[NegotiationAction, ...]:
        action_tuple = tuple(actions)
        if any(not isinstance(action, NegotiationAction) for action in action_tuple):
            raise TypeError("Every simultaneous proposal must be a NegotiationAction")
        actors = [action.actor_id for action in action_tuple]
        unknown = sorted(set(actors) - set(self.participants))
        if unknown:
            raise ValueError(
                "Unknown simultaneous batch actors: " + ", ".join(unknown)
            )
        duplicates = sorted({actor for actor in actors if actors.count(actor) > 1})
        if duplicates:
            raise ValueError(
                "Duplicate simultaneous batch actors: " + ", ".join(duplicates)
            )
        missing = [actor for actor in self.participants if actor not in actors]
        if missing:
            raise ValueError(
                "Missing simultaneous batch actors: " + ", ".join(missing)
            )
        if len(action_tuple) != len(self.participants):
            raise ValueError(
                "A simultaneous batch requires exactly one action per participant"
            )
        by_actor = {action.actor_id: action for action in action_tuple}
        ordered = tuple(by_actor[actor] for actor in self.participants)
        if len({action.action_id for action in ordered}) != len(ordered):
            raise ValueError("Simultaneous batch action IDs must be unique")
        return ordered

    def _evaluate_simultaneous_validators(
        self,
        actions: Sequence[NegotiationAction],
        contexts: Sequence[gm_modules.ModuleContext],
    ) -> list[tuple[ValidationDecision, ...]]:
        """Evaluate every proposal against the identical restored pre-round state."""
        tracker_snapshot = copy.deepcopy(self._state_tracker.get_state())
        offers_snapshot = dict(self._offers)
        indices_snapshot = dict(self._offer_legacy_indices)
        outcome_snapshot = self._outcome
        turn_snapshot = self._committed_turn_index
        module_snapshots = {
            module_id: copy.deepcopy(module.get_state())
            for module_id, module in self._modules.items()
        }

        def restore_pre_round() -> None:
            self._state_tracker.set_state(copy.deepcopy(tracker_snapshot))
            self._offers = dict(offers_snapshot)
            self._offer_legacy_indices = dict(indices_snapshot)
            tracker = self._state_tracker.get_negotiation(self.negotiation_id)
            self._legacy_offers = {
                offer_id: tracker.offers_history[index]
                for offer_id, index in self._offer_legacy_indices.items()
            }
            self._outcome = outcome_snapshot
            self._committed_turn_index = turn_snapshot
            for module_id, module_state in module_snapshots.items():
                self._modules[module_id].set_state(copy.deepcopy(module_state))

        decision_rows = []
        try:
            for action, context in zip(actions, contexts):
                restore_pre_round()
                decision_rows.append(tuple(
                    binding.evaluate(action, context)
                    for binding in self._validator_bindings()
                ))
        finally:
            restore_pre_round()
        return decision_rows

    def _batch_id(self, actions: Sequence[NegotiationAction]) -> str:
        return stable_id(
            "batch",
            {
                "batch_version": SIMULTANEOUS_BATCH_VERSION,
                "negotiation_id": self.negotiation_id,
                "participant_order": list(self.participants),
                "action_ids": [action.action_id for action in actions],
            },
        )

    def _verify_existing_batch(
        self,
        batch_id: str,
        actions: Sequence[NegotiationAction],
    ) -> None:
        for action in actions:
            event = self._events_by_action_id.get(action.action_id)
            if event is None:
                raise RuntimeError("Resolved action is absent from the event log")
            shared_data = thaw_json(event.module_context).get("shared_data", {})
            if shared_data.get("batch_id") != batch_id:
                raise RuntimeError("Action IDs belong to another simultaneous batch")

    def dispatch_event(self, event_id: str) -> None:
        """Deliver one locally committed event; existing receipts make this a no-op."""
        event = next((item for item in self._event_log if item.event_id == event_id), None)
        if event is None:
            raise KeyError(f"Unknown event: {event_id}")
        if not event.committed or event.event_id not in self._applied_event_ids:
            raise ValueError("Only applied committed events may be dispatched")
        self._dispatch_missing(event)

    def _validator_bindings(self) -> tuple[ValidatorBinding, ...]:
        bindings = [
            ValidatorBinding(
                validator_id="core_protocol",
                priority=-100,
                mode=ValidationMode.HARD,
                validator=self._validate_core,
            )
        ]
        if self._validator is not None:
            bindings.append(
                ValidatorBinding(
                    validator_id="negotiation_validator",
                    priority=-50,
                    mode=ValidationMode.HARD,
                    validator=self._validate_terms,
                )
            )
        bindings.extend(self._custom_validators)
        for module_id, module in self._modules.items():
            if not module.is_enabled():
                continue
            mode = self._module_validation_modes.get(module_id, ValidationMode.ADVISORY)
            bindings.append(
                ValidatorBinding(
                    validator_id=f"module:{module_id}",
                    priority=module.get_priority(),
                    mode=mode,
                    validator=(
                        lambda action, context, current=module: self._validate_module(
                            current,
                            action,
                            context,
                        )
                    ),
                )
            )
        return tuple(sorted(bindings, key=lambda item: (item.priority, item.validator_id)))

    def _validate_core(
        self,
        action: NegotiationAction,
        context: gm_modules.ModuleContext,
    ) -> tuple[bool, str | None]:
        del context
        if self._outcome is not None:
            return False, "Negotiation is already terminated"
        if action.actor_id not in self.participants:
            return False, "Actor is not a negotiation participant"
        if self.protocol == "alternating" and action.actor_id != self.next_actor:
            return False, f"Expected action from {self.next_actor}"
        if action.kind is ActionKind.OFFER:
            assert action.offer is not None
            if action.offer.recipient_id not in self.participants:
                return False, "Offer recipient is not a negotiation participant"
            if action.offer.offer_id in self._offers:
                return False, "Offer ID already exists"
        if action.kind in {ActionKind.ACCEPT, ActionKind.REJECT}:
            offer = self._offers.get(str(action.referenced_offer_id))
            if offer is None:
                return False, "Referenced offer does not exist"
            if action.actor_id != offer.recipient_id:
                return False, "Only the intended recipient may accept or reject"
            legacy = self._legacy_offers[offer.offer_id]
            if legacy.is_accepted or legacy.is_rejected:
                return False, "Referenced offer is no longer active"
        return True, None

    def _validate_terms(
        self,
        action: NegotiationAction,
        context: gm_modules.ModuleContext,
    ) -> tuple[bool, str | None]:
        if action.kind is not ActionKind.OFFER:
            return True, None
        assert action.offer is not None and self._validator is not None
        allowed, errors = self._validator.validate_offer(
            action.actor_id,
            thaw_json(action.offer.terms),
            thaw_json(context.shared_data),
        )
        return allowed, "; ".join(errors) if errors else None

    def _validate_module(
        self,
        module: gm_modules.NegotiationGMModule,
        action: NegotiationAction,
        context: gm_modules.ModuleContext,
    ) -> tuple[bool, str | None]:
        """Make module validation observational even for an impure implementation."""
        snapshot = copy.deepcopy(module.get_state())
        try:
            return module.validate_action(
                action.actor_id,
                self._action_text(action),
                context,
            )
        finally:
            module.set_state(snapshot)

    def _apply_committed_action(
        self,
        event: InteractionEvent,
        *,
        defer_terminal: bool = False,
        advance_round: bool = True,
    ) -> tuple[Agreement | None, Outcome | None]:
        if event.event_id in self._applied_event_ids:
            return self._derive_event_result(event)
        action = event.action
        agreement = None
        outcome = None
        if action.kind is ActionKind.OFFER:
            assert action.offer is not None
            legacy = self._state_tracker.record_offer(
                self.negotiation_id,
                action.offer.actor_id,
                action.offer.recipient_id,
                action.offer.offer_type,
                thaw_json(action.offer.terms),
            )
            self._offers[action.offer.offer_id] = action.offer
            self._legacy_offers[action.offer.offer_id] = legacy
            tracker_state = self._state_tracker.get_negotiation(self.negotiation_id)
            self._offer_legacy_indices[action.offer.offer_id] = (
                tracker_state.offers_history.index(legacy)
            )
        elif action.kind is ActionKind.ACCEPT:
            offer = self._offers[str(action.referenced_offer_id)]
            self._state_tracker.accept_offer(
                self.negotiation_id,
                self._legacy_offers[offer.offer_id],
                action.actor_id,
            )
            agreement = Agreement(
                negotiation_id=self.negotiation_id,
                offer_id=offer.offer_id,
                parties=(offer.actor_id, offer.recipient_id),
                terms=offer.terms,
            )
            outcome = Outcome(
                negotiation_id=self.negotiation_id,
                status=OutcomeStatus.AGREEMENT,
                reason="Offer accepted",
                agreement_id=agreement.agreement_id,
            )
            if not defer_terminal:
                self._state_tracker.terminate_negotiation(
                    self.negotiation_id,
                    reason="Offer accepted",
                    success=True,
                )
        elif action.kind is ActionKind.REJECT:
            offer = self._offers[str(action.referenced_offer_id)]
            self._state_tracker.reject_offer(
                self.negotiation_id,
                self._legacy_offers[offer.offer_id],
                action.actor_id,
                action.reason,
            )
        elif action.kind is ActionKind.WALK_AWAY:
            outcome = Outcome(
                negotiation_id=self.negotiation_id,
                status=OutcomeStatus.WALK_AWAY,
                reason=action.reason or "Participant walked away",
            )
            if not defer_terminal:
                self._state_tracker.terminate_negotiation(
                    self.negotiation_id,
                    reason=outcome.reason,
                    success=False,
                )

        self._applied_event_ids.add(event.event_id)
        self._committed_turn_index += 1
        if (
            advance_round
            and self._committed_turn_index % len(self.participants) == 0
        ):
            self._state_tracker.advance_round(self.negotiation_id)
        if outcome is not None and not defer_terminal:
            self._outcome = outcome
        return agreement, outcome

    def _derive_event_result(
        self,
        event: InteractionEvent,
    ) -> tuple[Agreement | None, Outcome | None]:
        """Reconstruct deterministic result records after apply-before-store crash."""
        action = event.action
        if action.kind is ActionKind.ACCEPT:
            offer = self._offers[str(action.referenced_offer_id)]
            agreement = Agreement(
                negotiation_id=self.negotiation_id,
                offer_id=offer.offer_id,
                parties=(offer.actor_id, offer.recipient_id),
                terms=offer.terms,
            )
            outcome = Outcome(
                negotiation_id=self.negotiation_id,
                status=OutcomeStatus.AGREEMENT,
                reason="Offer accepted",
                agreement_id=agreement.agreement_id,
            )
            return agreement, outcome
        if action.kind is ActionKind.WALK_AWAY:
            return None, Outcome(
                negotiation_id=self.negotiation_id,
                status=OutcomeStatus.WALK_AWAY,
                reason=action.reason or "Participant walked away",
            )
        return None, None

    def _dispatch_missing(
        self,
        event: InteractionEvent,
        context: gm_modules.ModuleContext | None = None,
    ) -> None:
        if not event.committed:
            return
        module_context = context or self._context_from_dict(event.module_context)
        for module_id, module in sorted(
            self._modules.items(),
            key=lambda item: (item[1].get_priority(), item[0]),
        ):
            if not module.is_enabled():
                continue
            module_version = self._module_version(module)
            receipt_key = (event.event_id, module_id, module_version)
            if receipt_key in self._receipt_keys:
                continue
            snapshot = copy.deepcopy(module.get_state())
            try:
                module.update_state(
                    self._action_text(event.action),
                    event.action.actor_id,
                    module_context,
                )
            except Exception as error:
                module.set_state(snapshot)
                raise ModuleDispatchError(event.event_id, module_id, error) from error
            receipt = DispatchReceipt(
                event_id=event.event_id,
                module_id=module_id,
                module_version=module_version,
            )
            self._receipts.append(receipt)
            self._receipt_keys.add(receipt_key)

    def _make_module_context(
        self,
        action: NegotiationAction,
        *,
        batch_id: str | None = None,
    ) -> gm_modules.ModuleContext:
        state = self._state_tracker.get_negotiation(self.negotiation_id)
        return gm_modules.ModuleContext(
            negotiation_id=self.negotiation_id,
            participants=list(self.participants),
            current_phase=state.phase,
            current_round=state.current_round,
            active_modules={
                participant: set(modules)
                for participant, modules in self._active_modules.items()
            },
            shared_data={
                "action_id": action.action_id,
                "action_kind": action.kind.value,
                "action": action.to_dict(),
                **(
                    {
                        "batch_id": batch_id,
                        "batch_version": SIMULTANEOUS_BATCH_VERSION,
                        "participant_order": list(self.participants),
                    }
                    if batch_id is not None
                    else {}
                ),
            },
        )

    @staticmethod
    def _context_to_dict(context: gm_modules.ModuleContext) -> dict[str, Any]:
        return {
            "negotiation_id": context.negotiation_id,
            "participants": list(context.participants),
            "current_phase": context.current_phase,
            "current_round": context.current_round,
            "active_modules": {
                participant: sorted(modules)
                for participant, modules in context.active_modules.items()
            },
            "shared_data": copy.deepcopy(context.shared_data),
        }

    @staticmethod
    def _context_from_dict(data: Mapping[str, Any]) -> gm_modules.ModuleContext:
        return gm_modules.ModuleContext(
            negotiation_id=str(data["negotiation_id"]),
            participants=[str(item) for item in data["participants"]],
            current_phase=str(data["current_phase"]),
            current_round=int(data["current_round"]),
            active_modules={
                str(participant): set(str(item) for item in modules)
                for participant, modules in data.get("active_modules", {}).items()
            },
            shared_data=dict(data.get("shared_data", {})),
        )

    @staticmethod
    def _action_text(action: NegotiationAction) -> str:
        return action.raw_text or json.dumps(action.to_dict(), sort_keys=True)

    @staticmethod
    def _module_version(module: gm_modules.NegotiationGMModule) -> str:
        explicit = getattr(module, "module_version", None)
        if explicit:
            return str(explicit)
        module_type = type(module)
        return f"{module_type.__module__}.{module_type.__qualname__}"

    def pre_act(self, action_spec: Any) -> str:
        del action_spec
        state = self._state_tracker.get_negotiation(self.negotiation_id)
        if self.protocol == "simultaneous" and self._outcome is None:
            next_actor = "all participants (submit_batch)"
        else:
            next_actor = self.next_actor or "none (terminated)"
        return (
            f"ADJUDICATION BOUNDARY ({self.negotiation_id}):\n"
            f"Phase: {state.phase}\nRound: {state.current_round}\n"
            f"Next actor: {next_actor}\nProposals: {self._proposal_sequence}\n"
            f"Committed actions: {self._committed_turn_index}\n"
            f"Rejection policy: {self.rejection_policy.value}"
        )

    def post_act(self, action_attempt: str) -> str:
        del action_attempt
        return ""

    def pre_observe(self, observation: str) -> str:
        del observation
        return ""

    def post_observe(self) -> str:
        return ""

    def update(self) -> None:
        return None

    @property
    def name(self) -> str:
        return "NegotiationAdjudicator"

    def get_state(self) -> dict[str, Any]:
        return {
            "adjudication_version": ADJUDICATION_VERSION,
            "negotiation_id": self.negotiation_id,
            "participants": list(self.participants),
            "protocol": self.protocol,
            "action_sequence": self._proposal_sequence,
            "committed_turn_index": self._committed_turn_index,
            "rejection_policy": self.rejection_policy.value,
            "rejection_policy_version": self.rejection_policy_version,
            "events": [event.to_dict() for event in self._event_log],
            "pending_action_ids": sorted(self._pending_events),
            "pending_batches": {
                batch_id: list(action_ids)
                for batch_id, action_ids in self._pending_batches.items()
            },
            "finalized_batch_ids": sorted(self._finalized_batch_ids),
            "simultaneous_batch_version": (
                SIMULTANEOUS_BATCH_VERSION
                if self.protocol == "simultaneous"
                else None
            ),
            "resolutions": [
                resolution.to_dict() for resolution in self._resolutions.values()
            ],
            "receipts": [receipt.to_dict() for receipt in self._receipts],
            "applied_event_ids": sorted(self._applied_event_ids),
            "offers": [offer.to_dict() for offer in self._offers.values()],
            "offer_legacy_indices": dict(self._offer_legacy_indices),
            "outcome": self._outcome.to_dict() if self._outcome else None,
            "tracker_state": copy.deepcopy(self._state_tracker.get_state()),
            "module_states": {
                module_id: copy.deepcopy(module.get_state())
                for module_id, module in self._modules.items()
            },
            "module_versions": {
                module_id: self._module_version(module)
                for module_id, module in self._modules.items()
            },
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Atomically restore a checkpoint or leave every live state unchanged."""
        original = self.get_state()
        try:
            self._restore_state(state)
        except Exception:
            try:
                self._restore_state(original)
            except Exception as rollback_error:  # pragma: no cover - catastrophic
                raise RuntimeError(
                    "Adjudicator checkpoint failed and rollback could not restore "
                    "the prior state"
                ) from rollback_error
            raise

    def _restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("adjudication_version") != ADJUDICATION_VERSION:
            raise ValueError("Adjudicator state uses an incompatible version")
        if state.get("negotiation_id") != self.negotiation_id:
            raise ValueError("Adjudicator state belongs to another negotiation")
        if tuple(state.get("participants", ())) != self.participants:
            raise ValueError("Adjudicator participants do not match restored state")
        if state.get("protocol") != self.protocol:
            raise ValueError("Adjudicator protocol does not match restored state")
        if RejectionPolicy(state["rejection_policy"]) is not self.rejection_policy:
            raise ValueError("Adjudicator rejection policy does not match restored state")
        if state["rejection_policy_version"] != self.rejection_policy_version:
            raise ValueError("Adjudicator rejection-policy version does not match")
        expected_batch_version = (
            SIMULTANEOUS_BATCH_VERSION
            if self.protocol == "simultaneous"
            else None
        )
        if state.get("simultaneous_batch_version") != expected_batch_version:
            raise ValueError("Adjudicator simultaneous-batch version does not match")
        module_states = state.get("module_states")
        module_versions = state.get("module_versions")
        if not isinstance(module_states, Mapping):
            raise ValueError("Adjudicator module_states must be a mapping")
        if not isinstance(module_versions, Mapping):
            raise ValueError("Adjudicator module_versions must be a mapping")
        expected_module_ids = set(self._modules)
        if set(module_states) != expected_module_ids:
            raise ValueError("Adjudicator module-state IDs do not match runtime modules")
        if set(module_versions) != expected_module_ids:
            raise ValueError("Adjudicator module-version IDs do not match runtime modules")
        for module_id, module in self._modules.items():
            if module_versions[module_id] != self._module_version(module):
                raise ValueError(
                    f"Adjudicator module version does not match: {module_id}"
                )
        for name in ("action_sequence", "committed_turn_index"):
            value = state.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Adjudicator {name} must be a non-negative integer")
        self._state_tracker.set_state(copy.deepcopy(state["tracker_state"]))
        self._event_log = [
            InteractionEvent.from_dict(item) for item in state.get("events", ())
        ]
        self._events_by_action_id = {
            event.action.action_id: event for event in self._event_log
        }
        resolutions = [
            ActionResolution.from_dict(item) for item in state.get("resolutions", ())
        ]
        if len({item.event.action.action_id for item in resolutions}) != len(
            resolutions
        ):
            raise ValueError("Restored resolutions contain duplicate actions")
        self._resolutions = {
            resolution.event.action.action_id: resolution for resolution in resolutions
        }
        self._receipts = [
            DispatchReceipt.from_dict(item) for item in state.get("receipts", ())
        ]
        self._receipt_keys = {
            (receipt.event_id, receipt.module_id, receipt.module_version)
            for receipt in self._receipts
        }
        applied_event_ids = tuple(state.get("applied_event_ids", ()))
        if any(not isinstance(item, str) or not item for item in applied_event_ids):
            raise ValueError("Restored applied-event IDs must be non-empty strings")
        if len(set(applied_event_ids)) != len(applied_event_ids):
            raise ValueError("Restored applied-event IDs must not contain duplicates")
        self._applied_event_ids = set(applied_event_ids)
        self._offers = {
            offer.offer_id: offer
            for offer in (Offer.from_dict(item) for item in state.get("offers", ()))
        }
        self._offer_legacy_indices = {
            str(offer_id): int(index)
            for offer_id, index in state.get("offer_legacy_indices", {}).items()
        }
        tracker = self._state_tracker.get_negotiation(self.negotiation_id)
        self._legacy_offers = {
            offer_id: tracker.offers_history[index]
            for offer_id, index in self._offer_legacy_indices.items()
        }
        self._proposal_sequence = state["action_sequence"]
        self._committed_turn_index = state["committed_turn_index"]
        self._pending_events = {
            action_id: self._events_by_action_id[action_id]
            for action_id in (str(item) for item in state.get("pending_action_ids", ()))
        }
        self._pending_batches = {
            str(batch_id): tuple(str(item) for item in action_ids)
            for batch_id, action_ids in state.get("pending_batches", {}).items()
        }
        self._finalized_batch_ids = {
            str(item) for item in state.get("finalized_batch_ids", ())
        }
        outcome = state.get("outcome")
        self._outcome = Outcome.from_dict(outcome) if outcome else None
        for module_id, module_state in module_states.items():
            self._modules[module_id].set_state(copy.deepcopy(module_state))
        self._validate_restored_state()

    def _validate_restored_state(self) -> None:
        sequences = [event.action_sequence for event in self._event_log]
        if sequences != list(range(len(self._event_log))):
            raise ValueError("Restored event action sequences are not contiguous")
        if self._proposal_sequence != len(self._event_log):
            raise ValueError("Restored proposal sequence does not match the event log")
        event_ids = {event.event_id for event in self._event_log}
        if len(event_ids) != len(self._event_log):
            raise ValueError("Restored event log contains duplicate event IDs")
        if len(self._events_by_action_id) != len(self._event_log):
            raise ValueError("Restored event log contains duplicate action IDs")
        if not self._applied_event_ids.issubset(event_ids):
            raise ValueError("Restored applied-event IDs are absent from the event log")
        events_by_id = {event.event_id: event for event in self._event_log}
        for action_id, resolution in self._resolutions.items():
            logged = events_by_id.get(resolution.event.event_id)
            if logged is None or logged != resolution.event:
                raise ValueError(
                    "Restored resolution does not resolve its logged event"
                )
            if action_id != resolution.event.action.action_id:
                raise ValueError("Restored resolution action lineage is inconsistent")
        if len(self._receipt_keys) != len(self._receipts):
            raise ValueError("Restored dispatch receipts contain duplicates")
        module_versions = {
            module_id: self._module_version(module)
            for module_id, module in self._modules.items()
        }
        for receipt in self._receipts:
            event = events_by_id.get(receipt.event_id)
            if event is None:
                raise ValueError("Restored receipt references an unknown event")
            if not event.committed or event.event_id not in self._applied_event_ids:
                raise ValueError("Restored receipt requires an applied committed event")
            if receipt.module_id not in module_versions:
                raise ValueError("Restored receipt references an unknown module")
            if receipt.module_version != module_versions[receipt.module_id]:
                raise ValueError("Restored receipt module version does not match runtime")
        logical_committed_turn = 0
        applied_committed_turn = 0
        saw_unapplied_commit = False
        for event in self._event_log:
            if event.negotiation_id != self.negotiation_id:
                raise ValueError("Restored event belongs to another negotiation")
            if event.rejection_policy is not self.rejection_policy:
                raise ValueError("Restored event uses another rejection policy")
            if event.rejection_policy_version != self.rejection_policy_version:
                raise ValueError("Restored event uses another rejection-policy version")
            if event.committed_turn_index != logical_committed_turn:
                raise ValueError("Restored event committed-turn sequence is inconsistent")
            if event.round_index != logical_committed_turn // len(self.participants):
                raise ValueError("Restored event round index is inconsistent")
            if event.committed:
                logical_committed_turn += 1
            if event.event_id in self._applied_event_ids:
                if not event.committed:
                    raise ValueError("A rejected event cannot be marked applied")
                if saw_unapplied_commit:
                    raise ValueError("Applied events cannot follow an unapplied commit")
                applied_committed_turn += 1
            elif event.committed:
                saw_unapplied_commit = True
        if self._committed_turn_index != applied_committed_turn:
            raise ValueError("Restored committed-turn index does not match applied events")
        expected_pending = {
            event.action.action_id
            for event in self._event_log
            if event.committed
            and event.action.action_id not in self._resolutions
        }
        if set(self._pending_events) != expected_pending:
            raise ValueError("Restored pending events do not match unresolved commits")
        if self.protocol == "alternating":
            if self._pending_batches or self._finalized_batch_ids:
                raise ValueError("Alternating state cannot contain pending batches")
        else:
            pending_batch_actions = {
                action_id
                for action_ids in self._pending_batches.values()
                for action_id in action_ids
            }
            if pending_batch_actions != set(self._pending_events):
                raise ValueError(
                    "Restored pending batches do not match unresolved actions"
                )
            for batch_id, action_ids in self._pending_batches.items():
                if len(action_ids) != len(self.participants):
                    raise ValueError("Restored simultaneous batch is incomplete")
                events = [self._pending_events[action_id] for action_id in action_ids]
                actions = [event.action for event in events]
                if self._batch_id(actions) != batch_id:
                    raise ValueError("Restored simultaneous batch ID is inconsistent")
            known_batch_ids = {
                thaw_json(event.module_context)
                .get("shared_data", {})
                .get("batch_id")
                for event in self._event_log
            }
            if not self._finalized_batch_ids.issubset(known_batch_ids):
                raise ValueError("Restored finalized batch IDs are absent from events")
