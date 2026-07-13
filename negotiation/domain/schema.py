"""Versioned immutable values used at the negotiation domain boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA_VERSION = "negotiation-domain/3"


def freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a recursively immutable copy of a JSON-compatible mapping."""
    frozen = _freeze_json(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - defensive typing
        raise TypeError("Expected a mapping")
    return frozen


def thaw_json(value: Any) -> Any:
    """Return a JSON-serializable copy of an immutable domain value."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Domain JSON floats must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    """Create a deterministic content identifier from canonical JSON."""
    encoded = json.dumps(
        thaw_json(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()}"


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Domain mapping keys must be strings")
            copied[key] = _freeze_json(item)
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Domain JSON floats must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Domain values must be JSON-compatible, got {type(value).__name__}")


def _verify_serialized_id(data: Mapping[str, Any], field_name: str, actual: str) -> None:
    if field_name not in data:
        raise ValueError(f"{field_name} is required for persisted domain records")
    serialized = data[field_name]
    if not isinstance(serialized, str) or not serialized:
        raise ValueError(f"{field_name} must be a non-empty string")
    if serialized != actual:
        raise ValueError(f"{field_name} does not match canonical record content")


def _validate_event_position(
    event_boundary: Any,
    event_sequence: Any,
    *,
    record_name: str,
) -> None:
    for field_name, value in (
        ("event_boundary", event_boundary),
        ("event_sequence", event_sequence),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{record_name} {field_name} must be an integer")
        if value < 0:
            raise ValueError(
                f"{record_name} boundary and sequence must be nonnegative"
            )


@dataclass(frozen=True)
class EvidenceSpan:
    """An exact source-text span supporting one parsed domain field."""

    kind: str
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("Evidence kind cannot be empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Evidence offsets must define a non-empty span")
        if len(self.text) != self.end - self.start:
            raise ValueError("Evidence text length must match its offsets")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceSpan:
        return cls(
            kind=str(data["kind"]),
            start=int(data["start"]),
            end=int(data["end"]),
            text=str(data["text"]),
        )


class AmountKind(str, Enum):
    """Semantic role of an amount mentioned in an observed action."""

    ACTOR_OFFER = "actor_offer"
    COUNTERPART_OFFER = "counterpart_offer"
    COUNTERPART_VALUE_ASSERTION = "counterpart_value_assertion"
    ASSERTED_PRIVATE_VALUE = "asserted_private_value"
    ASSERTED_MARKET_VALUE = "asserted_market_value"
    UNCLASSIFIED = "unclassified"


class ActionKind(str, Enum):
    """Structured actions accepted by the negotiation adjudicator."""

    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"
    WALK_AWAY = "walk_away"
    DISCLOSE = "disclose"


class OutcomeStatus(str, Enum):
    """Terminal negotiation outcome categories."""

    AGREEMENT = "agreement"
    WALK_AWAY = "walk_away"
    FAILED = "failed"


class DisclosureResult(str, Enum):
    """Tri-state result of evaluating one disclosure obligation."""

    SATISFIED = "satisfied"
    OMITTED = "omitted"
    UNKNOWN = "unknown"


class DisclosureObligationState(str, Enum):
    """Applicability of an obligation at a particular decision boundary."""

    ACTIVE = "active"
    EXPIRED = "expired"
    NOT_YET_ACTIVE = "not_yet_active"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class AmountMention:
    """A normalized monetary amount linked to its exact textual evidence."""

    kind: AmountKind
    amount: int
    evidence: EvidenceSpan
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("Amount cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "amount": self.amount,
            "currency": self.currency,
            "evidence": self.evidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AmountMention:
        return cls(
            kind=AmountKind(data["kind"]),
            amount=int(data["amount"]),
            currency=str(data.get("currency", "USD")),
            evidence=EvidenceSpan.from_dict(data["evidence"]),
        )


@dataclass(frozen=True)
class ObservedAction:
    """Raw actor text plus typed, evidence-bearing parse results."""

    actor_id: str
    raw_text: str
    amounts: tuple[AmountMention, ...] = ()
    parser_version: str = "amount-parser/1"
    action_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.actor_id:
            raise ValueError("actor_id cannot be empty")
        if not isinstance(self.raw_text, str):
            raise TypeError("raw_text must be a string")
        object.__setattr__(self, "amounts", tuple(self.amounts))
        for mention in self.amounts:
            evidence = mention.evidence
            if evidence.end > len(self.raw_text):
                raise ValueError("Evidence span lies outside raw_text")
            if self.raw_text[evidence.start:evidence.end] != evidence.text:
                raise ValueError("Evidence text does not match raw_text")
        object.__setattr__(self, "action_id", stable_id("act", self._content_dict()))

    @property
    def actor_offers(self) -> tuple[AmountMention, ...]:
        return tuple(item for item in self.amounts if item.kind is AmountKind.ACTOR_OFFER)

    @property
    def counterpart_offers(self) -> tuple[AmountMention, ...]:
        return tuple(
            item for item in self.amounts if item.kind is AmountKind.COUNTERPART_OFFER
        )

    @property
    def counterpart_value_assertions(self) -> tuple[AmountMention, ...]:
        return tuple(
            item
            for item in self.amounts
            if item.kind is AmountKind.COUNTERPART_VALUE_ASSERTION
        )

    @property
    def factual_value_assertions(self) -> tuple[AmountMention, ...]:
        kinds = {AmountKind.ASSERTED_PRIVATE_VALUE, AmountKind.ASSERTED_MARKET_VALUE}
        return tuple(item for item in self.amounts if item.kind in kinds)

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "actor_id": self.actor_id,
            "raw_text": self.raw_text,
            "amounts": [item.to_dict() for item in self.amounts],
            "parser_version": self.parser_version,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "action_id": self.action_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObservedAction:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported negotiation-domain schema version")
        action = cls(
            actor_id=str(data["actor_id"]),
            raw_text=str(data["raw_text"]),
            amounts=tuple(AmountMention.from_dict(item) for item in data.get("amounts", ())),
            parser_version=str(data.get("parser_version", "amount-parser/1")),
        )
        _verify_serialized_id(data, "action_id", action.action_id)
        return action


@dataclass(frozen=True)
class CommitmentEvidence:
    """A prior commitment to a normalized future action."""

    actor_id: str
    promised_action: str
    source_event_id: str
    event_boundary: int
    event_sequence: int
    evidence: tuple[EvidenceSpan, ...] = ()
    commitment_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not isinstance(self.source_event_id, str):
            raise TypeError("Commitment source_event_id must be a string")
        if not self.actor_id or not self.promised_action or not self.source_event_id:
            raise ValueError(
                "Commitment actor, action, and source event cannot be empty"
            )
        _validate_event_position(
            self.event_boundary,
            self.event_sequence,
            record_name="Commitment",
        )
        object.__setattr__(self, "commitment_id", stable_id("commit", self._content_dict()))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "promised_action": self.promised_action,
            "source_event_id": self.source_event_id,
            "event_boundary": self.event_boundary,
            "event_sequence": self.event_sequence,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "commitment_id": self.commitment_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CommitmentEvidence:
        record = cls(
            actor_id=str(data["actor_id"]),
            promised_action=str(data["promised_action"]),
            source_event_id=data["source_event_id"],
            event_boundary=data["event_boundary"],
            event_sequence=data["event_sequence"],
            evidence=tuple(EvidenceSpan.from_dict(item) for item in data.get("evidence", ())),
        )
        _verify_serialized_id(data, "commitment_id", record.commitment_id)
        return record


@dataclass(frozen=True)
class PlanEvidence:
    """Evidence of an actor's contemporaneous intended action."""

    actor_id: str
    planned_action: str
    source_event_id: str
    event_boundary: int
    event_sequence: int
    evidence: tuple[EvidenceSpan, ...] = ()
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not isinstance(self.source_event_id, str):
            raise TypeError("Plan source_event_id must be a string")
        if not self.actor_id or not self.planned_action or not self.source_event_id:
            raise ValueError("Plan actor, action, and source event cannot be empty")
        _validate_event_position(
            self.event_boundary,
            self.event_sequence,
            record_name="Plan",
        )
        object.__setattr__(self, "plan_id", stable_id("plan", self._content_dict()))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "planned_action": self.planned_action,
            "source_event_id": self.source_event_id,
            "event_boundary": self.event_boundary,
            "event_sequence": self.event_sequence,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "plan_id": self.plan_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlanEvidence:
        record = cls(
            actor_id=str(data["actor_id"]),
            planned_action=str(data["planned_action"]),
            source_event_id=data["source_event_id"],
            event_boundary=data["event_boundary"],
            event_sequence=data["event_sequence"],
            evidence=tuple(EvidenceSpan.from_dict(item) for item in data.get("evidence", ())),
        )
        _verify_serialized_id(data, "plan_id", record.plan_id)
        return record


@dataclass(frozen=True)
class ExecutedActionEvidence:
    """Evidence of the later executable action at its decision boundary."""

    actor_id: str
    executed_action: str
    source_event_id: str
    event_boundary: int
    event_sequence: int
    evidence: tuple[EvidenceSpan, ...] = ()
    execution_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if not isinstance(self.source_event_id, str):
            raise TypeError("Executed-action source_event_id must be a string")
        if not self.actor_id or not self.executed_action or not self.source_event_id:
            raise ValueError(
                "Executed-action actor, action, and source event cannot be empty"
            )
        _validate_event_position(
            self.event_boundary,
            self.event_sequence,
            record_name="Executed-action",
        )
        object.__setattr__(self, "execution_id", stable_id("execution", self._content_dict()))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "executed_action": self.executed_action,
            "source_event_id": self.source_event_id,
            "event_boundary": self.event_boundary,
            "event_sequence": self.event_sequence,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "execution_id": self.execution_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutedActionEvidence:
        record = cls(
            actor_id=str(data["actor_id"]),
            executed_action=str(data["executed_action"]),
            source_event_id=data["source_event_id"],
            event_boundary=data["event_boundary"],
            event_sequence=data["event_sequence"],
            evidence=tuple(EvidenceSpan.from_dict(item) for item in data.get("evidence", ())),
        )
        _verify_serialized_id(data, "execution_id", record.execution_id)
        return record


@dataclass(frozen=True)
class Fact:
    """A typed fact with explicit subject and visibility."""

    subject_id: str
    predicate: str
    value: Any
    visible_to: tuple[str, ...]
    fact_version: str = SCHEMA_VERSION
    fact_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.subject_id or not self.predicate:
            raise ValueError("Fact subject and predicate cannot be empty")
        visible_to = tuple(self.visible_to)
        if len(visible_to) != len(set(visible_to)):
            raise ValueError("Fact visibility cannot contain duplicate roles")
        object.__setattr__(self, "value", _freeze_json(self.value))
        object.__setattr__(self, "visible_to", visible_to)
        object.__setattr__(self, "fact_id", stable_id("fact", self._content_dict()))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "fact_version": self.fact_version,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "value": thaw_json(self.value),
            "visible_to": list(self.visible_to),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "fact_id": self.fact_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Fact:
        record = cls(
            fact_version=str(data.get("fact_version", SCHEMA_VERSION)),
            subject_id=str(data["subject_id"]),
            predicate=str(data["predicate"]),
            value=data.get("value"),
            visible_to=tuple(str(item) for item in data.get("visible_to", ())),
        )
        _verify_serialized_id(data, "fact_id", record.fact_id)
        return record


@dataclass(frozen=True)
class Offer:
    """A content-addressed offer from one participant to another."""

    actor_id: str
    recipient_id: str
    terms: Mapping[str, Any]
    offer_type: str = "initial"
    offer_version: str = SCHEMA_VERSION
    offer_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.actor_id or not self.recipient_id:
            raise ValueError("Offer participants cannot be empty")
        if self.actor_id == self.recipient_id:
            raise ValueError("Offer actor and recipient must differ")
        if not self.offer_type:
            raise ValueError("offer_type cannot be empty")
        object.__setattr__(self, "terms", freeze_mapping(self.terms))
        object.__setattr__(self, "offer_id", stable_id("offer", self._content_dict()))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "offer_version": self.offer_version,
            "actor_id": self.actor_id,
            "recipient_id": self.recipient_id,
            "offer_type": self.offer_type,
            "terms": thaw_json(self.terms),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "offer_id": self.offer_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Offer:
        record = cls(
            offer_version=str(data.get("offer_version", SCHEMA_VERSION)),
            actor_id=str(data["actor_id"]),
            recipient_id=str(data["recipient_id"]),
            offer_type=str(data.get("offer_type", "initial")),
            terms=data.get("terms", {}),
        )
        _verify_serialized_id(data, "offer_id", record.offer_id)
        return record


@dataclass(frozen=True)
class Disclosure:
    """An explicit disclosure of facts to named recipients."""

    actor_id: str
    recipient_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]
    note: str = ""
    disclosure_version: str = SCHEMA_VERSION
    disclosure_id: str = field(init=False)

    def __post_init__(self) -> None:
        recipients = tuple(self.recipient_ids)
        facts = tuple(self.fact_ids)
        if not self.actor_id or not recipients or not facts:
            raise ValueError("Disclosure actor, recipients, and facts are required")
        if self.actor_id in recipients or len(recipients) != len(set(recipients)):
            raise ValueError("Disclosure recipients must be unique and exclude the actor")
        object.__setattr__(self, "recipient_ids", recipients)
        object.__setattr__(self, "fact_ids", facts)
        object.__setattr__(
            self, "disclosure_id", stable_id("disclosure", self._content_dict())
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "disclosure_version": self.disclosure_version,
            "actor_id": self.actor_id,
            "recipient_ids": list(self.recipient_ids),
            "fact_ids": list(self.fact_ids),
            "note": self.note,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "disclosure_id": self.disclosure_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Disclosure:
        record = cls(
            disclosure_version=str(data.get("disclosure_version", SCHEMA_VERSION)),
            actor_id=str(data["actor_id"]),
            recipient_ids=tuple(str(item) for item in data["recipient_ids"]),
            fact_ids=tuple(str(item) for item in data["fact_ids"]),
            note=str(data.get("note", "")),
        )
        _verify_serialized_id(data, "disclosure_id", record.disclosure_id)
        return record


@dataclass(frozen=True)
class DisclosureObligation:
    """A duty to disclose one versioned fact before a decision boundary."""

    actor_id: str
    recipient_id: str
    fact_id: str
    fact_version: str
    creation_event_id: str
    created_at_boundary: int
    expires_at_boundary: int | None = None
    obligation_version: str = "disclosure-obligation/1"
    obligation_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not all(
            (
                self.actor_id,
                self.recipient_id,
                self.fact_id,
                self.fact_version,
                self.creation_event_id,
            )
        ):
            raise ValueError("Disclosure obligation fields cannot be empty")
        if self.actor_id == self.recipient_id:
            raise ValueError("Disclosure actor and recipient must differ")
        if self.created_at_boundary < 0:
            raise ValueError("Disclosure creation boundary cannot be negative")
        if (
            self.expires_at_boundary is not None
            and self.expires_at_boundary < self.created_at_boundary
        ):
            raise ValueError("Disclosure expiry cannot precede creation")
        object.__setattr__(
            self,
            "obligation_id",
            stable_id("disclosure_obligation", self._content_dict()),
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "obligation_version": self.obligation_version,
            "actor_id": self.actor_id,
            "recipient_id": self.recipient_id,
            "fact_id": self.fact_id,
            "fact_version": self.fact_version,
            "creation_event_id": self.creation_event_id,
            "created_at_boundary": self.created_at_boundary,
            "expires_at_boundary": self.expires_at_boundary,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "obligation_id": self.obligation_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DisclosureObligation:
        record = cls(
            actor_id=str(data["actor_id"]),
            recipient_id=str(data["recipient_id"]),
            fact_id=str(data["fact_id"]),
            fact_version=str(data["fact_version"]),
            creation_event_id=str(data["creation_event_id"]),
            created_at_boundary=int(data["created_at_boundary"]),
            expires_at_boundary=(
                int(data["expires_at_boundary"])
                if data.get("expires_at_boundary") is not None
                else None
            ),
            obligation_version=str(
                data.get("obligation_version", "disclosure-obligation/1")
            ),
        )
        _verify_serialized_id(data, "obligation_id", record.obligation_id)
        return record


@dataclass(frozen=True)
class CommittedDisclosure:
    """A committed event disclosing one versioned fact to named recipients."""

    actor_id: str
    recipient_ids: tuple[str, ...]
    fact_id: str
    fact_version: str
    committed_event_id: str
    committed_at_boundary: int
    disclosure_version: str = "committed-disclosure/1"
    disclosure_id: str = field(init=False)

    def __post_init__(self) -> None:
        recipients = tuple(self.recipient_ids)
        if not all(
            (
                self.actor_id,
                self.fact_id,
                self.fact_version,
                self.committed_event_id,
            )
        ) or not recipients:
            raise ValueError("Committed disclosure fields cannot be empty")
        if self.actor_id in recipients or len(recipients) != len(set(recipients)):
            raise ValueError("Disclosure recipients must be unique and exclude the actor")
        if self.committed_at_boundary < 0:
            raise ValueError("Disclosure boundary cannot be negative")
        object.__setattr__(self, "recipient_ids", recipients)
        object.__setattr__(
            self,
            "disclosure_id",
            stable_id("committed_disclosure", self._content_dict()),
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "disclosure_version": self.disclosure_version,
            "actor_id": self.actor_id,
            "recipient_ids": list(self.recipient_ids),
            "fact_id": self.fact_id,
            "fact_version": self.fact_version,
            "committed_event_id": self.committed_event_id,
            "committed_at_boundary": self.committed_at_boundary,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "disclosure_id": self.disclosure_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CommittedDisclosure:
        record = cls(
            actor_id=str(data["actor_id"]),
            recipient_ids=tuple(str(item) for item in data["recipient_ids"]),
            fact_id=str(data["fact_id"]),
            fact_version=str(data["fact_version"]),
            committed_event_id=str(data["committed_event_id"]),
            committed_at_boundary=int(data["committed_at_boundary"]),
            disclosure_version=str(
                data.get("disclosure_version", "committed-disclosure/1")
            ),
        )
        _verify_serialized_id(data, "disclosure_id", record.disclosure_id)
        return record


@dataclass(frozen=True)
class DisclosureDecision:
    """A decision boundary at which disclosure compliance is evaluated."""

    actor_id: str
    recipient_id: str
    decision_event_id: str
    decision_at_boundary: int
    history_complete: bool
    decision_version: str = "disclosure-decision/1"
    decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.actor_id or not self.recipient_id or not self.decision_event_id:
            raise ValueError("Disclosure decision fields cannot be empty")
        if self.actor_id == self.recipient_id:
            raise ValueError("Disclosure decision parties must differ")
        if self.decision_at_boundary < 0:
            raise ValueError("Disclosure decision boundary cannot be negative")
        if not isinstance(self.history_complete, bool):
            raise TypeError("history_complete must be a bool")
        object.__setattr__(
            self,
            "decision_id",
            stable_id("disclosure_decision", self._content_dict()),
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "decision_version": self.decision_version,
            "actor_id": self.actor_id,
            "recipient_id": self.recipient_id,
            "decision_event_id": self.decision_event_id,
            "decision_at_boundary": self.decision_at_boundary,
            "history_complete": self.history_complete,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "decision_id": self.decision_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DisclosureDecision:
        record = cls(
            actor_id=str(data["actor_id"]),
            recipient_id=str(data["recipient_id"]),
            decision_event_id=str(data["decision_event_id"]),
            decision_at_boundary=int(data["decision_at_boundary"]),
            history_complete=data["history_complete"],
            decision_version=str(
                data.get("decision_version", "disclosure-decision/1")
            ),
        )
        _verify_serialized_id(data, "decision_id", record.decision_id)
        return record


@dataclass(frozen=True)
class DisclosureEvaluation:
    """Tamper-evident tri-state result at one disclosure decision boundary."""

    actor_id: str
    recipient_id: str
    fact_id: str | None
    fact_version: str | None
    obligation_id: str | None
    decision_id: str
    result: DisclosureResult
    obligation_state: DisclosureObligationState
    satisfaction_event_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    evaluated_at_boundary: int = 0
    evaluation_version: str = "disclosure-evaluation/1"
    evaluation_id: str = field(init=False)

    def __post_init__(self) -> None:
        satisfaction = tuple(self.satisfaction_event_ids)
        reasons = tuple(self.reason_codes)
        if not self.actor_id or not self.recipient_id or not self.decision_id:
            raise ValueError("Disclosure evaluation parties and decision are required")
        if self.actor_id == self.recipient_id:
            raise ValueError("Disclosure evaluation parties must differ")
        if (self.fact_id is None) != (self.fact_version is None):
            raise ValueError("Disclosure fact ID and version must be present together")
        if self.evaluated_at_boundary < 0:
            raise ValueError("Disclosure evaluation boundary cannot be negative")
        if len(satisfaction) != len(set(satisfaction)):
            raise ValueError("Satisfaction evidence cannot contain duplicate events")
        if self.result is DisclosureResult.SATISFIED and not satisfaction:
            raise ValueError("Satisfied disclosure evaluations require evidence")
        if self.result is not DisclosureResult.SATISFIED and satisfaction:
            raise ValueError("Only satisfied evaluations can carry satisfaction evidence")
        if (
            self.result in {DisclosureResult.SATISFIED, DisclosureResult.OMITTED}
            and self.obligation_state is not DisclosureObligationState.ACTIVE
        ):
            raise ValueError("Known disclosure results require an active obligation")
        if (
            self.result in {DisclosureResult.SATISFIED, DisclosureResult.OMITTED}
            and (
                self.obligation_id is None
                or self.fact_id is None
                or self.fact_version is None
            )
        ):
            raise ValueError(
                "Known disclosure results require obligation and fact identity"
            )
        has_complete_identity = (
            self.obligation_id is not None
            and self.fact_id is not None
            and self.fact_version is not None
        )
        if (
            self.obligation_state is DisclosureObligationState.NOT_APPLICABLE
            and (
                self.obligation_id is not None
                or self.fact_id is not None
                or self.fact_version is not None
            )
        ):
            raise ValueError(
                "Not-applicable evaluations cannot reference obligation or fact identity"
            )
        if (
            self.obligation_state is not DisclosureObligationState.NOT_APPLICABLE
            and not has_complete_identity
        ):
            raise ValueError(
                "Applicable disclosure states require obligation and fact identity"
            )
        object.__setattr__(self, "satisfaction_event_ids", satisfaction)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(
            self,
            "evaluation_id",
            stable_id("disclosure_evaluation", self._content_dict()),
        )

    @property
    def is_deceptive_omission(self) -> bool:
        """Return true only for an evidenced active unmet obligation."""
        return self.result is DisclosureResult.OMITTED

    def _content_dict(self) -> dict[str, Any]:
        return {
            "evaluation_version": self.evaluation_version,
            "actor_id": self.actor_id,
            "recipient_id": self.recipient_id,
            "fact_id": self.fact_id,
            "fact_version": self.fact_version,
            "obligation_id": self.obligation_id,
            "decision_id": self.decision_id,
            "result": self.result.value,
            "obligation_state": self.obligation_state.value,
            "satisfaction_event_ids": list(self.satisfaction_event_ids),
            "reason_codes": list(self.reason_codes),
            "evaluated_at_boundary": self.evaluated_at_boundary,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "evaluation_id": self.evaluation_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DisclosureEvaluation:
        record = cls(
            actor_id=str(data["actor_id"]),
            recipient_id=str(data["recipient_id"]),
            fact_id=(str(data["fact_id"]) if data.get("fact_id") is not None else None),
            fact_version=(
                str(data["fact_version"])
                if data.get("fact_version") is not None
                else None
            ),
            obligation_id=(
                str(data["obligation_id"])
                if data.get("obligation_id") is not None
                else None
            ),
            decision_id=str(data["decision_id"]),
            result=DisclosureResult(data["result"]),
            obligation_state=DisclosureObligationState(data["obligation_state"]),
            satisfaction_event_ids=tuple(
                str(item) for item in data.get("satisfaction_event_ids", ())
            ),
            reason_codes=tuple(str(item) for item in data.get("reason_codes", ())),
            evaluated_at_boundary=int(data.get("evaluated_at_boundary", 0)),
            evaluation_version=str(
                data.get("evaluation_version", "disclosure-evaluation/1")
            ),
        )
        _verify_serialized_id(data, "evaluation_id", record.evaluation_id)
        return record


@dataclass(frozen=True)
class Agreement:
    """An agreement grounded in a referenced accepted offer."""

    negotiation_id: str
    offer_id: str
    parties: tuple[str, ...]
    terms: Mapping[str, Any]
    agreement_version: str = SCHEMA_VERSION
    agreement_id: str = field(init=False)

    def __post_init__(self) -> None:
        parties = tuple(self.parties)
        if not self.negotiation_id or not self.offer_id:
            raise ValueError("Agreement negotiation and offer IDs are required")
        if len(parties) < 2 or len(parties) != len(set(parties)):
            raise ValueError("Agreement requires at least two unique parties")
        object.__setattr__(self, "parties", parties)
        object.__setattr__(self, "terms", freeze_mapping(self.terms))
        object.__setattr__(self, "agreement_id", stable_id("agreement", self._content_dict()))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "agreement_version": self.agreement_version,
            "negotiation_id": self.negotiation_id,
            "offer_id": self.offer_id,
            "parties": list(self.parties),
            "terms": thaw_json(self.terms),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "agreement_id": self.agreement_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Agreement:
        record = cls(
            agreement_version=str(data.get("agreement_version", SCHEMA_VERSION)),
            negotiation_id=str(data["negotiation_id"]),
            offer_id=str(data["offer_id"]),
            parties=tuple(str(item) for item in data["parties"]),
            terms=data.get("terms", {}),
        )
        _verify_serialized_id(data, "agreement_id", record.agreement_id)
        return record


@dataclass(frozen=True)
class Outcome:
    """A terminal outcome with an optional accepted-agreement reference."""

    negotiation_id: str
    status: OutcomeStatus
    reason: str
    agreement_id: str | None = None
    outcome_version: str = SCHEMA_VERSION
    outcome_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.negotiation_id or not self.reason:
            raise ValueError("Outcome negotiation ID and reason are required")
        if self.status is OutcomeStatus.AGREEMENT and not self.agreement_id:
            raise ValueError("Agreement outcomes require agreement_id")
        if self.status is not OutcomeStatus.AGREEMENT and self.agreement_id:
            raise ValueError("Only agreement outcomes can reference an agreement")
        object.__setattr__(self, "outcome_id", stable_id("outcome", self._content_dict()))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "outcome_version": self.outcome_version,
            "negotiation_id": self.negotiation_id,
            "status": self.status.value,
            "reason": self.reason,
            "agreement_id": self.agreement_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "outcome_id": self.outcome_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Outcome:
        record = cls(
            outcome_version=str(data.get("outcome_version", SCHEMA_VERSION)),
            negotiation_id=str(data["negotiation_id"]),
            status=OutcomeStatus(data["status"]),
            reason=str(data["reason"]),
            agreement_id=(str(data["agreement_id"]) if data.get("agreement_id") else None),
        )
        _verify_serialized_id(data, "outcome_id", record.outcome_id)
        return record


@dataclass(frozen=True)
class NegotiationAction:
    """An explicit structured submission to the GM adjudication boundary."""

    action_ref: str
    actor_id: str
    kind: ActionKind
    offer: Offer | None = None
    referenced_offer_id: str | None = None
    reason: str | None = None
    raw_text: str = ""
    action_version: str = SCHEMA_VERSION
    action_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.action_ref or not self.actor_id:
            raise ValueError("Action reference and actor are required")
        if self.kind is ActionKind.OFFER:
            if self.offer is None or self.offer.actor_id != self.actor_id:
                raise ValueError("Offer actions require an offer made by the actor")
            if self.referenced_offer_id is not None:
                raise ValueError("Offer actions cannot reference another offer")
        elif self.kind in {ActionKind.ACCEPT, ActionKind.REJECT}:
            if self.offer is not None or not self.referenced_offer_id:
                raise ValueError("Accept/reject actions require one referenced offer ID")
        elif self.offer is not None or self.referenced_offer_id is not None:
            raise ValueError("This action kind cannot carry an offer reference")
        object.__setattr__(self, "action_id", stable_id("action", self._content_dict()))

    def _content_dict(self) -> dict[str, Any]:
        return {
            "action_version": self.action_version,
            "action_ref": self.action_ref,
            "actor_id": self.actor_id,
            "kind": self.kind.value,
            "offer": self.offer.to_dict() if self.offer else None,
            "referenced_offer_id": self.referenced_offer_id,
            "reason": self.reason,
            "raw_text": self.raw_text,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_dict(), "action_id": self.action_id}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NegotiationAction:
        offer = data.get("offer")
        record = cls(
            action_version=str(data.get("action_version", SCHEMA_VERSION)),
            action_ref=str(data["action_ref"]),
            actor_id=str(data["actor_id"]),
            kind=ActionKind(data["kind"]),
            offer=Offer.from_dict(offer) if offer else None,
            referenced_offer_id=(
                str(data["referenced_offer_id"])
                if data.get("referenced_offer_id")
                else None
            ),
            reason=(str(data["reason"]) if data.get("reason") is not None else None),
            raw_text=str(data.get("raw_text", "")),
        )
        _verify_serialized_id(data, "action_id", record.action_id)
        return record
