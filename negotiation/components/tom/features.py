"""Typed deterministic projection from scenario actions to ToM evidence.

This module never extracts semantics from prose.  It projects only validated
atomic records from ``ObservedAction`` plus explicitly supplied runtime facts.
Raw text is used solely for a source hash already validated against its spans.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
import math
import re
from typing import Annotated, Any, Iterable, Literal, Self
import unicodedata

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    StringConstraints,
    model_validator,
)

from interpretability.scenarios.schema import (
    Claim,
    Commitment,
    Disclosure,
    EvidenceSpan,
    ObservedAction,
    Offer,
    ParseStatus,
)
from negotiation.components.tom.schema import (
    Evidence,
    EvidenceChannel,
    EvidenceVisibility,
)


FEATURE_PROJECTION_SCHEMA_VERSION = "tom-feature-projection/1.0.0"
_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_.:/-]*$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_FORBIDDEN_LABEL_NAMES = frozenset(
    {
        "actual_deception",
        "actual_deception_label",
        "behavioral_misrepresentation",
        "deception_ground_truth",
        "ground_truth_deception",
    }
)


def _stable_identifier(value: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError("value must be a stable identifier")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("identifiers must use canonical NFC Unicode")
    return value


def _stable_category(value: str) -> str:
    if not _CATEGORY_PATTERN.fullmatch(value):
        raise ValueError("value must be a lowercase stable category")
    return value


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be numeric, not boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value must be finite")
    return 0.0 if result == 0.0 else result


def _finite_non_negative(value: Any) -> float:
    result = _finite_number(value)
    if result < 0.0:
        raise ValueError("value must be non-negative")
    return result


def _probability(value: Any) -> float:
    result = _finite_non_negative(value)
    if result > 1.0:
        raise ValueError("confidence must not exceed one")
    return result


StableIdentifier = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=256),
    AfterValidator(_stable_identifier),
]
StableCategory = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=128),
    AfterValidator(_stable_category),
]
FiniteNonNegative = Annotated[float, BeforeValidator(_finite_non_negative)]
Probability = Annotated[float, BeforeValidator(_probability)]
AuxiliaryValue = StrictBool | StrictInt | Annotated[
    float, BeforeValidator(_finite_number)
] | StrictStr


def _canonical_unique(values: tuple[str, ...], *, field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")
    if values != tuple(sorted(values)):
        raise ValueError(
            f"{field_name} must use canonical lexicographic ordering"
        )


def _sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _hash_ids(values: Iterable[str]) -> str:
    canonical = json.dumps(
        sorted(set(values)),
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _sha256_text(canonical)


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class ProjectionStatus(str, Enum):
    """Whether typed action evidence is usable without inventing a zero."""

    AVAILABLE = "available"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"
    UNAVAILABLE = "unavailable"


class ResponseStatus(str, Enum):
    """Explicit runtime knowledge about counterpart response delivery."""

    OBSERVED = "observed"
    ABSENT = "absent"
    UNAVAILABLE = "unavailable"


class StructuredActionCategory(str, Enum):
    """Typed categories supplied externally rather than inferred from prose."""

    ACCEPT = "accept"
    REJECT = "reject"
    REQUEST_EVIDENCE = "request_evidence"


class FeatureProjectionContext(_StrictFrozenModel):
    """Event-linked information explicitly authorized for one observer."""

    schema_version: Literal["tom-feature-projection/1.0.0"] = (
        FEATURE_PROJECTION_SCHEMA_VERSION
    )
    trial_id: StableIdentifier
    observer_id: StableIdentifier
    source_actor_id: StableIdentifier
    source_event_id: StableIdentifier
    source_call_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    visibility: EvidenceVisibility
    visible_to: tuple[StableIdentifier, ...] = ()
    accessible_fact_ids: tuple[StableIdentifier, ...] = ()
    accessible_event_ids: tuple[StableIdentifier, ...]
    structured_action_category: StructuredActionCategory | None = None
    response_status: ResponseStatus
    response_latency_seconds: FiniteNonNegative | None = None

    @model_validator(mode="after")
    def _validate_context(self) -> Self:
        for field_name in (
            "visible_to",
            "accessible_fact_ids",
            "accessible_event_ids",
        ):
            _canonical_unique(getattr(self, field_name), field_name=field_name)
        if self.visibility is EvidenceVisibility.ADJUDICATOR_ONLY:
            raise ValueError("adjudicator-only context cannot enter ToM evidence")
        if self.visibility is EvidenceVisibility.PUBLIC:
            if self.visible_to:
                raise ValueError("public context must not enumerate recipients")
        else:
            if not self.visible_to or self.observer_id not in self.visible_to:
                raise ValueError(
                    "explicit context must be authorized for the observer"
                )
        if self.source_event_id not in self.accessible_event_ids:
            raise ValueError("source event must be explicitly accessible")
        if (
            self.response_status is not ResponseStatus.OBSERVED
            and self.structured_action_category is not None
        ):
            raise ValueError(
                "unobserved responses cannot carry a structured action"
            )
        return self


class AuxiliaryFeatureInput(_StrictFrozenModel):
    """Precomputed weak/model-derived feature kept outside observable evidence."""

    schema_version: Literal["tom-feature-projection/1.0.0"] = (
        FEATURE_PROJECTION_SCHEMA_VERSION
    )
    feature_name: StableCategory
    value: AuxiliaryValue
    confidence: Probability
    channel: Literal[
        EvidenceChannel.LINGUISTIC, EvidenceChannel.MODEL_DERIVED
    ]
    extractor_version: StableIdentifier
    source_span_ids: tuple[StableIdentifier, ...] = ()

    @model_validator(mode="after")
    def _validate_auxiliary(self) -> Self:
        _canonical_unique(self.source_span_ids, field_name="source_span_ids")
        normalized = re.sub(r"[-./:]+", "_", self.feature_name)
        tokens = frozenset(normalized.split("_"))
        if (
            normalized in _FORBIDDEN_LABEL_NAMES
            or {"actual", "deception"}.issubset(tokens)
            or {"ground", "truth", "deception"}.issubset(tokens)
        ):
            raise ValueError("actual-deception labels are not ToM features")
        return self


class DeterministicFeatureProjector(_StrictFrozenModel):
    """Project validated atoms and runtime declarations into Evidence records."""

    version: StableIdentifier = "tom-feature-projector-1"
    amount_term_names: tuple[StableCategory, ...] = (
        "amount",
        "points",
        "price",
        "value",
    )

    @model_validator(mode="after")
    def _validate_projector(self) -> Self:
        _canonical_unique(self.amount_term_names, field_name="amount_term_names")
        return self

    def project(
        self,
        action: ObservedAction,
        context: FeatureProjectionContext,
        auxiliary_features: Iterable[AuxiliaryFeatureInput] = (),
    ) -> tuple[Evidence, ...]:
        """Return observable evidence followed by separated auxiliary records."""
        if not isinstance(action, ObservedAction):
            raise TypeError("action must be a validated ObservedAction")
        if not isinstance(context, FeatureProjectionContext):
            raise TypeError("context must be a FeatureProjectionContext")
        self._validate_linkage(action, context)
        spans = self._spans(action)
        span_by_id = {span.span_id: span for span in spans}
        observable = self._observable_evidence(action, context, spans)

        try:
            auxiliary = tuple(auxiliary_features)
        except TypeError as error:
            raise TypeError(
                "auxiliary_features must be an iterable of AuxiliaryFeatureInput"
            ) from error
        if any(not isinstance(item, AuxiliaryFeatureInput) for item in auxiliary):
            raise TypeError(
                "auxiliary_features must contain AuxiliaryFeatureInput records"
            )
        names = tuple(item.feature_name for item in auxiliary)
        if len(set(names)) != len(names):
            raise ValueError("auxiliary feature names must be unique")
        auxiliary = tuple(sorted(auxiliary, key=lambda item: item.feature_name))
        return (
            observable,
            *(
                self._auxiliary_evidence(
                    item,
                    action=action,
                    context=context,
                    span_by_id=span_by_id,
                    observable_id=observable.evidence_id,
                )
                for item in auxiliary
            ),
        )

    @staticmethod
    def _validate_linkage(
        action: ObservedAction, context: FeatureProjectionContext
    ) -> None:
        if action.trial_id != context.trial_id:
            raise ValueError("action and context trial IDs do not match")
        if action.actor_id != context.source_actor_id:
            raise ValueError("action actor does not match context source actor")
        mismatched_atoms = (
            any(claim.asserted_by != action.actor_id for claim in action.claims)
            or any(offer.actor_id != action.actor_id for offer in action.offers)
            or any(
                commitment.actor_id != action.actor_id
                for commitment in action.commitments
            )
            or any(
                disclosure.actor_id != action.actor_id
                for disclosure in action.disclosures
            )
        )
        if mismatched_atoms:
            raise ValueError("atomic evidence actor does not match source actor")
        atomic_ids = (
            *(claim.claim_id for claim in action.claims),
            *(offer.offer_id for offer in action.offers),
            *(item.commitment_id for item in action.commitments),
            *(item.disclosure_id for item in action.disclosures),
        )
        if len(set(atomic_ids)) != len(atomic_ids):
            raise ValueError("atomic evidence records must be unique")
        referenced_fact_ids = {
            claim.fact_id for claim in action.claims if claim.fact_id is not None
        } | {disclosure.fact_id for disclosure in action.disclosures}
        unauthorized = referenced_fact_ids - set(context.accessible_fact_ids)
        if unauthorized:
            raise ValueError(
                "atomic evidence references inaccessible facts: "
                + ", ".join(sorted(unauthorized))
            )
        if context.response_status is ResponseStatus.ABSENT:
            if (
                action.raw_text
                or action.claims
                or action.offers
                or action.commitments
                or action.disclosures
                or context.structured_action_category is not None
            ):
                raise ValueError("absent response cannot carry action evidence")
        if (
            context.response_status is not ResponseStatus.OBSERVED
            and context.structured_action_category is not None
        ):
            raise ValueError(
                "unobserved responses cannot carry a structured action"
            )
        if (
            action.parse_status is ParseStatus.FAILED
            and context.structured_action_category is not None
        ):
            raise ValueError("failed extraction cannot carry a structured action")

    @staticmethod
    def _spans(action: ObservedAction) -> tuple[EvidenceSpan, ...]:
        spans: dict[str, EvidenceSpan] = {}
        records: tuple[Claim | Offer | Commitment | Disclosure, ...] = (
            *action.claims,
            *action.offers,
            *action.commitments,
            *action.disclosures,
        )
        for record in records:
            for span in record.evidence_spans:
                spans[span.span_id] = span
            if isinstance(record, Offer):
                for term in record.terms:
                    for span in term.evidence_spans:
                        spans[span.span_id] = span
        return tuple(sorted(spans.values(), key=lambda span: span.span_id))

    def _observable_evidence(
        self,
        action: ObservedAction,
        context: FeatureProjectionContext,
        spans: tuple[EvidenceSpan, ...],
    ) -> Evidence:
        features: dict[str, Any] = {
            "action_id": action.action_id,
            "feature_projection_schema_version": (
                FEATURE_PROJECTION_SCHEMA_VERSION
            ),
            "parse_status": action.parse_status.value,
            "parser_name": action.parser_name,
            "parser_version": action.parser_version,
            "scenario_id": action.scenario_id,
        }
        status, category = self._status_and_category(action, context)
        features["atomic_action_category"] = category
        features["projection_status"] = status.value
        features["response_status"] = context.response_status.value
        if context.response_status is ResponseStatus.ABSENT:
            features["response_absent"] = True
        if context.response_latency_seconds is not None:
            features["response_latency_seconds"] = (
                context.response_latency_seconds
            )
            features["response_latency_status"] = "available"

        # Partial atoms remain linked below for auditability, but only a
        # fail-closed AVAILABLE projection may expose them as observations.
        if status is ProjectionStatus.AVAILABLE:
            if action.offers:
                self._add_offer_features(features, action.offers)
            if action.claims:
                self._add_claim_features(features, action.claims)
            if action.commitments:
                features["commitment_present"] = True
                features["commitment_count"] = len(action.commitments)
                features["commitment_action_ids_hash"] = _hash_ids(
                    item.promised_action_id for item in action.commitments
                )
                features["commitment_ids_hash"] = _hash_ids(
                    item.commitment_id for item in action.commitments
                )
            if action.disclosures:
                features["disclosure_present"] = True
                features["disclosure_count"] = len(action.disclosures)
                features["disclosure_ids_hash"] = _hash_ids(
                    item.disclosure_id for item in action.disclosures
                )
                features["disclosed_fact_ids_hash"] = _hash_ids(
                    item.fact_id for item in action.disclosures
                )
            if context.structured_action_category is not None:
                feature_name = {
                    StructuredActionCategory.ACCEPT: "acceptance_present",
                    StructuredActionCategory.REJECT: "rejection_present",
                    StructuredActionCategory.REQUEST_EVIDENCE: (
                        "request_evidence_present"
                    ),
                }[context.structured_action_category]
                features[feature_name] = True

        if spans:
            features["source_span_count"] = len(spans)
            features["source_span_ids_hash"] = _hash_ids(
                span.span_id for span in spans
            )
            source_span = (
                min(span.start for span in spans),
                max(span.end for span in spans),
            )
        else:
            features["source_span_status"] = "unavailable"
            source_span = None

        return Evidence(
            observer_id=context.observer_id,
            source_actor_id=context.source_actor_id,
            source_event_id=context.source_event_id,
            source_call_id=context.source_call_id,
            turn=context.turn,
            features=tuple(sorted(features.items())),
            channel=EvidenceChannel.OBSERVABLE,
            visibility=context.visibility,
            visible_to=context.visible_to,
            reliability=1.0,
            extractor_version=self.version,
            source_text_hash=_sha256_text(action.raw_text),
            source_span=source_span,
            summary=f"typed projection status {status.value}",
        )

    @staticmethod
    def _status_and_category(
        action: ObservedAction,
        context: FeatureProjectionContext,
    ) -> tuple[ProjectionStatus, str]:
        if context.response_status is ResponseStatus.ABSENT:
            return ProjectionStatus.MISSING, "response_absent"
        if context.response_status is ResponseStatus.UNAVAILABLE:
            return ProjectionStatus.UNAVAILABLE, "unavailable"
        if action.parse_status is ParseStatus.FAILED:
            return ProjectionStatus.UNAVAILABLE, "unavailable"
        if action.parse_status is ParseStatus.UNCERTAIN:
            return ProjectionStatus.AMBIGUOUS, "unavailable"
        if context.structured_action_category is not None:
            return ProjectionStatus.AVAILABLE, context.structured_action_category.value
        categories: list[str] = []
        if action.claims:
            categories.append("claim")
        if action.offers:
            categories.append("offer")
        if action.commitments:
            categories.append("commitment")
        if action.disclosures:
            categories.append("disclosure")
        if len(categories) == 1:
            return ProjectionStatus.AVAILABLE, categories[0]
        if len(categories) > 1:
            return ProjectionStatus.AVAILABLE, "multi_atomic"
        return ProjectionStatus.MISSING, "no_relevant_action"

    def _add_offer_features(
        self, features: dict[str, Any], offers: tuple[Offer, ...]
    ) -> None:
        features["offer_present"] = True
        features["offer_count"] = len(offers)
        terms = tuple(term for offer in offers for term in offer.terms)
        features["offer_terms_present"] = True
        features["offer_term_count"] = len(terms)
        features["offer_term_ids_hash"] = _hash_ids(term.term_id for term in terms)
        features["offer_term_names_hash"] = _hash_ids(
            term.name for term in terms
        )
        amounts: list[float] = []
        amount_term_present = False
        for term in terms:
            if term.name not in self.amount_term_names:
                continue
            amount_term_present = True
            if isinstance(term.value, bool) or not isinstance(
                term.value, (int, float)
            ):
                continue
            value = float(term.value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("typed offer amount must be finite and non-negative")
            amounts.append(value)
        if amount_term_present:
            features["offer_amount_present"] = True
        unique_amounts = tuple(sorted(set(amounts)))
        if len(unique_amounts) == 1:
            features["offer_amount"] = unique_amounts[0]
            features["offer_amount_status"] = "available"
        elif not unique_amounts:
            features["offer_amount_status"] = "unavailable"
        else:
            features["offer_amount_status"] = "ambiguous"

    @staticmethod
    def _add_claim_features(
        features: dict[str, Any], claims: tuple[Claim, ...]
    ) -> None:
        features["claim_present"] = True
        features["claim_count"] = len(claims)
        features["claim_predicates_hash"] = _hash_ids(
            claim.predicate for claim in claims
        )
        fact_ids = tuple(
            claim.fact_id for claim in claims if claim.fact_id is not None
        )
        if fact_ids:
            features["claim_fact_id_count"] = len(set(fact_ids))
            features["claim_fact_ids_hash"] = _hash_ids(fact_ids)
        features["claim_positive_polarity_count"] = sum(
            claim.polarity for claim in claims
        )
        features["claim_negative_polarity_count"] = sum(
            not claim.polarity for claim in claims
        )
        features["claim_semantics_source"] = "atomic_evidence"

    def _auxiliary_evidence(
        self,
        auxiliary: AuxiliaryFeatureInput,
        *,
        action: ObservedAction,
        context: FeatureProjectionContext,
        span_by_id: dict[str, EvidenceSpan],
        observable_id: str,
    ) -> Evidence:
        unknown_spans = set(auxiliary.source_span_ids) - set(span_by_id)
        if unknown_spans:
            raise ValueError(
                "auxiliary feature names unknown source spans: "
                + ", ".join(sorted(unknown_spans))
            )
        selected_spans = tuple(
            span_by_id[span_id] for span_id in auxiliary.source_span_ids
        )
        source_span = (
            (
                min(span.start for span in selected_spans),
                max(span.end for span in selected_spans),
            )
            if selected_spans
            else None
        )
        feature_prefix = f"aux.{auxiliary.feature_name}"
        features: dict[str, Any] = {
            f"{feature_prefix}.confidence": auxiliary.confidence,
            f"{feature_prefix}.status": "available",
            f"{feature_prefix}.value": auxiliary.value,
        }
        if auxiliary.source_span_ids:
            features[f"{feature_prefix}.source_span_ids_hash"] = _hash_ids(
                auxiliary.source_span_ids
            )
        return Evidence(
            observer_id=context.observer_id,
            source_actor_id=context.source_actor_id,
            source_event_id=context.source_event_id,
            source_call_id=context.source_call_id,
            turn=context.turn,
            features=tuple(sorted(features.items())),
            channel=auxiliary.channel,
            visibility=context.visibility,
            visible_to=context.visible_to,
            reliability=auxiliary.confidence,
            extractor_version=auxiliary.extractor_version,
            source_text_hash=_sha256_text(action.raw_text),
            source_span=source_span,
            parent_evidence_ids=(observable_id,),
            summary=f"separate auxiliary feature {auxiliary.feature_name}",
        )


__all__ = [
    "AuxiliaryFeatureInput",
    "DeterministicFeatureProjector",
    "FEATURE_PROJECTION_SCHEMA_VERSION",
    "FeatureProjectionContext",
    "ProjectionStatus",
    "ResponseStatus",
    "StructuredActionCategory",
]
