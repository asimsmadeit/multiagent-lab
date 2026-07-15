"""Permanent contracts for deterministic Theory of Mind feature projection."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Iterable

import pytest
from pydantic import BaseModel, ValidationError

from interpretability.scenarios.schema import (
    Claim,
    Commitment,
    Disclosure,
    EvidenceSpan,
    ObservedAction,
    Offer,
    OfferTerm,
    ParseStatus,
)
from negotiation.components.tom.features import (
    FEATURE_PROJECTION_SCHEMA_VERSION,
    AuxiliaryFeatureInput,
    DeterministicFeatureProjector,
    FeatureProjectionContext,
    ProjectionStatus,
    ResponseStatus,
    StructuredActionCategory,
)
from negotiation.components.tom.schema import (
    Evidence,
    EvidenceChannel,
    EvidenceVisibility,
)


_RAW_ATOMIC = (
    'I offer $70, claim "the brakes are not defective", '
    "promise cooperation, and disclose defect."
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


def _span(raw_text: str, text: str, kind: str) -> EvidenceSpan:
    start = raw_text.index(text)
    return EvidenceSpan(
        kind=kind,
        start=start,
        end=start + len(text),
        text=text,
    )


def _atomic_records(
    raw_text: str = _RAW_ATOMIC,
    *,
    actor_id: str = "actor",
) -> tuple[
    tuple[Claim, ...],
    tuple[Offer, ...],
    tuple[Commitment, ...],
    tuple[Disclosure, ...],
]:
    offer_span = _span(raw_text, "offer $70", "offer")
    amount_span = _span(raw_text, "$70", "offer_amount")
    claim_span = _span(
        raw_text,
        "the brakes are not defective",
        "negated_claim",
    )
    commitment_span = _span(
        raw_text,
        "promise cooperation",
        "commitment",
    )
    disclosure_span = _span(raw_text, "disclose defect", "disclosure")
    claim = Claim(
        subject_id="vehicle",
        predicate="has_defect",
        value=False,
        asserted_by=actor_id,
        polarity=False,
        fact_id="defect",
        evidence_spans=(claim_span,),
    )
    offer = Offer(
        actor_id=actor_id,
        recipient_ids=("counterpart",),
        terms=(
            OfferTerm(
                name="price",
                value=70,
                evidence_spans=(amount_span,),
            ),
        ),
        evidence_spans=(offer_span,),
    )
    commitment = Commitment(
        actor_id=actor_id,
        promised_action_id="cooperate",
        beneficiary_role_ids=("counterpart",),
        effective_round=3,
        evidence_spans=(commitment_span,),
    )
    disclosure = Disclosure(
        actor_id=actor_id,
        recipient_ids=("counterpart",),
        fact_id="defect",
        fact_version="1.0.0",
        evidence_spans=(disclosure_span,),
    )
    return (claim,), (offer,), (commitment,), (disclosure,)


def _action(
    *,
    raw_text: str = _RAW_ATOMIC,
    parse_status: ParseStatus = ParseStatus.COMPLETE,
    claims: tuple[Claim, ...] | None = None,
    offers: tuple[Offer, ...] | None = None,
    commitments: tuple[Commitment, ...] | None = None,
    disclosures: tuple[Disclosure, ...] | None = None,
    actor_id: str = "actor",
    trial_id: str = "trial-17",
) -> ObservedAction:
    if any(
        records is None
        for records in (claims, offers, commitments, disclosures)
    ):
        defaults = _atomic_records(raw_text, actor_id=actor_id)
        claims = defaults[0] if claims is None else claims
        offers = defaults[1] if offers is None else offers
        commitments = defaults[2] if commitments is None else commitments
        disclosures = defaults[3] if disclosures is None else disclosures
    return ObservedAction(
        scenario_id="vehicle_sale",
        spec_version="1.0.0",
        trial_id=trial_id,
        actor_id=actor_id,
        raw_text=raw_text,
        parse_status=parse_status,
        parser_name="typed_parser",
        parser_version="1.0.0",
        claims=claims,
        offers=offers,
        commitments=commitments,
        disclosures=disclosures,
        parse_error=(
            "partial extraction failed"
            if parse_status is ParseStatus.FAILED
            else None
        ),
    )


def _empty_action(
    raw_text: str = "Maybe later.",
    *,
    parse_status: ParseStatus = ParseStatus.NO_RELEVANT_ACTION,
    trial_id: str = "trial-17",
) -> ObservedAction:
    return _action(
        raw_text=raw_text,
        parse_status=parse_status,
        claims=(),
        offers=(),
        commitments=(),
        disclosures=(),
        trial_id=trial_id,
    )


def _context(**updates: Any) -> FeatureProjectionContext:
    payload: dict[str, Any] = {
        "trial_id": "trial-17",
        "observer_id": "counterpart",
        "source_actor_id": "actor",
        "source_event_id": "event-action",
        "source_call_id": "call-action",
        "turn": 2,
        "visibility": EvidenceVisibility.PUBLIC,
        "visible_to": (),
        "accessible_fact_ids": ("defect",),
        "accessible_event_ids": ("event-action",),
        "response_status": ResponseStatus.OBSERVED,
        "response_latency_seconds": 1.25,
    }
    payload.update(updates)
    return FeatureProjectionContext(**payload)


def _features(evidence: Evidence) -> dict[str, Any]:
    return dict(evidence.features)


def _project(
    action: ObservedAction | None = None,
    context: FeatureProjectionContext | None = None,
    auxiliary: Iterable[AuxiliaryFeatureInput] = (),
) -> tuple[Evidence, ...]:
    return DeterministicFeatureProjector().project(
        action or _action(),
        context or _context(),
        auxiliary,
    )


def _offer_action(
    raw_text: str,
    terms: tuple[tuple[str, Any, str], ...],
) -> ObservedAction:
    offer_span = _span(raw_text, raw_text, "offer")
    offer_terms = tuple(
        OfferTerm(
            name=name,
            value=value,
            evidence_spans=(_span(raw_text, source, f"term_{name}"),),
        )
        for name, value, source in terms
    )
    offer = Offer(
        actor_id="actor",
        recipient_ids=("counterpart",),
        terms=offer_terms,
        evidence_spans=(offer_span,),
    )
    return _action(
        raw_text=raw_text,
        claims=(),
        offers=(offer,),
        commitments=(),
        disclosures=(),
    )


def _auxiliary(
    feature_name: str = "linguistic_complexity",
    *,
    value: Any = 0.4,
    confidence: Any = 0.7,
    channel: EvidenceChannel = EvidenceChannel.LINGUISTIC,
    source_span_ids: tuple[str, ...] = (),
) -> AuxiliaryFeatureInput:
    return AuxiliaryFeatureInput(
        feature_name=feature_name,
        value=value,
        confidence=confidence,
        channel=channel,
        extractor_version="auxiliary-extractor-1",
        source_span_ids=source_span_ids,
    )


def test_multi_atomic_projection_is_versioned_and_scenario_relevant() -> None:
    evidence = _project()[0]
    features = _features(evidence)

    assert features["feature_projection_schema_version"] == (
        FEATURE_PROJECTION_SCHEMA_VERSION
    )
    assert features["scenario_id"] == "vehicle_sale"
    assert features["parse_status"] == ParseStatus.COMPLETE.value
    assert features["projection_status"] == ProjectionStatus.AVAILABLE.value
    assert features["atomic_action_category"] == "multi_atomic"
    assert features["parser_name"] == "typed_parser"
    assert features["parser_version"] == "1.0.0"
    assert features["action_id"] == _action().action_id
    assert evidence.extractor_version == "tom-feature-projector-1"


def test_available_offer_amount_terms_and_hashes_are_exact() -> None:
    action = _action()
    features = _features(_project(action)[0])
    term = action.offers[0].terms[0]

    assert features["offer_present"] is True
    assert features["offer_count"] == 1
    assert features["offer_terms_present"] is True
    assert features["offer_term_count"] == 1
    assert features["offer_term_ids_hash"] == _hash_ids((term.term_id,))
    assert features["offer_term_names_hash"] == _hash_ids(("price",))
    assert features["offer_amount_present"] is True
    assert features["offer_amount_status"] == "available"
    assert features["offer_amount"] == 70.0


def test_distinct_typed_offer_amounts_are_ambiguous_not_zero() -> None:
    action = _offer_action(
        "offer price $70 and amount $80",
        (("price", 70, "$70"), ("amount", 80, "$80")),
    )
    features = _features(_project(action)[0])

    assert features["offer_amount_present"] is True
    assert features["offer_amount_status"] == "ambiguous"
    assert "offer_amount" not in features


def test_offer_without_amount_term_is_typed_unavailable_not_zero() -> None:
    action = _offer_action(
        "offer delivery tomorrow",
        (("delivery", "tomorrow", "tomorrow"),),
    )
    features = _features(_project(action)[0])

    assert features["offer_amount_status"] == "unavailable"
    assert "offer_amount_present" not in features
    assert "offer_amount" not in features


def test_boolean_amount_term_is_not_coerced_to_numeric_amount() -> None:
    action = _offer_action("offer price true", (("price", True, "true"),))
    features = _features(_project(action)[0])

    assert features["offer_amount_present"] is True
    assert features["offer_amount_status"] == "unavailable"
    assert "offer_amount" not in features


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_scenario_schema_rejects_nonfinite_offer_amount(value: float) -> None:
    raw_text = "offer price value"
    source = _span(raw_text, "value", "offer_amount")

    with pytest.raises(ValidationError):
        OfferTerm(name="price", value=value, evidence_spans=(source,))


def test_projection_rejects_negative_typed_offer_amount() -> None:
    action = _offer_action("offer price -5", (("price", -5, "-5"),))

    with pytest.raises(ValueError, match="finite and non-negative"):
        _project(action)


def test_claim_fact_predicate_and_polarity_features_are_atomic() -> None:
    action = _action()
    claim = action.claims[0]
    features = _features(_project(action)[0])

    assert features["claim_present"] is True
    assert features["claim_count"] == 1
    assert features["claim_predicates_hash"] == _hash_ids((claim.predicate,))
    assert features["claim_fact_id_count"] == 1
    assert features["claim_fact_ids_hash"] == _hash_ids(("defect",))
    assert features["claim_positive_polarity_count"] == 0
    assert features["claim_negative_polarity_count"] == 1
    assert features["claim_semantics_source"] == "atomic_evidence"


def test_commitment_and_disclosure_features_retain_typed_identity() -> None:
    action = _action()
    commitment = action.commitments[0]
    disclosure = action.disclosures[0]
    features = _features(_project(action)[0])

    assert features["commitment_present"] is True
    assert features["commitment_count"] == 1
    assert features["commitment_ids_hash"] == _hash_ids(
        (commitment.commitment_id,)
    )
    assert features["commitment_action_ids_hash"] == _hash_ids(("cooperate",))
    assert features["disclosure_present"] is True
    assert features["disclosure_count"] == 1
    assert features["disclosure_ids_hash"] == _hash_ids(
        (disclosure.disclosure_id,)
    )
    assert features["disclosed_fact_ids_hash"] == _hash_ids(("defect",))


@pytest.mark.parametrize(
    ("category", "feature_name"),
    [
        (StructuredActionCategory.ACCEPT, "acceptance_present"),
        (StructuredActionCategory.REJECT, "rejection_present"),
        (
            StructuredActionCategory.REQUEST_EVIDENCE,
            "request_evidence_present",
        ),
    ],
)
def test_structured_action_category_is_the_only_typed_signal(
    category: StructuredActionCategory,
    feature_name: str,
) -> None:
    action = _empty_action("Structured action supplied by the event.")
    evidence = _project(
        action,
        _context(
            structured_action_category=category,
            response_latency_seconds=None,
        ),
    )[0]
    features = _features(evidence)

    assert features["projection_status"] == "available"
    assert features["atomic_action_category"] == category.value
    assert features[feature_name] is True


def test_raw_action_words_do_not_create_typed_action_features() -> None:
    raw_text = (
        "I accept, reject, request evidence, offer $5, claim a fact, "
        "promise, and disclose."
    )
    features = _features(
        _project(
            _empty_action(raw_text),
            _context(response_latency_seconds=None),
        )[0]
    )

    assert features["projection_status"] == "missing"
    assert features["atomic_action_category"] == "no_relevant_action"
    assert not {
        "acceptance_present",
        "rejection_present",
        "request_evidence_present",
        "offer_present",
        "claim_present",
        "commitment_present",
        "disclosure_present",
    } & features.keys()


def test_explicit_absence_and_latency_are_not_a_zero_observation() -> None:
    evidence = _project(
        _empty_action(""),
        _context(
            response_status=ResponseStatus.ABSENT,
            response_latency_seconds=5.0,
        ),
    )[0]
    features = _features(evidence)

    assert features["response_status"] == "absent"
    assert features["response_absent"] is True
    assert features["response_latency_seconds"] == 5.0
    assert features["response_latency_status"] == "available"
    assert features["projection_status"] == "missing"
    assert features["atomic_action_category"] == "response_absent"
    assert not any(name.endswith("_count") for name in features)


def test_absence_without_declared_latency_does_not_invent_zero() -> None:
    features = _features(
        _project(
            _empty_action(""),
            _context(
                response_status=ResponseStatus.ABSENT,
                response_latency_seconds=None,
            ),
        )[0]
    )

    assert features["response_absent"] is True
    assert "response_latency_seconds" not in features
    assert "response_latency_status" not in features


def test_absent_response_cannot_carry_raw_or_atomic_action() -> None:
    with pytest.raises(ValueError, match="absent response"):
        _project(
            _action(),
            _context(
                response_status=ResponseStatus.ABSENT,
                response_latency_seconds=3.0,
            ),
        )


def test_no_relevant_action_is_missing_without_numeric_atom_defaults() -> None:
    evidence = _project(
        _empty_action(),
        _context(response_latency_seconds=None),
    )[0]
    features = _features(evidence)

    assert features["parse_status"] == "no_relevant_action"
    assert features["projection_status"] == "missing"
    assert features["atomic_action_category"] == "no_relevant_action"
    assert features["source_span_status"] == "unavailable"
    assert not any(name.endswith("_count") for name in features)


@pytest.mark.parametrize(
    ("parse_status", "expected_status"),
    [
        (ParseStatus.UNCERTAIN, ProjectionStatus.AMBIGUOUS.value),
        (ParseStatus.FAILED, ProjectionStatus.UNAVAILABLE.value),
    ],
)
def test_partial_atoms_are_linked_but_withheld_when_parse_is_not_usable(
    parse_status: ParseStatus,
    expected_status: str,
) -> None:
    action = _action(parse_status=parse_status)
    evidence = _project(action)[0]
    features = _features(evidence)

    assert features["projection_status"] == expected_status
    assert features["atomic_action_category"] == "unavailable"
    assert features["source_span_count"] == 5
    assert not {
        "offer_present",
        "claim_present",
        "commitment_present",
        "disclosure_present",
    } & features.keys()
    assert not any(
        name.endswith("_count") and name != "source_span_count"
        for name in features
    )


def test_unavailable_response_with_complete_atoms_remains_unavailable() -> None:
    evidence = _project(
        _action(),
        _context(
            response_status=ResponseStatus.UNAVAILABLE,
            response_latency_seconds=None,
        ),
    )[0]
    features = _features(evidence)

    assert features["response_status"] == "unavailable"
    assert features["projection_status"] == "unavailable"
    assert features["atomic_action_category"] == "unavailable"
    assert not {
        "offer_present",
        "claim_present",
        "commitment_present",
        "disclosure_present",
    } & features.keys()


@pytest.mark.parametrize(
    "response_status",
    [ResponseStatus.ABSENT, ResponseStatus.UNAVAILABLE],
)
def test_unobserved_response_rejects_structured_action_category(
    response_status: ResponseStatus,
) -> None:
    with pytest.raises(ValueError, match="unobserved responses"):
        _context(
            response_status=response_status,
            structured_action_category=StructuredActionCategory.ACCEPT,
        )


def test_observable_evidence_has_exact_event_and_text_lineage() -> None:
    action = _action()
    context = _context()
    evidence = _project(action, context)[0]
    spans = {
        span.span_id: span
        for record in (
            *action.claims,
            *action.offers,
            *action.commitments,
            *action.disclosures,
        )
        for span in record.evidence_spans
    }
    for offer in action.offers:
        for term in offer.terms:
            spans.update({span.span_id: span for span in term.evidence_spans})
    features = _features(evidence)

    assert evidence.observer_id == context.observer_id == "counterpart"
    assert evidence.source_actor_id == context.source_actor_id == "actor"
    assert evidence.source_event_id == context.source_event_id == "event-action"
    assert evidence.source_call_id == context.source_call_id == "call-action"
    assert evidence.turn == context.turn == 2
    assert evidence.visibility is context.visibility is EvidenceVisibility.PUBLIC
    assert evidence.visible_to == ()
    assert evidence.source_text_hash == _sha256_text(action.raw_text)
    assert evidence.source_span == (
        min(span.start for span in spans.values()),
        max(span.end for span in spans.values()),
    )
    assert features["source_span_count"] == len(spans)
    assert features["source_span_ids_hash"] == _hash_ids(spans)
    assert features["action_id"] == action.action_id
    assert context.trial_id == action.trial_id == "trial-17"


def test_trial_mismatch_is_rejected_before_evidence_creation() -> None:
    with pytest.raises(ValueError, match="trial IDs do not match"):
        _project(_action(), _context(trial_id="trial-18"))


def test_trial_identity_is_retained_through_content_addressed_action() -> None:
    first_action = _action(trial_id="trial-17")
    second_action = _action(trial_id="trial-18")
    second_context = _context(trial_id="trial-18")
    first = _project(first_action)[0]
    second = _project(second_action, second_context)[0]

    assert _features(first)["action_id"] == first_action.action_id
    assert _features(second)["action_id"] == second_action.action_id
    assert first_action.action_id != second_action.action_id
    assert first.evidence_id != second.evidence_id


def test_explicit_visibility_is_preserved_for_authorized_observer() -> None:
    evidence = _project(
        context=_context(
            visibility=EvidenceVisibility.EXPLICIT,
            visible_to=("counterpart",),
        )
    )[0]

    assert evidence.visibility is EvidenceVisibility.EXPLICIT
    assert evidence.visible_to == ("counterpart",)
    assert evidence.is_visible_to("counterpart")
    assert not evidence.is_visible_to("actor")


@pytest.mark.parametrize(
    "updates",
    [
        {"visibility": EvidenceVisibility.ADJUDICATOR_ONLY},
        {
            "visibility": EvidenceVisibility.PUBLIC,
            "visible_to": ("counterpart",),
        },
        {"visibility": EvidenceVisibility.EXPLICIT, "visible_to": ()},
        {
            "visibility": EvidenceVisibility.EXPLICIT,
            "visible_to": ("actor",),
        },
    ],
)
def test_context_rejects_adjudicator_or_unauthorized_visibility(
    updates: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        _context(**updates)


def test_inaccessible_claim_or_disclosure_fact_is_rejected() -> None:
    with pytest.raises(ValueError, match="inaccessible facts"):
        _project(_action(), _context(accessible_fact_ids=()))


def test_context_actor_linkage_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="source actor"):
        _project(_action(), _context(source_actor_id="other_actor"))


def test_context_source_event_must_be_explicitly_accessible() -> None:
    with pytest.raises(ValidationError, match="explicitly accessible"):
        _context(source_event_id="other_event")


@pytest.mark.parametrize(
    "record_kind",
    ["claim", "offer", "commitment", "disclosure"],
)
def test_nested_atomic_actor_must_match_source_actor(record_kind: str) -> None:
    claims, offers, commitments, disclosures = _atomic_records()
    if record_kind == "claim":
        original = claims[0]
        claims = (
            Claim(
                subject_id=original.subject_id,
                predicate=original.predicate,
                value=original.value,
                asserted_by="intruder",
                polarity=original.polarity,
                fact_id=original.fact_id,
                evidence_spans=original.evidence_spans,
            ),
        )
    elif record_kind == "offer":
        original_offer = offers[0]
        offers = (
            Offer(
                actor_id="intruder",
                recipient_ids=original_offer.recipient_ids,
                terms=original_offer.terms,
                evidence_spans=original_offer.evidence_spans,
            ),
        )
    elif record_kind == "commitment":
        original_commitment = commitments[0]
        commitments = (
            Commitment(
                actor_id="intruder",
                promised_action_id=original_commitment.promised_action_id,
                beneficiary_role_ids=original_commitment.beneficiary_role_ids,
                effective_round=original_commitment.effective_round,
                evidence_spans=original_commitment.evidence_spans,
            ),
        )
    else:
        original_disclosure = disclosures[0]
        disclosures = (
            Disclosure(
                actor_id="intruder",
                recipient_ids=original_disclosure.recipient_ids,
                fact_id=original_disclosure.fact_id,
                fact_version=original_disclosure.fact_version,
                evidence_spans=original_disclosure.evidence_spans,
            ),
        )
    action = _action(
        claims=claims,
        offers=offers,
        commitments=commitments,
        disclosures=disclosures,
    )

    with pytest.raises(ValueError, match="atomic evidence actor"):
        _project(action)


def test_duplicate_atomic_records_are_rejected() -> None:
    claim = _atomic_records()[0][0]
    action = _action(claims=(claim, claim))

    with pytest.raises(ValueError, match="records must be unique"):
        _project(action)


def test_shared_atomic_span_is_deduplicated_in_source_lineage() -> None:
    raw_text = "shared statement"
    shared = _span(raw_text, raw_text, "shared")
    claim = Claim(
        subject_id="item",
        predicate="quality",
        value="high",
        asserted_by="actor",
        polarity=True,
        fact_id="quality",
        evidence_spans=(shared,),
    )
    commitment = Commitment(
        actor_id="actor",
        promised_action_id="quality",
        beneficiary_role_ids=("counterpart",),
        evidence_spans=(shared,),
    )
    action = _action(
        raw_text=raw_text,
        claims=(claim,),
        offers=(),
        commitments=(commitment,),
        disclosures=(),
    )
    features = _features(
        _project(
            action,
            _context(accessible_fact_ids=("defect", "quality")),
        )[0]
    )

    assert features["source_span_count"] == 1
    assert features["source_span_ids_hash"] == _hash_ids((shared.span_id,))


def test_observable_and_auxiliary_channels_are_separate_and_parent_linked() -> None:
    action = _action()
    claim_span = action.claims[0].evidence_spans[0]
    linguistic = _auxiliary(source_span_ids=(claim_span.span_id,))
    model = _auxiliary(
        "model_sentiment",
        value=-0.4,
        confidence=0.6,
        channel=EvidenceChannel.MODEL_DERIVED,
    )
    evidence = _project(action, auxiliary=(model, linguistic))

    assert tuple(item.channel for item in evidence) == (
        EvidenceChannel.OBSERVABLE,
        EvidenceChannel.LINGUISTIC,
        EvidenceChannel.MODEL_DERIVED,
    )
    observable, linguistic_evidence, model_evidence = evidence
    assert linguistic_evidence.reliability == 0.7
    assert model_evidence.reliability == 0.6
    assert linguistic_evidence.parent_evidence_ids == (observable.evidence_id,)
    assert model_evidence.parent_evidence_ids == (observable.evidence_id,)
    assert linguistic_evidence.source_event_id == observable.source_event_id
    assert linguistic_evidence.source_call_id == observable.source_call_id
    assert linguistic_evidence.source_text_hash == observable.source_text_hash
    assert linguistic_evidence.source_span == (
        claim_span.start,
        claim_span.end,
    )
    assert model_evidence.source_span is None
    assert _features(linguistic_evidence)[
        "aux.linguistic_complexity.confidence"
    ] == 0.7
    assert _features(model_evidence)["aux.model_sentiment.value"] == -0.4
    assert not any(
        name.startswith("aux.") for name, _ in observable.features
    )


@pytest.mark.parametrize(
    "confidence",
    [True, -0.1, 1.1, math.nan, math.inf, -math.inf],
)
def test_auxiliary_confidence_rejects_boolean_or_invalid_number(
    confidence: Any,
) -> None:
    with pytest.raises(ValidationError):
        _auxiliary(confidence=confidence)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_auxiliary_numeric_value_must_be_finite(value: float) -> None:
    with pytest.raises(ValidationError):
        _auxiliary(value=value)


def test_signed_auxiliary_value_is_preserved_without_becoming_observable() -> None:
    auxiliary = _auxiliary(
        "model_score",
        value=-0.75,
        channel=EvidenceChannel.MODEL_DERIVED,
    )
    observable, model = _project(auxiliary=(auxiliary,))

    assert "aux.model_score.value" not in _features(observable)
    assert _features(model)["aux.model_score.value"] == -0.75


@pytest.mark.parametrize(
    "feature_name",
    [
        "actual_deception",
        "actual-deception-score",
        "actual/deception/label",
        "deception_ground_truth",
        "ground.truth.deception",
        "behavioral_misrepresentation",
    ],
)
def test_auxiliary_input_rejects_actual_deception_label_variants(
    feature_name: str,
) -> None:
    with pytest.raises(ValidationError, match="actual-deception labels"):
        _auxiliary(feature_name, value=True)


def test_perceived_deception_cue_is_allowed_only_as_auxiliary_evidence() -> None:
    cue = _auxiliary(
        "perceived_deception_score",
        value=0.3,
        channel=EvidenceChannel.MODEL_DERIVED,
    )
    observable, auxiliary = _project(auxiliary=(cue,))

    assert "aux.perceived_deception_score.value" not in _features(observable)
    assert _features(auxiliary)["aux.perceived_deception_score.value"] == 0.3


def test_duplicate_auxiliary_feature_names_are_rejected() -> None:
    auxiliary = _auxiliary()

    with pytest.raises(ValueError, match="names must be unique"):
        _project(auxiliary=(auxiliary, auxiliary))


def test_duplicate_auxiliary_span_ids_are_rejected() -> None:
    span_id = _action().claims[0].evidence_spans[0].span_id

    with pytest.raises(ValidationError, match="duplicates"):
        _auxiliary(source_span_ids=(span_id, span_id))


def test_auxiliary_unknown_source_span_is_rejected() -> None:
    auxiliary = _auxiliary(
        source_span_ids=("evidence_span_" + "0" * 64,)
    )

    with pytest.raises(ValueError, match="unknown source spans"):
        _project(auxiliary=(auxiliary,))


@pytest.mark.parametrize("value", [42, None, object()])
def test_auxiliary_iterable_or_member_type_is_checked(value: Any) -> None:
    if value is None:
        supplied: Any = (None,)
    else:
        supplied = value
    with pytest.raises(TypeError, match="AuxiliaryFeatureInput"):
        _project(auxiliary=supplied)


def test_auxiliary_generator_is_materialized_and_canonically_ordered() -> None:
    linguistic = _auxiliary("z_linguistic")
    model = _auxiliary(
        "a_model",
        channel=EvidenceChannel.MODEL_DERIVED,
    )
    source = [linguistic, model]
    generated = (item for item in source)
    projected = _project(auxiliary=generated)
    source.clear()

    assert tuple(item.channel for item in projected) == (
        EvidenceChannel.OBSERVABLE,
        EvidenceChannel.MODEL_DERIVED,
        EvidenceChannel.LINGUISTIC,
    )
    assert _features(projected[1])["aux.a_model.value"] == 0.4
    assert _features(projected[2])["aux.z_linguistic.value"] == 0.4


def test_raw_text_is_hashed_but_never_emitted_as_a_feature_or_summary() -> None:
    action = _action()
    evidence = _project(action)[0]
    features = _features(evidence)

    assert evidence.source_text_hash == _sha256_text(action.raw_text)
    assert "raw_text" not in features
    assert action.raw_text not in features.values()
    assert action.raw_text not in (evidence.summary or "")
    assert "private_prompt" not in features
    assert "chain_of_thought" not in features


def test_quotation_and_negation_semantics_come_only_from_claim_atom() -> None:
    raw_text = 'I merely quote "not true" without endorsing it.'
    evidence_span = _span(raw_text, "not true", "quoted_claim")
    typed_positive_claim = Claim(
        subject_id="item",
        predicate="quality",
        value=True,
        asserted_by="actor",
        polarity=True,
        fact_id="quality",
        evidence_spans=(evidence_span,),
    )
    action = _action(
        raw_text=raw_text,
        claims=(typed_positive_claim,),
        offers=(),
        commitments=(),
        disclosures=(),
    )
    features = _features(
        _project(
            action,
            _context(accessible_fact_ids=("defect", "quality")),
        )[0]
    )

    assert features["claim_positive_polarity_count"] == 1
    assert features["claim_negative_polarity_count"] == 0
    assert features["claim_semantics_source"] == "atomic_evidence"
    assert "quoted" not in features
    assert "negated" not in features


@pytest.mark.parametrize(
    "latency",
    [True, -0.1, math.nan, math.inf, -math.inf],
)
def test_response_latency_rejects_boolean_or_invalid_number(latency: Any) -> None:
    with pytest.raises(ValidationError):
        _context(response_latency_seconds=latency)


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (_context, "turn"),
        (_auxiliary, "confidence"),
        (DeterministicFeatureProjector, "version"),
    ],
)
def test_feature_models_are_frozen(
    factory: Callable[[], BaseModel],
    field: str,
) -> None:
    record = factory()

    with pytest.raises(ValidationError, match="frozen"):
        setattr(record, field, getattr(record, field))


@pytest.mark.parametrize(
    ("factory", "payload"),
    [
        (
            FeatureProjectionContext,
            {
                **_context().model_dump(),
                "unexpected": True,
            },
        ),
        (
            AuxiliaryFeatureInput,
            {
                **_auxiliary().model_dump(),
                "unexpected": True,
            },
        ),
        (
            DeterministicFeatureProjector,
            {
                **DeterministicFeatureProjector().model_dump(),
                "unexpected": True,
            },
        ),
    ],
)
def test_feature_models_forbid_extra_fields(
    factory: type[BaseModel],
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        factory(**payload)


def test_context_does_not_accept_ambiguous_event_alias() -> None:
    payload = _context().model_dump()
    payload["event_id"] = payload.pop("source_event_id")

    with pytest.raises(ValidationError):
        FeatureProjectionContext(**payload)


def test_context_requires_canonical_unique_access_lists() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        _context(accessible_event_ids=("event-action", "event-action"))
    with pytest.raises(ValidationError, match="lexicographic"):
        _context(accessible_fact_ids=("quality", "defect"))


def test_projector_requires_canonical_unique_amount_term_names() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        DeterministicFeatureProjector(amount_term_names=("price", "price"))
    with pytest.raises(ValidationError, match="lexicographic"):
        DeterministicFeatureProjector(amount_term_names=("price", "amount"))


def test_projection_identity_and_feature_keys_are_deterministic() -> None:
    action = _action()
    context = _context()
    first = _project(action, context)
    second = _project(action, context)

    assert first == second
    assert tuple(item.evidence_id for item in first) == tuple(
        item.evidence_id for item in second
    )
    names = tuple(name for name, _ in first[0].features)
    assert names == tuple(sorted(names))
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "context",
    [
        _context(
            source_event_id="event-other",
            accessible_event_ids=("event-other",),
        ),
        _context(source_call_id="call-other"),
        _context(turn=3),
    ],
)
def test_provenance_changes_evidence_identity(
    context: FeatureProjectionContext,
) -> None:
    baseline = _project()[0]
    changed = _project(context=context)[0]

    assert changed.evidence_id != baseline.evidence_id


def test_projection_rejects_wrong_runtime_types() -> None:
    projector = DeterministicFeatureProjector()

    with pytest.raises(TypeError, match="ObservedAction"):
        projector.project(object(), _context())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FeatureProjectionContext"):
        projector.project(_action(), object())  # type: ignore[arg-type]
