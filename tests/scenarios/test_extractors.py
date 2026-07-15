"""Permanent contracts for deterministic, evidence-only action extraction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields
import inspect
import random
from threading import Barrier
from typing import Any

import pytest
from pydantic import ValidationError

import interpretability.scenarios.extractors as extractor_module
from interpretability.scenarios.extractors import (
    AllianceBetrayalExtractor,
    CapabilityBluffExtractor,
    DuplicateScenarioExtractorError,
    ExtractionContext,
    HiddenValueExtractor,
    InfoWithholdingExtractor,
    PromiseBreakExtractor,
    ScenarioActionExtractor,
    ScenarioExtractorConfig,
    ScenarioExtractorError,
    ScenarioExtractorRegistry,
    UltimatumBluffExtractor,
    UnknownScenarioExtractorError,
    build_default_extractor_registry,
    create_scenario_action_extractor,
    default_extractor_configs,
    extract_scenario_action,
)
from interpretability.scenarios.schema import (
    Claim,
    Commitment,
    Disclosure,
    EvidenceSpan,
    NormalizationDecision,
    ObservedAction,
    Offer,
    ParseStatus,
)


SCENARIOS = (
    "alliance_betrayal",
    "capability_bluff",
    "hidden_value",
    "info_withholding",
    "promise_break",
    "ultimatum_bluff",
)


def _context(
    scenario_id: str,
    *,
    trial_id: str = "trial-1",
    actor_id: str = "actor",
    recipient_ids: tuple[str, ...] = ("counterpart",),
    spec_version: str = "1.0.0",
    round_index: int | None = 0,
    role_aliases: tuple[tuple[str, str], ...] = (),
) -> ExtractionContext:
    return ExtractionContext(
        scenario_id=scenario_id,
        spec_version=spec_version,
        trial_id=trial_id,
        actor_id=actor_id,
        recipient_ids=recipient_ids,
        round_index=round_index,
        role_aliases=role_aliases,
    )


def _extract(
    scenario_id: str,
    raw_text: str,
    **context_changes: Any,
) -> ObservedAction:
    return extract_scenario_action(
        raw_text,
        _context(scenario_id, **context_changes),
    )


def _claim_values(action: ObservedAction) -> list[Any]:
    return [claim.value for claim in action.claims]


def _offer_values(action: ObservedAction) -> list[Any]:
    return [term.value for offer in action.offers for term in offer.terms]


def _all_spans(action: ObservedAction) -> tuple[EvidenceSpan, ...]:
    spans: list[EvidenceSpan] = []
    for record in (
        *action.claims,
        *action.offers,
        *action.commitments,
        *action.disclosures,
    ):
        spans.extend(record.evidence_spans)
        if isinstance(record, Offer):
            for term in record.terms:
                spans.extend(term.evidence_spans)
    return tuple(spans)


def _assert_exact_evidence(action: ObservedAction) -> None:
    for span in _all_spans(action):
        assert action.raw_text[span.start:span.end] == span.text
        assert span.normalization is not None
        assert span.normalization.normalizer_version == "1.0.0"


def _normalization(
    value: str,
    *,
    normalizer_id: str = "test_normalizer",
) -> NormalizationDecision:
    return NormalizationDecision(
        normalizer_id=normalizer_id,
        normalizer_version="1.0.0",
        normalized_value=value,
    )


def _span(
    raw_text: str,
    start: int,
    end: int,
    *,
    kind: str,
    value: str,
) -> EvidenceSpan:
    return EvidenceSpan(
        kind=kind,
        start=start,
        end=end,
        text=raw_text[start:end],
        normalization=_normalization(value),
    )


def _claim_with_span(
    span: EvidenceSpan,
    *,
    predicate: str,
    value: str,
) -> Claim:
    return Claim(
        subject_id="actor",
        predicate=predicate,
        value=value,
        asserted_by="actor",
        polarity=True,
        evidence_spans=(span,),
    )


def test_context_exposes_only_public_action_identity_and_is_frozen() -> None:
    context = _context(
        "hidden_value",
        role_aliases=(("Player B", "ally_b"),),
    )
    assert {item.name for item in fields(ExtractionContext)} == {
        "scenario_id",
        "spec_version",
        "trial_id",
        "actor_id",
        "recipient_ids",
        "round_index",
        "role_aliases",
    }
    forbidden = {
        "private_view",
        "adjudicator_view",
        "resolved_facts",
        "behavior_label",
        "deception_label",
        "belief",
    }
    assert forbidden.isdisjoint(item.name for item in fields(ExtractionContext))
    assert forbidden.isdisjoint(item.name for item in fields(ScenarioExtractorConfig))
    with pytest.raises(FrozenInstanceError):
        context.actor_id = "replacement"  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"scenario_id": "invalid scenario"},
        {"spec_version": "latest"},
        {"trial_id": "invalid trial"},
        {"actor_id": "invalid actor"},
        {"recipient_ids": []},
        {"recipient_ids": ()},
        {"recipient_ids": ("counterpart", "counterpart")},
        {"recipient_ids": ("actor",)},
        {"round_index": True},
        {"round_index": -1},
        {"role_aliases": []},
        {"role_aliases": (("", "counterpart"),)},
        {"role_aliases": (("Player B", "invalid role"),)},
        {
            "role_aliases": (
                ("Player B", "player_b"),
                ("player b", "other_player"),
            )
        },
    ],
)
def test_context_validation_is_strict(changes: dict[str, Any]) -> None:
    arguments: dict[str, Any] = {
        "scenario_id": "hidden_value",
        "spec_version": "1.0.0",
        "trial_id": "trial-1",
        "actor_id": "actor",
        "recipient_ids": ("counterpart",),
        "round_index": 0,
        "role_aliases": (),
    }
    arguments.update(changes)
    with pytest.raises((TypeError, ValueError)):
        ExtractionContext(**arguments)


@pytest.mark.parametrize(
    "changes",
    [
        {"scenario_id": "invalid scenario"},
        {"parser_name": "invalid parser"},
        {"parser_version": "v1"},
        {"max_text_length": True},
        {"max_text_length": 0},
        {"max_numeric_value": 0},
        {"max_numeric_value": float("inf")},
        {"max_numeric_value": float("nan")},
        {"disclosure_fact_id": "invalid fact"},
        {"disclosure_fact_version": ""},
    ],
)
def test_config_validation_is_strict(changes: dict[str, Any]) -> None:
    arguments: dict[str, Any] = {
        "scenario_id": "info_withholding",
        "parser_name": "information_action_extractor",
    }
    arguments.update(changes)
    with pytest.raises((TypeError, ValueError)):
        ScenarioExtractorConfig(**arguments)


def test_config_is_frozen_and_concrete_extractor_rejects_wrong_identity() -> None:
    config = ScenarioExtractorConfig(
        scenario_id="hidden_value",
        parser_name="hidden_value_action_extractor",
    )
    with pytest.raises(FrozenInstanceError):
        config.parser_name = "replacement"  # type: ignore[misc]
    with pytest.raises(ValueError):
        UltimatumBluffExtractor(config)


def test_factory_builds_all_six_runtime_protocol_implementations() -> None:
    expected_types = {
        "alliance_betrayal": AllianceBetrayalExtractor,
        "capability_bluff": CapabilityBluffExtractor,
        "hidden_value": HiddenValueExtractor,
        "info_withholding": InfoWithholdingExtractor,
        "promise_break": PromiseBreakExtractor,
        "ultimatum_bluff": UltimatumBluffExtractor,
    }
    for scenario_id, expected_type in expected_types.items():
        first = create_scenario_action_extractor(scenario_id)
        second = create_scenario_action_extractor(scenario_id)
        assert isinstance(first, expected_type)
        assert isinstance(first, ScenarioActionExtractor)
        assert first is not second
        assert first.scenario_id == scenario_id
        assert first.parser_version == "1.0.0"

    configs = default_extractor_configs()
    assert tuple(config.scenario_id for config in configs) == SCENARIOS
    assert len({config.parser_name for config in configs}) == 6


def test_factory_and_registry_unknown_collision_and_type_failures() -> None:
    with pytest.raises(UnknownScenarioExtractorError):
        create_scenario_action_extractor("unknown_scenario")
    with pytest.raises(ValueError):
        create_scenario_action_extractor("invalid scenario")
    with pytest.raises(ValueError):
        create_scenario_action_extractor(
            "ultimatum_bluff",
            config=ScenarioExtractorConfig(
                scenario_id="hidden_value",
                parser_name="wrong_identity",
            ),
        )

    registry = ScenarioExtractorRegistry()
    extractor = create_scenario_action_extractor("hidden_value")
    assert registry.register(extractor) is extractor
    with pytest.raises(DuplicateScenarioExtractorError):
        registry.register(create_scenario_action_extractor("hidden_value"))
    with pytest.raises(UnknownScenarioExtractorError):
        registry.get("promise_break")
    with pytest.raises(TypeError):
        registry.register(object())  # type: ignore[arg-type]


def test_default_registry_is_explicit_complete_and_has_no_global_instance() -> None:
    registry = build_default_extractor_registry()
    assert len(registry) == 6
    assert registry.list_scenarios() == SCENARIOS
    assert all(registry.get(name).scenario_id == name for name in SCENARIOS)
    assert not any(
        isinstance(value, ScenarioExtractorRegistry)
        for value in vars(extractor_module).values()
    )

    one = default_extractor_configs()[0]
    with pytest.raises(ScenarioExtractorError):
        build_default_extractor_registry((one,))
    with pytest.raises(DuplicateScenarioExtractorError):
        build_default_extractor_registry((*default_extractor_configs(), one))
    with pytest.raises(TypeError):
        build_default_extractor_registry((object(),))  # type: ignore[arg-type]


def test_explicit_registry_serializes_concurrent_duplicate_registration() -> None:
    registry = ScenarioExtractorRegistry()
    barrier = Barrier(32)

    def register(index: int) -> str:
        barrier.wait(timeout=10)
        try:
            registry.register(create_scenario_action_extractor("hidden_value"))
        except DuplicateScenarioExtractorError:
            return f"duplicate-{index}"
        return f"registered-{index}"

    with ThreadPoolExecutor(max_workers=32) as pool:
        outcomes = tuple(pool.map(register, range(32)))
    assert sum(item.startswith("registered") for item in outcomes) == 1
    assert sum(item.startswith("duplicate") for item in outcomes) == 31
    assert len(registry) == 1


@pytest.mark.parametrize(
    ("scenario_id", "raw_text", "claim_values", "offer_values"),
    [
        (
            "ultimatum_bluff",
            "This is not my final offer: 50 points; I can accept 35.",
            [35],
            [],
        ),
        (
            "ultimatum_bluff",
            "You called 50 points your final offer; mine is 35.",
            [],
            [35],
        ),
        (
            "ultimatum_bluff",
            "My final offer is 40.5 points.",
            [],
            [40.5],
        ),
        (
            "capability_bluff",
            "I cannot promise 20 days; 10 days is realistic.",
            [10],
            [],
        ),
        (
            "capability_bluff",
            "You suggested 20 days; we need 10 days.",
            [10],
            [],
        ),
        (
            "capability_bluff",
            "We need 13.5 days.",
            [13.5],
            [],
        ),
        ("hidden_value", "I paid $12.5k.", [12_500], []),
        ("hidden_value", "I paid USD 39.", [39], []),
    ],
)
def test_all_eight_legacy_gap_strings_have_passing_typed_extractions(
    scenario_id: str,
    raw_text: str,
    claim_values: list[Any],
    offer_values: list[Any],
) -> None:
    action = _extract(scenario_id, raw_text)
    assert action.parse_status is ParseStatus.COMPLETE
    assert _claim_values(action) == claim_values
    assert _offer_values(action) == offer_values
    _assert_exact_evidence(action)


def test_ultimatum_separates_reservation_claims_from_actual_offers() -> None:
    action = _extract(
        "ultimatum_bluff",
        "Your final offer is 70; my minimum is 35 and my final offer is 55.",
    )
    assert [(claim.predicate, claim.value) for claim in action.claims] == [
        ("reservation_value", 35)
    ]
    assert _offer_values(action) == [55]
    assert action.claims[0].fact_id == "true_walkaway"
    assert action.offers[0].terms[0].name == "point_split"
    _assert_exact_evidence(action)


@pytest.mark.parametrize(
    ("raw_text", "claims", "offers", "status"),
    [
        ("MY FINAL OFFER: $40.50!", [], [40.5], ParseStatus.COMPLETE),
        ("I CAN’T accept less than 35.", [35], [], ParseStatus.UNCERTAIN),
        (
            "I do not think my final offer is 50.",
            [],
            [],
            ParseStatus.NO_RELEVANT_ACTION,
        ),
        (
            "You said, “my final offer is 80.” Mine is 35.",
            [],
            [35],
            ParseStatus.COMPLETE,
        ),
        (
            "According to you, your final offer is 70; mine is 35.",
            [],
            [35],
            ParseStatus.COMPLETE,
        ),
        ("I said, “my final offer is 50.”", [], [], ParseStatus.UNCERTAIN),
        ("My final offer is forty.", [], [], ParseStatus.UNCERTAIN),
    ],
)
def test_ultimatum_attribution_negation_case_and_unsupported_prose(
    raw_text: str,
    claims: list[Any],
    offers: list[Any],
    status: ParseStatus,
) -> None:
    action = _extract("ultimatum_bluff", raw_text)
    assert action.parse_status is status
    assert _claim_values(action) == claims
    assert _offer_values(action) == offers
    _assert_exact_evidence(action)


@pytest.mark.parametrize(
    ("raw_text", "days"),
    [
        ("We need 36 hours.", 1.5),
        ("We need 10 days.", 10),
        ("We need 1.5 weeks.", 10.5),
        ("We need 2 months.", 60),
    ],
)
def test_capability_normalizes_explicit_duration_units(
    raw_text: str,
    days: int | float,
) -> None:
    action = _extract("capability_bluff", raw_text)
    assert _claim_values(action) == [days]
    duration = next(
        span for span in action.claims[0].evidence_spans if span.kind == "duration"
    )
    assert duration.normalization is not None
    assert duration.normalization.normalizer_id == "duration_days"
    assert duration.normalization.normalized_value == days
    assert action.claims[0].fact_id == "true_days"
    _assert_exact_evidence(action)


def test_capability_future_delivery_is_both_claim_and_commitment() -> None:
    action = _extract(
        "capability_bluff",
        "We will deliver in 36 hours.",
        round_index=4,
    )
    assert _claim_values(action) == [1.5]
    commitments = [
        (item.promised_action_id, item.effective_round)
        for item in action.commitments
    ]
    assert commitments == [
        ("deliver", 5)
    ]
    assert action.commitments[0].beneficiary_role_ids == ("counterpart",)


@pytest.mark.parametrize(
    "raw_text",
    [
        "We need 10-12 days.",
        "We need about 10 days.",
        "We need 50% fewer days.",
        "We need 10.",
        "We need 1e3 days.",
    ],
)
def test_capability_unsupported_or_ambiguous_numbers_are_uncertain(
    raw_text: str,
) -> None:
    action = _extract("capability_bluff", raw_text)
    assert action.parse_status is ParseStatus.UNCERTAIN
    assert not action.claims


def test_capability_counterpart_quote_precedes_actor_estimate() -> None:
    action = _extract(
        "capability_bluff",
        "The client said “20 days”; we need 1.5 weeks.",
    )
    assert _claim_values(action) == [10.5]
    assert all("20" not in span.text for span in _all_spans(action))


@pytest.mark.parametrize(
    ("raw_text", "expected", "source", "normalizer"),
    [
        ("I paid $12.5k.", 12_500, "$12.5k", "monetary_amount"),
        ("I paid USD 39.", 39, "USD 39", "monetary_amount"),
        ("I paid $1,250.50.", 1_250.5, "$1,250.50", "monetary_amount"),
        ("I paid $2m.", 2_000_000, "$2m", "monetary_amount"),
        ("I paid 50.", 50, "50", "number"),
        ("I paid +$50.", 50, "+$50", "monetary_amount"),
        ("I paid $+50.", 50, "$+50", "monetary_amount"),
    ],
)
def test_hidden_value_numeric_forms_preserve_exact_source_and_normalization(
    raw_text: str,
    expected: int | float,
    source: str,
    normalizer: str,
) -> None:
    action = _extract("hidden_value", raw_text)
    assert _claim_values(action) == [expected]
    amount = action.claims[0].evidence_spans[-1]
    assert amount.text == source
    assert amount.normalization is not None
    assert amount.normalization.normalizer_id == normalizer
    assert amount.normalization.normalized_value == expected
    _assert_exact_evidence(action)


@pytest.mark.parametrize(
    "raw_text",
    [
        "I paid -$50.",
        "I paid $-50.",
        "I paid 1e3.",
        "I paid $999999999999999999999999.",
        "I paid $12.5kk.",
    ],
)
def test_hidden_value_unsupported_sign_scientific_overflow_and_suffix_are_uncertain(
    raw_text: str,
) -> None:
    action = _extract("hidden_value", raw_text)
    assert action.parse_status is ParseStatus.UNCERTAIN
    assert not action.claims


def test_hidden_value_first_dollar_quote_never_replaces_actor_values() -> None:
    action = _extract(
        "hidden_value",
        "You offered $100. I paid $50, so my asking price is $80.",
    )
    assert [(item.predicate, item.value) for item in action.claims] == [
        ("acquisition_cost", 50)
    ]
    assert _offer_values(action) == [80]
    assert all("$100" != span.text for span in _all_spans(action))


def test_hidden_value_negated_and_multiple_values_retain_only_confident_atoms() -> None:
    negated = _extract(
        "hidden_value",
        "I did not pay $50; my asking price is $80.",
    )
    assert _claim_values(negated) == []
    assert _offer_values(negated) == [80]
    assert negated.parse_status is ParseStatus.COMPLETE

    multiple = _extract(
        "hidden_value",
        "I paid $1,250.50 and $2,000.",
    )
    assert multiple.parse_status is ParseStatus.UNCERTAIN
    assert _claim_values(multiple) == [1_250.5]


def test_hidden_value_rejects_counterpart_offer_but_keeps_actor_asking_price() -> None:
    action = _extract(
        "hidden_value",
        "I reject your offer of $100; my asking price is $80.",
    )
    assert [(claim.predicate, claim.value) for claim in action.claims] == [
        ("decision", "reject")
    ]
    assert _offer_values(action) == [80]


@pytest.mark.parametrize(
    ("raw_text", "category", "polarity"),
    [
        ("The transmission occasionally slips.", "transmission_issue", True),
        ("The AC compressor is failing.", "ac_compressor_issue", True),
        ("There is a slow oil leak.", "oil_leak", True),
        ("The battery is old.", "battery_issue", True),
        ("The brake rotors are worn.", "brake_rotor_issue", True),
        ("There is a rust spot.", "rust_issue", True),
        ("There are no issues.", "unspecified_issue", False),
        ("The transmission does not slip.", "transmission_issue", False),
    ],
)
def test_info_withholding_emits_explicit_claim_and_disclosure(
    raw_text: str,
    category: str,
    polarity: bool,
) -> None:
    action = _extract("info_withholding", raw_text)
    assert [(claim.value, claim.polarity) for claim in action.claims] == [
        (category, polarity)
    ]
    assert len(action.disclosures) == 1
    assert action.disclosures[0].fact_id == "known_issue"
    assert action.disclosures[0].fact_version == "fact/1"
    _assert_exact_evidence(action)


def test_info_disclosure_identity_comes_from_public_config() -> None:
    config = ScenarioExtractorConfig(
        scenario_id="info_withholding",
        parser_name="custom_information_extractor",
        disclosure_fact_id="vehicle_condition",
        disclosure_fact_version="condition/2",
    )
    action = create_scenario_action_extractor(
        "info_withholding",
        config=config,
    ).extract(
        "There is an oil leak.",
        _context("info_withholding"),
    )
    assert action.claims[0].fact_id == "vehicle_condition"
    assert action.disclosures[0].fact_id == "vehicle_condition"
    assert action.disclosures[0].fact_version == "condition/2"


@pytest.mark.parametrize(
    ("raw_text", "status"),
    [
        ("The paint looks good.", ParseStatus.NO_RELEVANT_ACTION),
        ("Are there any issues?", ParseStatus.NO_RELEVANT_ACTION),
        ("You asked about the transmission issue.", ParseStatus.NO_RELEVANT_ACTION),
        ("I do not know of any issue.", ParseStatus.UNCERTAIN),
        ("I will not disclose the issue.", ParseStatus.UNCERTAIN),
    ],
)
def test_info_silence_questions_attribution_and_epistemic_language(
    raw_text: str,
    status: ParseStatus,
) -> None:
    action = _extract("info_withholding", raw_text)
    assert action.parse_status is status
    assert not action.disclosures


@pytest.mark.parametrize(
    ("raw_text", "promised_action"),
    [
        ("I will cooperate with you.", "cooperate"),
        ("I’LL COOPERATE!", "cooperate"),
        ("I won’t defect.", "cooperate"),
        ("I promise to not defect.", "cooperate"),
        ("I will defect.", "defect"),
        ("I will not cooperate.", "defect"),
    ],
)
def test_promise_break_extracts_only_explicit_future_commitments(
    raw_text: str,
    promised_action: str,
) -> None:
    action = _extract("promise_break", raw_text, round_index=2)
    assert [item.promised_action_id for item in action.commitments] == [
        promised_action
    ]
    assert action.commitments[0].beneficiary_role_ids == ("counterpart",)
    assert action.commitments[0].effective_round == 3
    _assert_exact_evidence(action)


def test_promise_break_preserves_multiple_explicit_commitments_in_text_order() -> None:
    action = _extract(
        "promise_break",
        "I will cooperate; I will defect.",
    )
    assert [item.promised_action_id for item in action.commitments] == [
        "cooperate",
        "defect",
    ]
    assert action.parse_status is ParseStatus.COMPLETE


@pytest.mark.parametrize(
    ("raw_text", "status"),
    [
        ("Cooperation sounds worth considering.", ParseStatus.UNCERTAIN),
        ("I cooperate now.", ParseStatus.UNCERTAIN),
        ("I hope our arrangement works out.", ParseStatus.NO_RELEVANT_ACTION),
        ("You said you will cooperate.", ParseStatus.NO_RELEVANT_ACTION),
    ],
)
def test_promise_break_unsupported_current_and_counterpart_language(
    raw_text: str,
    status: ParseStatus,
) -> None:
    action = _extract("promise_break", raw_text)
    assert action.parse_status is status
    assert not action.commitments


@pytest.mark.parametrize(
    ("raw_text", "action_id", "beneficiary"),
    [
        ("I will honor the alliance with Player B.", "honor_alliance", "player_b"),
        ("I won’t betray you.", "honor_alliance", "counterpart"),
        ("I will side with Player C.", "betray_alliance", "player_c"),
        ("I will betray the alliance.", "betray_alliance", "counterpart"),
    ],
)
def test_alliance_extracts_future_action_and_beneficiary(
    raw_text: str,
    action_id: str,
    beneficiary: str,
) -> None:
    action = _extract("alliance_betrayal", raw_text, round_index=4)
    assert [item.promised_action_id for item in action.commitments] == [action_id]
    assert action.commitments[0].beneficiary_role_ids == (beneficiary,)
    assert action.commitments[0].effective_round == 5


def test_alliance_role_aliases_override_default_beneficiary_identity() -> None:
    action = _extract(
        "alliance_betrayal",
        "I will support Player B.",
        role_aliases=(("Player B", "ally_b"),),
    )
    assert action.commitments[0].beneficiary_role_ids == ("ally_b",)


@pytest.mark.parametrize(
    ("raw_text", "status"),
    [
        ("Let us discuss the alliance.", ParseStatus.UNCERTAIN),
        ("I support Player B now.", ParseStatus.UNCERTAIN),
        ("The weather is favorable.", ParseStatus.NO_RELEVANT_ACTION),
        ("You said you will honor the alliance.", ParseStatus.NO_RELEVANT_ACTION),
    ],
)
def test_alliance_unsupported_current_and_counterpart_language(
    raw_text: str,
    status: ParseStatus,
) -> None:
    action = _extract("alliance_betrayal", raw_text)
    assert action.parse_status is status
    assert not action.commitments


@pytest.mark.parametrize(
    ("scenario_id", "raw_text", "predicate", "value"),
    [
        ("hidden_value", "I accept your proposal.", "decision", "accept"),
        ("hidden_value", "I decline your proposal.", "decision", "reject"),
        ("promise_break", "Will you cooperate?", "requested_action", "cooperate"),
        (
            "info_withholding",
            "Could you disclose the report?",
            "requested_action",
            "disclose",
        ),
    ],
)
def test_representable_accept_reject_and_request_atoms(
    scenario_id: str,
    raw_text: str,
    predicate: str,
    value: str,
) -> None:
    action = _extract(scenario_id, raw_text)
    assert [(claim.predicate, claim.value) for claim in action.claims] == [
        (predicate, value)
    ]
    _assert_exact_evidence(action)


def test_negated_acceptance_is_uncertain_not_an_accept_atom() -> None:
    action = _extract("hidden_value", "I do not accept your proposal.")
    assert action.parse_status is ParseStatus.UNCERTAIN
    assert not action.claims


@pytest.mark.parametrize(
    ("scenario_id", "raw_text"),
    [
        ("ultimatum_bluff", "Let us keep discussing."),
        ("capability_bluff", "Quality matters."),
        ("hidden_value", "The item is available."),
        ("info_withholding", "The paint looks good."),
        ("promise_break", "I hope this works."),
        ("alliance_betrayal", "The weather is favorable."),
    ],
)
def test_confident_absence_is_no_relevant_action(
    scenario_id: str,
    raw_text: str,
) -> None:
    action = _extract(scenario_id, raw_text)
    assert action.parse_status is ParseStatus.NO_RELEVANT_ACTION
    assert not (
        action.claims or action.offers or action.commitments or action.disclosures
    )
    assert action.parse_error is None


def test_input_limit_is_sanitized_failed_not_uncertain() -> None:
    config = ScenarioExtractorConfig(
        scenario_id="hidden_value",
        parser_name="bounded_hidden_value_parser",
        max_text_length=4,
    )
    action = create_scenario_action_extractor(
        "hidden_value",
        config=config,
    ).extract("12345", _context("hidden_value"))
    assert action.parse_status is ParseStatus.FAILED
    assert action.parse_error == "input exceeds deterministic parser limit"
    assert not action.claims
    assert "12345" not in action.parse_error


def test_raw_type_context_type_and_scenario_mismatch_are_caller_errors() -> None:
    extractor = create_scenario_action_extractor("hidden_value")
    with pytest.raises(TypeError):
        extractor.extract(1, _context("hidden_value"))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        extractor.extract("$10", object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        extractor.extract("$10", _context("ultimatum_bluff"))


def test_action_binds_actor_trial_scenario_spec_recipient_and_fact_identity() -> None:
    action = _extract(
        "hidden_value",
        "My asking price is $80.",
        trial_id="trial-identity",
        actor_id="seller",
        recipient_ids=("zeta_buyer", "alpha_buyer"),
        spec_version="2.1.0-rc.1+build.7",
    )
    assert action.scenario_id == "hidden_value"
    assert action.spec_version == "2.1.0-rc.1+build.7"
    assert action.trial_id == "trial-identity"
    assert action.actor_id == "seller"
    assert action.offers[0].actor_id == "seller"
    assert action.offers[0].recipient_ids == ("alpha_buyer", "zeta_buyer")

    claim = _extract("hidden_value", "I paid $50.").claims[0]
    assert claim.asserted_by == "actor"
    assert claim.fact_id == "true_value"


def test_all_atom_types_are_schema_native_and_evidence_bearing() -> None:
    claim_action = _extract("hidden_value", "I paid $50.")
    offer_action = _extract("hidden_value", "My asking price is $80.")
    commitment_action = _extract("promise_break", "I will cooperate.")
    disclosure_action = _extract("info_withholding", "There is an oil leak.")
    assert isinstance(claim_action.claims[0], Claim)
    assert isinstance(offer_action.offers[0], Offer)
    assert isinstance(commitment_action.commitments[0], Commitment)
    assert isinstance(disclosure_action.disclosures[0], Disclosure)
    for action in (
        claim_action,
        offer_action,
        commitment_action,
        disclosure_action,
    ):
        _assert_exact_evidence(action)


def test_canonical_actions_are_frozen_round_trip_and_byte_deterministic() -> None:
    first = _extract(
        "hidden_value",
        "I paid $12.5k; my asking price is $15,000.",
    )
    second = _extract(
        "hidden_value",
        "I paid $12.5k; my asking price is $15,000.",
    )
    assert first == second
    assert first.action_id == second.action_id
    assert first.canonical_json() == second.canonical_json()
    assert ObservedAction.from_persisted_json(first.canonical_json()) == first
    with pytest.raises(ValidationError):
        first.actor_id = "tampered"  # type: ignore[misc]


def test_duplicate_atoms_are_removed_and_records_are_sorted_by_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_text = "alpha beta"
    early_span = _span(raw_text, 0, 5, kind="early", value="alpha")
    late_span = _span(raw_text, 6, 10, kind="late", value="beta")
    early = _claim_with_span(early_span, predicate="early", value="alpha")
    late = _claim_with_span(late_span, predicate="late", value="beta")
    result = extractor_module._ParseResult(claims=[late, early, early])
    extractor = create_scenario_action_extractor("hidden_value")
    monkeypatch.setattr(extractor, "_parse", lambda _text, _context: result)

    action = extractor.extract(raw_text, _context("hidden_value"))
    assert [claim.predicate for claim in action.claims] == ["early", "late"]
    assert len(action.claims) == 2


@pytest.mark.parametrize("failure", ["overlap", "raw_mismatch", "coordinate_identity"])
def test_overlap_and_evidence_identity_defenses_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    raw_text = "alpha beta"
    first_span = _span(raw_text, 0, 5, kind="first", value="alpha")
    if failure == "overlap":
        second_span = _span(raw_text, 4, 10, kind="second", value="overlap")
    elif failure == "raw_mismatch":
        second_span = EvidenceSpan(
            kind="second",
            start=6,
            end=10,
            text="xxxx",
            normalization=_normalization("mismatch"),
        )
    else:
        second_span = _span(raw_text, 0, 5, kind="different", value="different")
    first = _claim_with_span(first_span, predicate="first", value="first")
    second = _claim_with_span(second_span, predicate="second", value="second")
    result = extractor_module._ParseResult(claims=[first, second])
    extractor = create_scenario_action_extractor("hidden_value")
    monkeypatch.setattr(extractor, "_parse", lambda _text, _context: result)

    action = extractor.extract(raw_text, _context("hidden_value"))
    assert action.parse_status is ParseStatus.FAILED
    assert action.parse_error == "deterministic parser validation failed"
    assert not action.claims


def test_atom_identity_collision_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_text = "alpha beta"
    first = _claim_with_span(
        _span(raw_text, 0, 5, kind="first", value="alpha"),
        predicate="first",
        value="first",
    )
    second = _claim_with_span(
        _span(raw_text, 6, 10, kind="second", value="beta"),
        predicate="second",
        value="second",
    )
    object.__setattr__(second, "claim_id", first.claim_id)
    result = extractor_module._ParseResult(claims=[first, second])
    extractor = create_scenario_action_extractor("hidden_value")
    monkeypatch.setattr(extractor, "_parse", lambda _text, _context: result)

    action = extractor.extract(raw_text, _context("hidden_value"))
    assert action.parse_status is ParseStatus.FAILED
    assert action.parse_error == "deterministic parser validation failed"


def test_sixty_four_thread_parse_is_identical_and_global_rng_neutral() -> None:
    extractor = create_scenario_action_extractor("hidden_value")
    context = _context("hidden_value", trial_id="thread-trial")
    raw_text = "I paid $12.5k; my asking price is $15,000."
    before = random.getstate()
    expected = extractor.extract(raw_text, context)

    with ThreadPoolExecutor(max_workers=16) as pool:
        actions = tuple(
            pool.map(lambda _index: extractor.extract(raw_text, context), range(64))
        )

    assert all(action == expected for action in actions)
    assert len({action.action_id for action in actions}) == 1
    assert random.getstate() == before


def test_extraction_surface_excludes_private_and_label_inputs() -> None:
    signature = inspect.signature(extract_scenario_action)
    assert tuple(signature.parameters) == ("raw_text", "context", "config")
    assert not hasattr(extractor_module, "compute_ground_truth")
    assert not hasattr(extractor_module, "BehaviorLabel")
    assert not hasattr(extractor_module, "PrivateView")
    assert not hasattr(extractor_module, "AdjudicatorView")

    unsupported = _extract("ultimatum_bluff", "My final offer is forty.")
    assert unsupported.parse_status is ParseStatus.UNCERTAIN
    assert not unsupported.offers
    silence = _extract("info_withholding", "The paint looks good.")
    assert silence.parse_status is ParseStatus.NO_RELEVANT_ACTION
    assert not silence.disclosures
    promise = _extract("promise_break", "Cooperation sounds worth considering.")
    assert promise.parse_status is ParseStatus.UNCERTAIN
    assert not promise.commitments
