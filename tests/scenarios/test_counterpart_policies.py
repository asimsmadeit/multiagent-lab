"""Contracts for model-free controlled counterpart-policy fixtures."""

from __future__ import annotations

import json
import math
import random
from typing import Any, Callable

import pytest
from pydantic import BaseModel, ValidationError

from interpretability.scenarios.counterpart_policies import (
    ALL_POLICY_ACTIONS,
    CONTROLLED_POLICY_MAX_AMOUNT,
    CONTROLLED_POLICY_SCHEMA_VERSION,
    AbsentPolicy,
    ActionDistribution,
    AdjudicatorPolicyTruth,
    ExecutableCounterpartPolicy,
    ObservationAccess,
    PolicyAction,
    PolicyHypothesis,
    PublicObservation,
    ThresholdRationalPolicy,
    VerificationSensitivePolicy,
    VerificationStatus,
    make_adjudicator_policy_truth,
)


def _observation(
    *,
    trial_id: str = "trial-1",
    turn: int = 2,
    observer_id: str = "Buyer",
    source_actor_id: str = "Seller",
    source_event_id: str = "event-2",
    access: ObservationAccess = ObservationAccess.PUBLIC,
    fact_ids: tuple[str, ...] = ("fact-price",),
    event_ids: tuple[str, ...] = ("event-2",),
    legal_actions: tuple[PolicyAction, ...] = ALL_POLICY_ACTIONS,
    offer_amount: float | None = 80.0,
    claim_fact_id: str | None = "fact-price",
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED,
) -> PublicObservation:
    return PublicObservation(
        trial_id=trial_id,
        turn=turn,
        observer_id=observer_id,
        source_actor_id=source_actor_id,
        source_event_id=source_event_id,
        access=access,
        accessible_fact_ids=fact_ids,
        accessible_event_ids=event_ids,
        legal_actions=legal_actions,
        offer_amount=offer_amount,
        claim_fact_id=claim_fact_id,
        verification_status=verification_status,
    )


def _no_claim_observation(**changes: Any) -> PublicObservation:
    values: dict[str, Any] = {
        "fact_ids": (),
        "claim_fact_id": None,
        "verification_status": VerificationStatus.UNAVAILABLE,
    }
    values.update(changes)
    return _observation(**values)


def _threshold(
    *,
    hypothesis: PolicyHypothesis = PolicyHypothesis.DEFAULT,
    threshold: float = 70.0,
    accept_probability: float = 0.9,
    counter_probability_below_threshold: float = 0.8,
    version: str = "threshold-test-1",
) -> ThresholdRationalPolicy:
    return ThresholdRationalPolicy(
        hypothesis=hypothesis,
        threshold=threshold,
        accept_probability=accept_probability,
        counter_probability_below_threshold=counter_probability_below_threshold,
        version=version,
    )


def _verification_policy(
    hypothesis: PolicyHypothesis = PolicyHypothesis.SKEPTICAL,
    *,
    threshold: float = 70.0,
    request_probability: float = 0.8,
    version: str = "verification-test-1",
) -> VerificationSensitivePolicy:
    return VerificationSensitivePolicy(
        hypothesis=hypothesis,
        threshold=threshold,
        request_probability=request_probability,
        version=version,
    )


def _identity(record: BaseModel) -> str:
    if isinstance(record, PublicObservation):
        return record.observation_id
    if isinstance(record, ActionDistribution):
        return record.distribution_id
    if isinstance(record, AdjudicatorPolicyTruth):
        return record.truth_record_id
    if isinstance(
        record,
        (ThresholdRationalPolicy, VerificationSensitivePolicy, AbsentPolicy),
    ):
        return record.policy_id
    raise TypeError(type(record).__name__)


def _truth() -> AdjudicatorPolicyTruth:
    return make_adjudicator_policy_truth(
        _verification_policy(),
        trial_id="trial-1",
        assignment_event_id="assignment-1",
    )


@pytest.fixture(
    params=[
        (0.0, 0.0, 0.0),
        (25.0, 0.25, 0.75),
        (500_000.0, 0.5, 0.5),
        (CONTROLLED_POLICY_MAX_AMOUNT, 1.0, 1.0),
    ],
    ids=["lower-bound", "lower-transfer", "upper-transfer", "upper-bound"],
)
def threshold_transfer_policy(
    request: pytest.FixtureRequest,
) -> ThresholdRationalPolicy:
    threshold, accept, counter = request.param
    return _threshold(
        threshold=threshold,
        accept_probability=accept,
        counter_probability_below_threshold=counter,
        version=(
            f"threshold-transfer-{threshold:g}-{accept:g}-{counter:g}"
            .replace("+", "p")
        ),
    )


@pytest.fixture(
    params=[
        (PolicyHypothesis.SKEPTICAL, 0.0),
        (PolicyHypothesis.SKEPTICAL, 1.0),
        (PolicyHypothesis.CREDULOUS, 0.15),
        (PolicyHypothesis.CREDULOUS, 0.85),
    ],
    ids=["skeptical-never", "skeptical-always", "credulous-low", "credulous-high"],
)
def verification_transfer_policy(
    request: pytest.FixtureRequest,
) -> VerificationSensitivePolicy:
    hypothesis, tendency = request.param
    return _verification_policy(
        hypothesis,
        request_probability=tendency,
        version=f"verification-transfer-{hypothesis.value}-{tendency:g}",
    )


def test_policy_hypotheses_match_legacy_experiment_names() -> None:
    assert tuple(item.value for item in PolicyHypothesis) == (
        "default",
        "skeptical",
        "credulous",
        "informed",
        "absent",
    )


def test_threshold_policy_accepts_exactly_at_boundary() -> None:
    policy = _threshold(threshold=70.0)
    below = policy.next_action_distribution(
        _observation(offer_amount=69.999999)
    )
    boundary = policy.next_action_distribution(_observation(offer_amount=70.0))
    above = policy.next_action_distribution(_observation(offer_amount=70.000001))

    assert below.probability(PolicyAction.COUNTER) == pytest.approx(0.8)
    assert below.probability(PolicyAction.REJECT) == pytest.approx(0.2)
    for distribution in (boundary, above):
        assert distribution.probability(PolicyAction.ACCEPT) == pytest.approx(0.9)
        assert distribution.probability(PolicyAction.COUNTER) == pytest.approx(0.1)


def test_threshold_policy_without_offer_records_unknown_action() -> None:
    distribution = _threshold().next_action_distribution(
        _no_claim_observation(offer_amount=None)
    )

    assert distribution.probability(PolicyAction.UNKNOWN) == 1.0
    assert sum(
        distribution.probability(action)
        for action in ALL_POLICY_ACTIONS
        if action is not PolicyAction.UNKNOWN
    ) == 0.0


def test_default_and_informed_threshold_policies_share_rules_not_identity() -> None:
    default = _threshold(hypothesis=PolicyHypothesis.DEFAULT)
    informed = _threshold(hypothesis=PolicyHypothesis.INFORMED)
    observation = _observation()

    assert default.next_action_distribution(observation) == (
        informed.next_action_distribution(observation)
    )
    assert default.policy_id != informed.policy_id
    assert default.hypothesis is PolicyHypothesis.DEFAULT
    assert informed.hypothesis is PolicyHypothesis.INFORMED


def test_skeptical_unverified_claim_requests_evidence_then_counters() -> None:
    distribution = _verification_policy().next_action_distribution(
        _observation(verification_status=VerificationStatus.UNVERIFIED)
    )

    assert distribution.probability(PolicyAction.REQUEST_EVIDENCE) == 0.8
    assert distribution.probability(PolicyAction.COUNTER) == pytest.approx(0.2)
    assert distribution.probability(PolicyAction.ACCEPT) == 0.0


def test_credulous_unverified_claim_requests_less_and_accepts() -> None:
    policy = _verification_policy(
        PolicyHypothesis.CREDULOUS, request_probability=0.2
    )
    distribution = policy.next_action_distribution(
        _observation(verification_status=VerificationStatus.UNVERIFIED)
    )

    assert distribution.probability(PolicyAction.REQUEST_EVIDENCE) == 0.2
    assert distribution.probability(PolicyAction.ACCEPT) == 0.8
    assert distribution.probability(PolicyAction.COUNTER) == 0.0
    assert policy.verification_tendency == 0.2


def test_supplied_true_verification_returns_to_threshold_rule() -> None:
    policy = _verification_policy()
    above = policy.next_action_distribution(
        _observation(verification_status=VerificationStatus.VERIFIED_TRUE)
    )
    below = policy.next_action_distribution(
        _observation(
            offer_amount=60.0,
            verification_status=VerificationStatus.VERIFIED_TRUE,
        )
    )

    assert above.probability(PolicyAction.ACCEPT) == pytest.approx(0.9)
    assert above.probability(PolicyAction.REQUEST_EVIDENCE) == 0.0
    assert below.probability(PolicyAction.COUNTER) == pytest.approx(0.8)
    assert below.probability(PolicyAction.REJECT) == pytest.approx(0.2)


def test_failed_verification_rejects_with_declared_probability() -> None:
    distribution = _verification_policy().next_action_distribution(
        _observation(verification_status=VerificationStatus.VERIFIED_FALSE)
    )

    assert distribution.probability(PolicyAction.REJECT) == 0.95
    assert distribution.probability(PolicyAction.REQUEST_EVIDENCE) == pytest.approx(0.05)


def test_absent_policy_has_only_no_response_mass() -> None:
    policy = AbsentPolicy(version="absent-test-1")
    distribution = policy.next_action_distribution(_observation())

    assert policy.hypothesis is PolicyHypothesis.ABSENT
    assert policy.acceptance_threshold is None
    assert policy.verification_tendency == 0.0
    assert distribution.probability(PolicyAction.NO_RESPONSE) == 1.0
    assert all(
        distribution.probability(action) == 0.0
        for action in ALL_POLICY_ACTIONS
        if action is not PolicyAction.NO_RESPONSE
    )


def test_action_vocabulary_is_complete_canonical_and_normalized() -> None:
    distribution = _threshold().next_action_distribution(_observation())

    assert ALL_POLICY_ACTIONS == tuple(
        sorted(PolicyAction, key=lambda action: action.value)
    )
    assert distribution.actions == ALL_POLICY_ACTIONS
    assert PolicyAction.UNKNOWN in distribution.actions
    assert PolicyAction.NO_RESPONSE in distribution.actions
    assert len(distribution.actions) == len(PolicyAction)
    assert math.fsum(distribution.probabilities) == 1.0


def test_seeded_sampling_is_reproducible_and_does_not_touch_global_rng() -> None:
    policy = _verification_policy()
    observation = _observation()
    random.seed(8172)
    before = random.getstate()

    first = tuple(policy.sample_action(observation, seed=index) for index in range(100))
    after = random.getstate()
    second = tuple(policy.sample_action(observation, seed=index) for index in range(100))

    assert first == second
    assert before == after == random.getstate()


@pytest.mark.parametrize("domain", ["policy", "version", "trial", "turn"])
def test_sampling_keys_are_domain_separated(domain: str) -> None:
    distribution = _verification_policy().next_action_distribution(_observation())

    def sequence(**changes: Any) -> tuple[PolicyAction, ...]:
        values = {
            "policy_id": "policy-a",
            "policy_version": "version-a",
            "trial_id": "trial-a",
            "turn": 1,
        }
        values.update(changes)
        return tuple(
            distribution.sample(seed=seed, **values) for seed in range(100)
        )

    baseline = sequence()
    changed = {
        "policy": sequence(policy_id="policy-b"),
        "version": sequence(policy_version="version-b"),
        "trial": sequence(trial_id="trial-b"),
        "turn": sequence(turn=2),
    }[domain]
    assert changed != baseline


def test_sampling_never_returns_zero_probability_action() -> None:
    policy = _threshold()
    distribution = policy.next_action_distribution(_observation())
    support = {
        action
        for action in distribution.actions
        if distribution.probability(action) > 0.0
    }

    samples = {
        policy.sample_action(_observation(), seed=seed) for seed in range(2000)
    }
    assert samples <= support
    assert samples == support == {PolicyAction.ACCEPT, PolicyAction.COUNTER}


def test_sampling_frequency_sanity_matches_known_distribution() -> None:
    policy = _verification_policy()
    observation = _observation()
    samples = tuple(
        policy.sample_action(observation, seed=seed) for seed in range(5000)
    )
    request_fraction = samples.count(PolicyAction.REQUEST_EVIDENCE) / len(samples)

    assert request_fraction == pytest.approx(0.8, abs=0.025)


def test_public_observation_requires_explicit_fact_and_event_access() -> None:
    with pytest.raises(ValidationError, match="source_event_id must be"):
        _observation(event_ids=("event-other",))
    with pytest.raises(ValidationError, match="claim_fact_id must be"):
        _observation(fact_ids=("fact-other",))


@pytest.mark.parametrize(
    "access", [ObservationAccess.PRIVATE, ObservationAccess.ADJUDICATOR_ONLY]
)
def test_private_and_adjudicator_observations_fail_closed(
    access: ObservationAccess,
) -> None:
    with pytest.raises(ValidationError, match="cannot enter a public"):
        _observation(access=access)


@pytest.mark.parametrize(
    ("claim_fact_id", "status", "match"),
    [
        (None, VerificationStatus.UNVERIFIED, "requires an accessible claim"),
        (None, VerificationStatus.VERIFIED_TRUE, "requires an accessible claim"),
        ("fact-price", VerificationStatus.UNAVAILABLE, "must explicitly be"),
    ],
)
def test_verification_status_requires_consistent_public_claim(
    claim_fact_id: str | None,
    status: VerificationStatus,
    match: str,
) -> None:
    fact_ids = ("fact-price",) if claim_fact_id else ()
    with pytest.raises(ValidationError, match=match):
        _observation(
            fact_ids=fact_ids,
            claim_fact_id=claim_fact_id,
            verification_status=status,
        )


def test_actor_serialization_has_no_policy_truth_fields() -> None:
    observation = _observation()
    payload = observation.actor_visible_dict()
    forbidden = {
        "policy",
        "policy_id",
        "policy_version",
        "policy_hypothesis",
        "true_policy",
        "true_policy_hypothesis",
    }

    assert forbidden.isdisjoint(payload)
    assert payload["accessible_fact_ids"] == ["fact-price"]
    assert payload["accessible_event_ids"] == ["event-2"]
    assert "skeptical" not in json.dumps(payload)

    tampered = observation.model_dump()
    tampered["true_policy"] = "skeptical"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PublicObservation(**tampered)


@pytest.mark.parametrize(
    "leaked_identifier",
    [
        "fact_true_policy",
        "fact-true-policy",
        "counterpart.policy.type",
        "counterpartPolicyTruth",
        "PolicyTruth",
        "event/policy_truth/value",
    ],
)
def test_actor_visible_identifier_spellings_fail_closed(
    leaked_identifier: str,
) -> None:
    with pytest.raises(ValidationError, match="must not expose policy truth"):
        _no_claim_observation(
            source_event_id=leaked_identifier,
            event_ids=(leaked_identifier,),
        )


def test_adjudicator_truth_is_separate_round_trippable_and_actor_forbidden() -> None:
    policy = _verification_policy()
    truth = make_adjudicator_policy_truth(
        policy,
        trial_id="trial-1",
        assignment_event_id="assignment-1",
    )
    restored = AdjudicatorPolicyTruth.model_validate_json(truth.canonical_json())

    assert restored == truth
    assert restored.truth_record_id == truth.truth_record_id
    assert truth.policy_id == policy.policy_id
    assert truth.policy_content_hash == policy.content_hash()
    assert truth.true_policy_hypothesis is PolicyHypothesis.SKEPTICAL
    with pytest.raises(PermissionError, match="not actor-visible"):
        truth.actor_visible_dict()


def test_adjudicator_truth_tampering_changes_content_identity() -> None:
    truth = _truth()
    payload = truth.model_dump()
    payload["assignment_event_id"] = "assignment-2"
    tampered = AdjudicatorPolicyTruth(**payload)

    assert tampered.truth_record_id != truth.truth_record_id
    assert tampered.canonical_json() != truth.canonical_json()


def test_adjudicator_truth_requires_executable_canonical_policy() -> None:
    with pytest.raises(TypeError, match="does not satisfy"):
        make_adjudicator_policy_truth(
            object(), trial_id="trial-1", assignment_event_id="assignment-1"
        )


@pytest.mark.parametrize(
    "factory",
    [
        _observation,
        lambda: _threshold().next_action_distribution(_observation()),
        _threshold,
        _verification_policy,
        AbsentPolicy,
        _truth,
    ],
    ids=[
        "observation",
        "distribution",
        "threshold-policy",
        "verification-policy",
        "absent-policy",
        "adjudicator-truth",
    ],
)
def test_public_records_have_stable_ids_round_trip_and_are_frozen(
    factory: Callable[[], BaseModel],
) -> None:
    record = factory()
    restored = type(record).model_validate_json(record.canonical_json())

    assert restored == record
    assert _identity(restored) == _identity(record)
    mutable_field = "version" if hasattr(record, "version") else "schema_version"
    with pytest.raises(ValidationError, match="frozen"):
        setattr(record, mutable_field, "tampered")

    payload = record.model_dump()
    payload["future_field"] = "forbidden"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        type(record)(**payload)


def test_stable_ids_change_with_scientifically_relevant_content() -> None:
    first_observation = _observation(turn=1)
    second_observation = _observation(turn=2)
    first_policy = _threshold(version="threshold-a")
    second_policy = _threshold(version="threshold-b")
    first_distribution = first_policy.next_action_distribution(first_observation)
    second_distribution = first_policy.next_action_distribution(
        _observation(turn=1, offer_amount=60.0)
    )

    assert first_observation.observation_id != second_observation.observation_id
    assert first_policy.policy_id != second_policy.policy_id
    assert first_distribution.distribution_id != second_distribution.distribution_id


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"turn": True}, "valid integer"),
        ({"turn": -1}, "greater than or equal to 0"),
        ({"offer_amount": True}, "real number"),
        ({"offer_amount": -0.1}, "declared controlled-policy bounds"),
        ({"offer_amount": float("nan")}, "finite"),
        ({"offer_amount": float("inf")}, "finite"),
        ({"accessible_fact_ids": ("z", "a")}, "lexicographic"),
        ({"accessible_fact_ids": ("same", "same")}, "duplicates"),
        ({"accessible_event_ids": ("z", "a")}, "lexicographic"),
        ({"legal_actions": tuple(reversed(ALL_POLICY_ACTIONS))}, "lexicographic"),
        ({"legal_actions": (PolicyAction.UNKNOWN,)}, "no_response"),
        ({"legal_actions": (PolicyAction.NO_RESPONSE,)}, "unknown"),
    ],
)
def test_observation_rejects_invalid_bounds_and_unstable_sets(
    mutation: dict[str, Any], match: str
) -> None:
    payload = _observation().model_dump()
    payload.update(mutation)

    with pytest.raises(ValidationError, match=match):
        PublicObservation(**payload)


@pytest.mark.parametrize(
    "invalid", [True, -0.1, 1.1, float("nan"), float("inf"), float("-inf")]
)
def test_action_distribution_rejects_invalid_probabilities(invalid: Any) -> None:
    probabilities = [0.0] * len(ALL_POLICY_ACTIONS)
    probabilities[0] = invalid

    with pytest.raises(ValidationError):
        ActionDistribution(
            actions=ALL_POLICY_ACTIONS,
            probabilities=tuple(probabilities),
            legal_actions=ALL_POLICY_ACTIONS,
        )


def test_action_distribution_rejects_category_and_legality_mismatch() -> None:
    valid = _threshold().next_action_distribution(_observation())

    with pytest.raises(ValidationError, match="canonical PolicyAction vocabulary"):
        ActionDistribution(
            actions=ALL_POLICY_ACTIONS[:-1],
            probabilities=valid.probabilities[:-1],
            legal_actions=ALL_POLICY_ACTIONS,
        )
    with pytest.raises(ValidationError, match="canonical PolicyAction vocabulary"):
        ActionDistribution(
            actions=tuple(reversed(ALL_POLICY_ACTIONS)),
            probabilities=valid.probabilities,
            legal_actions=ALL_POLICY_ACTIONS,
        )
    restricted = (PolicyAction.NO_RESPONSE, PolicyAction.UNKNOWN)
    with pytest.raises(ValidationError, match="illegal action"):
        ActionDistribution(
            actions=ALL_POLICY_ACTIONS,
            probabilities=valid.probabilities,
            legal_actions=restricted,
        )
    with pytest.raises(ValidationError, match="sum to one"):
        ActionDistribution(
            actions=ALL_POLICY_ACTIONS,
            probabilities=(0.1,) * len(ALL_POLICY_ACTIONS),
            legal_actions=ALL_POLICY_ACTIONS,
        )


@pytest.mark.parametrize(
    "invalid", [True, -0.1, float("nan"), float("inf"),
                 CONTROLLED_POLICY_MAX_AMOUNT + 1]
)
def test_policy_threshold_rejects_invalid_amounts(invalid: Any) -> None:
    with pytest.raises(ValidationError):
        _threshold(threshold=invalid)
    with pytest.raises(ValidationError):
        _verification_policy(threshold=invalid)


@pytest.mark.parametrize(
    "field",
    [
        "accept_probability",
        "counter_probability_below_threshold",
    ],
)
@pytest.mark.parametrize("invalid", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_threshold_policy_rejects_invalid_probabilities(
    field: str, invalid: Any
) -> None:
    values = {field: invalid}
    with pytest.raises(ValidationError):
        _threshold(**values)


@pytest.mark.parametrize(
    "invalid", [True, -0.1, 1.1, float("nan"), float("inf")]
)
def test_verification_policy_rejects_invalid_request_probability(
    invalid: Any,
) -> None:
    with pytest.raises(ValidationError):
        _verification_policy(request_probability=invalid)


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"seed": True}, TypeError),
        ({"seed": -1}, ValueError),
        ({"turn": True}, TypeError),
        ({"turn": -1}, ValueError),
        ({"policy_id": ""}, TypeError),
        ({"policy_version": ""}, TypeError),
        ({"trial_id": ""}, TypeError),
    ],
)
def test_distribution_sampling_rejects_invalid_key_parts(
    changes: dict[str, Any], error: type[Exception]
) -> None:
    distribution = _threshold().next_action_distribution(_observation())
    values: dict[str, Any] = {
        "policy_id": "policy-1",
        "policy_version": "version-1",
        "trial_id": "trial-1",
        "turn": 1,
        "seed": 1,
    }
    values.update(changes)

    with pytest.raises(error):
        distribution.sample(**values)


def test_policies_are_model_free_and_satisfy_executable_protocol() -> None:
    policies = (
        _threshold(),
        _verification_policy(),
        AbsentPolicy(),
    )

    for policy in policies:
        assert isinstance(policy, ExecutableCounterpartPolicy)
        assert policy.observation_access == (
            ObservationAccess.PUBLIC,
            ObservationAccess.SHARED,
        )
        assert not hasattr(policy, "sample_text")
        assert not hasattr(policy, "language_model")


def test_policy_assignment_is_independent_of_actor_role_identity() -> None:
    policy = _threshold()
    seller_observes = _observation(
        observer_id="Seller",
        source_actor_id="Buyer",
    )
    buyer_observes = _observation(
        observer_id="Buyer",
        source_actor_id="Seller",
    )

    assert seller_observes.observation_id != buyer_observes.observation_id
    assert policy.policy_id == _threshold().policy_id
    assert policy.next_action_distribution(seller_observes) == (
        policy.next_action_distribution(buyer_observes)
    )


def test_threshold_transfer_parameter_fixture_is_executable(
    threshold_transfer_policy: ThresholdRationalPolicy,
) -> None:
    threshold = threshold_transfer_policy.acceptance_threshold
    observation = _no_claim_observation(offer_amount=threshold)
    distribution = threshold_transfer_policy.next_action_distribution(observation)

    assert distribution.probability(PolicyAction.ACCEPT) == (
        threshold_transfer_policy.accept_probability
    )
    assert math.fsum(distribution.probabilities) == 1.0


def test_verification_transfer_parameter_fixture_is_executable(
    verification_transfer_policy: VerificationSensitivePolicy,
) -> None:
    distribution = verification_transfer_policy.next_action_distribution(
        _observation()
    )

    assert distribution.probability(PolicyAction.REQUEST_EVIDENCE) == (
        verification_transfer_policy.request_probability
    )
    assert verification_transfer_policy.verification_tendency == (
        verification_transfer_policy.request_probability
    )
    assert math.fsum(distribution.probabilities) == 1.0


def test_schema_version_is_explicit_on_persisted_boundary_records() -> None:
    assert CONTROLLED_POLICY_SCHEMA_VERSION == "controlled-policy/1.0.0"
    assert _observation().schema_version == CONTROLLED_POLICY_SCHEMA_VERSION
    assert (
        _threshold().next_action_distribution(_observation()).schema_version
        == CONTROLLED_POLICY_SCHEMA_VERSION
    )
    assert _truth().schema_version == CONTROLLED_POLICY_SCHEMA_VERSION
