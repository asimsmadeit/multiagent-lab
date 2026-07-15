"""Permanent model-free contracts for Theory of Mind v2 belief updaters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import pytest

from negotiation.components.tom.schema import (
    BeliefDistribution,
    EpistemicStatus,
    Evidence,
    EvidenceChannel,
    EvidenceVisibility,
    GroundTruthKind,
    UpdateMethod,
)
from negotiation.components.tom.updater import (
    BayesianUpdater,
    BeliefUpdater,
    FrequencyBaselineUpdater,
    FrozenPriorUpdater,
    ObservationModel,
)


def _belief(
    probabilities: tuple[float, ...] = (0.5, 0.5),
    *,
    target: str = "policy_type",
    categories: tuple[str, ...] = ("skeptical", "unknown"),
) -> BeliefDistribution:
    return BeliefDistribution(
        target=target,
        categories=categories,
        probabilities=probabilities,
        epistemic_status=EpistemicStatus.PRIOR,
        ground_truth_kind=GroundTruthKind.OBJECTIVE,
    )


def _evidence(
    event_id: str = "event-1",
    *,
    features: tuple[tuple[str, Any], ...] = (("requested_evidence", True),),
) -> Evidence:
    return Evidence(
        observer_id="Seller",
        source_actor_id="Buyer",
        source_event_id=event_id,
        turn=1,
        features=features,
        channel=EvidenceChannel.OBSERVABLE,
        visibility=EvidenceVisibility.PUBLIC,
        reliability=1.0,
        extractor_version="rules-1",
    )


@dataclass(frozen=True)
class _TableModel:
    values: Mapping[str, Mapping[str, Any]]
    target: str = "policy_type"
    categories: Any = ("skeptical", "unknown")
    version: str = "table-model-1"

    def likelihood(self, hypothesis: str, evidence: Evidence) -> Any:
        return self.values[evidence.source_event_id][hypothesis]


@dataclass(frozen=True)
class _ConstantModel:
    values: Mapping[str, Any]
    target: str = "policy_type"
    categories: tuple[str, ...] = ("skeptical", "unknown")
    version: str = "constant-model-1"

    def likelihood(self, hypothesis: str, evidence: Evidence) -> Any:
        del evidence
        return self.values[hypothesis]


class _MissingVersionModel:
    target = "policy_type"
    categories = ("skeptical", "unknown")

    @staticmethod
    def likelihood(hypothesis: str, evidence: Evidence) -> float:
        del hypothesis, evidence
        return 0.5


def _table(
    event_id: str = "event-1",
    values: tuple[Any, Any] = (0.8, 0.2),
    **kwargs: Any,
) -> _TableModel:
    return _TableModel(
        {event_id: {"skeptical": values[0], "unknown": values[1]}},
        **kwargs,
    )


def _frequency_prior() -> BeliefDistribution:
    return _belief(
        (0.5, 0.3, 0.2),
        target="next_action",
        categories=("accept", "counter", "unknown"),
    )


def _frequency_updater(
    mapping: Mapping[str, str] | None = None,
    *,
    pseudocount: float = 10.0,
) -> FrequencyBaselineUpdater:
    return FrequencyBaselineUpdater(
        feature_name="observed_action",
        category_mapping=mapping
        or {
            "accepted": "accept",
            "countered": "counter",
            "other": "unknown",
        },
        mapping_version="action-map-1",
        pseudocount=pseudocount,
        version="frequency-test-1",
    )


def _action_evidence(event_id: str, observed: Any) -> Evidence:
    return _evidence(
        event_id,
        features=(("observed_action", observed),),
    )


def test_runtime_checkable_protocols_distinguish_valid_objects() -> None:
    model = _table()

    assert isinstance(model, ObservationModel)
    assert not isinstance(_MissingVersionModel(), ObservationModel)
    assert not isinstance(object(), ObservationModel)
    for updater in (
        BayesianUpdater(),
        FrozenPriorUpdater(),
        _frequency_updater(),
    ):
        assert isinstance(updater, BeliefUpdater)
    assert not isinstance(object(), BeliefUpdater)


def test_one_evidence_bayesian_posterior_is_exact_and_versioned() -> None:
    prior = _belief()
    evidence = _evidence()
    update = BayesianUpdater(version="bayes-test-1").update(
        prior, (evidence,), _table()
    )

    assert update.posterior.probability("skeptical") == pytest.approx(0.8)
    assert update.posterior.probability("unknown") == pytest.approx(0.2)
    assert math.fsum(update.posterior.probabilities) == 1.0
    assert update.posterior.target == prior.target
    assert update.posterior.categories == prior.categories
    assert update.posterior.unknown_category == prior.unknown_category
    assert update.posterior.ground_truth_kind is prior.ground_truth_kind
    assert update.posterior.epistemic_status is EpistemicStatus.UPDATED
    assert update.method is UpdateMethod.BAYESIAN
    assert update.updater_version == "bayes-test-1"
    assert update.observation_model_version == "table-model-1"
    assert update.prior_state_hash == prior.state_hash
    assert update.posterior_state_hash == update.posterior.state_hash
    assert update.evidence_ids == (evidence.evidence_id,)
    assert update.update_id.startswith("tom_update_")


def test_multiple_evidence_bayesian_posterior_multiplies_likelihoods() -> None:
    first = _evidence("event-1")
    second = _evidence("event-2")
    model = _TableModel(
        {
            "event-1": {"skeptical": 0.8, "unknown": 0.2},
            "event-2": {"skeptical": 0.75, "unknown": 0.25},
        }
    )

    update = BayesianUpdater().update(_belief(), (first, second), model)

    assert update.posterior.probability("skeptical") == 12 / 13
    assert update.posterior.probability("unknown") == 1 / 13
    assert update.likelihoods == pytest.approx((1.0, 1 / 12))


def test_log_space_update_resists_naive_product_underflow() -> None:
    evidence = tuple(_evidence(f"event-{index:04d}") for index in range(400))
    model = _ConstantModel({"skeptical": 1e-200, "unknown": 1e-201})

    assert math.prod([1e-200] * len(evidence)) == 0.0
    assert math.prod([1e-201] * len(evidence)) == 0.0
    update = BayesianUpdater().update(_belief(), evidence, model)

    assert update.posterior.probability("skeptical") > 1.0 - 1e-12
    assert update.posterior.probability("unknown") < 1e-12
    assert math.fsum(update.posterior.probabilities) == 1.0


def test_zero_prior_mass_cannot_be_resurrected_by_likelihood() -> None:
    prior = _belief((1.0, 0.0))
    evidence = _evidence()
    update = BayesianUpdater().update(
        prior, (evidence,), _table(values=(0.1, 0.9))
    )

    assert update.posterior.probabilities == (1.0, 0.0)

    with pytest.raises(ValueError, match="no positive posterior mass"):
        BayesianUpdater().update(
            prior, (evidence,), _table(values=(0.0, 1.0))
        )


def test_one_zero_likelihood_eliminates_only_that_hypothesis() -> None:
    update = BayesianUpdater().update(
        _belief(), (_evidence(),), _table(values=(0.0, 0.5))
    )

    assert update.posterior.probabilities == (0.0, 1.0)
    assert update.likelihoods == (0.0, 1.0)


def test_all_zero_likelihood_requires_explicit_smoothing() -> None:
    prior = _belief((0.6, 0.4))
    evidence = _evidence()
    model = _table(values=(0.0, 0.0))

    with pytest.raises(ValueError, match="all-zero likelihood vector"):
        BayesianUpdater().update(prior, (evidence,), model)

    smoothed = BayesianUpdater(smoothing=0.01).update(
        prior, (evidence,), model
    )
    assert smoothed.posterior.probabilities == pytest.approx(
        prior.probabilities
    )
    assert any(
        warning.startswith("all_zero_likelihood_smoothed:")
        for warning in smoothed.warnings
    )
    assert any(
        warning.startswith("likelihood_floor_applied:")
        for warning in smoothed.warnings
    )


def test_smoothing_floors_zero_without_masking_other_likelihood() -> None:
    update = BayesianUpdater(smoothing=0.01).update(
        _belief(), (_evidence(),), _table(values=(0.0, 0.5))
    )

    assert update.posterior.probability("skeptical") == pytest.approx(1 / 51)
    assert update.posterior.probability("unknown") == pytest.approx(50 / 51)
    assert update.likelihoods == pytest.approx((0.02, 1.0))


def test_partially_missing_likelihood_vector_is_rejected() -> None:
    with pytest.raises(ValueError, match="partially missing"):
        BayesianUpdater().update(
            _belief(), (_evidence(),), _table(values=(None, 0.5))
        )


def test_all_missing_likelihood_is_retained_as_explicit_missing_evidence() -> None:
    prior = _belief()
    evidence = _evidence()
    update = BayesianUpdater().update(
        prior, (evidence,), _table(values=(None, None))
    )

    assert update.posterior.probabilities == prior.probabilities
    assert update.likelihoods == (1.0, 1.0)
    assert update.evidence_ids == (evidence.evidence_id,)
    assert update.warnings == (f"missing_evidence:{evidence.evidence_id}",)


def test_empty_bayesian_evidence_is_explicit_and_preserves_prior() -> None:
    prior = _belief()
    update = BayesianUpdater().update(prior, (), _table())

    assert update.posterior.probabilities == prior.probabilities
    assert update.evidence == ()
    assert update.warnings == ("no_evidence",)


def test_informative_bayesian_evidence_reduces_entropy() -> None:
    update = BayesianUpdater().update(
        _belief(), (_evidence(),), _table(values=(0.99, 0.01))
    )

    assert update.posterior.entropy < update.prior.entropy
    assert update.entropy_change == pytest.approx(
        update.posterior.entropy - update.prior.entropy
    )
    assert update.entropy_change < 0.0


def test_evidence_order_is_canonical_and_update_identity_is_invariant() -> None:
    first = _evidence("event-1")
    second = _evidence("event-2")
    model = _TableModel(
        {
            "event-1": {"skeptical": 0.8, "unknown": 0.2},
            "event-2": {"skeptical": 0.4, "unknown": 0.6},
        }
    )
    updater = BayesianUpdater()

    forward = updater.update(_belief(), (first, second), model)
    reverse = updater.update(_belief(), (second, first), model)

    expected_ids = tuple(sorted((first.evidence_id, second.evidence_id)))
    assert forward.evidence_ids == reverse.evidence_ids == expected_ids
    assert forward.posterior == reverse.posterior
    assert forward.update_id == reverse.update_id


def test_duplicate_evidence_is_rejected_before_updating() -> None:
    evidence = _evidence()
    with pytest.raises(ValueError, match="duplicate records"):
        BayesianUpdater().update(_belief(), (evidence, evidence), _table())


@pytest.mark.parametrize(
    ("model", "error", "match"),
    [
        (None, ValueError, "requires an observation model"),
        (_MissingVersionModel(), TypeError, "does not satisfy"),
        (_table(target="different_target"), ValueError, "target does not match"),
        (_table(categories=("default", "unknown")), ValueError,
         "categories must exactly match"),
        (_table(categories=["skeptical", "unknown"]), TypeError,
         "categories must be a tuple"),
        (_table(version=""), TypeError, "non-empty string"),
        (_table(version=" table-1 "), ValueError, "surrounding whitespace"),
    ],
)
def test_bayesian_model_contract_mismatch_is_rejected(
    model: Any, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        BayesianUpdater().update(_belief(), (_evidence(),), model)


@pytest.mark.parametrize(
    "invalid",
    [True, False, -0.1, float("nan"), float("inf"), float("-inf"), "0.5"],
)
def test_bayesian_rejects_invalid_likelihood_values(invalid: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        BayesianUpdater().update(
            _belief(), (_evidence(),), _table(values=(invalid, 0.2))
        )


@pytest.mark.parametrize(
    "invalid", [True, False, -0.1, 1.1, float("nan"), float("inf"), "0.1"]
)
def test_bayesian_rejects_invalid_smoothing(invalid: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        BayesianUpdater(smoothing=invalid)


def test_contradictory_zero_likelihoods_reject_zero_cumulative_mass() -> None:
    first = _evidence("event-1")
    second = _evidence("event-2")
    model = _TableModel(
        {
            "event-1": {"skeptical": 1.0, "unknown": 0.0},
            "event-2": {"skeptical": 0.0, "unknown": 1.0},
        }
    )

    with pytest.raises(ValueError, match="zero cumulative likelihood"):
        BayesianUpdater().update(_belief(), (first, second), model)


def test_frozen_prior_is_invariant_and_records_ignored_evidence() -> None:
    prior = _belief((0.7, 0.3))
    evidence = _evidence()
    update = FrozenPriorUpdater(version="frozen-test-1").update(
        prior, (evidence,)
    )

    assert update.posterior.probabilities == prior.probabilities
    assert update.posterior.epistemic_status is EpistemicStatus.FROZEN
    assert update.entropy_change == 0.0
    assert update.method is UpdateMethod.FROZEN_PRIOR
    assert update.updater_version == "frozen-test-1"
    assert update.observation_model_version == "not-used-1"
    assert update.likelihoods == (1.0, 1.0)
    assert update.evidence_ids == (evidence.evidence_id,)
    assert update.warnings == ("evidence_ignored_by_frozen_prior",)


def test_frozen_prior_without_evidence_and_with_model_are_explicit() -> None:
    prior = _belief()
    empty = FrozenPriorUpdater().update(prior, ())
    with_model = FrozenPriorUpdater().update(prior, (_evidence(),), _table())

    assert empty.warnings == ("no_evidence",)
    assert empty.evidence_ids == ()
    assert with_model.observation_model_version == "table-model-1"
    with pytest.raises(ValueError, match="target does not match"):
        FrozenPriorUpdater().update(
            prior, (_evidence(),), _table(target="wrong_target")
        )


def test_frequency_prior_weighted_pseudocount_math_and_provenance() -> None:
    prior = _frequency_prior()
    evidence = (
        _action_evidence("event-a1", "accepted"),
        _action_evidence("event-a2", "accepted"),
        _action_evidence("event-c1", "countered"),
    )
    update = _frequency_updater().update(prior, evidence)

    assert update.likelihoods == (7.0, 4.0, 2.0)
    assert update.posterior.probabilities == pytest.approx((7 / 13, 4 / 13, 2 / 13))
    assert math.fsum(update.posterior.probabilities) == 1.0
    assert update.method is UpdateMethod.FREQUENCY_BASELINE
    assert update.posterior.epistemic_status is EpistemicStatus.UPDATED
    assert update.updater_version == "frequency-test-1"
    assert update.observation_model_version == "action-map-1"
    assert update.posterior.target == prior.target
    assert update.posterior.categories == prior.categories
    assert update.posterior.ground_truth_kind is prior.ground_truth_kind
    assert update.evidence_ids == tuple(sorted(item.evidence_id for item in evidence))


def test_frequency_update_is_order_invariant_and_canonical() -> None:
    evidence = (
        _action_evidence("event-a", "accepted"),
        _action_evidence("event-c", "countered"),
        _action_evidence("event-u", "other"),
    )
    updater = _frequency_updater()

    forward = updater.update(_frequency_prior(), evidence)
    reverse = updater.update(_frequency_prior(), tuple(reversed(evidence)))

    assert forward.posterior == reverse.posterior
    assert forward.evidence_ids == reverse.evidence_ids
    assert forward.update_id == reverse.update_id


def test_frequency_mapping_is_copied_and_exposed_immutably() -> None:
    source = {
        "accepted": "accept",
        "countered": "counter",
        "other": "unknown",
    }
    updater = _frequency_updater(source)
    source["accepted"] = "unknown"

    assert updater.category_mapping["accepted"] == "accept"
    with pytest.raises(TypeError):
        updater.category_mapping["accepted"] = "unknown"


def test_frequency_missing_feature_is_explicit_and_does_not_add_count() -> None:
    prior = _frequency_prior()
    evidence = _evidence("event-missing")
    update = _frequency_updater().update(prior, (evidence,))

    assert update.posterior.probabilities == prior.probabilities
    assert update.likelihoods == pytest.approx((5.0, 3.0, 2.0))
    assert update.evidence_ids == (evidence.evidence_id,)
    assert update.warnings[0].startswith("missing_feature:observed_action:")


@pytest.mark.parametrize("observed", [True, 1, 1.0])
def test_frequency_rejects_nonstring_observed_categories(observed: Any) -> None:
    with pytest.raises(TypeError, match="declared string category"):
        _frequency_updater().update(
            _frequency_prior(),
            (_action_evidence("event-invalid", observed),),
        )


def test_frequency_rejects_undeclared_observed_category() -> None:
    with pytest.raises(ValueError, match="undeclared evidence category"):
        _frequency_updater().update(
            _frequency_prior(),
            (_action_evidence("event-reject", "rejected"),),
        )


@pytest.mark.parametrize(
    "mapping",
    [
        {"accepted": "accept", "countered": "counter"},
        {
            "accepted": "accept",
            "countered": "counter",
            "other": "unknown",
            "reject": "reject",
        },
    ],
)
def test_frequency_mapping_must_cover_exact_prior_categories(
    mapping: Mapping[str, str],
) -> None:
    updater = _frequency_updater(mapping)
    with pytest.raises(ValueError, match="exactly match prior categories"):
        updater.update(_frequency_prior(), ())


@pytest.mark.parametrize(
    "invalid", [True, 0.0, -1.0, float("nan"), float("inf"), "1.0"]
)
def test_frequency_rejects_invalid_or_overflowing_pseudocount(invalid: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        _frequency_updater(pseudocount=invalid)


def test_frequency_rejects_observation_model_and_duplicate_evidence() -> None:
    evidence = _action_evidence("event-a", "accepted")
    updater = _frequency_updater()

    with pytest.raises(ValueError, match="declared mapping"):
        updater.update(_frequency_prior(), (evidence,), _table())
    with pytest.raises(ValueError, match="duplicate records"):
        updater.update(_frequency_prior(), (evidence, evidence))


def test_frequency_empty_evidence_preserves_prior_with_warning() -> None:
    prior = _frequency_prior()
    update = _frequency_updater().update(prior, ())

    assert update.posterior.probabilities == prior.probabilities
    assert update.evidence_ids == ()
    assert update.warnings == ("no_evidence",)
    assert update.prior_state_hash == prior.state_hash
    assert update.posterior_state_hash == update.posterior.state_hash
