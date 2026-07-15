"""Deterministic belief updaters for Theory of Mind v2.

The implementations in this module are model-free.  They consume only declared
observation likelihoods or categorical evidence and emit auditable schema
records; no random source or language model is consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Iterable, Mapping, Protocol, runtime_checkable
import unicodedata

from negotiation.components.tom.schema import (
    BeliefDistribution,
    BeliefUpdate,
    EpistemicStatus,
    Evidence,
    UpdateMethod,
)


_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_.:/-]*$")


def _require_version(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not have surrounding whitespace")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{field_name} must use canonical NFC Unicode")
    return value


def _require_feature_name(value: str) -> str:
    _require_version(value, field_name="feature_name")
    if not _CATEGORY_PATTERN.fullmatch(value):
        raise ValueError("feature_name must be a stable lowercase category")
    return value


def _finite_non_negative(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric, not boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    if result < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return 0.0 if result == 0.0 else result


def _canonical_evidence(evidence: Iterable[Evidence]) -> tuple[Evidence, ...]:
    try:
        items = tuple(evidence)
    except TypeError as error:
        raise TypeError("evidence must be an iterable of Evidence records") from error
    if any(not isinstance(item, Evidence) for item in items):
        raise TypeError("evidence must contain only Evidence records")
    evidence_ids = tuple(item.evidence_id for item in items)
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("evidence must not contain duplicate records")
    return tuple(sorted(items, key=lambda item: item.evidence_id))


def _posterior_distribution(
    prior: BeliefDistribution,
    probabilities: tuple[float, ...],
    *,
    status: EpistemicStatus,
) -> BeliefDistribution:
    return BeliefDistribution(
        target=prior.target,
        categories=prior.categories,
        probabilities=probabilities,
        unknown_category=prior.unknown_category,
        epistemic_status=status,
        ground_truth_kind=prior.ground_truth_kind,
    )


def _normalize(values: tuple[float, ...], *, context: str) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{context} must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{context} must contain finite non-negative values")
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{context} has no positive mass")
    normalized = [value / total for value in values]
    largest = max(range(len(normalized)), key=normalized.__getitem__)
    normalized[largest] += 1.0 - math.fsum(normalized)
    return tuple(normalized)


@runtime_checkable
class ObservationModel(Protocol):
    """Finite observation likelihood contract for a declared belief target."""

    version: str
    target: str
    categories: tuple[str, ...]

    def likelihood(
        self, hypothesis: str, evidence: Evidence
    ) -> float | int | None:
        """Return p(evidence | hypothesis), or ``None`` when unavailable."""


@runtime_checkable
class BeliefUpdater(Protocol):
    """Common update interface for dynamic and comparison conditions."""

    version: str

    def update(
        self,
        prior: BeliefDistribution,
        evidence: Iterable[Evidence],
        observation_model: ObservationModel | None = None,
    ) -> BeliefUpdate:
        """Return an immutable, evidence-linked update record."""


def _validate_observation_model(
    prior: BeliefDistribution, model: ObservationModel | None
) -> ObservationModel:
    if model is None:
        raise ValueError("BayesianUpdater requires an observation model")
    if not isinstance(model, ObservationModel):
        raise TypeError("observation_model does not satisfy ObservationModel")
    _require_version(model.version, field_name="observation_model.version")
    if model.target != prior.target:
        raise ValueError("observation model target does not match prior target")
    if not isinstance(model.categories, tuple):
        raise TypeError("observation model categories must be a tuple")
    if model.categories != prior.categories:
        raise ValueError(
            "observation model categories must exactly match prior categories"
        )
    return model


@dataclass(frozen=True, slots=True)
class BayesianUpdater:
    """Accumulate independent evidence likelihoods in log space."""

    smoothing: float = 0.0
    version: str = "bayesian-logspace-1"

    def __post_init__(self) -> None:
        smoothing = _finite_non_negative(
            self.smoothing, field_name="smoothing"
        )
        if smoothing > 1.0:
            raise ValueError("smoothing must not exceed one")
        _require_version(self.version, field_name="version")
        object.__setattr__(self, "smoothing", smoothing)

    def update(
        self,
        prior: BeliefDistribution,
        evidence: Iterable[Evidence],
        observation_model: ObservationModel | None = None,
    ) -> BeliefUpdate:
        if not isinstance(prior, BeliefDistribution):
            raise TypeError("prior must be a BeliefDistribution")
        model = _validate_observation_model(prior, observation_model)
        items = _canonical_evidence(evidence)
        cumulative_log_likelihood = [0.0] * len(prior.categories)
        warnings: list[str] = []
        used_evidence = 0

        if not items:
            warnings.append("no_evidence")

        for item in items:
            raw_values = tuple(
                model.likelihood(hypothesis, item)
                for hypothesis in prior.categories
            )
            missing = tuple(value is None for value in raw_values)
            if all(missing):
                warnings.append(f"missing_evidence:{item.evidence_id}")
                continue
            if any(missing):
                raise ValueError(
                    "observation model returned a partially missing likelihood "
                    f"vector for {item.evidence_id}"
                )

            likelihoods = tuple(
                _finite_non_negative(value, field_name="likelihood")
                for value in raw_values
            )
            if not any(likelihoods):
                if self.smoothing == 0.0:
                    raise ValueError(
                        "observation model returned an all-zero likelihood vector"
                    )
                warnings.append(
                    f"all_zero_likelihood_smoothed:{item.evidence_id}"
                )
            if self.smoothing > 0.0 and any(
                value < self.smoothing for value in likelihoods
            ):
                warnings.append(f"likelihood_floor_applied:{item.evidence_id}")

            effective = tuple(
                max(value, self.smoothing) for value in likelihoods
            )
            for index, value in enumerate(effective):
                if value == 0.0:
                    cumulative_log_likelihood[index] = -math.inf
                elif cumulative_log_likelihood[index] != -math.inf:
                    cumulative_log_likelihood[index] += math.log(value)
            used_evidence += 1

        if used_evidence == 0:
            stabilized_likelihoods = (1.0,) * len(prior.categories)
        else:
            finite_logs = [
                value for value in cumulative_log_likelihood if math.isfinite(value)
            ]
            if not finite_logs:
                raise ValueError(
                    "evidence gives every hypothesis zero cumulative likelihood"
                )
            maximum_log_likelihood = max(finite_logs)
            stabilized_likelihoods = tuple(
                0.0
                if value == -math.inf
                else math.exp(value - maximum_log_likelihood)
                for value in cumulative_log_likelihood
            )

        log_weights = tuple(
            -math.inf
            if probability == 0.0 or log_likelihood == -math.inf
            else math.log(probability) + log_likelihood
            for probability, log_likelihood in zip(
                prior.probabilities, cumulative_log_likelihood, strict=True
            )
        )
        finite_weights = [value for value in log_weights if math.isfinite(value)]
        if not finite_weights:
            raise ValueError("evidence gives the prior no positive posterior mass")
        maximum_log_weight = max(finite_weights)
        posterior_weights = tuple(
            0.0
            if value == -math.inf
            else math.exp(value - maximum_log_weight)
            for value in log_weights
        )
        posterior = _posterior_distribution(
            prior,
            _normalize(posterior_weights, context="posterior weights"),
            status=EpistemicStatus.UPDATED,
        )
        return BeliefUpdate(
            prior=prior,
            evidence=items,
            likelihoods=stabilized_likelihoods,
            posterior=posterior,
            method=UpdateMethod.BAYESIAN,
            updater_version=self.version,
            observation_model_version=model.version,
            warnings=tuple(sorted(set(warnings))),
        )


@dataclass(frozen=True, slots=True)
class FrozenPriorUpdater:
    """Comparison condition that records evidence without using it."""

    version: str = "frozen-prior-1"
    observation_model_version: str = "not-used-1"

    def __post_init__(self) -> None:
        _require_version(self.version, field_name="version")
        _require_version(
            self.observation_model_version,
            field_name="observation_model_version",
        )

    def update(
        self,
        prior: BeliefDistribution,
        evidence: Iterable[Evidence],
        observation_model: ObservationModel | None = None,
    ) -> BeliefUpdate:
        if not isinstance(prior, BeliefDistribution):
            raise TypeError("prior must be a BeliefDistribution")
        items = _canonical_evidence(evidence)
        model_version = self.observation_model_version
        if observation_model is not None:
            model = _validate_observation_model(prior, observation_model)
            model_version = model.version
        posterior = _posterior_distribution(
            prior, prior.probabilities, status=EpistemicStatus.FROZEN
        )
        warning = (
            "evidence_ignored_by_frozen_prior" if items else "no_evidence"
        )
        return BeliefUpdate(
            prior=prior,
            evidence=items,
            likelihoods=(1.0,) * len(prior.categories),
            posterior=posterior,
            method=UpdateMethod.FROZEN_PRIOR,
            updater_version=self.version,
            observation_model_version=model_version,
            warnings=(warning,),
        )


@dataclass(frozen=True, slots=True, init=False)
class FrequencyBaselineUpdater:
    """Update empirical category counts from one declared evidence feature."""

    feature_name: str
    mapping_version: str
    pseudocount: float
    version: str
    _mapping_items: tuple[tuple[str, str], ...]

    def __init__(
        self,
        *,
        feature_name: str,
        category_mapping: Mapping[str, str],
        mapping_version: str,
        pseudocount: float = 1.0,
        version: str = "frequency-baseline-1",
    ) -> None:
        _require_feature_name(feature_name)
        _require_version(mapping_version, field_name="mapping_version")
        _require_version(version, field_name="version")
        count = _finite_non_negative(pseudocount, field_name="pseudocount")
        if count == 0.0:
            raise ValueError("pseudocount must be positive")
        if not isinstance(category_mapping, Mapping) or not category_mapping:
            raise TypeError("category_mapping must be a non-empty mapping")

        items: list[tuple[str, str]] = []
        for observed, category in category_mapping.items():
            _require_version(observed, field_name="observed category")
            if not isinstance(category, str) or not _CATEGORY_PATTERN.fullmatch(
                category
            ):
                raise ValueError(
                    "mapped categories must be stable lowercase identifiers"
                )
            items.append((observed, category))
        items.sort()

        object.__setattr__(self, "feature_name", feature_name)
        object.__setattr__(self, "mapping_version", mapping_version)
        object.__setattr__(self, "pseudocount", count)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "_mapping_items", tuple(items))

    @property
    def category_mapping(self) -> Mapping[str, str]:
        """Return an immutable copy of the declared observation mapping."""
        return MappingProxyType(dict(self._mapping_items))

    def update(
        self,
        prior: BeliefDistribution,
        evidence: Iterable[Evidence],
        observation_model: ObservationModel | None = None,
    ) -> BeliefUpdate:
        if not isinstance(prior, BeliefDistribution):
            raise TypeError("prior must be a BeliefDistribution")
        if observation_model is not None:
            raise ValueError(
                "FrequencyBaselineUpdater uses its declared mapping, not an "
                "observation model"
            )
        mapped_categories = {category for _, category in self._mapping_items}
        if mapped_categories != set(prior.categories):
            raise ValueError(
                "category_mapping outputs must exactly match prior categories"
            )

        items = _canonical_evidence(evidence)
        mapping = dict(self._mapping_items)
        counts = [
            self.pseudocount * probability
            for probability in prior.probabilities
        ]
        warnings: list[str] = []
        if not items:
            warnings.append("no_evidence")

        for item in items:
            features = dict(item.features)
            if self.feature_name not in features:
                warnings.append(
                    f"missing_feature:{self.feature_name}:{item.evidence_id}"
                )
                continue
            observed = features[self.feature_name]
            if not isinstance(observed, str) or isinstance(observed, bool):
                raise TypeError(
                    f"feature {self.feature_name!r} must contain a declared "
                    "string category"
                )
            if observed not in mapping:
                raise ValueError(
                    f"undeclared evidence category {observed!r} for feature "
                    f"{self.feature_name!r}"
                )
            category = mapping[observed]
            counts[prior.categories.index(category)] += 1.0

        if any(not math.isfinite(value) for value in counts):
            raise ValueError("frequency counts overflowed to a non-finite value")
        probabilities = _normalize(tuple(counts), context="frequency counts")
        posterior = _posterior_distribution(
            prior, probabilities, status=EpistemicStatus.UPDATED
        )
        return BeliefUpdate(
            prior=prior,
            evidence=items,
            likelihoods=tuple(counts),
            posterior=posterior,
            method=UpdateMethod.FREQUENCY_BASELINE,
            updater_version=self.version,
            observation_model_version=self.mapping_version,
            warnings=tuple(sorted(set(warnings))),
        )


__all__ = [
    "BayesianUpdater",
    "BeliefUpdater",
    "FrequencyBaselineUpdater",
    "FrozenPriorUpdater",
    "ObservationModel",
]
