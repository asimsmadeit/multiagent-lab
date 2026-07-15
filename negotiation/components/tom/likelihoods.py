"""Finite observation models for Theory of Mind v2 belief updating.

Exact controlled-policy tables and training-fitted tables have distinct,
validated provenance.  Fitting is deterministic, train-only, group-safe, and
contains no language-model or random-number dependency.
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
    StringConstraints,
    model_validator,
)

from negotiation.components.tom.schema import BeliefDistribution, Evidence


LIKELIHOOD_SCHEMA_VERSION = "tom-likelihood/1.0.0"
MISSING_OBSERVATION = "missing"
OTHER_OBSERVATION = "other"
_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_.:/-]*$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


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


def _sha256(value: str) -> str:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("value must be a lowercase sha256 digest")
    return value


def _finite_non_negative(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be numeric, not boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value must be finite")
    if result < 0.0:
        raise ValueError("value must be non-negative")
    return 0.0 if result == 0.0 else result


def _positive_finite(value: Any) -> float:
    result = _finite_non_negative(value)
    if result <= 0.0:
        raise ValueError("value must be positive")
    return result


def _probability(value: Any) -> float:
    result = _finite_non_negative(value)
    if result > 1.0:
        raise ValueError("probability must not exceed one")
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
Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True),
    AfterValidator(_sha256),
]
Probability = Annotated[float, BeforeValidator(_probability)]
PositiveFinite = Annotated[float, BeforeValidator(_positive_finite)]


def _canonical_unique(values: tuple[str, ...], *, field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")
    if values != tuple(sorted(values)):
        raise ValueError(
            f"{field_name} must use canonical lexicographic ordering"
        )


def _normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("probability column has no finite positive mass")
    normalized = [value / total for value in values]
    largest = max(range(len(normalized)), key=normalized.__getitem__)
    normalized[largest] += 1.0 - math.fsum(normalized)
    return tuple(normalized)


class _CanonicalModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def content_hash(self) -> str:
        material = f"{type(self).__name__}\0{self.canonical_json()}".encode()
        return f"sha256:{hashlib.sha256(material).hexdigest()}"


class LikelihoodTableOrigin(str, Enum):
    """Whether probabilities are controlled truths or training estimates."""

    EXACT = "exact"
    FITTED_TRAIN_ONLY = "fitted_train_only"


class DataPartition(str, Enum):
    """Dataset partition recorded before any likelihood estimation."""

    TRAIN = "train"
    DEVELOPMENT = "development"
    TEST = "test"


class LabeledPolicyObservation(_CanonicalModel):
    """Immutable weighted training observation with dyad/group provenance."""

    schema_version: Literal["tom-likelihood/1.0.0"] = LIKELIHOOD_SCHEMA_VERSION
    sample_id: StableIdentifier
    evidence_id: StableIdentifier
    dyad_id: StableIdentifier
    group_id: StableIdentifier
    partition: DataPartition
    hypothesis: StableCategory
    feature_name: StableCategory
    observation_value: StableCategory | None
    count: PositiveFinite = 1.0

    @property
    def row_id(self) -> str:
        return f"likelihood_row_{self.content_hash().removeprefix('sha256:')}"


class LikelihoodFitMetadata(_CanonicalModel):
    """Complete lineage proving that one table was fitted on train groups."""

    schema_version: Literal["tom-likelihood/1.0.0"] = LIKELIHOOD_SCHEMA_VERSION
    estimator_kind: Literal["training_only"] = "training_only"
    estimator_version: StableIdentifier
    training_partition: Literal["train"] = "train"
    source_dataset_hash: Sha256Digest
    source_manifest_hash: Sha256Digest
    sample_ids: tuple[StableIdentifier, ...]
    group_ids: tuple[StableIdentifier, ...]
    dyad_ids: tuple[StableIdentifier, ...]
    held_out_group_denylist: tuple[StableIdentifier, ...]
    additive_smoothing: PositiveFinite
    shrinkage: Probability
    weighted_row_count: PositiveFinite

    @model_validator(mode="after")
    def _validate_lineage(self) -> Self:
        for field_name in (
            "sample_ids",
            "group_ids",
            "dyad_ids",
            "held_out_group_denylist",
        ):
            values = getattr(self, field_name)
            _canonical_unique(values, field_name=field_name)
        if not self.sample_ids or not self.group_ids or not self.dyad_ids:
            raise ValueError("fitted metadata must retain non-empty train lineage")
        train_units = set(self.group_ids) | set(self.dyad_ids)
        overlap = train_units & set(self.held_out_group_denylist)
        if overlap:
            raise ValueError(
                "training groups overlap the held-out denylist: "
                + ", ".join(sorted(overlap))
            )
        return self

    @property
    def fit_id(self) -> str:
        return f"likelihood_fit_{self.content_hash().removeprefix('sha256:')}"


class LikelihoodTable(_CanonicalModel):
    """Finite p(observation | hypothesis) columns for one feature."""

    schema_version: Literal["tom-likelihood/1.0.0"] = LIKELIHOOD_SCHEMA_VERSION
    version: StableIdentifier
    target: StableCategory
    feature_name: StableCategory
    hypotheses: tuple[StableCategory, ...]
    observation_vocabulary: tuple[StableCategory, ...]
    columns: tuple[tuple[Probability, ...], ...]
    origin: LikelihoodTableOrigin
    fit_metadata: LikelihoodFitMetadata | None = None

    @model_validator(mode="after")
    def _validate_table(self) -> Self:
        _canonical_unique(self.hypotheses, field_name="hypotheses")
        _canonical_unique(
            self.observation_vocabulary,
            field_name="observation_vocabulary",
        )
        if len(self.hypotheses) < 2:
            raise ValueError("likelihood tables require at least two hypotheses")
        if MISSING_OBSERVATION not in self.observation_vocabulary:
            raise ValueError("observation vocabulary must include missing")
        if OTHER_OBSERVATION not in self.observation_vocabulary:
            raise ValueError("observation vocabulary must include other")
        if len(self.columns) != len(self.hypotheses):
            raise ValueError("columns must align with hypotheses")
        for hypothesis, column in zip(
            self.hypotheses, self.columns, strict=True
        ):
            if len(column) != len(self.observation_vocabulary):
                raise ValueError(
                    f"column for {hypothesis!r} must align with observations"
                )
            if not math.isclose(
                math.fsum(column), 1.0, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"column for {hypothesis!r} must sum to one"
                )
        if self.origin is LikelihoodTableOrigin.EXACT:
            if self.fit_metadata is not None:
                raise ValueError("exact tables must not carry fitted metadata")
        elif self.fit_metadata is None:
            raise ValueError("fitted tables require train-only fit metadata")
        return self

    @property
    def table_id(self) -> str:
        return f"likelihood_table_{self.content_hash().removeprefix('sha256:')}"

    def probability(self, hypothesis: str, observation: str) -> float:
        try:
            hypothesis_index = self.hypotheses.index(hypothesis)
        except ValueError as error:
            raise KeyError(hypothesis) from error
        try:
            observation_index = self.observation_vocabulary.index(observation)
        except ValueError as error:
            raise KeyError(observation) from error
        return self.columns[hypothesis_index][observation_index]


class ControlledPolicyObservationModel(_CanonicalModel):
    """ObservationModel adapter over one immutable likelihood table."""

    table: LikelihoodTable

    @property
    def version(self) -> str:
        return self.table.version

    @property
    def target(self) -> str:
        return self.table.target

    @property
    def categories(self) -> tuple[str, ...]:
        return self.table.hypotheses

    @property
    def model_id(self) -> str:
        return f"observation_model_{self.content_hash().removeprefix('sha256:')}"

    @property
    def fit_metadata(self) -> LikelihoodFitMetadata | None:
        return self.table.fit_metadata

    def likelihood(
        self, hypothesis: str, evidence: Evidence
    ) -> float | None:
        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must be an Evidence record")
        if hypothesis not in self.categories:
            raise ValueError(
                f"hypothesis {hypothesis!r} is outside the table categories"
            )
        features = dict(evidence.features)
        if self.table.feature_name not in features:
            return None
        observed = features[self.table.feature_name]
        if not isinstance(observed, str) or isinstance(observed, bool):
            raise TypeError(
                f"feature {self.table.feature_name!r} must be a string category"
            )
        category = (
            observed
            if observed in self.table.observation_vocabulary
            else OTHER_OBSERVATION
        )
        return self.table.probability(hypothesis, category)


class TrainingOnlyLikelihoodEstimator(_CanonicalModel):
    """Fit a finite table using only pre-partitioned training groups."""

    version: StableIdentifier = "training-likelihood-estimator-1"
    additive_smoothing: PositiveFinite = 1.0
    shrinkage: Probability = 0.1

    def fit(
        self,
        *,
        prior: BeliefDistribution,
        feature_name: str,
        observation_vocabulary: tuple[str, ...],
        rows: Iterable[LabeledPolicyObservation],
        source_dataset_hash: str,
        source_manifest_hash: str,
        held_out_group_denylist: tuple[str, ...],
        table_version: str,
    ) -> LikelihoodTable:
        if not isinstance(prior, BeliefDistribution):
            raise TypeError("prior must be a BeliefDistribution")
        feature = _stable_category(feature_name)
        vocabulary = tuple(observation_vocabulary)
        _canonical_unique(vocabulary, field_name="observation_vocabulary")
        if MISSING_OBSERVATION not in vocabulary:
            raise ValueError("observation vocabulary must include missing")
        if OTHER_OBSERVATION not in vocabulary:
            raise ValueError("observation vocabulary must include other")
        if any(not isinstance(value, str) for value in vocabulary):
            raise TypeError("observation vocabulary must contain strings")
        for value in vocabulary:
            _stable_category(value)
        denylist = tuple(held_out_group_denylist)
        _canonical_unique(denylist, field_name="held_out_group_denylist")
        for value in denylist:
            _stable_identifier(value)
        _sha256(source_dataset_hash)
        _sha256(source_manifest_hash)
        _stable_identifier(table_version)

        try:
            examples = tuple(rows)
        except TypeError as error:
            raise TypeError(
                "rows must be an iterable of LabeledPolicyObservation"
            ) from error
        if not examples:
            raise ValueError("likelihood estimation requires non-empty train data")
        if any(not isinstance(row, LabeledPolicyObservation) for row in examples):
            raise TypeError("rows must contain LabeledPolicyObservation records")

        sample_ids = tuple(row.sample_id for row in examples)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("training rows contain duplicate sample IDs")
        examples = tuple(sorted(examples, key=lambda row: row.sample_id))
        denyset = set(denylist)
        dyad_groups: dict[str, str] = {}
        counts = {
            hypothesis: [0.0] * len(vocabulary)
            for hypothesis in prior.categories
        }

        for row in examples:
            if row.partition is not DataPartition.TRAIN:
                raise ValueError(
                    f"row {row.sample_id!r} is not in the train partition"
                )
            if row.group_id in denyset or row.dyad_id in denyset:
                raise ValueError(
                    f"row {row.sample_id!r} overlaps a held-out group"
                )
            previous_group = dyad_groups.setdefault(row.dyad_id, row.group_id)
            if previous_group != row.group_id:
                raise ValueError(
                    f"dyad {row.dyad_id!r} spans multiple training groups"
                )
            if row.feature_name != feature:
                raise ValueError(
                    f"row {row.sample_id!r} uses an unknown feature"
                )
            if row.hypothesis not in prior.categories:
                raise ValueError(
                    f"row {row.sample_id!r} uses an unknown hypothesis"
                )
            observation = row.observation_value
            if observation is None:
                observation = MISSING_OBSERVATION
            elif observation not in vocabulary:
                observation = OTHER_OBSERVATION
            counts[row.hypothesis][vocabulary.index(observation)] += row.count

        pooled_raw = tuple(
            math.fsum(counts[hypothesis][index] for hypothesis in prior.categories)
            for index in range(len(vocabulary))
        )
        pooled = _normalize(
            tuple(value + self.additive_smoothing for value in pooled_raw)
        )
        columns: list[tuple[float, ...]] = []
        for hypothesis in prior.categories:
            additive_column = _normalize(
                tuple(
                    value + self.additive_smoothing
                    for value in counts[hypothesis]
                )
            )
            shrunk = _normalize(
                tuple(
                    (1.0 - self.shrinkage) * conditional
                    + self.shrinkage * pooled_probability
                    for conditional, pooled_probability in zip(
                        additive_column, pooled, strict=True
                    )
                )
            )
            columns.append(shrunk)

        metadata = LikelihoodFitMetadata(
            estimator_version=self.version,
            source_dataset_hash=source_dataset_hash,
            source_manifest_hash=source_manifest_hash,
            sample_ids=tuple(sorted(sample_ids)),
            group_ids=tuple(sorted({row.group_id for row in examples})),
            dyad_ids=tuple(sorted({row.dyad_id for row in examples})),
            held_out_group_denylist=denylist,
            additive_smoothing=self.additive_smoothing,
            shrinkage=self.shrinkage,
            weighted_row_count=math.fsum(row.count for row in examples),
        )
        return LikelihoodTable(
            version=table_version,
            target=prior.target,
            feature_name=feature,
            hypotheses=prior.categories,
            observation_vocabulary=vocabulary,
            columns=tuple(columns),
            origin=LikelihoodTableOrigin.FITTED_TRAIN_ONLY,
            fit_metadata=metadata,
        )


__all__ = [
    "DataPartition",
    "LIKELIHOOD_SCHEMA_VERSION",
    "LabeledPolicyObservation",
    "LikelihoodFitMetadata",
    "LikelihoodTable",
    "LikelihoodTableOrigin",
    "MISSING_OBSERVATION",
    "OTHER_OBSERVATION",
    "ControlledPolicyObservationModel",
    "TrainingOnlyLikelihoodEstimator",
]
