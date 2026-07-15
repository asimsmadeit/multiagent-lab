"""Contracts for finite and train-only Theory of Mind likelihood models."""

from __future__ import annotations

import math
from typing import Any, Callable

import pytest
from pydantic import BaseModel, ValidationError

from negotiation.components.tom.likelihoods import (
    LIKELIHOOD_SCHEMA_VERSION,
    MISSING_OBSERVATION,
    OTHER_OBSERVATION,
    ControlledPolicyObservationModel,
    DataPartition,
    LabeledPolicyObservation,
    LikelihoodFitMetadata,
    LikelihoodTable,
    LikelihoodTableOrigin,
    TrainingOnlyLikelihoodEstimator,
)
from negotiation.components.tom.schema import (
    BeliefDistribution,
    EpistemicStatus,
    Evidence,
    EvidenceChannel,
    EvidenceVisibility,
    GroundTruthKind,
    UpdateMethod,
)
from negotiation.components.tom.updater import BayesianUpdater, ObservationModel


_DATASET_HASH = "sha256:" + "a" * 64
_MANIFEST_HASH = "sha256:" + "b" * 64
_VOCABULARY = ("accept", "missing", "other", "request")


def _prior(
    *,
    target: str = "policy_type",
    categories: tuple[str, ...] = ("skeptical", "unknown"),
    probabilities: tuple[float, ...] = (0.5, 0.5),
) -> BeliefDistribution:
    return BeliefDistribution(
        target=target,
        categories=categories,
        probabilities=probabilities,
        epistemic_status=EpistemicStatus.PRIOR,
        ground_truth_kind=GroundTruthKind.OBJECTIVE,
    )


def _evidence(
    *,
    event_id: str = "event-1",
    features: tuple[tuple[str, Any], ...] = (("observed_action", "request"),),
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


def _exact_table(**changes: Any) -> LikelihoodTable:
    values: dict[str, Any] = {
        "version": "exact-table-1",
        "target": "policy_type",
        "feature_name": "observed_action",
        "hypotheses": ("skeptical", "unknown"),
        "observation_vocabulary": _VOCABULARY,
        "columns": (
            (0.1, 0.1, 0.1, 0.7),
            (0.6, 0.1, 0.2, 0.1),
        ),
        "origin": LikelihoodTableOrigin.EXACT,
    }
    values.update(changes)
    return LikelihoodTable(**values)


def _row(
    sample_id: str,
    *,
    evidence_id: str | None = None,
    dyad_id: str = "dyad-1",
    group_id: str = "group-1",
    partition: DataPartition = DataPartition.TRAIN,
    hypothesis: str = "skeptical",
    feature_name: str = "observed_action",
    observation_value: str | None = "request",
    count: float = 1.0,
) -> LabeledPolicyObservation:
    return LabeledPolicyObservation(
        sample_id=sample_id,
        evidence_id=evidence_id or f"evidence-{sample_id}",
        dyad_id=dyad_id,
        group_id=group_id,
        partition=partition,
        hypothesis=hypothesis,
        feature_name=feature_name,
        observation_value=observation_value,
        count=count,
    )


def _training_rows() -> tuple[LabeledPolicyObservation, ...]:
    return (
        _row("sample-1", observation_value="request", count=3.0),
        _row("sample-2", observation_value="accept", count=1.0),
        _row(
            "sample-3",
            dyad_id="dyad-2",
            group_id="group-2",
            hypothesis="unknown",
            observation_value="accept",
            count=3.0,
        ),
        _row(
            "sample-4",
            dyad_id="dyad-2",
            group_id="group-2",
            hypothesis="unknown",
            observation_value="request",
            count=1.0,
        ),
    )


def _estimator(
    *,
    additive_smoothing: float = 1.0,
    shrinkage: float = 0.2,
    version: str = "estimator-1",
) -> TrainingOnlyLikelihoodEstimator:
    return TrainingOnlyLikelihoodEstimator(
        version=version,
        additive_smoothing=additive_smoothing,
        shrinkage=shrinkage,
    )


def _fit(
    *,
    rows: Any = None,
    estimator: TrainingOnlyLikelihoodEstimator | None = None,
    prior: BeliefDistribution | None = None,
    feature_name: str = "observed_action",
    observation_vocabulary: tuple[str, ...] = _VOCABULARY,
    source_dataset_hash: str = _DATASET_HASH,
    source_manifest_hash: str = _MANIFEST_HASH,
    held_out_group_denylist: tuple[str, ...] = ("heldout-1",),
    table_version: str = "fitted-table-1",
) -> LikelihoodTable:
    return (estimator or _estimator()).fit(
        prior=prior or _prior(),
        feature_name=feature_name,
        observation_vocabulary=observation_vocabulary,
        rows=_training_rows() if rows is None else rows,
        source_dataset_hash=source_dataset_hash,
        source_manifest_hash=source_manifest_hash,
        held_out_group_denylist=held_out_group_denylist,
        table_version=table_version,
    )


def _identity(record: BaseModel) -> str:
    if isinstance(record, LikelihoodTable):
        return record.table_id
    if isinstance(record, ControlledPolicyObservationModel):
        return record.model_id
    if isinstance(record, LabeledPolicyObservation):
        return record.row_id
    if isinstance(record, LikelihoodFitMetadata):
        return record.fit_id
    if isinstance(record, TrainingOnlyLikelihoodEstimator):
        return record.content_hash()
    raise TypeError(type(record).__name__)


def test_exact_table_columns_are_normalized_and_addressable() -> None:
    table = _exact_table()

    assert table.hypotheses == ("skeptical", "unknown")
    assert table.observation_vocabulary == _VOCABULARY
    assert all(math.fsum(column) == 1.0 for column in table.columns)
    assert table.probability("skeptical", "request") == 0.7
    assert table.probability("unknown", "other") == 0.2
    with pytest.raises(KeyError, match="absent"):
        table.probability("absent", "request")
    with pytest.raises(KeyError, match="reject"):
        table.probability("skeptical", "reject")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"hypotheses": ("unknown", "skeptical")}, "lexicographic"),
        ({"hypotheses": ("unknown", "unknown")}, "duplicates"),
        ({"hypotheses": ("unknown",)}, "at least two"),
        ({"observation_vocabulary": tuple(reversed(_VOCABULARY))},
         "lexicographic"),
        ({"observation_vocabulary": ("accept", "missing", "missing", "other")},
         "duplicates"),
        ({"observation_vocabulary": ("accept", "other", "request")},
         "include missing"),
        ({"observation_vocabulary": ("accept", "missing", "request")},
         "include other"),
        ({"columns": ((0.1, 0.1, 0.1, 0.7),)}, "align with hypotheses"),
        ({"columns": ((0.1, 0.1, 0.8), (0.6, 0.1, 0.2, 0.1))},
         "align with observations"),
        ({"columns": ((0.1, 0.1, 0.1, 0.6), (0.6, 0.1, 0.2, 0.1))},
         "sum to one"),
    ],
)
def test_table_rejects_vocabulary_order_and_column_mismatch(
    mutation: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        _exact_table(**mutation)


@pytest.mark.parametrize(
    "invalid", [True, -0.1, 1.1, float("nan"), float("inf"), float("-inf")]
)
def test_table_rejects_invalid_probabilities(invalid: Any) -> None:
    columns = [list(column) for column in _exact_table().columns]
    columns[0][0] = invalid

    with pytest.raises(ValidationError):
        _exact_table(columns=tuple(tuple(column) for column in columns))


def test_model_distinguishes_missing_explicit_missing_and_other() -> None:
    model = ControlledPolicyObservationModel(table=_exact_table())
    genuinely_missing = _evidence(features=(("different_feature", "request"),))
    explicit_missing = _evidence(
        event_id="event-2",
        features=(("observed_action", MISSING_OBSERVATION),),
    )
    unseen = _evidence(
        event_id="event-3",
        features=(("observed_action", "novel_action"),),
    )

    assert model.likelihood("skeptical", genuinely_missing) is None
    assert model.likelihood("skeptical", explicit_missing) == 0.1
    assert model.likelihood("skeptical", unseen) == (
        model.table.probability("skeptical", OTHER_OBSERVATION)
    )


def test_model_lookup_satisfies_observation_protocol_and_rejects_bad_inputs() -> None:
    model = ControlledPolicyObservationModel(table=_exact_table())

    assert isinstance(model, ObservationModel)
    assert model.version == "exact-table-1"
    assert model.target == "policy_type"
    assert model.categories == ("skeptical", "unknown")
    assert model.fit_metadata is None
    assert model.likelihood("skeptical", _evidence()) == 0.7
    with pytest.raises(ValueError, match="outside the table categories"):
        model.likelihood("absent", _evidence())
    with pytest.raises(TypeError, match="Evidence record"):
        model.likelihood("skeptical", object())
    for value in (True, 1, 1.0):
        with pytest.raises(TypeError, match="string category"):
            model.likelihood(
                "skeptical",
                _evidence(features=(("observed_action", value),)),
            )


@pytest.mark.parametrize(
    "factory",
    [
        _exact_table,
        lambda: ControlledPolicyObservationModel(table=_exact_table()),
        lambda: _training_rows()[0],
        lambda: _fit().fit_metadata,
        _estimator,
    ],
    ids=["table", "model", "row", "fit-metadata", "estimator"],
)
def test_records_round_trip_have_stable_ids_and_are_frozen(
    factory: Callable[[], BaseModel | None],
) -> None:
    record = factory()
    assert record is not None
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


def test_content_tampering_changes_each_stable_identity() -> None:
    table = _exact_table()
    changed_table = _exact_table(version="exact-table-2")
    row = _training_rows()[0]
    row_payload = row.model_dump()
    row_payload["count"] = 2.0
    changed_row = LabeledPolicyObservation(**row_payload)
    metadata = _fit().fit_metadata
    assert metadata is not None
    metadata_payload = metadata.model_dump()
    metadata_payload["source_dataset_hash"] = "sha256:" + "c" * 64
    changed_metadata = LikelihoodFitMetadata(**metadata_payload)

    assert table.table_id != changed_table.table_id
    assert row.row_id != changed_row.row_id
    assert metadata.fit_id != changed_metadata.fit_id
    assert ControlledPolicyObservationModel(table=table).model_id != (
        ControlledPolicyObservationModel(table=changed_table).model_id
    )


def test_exact_and_fitted_provenance_cannot_masquerade_as_each_other() -> None:
    fitted = _fit()
    assert fitted.fit_metadata is not None

    with pytest.raises(ValidationError, match="must not carry fitted metadata"):
        _exact_table(fit_metadata=fitted.fit_metadata)
    with pytest.raises(ValidationError, match="require train-only fit metadata"):
        _exact_table(origin=LikelihoodTableOrigin.FITTED_TRAIN_ONLY)

    assert _exact_table().origin is LikelihoodTableOrigin.EXACT
    assert fitted.origin is LikelihoodTableOrigin.FITTED_TRAIN_ONLY
    assert fitted.fit_metadata.estimator_kind == "training_only"
    assert fitted.fit_metadata.training_partition == "train"


def test_training_row_identity_covers_partition_group_and_count() -> None:
    row = _row("sample-1")
    variants = []
    for field, value in (
        ("partition", DataPartition.DEVELOPMENT),
        ("group_id", "group-2"),
        ("dyad_id", "dyad-2"),
        ("count", 2.0),
    ):
        payload = row.model_dump()
        payload[field] = value
        variants.append(LabeledPolicyObservation(**payload))

    assert len({row.row_id, *(variant.row_id for variant in variants)}) == 5


@pytest.mark.parametrize(
    "invalid", [True, False, 0.0, -0.1, float("nan"), float("inf"), "1"]
)
def test_training_rows_reject_invalid_counts(invalid: Any) -> None:
    with pytest.raises(ValidationError):
        _row("sample-invalid", count=invalid)


def test_fit_metadata_retains_exact_declared_training_lineage() -> None:
    fitted = _fit()
    metadata = fitted.fit_metadata
    assert metadata is not None

    assert metadata.sample_ids == (
        "sample-1",
        "sample-2",
        "sample-3",
        "sample-4",
    )
    assert metadata.group_ids == ("group-1", "group-2")
    assert metadata.dyad_ids == ("dyad-1", "dyad-2")
    assert metadata.held_out_group_denylist == ("heldout-1",)
    assert metadata.source_dataset_hash == _DATASET_HASH
    assert metadata.source_manifest_hash == _MANIFEST_HASH
    assert metadata.weighted_row_count == 8.0
    assert metadata.additive_smoothing == 1.0
    assert metadata.shrinkage == 0.2


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"sample_ids": ()}, "non-empty train lineage"),
        ({"sample_ids": ("z", "a")}, "lexicographic"),
        ({"sample_ids": ("same", "same")}, "duplicates"),
        ({"group_ids": ("heldout-1",)}, "overlap"),
        ({"dyad_ids": ("heldout-1",)}, "overlap"),
        ({"held_out_group_denylist": ("z", "a")}, "lexicographic"),
        ({"source_dataset_hash": "not-a-hash"}, "sha256 digest"),
        ({"source_manifest_hash": "sha256:" + "A" * 64}, "sha256 digest"),
        ({"training_partition": "test"}, "train"),
        ({"estimator_kind": "exact"}, "training_only"),
    ],
)
def test_fit_metadata_rejects_invalid_lineage_and_hashes(
    mutation: dict[str, Any], match: str
) -> None:
    metadata = _fit().fit_metadata
    assert metadata is not None
    payload = metadata.model_dump()
    payload.update(mutation)

    with pytest.raises(ValidationError, match=match):
        LikelihoodFitMetadata(**payload)


def test_estimator_is_deterministic_under_row_order() -> None:
    rows = _training_rows()
    forward = _fit(rows=rows)
    reverse = _fit(rows=tuple(reversed(rows)))
    generator = _fit(rows=(row for row in rows))

    assert forward == reverse == generator
    assert forward.table_id == reverse.table_id == generator.table_id


def test_additive_smoothing_formula_is_exact_without_shrinkage() -> None:
    fitted = _fit(estimator=_estimator(shrinkage=0.0))

    assert fitted.columns[0] == pytest.approx((0.25, 0.125, 0.125, 0.5))
    assert fitted.columns[1] == pytest.approx((0.5, 0.125, 0.125, 0.25))
    assert all(math.fsum(column) == 1.0 for column in fitted.columns)


def test_pooled_shrinkage_formula_is_exact() -> None:
    fitted = _fit(estimator=_estimator(shrinkage=0.2))

    assert fitted.columns[0] == pytest.approx(
        (17 / 60, 7 / 60, 7 / 60, 29 / 60)
    )
    assert fitted.columns[1] == pytest.approx(
        (29 / 60, 7 / 60, 7 / 60, 17 / 60)
    )


def test_zero_count_hypothesis_uses_additive_distribution() -> None:
    rows = (
        _row("sample-1", observation_value="request", count=3.0),
        _row("sample-2", observation_value="accept", count=1.0),
    )
    fitted = _fit(rows=rows, estimator=_estimator(shrinkage=0.0))

    assert fitted.columns[0] == pytest.approx((0.25, 0.125, 0.125, 0.5))
    assert fitted.columns[1] == pytest.approx((0.25, 0.25, 0.25, 0.25))


def test_full_shrinkage_uses_pooled_training_source_for_every_hypothesis() -> None:
    fitted = _fit(estimator=_estimator(shrinkage=1.0))
    pooled = (5 / 12, 1 / 12, 1 / 12, 5 / 12)

    assert fitted.columns[0] == pytest.approx(pooled)
    assert fitted.columns[1] == pytest.approx(pooled)


@pytest.mark.parametrize("partition", [DataPartition.DEVELOPMENT, DataPartition.TEST])
def test_estimator_rejects_every_non_train_partition(
    partition: DataPartition,
) -> None:
    row = _row("sample-nontrain", partition=partition)

    with pytest.raises(ValueError, match="not in the train partition"):
        _fit(rows=(row,))


def test_estimator_rejects_duplicate_sample_ids() -> None:
    first = _row("duplicate", evidence_id="evidence-1")
    second = _row("duplicate", evidence_id="evidence-2")

    with pytest.raises(ValueError, match="duplicate sample IDs"):
        _fit(rows=(first, second))


@pytest.mark.parametrize(
    ("row", "denylist"),
    [
        (_row("sample-group", group_id="heldout-1"), ("heldout-1",)),
        (_row("sample-dyad", dyad_id="heldout-1"), ("heldout-1",)),
    ],
    ids=["group-overlap", "dyad-overlap"],
)
def test_estimator_rejects_group_or_dyad_heldout_overlap(
    row: LabeledPolicyObservation, denylist: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="overlaps a held-out group"):
        _fit(rows=(row,), held_out_group_denylist=denylist)


def test_estimator_rejects_one_dyad_spanning_multiple_groups() -> None:
    rows = (
        _row("sample-1", dyad_id="dyad-shared", group_id="group-1"),
        _row("sample-2", dyad_id="dyad-shared", group_id="group-2"),
    )

    with pytest.raises(ValueError, match="spans multiple training groups"):
        _fit(rows=rows)


@pytest.mark.parametrize(
    ("field", "invalid", "match"),
    [
        ("source_dataset_hash", "sha256:short", "sha256 digest"),
        ("source_manifest_hash", "not-a-hash", "sha256 digest"),
        ("table_version", "bad+version", "stable identifier"),
    ],
)
def test_estimator_rejects_dataset_manifest_and_table_version_mismatch(
    field: str, invalid: str, match: str
) -> None:
    values = {field: invalid}
    with pytest.raises(ValueError, match=match):
        _fit(**values)


def test_estimator_rejects_unknown_feature_and_hypothesis() -> None:
    with pytest.raises(ValueError, match="unknown feature"):
        _fit(rows=(_row("sample-feature", feature_name="other_feature"),))
    with pytest.raises(ValueError, match="unknown hypothesis"):
        _fit(rows=(_row("sample-hypothesis", hypothesis="absent"),))


def test_estimator_maps_unknown_observation_to_explicit_other() -> None:
    row = _row("sample-novel", observation_value="novel_action", count=3.0)
    fitted = _fit(rows=(row,), estimator=_estimator(shrinkage=0.0))
    other_index = fitted.observation_vocabulary.index(OTHER_OBSERVATION)

    assert fitted.columns[0][other_index] == pytest.approx(4 / 7)
    assert fitted.fit_metadata is not None
    assert fitted.fit_metadata.sample_ids == ("sample-novel",)


def test_estimator_counts_explicit_none_as_missing_category() -> None:
    row = _row("sample-missing", observation_value=None, count=3.0)
    fitted = _fit(rows=(row,), estimator=_estimator(shrinkage=0.0))
    missing_index = fitted.observation_vocabulary.index(MISSING_OBSERVATION)

    assert fitted.columns[0][missing_index] == pytest.approx(4 / 7)


def test_estimator_rejects_empty_nonrecord_and_noniterable_rows() -> None:
    with pytest.raises(ValueError, match="non-empty train data"):
        _fit(rows=())
    with pytest.raises(TypeError, match="must contain"):
        _fit(rows=(object(),))
    with pytest.raises(TypeError, match="must be an iterable"):
        _fit(rows=42)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("additive_smoothing", True),
        ("additive_smoothing", 0.0),
        ("additive_smoothing", -0.1),
        ("additive_smoothing", float("nan")),
        ("additive_smoothing", float("inf")),
        ("shrinkage", True),
        ("shrinkage", -0.1),
        ("shrinkage", 1.1),
        ("shrinkage", float("nan")),
        ("shrinkage", float("inf")),
    ],
)
def test_estimator_rejects_invalid_hyperparameters(
    field: str, invalid: Any
) -> None:
    with pytest.raises(ValidationError):
        _estimator(**{field: invalid})


@pytest.mark.parametrize(
    ("vocabulary", "match"),
    [
        (("other", "missing", "accept"), "lexicographic"),
        (("accept", "missing", "missing", "other"), "duplicates"),
        (("accept", "other"), "include missing"),
        (("accept", "missing"), "include other"),
        (("BadValue", "accept", "missing", "other"), "lowercase stable"),
    ],
)
def test_estimator_rejects_invalid_observation_vocabulary(
    vocabulary: tuple[str, ...], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _fit(observation_vocabulary=vocabulary)


def test_fitted_table_contains_only_declared_train_rows() -> None:
    declared = _training_rows()
    fitted = _fit(rows=declared)
    metadata = fitted.fit_metadata
    assert metadata is not None

    assert set(metadata.sample_ids) == {row.sample_id for row in declared}
    assert set(metadata.group_ids) == {row.group_id for row in declared}
    assert set(metadata.dyad_ids) == {row.dyad_id for row in declared}
    assert not set(metadata.group_ids) & set(metadata.held_out_group_denylist)
    assert not set(metadata.dyad_ids) & set(metadata.held_out_group_denylist)


def test_updater_integration_has_exact_posterior_and_lineage() -> None:
    prior = _prior()
    evidence = _evidence()
    model = ControlledPolicyObservationModel(table=_exact_table())
    update = BayesianUpdater().update(prior, (evidence,), model)

    assert update.posterior.probability("skeptical") == pytest.approx(0.875)
    assert update.posterior.probability("unknown") == pytest.approx(0.125)
    assert update.method is UpdateMethod.BAYESIAN
    assert update.observation_model_version == model.version
    assert update.evidence_ids == (evidence.evidence_id,)


def test_updater_integration_preserves_prior_for_genuinely_missing_feature() -> None:
    prior = _prior(probabilities=(0.6, 0.4))
    evidence = _evidence(features=(("different_feature", "request"),))
    model = ControlledPolicyObservationModel(table=_exact_table())
    update = BayesianUpdater().update(prior, (evidence,), model)

    assert update.posterior.probabilities == prior.probabilities
    assert update.likelihoods == (1.0, 1.0)
    assert update.warnings == (f"missing_evidence:{evidence.evidence_id}",)


def test_updater_rejects_observation_model_target_and_category_mismatch() -> None:
    evidence = _evidence()
    model = ControlledPolicyObservationModel(table=_exact_table())
    wrong_target = _prior(target="next_action")
    wrong_categories = _prior(
        categories=("default", "unknown"), probabilities=(0.5, 0.5)
    )

    with pytest.raises(ValueError, match="target does not match"):
        BayesianUpdater().update(wrong_target, (evidence,), model)
    with pytest.raises(ValueError, match="categories must exactly match"):
        BayesianUpdater().update(wrong_categories, (evidence,), model)


def test_likelihood_schema_version_is_explicit() -> None:
    assert LIKELIHOOD_SCHEMA_VERSION == "tom-likelihood/1.0.0"
    assert _exact_table().schema_version == LIKELIHOOD_SCHEMA_VERSION
    assert _training_rows()[0].schema_version == LIKELIHOOD_SCHEMA_VERSION
    metadata = _fit().fit_metadata
    assert metadata is not None
    assert metadata.schema_version == LIKELIHOOD_SCHEMA_VERSION
