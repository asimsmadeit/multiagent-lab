"""Information-safe publication records for Theory of Mind instrumentation.

This module is deliberately owned by the negotiation layer.  It exposes only
strict JSON-domain records and dictionaries; the event layer may wrap those
values without creating a dependency from negotiation to interpretability.
Raw observations, prompts, outputs, private values, and hidden reasoning are
never retained here.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
import hashlib
import json
import math
import re
from typing import Annotated, Any, Literal, Self
import unicodedata

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    model_validator,
)

from negotiation.components.tom.policy import (
    ActionRecommendationScore,
    PolicyInterventionCondition,
    PolicyRecommendationResult,
    PolicyRequest,
)
from negotiation.components.tom.recursion import (
    AccessBasis,
    AuthorizedEvidenceUse,
    EvidenceUseKind,
    RecursiveBeliefResult,
    RecursiveTargetKind,
)
from negotiation.components.tom.schema import (
    BeliefDistribution,
    BeliefUpdate,
    EpistemicStatus,
    Evidence,
    EvidenceChannel,
    EvidenceVisibility,
    GroundTruthKind,
    PartnerBeliefState,
    ToMDecisionTrace,
    UpdateMethod,
)


INSTRUMENTATION_SCHEMA_VERSION = "tom-instrumentation/1.0.0"
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_EVENT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_SENSITIVE_TOKENS = frozenset(
    {
        "access_token",
        "actual_deception",
        "adjudicator",
        "api_key",
        "apikey",
        "authorization",
        "chain_of_thought",
        "cot",
        "credential",
        "credentials",
        "deception",
        "deception_label",
        "deceptive",
        "ground_truth",
        "ground_truth_deception",
        "hidden_reasoning",
        "oracle",
        "output_text",
        "password",
        "private",
        "private_fact",
        "private_prompt",
        "prompt_text",
        "raw_observation",
        "raw_output",
        "raw_prompt",
        "raw_text",
        "reasoning",
        "reasoning_trace",
        "secret",
        "self_report",
    }
)


def _stable_identifier(value: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError("value must be an event-safe stable identifier")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("identifiers must use canonical NFC Unicode")
    return value


def _stable_category(value: str) -> str:
    if not isinstance(value, str) or not _CATEGORY_PATTERN.fullmatch(value):
        raise ValueError("value must be a lowercase stable category")
    _reject_sensitive_channel(value, field_name="category")
    return value


def _event_id(value: str) -> str:
    if not isinstance(value, str) or not _EVENT_ID_PATTERN.fullmatch(value):
        raise ValueError("event IDs must be lowercase canonical UUID strings")
    return value


def _sha256(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("value must be a prefixed lowercase SHA-256 digest")
    return value


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be numeric, not boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value must be finite")
    return 0.0 if result == 0.0 else result


def _non_negative_number(value: Any) -> float:
    result = _finite_number(value)
    if result < 0.0:
        raise ValueError("value must be non-negative")
    return result


def _probability(value: Any) -> float:
    result = _non_negative_number(value)
    if result > 1.0:
        raise ValueError("probability must not exceed one")
    return result


def _normalized_tokens(value: str) -> frozenset[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    tokens = normalized.split("_") if normalized else []
    compounds = {
        "_".join(tokens[index : index + width])
        for width in (2, 3)
        for index in range(max(0, len(tokens) - width + 1))
    }
    return frozenset((*tokens, *compounds))


def _reject_sensitive_channel(value: str, *, field_name: str) -> None:
    if _normalized_tokens(value) & _SENSITIVE_TOKENS:
        raise ValueError(f"{field_name} names a prohibited information channel")


def _canonical_unique(values: tuple[str, ...], *, field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")
    if values != tuple(sorted(values)):
        raise ValueError(
            f"{field_name} must use canonical lexicographic ordering"
        )


def _canonical_pair_keys(
    values: tuple[tuple[str, Any], ...], *, field_name: str
) -> None:
    _canonical_unique(
        tuple(key for key, _ in values), field_name=field_name
    )


def _hash_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _hash_json(value: Any) -> str:
    material = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(material).hexdigest()}"


def _derived_id(digest: str, *, prefix: str) -> str:
    return f"{prefix}{digest.removeprefix('sha256:')}"


def _assert_derived_id(
    value: str, digest: str, *, prefix: str, field_name: str
) -> None:
    if value != _derived_id(digest, prefix=prefix):
        raise ValueError(f"{field_name} does not match its content hash")


StableIdentifier = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=256),
    AfterValidator(_stable_identifier),
]
SafeCategory = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=128),
    AfterValidator(_stable_category),
]
CanonicalEventId = Annotated[
    str,
    StringConstraints(strict=True),
    AfterValidator(_event_id),
]
Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True),
    AfterValidator(_sha256),
]
FiniteNumber = Annotated[float, BeforeValidator(_finite_number)]
NonNegativeNumber = Annotated[float, BeforeValidator(_non_negative_number)]
Probability = Annotated[float, BeforeValidator(_probability)]
SummaryScalar = StrictBool | StrictInt | FiniteNumber


class _CanonicalModel(BaseModel):
    """Strict immutable base with deterministic scientific identity."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    def content_hash(self) -> str:
        material = f"{type(self).__name__}\0{self.canonical_json()}".encode()
        return f"sha256:{hashlib.sha256(material).hexdigest()}"

    @property
    def record_id(self) -> str:
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(self).__name__).lower()
        return _derived_id(self.content_hash(), prefix=f"{name}_")

    def redacted_export(self) -> dict[str, Any]:
        """Return an isolated JSON projection without auxiliary summaries."""
        payload = _remove_summaries(self.canonical_dict())
        payload["visibility"] = PublicationVisibility.REDACTED_EXPORT.value
        payload["audience"] = ["research_export"]
        payload["publication_hash"] = self.content_hash()
        payload["publication_id"] = self.record_id
        return payload


def _remove_summaries(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"summary", "context_summary"}:
                continue
            if key == "visibility" and item == PublicationVisibility.RESTRICTED.value:
                result[key] = PublicationVisibility.REDACTED_EXPORT.value
            elif key == "audience":
                result[key] = ["research_export"]
            else:
                result[key] = _remove_summaries(item)
        return result
    if isinstance(value, list):
        return [_remove_summaries(item) for item in value]
    return value


class PublicationVisibility(str, Enum):
    """Visibility of a primary record or its public research projection."""

    RESTRICTED = "restricted"
    REDACTED_EXPORT = "redacted_export"


class DecisionPublicationPhase(str, Enum):
    """Whether a policy trace has been bound to the acting model call."""

    PRE_ACTION = "pre_action"
    POST_ACTION = "post_action"


class StructuredSummary(_CanonicalModel):
    """Small numeric summary with no free-form text channel."""

    schema_version: Literal["tom-instrumentation/1.0.0"] = (
        INSTRUMENTATION_SCHEMA_VERSION
    )
    entries: Annotated[
        tuple[tuple[SafeCategory, SummaryScalar], ...],
        Field(min_length=1, max_length=16),
    ]

    @model_validator(mode="after")
    def _validate_summary(self) -> Self:
        _canonical_pair_keys(self.entries, field_name="summary entries")
        if len(self.canonical_json()) > 512:
            raise ValueError("structured summary exceeds the 512-character bound")
        return self

    @property
    def summary_hash(self) -> str:
        return self.content_hash()


class ScopedModelCall(_CanonicalModel):
    """Explicit call scope; purpose and event validation remain event concerns."""

    schema_version: Literal["tom-instrumentation/1.0.0"] = (
        INSTRUMENTATION_SCHEMA_VERSION
    )
    trial_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    actor_id: StableIdentifier
    call_id: StableIdentifier

    @model_validator(mode="after")
    def _validate_call(self) -> Self:
        _reject_sensitive_channel(self.trial_id, field_name="call trial")
        _reject_sensitive_channel(self.actor_id, field_name="call actor")
        _reject_sensitive_channel(self.call_id, field_name="call ID")
        return self


class EvidenceEventLink(_CanonicalModel):
    """Explicit mapping from a ToM evidence identity to an event UUID."""

    schema_version: Literal["tom-instrumentation/1.0.0"] = (
        INSTRUMENTATION_SCHEMA_VERSION
    )
    trial_id: StableIdentifier
    through_turn: Annotated[StrictInt, Field(ge=0)]
    actor_id: StableIdentifier
    counterpart_id: StableIdentifier
    evidence_id: StableIdentifier
    evidence_source_event_id: StableIdentifier
    evidence_event_id: CanonicalEventId

    @model_validator(mode="after")
    def _validate_link(self) -> Self:
        if self.actor_id == self.counterpart_id:
            raise ValueError("actor and counterpart must be distinct")
        for field_name in (
            "trial_id",
            "actor_id",
            "counterpart_id",
            "evidence_id",
            "evidence_source_event_id",
        ):
            _reject_sensitive_channel(
                getattr(self, field_name), field_name=field_name
            )
        return self

    @classmethod
    def bind(
        cls,
        *,
        trial_id: str,
        through_turn: int,
        actor_id: str,
        counterpart_id: str,
        evidence: Evidence,
        evidence_event_id: str,
    ) -> Self:
        return cls(
            trial_id=trial_id,
            through_turn=through_turn,
            actor_id=actor_id,
            counterpart_id=counterpart_id,
            evidence_id=evidence.evidence_id,
            evidence_source_event_id=evidence.source_event_id,
            evidence_event_id=evidence_event_id,
        )


class DistributionSnapshot(_CanonicalModel):
    """Complete, reconstructable probability distribution plus entropy."""

    schema_version: Literal["tom-instrumentation/1.0.0"] = (
        INSTRUMENTATION_SCHEMA_VERSION
    )
    distribution_schema_version: Literal["2.0.0"]
    target: SafeCategory
    categories: Annotated[tuple[SafeCategory, ...], Field(min_length=2)]
    probabilities: Annotated[tuple[Probability, ...], Field(min_length=2)]
    unknown_category: SafeCategory
    epistemic_status: EpistemicStatus
    ground_truth_kind: GroundTruthKind
    distribution_hash: Sha256Digest
    entropy: NonNegativeNumber

    @model_validator(mode="after")
    def _validate_distribution(self) -> Self:
        if self.epistemic_status is EpistemicStatus.ORACLE:
            raise ValueError("oracle distributions cannot be instrumented")
        _canonical_unique(self.categories, field_name="distribution categories")
        reconstructed = BeliefDistribution(
            schema_version=self.distribution_schema_version,
            target=self.target,
            categories=self.categories,
            probabilities=self.probabilities,
            unknown_category=self.unknown_category,
            epistemic_status=self.epistemic_status,
            ground_truth_kind=self.ground_truth_kind,
        )
        if reconstructed.state_hash != self.distribution_hash:
            raise ValueError("distribution hash does not match distribution")
        if not math.isclose(
            reconstructed.entropy,
            self.entropy,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("distribution entropy does not match probabilities")
        return self

    @classmethod
    def from_distribution(cls, value: BeliefDistribution) -> Self:
        _validate_distribution_input(value)
        return cls(
            distribution_schema_version=value.schema_version,
            target=value.target,
            categories=value.categories,
            probabilities=value.probabilities,
            unknown_category=value.unknown_category,
            epistemic_status=value.epistemic_status,
            ground_truth_kind=value.ground_truth_kind,
            distribution_hash=value.state_hash,
            entropy=value.entropy,
        )


def _validate_distribution_input(value: BeliefDistribution) -> None:
    if not isinstance(value, BeliefDistribution):
        raise TypeError("distribution must be a BeliefDistribution")
    if value.epistemic_status is EpistemicStatus.ORACLE:
        raise ValueError("oracle distributions cannot be instrumented")
    for category in (value.target, value.unknown_category, *value.categories):
        _stable_category(category)


class EvidenceLineage(_CanonicalModel):
    """Evidence provenance without feature values, text, spans, or summaries."""

    schema_version: Literal["tom-instrumentation/1.0.0"] = (
        INSTRUMENTATION_SCHEMA_VERSION
    )
    trial_id: StableIdentifier
    publication_turn: Annotated[StrictInt, Field(ge=0)]
    actor_id: StableIdentifier
    counterpart_id: StableIdentifier
    evidence_id: StableIdentifier
    evidence_hash: Sha256Digest
    evidence_event_id: CanonicalEventId
    source_event_id: StableIdentifier
    source_call_id: StableIdentifier | None
    source_actor_id: StableIdentifier
    observer_id: StableIdentifier
    evidence_turn: Annotated[StrictInt, Field(ge=0)]
    feature_names: Annotated[tuple[SafeCategory, ...], Field(min_length=1)]
    channel: EvidenceChannel
    visibility: Literal[EvidenceVisibility.PUBLIC]
    reliability: Probability
    extractor_version: StableIdentifier
    source_text_hash: Sha256Digest | None
    parent_evidence_ids: tuple[StableIdentifier, ...]

    @model_validator(mode="after")
    def _validate_evidence(self) -> Self:
        if self.actor_id == self.counterpart_id:
            raise ValueError("actor and counterpart must be distinct")
        if self.evidence_turn > self.publication_turn:
            raise ValueError("future-turn evidence cannot be published")
        _canonical_unique(self.feature_names, field_name="feature_names")
        _canonical_unique(
            self.parent_evidence_ids, field_name="parent_evidence_ids"
        )
        _assert_derived_id(
            self.evidence_id,
            self.evidence_hash,
            prefix="evidence_",
            field_name="evidence_id",
        )
        for field_name in (
            "actor_id",
            "counterpart_id",
            "source_event_id",
            "source_actor_id",
            "observer_id",
            "extractor_version",
        ):
            _reject_sensitive_channel(
                getattr(self, field_name), field_name=field_name
            )
        if self.source_call_id is not None:
            _reject_sensitive_channel(self.source_call_id, field_name="source_call_id")
        for item in self.parent_evidence_ids:
            _reject_sensitive_channel(item, field_name="parent_evidence_ids")
        return self

    @classmethod
    def from_evidence(
        cls,
        *,
        trial_id: str,
        publication_turn: int,
        actor_id: str,
        counterpart_id: str,
        evidence: Evidence,
        event_link: EvidenceEventLink,
    ) -> Self:
        _validate_evidence_input(evidence)
        expected_scope = (
            trial_id,
            publication_turn,
            actor_id,
            counterpart_id,
            evidence.evidence_id,
            evidence.source_event_id,
        )
        actual_scope = (
            event_link.trial_id,
            event_link.through_turn,
            event_link.actor_id,
            event_link.counterpart_id,
            event_link.evidence_id,
            event_link.evidence_source_event_id,
        )
        if actual_scope != expected_scope:
            raise ValueError("evidence event link does not match evidence scope")
        return cls(
            trial_id=trial_id,
            publication_turn=publication_turn,
            actor_id=actor_id,
            counterpart_id=counterpart_id,
            evidence_id=evidence.evidence_id,
            evidence_hash=evidence.content_hash(),
            evidence_event_id=event_link.evidence_event_id,
            source_event_id=evidence.source_event_id,
            source_call_id=evidence.source_call_id,
            source_actor_id=evidence.source_actor_id,
            observer_id=evidence.observer_id,
            evidence_turn=evidence.turn,
            feature_names=tuple(name for name, _ in evidence.features),
            channel=evidence.channel,
            visibility=evidence.visibility,
            reliability=evidence.reliability,
            extractor_version=evidence.extractor_version,
            source_text_hash=evidence.source_text_hash,
            parent_evidence_ids=evidence.parent_evidence_ids,
        )


def _validate_evidence_input(value: Evidence) -> None:
    if not isinstance(value, Evidence):
        raise TypeError("evidence must be an Evidence record")
    if value.visibility is not EvidenceVisibility.PUBLIC:
        raise ValueError("private or adjudicator evidence cannot be published")
    for feature_name, _ in value.features:
        _stable_category(feature_name)
    for identifier in (
        value.observer_id,
        value.source_actor_id,
        value.source_event_id,
        value.extractor_version,
        *value.parent_evidence_ids,
    ):
        _stable_identifier(identifier)
        _reject_sensitive_channel(identifier, field_name="evidence provenance")
    if value.source_call_id is not None:
        _stable_identifier(value.source_call_id)
        _reject_sensitive_channel(value.source_call_id, field_name="source call")


class BeliefUpdateSnapshot(_CanonicalModel):
    """Exact update linkage with no evidence values or warning text."""

    schema_version: Literal["tom-instrumentation/1.0.0"] = (
        INSTRUMENTATION_SCHEMA_VERSION
    )
    trial_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    actor_id: StableIdentifier
    counterpart_id: StableIdentifier
    update_id: StableIdentifier
    update_hash: Sha256Digest
    previous_update_id: StableIdentifier | None
    target: SafeCategory
    prior: DistributionSnapshot
    posterior: DistributionSnapshot
    evidence_ids: tuple[StableIdentifier, ...]
    likelihoods: tuple[NonNegativeNumber, ...]
    method: UpdateMethod
    updater_version: StableIdentifier
    observation_model_version: StableIdentifier
    prior_entropy: NonNegativeNumber
    posterior_entropy: NonNegativeNumber
    entropy_change: FiniteNumber
    warnings_hash: Sha256Digest | None

    @model_validator(mode="after")
    def _validate_update(self) -> Self:
        if self.actor_id == self.counterpart_id:
            raise ValueError("actor and counterpart must be distinct")
        if self.method is UpdateMethod.ORACLE:
            raise ValueError("oracle updates cannot be instrumented")
        _assert_derived_id(
            self.update_id,
            self.update_hash,
            prefix="tom_update_",
            field_name="update_id",
        )
        _canonical_unique(self.evidence_ids, field_name="update evidence_ids")
        if self.target != self.prior.target or self.target != self.posterior.target:
            raise ValueError("update target does not match distributions")
        if len(self.likelihoods) != len(self.prior.categories):
            raise ValueError("likelihoods do not align with update categories")
        if self.prior.categories != self.posterior.categories:
            raise ValueError("update category spaces differ")
        comparisons = (
            (self.prior_entropy, self.prior.entropy, "prior entropy"),
            (self.posterior_entropy, self.posterior.entropy, "posterior entropy"),
            (
                self.entropy_change,
                self.posterior.entropy - self.prior.entropy,
                "entropy change",
            ),
        )
        for actual, expected, field_name in comparisons:
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{field_name} does not match distributions")
        for field_name in (
            "actor_id",
            "counterpart_id",
            "updater_version",
            "observation_model_version",
        ):
            _reject_sensitive_channel(
                getattr(self, field_name), field_name=field_name
            )
        return self

    @classmethod
    def from_update(
        cls,
        *,
        trial_id: str,
        turn: int,
        actor_id: str,
        counterpart_id: str,
        update: BeliefUpdate,
    ) -> Self:
        if not isinstance(update, BeliefUpdate):
            raise TypeError("update must be a BeliefUpdate")
        if update.method is UpdateMethod.ORACLE:
            raise ValueError("oracle updates cannot be instrumented")
        warnings_hash = (
            _hash_json(list(update.warnings)) if update.warnings else None
        )
        return cls(
            trial_id=trial_id,
            turn=turn,
            actor_id=actor_id,
            counterpart_id=counterpart_id,
            update_id=update.update_id,
            update_hash=update.content_hash(),
            previous_update_id=update.previous_update_id,
            target=update.posterior.target,
            prior=DistributionSnapshot.from_distribution(update.prior),
            posterior=DistributionSnapshot.from_distribution(update.posterior),
            evidence_ids=update.evidence_ids,
            likelihoods=update.likelihoods,
            method=update.method,
            updater_version=update.updater_version,
            observation_model_version=update.observation_model_version,
            prior_entropy=update.prior.entropy,
            posterior_entropy=update.posterior.entropy,
            entropy_change=update.entropy_change,
            warnings_hash=warnings_hash,
        )


def _state_distributions(
    state: PartnerBeliefState,
) -> tuple[BeliefDistribution, ...]:
    return (
        state.policy_type,
        state.expected_next_action,
        state.reservation_value,
        *state.goal_beliefs,
        *state.constraint_beliefs,
        *state.fact_beliefs,
        state.trustworthiness,
    )


def _validate_state_input(state: PartnerBeliefState) -> None:
    if not isinstance(state, PartnerBeliefState):
        raise TypeError("state must be a PartnerBeliefState")
    for identifier in (state.observer_id, state.counterpart_id):
        _stable_identifier(identifier)
        _reject_sensitive_channel(identifier, field_name="state actor")
    for distribution in _state_distributions(state):
        _validate_distribution_input(distribution)


def _state_summary(
    state: PartnerBeliefState,
    updates: tuple[BeliefUpdateSnapshot, ...],
    evidence: tuple[EvidenceLineage, ...],
) -> StructuredSummary:
    return StructuredSummary(
        entries=(
            ("belief_target_count", len(_state_distributions(state))),
            ("evidence_count", len(evidence)),
            ("state_version", state.state_version),
            ("update_count", len(updates)),
        )
    )


class PartnerBeliefPublication(_CanonicalModel):
    """Event-safe partner state with complete update/evidence provenance."""

    schema_version: Literal["tom-instrumentation/1.0.0"] = (
        INSTRUMENTATION_SCHEMA_VERSION
    )
    trial_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    actor_id: StableIdentifier
    counterpart_id: StableIdentifier
    visibility: Literal[PublicationVisibility.RESTRICTED] = (
        PublicationVisibility.RESTRICTED
    )
    audience: Annotated[tuple[SafeCategory, ...], Field(min_length=1)] = (
        "experiment_runtime",
    )
    state: PartnerBeliefState
    state_schema_version: Literal["2.0.0"]
    state_hash: Sha256Digest
    distributions: Annotated[
        tuple[DistributionSnapshot, ...], Field(min_length=1)
    ]
    updates: tuple[BeliefUpdateSnapshot, ...]
    evidence: tuple[EvidenceLineage, ...]
    source_model_call: ScopedModelCall | None = None
    summary: StructuredSummary

    @model_validator(mode="after")
    def _validate_publication(self) -> Self:
        _validate_scope(
            self.trial_id,
            self.turn,
            self.actor_id,
            self.counterpart_id,
            self.audience,
        )
        _validate_state_input(self.state)
        if (self.actor_id, self.counterpart_id) != (
            self.state.observer_id,
            self.state.counterpart_id,
        ):
            raise ValueError("publication actors do not match partner state")
        if self.state_schema_version != self.state.schema_version:
            raise ValueError("state schema version does not match state")
        if self.state_hash != self.state.state_hash:
            raise ValueError("state hash does not match partner state")

        expected_distributions = tuple(
            sorted(
                (
                    DistributionSnapshot.from_distribution(item)
                    for item in _state_distributions(self.state)
                ),
                key=lambda item: item.target,
            )
        )
        if self.distributions != expected_distributions:
            raise ValueError("published distributions do not match partner state")

        update_ids = tuple(item.update_id for item in self.updates)
        _canonical_unique(update_ids, field_name="published updates")
        if update_ids != self.state.update_ids:
            raise ValueError("published updates do not match state update IDs")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        _canonical_unique(evidence_ids, field_name="published evidence")
        if evidence_ids != self.state.evidence_ids:
            raise ValueError("published evidence does not match state evidence IDs")
        event_ids = tuple(item.evidence_event_id for item in self.evidence)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("evidence event IDs must be unique")

        expected_scope = (
            self.trial_id,
            self.turn,
            self.actor_id,
            self.counterpart_id,
        )
        for update in self.updates:
            if (
                update.trial_id,
                update.turn,
                update.actor_id,
                update.counterpart_id,
            ) != expected_scope:
                raise ValueError("update crosses publication scope")
        for item in self.evidence:
            if (
                item.trial_id,
                item.publication_turn,
                item.actor_id,
                item.counterpart_id,
            ) != expected_scope:
                raise ValueError("evidence crosses publication scope")
            if item.observer_id != self.actor_id:
                raise ValueError("state evidence observer must be the actor")
        if self.source_model_call is not None and (
            self.source_model_call.trial_id,
            self.source_model_call.turn,
            self.source_model_call.actor_id,
        ) != (self.trial_id, self.turn, self.actor_id):
            raise ValueError("source model call crosses publication scope")

        self._validate_update_lineage()
        if self.summary != _state_summary(self.state, self.updates, self.evidence):
            raise ValueError("state summary does not match publication")
        return self

    def _validate_update_lineage(self) -> None:
        by_id = {item.update_id: item for item in self.updates}
        flattened_evidence = tuple(
            evidence_id
            for update in self.updates
            for evidence_id in update.evidence_ids
        )
        if len(set(flattened_evidence)) != len(flattened_evidence):
            raise ValueError("each evidence ID must occur in exactly one update")
        if tuple(sorted(flattened_evidence)) != self.state.evidence_ids:
            raise ValueError("update evidence does not exactly cover state evidence")

        by_target: dict[str, list[BeliefUpdateSnapshot]] = {}
        referenced_previous: set[str] = set()
        for update in self.updates:
            by_target.setdefault(update.target, []).append(update)
            if update.previous_update_id is None:
                continue
            previous = by_id.get(update.previous_update_id)
            if previous is None:
                raise ValueError("previous update is absent from publication")
            if update.previous_update_id in referenced_previous:
                raise ValueError("update lineage branches from one predecessor")
            referenced_previous.add(update.previous_update_id)
            if previous.target != update.target:
                raise ValueError("update chain changes belief target")
            if previous.posterior != update.prior:
                raise ValueError("prior does not equal predecessor posterior")

        state_by_target = {item.target: item for item in self.distributions}
        for target, chain in by_target.items():
            roots = tuple(item for item in chain if item.previous_update_id is None)
            leaves = tuple(
                item for item in chain if item.update_id not in referenced_previous
            )
            if len(roots) != 1 or len(leaves) != 1:
                raise ValueError("each target must have one linear update chain")
            visited: set[str] = set()
            cursor = leaves[0]
            while True:
                if cursor.update_id in visited:
                    raise ValueError("update lineage contains a cycle")
                visited.add(cursor.update_id)
                if cursor.previous_update_id is None:
                    break
                cursor = by_id[cursor.previous_update_id]
            if len(visited) != len(chain):
                raise ValueError("update target contains disconnected chains")
            if target not in state_by_target:
                raise ValueError("update target is absent from partner state")
            if leaves[0].posterior != state_by_target[target]:
                raise ValueError("update chain does not terminate at partner state")

    @classmethod
    def from_state(
        cls,
        *,
        trial_id: str,
        turn: int,
        state: PartnerBeliefState,
        updates: Iterable[BeliefUpdate],
        evidence_event_links: Iterable[EvidenceEventLink],
        source_model_call: ScopedModelCall | None = None,
        audience: tuple[str, ...] = ("experiment_runtime",),
    ) -> Self:
        _validate_state_input(state)
        update_values = tuple(updates)
        if any(not isinstance(item, BeliefUpdate) for item in update_values):
            raise TypeError("updates must contain BeliefUpdate records")
        update_values = tuple(sorted(update_values, key=lambda item: item.update_id))
        links = tuple(evidence_event_links)
        if any(not isinstance(item, EvidenceEventLink) for item in links):
            raise TypeError("evidence_event_links must contain EvidenceEventLink records")
        link_by_evidence = {item.evidence_id: item for item in links}
        if len(link_by_evidence) != len(links):
            raise ValueError("evidence event links must have unique evidence IDs")

        evidence_by_id: dict[str, Evidence] = {}
        for update in update_values:
            for item in update.evidence:
                if item.evidence_id in evidence_by_id:
                    raise ValueError("evidence cannot be reused across updates")
                evidence_by_id[item.evidence_id] = item
        if set(link_by_evidence) != set(evidence_by_id):
            raise ValueError("event links must exactly cover update evidence")

        update_snapshots = tuple(
            BeliefUpdateSnapshot.from_update(
                trial_id=trial_id,
                turn=turn,
                actor_id=state.observer_id,
                counterpart_id=state.counterpart_id,
                update=item,
            )
            for item in update_values
        )
        evidence_snapshots = tuple(
            EvidenceLineage.from_evidence(
                trial_id=trial_id,
                publication_turn=turn,
                actor_id=state.observer_id,
                counterpart_id=state.counterpart_id,
                evidence=evidence_by_id[evidence_id],
                event_link=link_by_evidence[evidence_id],
            )
            for evidence_id in sorted(evidence_by_id)
        )
        distributions = tuple(
            sorted(
                (
                    DistributionSnapshot.from_distribution(item)
                    for item in _state_distributions(state)
                ),
                key=lambda item: item.target,
            )
        )
        return cls(
            trial_id=trial_id,
            turn=turn,
            actor_id=state.observer_id,
            counterpart_id=state.counterpart_id,
            audience=audience,
            state=state,
            state_schema_version=state.schema_version,
            state_hash=state.state_hash,
            distributions=distributions,
            updates=update_snapshots,
            evidence=evidence_snapshots,
            source_model_call=source_model_call,
            summary=_state_summary(state, update_snapshots, evidence_snapshots),
        )

    @property
    def state_id(self) -> str:
        return _derived_id(self.state_hash, prefix="tom_state_")

    def to_tom_state_updated_fields(self) -> dict[str, Any]:
        """Return exact JSON fields for a future ToMStateUpdatedPayload."""
        if not self.evidence:
            raise ValueError("ToMStateUpdated requires at least one evidence event")
        if self.source_model_call is None:
            raise ValueError("ToMStateUpdated requires a real source model call")
        return {
            "state_id": self.state_id,
            "actor_id": self.actor_id,
            "counterpart_actor_id": self.counterpart_id,
            "state_schema_version": self.state_schema_version,
            "state_hash": self.state_hash.removeprefix("sha256:"),
            "evidence_event_ids": tuple(
                item.evidence_event_id for item in self.evidence
            ),
            "source_model_call_id": self.source_model_call.call_id,
        }


def _validate_scope(
    trial_id: str,
    turn: int,
    actor_id: str,
    counterpart_id: str,
    audience: tuple[str, ...],
) -> None:
    del turn
    if actor_id == counterpart_id:
        raise ValueError("actor and counterpart must be distinct")
    for value, field_name in (
        (trial_id, "trial_id"),
        (actor_id, "actor_id"),
        (counterpart_id, "counterpart_id"),
    ):
        _reject_sensitive_channel(value, field_name=field_name)
    _canonical_unique(audience, field_name="audience")


LikelihoodSnapshot = tuple[SafeCategory, Probability]
EventAccessSnapshot = tuple[StableIdentifier, AccessBasis]


class RecursiveAuthorizationSnapshot(_CanonicalModel):
    """Authorization proof with private fact identifiers reduced to hashes."""

    schema_version: Literal["tom-instrumentation/1.0.0"] = (
        INSTRUMENTATION_SCHEMA_VERSION
    )
    trial_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    actor_id: StableIdentifier
    counterpart_id: StableIdentifier
    target: SafeCategory
    target_spec_id: StableIdentifier
    use_id: StableIdentifier
    use_hash: Sha256Digest
    input_id: StableIdentifier
    input_hash: Sha256Digest
    evidence_id: StableIdentifier
    evidence_hash: Sha256Digest
    evidence_turn: Annotated[StrictInt, Field(ge=0)]
    source_event_id: StableIdentifier
    source_actor_id: StableIdentifier
    observer_id: StableIdentifier
    information_path: Annotated[
        tuple[StableIdentifier, ...], Field(min_length=2, max_length=3)
    ]
    authorized_actor_ids: Annotated[
        tuple[StableIdentifier, ...], Field(min_length=2)
    ]
    access_record_ids: Annotated[
        tuple[StableIdentifier, ...], Field(min_length=1)
    ]
    event_access: Annotated[
        tuple[EventAccessSnapshot, ...], Field(min_length=2)
    ]
    fact_access_hash: Sha256Digest
    input_fact_ids_hash: Sha256Digest
    use_kind: EvidenceUseKind
    likelihoods: tuple[LikelihoodSnapshot, ...]
    hard_category_hash: Sha256Digest | None
    calibration_id: StableIdentifier | None
    update_input_version: StableIdentifier | None

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        _validate_scope(
            self.trial_id,
            self.turn,
            self.actor_id,
            self.counterpart_id,
            ("experiment_runtime",),
        )
        if self.evidence_turn > self.turn:
            raise ValueError("future-turn recursive evidence cannot be published")
        _assert_derived_id(
            self.use_id,
            self.use_hash,
            prefix="authorized_evidence_",
            field_name="use_id",
        )
        _assert_derived_id(
            self.input_id,
            self.input_hash,
            prefix="recursive_input_",
            field_name="input_id",
        )
        _assert_derived_id(
            self.evidence_id,
            self.evidence_hash,
            prefix="evidence_",
            field_name="evidence_id",
        )
        _canonical_unique(
            self.authorized_actor_ids, field_name="authorized_actor_ids"
        )
        _canonical_unique(
            self.access_record_ids, field_name="access_record_ids"
        )
        _canonical_pair_keys(self.likelihoods, field_name="likelihoods")
        event_actors = tuple(actor_id for actor_id, _ in self.event_access)
        _canonical_unique(event_actors, field_name="event_access")
        if set(event_actors) != set(self.authorized_actor_ids):
            raise ValueError("event access must cover every authorized actor")
        if self.authorized_actor_ids != tuple(sorted(set(self.information_path))):
            raise ValueError("authorized actors do not match information path")
        expected_path = (
            (self.actor_id, self.counterpart_id)
            if len(self.information_path) == 2
            else (self.actor_id, self.counterpart_id, self.actor_id)
        )
        if self.information_path != expected_path:
            raise ValueError("recursive authorization has the wrong information path")
        if self.use_kind is EvidenceUseKind.HARD_KNOWLEDGE:
            if self.likelihoods:
                raise ValueError("hard knowledge cannot publish likelihoods")
            if self.hard_category_hash is None:
                raise ValueError("hard knowledge requires a category hash")
            if self.calibration_id is not None or self.update_input_version is not None:
                raise ValueError("hard knowledge cannot claim soft calibration")
        else:
            if not self.likelihoods:
                raise ValueError("soft evidence requires likelihoods")
            if self.hard_category_hash is not None:
                raise ValueError("soft evidence cannot carry a hard category hash")
            if self.calibration_id is None or self.update_input_version is None:
                raise ValueError("soft evidence requires calibration provenance")
        for field_name in (
            "actor_id",
            "counterpart_id",
            "source_event_id",
            "source_actor_id",
            "observer_id",
        ):
            _reject_sensitive_channel(
                getattr(self, field_name), field_name=field_name
            )
        for field_name in ("calibration_id", "update_input_version"):
            value = getattr(self, field_name)
            if value is not None:
                _reject_sensitive_channel(value, field_name=field_name)
        return self

    @classmethod
    def from_use(
        cls,
        *,
        trial_id: str,
        turn: int,
        actor_id: str,
        counterpart_id: str,
        use: AuthorizedEvidenceUse,
    ) -> Self:
        if not isinstance(use, AuthorizedEvidenceUse):
            raise TypeError("use must be an AuthorizedEvidenceUse")
        item = use.evidence_input
        _validate_evidence_input(item.evidence)
        if item.trial_id != trial_id:
            raise ValueError("recursive evidence crosses trial boundary")
        if item.turn > turn:
            raise ValueError("recursive evidence comes from a future turn")
        hard_hash = (
            _hash_text(item.hard_category)
            if item.hard_category is not None
            else None
        )
        return cls(
            trial_id=trial_id,
            turn=turn,
            actor_id=actor_id,
            counterpart_id=counterpart_id,
            target=item.target,
            target_spec_id=use.target_spec_id,
            use_id=use.use_id,
            use_hash=use.content_hash(),
            input_id=item.input_id,
            input_hash=item.content_hash(),
            evidence_id=item.evidence_id,
            evidence_hash=item.evidence.content_hash(),
            evidence_turn=item.turn,
            source_event_id=item.source_event_id,
            source_actor_id=item.source_actor_id,
            observer_id=item.observer_id,
            information_path=use.information_path,
            authorized_actor_ids=use.authorized_actor_ids,
            access_record_ids=use.access_record_ids,
            event_access=use.event_access,
            fact_access_hash=_hash_json(use.model_dump(mode="json")["fact_access"]),
            input_fact_ids_hash=_hash_json(list(item.fact_ids)),
            use_kind=item.use_kind,
            likelihoods=item.likelihoods,
            hard_category_hash=hard_hash,
            calibration_id=item.calibration_id,
            update_input_version=item.update_input_version,
        )


class RecursiveResultSnapshot(_CanonicalModel):
    """One recursive result with scorable state and authorization lineage."""

    schema_version: Literal["tom-instrumentation/1.0.0"] = (
        INSTRUMENTATION_SCHEMA_VERSION
    )
    trial_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    actor_id: StableIdentifier
    counterpart_id: StableIdentifier
    result_id: StableIdentifier
    result_hash: Sha256Digest
    recursive_state_hash: Sha256Digest
    root_state_hash: Sha256Digest
    depth: Literal[1, 2]
    information_path: tuple[StableIdentifier, ...]
    target: SafeCategory
    target_kind: RecursiveTargetKind
    target_spec_id: StableIdentifier
    target_spec_hash: Sha256Digest
    subject_actor_id: StableIdentifier
    scoring_reference_hash: Sha256Digest
    expected_ground_truth_kind: GroundTruthKind
    target_distribution: DistributionSnapshot
    evidence_ids: Annotated[tuple[StableIdentifier, ...], Field(min_length=1)]
    authorizations: Annotated[
        tuple[RecursiveAuthorizationSnapshot, ...], Field(min_length=1)
    ]
    access_record_ids: Annotated[
        tuple[StableIdentifier, ...], Field(min_length=1)
    ]
    permitted_external_sources: tuple[StableIdentifier, ...]
    builder_version: StableIdentifier

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        _validate_scope(
            self.trial_id,
            self.turn,
            self.actor_id,
            self.counterpart_id,
            ("experiment_runtime",),
        )
        _assert_derived_id(
            self.result_id,
            self.result_hash,
            prefix="recursive_result_",
            field_name="result_id",
        )
        _assert_derived_id(
            self.target_spec_id,
            self.target_spec_hash,
            prefix="recursive_target_",
            field_name="target_spec_id",
        )
        expected_path = (
            (self.actor_id, self.counterpart_id)
            if self.depth == 1
            else (self.actor_id, self.counterpart_id, self.actor_id)
        )
        if self.information_path != expected_path:
            raise ValueError("recursive result information path is inconsistent")
        expected_subject = self.counterpart_id if self.depth == 1 else self.actor_id
        if self.subject_actor_id != expected_subject:
            raise ValueError("recursive target subject is inconsistent with depth")
        if self.target_distribution.target != self.target:
            raise ValueError("recursive target distribution names a different target")
        evidence_ids = tuple(item.evidence_id for item in self.authorizations)
        _canonical_unique(evidence_ids, field_name="recursive authorizations")
        if evidence_ids != self.evidence_ids:
            raise ValueError("recursive evidence and authorizations differ")
        _canonical_unique(self.access_record_ids, field_name="access_record_ids")
        _canonical_unique(
            self.permitted_external_sources,
            field_name="permitted_external_sources",
        )
        for item in self.permitted_external_sources:
            _reject_sensitive_channel(
                item, field_name="permitted_external_sources"
            )
        _reject_sensitive_channel(
            self.subject_actor_id, field_name="subject_actor_id"
        )
        expected_scope = (
            self.trial_id,
            self.turn,
            self.actor_id,
            self.counterpart_id,
        )
        for item in self.authorizations:
            if (
                item.trial_id,
                item.turn,
                item.actor_id,
                item.counterpart_id,
            ) != expected_scope:
                raise ValueError("authorization crosses recursive result scope")
            if item.target != self.target:
                raise ValueError("authorization targets a different variable")
            if item.target_spec_id != self.target_spec_id:
                raise ValueError("authorization references a different target spec")
            if item.information_path != self.information_path:
                raise ValueError("authorization information path differs")
            if item.access_record_ids != self.access_record_ids:
                raise ValueError("authorization access records differ from result")
        _reject_sensitive_channel(self.builder_version, field_name="builder_version")
        return self

    @classmethod
    def from_result(cls, value: RecursiveBeliefResult) -> Self:
        if not isinstance(value, RecursiveBeliefResult):
            raise TypeError("result must be a RecursiveBeliefResult")
        authorizations = tuple(
            sorted(
                (
                    RecursiveAuthorizationSnapshot.from_use(
                        trial_id=value.trial_id,
                        turn=value.turn,
                        actor_id=value.state.root_observer_id,
                        counterpart_id=value.state.counterpart_id,
                        use=item,
                    )
                    for item in value.evidence_uses
                ),
                key=lambda item: item.evidence_id,
            )
        )
        return cls(
            trial_id=value.trial_id,
            turn=value.turn,
            actor_id=value.state.root_observer_id,
            counterpart_id=value.state.counterpart_id,
            result_id=value.result_id,
            result_hash=value.content_hash(),
            recursive_state_hash=value.state.state_hash,
            root_state_hash=value.state.root_state_hash,
            depth=value.state.depth,
            information_path=value.state.information_path,
            target=value.target.target,
            target_kind=value.target.kind,
            target_spec_id=value.target.target_spec_id,
            target_spec_hash=value.target.content_hash(),
            subject_actor_id=value.target.subject_actor_id,
            scoring_reference_hash=_hash_text(value.target.scoring_reference_id),
            expected_ground_truth_kind=value.target.expected_ground_truth_kind,
            target_distribution=DistributionSnapshot.from_distribution(
                value.state.target_belief
            ),
            evidence_ids=tuple(item.evidence_id for item in authorizations),
            authorizations=authorizations,
            access_record_ids=value.access_record_ids,
            permitted_external_sources=value.state.permitted_external_sources,
            builder_version=value.builder_version,
        )


def _recursion_summary(
    results: tuple[RecursiveResultSnapshot, ...],
    evidence: tuple[EvidenceLineage, ...],
) -> StructuredSummary:
    return StructuredSummary(
        entries=(
            ("evidence_count", len(evidence)),
            ("max_depth", max(item.depth for item in results)),
            ("result_count", len(results)),
            ("target_count", len({item.target for item in results})),
        )
    )


class RecursiveBeliefPublication(_CanonicalModel):
    """Canonical collection of recursive beliefs sharing one root state."""

    schema_version: Literal["tom-instrumentation/1.0.0"] = (
        INSTRUMENTATION_SCHEMA_VERSION
    )
    trial_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    actor_id: StableIdentifier
    counterpart_id: StableIdentifier
    visibility: Literal[PublicationVisibility.RESTRICTED] = (
        PublicationVisibility.RESTRICTED
    )
    audience: Annotated[tuple[SafeCategory, ...], Field(min_length=1)] = (
        "experiment_runtime",
    )
    root_state_hash: Sha256Digest
    root_state_version: Annotated[StrictInt, Field(ge=0)]
    results: Annotated[tuple[RecursiveResultSnapshot, ...], Field(min_length=1)]
    evidence: Annotated[tuple[EvidenceLineage, ...], Field(min_length=1)]
    summary: StructuredSummary

    @model_validator(mode="after")
    def _validate_publication(self) -> Self:
        _validate_scope(
            self.trial_id,
            self.turn,
            self.actor_id,
            self.counterpart_id,
            self.audience,
        )
        result_ids = tuple(item.result_id for item in self.results)
        _canonical_unique(result_ids, field_name="recursive results")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        _canonical_unique(evidence_ids, field_name="recursive evidence")
        event_ids = tuple(item.evidence_event_id for item in self.evidence)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("recursive evidence event IDs must be unique")
        expected_scope = (
            self.trial_id,
            self.turn,
            self.actor_id,
            self.counterpart_id,
        )
        result_evidence: set[str] = set()
        target_keys: set[tuple[int, str]] = set()
        access_record_sets: set[tuple[str, ...]] = set()
        for item in self.results:
            if (
                item.trial_id,
                item.turn,
                item.actor_id,
                item.counterpart_id,
            ) != expected_scope:
                raise ValueError("recursive result crosses publication scope")
            if item.root_state_hash != self.root_state_hash:
                raise ValueError("recursive result references a different root state")
            target_key = (item.depth, item.target)
            if target_key in target_keys:
                raise ValueError("recursive depth/target pairs must be unique")
            target_keys.add(target_key)
            result_evidence.update(item.evidence_ids)
            access_record_sets.add(item.access_record_ids)
        if len(access_record_sets) != 1:
            raise ValueError("recursive results disagree on access snapshots")
        if tuple(sorted(result_evidence)) != evidence_ids:
            raise ValueError("recursive evidence does not exactly cover results")
        for item in self.evidence:
            if (
                item.trial_id,
                item.publication_turn,
                item.actor_id,
                item.counterpart_id,
            ) != expected_scope:
                raise ValueError("recursive evidence crosses publication scope")
            if item.observer_id not in {self.actor_id, self.counterpart_id}:
                raise ValueError("recursive evidence observer is outside the dyad")
        if self.summary != _recursion_summary(self.results, self.evidence):
            raise ValueError("recursive summary does not match publication")
        return self

    @classmethod
    def from_results(
        cls,
        *,
        root_state: PartnerBeliefState,
        results: Iterable[RecursiveBeliefResult],
        evidence_event_links: Iterable[EvidenceEventLink],
        audience: tuple[str, ...] = ("experiment_runtime",),
    ) -> Self:
        _validate_state_input(root_state)
        result_values = tuple(results)
        if not result_values:
            raise ValueError("recursive publication requires at least one result")
        if any(not isinstance(item, RecursiveBeliefResult) for item in result_values):
            raise TypeError("results must contain RecursiveBeliefResult records")
        first = result_values[0]
        trial_id = first.trial_id
        turn = first.turn
        expected = (
            root_state.state_hash,
            root_state.observer_id,
            root_state.counterpart_id,
        )
        evidence_by_id: dict[str, Evidence] = {}
        for result in result_values:
            if result.trial_id != trial_id:
                raise ValueError("recursive result crosses trial boundary")
            if result.turn != turn:
                raise ValueError("recursive results must share one turn")
            if (
                result.state.root_state_hash,
                result.state.root_observer_id,
                result.state.counterpart_id,
            ) != expected:
                raise ValueError("recursive result does not match root state")
            for use in result.evidence_uses:
                item = use.evidence_input
                if item.trial_id != trial_id:
                    raise ValueError("recursive evidence crosses trial boundary")
                if item.turn > turn:
                    raise ValueError("future recursive evidence cannot be published")
                existing = evidence_by_id.get(item.evidence_id)
                if existing is not None and existing != item.evidence:
                    raise ValueError("one evidence ID resolves to different records")
                evidence_by_id[item.evidence_id] = item.evidence

        links = tuple(evidence_event_links)
        if any(not isinstance(item, EvidenceEventLink) for item in links):
            raise TypeError("evidence_event_links must contain EvidenceEventLink records")
        link_by_evidence = {item.evidence_id: item for item in links}
        if len(link_by_evidence) != len(links):
            raise ValueError("evidence event links must have unique evidence IDs")
        if set(link_by_evidence) != set(evidence_by_id):
            raise ValueError("event links must exactly cover recursive evidence")

        snapshots = tuple(
            sorted(
                (RecursiveResultSnapshot.from_result(item) for item in result_values),
                key=lambda item: item.result_id,
            )
        )
        evidence_snapshots = tuple(
            EvidenceLineage.from_evidence(
                trial_id=trial_id,
                publication_turn=turn,
                actor_id=root_state.observer_id,
                counterpart_id=root_state.counterpart_id,
                evidence=evidence_by_id[evidence_id],
                event_link=link_by_evidence[evidence_id],
            )
            for evidence_id in sorted(evidence_by_id)
        )
        return cls(
            trial_id=trial_id,
            turn=turn,
            actor_id=root_state.observer_id,
            counterpart_id=root_state.counterpart_id,
            audience=audience,
            root_state_hash=root_state.state_hash,
            root_state_version=root_state.state_version,
            results=snapshots,
            evidence=evidence_snapshots,
            summary=_recursion_summary(snapshots, evidence_snapshots),
        )


def _validate_policy_request_safe(request: PolicyRequest) -> None:
    if not isinstance(request, PolicyRequest):
        raise TypeError("request must be a PolicyRequest")
    identifiers = (
        request.trial_id,
        request.actor_id,
        request.counterpart_id,
        request.objective.objective_id,
        request.objective.version,
        *request.legal_actions,
    )
    for value in identifiers:
        _reject_sensitive_channel(value, field_name="policy request")
    if request.constraints is not None:
        constraint_values = (
            request.constraints.constraint_set_id,
            request.constraints.version,
            *request.constraints.permitted_actions,
            *request.constraints.forbidden_actions,
        )
        for value in constraint_values:
            _reject_sensitive_channel(value, field_name="policy constraints")
        if request.constraints.required_action is not None:
            _reject_sensitive_channel(
                request.constraints.required_action,
                field_name="required action",
            )
    for candidate in request.candidates:
        _reject_sensitive_channel(candidate.action, field_name="candidate action")
        _reject_sensitive_channel(
            candidate.contract_version, field_name="candidate contract"
        )
        for response in candidate.responses:
            _reject_sensitive_channel(
                response.counterpart_action,
                field_name="counterpart response",
            )


def _decision_summary(
    action_scores: tuple[ActionRecommendationScore, ...],
    belief_update_ids: tuple[str, ...],
    *,
    linked: bool,
) -> StructuredSummary:
    return StructuredSummary(
        entries=(
            ("action_count", len(action_scores)),
            (
                "eligible_action_count",
                sum(1 for item in action_scores if item.eligible),
            ),
            ("linked", linked),
            ("update_count", len(belief_update_ids)),
        )
    )


class PolicyDecisionPublication(_CanonicalModel):
    """Policy recommendation before or after its immutable action-call link."""

    schema_version: Literal["tom-instrumentation/1.0.0"] = (
        INSTRUMENTATION_SCHEMA_VERSION
    )
    trial_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    actor_id: StableIdentifier
    counterpart_id: StableIdentifier
    visibility: Literal[PublicationVisibility.RESTRICTED] = (
        PublicationVisibility.RESTRICTED
    )
    audience: Annotated[tuple[SafeCategory, ...], Field(min_length=1)] = (
        "experiment_runtime",
    )
    phase: DecisionPublicationPhase
    request: PolicyRequest
    request_id: StableIdentifier
    request_hash: Sha256Digest
    recommendation_id: StableIdentifier
    recommendation_hash: Sha256Digest
    belief_state_hash: Sha256Digest
    state_version: Annotated[StrictInt, Field(gt=0)]
    belief_update_ids: Annotated[
        tuple[StableIdentifier, ...], Field(min_length=1)
    ]
    objective_hash: Sha256Digest
    constraint_hash: Sha256Digest | None
    intervention_condition: PolicyInterventionCondition
    intervention_provenance_ids: Annotated[
        tuple[StableIdentifier, ...], Field(min_length=1)
    ]
    chosen_action: SafeCategory
    action_scores: Annotated[
        tuple[ActionRecommendationScore, ...], Field(min_length=1)
    ]
    score_ids: Annotated[tuple[StableIdentifier, ...], Field(min_length=1)]
    context_summary_hash: Sha256Digest
    advisor_version: StableIdentifier
    unlinked_trace_id: StableIdentifier
    unlinked_trace: ToMDecisionTrace
    linked_trace_id: StableIdentifier | None = None
    linked_trace: ToMDecisionTrace | None = None
    action_call: ScopedModelCall | None = None
    summary: StructuredSummary

    @model_validator(mode="after")
    def _validate_publication(self) -> Self:
        _validate_scope(
            self.trial_id,
            self.turn,
            self.actor_id,
            self.counterpart_id,
            self.audience,
        )
        _validate_policy_request_safe(self.request)
        if (
            self.request.trial_id,
            self.request.turn,
            self.request.actor_id,
            self.request.counterpart_id,
        ) != (
            self.trial_id,
            self.turn,
            self.actor_id,
            self.counterpart_id,
        ):
            raise ValueError("policy request crosses publication scope")
        if self.request.request_id != self.request_id:
            raise ValueError("request ID does not match request")
        if self.request.content_hash() != self.request_hash:
            raise ValueError("request hash does not match request")
        _assert_derived_id(
            self.request_id,
            self.request_hash,
            prefix="policy_request_",
            field_name="request_id",
        )
        _assert_derived_id(
            self.recommendation_id,
            self.recommendation_hash,
            prefix="policy_recommendation_",
            field_name="recommendation_id",
        )
        if self.request.belief_state_hash != self.belief_state_hash:
            raise ValueError("request references a different belief state")
        if self.request.state_version != self.state_version:
            raise ValueError("request state version does not match publication")
        if self.request.objective.objective_hash != self.objective_hash:
            raise ValueError("objective hash does not match request")
        expected_constraint_hash = (
            self.request.constraints.constraint_hash
            if self.request.constraints is not None
            else None
        )
        if self.constraint_hash != expected_constraint_hash:
            raise ValueError("constraint hash does not match request")
        if self.intervention_condition is not self.request.intervention_condition:
            raise ValueError("intervention condition does not match request")
        _canonical_unique(
            self.intervention_provenance_ids,
            field_name="intervention_provenance_ids",
        )
        for item in self.intervention_provenance_ids:
            _reject_sensitive_channel(item, field_name="intervention provenance")

        score_actions = tuple(item.action for item in self.action_scores)
        _canonical_unique(score_actions, field_name="action_scores")
        if score_actions != self.request.legal_actions:
            raise ValueError("scores do not cover the request action space")
        expected_score_ids = tuple(item.score_id for item in self.action_scores)
        if self.score_ids != expected_score_ids:
            raise ValueError("score IDs do not match score decomposition")
        if self.chosen_action not in score_actions:
            raise ValueError("chosen action is absent from scores")
        eligible = tuple(item for item in self.action_scores if item.eligible)
        expected_chosen = min(
            eligible,
            key=lambda item: (-item.objective_score, item.action),
        )
        if self.chosen_action != expected_chosen.action:
            raise ValueError("chosen action violates deterministic score ordering")
        for item in self.action_scores:
            _reject_sensitive_channel(item.action, field_name="score action")
            _validate_distribution_input(item.conditional_prediction)

        trace = self.unlinked_trace
        if trace.action_linked:
            raise ValueError("unlinked trace must not carry an action call")
        if trace.trace_id != self.unlinked_trace_id:
            raise ValueError("unlinked trace ID does not match trace")
        if (
            trace.trial_id,
            trace.turn,
            trace.actor_id,
            trace.counterpart_id,
        ) != (
            self.trial_id,
            self.turn,
            self.actor_id,
            self.counterpart_id,
        ):
            raise ValueError("decision trace crosses publication scope")
        if trace.belief_state_hash != self.belief_state_hash:
            raise ValueError("decision trace references a different belief state")
        if trace.belief_update_ids != self.belief_update_ids:
            raise ValueError("decision trace references different belief updates")
        if trace.chosen_action != self.chosen_action:
            raise ValueError("decision trace chose a different action")
        if trace.intervention_condition != self.intervention_condition.value:
            raise ValueError("decision trace has a different intervention")
        if trace.objective != self.request.objective.objective_id:
            raise ValueError("decision trace has a different objective")
        if trace.recommendation_summary_hash != self.context_summary_hash:
            raise ValueError("decision trace summary hash differs")
        if trace.advisor_version != self.advisor_version:
            raise ValueError("decision trace advisor version differs")
        _validate_distribution_input(trace.predicted_counterpart_action)
        if trace.legal_actions != tuple(item.action for item in eligible):
            raise ValueError("trace legal actions differ from eligible scores")
        if trace.expected_utilities != tuple(
            item.objective_score for item in eligible
        ):
            raise ValueError("trace objective scores differ from score records")
        if trace.conditional_predictions != tuple(
            item.conditional_prediction for item in eligible
        ):
            raise ValueError("trace predictions differ from score records")
        _canonical_unique(self.belief_update_ids, field_name="belief_update_ids")
        for item in self.belief_update_ids:
            _reject_sensitive_channel(item, field_name="belief_update_ids")
        _reject_sensitive_channel(self.advisor_version, field_name="advisor_version")

        if self.phase is DecisionPublicationPhase.PRE_ACTION:
            if any(
                value is not None
                for value in (
                    self.linked_trace_id,
                    self.linked_trace,
                    self.action_call,
                )
            ):
                raise ValueError("pre-action publication must remain unlinked")
        else:
            if (
                self.linked_trace_id is None
                or self.linked_trace is None
                or self.action_call is None
            ):
                raise ValueError("post-action publication requires a linked trace")
            if (
                self.action_call.trial_id,
                self.action_call.turn,
                self.action_call.actor_id,
            ) != (self.trial_id, self.turn, self.actor_id):
                raise ValueError("acting call crosses decision scope")
            expected_linked = self.unlinked_trace.link_action(
                self.action_call.call_id
            )
            if self.linked_trace != expected_linked:
                raise ValueError("linked trace is not the exact immutable successor")
            if self.linked_trace.trace_id != self.linked_trace_id:
                raise ValueError("linked trace ID does not match trace")
            if self.linked_trace.chosen_action_call_id != self.action_call.call_id:
                raise ValueError("linked trace and acting call IDs differ")
        expected_summary = _decision_summary(
            self.action_scores,
            self.belief_update_ids,
            linked=self.phase is DecisionPublicationPhase.POST_ACTION,
        )
        if self.summary != expected_summary:
            raise ValueError("decision summary does not match publication")
        return self

    @classmethod
    def from_recommendation(
        cls,
        *,
        state: PartnerBeliefState,
        request: PolicyRequest,
        recommendation: PolicyRecommendationResult,
        intervention_provenance_ids: tuple[str, ...],
        action_call: ScopedModelCall | None = None,
        audience: tuple[str, ...] = ("experiment_runtime",),
    ) -> Self:
        _validate_state_input(state)
        _validate_policy_request_safe(request)
        if not isinstance(recommendation, PolicyRecommendationResult):
            raise TypeError("recommendation must be a PolicyRecommendationResult")
        if request.request_id != recommendation.request_id:
            raise ValueError("request and recommendation IDs differ")
        if request.belief_state_hash != state.state_hash:
            raise ValueError("request does not reference the supplied state")
        if request.state_version != state.state_version:
            raise ValueError("request and state versions differ")
        if (request.actor_id, request.counterpart_id) != (
            state.observer_id,
            state.counterpart_id,
        ):
            raise ValueError("request actors do not match supplied state")
        if recommendation.belief_state_hash != state.state_hash:
            raise ValueError("recommendation does not reference supplied state")
        if recommendation.belief_update_ids != state.update_ids:
            raise ValueError("recommendation updates do not match supplied state")
        if recommendation.trace.trial_id != request.trial_id:
            raise ValueError("recommendation trace crosses trial boundary")
        if recommendation.trace.turn != request.turn:
            raise ValueError("recommendation trace uses a different turn")
        if (
            recommendation.trace.actor_id,
            recommendation.trace.counterpart_id,
        ) != (request.actor_id, request.counterpart_id):
            raise ValueError("recommendation trace actors differ from request")
        pre_action = cls(
            trial_id=request.trial_id,
            turn=request.turn,
            actor_id=request.actor_id,
            counterpart_id=request.counterpart_id,
            audience=audience,
            phase=DecisionPublicationPhase.PRE_ACTION,
            request=request,
            request_id=request.request_id,
            request_hash=request.content_hash(),
            recommendation_id=recommendation.recommendation_id,
            recommendation_hash=recommendation.content_hash(),
            belief_state_hash=recommendation.belief_state_hash,
            state_version=state.state_version,
            belief_update_ids=recommendation.belief_update_ids,
            objective_hash=request.objective.objective_hash,
            constraint_hash=(
                request.constraints.constraint_hash
                if request.constraints is not None
                else None
            ),
            intervention_condition=request.intervention_condition,
            intervention_provenance_ids=tuple(
                sorted(intervention_provenance_ids)
            ),
            chosen_action=recommendation.chosen_action,
            action_scores=recommendation.action_scores,
            score_ids=tuple(item.score_id for item in recommendation.action_scores),
            context_summary_hash=recommendation.context_summary_hash,
            advisor_version=recommendation.advisor_version,
            unlinked_trace_id=recommendation.trace.trace_id,
            unlinked_trace=recommendation.trace,
            summary=_decision_summary(
                recommendation.action_scores,
                recommendation.belief_update_ids,
                linked=False,
            ),
        )
        return pre_action if action_call is None else pre_action.link_action(action_call)

    def link_action(self, action_call: ScopedModelCall) -> Self:
        """Return the exact post-action publication without mutating this one."""
        if self.phase is DecisionPublicationPhase.POST_ACTION:
            raise ValueError("decision publication is already action-linked")
        if not isinstance(action_call, ScopedModelCall):
            raise TypeError("action_call must be a ScopedModelCall")
        linked_trace = self.unlinked_trace.link_action(action_call.call_id)
        payload = self.model_dump()
        payload.update(
            {
                "phase": DecisionPublicationPhase.POST_ACTION,
                "linked_trace_id": linked_trace.trace_id,
                "linked_trace": linked_trace,
                "action_call": action_call,
                "summary": _decision_summary(
                    self.action_scores,
                    self.belief_update_ids,
                    linked=True,
                ),
            }
        )
        return type(self)(**payload)


def _bundle_summary(
    state: PartnerBeliefPublication,
    recursion: RecursiveBeliefPublication | None,
    decisions: tuple[PolicyDecisionPublication, ...],
) -> StructuredSummary:
    return StructuredSummary(
        entries=(
            ("decision_count", len(decisions)),
            ("evidence_count", len(state.evidence)),
            ("recursive_result_count", 0 if recursion is None else len(recursion.results)),
            ("update_count", len(state.updates)),
        )
    )


class ToMInstrumentationBundle(_CanonicalModel):
    """Cross-checked state, recursion, and pre/post policy publications."""

    schema_version: Literal["tom-instrumentation/1.0.0"] = (
        INSTRUMENTATION_SCHEMA_VERSION
    )
    trial_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    actor_id: StableIdentifier
    counterpart_id: StableIdentifier
    visibility: Literal[PublicationVisibility.RESTRICTED] = (
        PublicationVisibility.RESTRICTED
    )
    audience: Annotated[tuple[SafeCategory, ...], Field(min_length=1)] = (
        "experiment_runtime",
    )
    state: PartnerBeliefPublication
    recursion: RecursiveBeliefPublication | None = None
    decisions: Annotated[
        tuple[PolicyDecisionPublication, ...], Field(max_length=2)
    ] = ()
    summary: StructuredSummary

    @model_validator(mode="after")
    def _validate_bundle(self) -> Self:
        _validate_scope(
            self.trial_id,
            self.turn,
            self.actor_id,
            self.counterpart_id,
            self.audience,
        )
        expected_scope = (
            self.trial_id,
            self.turn,
            self.actor_id,
            self.counterpart_id,
        )
        if (
            self.state.trial_id,
            self.state.turn,
            self.state.actor_id,
            self.state.counterpart_id,
        ) != expected_scope:
            raise ValueError("state publication crosses bundle scope")
        if self.state.audience != self.audience:
            raise ValueError("state publication has a different audience")
        if self.recursion is not None:
            if (
                self.recursion.trial_id,
                self.recursion.turn,
                self.recursion.actor_id,
                self.recursion.counterpart_id,
            ) != expected_scope:
                raise ValueError("recursive publication crosses bundle scope")
            if self.recursion.audience != self.audience:
                raise ValueError("recursive publication has a different audience")
            if self.recursion.root_state_hash != self.state.state_hash:
                raise ValueError("recursion root and published state hashes differ")
            if self.recursion.root_state_version != self.state.state.state_version:
                raise ValueError("recursion root and state versions differ")

        phases = tuple(item.phase for item in self.decisions)
        if len(set(phases)) != len(phases):
            raise ValueError("bundle cannot duplicate a decision phase")
        canonical_phases = tuple(
            phase
            for phase in (
                DecisionPublicationPhase.PRE_ACTION,
                DecisionPublicationPhase.POST_ACTION,
            )
            if phase in phases
        )
        if phases != canonical_phases:
            raise ValueError("decision phases must use pre-action/post-action order")
        for decision in self.decisions:
            if (
                decision.trial_id,
                decision.turn,
                decision.actor_id,
                decision.counterpart_id,
            ) != expected_scope:
                raise ValueError("decision publication crosses bundle scope")
            if decision.audience != self.audience:
                raise ValueError("decision publication has a different audience")
            if decision.belief_state_hash != self.state.state_hash:
                raise ValueError("decision and published state hashes differ")
            if decision.state_version != self.state.state.state_version:
                raise ValueError("decision and state versions differ")
            if decision.belief_update_ids != self.state.state.update_ids:
                raise ValueError("decision and state update IDs differ")
            if (
                decision.phase is DecisionPublicationPhase.POST_ACTION
                and self.state.source_model_call is not None
                and decision.action_call is not None
                and decision.action_call.call_id
                == self.state.source_model_call.call_id
            ):
                raise ValueError("state inference and acting calls must be distinct")
        if len(self.decisions) == 2:
            before, after = self.decisions
            shared_fields = (
                "request_id",
                "recommendation_id",
                "recommendation_hash",
                "unlinked_trace_id",
                "belief_state_hash",
                "chosen_action",
            )
            if any(
                getattr(before, field_name) != getattr(after, field_name)
                for field_name in shared_fields
            ):
                raise ValueError("pre/post decisions are not the same recommendation")
        if self.summary != _bundle_summary(
            self.state, self.recursion, self.decisions
        ):
            raise ValueError("bundle summary does not match publications")
        return self

    @classmethod
    def from_publications(
        cls,
        *,
        state: PartnerBeliefPublication,
        recursion: RecursiveBeliefPublication | None = None,
        decisions: Iterable[PolicyDecisionPublication] = (),
    ) -> Self:
        if not isinstance(state, PartnerBeliefPublication):
            raise TypeError("state must be a PartnerBeliefPublication")
        if recursion is not None and not isinstance(
            recursion, RecursiveBeliefPublication
        ):
            raise TypeError("recursion must be a RecursiveBeliefPublication")
        decision_values = tuple(decisions)
        if any(
            not isinstance(item, PolicyDecisionPublication)
            for item in decision_values
        ):
            raise TypeError("decisions must contain PolicyDecisionPublication records")
        phase_order = {
            DecisionPublicationPhase.PRE_ACTION: 0,
            DecisionPublicationPhase.POST_ACTION: 1,
        }
        decision_values = tuple(
            sorted(decision_values, key=lambda item: phase_order[item.phase])
        )
        return cls(
            trial_id=state.trial_id,
            turn=state.turn,
            actor_id=state.actor_id,
            counterpart_id=state.counterpart_id,
            audience=state.audience,
            state=state,
            recursion=recursion,
            decisions=decision_values,
            summary=_bundle_summary(state, recursion, decision_values),
        )


def tom_state_updated_fields(
    publication: PartnerBeliefPublication,
) -> dict[str, Any]:
    """Functional form of the event-layer adapter."""
    if not isinstance(publication, PartnerBeliefPublication):
        raise TypeError("publication must be a PartnerBeliefPublication")
    return publication.to_tom_state_updated_fields()


__all__ = [
    "BeliefUpdateSnapshot",
    "DecisionPublicationPhase",
    "DistributionSnapshot",
    "EvidenceEventLink",
    "EvidenceLineage",
    "INSTRUMENTATION_SCHEMA_VERSION",
    "PartnerBeliefPublication",
    "PolicyDecisionPublication",
    "PublicationVisibility",
    "RecursiveAuthorizationSnapshot",
    "RecursiveBeliefPublication",
    "RecursiveResultSnapshot",
    "ScopedModelCall",
    "StructuredSummary",
    "ToMInstrumentationBundle",
    "tom_state_updated_fields",
]
