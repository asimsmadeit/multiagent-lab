"""Canonical, information-safe schemas for Theory of Mind v2.

These records contain inspectable partner-policy variables, not hidden reasoning.
All scientific identities are content hashes of strict, immutable payloads.
"""

from __future__ import annotations

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
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)


TOM_SCHEMA_VERSION = "2.0.0"
_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_.:/-]*$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _stable_identifier(value: str) -> str:
    if value != value.strip():
        raise ValueError("identifiers must not have surrounding whitespace")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("identifiers must use canonical NFC Unicode")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError("identifiers must not contain control characters")
    return value


def _stable_category(value: str) -> str:
    _stable_identifier(value)
    if not _CATEGORY_PATTERN.fullmatch(value):
        raise ValueError(
            "categories must be lowercase, stable machine identifiers"
        )
    return value


def _stable_text(value: str) -> str:
    _stable_identifier(value)
    return value


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("value must be a real number, not a boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value must be finite")
    return 0.0 if result == 0.0 else result


def _probability(value: Any) -> float:
    result = _finite_number(value)
    if result < 0.0 or result > 1.0:
        raise ValueError("probabilities must be between zero and one")
    return result


def _non_negative(value: Any) -> float:
    result = _finite_number(value)
    if result < 0.0:
        raise ValueError("value must be non-negative")
    return result


def _sha256_digest(value: str) -> str:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("value must be a lowercase sha256 digest")
    return value


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
ShortText = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=512),
    AfterValidator(_stable_text),
]
Probability = Annotated[float, BeforeValidator(_probability)]
FiniteNumber = Annotated[float, BeforeValidator(_finite_number)]
NonNegativeNumber = Annotated[float, BeforeValidator(_non_negative)]
Sha256Digest = Annotated[
    str,
    StringConstraints(strict=True),
    AfterValidator(_sha256_digest),
]
FeatureValue = StrictBool | StrictInt | FiniteNumber | StrictStr


def _require_canonical_unique(
    values: tuple[str, ...], *, field_name: str
) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")
    if values != tuple(sorted(values)):
        raise ValueError(
            f"{field_name} must use canonical lexicographic ordering"
        )


class _CanonicalModel(BaseModel):
    """Strict frozen base with deterministic JSON and hashing."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    def canonical_dict(self) -> dict[str, Any]:
        """Return the JSON-domain payload used for scientific identity."""
        return self.model_dump(mode="json")

    def canonical_json(self) -> str:
        """Serialize deterministically without accepting non-finite numbers."""
        return json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def content_hash(self) -> str:
        """Return a type-separated, full-length content digest."""
        material = f"{type(self).__name__}\0{self.canonical_json()}".encode()
        return f"sha256:{hashlib.sha256(material).hexdigest()}"


class EvidenceVisibility(str, Enum):
    """Who may condition a belief on an evidence item."""

    PUBLIC = "public"
    EXPLICIT = "explicit"
    ADJUDICATOR_ONLY = "adjudicator_only"


class EvidenceChannel(str, Enum):
    """Provenance class; weak cues remain separate from observed behavior."""

    OBSERVABLE = "observable"
    LINGUISTIC = "linguistic"
    MODEL_DERIVED = "model_derived"


class GroundTruthKind(str, Enum):
    """Whether a belief target has defensible scoring ground truth."""

    OBJECTIVE = "objective"
    INFERRED = "inferred"
    NONE = "none"


class EpistemicStatus(str, Enum):
    """How the probabilities should be interpreted."""

    PRIOR = "prior"
    UPDATED = "updated"
    FROZEN = "frozen"
    ORACLE = "oracle"


class UpdateMethod(str, Enum):
    """Declared update condition, including required baselines."""

    BAYESIAN = "bayesian"
    FREQUENCY_BASELINE = "frequency_baseline"
    FROZEN_PRIOR = "frozen_prior"
    ORACLE = "oracle"
    ESTIMATED = "estimated"


class Evidence(_CanonicalModel):
    """One immutable, event-linked observation available to an observer."""

    schema_version: Literal["2.0.0"] = TOM_SCHEMA_VERSION
    observer_id: StableIdentifier
    source_actor_id: StableIdentifier
    source_event_id: StableIdentifier
    source_call_id: StableIdentifier | None = None
    turn: Annotated[StrictInt, Field(ge=0)]
    features: Annotated[
        tuple[tuple[StableCategory, FeatureValue], ...], Field(min_length=1)
    ]
    channel: EvidenceChannel
    visibility: EvidenceVisibility
    visible_to: tuple[StableIdentifier, ...] = ()
    reliability: Probability
    extractor_version: StableIdentifier
    source_text_hash: Sha256Digest | None = None
    source_span: tuple[StrictInt, StrictInt] | None = None
    parent_evidence_ids: tuple[StableIdentifier, ...] = ()
    summary: ShortText | None = None

    @model_validator(mode="after")
    def _validate_evidence(self) -> Self:
        feature_names = tuple(name for name, _ in self.features)
        _require_canonical_unique(feature_names, field_name="features")
        _require_canonical_unique(self.visible_to, field_name="visible_to")
        _require_canonical_unique(
            self.parent_evidence_ids, field_name="parent_evidence_ids"
        )

        if self.visibility is EvidenceVisibility.PUBLIC:
            if self.visible_to:
                raise ValueError("public evidence must not enumerate recipients")
        else:
            if not self.visible_to:
                raise ValueError("non-public evidence must enumerate recipients")
            if self.observer_id not in self.visible_to:
                raise ValueError("the evidence observer must be a recipient")

        if self.source_span is not None:
            start, end = self.source_span
            if isinstance(start, bool) or isinstance(end, bool):
                raise ValueError("source_span offsets must be integers")
            if start < 0 or end <= start:
                raise ValueError("source_span must be a non-empty forward span")
            if self.source_text_hash is None:
                raise ValueError("a source span requires source_text_hash")
        return self

    @property
    def evidence_id(self) -> str:
        """Content-addressed identity including provenance and visibility."""
        return f"evidence_{self.content_hash().removeprefix('sha256:')}"

    def is_visible_to(self, actor_id: str) -> bool:
        """Return whether an actor may use this evidence in a belief path."""
        if self.visibility is EvidenceVisibility.ADJUDICATOR_ONLY:
            return False
        return (
            self.visibility is EvidenceVisibility.PUBLIC
            or actor_id in self.visible_to
        )


class BeliefDistribution(_CanonicalModel):
    """A finite, declared hypothesis space with normalized probabilities."""

    schema_version: Literal["2.0.0"] = TOM_SCHEMA_VERSION
    target: StableCategory
    categories: Annotated[tuple[StableCategory, ...], Field(min_length=2)]
    probabilities: Annotated[tuple[Probability, ...], Field(min_length=2)]
    unknown_category: StableCategory = "unknown"
    epistemic_status: EpistemicStatus
    ground_truth_kind: GroundTruthKind

    @model_validator(mode="after")
    def _validate_distribution(self) -> Self:
        _require_canonical_unique(self.categories, field_name="categories")
        if len(self.categories) != len(self.probabilities):
            raise ValueError("categories and probabilities must be aligned")
        if self.unknown_category not in self.categories:
            raise ValueError("the declared unknown category must be represented")
        total = math.fsum(self.probabilities)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("probabilities must sum to one")
        return self

    @property
    def state_hash(self) -> str:
        return self.content_hash()

    @property
    def entropy(self) -> float:
        """Shannon entropy in nats."""
        return -math.fsum(
            probability * math.log(probability)
            for probability in self.probabilities
            if probability > 0.0
        )

    def probability(self, category: str) -> float:
        """Return one named probability without positional guessing."""
        try:
            index = self.categories.index(category)
        except ValueError as error:
            raise KeyError(category) from error
        return self.probabilities[index]


class PartnerBeliefState(_CanonicalModel):
    """Complete level-zero partner-policy state for one counterpart."""

    schema_version: Literal["2.0.0"] = TOM_SCHEMA_VERSION
    observer_id: StableIdentifier
    counterpart_id: StableIdentifier
    state_version: Annotated[StrictInt, Field(ge=0)]
    policy_type: BeliefDistribution
    expected_next_action: BeliefDistribution
    reservation_value: BeliefDistribution
    goal_beliefs: Annotated[
        tuple[BeliefDistribution, ...], Field(min_length=1)
    ]
    constraint_beliefs: Annotated[
        tuple[BeliefDistribution, ...], Field(min_length=1)
    ]
    fact_beliefs: Annotated[
        tuple[BeliefDistribution, ...], Field(min_length=1)
    ]
    trustworthiness: BeliefDistribution
    evidence_ids: tuple[StableIdentifier, ...] = ()
    update_ids: tuple[StableIdentifier, ...] = ()

    @model_validator(mode="after")
    def _validate_state(self) -> Self:
        if self.observer_id == self.counterpart_id:
            raise ValueError("observer and counterpart must be distinct")
        _require_canonical_unique(self.evidence_ids, field_name="evidence_ids")
        _require_canonical_unique(self.update_ids, field_name="update_ids")

        beliefs = (
            self.policy_type,
            self.expected_next_action,
            self.reservation_value,
            *self.goal_beliefs,
            *self.constraint_beliefs,
            *self.fact_beliefs,
            self.trustworthiness,
        )
        targets = tuple(belief.target for belief in beliefs)
        if len(set(targets)) != len(targets):
            raise ValueError("belief targets must be unique within a state")
        return self

    @property
    def state_hash(self) -> str:
        return self.content_hash()


class RecursiveBeliefState(_CanonicalModel):
    """A level-one or level-two belief constrained by an information path."""

    schema_version: Literal["2.0.0"] = TOM_SCHEMA_VERSION
    depth: Literal[1, 2]
    root_observer_id: StableIdentifier
    counterpart_id: StableIdentifier
    root_state_hash: Sha256Digest
    information_path: tuple[StableIdentifier, ...]
    target_belief: BeliefDistribution
    evidence: Annotated[tuple[Evidence, ...], Field(min_length=1)]
    permitted_external_sources: tuple[StableIdentifier, ...] = ()

    @field_validator("depth", mode="before")
    @classmethod
    def _reject_boolean_depth(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("depth must be the integer one or two")
        return value

    @model_validator(mode="after")
    def _validate_information_path(self) -> Self:
        if self.root_observer_id == self.counterpart_id:
            raise ValueError("root observer and counterpart must be distinct")
        expected_path = (
            (self.root_observer_id, self.counterpart_id)
            if self.depth == 1
            else (
                self.root_observer_id,
                self.counterpart_id,
                self.root_observer_id,
            )
        )
        if self.information_path != expected_path:
            raise ValueError(
                "information_path does not match the declared recursion target"
            )
        _require_canonical_unique(
            self.permitted_external_sources,
            field_name="permitted_external_sources",
        )
        if set(self.permitted_external_sources) & set(self.information_path):
            raise ValueError("external sources must be external to the path")

        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        _require_canonical_unique(evidence_ids, field_name="evidence")
        known_sources = set(self.information_path) | set(
            self.permitted_external_sources
        )
        required_observers = set(self.information_path)
        for item in self.evidence:
            if item.visibility is EvidenceVisibility.ADJUDICATOR_ONLY:
                raise ValueError("adjudicator-only evidence cannot enter ToM")
            if item.source_actor_id not in known_sources:
                raise ValueError("evidence source is outside the modeled path")
            if item.observer_id not in known_sources:
                raise ValueError("evidence observer is outside the modeled path")
            if not all(item.is_visible_to(actor) for actor in required_observers):
                raise ValueError(
                    "evidence is unavailable along the modeled information path"
                )
        return self

    @property
    def state_hash(self) -> str:
        return self.content_hash()


class BeliefUpdate(_CanonicalModel):
    """Auditable linkage from one distribution to its posterior."""

    schema_version: Literal["2.0.0"] = TOM_SCHEMA_VERSION
    prior: BeliefDistribution
    evidence: tuple[Evidence, ...]
    likelihoods: tuple[NonNegativeNumber, ...]
    posterior: BeliefDistribution
    method: UpdateMethod
    updater_version: StableIdentifier
    observation_model_version: StableIdentifier
    previous_update_id: StableIdentifier | None = None
    warnings: tuple[ShortText, ...] = ()

    @model_validator(mode="after")
    def _validate_linkage(self) -> Self:
        if self.prior.target != self.posterior.target:
            raise ValueError("prior and posterior targets must match")
        if self.prior.categories != self.posterior.categories:
            raise ValueError("prior and posterior categories must match exactly")
        if self.prior.unknown_category != self.posterior.unknown_category:
            raise ValueError("prior and posterior unknown categories must match")
        if self.prior.ground_truth_kind is not self.posterior.ground_truth_kind:
            raise ValueError("ground-truth semantics cannot change during update")
        if len(self.likelihoods) != len(self.prior.categories):
            raise ValueError("likelihoods must align with prior categories")

        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        _require_canonical_unique(evidence_ids, field_name="evidence")
        _require_canonical_unique(self.warnings, field_name="warnings")
        return self

    @property
    def prior_state_hash(self) -> str:
        return self.prior.state_hash

    @property
    def posterior_state_hash(self) -> str:
        return self.posterior.state_hash

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence)

    @property
    def entropy_change(self) -> float:
        """Posterior entropy minus prior entropy, in nats."""
        return self.posterior.entropy - self.prior.entropy

    @property
    def update_id(self) -> str:
        return f"tom_update_{self.content_hash().removeprefix('sha256:')}"


class ToMDecisionTrace(_CanonicalModel):
    """Structured belief-to-action decision record without chain-of-thought."""

    schema_version: Literal["2.0.0"] = TOM_SCHEMA_VERSION
    trial_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    actor_id: StableIdentifier
    counterpart_id: StableIdentifier
    belief_state_hash: Sha256Digest
    belief_update_ids: Annotated[
        tuple[StableIdentifier, ...], Field(min_length=1)
    ]
    predicted_counterpart_action: BeliefDistribution
    legal_actions: Annotated[tuple[StableCategory, ...], Field(min_length=1)]
    expected_utilities: tuple[FiniteNumber, ...]
    conditional_predictions: tuple[BeliefDistribution, ...]
    chosen_action: StableCategory
    chosen_action_call_id: StableIdentifier | None = None
    intervention_condition: StableCategory
    objective: StableCategory
    advisor_version: StableIdentifier
    recommendation_summary_hash: Sha256Digest | None = None

    @model_validator(mode="after")
    def _validate_decision(self) -> Self:
        if self.actor_id == self.counterpart_id:
            raise ValueError("actor and counterpart must be distinct")
        _require_canonical_unique(
            self.belief_update_ids, field_name="belief_update_ids"
        )
        _require_canonical_unique(self.legal_actions, field_name="legal_actions")
        if self.chosen_action not in self.legal_actions:
            raise ValueError("chosen_action must be legal")
        if len(self.expected_utilities) != len(self.legal_actions):
            raise ValueError("utilities must align with legal_actions")
        if len(self.conditional_predictions) != len(self.legal_actions):
            raise ValueError("predictions must align with legal_actions")
        for prediction in self.conditional_predictions:
            if (
                prediction.target
                != self.predicted_counterpart_action.target
                or prediction.categories
                != self.predicted_counterpart_action.categories
            ):
                raise ValueError(
                    "conditional predictions must share the declared action space"
                )
        return self

    @property
    def trace_id(self) -> str:
        return f"tom_trace_{self.content_hash().removeprefix('sha256:')}"

    @property
    def action_linked(self) -> bool:
        return self.chosen_action_call_id is not None

    def link_action(self, call_id: str) -> "ToMDecisionTrace":
        """Return a new immutable trace linked to the acting model call."""
        if self.chosen_action_call_id is not None:
            raise ValueError("decision trace is already linked to an action")
        payload = self.model_dump()
        payload["chosen_action_call_id"] = call_id
        return type(self)(**payload)


__all__ = [
    "BeliefDistribution",
    "BeliefUpdate",
    "EpistemicStatus",
    "Evidence",
    "EvidenceChannel",
    "EvidenceVisibility",
    "GroundTruthKind",
    "PartnerBeliefState",
    "RecursiveBeliefState",
    "TOM_SCHEMA_VERSION",
    "ToMDecisionTrace",
    "UpdateMethod",
]
