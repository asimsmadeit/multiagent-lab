"""Information-safe construction of inspectable recursive partner beliefs.

The builder in this module never reads prose, prompts, or hidden reasoning.  It
combines a declared, scorable target with event-linked evidence and explicit
role-view access snapshots.  Visibility and delivery are hard constraints;
soft inference is possible only through caller-supplied calibrated likelihoods.
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
    StrictInt,
    StringConstraints,
    model_validator,
)

from negotiation.components.tom.schema import (
    BeliefDistribution,
    EpistemicStatus,
    Evidence,
    EvidenceChannel,
    EvidenceVisibility,
    GroundTruthKind,
    PartnerBeliefState,
    RecursiveBeliefState,
)


RECURSION_SCHEMA_VERSION = "tom-recursion/1.0.0"
_CATEGORY_PATTERN = re.compile(r"^[a-z][a-z0-9_.:/-]*$")
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_GENERIC_TARGETS = frozenset(
    {
        "belief",
        "counterpart_belief",
        "mental_state",
        "recursive_belief",
        "recursive_reasoning",
        "unknown",
    }
)
_FORBIDDEN_EVIDENCE_FEATURE_TOKENS = frozenset(
    {
        "adjudicator",
        "chain_of_thought",
        "cot",
        "private_prompt",
        "prompt_text",
        "raw_prompt",
        "raw_text",
        "reasoning_trace",
    }
)


def _stable_identifier(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be a string identifier")
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError("value must be a stable identifier")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("identifiers must use canonical NFC Unicode")
    return value


def _stable_category(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("value must be a string category")
    if not _CATEGORY_PATTERN.fullmatch(value):
        raise ValueError("value must be a lowercase stable category")
    return value


def _finite_probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("probability must be numeric, not boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("probability must be finite")
    if result < 0.0 or result > 1.0:
        raise ValueError("probability must be between zero and one")
    return 0.0 if result == 0.0 else result


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
Probability = Annotated[float, BeforeValidator(_finite_probability)]


def _canonical_unique(values: tuple[str, ...], *, field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")
    if values != tuple(sorted(values)):
        raise ValueError(
            f"{field_name} must use canonical lexicographic ordering"
        )


def _canonical_pairs(
    values: tuple[tuple[str, Any], ...], *, field_name: str
) -> None:
    keys = tuple(item[0] for item in values)
    _canonical_unique(keys, field_name=field_name)


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


class RecursiveTargetKind(str, Enum):
    """Closed set of variables with defined scoring interpretations."""

    POLICY_TYPE = "policy_type"
    NEXT_ACTION = "next_action"
    RESERVATION_VALUE = "reservation_value"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    FACT = "fact"
    TRUSTWORTHINESS = "trustworthiness"


class EvidenceUseKind(str, Enum):
    """Whether evidence supplies exact knowledge or a calibrated update."""

    HARD_KNOWLEDGE = "hard_knowledge"
    SOFT_LIKELIHOOD = "soft_likelihood"


class AccessBasis(str, Enum):
    """Why one event or fact is available to a modeled actor."""

    PUBLIC = "public"
    ROLE_VIEW = "role_view"
    DELIVERED = "delivered"


class InformationAccessRecord(_CanonicalModel):
    """One actor's complete authorized information snapshot for a trial turn."""

    schema_version: Literal["tom-recursion/1.0.0"] = RECURSION_SCHEMA_VERSION
    trial_id: StableIdentifier
    actor_id: StableIdentifier
    through_turn: Annotated[StrictInt, Field(ge=0)]
    public_fact_ids: tuple[StableIdentifier, ...] = ()
    role_view_fact_ids: tuple[StableIdentifier, ...] = ()
    public_event_ids: tuple[StableIdentifier, ...] = ()
    delivered_event_ids: tuple[StableIdentifier, ...] = ()
    adjudicator_only_fact_ids: tuple[StableIdentifier, ...] = ()
    adjudicator_only_event_ids: tuple[StableIdentifier, ...] = ()
    adjudicator_actor_ids: tuple[StableIdentifier, ...] = ("adjudicator",)

    @model_validator(mode="after")
    def _validate_access_partition(self) -> Self:
        fields = (
            "public_fact_ids",
            "role_view_fact_ids",
            "public_event_ids",
            "delivered_event_ids",
            "adjudicator_only_fact_ids",
            "adjudicator_only_event_ids",
            "adjudicator_actor_ids",
        )
        for field_name in fields:
            _canonical_unique(getattr(self, field_name), field_name=field_name)
        if not self.adjudicator_actor_ids:
            raise ValueError("adjudicator_actor_ids must not be empty")
        fact_sets = (
            set(self.public_fact_ids),
            set(self.role_view_fact_ids),
            set(self.adjudicator_only_fact_ids),
        )
        event_sets = (
            set(self.public_event_ids),
            set(self.delivered_event_ids),
            set(self.adjudicator_only_event_ids),
        )
        if any(
            left & right
            for index, left in enumerate(fact_sets)
            for right in fact_sets[index + 1 :]
        ):
            raise ValueError("fact access classes must be disjoint")
        if any(
            left & right
            for index, left in enumerate(event_sets)
            for right in event_sets[index + 1 :]
        ):
            raise ValueError("event access classes must be disjoint")
        if self.actor_id in self.adjudicator_actor_ids:
            raise ValueError("modeled actor cannot be an adjudicator")
        return self

    @property
    def access_id(self) -> str:
        return f"information_access_{self.content_hash().removeprefix('sha256:')}"

    def event_basis(self, event_id: str) -> AccessBasis | None:
        if event_id in self.public_event_ids:
            return AccessBasis.PUBLIC
        if event_id in self.delivered_event_ids:
            return AccessBasis.DELIVERED
        return None

    def fact_basis(self, fact_id: str) -> AccessBasis | None:
        if fact_id in self.public_fact_ids:
            return AccessBasis.PUBLIC
        if fact_id in self.role_view_fact_ids:
            return AccessBasis.ROLE_VIEW
        return None


class ScorableRecursiveTarget(_CanonicalModel):
    """Named variable and independent scoring reference for one recursion."""

    schema_version: Literal["tom-recursion/1.0.0"] = RECURSION_SCHEMA_VERSION
    target: StableCategory
    kind: RecursiveTargetKind
    subject_actor_id: StableIdentifier
    scoring_reference_id: StableIdentifier
    expected_ground_truth_kind: GroundTruthKind
    required_fact_ids: tuple[StableIdentifier, ...] = ()

    @model_validator(mode="after")
    def _validate_scorable_target(self) -> Self:
        _canonical_unique(
            self.required_fact_ids, field_name="required_fact_ids"
        )
        if self.target in _GENERIC_TARGETS:
            raise ValueError("generic/free-form recursive targets are forbidden")
        exact_names = {
            RecursiveTargetKind.POLICY_TYPE: "policy_type",
            RecursiveTargetKind.NEXT_ACTION: "next_action",
            RecursiveTargetKind.RESERVATION_VALUE: "reservation_value",
            RecursiveTargetKind.TRUSTWORTHINESS: "trustworthiness",
        }
        if self.kind in exact_names and self.target != exact_names[self.kind]:
            raise ValueError("target name does not match its scorable kind")
        prefixes = {
            RecursiveTargetKind.GOAL: "goal.",
            RecursiveTargetKind.CONSTRAINT: "constraint.",
            RecursiveTargetKind.FACT: "fact.",
        }
        if self.kind in prefixes:
            prefix = prefixes[self.kind]
            if not self.target.startswith(prefix) or len(self.target) == len(prefix):
                raise ValueError("target name does not match its scorable kind")
        if self.kind is RecursiveTargetKind.FACT and not self.required_fact_ids:
            raise ValueError("fact targets require at least one named fact ID")
        if self.expected_ground_truth_kind is GroundTruthKind.NONE:
            raise ValueError("unscorable targets cannot be recursively constructed")
        return self

    @property
    def target_spec_id(self) -> str:
        return f"recursive_target_{self.content_hash().removeprefix('sha256:')}"


LikelihoodEntry = tuple[StableCategory, Probability]


class RecursiveEvidenceInput(_CanonicalModel):
    """Evidence plus exact lineage and an explicit update interpretation."""

    schema_version: Literal["tom-recursion/1.0.0"] = RECURSION_SCHEMA_VERSION
    trial_id: StableIdentifier
    target: StableCategory
    evidence_id: StableIdentifier
    source_event_id: StableIdentifier
    source_actor_id: StableIdentifier
    observer_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    fact_ids: tuple[StableIdentifier, ...] = ()
    evidence: Evidence
    use_kind: EvidenceUseKind
    hard_category: StableCategory | None = None
    likelihoods: tuple[LikelihoodEntry, ...] = ()
    calibration_id: StableIdentifier | None = None
    update_input_version: StableIdentifier | None = None

    @model_validator(mode="after")
    def _validate_input(self) -> Self:
        _canonical_unique(self.fact_ids, field_name="fact_ids")
        _canonical_pairs(self.likelihoods, field_name="likelihoods")
        lineage = (
            (self.evidence_id, self.evidence.evidence_id, "evidence ID"),
            (
                self.source_event_id,
                self.evidence.source_event_id,
                "source event",
            ),
            (
                self.source_actor_id,
                self.evidence.source_actor_id,
                "source actor",
            ),
            (self.observer_id, self.evidence.observer_id, "observer"),
            (self.turn, self.evidence.turn, "turn"),
        )
        for declared, actual, name in lineage:
            if declared != actual:
                raise ValueError(f"declared {name} does not match Evidence")
        if self.evidence.visibility is EvidenceVisibility.ADJUDICATOR_ONLY:
            raise ValueError("adjudicator-only evidence cannot enter recursion")
        for feature_name, _ in self.evidence.features:
            normalized = re.sub(r"[-./:]+", "_", feature_name)
            if (
                normalized in _FORBIDDEN_EVIDENCE_FEATURE_TOKENS
                or "actual_deception" in normalized
                or "ground_truth_deception" in normalized
                or "deception_ground_truth" in normalized
            ):
                raise ValueError(
                    "private, adjudicator, raw-prompt, or label features "
                    "cannot enter recursion"
                )
        if self.use_kind is EvidenceUseKind.HARD_KNOWLEDGE:
            if self.evidence.channel is not EvidenceChannel.OBSERVABLE:
                raise ValueError("hard knowledge must use observable evidence")
            if self.evidence.reliability != 1.0:
                raise ValueError("hard knowledge requires reliability one")
            if self.hard_category is None:
                raise ValueError("hard knowledge requires a typed category")
            if self.likelihoods:
                raise ValueError("hard knowledge cannot carry soft likelihoods")
            if self.calibration_id is not None or self.update_input_version is not None:
                raise ValueError("hard knowledge cannot claim soft calibration")
        else:
            if self.hard_category is not None:
                raise ValueError("soft evidence cannot carry a hard category")
            if not self.likelihoods:
                raise ValueError("soft evidence requires declared likelihoods")
            if self.calibration_id is None or self.update_input_version is None:
                raise ValueError(
                    "soft evidence requires calibration and update versions"
                )
            if self.evidence.reliability <= 0.0:
                raise ValueError("soft evidence requires positive reliability")
            if not any(value > 0.0 for _, value in self.likelihoods):
                raise ValueError("soft likelihoods require positive support")
        return self

    @property
    def input_id(self) -> str:
        return f"recursive_input_{self.content_hash().removeprefix('sha256:')}"

    @classmethod
    def hard(
        cls,
        *,
        trial_id: str,
        target: str,
        evidence: Evidence,
        hard_category: str,
        fact_ids: tuple[str, ...] = (),
    ) -> Self:
        return cls(
            trial_id=trial_id,
            target=target,
            evidence_id=evidence.evidence_id,
            source_event_id=evidence.source_event_id,
            source_actor_id=evidence.source_actor_id,
            observer_id=evidence.observer_id,
            turn=evidence.turn,
            fact_ids=fact_ids,
            evidence=evidence,
            use_kind=EvidenceUseKind.HARD_KNOWLEDGE,
            hard_category=hard_category,
        )

    @classmethod
    def soft(
        cls,
        *,
        trial_id: str,
        target: str,
        evidence: Evidence,
        likelihoods: tuple[LikelihoodEntry, ...],
        calibration_id: str,
        update_input_version: str,
        fact_ids: tuple[str, ...] = (),
    ) -> Self:
        return cls(
            trial_id=trial_id,
            target=target,
            evidence_id=evidence.evidence_id,
            source_event_id=evidence.source_event_id,
            source_actor_id=evidence.source_actor_id,
            observer_id=evidence.observer_id,
            turn=evidence.turn,
            fact_ids=fact_ids,
            evidence=evidence,
            use_kind=EvidenceUseKind.SOFT_LIKELIHOOD,
            likelihoods=likelihoods,
            calibration_id=calibration_id,
            update_input_version=update_input_version,
        )


EventAccessEntry = tuple[StableIdentifier, AccessBasis]
FactAccessEntry = tuple[StableIdentifier, StableIdentifier, AccessBasis]


class AuthorizedEvidenceUse(_CanonicalModel):
    """Structured explanation of why one evidence item may inform a target."""

    schema_version: Literal["tom-recursion/1.0.0"] = RECURSION_SCHEMA_VERSION
    target_spec_id: StableIdentifier
    evidence_input: RecursiveEvidenceInput
    information_path: tuple[StableIdentifier, ...]
    authorized_actor_ids: tuple[StableIdentifier, ...]
    access_record_ids: tuple[StableIdentifier, ...]
    event_access: tuple[EventAccessEntry, ...]
    fact_access: tuple[FactAccessEntry, ...] = ()

    @model_validator(mode="after")
    def _validate_authorization_record(self) -> Self:
        _canonical_unique(
            self.authorized_actor_ids, field_name="authorized_actor_ids"
        )
        _canonical_unique(
            self.access_record_ids, field_name="access_record_ids"
        )
        if not self.information_path:
            raise ValueError("information_path must not be empty")
        event_keys = tuple(actor_id for actor_id, _ in self.event_access)
        _canonical_unique(event_keys, field_name="event_access")
        fact_keys = tuple(
            f"{actor_id}\0{fact_id}"
            for actor_id, fact_id, _ in self.fact_access
        )
        _canonical_unique(fact_keys, field_name="fact_access")
        if set(event_keys) != set(self.authorized_actor_ids):
            raise ValueError("event access must explain every authorized actor")
        return self

    @property
    def use_id(self) -> str:
        return f"authorized_evidence_{self.content_hash().removeprefix('sha256:')}"


class RecursiveBeliefResult(_CanonicalModel):
    """Recursive state plus its scorable target and authorization explanation."""

    schema_version: Literal["tom-recursion/1.0.0"] = RECURSION_SCHEMA_VERSION
    trial_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    target: ScorableRecursiveTarget
    state: RecursiveBeliefState
    evidence_uses: tuple[AuthorizedEvidenceUse, ...]
    access_record_ids: tuple[StableIdentifier, ...]
    builder_version: StableIdentifier

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        _canonical_unique(
            self.access_record_ids, field_name="access_record_ids"
        )
        if not self.evidence_uses:
            raise ValueError("recursive results require authorized evidence")
        use_ids = tuple(item.use_id for item in self.evidence_uses)
        _canonical_unique(use_ids, field_name="evidence_uses")
        if self.state.target_belief.target != self.target.target:
            raise ValueError("result target does not match recursive state")
        if any(
            item.target_spec_id != self.target.target_spec_id
            for item in self.evidence_uses
        ):
            raise ValueError("evidence use references a different target")
        if any(
            item.information_path != self.state.information_path
            for item in self.evidence_uses
        ):
            raise ValueError("evidence use has a different information path")
        state_evidence_ids = tuple(item.evidence_id for item in self.state.evidence)
        used_evidence_ids = tuple(
            sorted(item.evidence_input.evidence_id for item in self.evidence_uses)
        )
        if state_evidence_ids != used_evidence_ids:
            raise ValueError("recursive state and explanation evidence differ")
        return self

    @property
    def result_id(self) -> str:
        return f"recursive_result_{self.content_hash().removeprefix('sha256:')}"


class DeterministicRecursionBuilder(_CanonicalModel):
    """Build level-one and level-two beliefs under explicit information flow."""

    version: StableIdentifier = "deterministic-recursion-1"
    default_depth: Literal[2] = 2
    maximum_depth: Literal[2] = 2

    def build_level(
        self,
        *,
        trial_id: str,
        turn: int,
        root_state: PartnerBeliefState,
        target: ScorableRecursiveTarget,
        access_records: Iterable[InformationAccessRecord],
        evidence_inputs: Iterable[RecursiveEvidenceInput],
        depth: int | None = None,
        permitted_external_source_ids: tuple[str, ...] = (),
    ) -> RecursiveBeliefResult:
        """Build one recursive level from authorized, target-linked evidence."""
        selected_depth = self.default_depth if depth is None else depth
        self._validate_depth(selected_depth)
        if not isinstance(turn, int) or isinstance(turn, bool) or turn < 0:
            raise ValueError("turn must be a non-negative integer")
        _stable_identifier(trial_id)
        if not isinstance(root_state, PartnerBeliefState):
            raise TypeError("root_state must be a PartnerBeliefState")
        if not isinstance(target, ScorableRecursiveTarget):
            raise TypeError("target must be a ScorableRecursiveTarget")

        path = self._information_path(root_state, selected_depth)
        expected_subject = (
            root_state.counterpart_id
            if selected_depth == 1
            else root_state.observer_id
        )
        if target.subject_actor_id != expected_subject:
            raise ValueError(
                "target subject does not match the recursive level semantics"
            )
        prior = self._target_prior(root_state, target)
        access_by_actor = self._access_map(
            access_records,
            trial_id=trial_id,
            turn=turn,
            root_state=root_state,
        )
        external_sources = tuple(permitted_external_source_ids)
        _canonical_unique(
            external_sources, field_name="permitted_external_source_ids"
        )
        for source_id in external_sources:
            _stable_identifier(source_id)
        adjudicator_actors = set(
            next(iter(access_by_actor.values())).adjudicator_actor_ids
        )
        if set(external_sources) & (set(path) | adjudicator_actors):
            raise ValueError(
                "external sources must be outside the path and adjudicators"
            )

        inputs = self._evidence_inputs(
            evidence_inputs,
            trial_id=trial_id,
            turn=turn,
            target=target,
        )
        uses = tuple(
            self._authorize_input(
                item,
                target=target,
                path=path,
                access_by_actor=access_by_actor,
                root_state=root_state,
                depth=selected_depth,
                external_sources=external_sources,
            )
            for item in inputs
        )
        posterior = self._posterior(prior, inputs)
        evidence = tuple(
            sorted(
                (item.evidence for item in inputs),
                key=lambda item: item.evidence_id,
            )
        )
        state = RecursiveBeliefState(
            depth=selected_depth,
            root_observer_id=root_state.observer_id,
            counterpart_id=root_state.counterpart_id,
            root_state_hash=root_state.state_hash,
            information_path=path,
            target_belief=posterior,
            evidence=evidence,
            permitted_external_sources=external_sources,
        )
        uses = tuple(sorted(uses, key=lambda item: item.use_id))
        return RecursiveBeliefResult(
            trial_id=trial_id,
            turn=turn,
            target=target,
            state=state,
            evidence_uses=uses,
            access_record_ids=tuple(
                sorted(item.access_id for item in access_by_actor.values())
            ),
            builder_version=self.version,
        )

    def build_both_levels(
        self,
        *,
        trial_id: str,
        turn: int,
        root_state: PartnerBeliefState,
        level_one_target: ScorableRecursiveTarget,
        level_two_target: ScorableRecursiveTarget,
        access_records: Iterable[InformationAccessRecord],
        level_one_evidence: Iterable[RecursiveEvidenceInput],
        level_two_evidence: Iterable[RecursiveEvidenceInput],
        permitted_external_source_ids: tuple[str, ...] = (),
    ) -> tuple[RecursiveBeliefResult, RecursiveBeliefResult]:
        """Build both levels independently; level two is never confidence decay."""
        records = tuple(access_records)
        return (
            self.build_level(
                trial_id=trial_id,
                turn=turn,
                root_state=root_state,
                target=level_one_target,
                access_records=records,
                evidence_inputs=level_one_evidence,
                depth=1,
                permitted_external_source_ids=permitted_external_source_ids,
            ),
            self.build_level(
                trial_id=trial_id,
                turn=turn,
                root_state=root_state,
                target=level_two_target,
                access_records=records,
                evidence_inputs=level_two_evidence,
                depth=2,
                permitted_external_source_ids=permitted_external_source_ids,
            ),
        )

    def _validate_depth(self, depth: Any) -> None:
        if not isinstance(depth, int) or isinstance(depth, bool):
            raise ValueError("recursive depth must be the integer one or two")
        if depth < 1 or depth > self.maximum_depth:
            raise ValueError("recursive depth must be one or two")

    @staticmethod
    def _information_path(
        root_state: PartnerBeliefState, depth: int
    ) -> tuple[str, ...]:
        if depth == 1:
            return (root_state.observer_id, root_state.counterpart_id)
        return (
            root_state.observer_id,
            root_state.counterpart_id,
            root_state.observer_id,
        )

    @staticmethod
    def _target_prior(
        root_state: PartnerBeliefState,
        target: ScorableRecursiveTarget,
    ) -> BeliefDistribution:
        beliefs = (
            root_state.policy_type,
            root_state.expected_next_action,
            root_state.reservation_value,
            *root_state.goal_beliefs,
            *root_state.constraint_beliefs,
            *root_state.fact_beliefs,
            root_state.trustworthiness,
        )
        matches = tuple(item for item in beliefs if item.target == target.target)
        if len(matches) != 1:
            raise ValueError(
                "recursive target is absent or not uniquely scorable in root state"
            )
        prior = matches[0]
        if prior.ground_truth_kind is GroundTruthKind.NONE:
            raise ValueError("recursive target has no scorable ground truth")
        if prior.ground_truth_kind is not target.expected_ground_truth_kind:
            raise ValueError("target ground-truth semantics do not match root state")
        if prior.unknown_category not in prior.categories:
            raise ValueError("recursive target must retain an unknown bucket")
        return prior

    @staticmethod
    def _access_map(
        records: Iterable[InformationAccessRecord],
        *,
        trial_id: str,
        turn: int,
        root_state: PartnerBeliefState,
    ) -> dict[str, InformationAccessRecord]:
        try:
            items = tuple(records)
        except TypeError as error:
            raise TypeError(
                "access_records must be an iterable of InformationAccessRecord"
            ) from error
        if any(not isinstance(item, InformationAccessRecord) for item in items):
            raise TypeError(
                "access_records must contain InformationAccessRecord values"
            )
        expected_actors = {root_state.observer_id, root_state.counterpart_id}
        actor_ids = tuple(item.actor_id for item in items)
        if len(set(actor_ids)) != len(actor_ids):
            raise ValueError("access records must have unique actor IDs")
        if set(actor_ids) != expected_actors:
            raise ValueError(
                "access records must cover exactly root and counterpart actors"
            )
        for item in items:
            if item.trial_id != trial_id:
                raise ValueError("access record crosses trial boundary")
            if item.through_turn != turn:
                raise ValueError("access record turn does not match build turn")
        first = items[0]
        shared_fields = (
            "public_fact_ids",
            "public_event_ids",
            "adjudicator_only_fact_ids",
            "adjudicator_only_event_ids",
            "adjudicator_actor_ids",
        )
        for item in items[1:]:
            if any(
                getattr(item, field_name) != getattr(first, field_name)
                for field_name in shared_fields
            ):
                raise ValueError(
                    "access records disagree on public/adjudicator universe"
                )
        return {item.actor_id: item for item in items}

    @staticmethod
    def _evidence_inputs(
        values: Iterable[RecursiveEvidenceInput],
        *,
        trial_id: str,
        turn: int,
        target: ScorableRecursiveTarget,
    ) -> tuple[RecursiveEvidenceInput, ...]:
        try:
            inputs = tuple(values)
        except TypeError as error:
            raise TypeError(
                "evidence_inputs must be an iterable of RecursiveEvidenceInput"
            ) from error
        if not inputs:
            raise ValueError("insufficient authorized evidence for recursive target")
        if any(not isinstance(item, RecursiveEvidenceInput) for item in inputs):
            raise TypeError(
                "evidence_inputs must contain RecursiveEvidenceInput values"
            )
        for item in inputs:
            if item.trial_id != trial_id:
                raise ValueError("recursive evidence crosses trial boundary")
            if item.target != target.target:
                raise ValueError("recursive evidence targets a different variable")
            if item.turn > turn:
                raise ValueError("future-turn evidence cannot enter recursion")
        evidence_ids = tuple(item.evidence_id for item in inputs)
        event_ids = tuple(item.source_event_id for item in inputs)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("recursive evidence IDs must be unique")
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("recursive source event IDs must be unique")
        return tuple(sorted(inputs, key=lambda item: item.evidence_id))

    @staticmethod
    def _authorize_input(
        item: RecursiveEvidenceInput,
        *,
        target: ScorableRecursiveTarget,
        path: tuple[str, ...],
        access_by_actor: dict[str, InformationAccessRecord],
        root_state: PartnerBeliefState,
        depth: int,
        external_sources: tuple[str, ...],
    ) -> AuthorizedEvidenceUse:
        actors = tuple(sorted(set(path)))
        adjudicator_actors = set(
            next(iter(access_by_actor.values())).adjudicator_actor_ids
        )
        if item.source_actor_id in adjudicator_actors:
            raise ValueError("adjudicator source cannot enter recursion")
        allowed_sources = set(path) | set(external_sources)
        if item.source_actor_id not in allowed_sources:
            raise ValueError("evidence source actor is outside the modeled path")
        if item.observer_id != root_state.counterpart_id:
            raise ValueError(
                "recursive evidence must represent the counterpart's view"
            )
        if depth == 2 and item.source_actor_id != root_state.observer_id:
            raise ValueError(
                "level-two evidence must concern the root actor's observable behavior"
            )
        if not all(item.evidence.is_visible_to(actor) for actor in actors):
            raise ValueError("evidence is not visible along the complete path")

        required_facts = tuple(
            sorted(set(item.fact_ids) | set(target.required_fact_ids))
        )
        event_access: list[EventAccessEntry] = []
        fact_access: list[FactAccessEntry] = []
        for actor_id in actors:
            access = access_by_actor[actor_id]
            if item.source_event_id in access.adjudicator_only_event_ids:
                raise ValueError("adjudicator-only event cannot enter recursion")
            event_basis = access.event_basis(item.source_event_id)
            if event_basis is None:
                raise ValueError(
                    f"event is unavailable to modeled actor {actor_id!r}"
                )
            event_access.append((actor_id, event_basis))
            for fact_id in required_facts:
                if fact_id in access.adjudicator_only_fact_ids:
                    raise ValueError(
                        "adjudicator-only fact cannot enter recursion"
                    )
                fact_basis = access.fact_basis(fact_id)
                if fact_basis is None:
                    raise ValueError(
                        f"fact is unavailable to modeled actor {actor_id!r}"
                    )
                fact_access.append((actor_id, fact_id, fact_basis))

        return AuthorizedEvidenceUse(
            target_spec_id=target.target_spec_id,
            evidence_input=item,
            information_path=path,
            authorized_actor_ids=actors,
            access_record_ids=tuple(
                sorted(access_by_actor[actor].access_id for actor in actors)
            ),
            event_access=tuple(sorted(event_access, key=lambda value: value[0])),
            fact_access=tuple(
                sorted(fact_access, key=lambda value: (value[0], value[1]))
            ),
        )

    @staticmethod
    def _posterior(
        prior: BeliefDistribution,
        inputs: tuple[RecursiveEvidenceInput, ...],
    ) -> BeliefDistribution:
        hard = tuple(
            item
            for item in inputs
            if item.use_kind is EvidenceUseKind.HARD_KNOWLEDGE
        )
        soft = tuple(
            item
            for item in inputs
            if item.use_kind is EvidenceUseKind.SOFT_LIKELIHOOD
        )
        if hard and soft:
            raise ValueError(
                "hard knowledge and soft likelihoods require separate targets"
            )
        if hard:
            categories = {item.hard_category for item in hard}
            if len(categories) != 1:
                raise ValueError("hard evidence gives contradictory categories")
            category = next(iter(categories))
            if category not in prior.categories:
                raise ValueError("hard category is outside target hypotheses")
            probabilities = tuple(
                1.0 if candidate == category else 0.0
                for candidate in prior.categories
            )
        else:
            expected_categories = prior.categories
            log_weights: list[float] = []
            for category, prior_probability in zip(
                expected_categories,
                prior.probabilities,
                strict=True,
            ):
                if prior_probability <= 0.0:
                    log_weights.append(-math.inf)
                    continue
                log_weight = math.log(prior_probability)
                supported = True
                for item in soft:
                    likelihood_by_category = dict(item.likelihoods)
                    if tuple(likelihood_by_category) != expected_categories:
                        raise ValueError(
                            "soft likelihood categories must exactly align with target"
                        )
                    likelihood = likelihood_by_category[category]
                    if likelihood <= 0.0:
                        supported = False
                        break
                    log_weight += math.log(likelihood)
                log_weights.append(log_weight if supported else -math.inf)
            finite = tuple(value for value in log_weights if math.isfinite(value))
            if not finite:
                raise ValueError("soft evidence leaves no positive posterior mass")
            maximum = max(finite)
            weights = tuple(
                0.0 if not math.isfinite(value) else math.exp(value - maximum)
                for value in log_weights
            )
            total = math.fsum(weights)
            probabilities_list = [value / total for value in weights]
            largest = max(
                range(len(probabilities_list)),
                key=probabilities_list.__getitem__,
            )
            probabilities_list[largest] += 1.0 - math.fsum(probabilities_list)
            probabilities = tuple(probabilities_list)

        return BeliefDistribution(
            target=prior.target,
            categories=prior.categories,
            probabilities=probabilities,
            unknown_category=prior.unknown_category,
            epistemic_status=EpistemicStatus.UPDATED,
            ground_truth_kind=prior.ground_truth_kind,
        )


def build_recursive_level(
    *,
    trial_id: str,
    turn: int,
    root_state: PartnerBeliefState,
    target: ScorableRecursiveTarget,
    access_records: Iterable[InformationAccessRecord],
    evidence_inputs: Iterable[RecursiveEvidenceInput],
    depth: int = 2,
    permitted_external_source_ids: tuple[str, ...] = (),
    builder: DeterministicRecursionBuilder | None = None,
) -> RecursiveBeliefResult:
    """Convenience helper for a single level, defaulting to maximum depth two."""
    selected_builder = builder or DeterministicRecursionBuilder()
    if not isinstance(selected_builder, DeterministicRecursionBuilder):
        raise TypeError("builder must be a DeterministicRecursionBuilder")
    return selected_builder.build_level(
        trial_id=trial_id,
        turn=turn,
        root_state=root_state,
        target=target,
        access_records=access_records,
        evidence_inputs=evidence_inputs,
        depth=depth,
        permitted_external_source_ids=permitted_external_source_ids,
    )


def build_recursive_levels(
    *,
    trial_id: str,
    turn: int,
    root_state: PartnerBeliefState,
    level_one_target: ScorableRecursiveTarget,
    level_two_target: ScorableRecursiveTarget,
    access_records: Iterable[InformationAccessRecord],
    level_one_evidence: Iterable[RecursiveEvidenceInput],
    level_two_evidence: Iterable[RecursiveEvidenceInput],
    permitted_external_source_ids: tuple[str, ...] = (),
    builder: DeterministicRecursionBuilder | None = None,
) -> tuple[RecursiveBeliefResult, RecursiveBeliefResult]:
    """Convenience helper that independently constructs levels one and two."""
    selected_builder = builder or DeterministicRecursionBuilder()
    if not isinstance(selected_builder, DeterministicRecursionBuilder):
        raise TypeError("builder must be a DeterministicRecursionBuilder")
    return selected_builder.build_both_levels(
        trial_id=trial_id,
        turn=turn,
        root_state=root_state,
        level_one_target=level_one_target,
        level_two_target=level_two_target,
        access_records=access_records,
        level_one_evidence=level_one_evidence,
        level_two_evidence=level_two_evidence,
        permitted_external_source_ids=permitted_external_source_ids,
    )


__all__ = [
    "AccessBasis",
    "AuthorizedEvidenceUse",
    "DeterministicRecursionBuilder",
    "EvidenceUseKind",
    "InformationAccessRecord",
    "RECURSION_SCHEMA_VERSION",
    "RecursiveBeliefResult",
    "RecursiveEvidenceInput",
    "RecursiveTargetKind",
    "ScorableRecursiveTarget",
    "build_recursive_level",
    "build_recursive_levels",
]
