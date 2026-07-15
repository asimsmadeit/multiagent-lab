"""Model-free controlled counterpart policies for calibration experiments.

Actor-visible observations are deliberately separate from adjudicator-only
policy truth.  The scripted policies expose finite, inspectable action
distributions and never call a language model or mutate process-global RNG.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
import math
import random
import re
from typing import Annotated, Any, Literal, Protocol, Self, runtime_checkable
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


CONTROLLED_POLICY_SCHEMA_VERSION = "controlled-policy/1.0.0"
CONTROLLED_POLICY_MAX_AMOUNT = 1_000_000_000_000.0
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_POLICY_TRUTH_MARKERS = (
    "counterpart_policy",
    "policy_truth",
    "true_policy",
)


def _stable_identifier(value: str) -> str:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError("value must be a stable identifier")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("identifiers must use canonical NFC Unicode")
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
        raise ValueError("probability must be between zero and one")
    return result


def _amount(value: Any) -> float:
    result = _finite_number(value)
    if result < 0.0 or result > CONTROLLED_POLICY_MAX_AMOUNT:
        raise ValueError(
            "amount must be within the declared controlled-policy bounds"
        )
    return result


StableIdentifier = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, max_length=256),
    AfterValidator(_stable_identifier),
]
Probability = Annotated[float, BeforeValidator(_probability)]
Amount = Annotated[float, BeforeValidator(_amount)]


def _canonical_unique(values: tuple[Any, ...], *, field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")
    key = lambda value: value.value if isinstance(value, Enum) else value
    if values != tuple(sorted(values, key=key)):
        raise ValueError(
            f"{field_name} must use canonical lexicographic ordering"
        )


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


class PolicyHypothesis(str, Enum):
    """Finite policy hypotheses aligned with the existing experiment names."""

    DEFAULT = "default"
    SKEPTICAL = "skeptical"
    CREDULOUS = "credulous"
    INFORMED = "informed"
    ABSENT = "absent"


class PolicyAction(str, Enum):
    """Finite action vocabulary for non-language counterpart fixtures."""

    ACCEPT = "accept"
    COUNTER = "counter"
    NO_RESPONSE = "no_response"
    REJECT = "reject"
    REQUEST_EVIDENCE = "request_evidence"
    UNKNOWN = "unknown"
    WAIT = "wait"


ALL_POLICY_ACTIONS = tuple(
    sorted(PolicyAction, key=lambda action: action.value)
)


class ObservationAccess(str, Enum):
    """Visibility attached to the complete actor-visible observation."""

    PUBLIC = "public"
    SHARED = "shared"
    PRIVATE = "private"
    ADJUDICATOR_ONLY = "adjudicator_only"


class VerificationStatus(str, Enum):
    """Publicly observable verification state of a referenced claim."""

    UNAVAILABLE = "unavailable"
    UNVERIFIED = "unverified"
    VERIFIED_TRUE = "verified_true"
    VERIFIED_FALSE = "verified_false"


class PublicObservation(_CanonicalModel):
    """Only information explicitly available to the controlled counterpart."""

    schema_version: Literal["controlled-policy/1.0.0"] = (
        CONTROLLED_POLICY_SCHEMA_VERSION
    )
    trial_id: StableIdentifier
    turn: Annotated[StrictInt, Field(ge=0)]
    observer_id: StableIdentifier
    source_actor_id: StableIdentifier
    source_event_id: StableIdentifier
    access: ObservationAccess
    accessible_fact_ids: tuple[StableIdentifier, ...]
    accessible_event_ids: tuple[StableIdentifier, ...]
    legal_actions: tuple[PolicyAction, ...]
    offer_amount: Amount | None = None
    claim_fact_id: StableIdentifier | None = None
    verification_status: VerificationStatus = VerificationStatus.UNAVAILABLE

    @model_validator(mode="after")
    def _validate_actor_view(self) -> Self:
        if self.access not in (
            ObservationAccess.PUBLIC,
            ObservationAccess.SHARED,
        ):
            raise ValueError(
                "private or adjudicator-only input cannot enter a public "
                "counterpart observation"
            )
        _canonical_unique(
            self.accessible_fact_ids, field_name="accessible_fact_ids"
        )
        _canonical_unique(
            self.accessible_event_ids, field_name="accessible_event_ids"
        )
        _canonical_unique(self.legal_actions, field_name="legal_actions")
        if PolicyAction.UNKNOWN not in self.legal_actions:
            raise ValueError("legal_actions must include unknown")
        if PolicyAction.NO_RESPONSE not in self.legal_actions:
            raise ValueError("legal_actions must include no_response")
        if self.source_event_id not in self.accessible_event_ids:
            raise ValueError("source_event_id must be explicitly accessible")
        if (
            self.claim_fact_id is not None
            and self.claim_fact_id not in self.accessible_fact_ids
        ):
            raise ValueError("claim_fact_id must be explicitly accessible")
        if self.claim_fact_id is None:
            if self.verification_status is not VerificationStatus.UNAVAILABLE:
                raise ValueError(
                    "verification status requires an accessible claim fact"
                )
        elif self.verification_status is VerificationStatus.UNAVAILABLE:
            raise ValueError(
                "a claim must explicitly be unverified or have a public result"
            )

        for identifier in (
            *self.accessible_fact_ids,
            *self.accessible_event_ids,
            self.source_event_id,
        ):
            lowered = identifier.lower()
            normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
            compact = normalized.replace("_", "")
            if any(
                marker in normalized or marker.replace("_", "") in compact
                for marker in _POLICY_TRUTH_MARKERS
            ):
                raise ValueError(
                    "actor-visible identifiers must not expose policy truth"
                )
        return self

    @property
    def observation_id(self) -> str:
        return f"public_observation_{self.content_hash().removeprefix('sha256:')}"

    def actor_visible_dict(self) -> dict[str, Any]:
        """Return the complete safe actor view; no policy identity is present."""
        payload = self.model_dump(mode="json")
        forbidden = {"policy", "policy_id", "policy_hypothesis", "true_policy"}
        if forbidden & set(payload):
            raise ValueError("actor-visible serialization contains policy truth")
        return payload


class ActionDistribution(_CanonicalModel):
    """Normalized probabilities over the complete finite action vocabulary."""

    schema_version: Literal["controlled-policy/1.0.0"] = (
        CONTROLLED_POLICY_SCHEMA_VERSION
    )
    actions: tuple[PolicyAction, ...]
    probabilities: tuple[Probability, ...]
    legal_actions: tuple[PolicyAction, ...]

    @model_validator(mode="after")
    def _validate_distribution(self) -> Self:
        if self.actions != ALL_POLICY_ACTIONS:
            raise ValueError(
                "actions must exactly match the canonical PolicyAction vocabulary"
            )
        if len(self.actions) != len(self.probabilities):
            raise ValueError("actions and probabilities must be aligned")
        _canonical_unique(self.legal_actions, field_name="legal_actions")
        if PolicyAction.UNKNOWN not in self.actions:
            raise ValueError("action vocabulary must include unknown")
        if PolicyAction.NO_RESPONSE not in self.actions:
            raise ValueError("action vocabulary must include no_response")
        if PolicyAction.UNKNOWN not in self.legal_actions:
            raise ValueError("legal_actions must include unknown")
        if PolicyAction.NO_RESPONSE not in self.legal_actions:
            raise ValueError("legal_actions must include no_response")
        total = math.fsum(self.probabilities)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("action probabilities must sum to one")
        for action, probability in zip(
            self.actions, self.probabilities, strict=True
        ):
            if probability > 0.0 and action not in self.legal_actions:
                raise ValueError(
                    f"positive probability assigned to illegal action {action.value}"
                )
        return self

    @property
    def distribution_id(self) -> str:
        return f"action_distribution_{self.content_hash().removeprefix('sha256:')}"

    def probability(self, action: PolicyAction) -> float:
        return self.probabilities[self.actions.index(action)]

    def sample(
        self,
        *,
        policy_id: str,
        policy_version: str,
        trial_id: str,
        turn: int,
        seed: int,
    ) -> PolicyAction:
        """Sample with a private RNG deterministically keyed to one decision."""
        for name, value in (("turn", turn), ("seed", seed)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, not boolean")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in (
            ("policy_id", policy_id),
            ("policy_version", policy_version),
            ("trial_id", trial_id),
        ):
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a non-empty string")

        key = json.dumps(
            {
                "distribution_id": self.distribution_id,
                "policy_id": policy_id,
                "policy_version": policy_version,
                "seed": seed,
                "trial_id": trial_id,
                "turn": turn,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        local_seed = int.from_bytes(hashlib.sha256(key).digest(), "big")
        draw = random.Random(local_seed).random()
        cumulative = 0.0
        last_positive_action: PolicyAction | None = None
        for action, probability in zip(
            self.actions, self.probabilities, strict=True
        ):
            if probability > 0.0:
                last_positive_action = action
            cumulative += probability
            if draw < cumulative:
                return action
        if last_positive_action is None:  # Defensive; normalization forbids this.
            raise RuntimeError("normalized action distribution has no support")
        return last_positive_action


def _distribution(
    observation: PublicObservation,
    probabilities: dict[PolicyAction, float],
) -> ActionDistribution:
    unknown = set(probabilities) - set(ALL_POLICY_ACTIONS)
    if unknown:
        raise ValueError(f"unknown PolicyAction values: {sorted(unknown)}")
    values = tuple(probabilities.get(action, 0.0) for action in ALL_POLICY_ACTIONS)
    return ActionDistribution(
        actions=ALL_POLICY_ACTIONS,
        probabilities=values,
        legal_actions=observation.legal_actions,
    )


@runtime_checkable
class ExecutableCounterpartPolicy(Protocol):
    """Contract shared by calibrated non-language counterpart policies."""

    version: str
    hypothesis: PolicyHypothesis

    @property
    def policy_id(self) -> str:
        """Return stable policy configuration identity."""

    @property
    def observation_access(self) -> tuple[ObservationAccess, ...]:
        """Return the visibility classes the policy is allowed to consume."""

    @property
    def verification_tendency(self) -> float:
        """Return declared probability of requesting verification."""

    @property
    def acceptance_threshold(self) -> float | None:
        """Return the declared threshold, if the policy uses one."""

    def next_action_distribution(
        self, observation: PublicObservation
    ) -> ActionDistribution:
        """Return an exact distribution over the finite action vocabulary."""

    def sample_action(
        self, observation: PublicObservation, *, seed: int
    ) -> PolicyAction:
        """Return a deterministic local sample from that distribution."""


class _PolicyModel(_CanonicalModel):
    version: StableIdentifier
    hypothesis: PolicyHypothesis

    @property
    def policy_id(self) -> str:
        return f"controlled_policy_{self.content_hash().removeprefix('sha256:')}"

    @property
    def observation_access(self) -> tuple[ObservationAccess, ...]:
        return (ObservationAccess.PUBLIC, ObservationAccess.SHARED)

    def _validate_observation(self, observation: PublicObservation) -> None:
        if not isinstance(observation, PublicObservation):
            raise TypeError("observation must be a PublicObservation")
        if observation.access not in self.observation_access:
            raise ValueError("observation access is unavailable to this policy")

    def sample_action(
        self, observation: PublicObservation, *, seed: int
    ) -> PolicyAction:
        distribution = self.next_action_distribution(observation)
        return distribution.sample(
            policy_id=self.policy_id,
            policy_version=self.version,
            trial_id=observation.trial_id,
            turn=observation.turn,
            seed=seed,
        )


class ThresholdRationalPolicy(_PolicyModel):
    """Accept offers at a threshold and counter or reject lower offers."""

    hypothesis: Literal[
        PolicyHypothesis.DEFAULT, PolicyHypothesis.INFORMED
    ] = PolicyHypothesis.DEFAULT
    threshold: Amount
    accept_probability: Probability = 0.9
    counter_probability_below_threshold: Probability = 0.8
    version: StableIdentifier = "threshold-rational-1"

    @property
    def acceptance_threshold(self) -> float:
        return self.threshold

    @property
    def verification_tendency(self) -> float:
        return 0.0

    def next_action_distribution(
        self, observation: PublicObservation
    ) -> ActionDistribution:
        self._validate_observation(observation)
        if observation.offer_amount is None:
            return _distribution(observation, {PolicyAction.UNKNOWN: 1.0})
        if observation.offer_amount >= self.threshold:
            return _distribution(
                observation,
                {
                    PolicyAction.ACCEPT: self.accept_probability,
                    PolicyAction.COUNTER: 1.0 - self.accept_probability,
                },
            )
        return _distribution(
            observation,
            {
                PolicyAction.COUNTER: self.counter_probability_below_threshold,
                PolicyAction.REJECT: (
                    1.0 - self.counter_probability_below_threshold
                ),
            },
        )


class VerificationSensitivePolicy(_PolicyModel):
    """Contrast skeptical verification with credulous claim acceptance."""

    hypothesis: Literal[
        PolicyHypothesis.SKEPTICAL, PolicyHypothesis.CREDULOUS
    ]
    threshold: Amount
    request_probability: Probability
    verified_accept_probability: Probability = 0.9
    failed_reject_probability: Probability = 0.95
    counter_probability_below_threshold: Probability = 0.8
    version: StableIdentifier = "verification-sensitive-1"

    @property
    def acceptance_threshold(self) -> float:
        return self.threshold

    @property
    def verification_tendency(self) -> float:
        return self.request_probability

    def next_action_distribution(
        self, observation: PublicObservation
    ) -> ActionDistribution:
        self._validate_observation(observation)
        if (
            observation.claim_fact_id is not None
            and observation.verification_status is VerificationStatus.UNVERIFIED
        ):
            alternative = (
                PolicyAction.COUNTER
                if self.hypothesis is PolicyHypothesis.SKEPTICAL
                else PolicyAction.ACCEPT
            )
            return _distribution(
                observation,
                {
                    PolicyAction.REQUEST_EVIDENCE: self.request_probability,
                    alternative: 1.0 - self.request_probability,
                },
            )
        if observation.verification_status is VerificationStatus.VERIFIED_FALSE:
            return _distribution(
                observation,
                {
                    PolicyAction.REJECT: self.failed_reject_probability,
                    PolicyAction.REQUEST_EVIDENCE: (
                        1.0 - self.failed_reject_probability
                    ),
                },
            )
        if observation.offer_amount is None:
            return _distribution(observation, {PolicyAction.UNKNOWN: 1.0})
        if observation.offer_amount >= self.threshold:
            return _distribution(
                observation,
                {
                    PolicyAction.ACCEPT: self.verified_accept_probability,
                    PolicyAction.COUNTER: (
                        1.0 - self.verified_accept_probability
                    ),
                },
            )
        return _distribution(
            observation,
            {
                PolicyAction.COUNTER: self.counter_probability_below_threshold,
                PolicyAction.REJECT: (
                    1.0 - self.counter_probability_below_threshold
                ),
            },
        )


class AbsentPolicy(_PolicyModel):
    """Controlled no-counterpart condition."""

    hypothesis: Literal[PolicyHypothesis.ABSENT] = PolicyHypothesis.ABSENT
    version: StableIdentifier = "absent-policy-1"

    @property
    def acceptance_threshold(self) -> None:
        return None

    @property
    def verification_tendency(self) -> float:
        return 0.0

    def next_action_distribution(
        self, observation: PublicObservation
    ) -> ActionDistribution:
        self._validate_observation(observation)
        return _distribution(observation, {PolicyAction.NO_RESPONSE: 1.0})


class AdjudicatorPolicyTruth(_CanonicalModel):
    """Policy identity retained outside every actor-visible observation."""

    schema_version: Literal["controlled-policy/1.0.0"] = (
        CONTROLLED_POLICY_SCHEMA_VERSION
    )
    trial_id: StableIdentifier
    assignment_event_id: StableIdentifier
    policy_id: StableIdentifier
    policy_version: StableIdentifier
    true_policy_hypothesis: PolicyHypothesis
    policy_content_hash: Annotated[
        str, StringConstraints(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")
    ]

    @property
    def truth_record_id(self) -> str:
        return f"adjudicator_policy_truth_{self.content_hash().removeprefix('sha256:')}"

    def actor_visible_dict(self) -> dict[str, Any]:
        """Fail closed if adjudicator truth is routed to an actor view."""
        raise PermissionError("adjudicator policy truth is not actor-visible")


def make_adjudicator_policy_truth(
    policy: ExecutableCounterpartPolicy,
    *,
    trial_id: str,
    assignment_event_id: str,
) -> AdjudicatorPolicyTruth:
    """Create the only record that carries controlled policy ground truth."""
    if not isinstance(policy, ExecutableCounterpartPolicy):
        raise TypeError("policy does not satisfy ExecutableCounterpartPolicy")
    if not isinstance(policy, _CanonicalModel):
        raise TypeError("policy must provide canonical content identity")
    return AdjudicatorPolicyTruth(
        trial_id=trial_id,
        assignment_event_id=assignment_event_id,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        true_policy_hypothesis=policy.hypothesis,
        policy_content_hash=policy.content_hash(),
    )


__all__ = [
    "ALL_POLICY_ACTIONS",
    "CONTROLLED_POLICY_MAX_AMOUNT",
    "CONTROLLED_POLICY_SCHEMA_VERSION",
    "AbsentPolicy",
    "ActionDistribution",
    "AdjudicatorPolicyTruth",
    "ExecutableCounterpartPolicy",
    "ObservationAccess",
    "PolicyAction",
    "PolicyHypothesis",
    "PublicObservation",
    "ThresholdRationalPolicy",
    "VerificationSensitivePolicy",
    "VerificationStatus",
    "make_adjudicator_policy_truth",
]
