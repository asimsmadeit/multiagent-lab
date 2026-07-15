"""Typed contracts for bilateral dyadic capture (Plan 4, Phases 1-4).

Records are normalized by actor turn rather than "main" versus "counterpart".
Every activation view is bound to one model call, one actor, one capture
stage, and one content-addressed artifact; a stage a backend cannot provide is
recorded as explicitly unavailable rather than silently substituted.

Digest conventions follow the neighbouring packages: canonical scenario/event
content references are ``sha256:``-prefixed (``interpretability.scenarios``),
while Theory-of-Mind belief-state hashes are plain lowercase hex digests
(``negotiation.components.tom``).
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Annotated, Any, ClassVar, Mapping, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    model_validator,
)
from typing import get_args, get_origin

from interpretability.scenarios.schema import canonical_json, canonical_sha256

DYAD_SCHEMA_VERSION = "1.0"

_PREFIXED_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_PLAIN_SHA256 = re.compile(r"^[0-9a-f]{64}$")


Identifier = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=256)
]
ShortText = Annotated[
    str, StringConstraints(strict=True, min_length=1, max_length=512)
]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


def _prefixed_digest(value: str) -> str:
    if not _PREFIXED_SHA256.fullmatch(value):
        raise ValueError("expected a 'sha256:'-prefixed lowercase hex digest")
    return value


def _plain_digest(value: str) -> str:
    if not _PLAIN_SHA256.fullmatch(value):
        raise ValueError("expected a plain lowercase sha256 hex digest")
    return value


from pydantic import AfterValidator  # noqa: E402  (grouped with peers above)

CanonicalRef = Annotated[
    str, StringConstraints(strict=True), AfterValidator(_prefixed_digest)
]
BeliefStateDigest = Annotated[
    str, StringConstraints(strict=True), AfterValidator(_plain_digest)
]


class CaptureMode(str, Enum):
    """Which side(s) of the dyad expose internals (Plan 4, Phase 1)."""

    BOTH_WHITE_BOX = "both_white_box"
    ACTOR_WHITE_BOX = "actor_white_box"
    RECEIVER_WHITE_BOX = "receiver_white_box"
    TEXT_ONLY = "text_only"


class CaptureStage(str, Enum):
    """Backend-neutral activation capture stages (Plan 4, Phase 4)."""

    PREFILL_LAST = "prefill_last"
    GENERATED_LAST = "generated_last"
    GENERATED_MEAN = "generated_mean"
    EVIDENCE_SPAN = "evidence_span"
    MESSAGE_READ_SPAN = "message_read_span"


#: Stages defined over a token span rather than a single position.
SPAN_STAGES = frozenset(
    {CaptureStage.GENERATED_MEAN, CaptureStage.EVIDENCE_SPAN,
     CaptureStage.MESSAGE_READ_SPAN}
)


class StageUnavailableReason(str, Enum):
    """Why a declared capture stage has no activation view."""

    BACKEND_UNSUPPORTED = "backend_unsupported"
    CAPTURE_MODE_DISABLED = "capture_mode_disabled"
    TOKEN_ALIGNMENT_FAILED = "token_alignment_failed"
    ARTIFACT_MISSING = "artifact_missing"


class Aggregation(str, Enum):
    """Token aggregation applied to the captured tensor."""

    NONE = "none"
    MEAN = "mean"


class FirstMoverSource(str, Enum):
    """Whether first mover was randomized or fixed by the scenario."""

    RANDOMIZED = "randomized"
    SCENARIO_FIXED = "scenario_fixed"


def _annotation_contains_tuple(annotation: Any) -> bool:
    if get_origin(annotation) is tuple:
        return True
    return any(_annotation_contains_tuple(item) for item in get_args(annotation))


class _DyadBaseModel(BaseModel):
    """Shared strict/frozen policy for every dyadic record."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        allow_inf_nan=False,
    )

    @model_validator(mode="before")
    @classmethod
    def _restore_json_arrays_as_tuples(
        cls, value: Any, info: ValidationInfo
    ) -> Any:
        if not (info.context and info.context.get("persisted")):
            return value
        if not isinstance(value, Mapping):
            return value
        restored = dict(value)
        for field_name, field_info in cls.model_fields.items():
            field_value = restored.get(field_name)
            if isinstance(field_value, list) and _annotation_contains_tuple(
                field_info.annotation
            ):
                restored[field_name] = tuple(field_value)
        return restored


class _ContentBoundModel(_DyadBaseModel):
    """Derive and verify one subclass-specific content identifier."""

    _content_field: ClassVar[str]

    @model_validator(mode="before")
    @classmethod
    def _require_persisted_identifier(
        cls, value: Any, info: ValidationInfo
    ) -> Any:
        if info.context and info.context.get("persisted"):
            if not isinstance(value, Mapping):
                raise TypeError("persisted dyadic objects must be JSON objects")
            if "schema_version" not in value:
                raise ValueError(
                    "schema_version is required on persisted dyadic objects"
                )
            persisted_id = value.get(cls._content_field)
            if not isinstance(persisted_id, str) or not persisted_id:
                raise ValueError(
                    f"{cls._content_field} is required on persisted objects"
                )
        return value

    @model_validator(mode="after")
    def _bind_content_identifier(self) -> Self:
        payload = self.model_dump(mode="json", exclude={self._content_field})
        expected = canonical_sha256(payload)
        supplied = getattr(self, self._content_field)
        if supplied and supplied != expected:
            raise ValueError(
                f"{self._content_field} does not match canonical content"
            )
        object.__setattr__(self, self._content_field, expected)
        return self

    def canonical_json(self) -> str:
        return canonical_json(self)

    @classmethod
    def from_persisted_json(cls, value: str | bytes) -> Self:
        return cls.model_validate_json(value, context={"persisted": True})

    @classmethod
    def from_persisted(cls, value: Mapping[str, Any]) -> Self:
        return cls.model_validate_json(
            json.dumps(value, ensure_ascii=False, allow_nan=False),
            context={"persisted": True},
        )


class ModelInstanceDescriptor(_DyadBaseModel):
    """One loaded model instance; weight sharing is derived, never asserted."""

    schema_version: str = DYAD_SCHEMA_VERSION
    model_id: Identifier
    revision: Identifier
    tokenizer_id: Identifier
    architecture: Identifier
    n_layers: PositiveInt
    hidden_size: PositiveInt | None = None

    def shares_weights_with(self, other: "ModelInstanceDescriptor") -> bool:
        return (self.model_id, self.revision) == (other.model_id, other.revision)

    def shares_tokenizer_with(self, other: "ModelInstanceDescriptor") -> bool:
        return self.tokenizer_id == other.tokenizer_id

    def representation_compatible_with(
        self, other: "ModelInstanceDescriptor"
    ) -> bool:
        """Same nominal residual geometry; NOT a claim of aligned features."""
        return (
            self.architecture == other.architecture
            and self.n_layers == other.n_layers
            and self.hidden_size is not None
            and self.hidden_size == other.hidden_size
        )


class AgentDescriptor(_DyadBaseModel):
    """Stable agent identity, separated from role, policy, and model."""

    schema_version: str = DYAD_SCHEMA_VERSION
    agent_id: Identifier
    role: Identifier
    policy_class: Identifier
    module_config_ref: CanonicalRef
    model: ModelInstanceDescriptor
    white_box: bool


class DyadCaptureConfig(_ContentBoundModel):
    """Declared capture mode plus the two agent descriptors it governs."""

    _content_field: ClassVar[str] = "config_id"

    schema_version: str = DYAD_SCHEMA_VERSION
    config_id: str = ""
    capture_mode: CaptureMode
    agents: tuple[AgentDescriptor, AgentDescriptor]
    capture_subject_role: Identifier | None = None

    @model_validator(mode="after")
    def _validate_capture_consistency(self) -> Self:
        first, second = self.agents
        if first.agent_id == second.agent_id:
            raise ValueError("dyad agents must have distinct agent_ids")
        if first.role == second.role:
            raise ValueError("dyad agents must have distinct roles")
        white_roles = tuple(a.role for a in self.agents if a.white_box)
        expected_count = {
            CaptureMode.BOTH_WHITE_BOX: 2,
            CaptureMode.ACTOR_WHITE_BOX: 1,
            CaptureMode.RECEIVER_WHITE_BOX: 1,
            CaptureMode.TEXT_ONLY: 0,
        }[self.capture_mode]
        if len(white_roles) != expected_count:
            raise ValueError(
                f"capture mode {self.capture_mode.value} requires "
                f"{expected_count} white-box agent(s), found {len(white_roles)}"
            )
        single_sided = expected_count == 1
        if single_sided:
            if self.capture_subject_role is None:
                raise ValueError(
                    "single-sided capture modes require capture_subject_role"
                )
            if self.capture_subject_role != white_roles[0]:
                raise ValueError(
                    "capture_subject_role must name the white-box agent's role"
                )
        elif self.capture_subject_role is not None:
            raise ValueError(
                "capture_subject_role is only valid for single-sided modes"
            )
        return self

    def white_box_agent_ids(self) -> frozenset[str]:
        return frozenset(a.agent_id for a in self.agents if a.white_box)


class TokenSelection(_DyadBaseModel):
    """Exact token provenance for one captured tensor."""

    schema_version: str = DYAD_SCHEMA_VERSION
    start_token: NonNegativeInt
    end_token: PositiveInt
    char_start: NonNegativeInt | None = None
    char_end: PositiveInt | None = None
    offset_mapping_ref: CanonicalRef | None = None
    selected_text_ref: CanonicalRef | None = None

    @model_validator(mode="after")
    def _validate_spans(self) -> Self:
        if self.end_token <= self.start_token:
            raise ValueError("end_token must be greater than start_token")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be set together")
        if self.char_end is not None and self.char_end <= (self.char_start or 0):
            raise ValueError("char_end must be greater than char_start")
        return self

    def token_count(self) -> int:
        return self.end_token - self.start_token


class ActivationView(_ContentBoundModel):
    """One captured tensor bound to a call, stage, tokens, and artifact."""

    _content_field: ClassVar[str] = "view_id"

    schema_version: str = DYAD_SCHEMA_VERSION
    view_id: str = ""
    model_call_id: Identifier
    actor_id: Identifier
    hook_name: Identifier
    layer: NonNegativeInt
    capture_stage: CaptureStage
    aggregation: Aggregation
    tokens: TokenSelection | None = None
    artifact_ref: CanonicalRef
    dtype: Identifier
    shape: tuple[PositiveInt, ...]

    @model_validator(mode="after")
    def _validate_stage_semantics(self) -> Self:
        if not self.shape:
            raise ValueError("shape must contain at least one dimension")
        span_stage = self.capture_stage in SPAN_STAGES
        if span_stage and self.tokens is None:
            raise ValueError(
                f"stage {self.capture_stage.value} requires token provenance"
            )
        if self.capture_stage is CaptureStage.GENERATED_MEAN:
            if self.aggregation is not Aggregation.MEAN:
                raise ValueError("generated_mean requires mean aggregation")
        elif self.aggregation is Aggregation.MEAN:
            raise ValueError(
                "mean aggregation is only defined for the generated_mean stage"
            )
        single_position = self.capture_stage in (
            CaptureStage.PREFILL_LAST, CaptureStage.GENERATED_LAST
        )
        if single_position and self.tokens is not None:
            if self.tokens.token_count() != 1:
                raise ValueError(
                    f"stage {self.capture_stage.value} selects exactly one token"
                )
        return self


class ActorTurn(_ContentBoundModel):
    """One acting model call by one agent within a dyadic trial."""

    _content_field: ClassVar[str] = "turn_id"

    schema_version: str = DYAD_SCHEMA_VERSION
    turn_id: str = ""
    dyad_id: Identifier
    trial_id: Identifier
    round_index: NonNegativeInt
    turn_ordinal: NonNegativeInt
    actor_id: Identifier
    actor_role: Identifier
    recipient_id: Identifier
    recipient_role: Identifier
    action_event_id: Identifier
    model_call_id: Identifier
    action_ref: CanonicalRef
    pre_action_belief_hash: BeliefStateDigest | None = None
    outcome_event_ids: tuple[Identifier, ...] = ()
    activations: dict[CaptureStage, ActivationView] = Field(
        default_factory=dict
    )
    unavailable_stages: dict[CaptureStage, StageUnavailableReason] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _validate_turn(self) -> Self:
        if self.actor_id == self.recipient_id:
            raise ValueError("actor_id and recipient_id must differ")
        if self.actor_role == self.recipient_role:
            raise ValueError("actor_role and recipient_role must differ")
        if len(set(self.outcome_event_ids)) != len(self.outcome_event_ids):
            raise ValueError("outcome_event_ids must be unique")
        overlap = set(self.activations) & set(self.unavailable_stages)
        if overlap:
            names = ", ".join(sorted(stage.value for stage in overlap))
            raise ValueError(
                f"stages cannot be both captured and unavailable: {names}"
            )
        for stage, view in self.activations.items():
            if view.capture_stage is not stage:
                raise ValueError(
                    f"activation keyed {stage.value} carries stage "
                    f"{view.capture_stage.value}"
                )
            if stage is CaptureStage.MESSAGE_READ_SPAN:
                raise ValueError(
                    "message_read_span belongs to Reception, not ActorTurn"
                )
            if view.actor_id != self.actor_id:
                raise ValueError(
                    "activation view actor_id must match the acting agent"
                )
            if view.model_call_id != self.model_call_id:
                raise ValueError(
                    "activation view must come from this turn's model call"
                )
        return self


class Reception(_ContentBoundModel):
    """How the recipient processed one delivered partner message."""

    _content_field: ClassVar[str] = "reception_id"

    schema_version: str = DYAD_SCHEMA_VERSION
    reception_id: str = ""
    dyad_id: Identifier
    trial_id: Identifier
    recipient_id: Identifier
    source_action_event_id: Identifier
    observation_event_id: Identifier
    processing_call_ids: tuple[Identifier, ...] = ()
    pre_observation_belief_hash: BeliefStateDigest | None = None
    post_observation_belief_hash: BeliefStateDigest | None = None
    belief_update_ids: tuple[Identifier, ...] = ()
    message_read_view: ActivationView | None = None
    next_action_event_id: Identifier | None = None
    next_model_call_id: Identifier | None = None

    @model_validator(mode="after")
    def _validate_reception(self) -> Self:
        if len(set(self.processing_call_ids)) != len(self.processing_call_ids):
            raise ValueError("processing_call_ids must be unique")
        if len(set(self.belief_update_ids)) != len(self.belief_update_ids):
            raise ValueError("belief_update_ids must be unique")
        if (self.next_action_event_id is None) != (
            self.next_model_call_id is None
        ):
            raise ValueError(
                "next_action_event_id and next_model_call_id are set together"
            )
        view = self.message_read_view
        if view is not None:
            if view.capture_stage is not CaptureStage.MESSAGE_READ_SPAN:
                raise ValueError(
                    "message_read_view must use the message_read_span stage"
                )
            if view.actor_id != self.recipient_id:
                raise ValueError(
                    "message_read_view must be captured from the recipient"
                )
        return self


class DyadLink(_ContentBoundModel):
    """Sent action -> reception -> response, with causal event lineage."""

    _content_field: ClassVar[str] = "link_id"

    schema_version: str = DYAD_SCHEMA_VERSION
    link_id: str = ""
    dyad_id: Identifier
    trial_id: Identifier
    sender_turn_id: Identifier
    action_event_id: Identifier
    reception: Reception
    response_turn_id: Identifier | None = None
    causal_parent_event_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def _validate_link(self) -> Self:
        if self.reception.source_action_event_id != self.action_event_id:
            raise ValueError(
                "reception.source_action_event_id must match action_event_id"
            )
        if self.reception.dyad_id != self.dyad_id:
            raise ValueError("reception dyad_id must match link dyad_id")
        if self.reception.trial_id != self.trial_id:
            raise ValueError("reception trial_id must match link trial_id")
        if self.response_turn_id == self.sender_turn_id:
            raise ValueError("a turn cannot respond to itself")
        if len(set(self.causal_parent_event_ids)) != len(
            self.causal_parent_event_ids
        ):
            raise ValueError("causal_parent_event_ids must be unique")
        return self


class RoleAssignment(_ContentBoundModel):
    """Role and first-mover assignment for one trial (Plan 4, Phase 3)."""

    _content_field: ClassVar[str] = "assignment_id"

    schema_version: str = DYAD_SCHEMA_VERSION
    assignment_id: str = ""
    trial_id: Identifier
    trial_family_id: Identifier
    agent_roles: dict[Identifier, Identifier]
    first_mover_agent_id: Identifier
    first_mover_source: FirstMoverSource
    mirror_of_assignment_id: Identifier | None = None
    assignment_seed: NonNegativeInt | None = None

    @model_validator(mode="after")
    def _validate_assignment(self) -> Self:
        if len(self.agent_roles) != 2:
            raise ValueError("a dyadic assignment names exactly two agents")
        roles = list(self.agent_roles.values())
        if roles[0] == roles[1]:
            raise ValueError("the two agents must hold distinct roles")
        if self.first_mover_agent_id not in self.agent_roles:
            raise ValueError("first_mover_agent_id must be an assigned agent")
        if (
            self.first_mover_source is FirstMoverSource.RANDOMIZED
            and self.assignment_seed is None
        ):
            raise ValueError("randomized first mover requires assignment_seed")
        return self

    def role_of(self, agent_id: str) -> str:
        return self.agent_roles[agent_id]

    def mirrored(self) -> dict[str, str]:
        """Return the role mapping with the two roles exchanged."""
        (a, role_a), (b, role_b) = self.agent_roles.items()
        return {a: role_b, b: role_a}


class DyadTrace(_ContentBoundModel):
    """Validated container joining config, assignment, turns, and links."""

    _content_field: ClassVar[str] = "trace_id"

    schema_version: str = DYAD_SCHEMA_VERSION
    trace_id: str = ""
    dyad_id: Identifier
    trial_id: Identifier
    capture: DyadCaptureConfig
    assignment: RoleAssignment
    turns: tuple[ActorTurn, ...] = ()
    links: tuple[DyadLink, ...] = ()

    @model_validator(mode="after")
    def _validate_trace(self) -> Self:
        if self.assignment.trial_id != self.trial_id:
            raise ValueError("assignment trial_id must match trace trial_id")
        config_agents = {a.agent_id for a in self.capture.agents}
        assigned_agents = set(self.assignment.agent_roles)
        if config_agents != assigned_agents:
            raise ValueError(
                "capture config and role assignment must name the same agents"
            )
        for agent in self.capture.agents:
            if self.assignment.agent_roles[agent.agent_id] != agent.role:
                raise ValueError(
                    f"agent {agent.agent_id} role differs between capture "
                    "config and role assignment"
                )
        white_box = self.capture.white_box_agent_ids()
        ordinals: list[int] = []
        turn_ids: dict[str, ActorTurn] = {}
        for turn in self.turns:
            if turn.dyad_id != self.dyad_id or turn.trial_id != self.trial_id:
                raise ValueError(
                    f"turn {turn.turn_id} does not belong to this trace"
                )
            if turn.actor_id not in assigned_agents:
                raise ValueError(f"unknown actor {turn.actor_id}")
            if turn.recipient_id not in assigned_agents:
                raise ValueError(f"unknown recipient {turn.recipient_id}")
            if self.assignment.agent_roles[turn.actor_id] != turn.actor_role:
                raise ValueError(
                    f"turn {turn.turn_id} actor_role contradicts assignment"
                )
            if (
                self.assignment.agent_roles[turn.recipient_id]
                != turn.recipient_role
            ):
                raise ValueError(
                    f"turn {turn.turn_id} recipient_role contradicts assignment"
                )
            if turn.activations and turn.actor_id not in white_box:
                raise ValueError(
                    f"turn {turn.turn_id} captured activations for an agent "
                    "outside the declared capture mode"
                )
            ordinals.append(turn.turn_ordinal)
            turn_ids[turn.turn_id] = turn
        if ordinals != sorted(ordinals) or len(set(ordinals)) != len(ordinals):
            raise ValueError("turn ordinals must be strictly increasing")
        for link in self.links:
            if link.dyad_id != self.dyad_id or link.trial_id != self.trial_id:
                raise ValueError(
                    f"link {link.link_id} does not belong to this trace"
                )
            sender = turn_ids.get(link.sender_turn_id)
            if sender is None:
                raise ValueError(
                    f"link {link.link_id} references unknown sender turn"
                )
            if sender.action_event_id != link.action_event_id:
                raise ValueError(
                    f"link {link.link_id} action_event_id contradicts sender"
                )
            if link.reception.recipient_id != sender.recipient_id:
                raise ValueError(
                    f"link {link.link_id} reception recipient contradicts "
                    "sender turn recipient"
                )
            view = link.reception.message_read_view
            if view is not None and sender.recipient_id not in white_box:
                raise ValueError(
                    f"link {link.link_id} captured receiver-side activations "
                    "for an agent outside the declared capture mode"
                )
            if link.response_turn_id is not None:
                response = turn_ids.get(link.response_turn_id)
                if response is None:
                    raise ValueError(
                        f"link {link.link_id} references unknown response turn"
                    )
                if response.actor_id != sender.recipient_id:
                    raise ValueError(
                        f"link {link.link_id} response actor must be the "
                        "reception recipient"
                    )
                if response.turn_ordinal <= sender.turn_ordinal:
                    raise ValueError(
                        f"link {link.link_id} response must come after the "
                        "sent action"
                    )
        return self
