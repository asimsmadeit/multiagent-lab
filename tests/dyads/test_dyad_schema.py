"""Contract tests for the bilateral dyadic schema (Plan 4)."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from interpretability.dyads.schema import (
    ActivationView,
    ActorTurn,
    AgentDescriptor,
    Aggregation,
    CaptureMode,
    CaptureStage,
    DyadCaptureConfig,
    DyadLink,
    DyadTrace,
    FirstMoverSource,
    ModelInstanceDescriptor,
    Reception,
    RoleAssignment,
    StageUnavailableReason,
    TokenSelection,
)


def _ref(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _belief(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def make_model(**overrides) -> ModelInstanceDescriptor:
    fields = {
        "model_id": "google/gemma-7b-it",
        "revision": "rev-a",
        "tokenizer_id": "google/gemma-7b-it",
        "architecture": "gemma",
        "n_layers": 28,
        "hidden_size": 3072,
    }
    fields.update(overrides)
    return ModelInstanceDescriptor(**fields)


def make_agent(agent_id: str, role: str, *, white_box: bool, **overrides):
    fields = {
        "agent_id": agent_id,
        "role": role,
        "policy_class": "llm_negotiator",
        "module_config_ref": _ref(f"modules:{agent_id}"),
        "model": make_model(),
        "white_box": white_box,
    }
    fields.update(overrides)
    return AgentDescriptor(**fields)


def make_config(
    mode: CaptureMode = CaptureMode.BOTH_WHITE_BOX,
    *,
    subject_role: str | None = None,
    a_white: bool = True,
    b_white: bool = True,
) -> DyadCaptureConfig:
    return DyadCaptureConfig(
        capture_mode=mode,
        agents=(
            make_agent("agent_a", "potential_deceiver", white_box=a_white),
            make_agent("agent_b", "counterpart", white_box=b_white),
        ),
        capture_subject_role=subject_role,
    )


def make_view(
    *,
    stage: CaptureStage = CaptureStage.GENERATED_LAST,
    actor_id: str = "agent_a",
    model_call_id: str = "call_a1",
    aggregation: Aggregation = Aggregation.NONE,
    tokens: TokenSelection | None = None,
) -> ActivationView:
    return ActivationView(
        model_call_id=model_call_id,
        actor_id=actor_id,
        hook_name="blocks.14.hook_resid_post",
        layer=14,
        capture_stage=stage,
        aggregation=aggregation,
        tokens=tokens,
        artifact_ref=_ref(f"artifact:{stage.value}:{actor_id}:{model_call_id}"),
        dtype="float32",
        shape=(3072,),
    )


def span(start: int = 0, end: int = 4) -> TokenSelection:
    return TokenSelection(start_token=start, end_token=end)


def make_turn(
    *,
    actor_id: str = "agent_a",
    actor_role: str = "potential_deceiver",
    recipient_id: str = "agent_b",
    recipient_role: str = "counterpart",
    ordinal: int = 0,
    model_call_id: str = "call_a1",
    action_event_id: str = "evt_action_a1",
    activations: dict | None = None,
    unavailable: dict | None = None,
    belief: str | None = None,
) -> ActorTurn:
    return ActorTurn(
        dyad_id="dyad_1",
        trial_id="trial_1",
        round_index=ordinal // 2,
        turn_ordinal=ordinal,
        actor_id=actor_id,
        actor_role=actor_role,
        recipient_id=recipient_id,
        recipient_role=recipient_role,
        action_event_id=action_event_id,
        model_call_id=model_call_id,
        action_ref=_ref(f"action:{action_event_id}"),
        pre_action_belief_hash=belief,
        activations=activations or {},
        unavailable_stages=unavailable or {},
    )


def make_reception(
    *,
    recipient_id: str = "agent_b",
    source_action_event_id: str = "evt_action_a1",
    message_view: ActivationView | None = None,
    next_action: str | None = "evt_action_b1",
    next_call: str | None = "call_b1",
) -> Reception:
    return Reception(
        dyad_id="dyad_1",
        trial_id="trial_1",
        recipient_id=recipient_id,
        source_action_event_id=source_action_event_id,
        observation_event_id="evt_obs_b1",
        pre_observation_belief_hash=_belief("pre"),
        post_observation_belief_hash=_belief("post"),
        message_read_view=message_view,
        next_action_event_id=next_action,
        next_model_call_id=next_call,
    )


def make_assignment(**overrides) -> RoleAssignment:
    fields = {
        "trial_id": "trial_1",
        "trial_family_id": "family_1",
        "agent_roles": {
            "agent_a": "potential_deceiver",
            "agent_b": "counterpart",
        },
        "first_mover_agent_id": "agent_a",
        "first_mover_source": FirstMoverSource.RANDOMIZED,
        "assignment_seed": 7,
    }
    fields.update(overrides)
    return RoleAssignment(**fields)


def make_trace(
    *,
    config: DyadCaptureConfig | None = None,
    turns: tuple[ActorTurn, ...] | None = None,
    links: tuple[DyadLink, ...] = (),
) -> DyadTrace:
    return DyadTrace(
        dyad_id="dyad_1",
        trial_id="trial_1",
        capture=config or make_config(),
        assignment=make_assignment(),
        turns=turns if turns is not None else (make_turn(),),
        links=links,
    )


# ---------------------------------------------------------------------------
# Base policy: strictness, immutability, persistence
# ---------------------------------------------------------------------------


def test_round_trip_preserves_content_identity() -> None:
    trace = make_trace(
        turns=(
            make_turn(
                activations={
                    CaptureStage.GENERATED_LAST: make_view(),
                    CaptureStage.PREFILL_LAST: make_view(
                        stage=CaptureStage.PREFILL_LAST
                    ),
                },
                belief=_belief("state"),
            ),
        )
    )
    restored = DyadTrace.from_persisted_json(trace.canonical_json())
    assert restored == trace
    assert restored.trace_id == trace.trace_id


def test_activation_dict_insertion_order_does_not_change_identity() -> None:
    view_last = make_view()
    view_prefill = make_view(stage=CaptureStage.PREFILL_LAST)
    turn_ab = make_turn(
        activations={
            CaptureStage.GENERATED_LAST: view_last,
            CaptureStage.PREFILL_LAST: view_prefill,
        }
    )
    turn_ba = make_turn(
        activations={
            CaptureStage.PREFILL_LAST: view_prefill,
            CaptureStage.GENERATED_LAST: view_last,
        }
    )
    assert turn_ab.turn_id == turn_ba.turn_id


def test_persisted_objects_require_schema_version_and_identity() -> None:
    turn = make_turn()
    payload = turn.model_dump(mode="json")
    missing_version = {k: v for k, v in payload.items() if k != "schema_version"}
    with pytest.raises(ValidationError):
        ActorTurn.from_persisted(missing_version)
    missing_id = {k: v for k, v in payload.items() if k != "turn_id"}
    with pytest.raises(ValidationError):
        ActorTurn.from_persisted(missing_id)


def test_tampered_content_identity_is_rejected() -> None:
    turn = make_turn()
    payload = turn.model_dump(mode="json")
    payload["turn_ordinal"] = 5
    with pytest.raises(ValidationError, match="does not match canonical"):
        ActorTurn.from_persisted(payload)


def test_models_are_frozen_and_reject_unknown_fields() -> None:
    view = make_view()
    with pytest.raises(ValidationError):
        view.layer = 3  # type: ignore[misc]
    with pytest.raises(ValidationError):
        make_model(surprise="field")


# ---------------------------------------------------------------------------
# ModelInstanceDescriptor
# ---------------------------------------------------------------------------


def test_weight_and_representation_sharing_are_derived() -> None:
    base = make_model()
    same_weights = make_model(tokenizer_id="other/tokenizer")
    other_rev = make_model(revision="rev-b")
    assert base.shares_weights_with(same_weights)
    assert not base.shares_weights_with(other_rev)
    assert base.representation_compatible_with(other_rev)
    assert not base.representation_compatible_with(
        make_model(architecture="llama")
    )
    no_hidden = make_model(hidden_size=None)
    assert not no_hidden.representation_compatible_with(base)


# ---------------------------------------------------------------------------
# DyadCaptureConfig
# ---------------------------------------------------------------------------


def test_capture_mode_white_box_counts_are_enforced() -> None:
    with pytest.raises(ValidationError, match="requires 2 white-box"):
        make_config(CaptureMode.BOTH_WHITE_BOX, b_white=False)
    with pytest.raises(ValidationError, match="requires 0 white-box"):
        make_config(CaptureMode.TEXT_ONLY)
    with pytest.raises(ValidationError, match="requires 1 white-box"):
        make_config(CaptureMode.ACTOR_WHITE_BOX, subject_role="potential_deceiver")


def test_single_sided_modes_bind_the_captured_role() -> None:
    config = make_config(
        CaptureMode.ACTOR_WHITE_BOX,
        subject_role="potential_deceiver",
        b_white=False,
    )
    assert config.white_box_agent_ids() == frozenset({"agent_a"})
    with pytest.raises(ValidationError, match="capture_subject_role"):
        make_config(CaptureMode.RECEIVER_WHITE_BOX, a_white=False)
    with pytest.raises(ValidationError, match="must name the white-box"):
        make_config(
            CaptureMode.RECEIVER_WHITE_BOX,
            subject_role="potential_deceiver",
            a_white=False,
        )
    with pytest.raises(ValidationError, match="only valid for single-sided"):
        make_config(
            CaptureMode.BOTH_WHITE_BOX, subject_role="potential_deceiver"
        )


def test_dyad_agents_must_be_distinct() -> None:
    with pytest.raises(ValidationError, match="distinct agent_ids"):
        DyadCaptureConfig(
            capture_mode=CaptureMode.BOTH_WHITE_BOX,
            agents=(
                make_agent("agent_a", "potential_deceiver", white_box=True),
                make_agent("agent_a", "counterpart", white_box=True),
            ),
        )
    with pytest.raises(ValidationError, match="distinct roles"):
        DyadCaptureConfig(
            capture_mode=CaptureMode.BOTH_WHITE_BOX,
            agents=(
                make_agent("agent_a", "counterpart", white_box=True),
                make_agent("agent_b", "counterpart", white_box=True),
            ),
        )


# ---------------------------------------------------------------------------
# TokenSelection and ActivationView
# ---------------------------------------------------------------------------


def test_token_selection_rejects_degenerate_spans() -> None:
    with pytest.raises(ValidationError):
        TokenSelection(start_token=4, end_token=4)
    with pytest.raises(ValidationError):
        TokenSelection(start_token=0, end_token=2, char_start=3)
    with pytest.raises(ValidationError):
        TokenSelection(start_token=0, end_token=2, char_start=5, char_end=5)


def test_span_stages_require_token_provenance() -> None:
    for stage in (
        CaptureStage.GENERATED_MEAN,
        CaptureStage.EVIDENCE_SPAN,
        CaptureStage.MESSAGE_READ_SPAN,
    ):
        with pytest.raises(ValidationError, match="requires token provenance"):
            make_view(
                stage=stage,
                aggregation=(
                    Aggregation.MEAN
                    if stage is CaptureStage.GENERATED_MEAN
                    else Aggregation.NONE
                ),
            )


def test_mean_aggregation_is_bound_to_generated_mean() -> None:
    with pytest.raises(ValidationError, match="requires mean aggregation"):
        make_view(stage=CaptureStage.GENERATED_MEAN, tokens=span())
    with pytest.raises(ValidationError, match="only defined for"):
        make_view(aggregation=Aggregation.MEAN, tokens=span(0, 1))
    view = make_view(
        stage=CaptureStage.GENERATED_MEAN,
        aggregation=Aggregation.MEAN,
        tokens=span(),
    )
    assert view.tokens is not None and view.tokens.token_count() == 4


def test_single_position_stages_select_exactly_one_token() -> None:
    with pytest.raises(ValidationError, match="exactly one token"):
        make_view(stage=CaptureStage.GENERATED_LAST, tokens=span(0, 3))
    view = make_view(stage=CaptureStage.PREFILL_LAST, tokens=span(11, 12))
    assert view.tokens is not None and view.tokens.token_count() == 1


def test_activation_view_rejects_malformed_refs_and_shapes() -> None:
    with pytest.raises(ValidationError):
        ActivationView(
            model_call_id="call_a1",
            actor_id="agent_a",
            hook_name="blocks.14.hook_resid_post",
            layer=14,
            capture_stage=CaptureStage.GENERATED_LAST,
            aggregation=Aggregation.NONE,
            artifact_ref="not-a-digest",
            dtype="float32",
            shape=(3072,),
        )
    with pytest.raises(ValidationError, match="at least one dimension"):
        ActivationView(
            model_call_id="call_a1",
            actor_id="agent_a",
            hook_name="blocks.14.hook_resid_post",
            layer=14,
            capture_stage=CaptureStage.GENERATED_LAST,
            aggregation=Aggregation.NONE,
            artifact_ref=_ref("artifact"),
            dtype="float32",
            shape=(),
        )


# ---------------------------------------------------------------------------
# ActorTurn
# ---------------------------------------------------------------------------


def test_actor_turn_identity_and_role_separation() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        make_turn(recipient_id="agent_a")
    with pytest.raises(ValidationError, match="must differ"):
        make_turn(recipient_role="potential_deceiver")


def test_actor_turn_belief_hash_uses_plain_digest() -> None:
    turn = make_turn(belief=_belief("state"))
    assert turn.pre_action_belief_hash == _belief("state")
    with pytest.raises(ValidationError):
        make_turn(belief=_ref("state"))  # prefixed form rejected


def test_actor_turn_stage_map_consistency() -> None:
    with pytest.raises(ValidationError, match="carries stage"):
        make_turn(activations={CaptureStage.PREFILL_LAST: make_view()})
    with pytest.raises(ValidationError, match="belongs to Reception"):
        make_turn(
            activations={
                CaptureStage.MESSAGE_READ_SPAN: make_view(
                    stage=CaptureStage.MESSAGE_READ_SPAN, tokens=span()
                )
            }
        )
    with pytest.raises(ValidationError, match="match the acting agent"):
        make_turn(activations={
            CaptureStage.GENERATED_LAST: make_view(actor_id="agent_b")
        })
    with pytest.raises(ValidationError, match="this turn's model call"):
        make_turn(activations={
            CaptureStage.GENERATED_LAST: make_view(model_call_id="call_zz")
        })
    with pytest.raises(ValidationError, match="both captured and unavailable"):
        make_turn(
            activations={CaptureStage.GENERATED_LAST: make_view()},
            unavailable={
                CaptureStage.GENERATED_LAST:
                    StageUnavailableReason.BACKEND_UNSUPPORTED
            },
        )


def test_unavailable_stages_are_explicit_not_substituted() -> None:
    turn = make_turn(
        activations={CaptureStage.GENERATED_LAST: make_view()},
        unavailable={
            CaptureStage.EVIDENCE_SPAN:
                StageUnavailableReason.TOKEN_ALIGNMENT_FAILED
        },
    )
    assert (
        turn.unavailable_stages[CaptureStage.EVIDENCE_SPAN]
        is StageUnavailableReason.TOKEN_ALIGNMENT_FAILED
    )
    assert CaptureStage.EVIDENCE_SPAN not in turn.activations


# ---------------------------------------------------------------------------
# Reception and DyadLink
# ---------------------------------------------------------------------------


def test_reception_message_view_contract() -> None:
    good = make_reception(
        message_view=make_view(
            stage=CaptureStage.MESSAGE_READ_SPAN,
            actor_id="agent_b",
            model_call_id="call_b1",
            tokens=span(),
        )
    )
    assert good.message_read_view is not None
    with pytest.raises(ValidationError, match="message_read_span stage"):
        make_reception(
            message_view=make_view(actor_id="agent_b", model_call_id="call_b1")
        )
    with pytest.raises(ValidationError, match="captured from the recipient"):
        make_reception(
            message_view=make_view(
                stage=CaptureStage.MESSAGE_READ_SPAN,
                actor_id="agent_a",
                tokens=span(),
            )
        )


def test_reception_next_action_fields_are_paired() -> None:
    with pytest.raises(ValidationError, match="set together"):
        make_reception(next_action="evt_action_b1", next_call=None)
    silent = make_reception(next_action=None, next_call=None)
    assert silent.next_action_event_id is None


def test_dyad_link_cross_references() -> None:
    reception = make_reception()
    link = DyadLink(
        dyad_id="dyad_1",
        trial_id="trial_1",
        sender_turn_id="turn_x",
        action_event_id="evt_action_a1",
        reception=reception,
        response_turn_id="turn_y",
    )
    assert link.link_id.startswith("sha256:")
    with pytest.raises(ValidationError, match="must match action_event_id"):
        DyadLink(
            dyad_id="dyad_1",
            trial_id="trial_1",
            sender_turn_id="turn_x",
            action_event_id="evt_other",
            reception=reception,
        )
    with pytest.raises(ValidationError, match="cannot respond to itself"):
        DyadLink(
            dyad_id="dyad_1",
            trial_id="trial_1",
            sender_turn_id="turn_x",
            action_event_id="evt_action_a1",
            reception=reception,
            response_turn_id="turn_x",
        )


# ---------------------------------------------------------------------------
# RoleAssignment
# ---------------------------------------------------------------------------


def test_role_assignment_contract() -> None:
    with pytest.raises(ValidationError, match="exactly two agents"):
        make_assignment(agent_roles={"agent_a": "potential_deceiver"})
    with pytest.raises(ValidationError, match="distinct roles"):
        make_assignment(
            agent_roles={"agent_a": "counterpart", "agent_b": "counterpart"}
        )
    with pytest.raises(ValidationError, match="must be an assigned agent"):
        make_assignment(first_mover_agent_id="agent_z")
    with pytest.raises(ValidationError, match="requires assignment_seed"):
        make_assignment(assignment_seed=None)
    fixed = make_assignment(
        first_mover_source=FirstMoverSource.SCENARIO_FIXED,
        assignment_seed=None,
    )
    assert fixed.first_mover_source is FirstMoverSource.SCENARIO_FIXED


def test_mirrored_assignment_swaps_roles_only() -> None:
    assignment = make_assignment()
    mirrored = make_assignment(
        agent_roles=assignment.mirrored(),
        mirror_of_assignment_id=assignment.assignment_id,
    )
    assert mirrored.agent_roles["agent_a"] == "counterpart"
    assert mirrored.agent_roles["agent_b"] == "potential_deceiver"
    assert mirrored.trial_family_id == assignment.trial_family_id
    assert mirrored.assignment_id != assignment.assignment_id


# ---------------------------------------------------------------------------
# DyadTrace
# ---------------------------------------------------------------------------


def _swapped_config() -> DyadCaptureConfig:
    return DyadCaptureConfig(
        capture_mode=CaptureMode.BOTH_WHITE_BOX,
        agents=(
            make_agent("agent_a", "counterpart", white_box=True),
            make_agent("agent_b", "potential_deceiver", white_box=True),
        ),
    )


def test_trace_requires_config_and_assignment_agreement() -> None:
    with pytest.raises(ValidationError, match="role differs"):
        DyadTrace(
            dyad_id="dyad_1",
            trial_id="trial_1",
            capture=_swapped_config(),
            assignment=make_assignment(),
            turns=(),
        )
    with pytest.raises(ValidationError, match="same agents"):
        DyadTrace(
            dyad_id="dyad_1",
            trial_id="trial_1",
            capture=DyadCaptureConfig(
                capture_mode=CaptureMode.BOTH_WHITE_BOX,
                agents=(
                    make_agent("agent_z", "potential_deceiver", white_box=True),
                    make_agent("agent_b", "counterpart", white_box=True),
                ),
            ),
            assignment=make_assignment(),
            turns=(),
        )


def test_trace_role_swap_is_a_valid_distinct_family_member() -> None:
    mirrored_assignment = make_assignment(
        agent_roles={
            "agent_a": "counterpart",
            "agent_b": "potential_deceiver",
        },
        first_mover_agent_id="agent_b",
    )
    trace = DyadTrace(
        dyad_id="dyad_1",
        trial_id="trial_1",
        capture=_swapped_config(),
        assignment=mirrored_assignment,
        turns=(
            make_turn(
                actor_id="agent_b",
                actor_role="potential_deceiver",
                recipient_id="agent_a",
                recipient_role="counterpart",
                model_call_id="call_b1",
                action_event_id="evt_action_b1",
                activations={
                    CaptureStage.GENERATED_LAST: make_view(
                        actor_id="agent_b", model_call_id="call_b1"
                    )
                },
            ),
        ),
    )
    assert trace.assignment.trial_family_id == "family_1"


def test_trace_rejects_ordinal_and_membership_violations() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        make_trace(
            turns=(
                make_turn(ordinal=1),
                make_turn(
                    ordinal=1,
                    actor_id="agent_b",
                    actor_role="counterpart",
                    recipient_id="agent_a",
                    recipient_role="potential_deceiver",
                    model_call_id="call_b1",
                    action_event_id="evt_action_b1",
                ),
            )
        )
    with pytest.raises(ValidationError, match="unknown actor"):
        make_trace(
            turns=(
                make_turn(actor_id="agent_z", action_event_id="evt_z"),
            )
        )
    with pytest.raises(ValidationError, match="contradicts assignment"):
        make_trace(
            turns=(
                make_turn(
                    actor_role="counterpart",
                    recipient_role="potential_deceiver",
                ),
            )
        )


def test_trace_enforces_declared_capture_mode_for_both_sides() -> None:
    actor_only = make_config(
        CaptureMode.ACTOR_WHITE_BOX,
        subject_role="potential_deceiver",
        b_white=False,
    )
    with pytest.raises(ValidationError, match="outside the declared capture"):
        make_trace(
            config=actor_only,
            turns=(
                make_turn(
                    actor_id="agent_b",
                    actor_role="counterpart",
                    recipient_id="agent_a",
                    recipient_role="potential_deceiver",
                    model_call_id="call_b1",
                    action_event_id="evt_action_b1",
                    activations={
                        CaptureStage.GENERATED_LAST: make_view(
                            actor_id="agent_b", model_call_id="call_b1"
                        )
                    },
                ),
            ),
        )
    sender_turn = make_turn()
    receiver_view_link = DyadLink(
        dyad_id="dyad_1",
        trial_id="trial_1",
        sender_turn_id=sender_turn.turn_id,
        action_event_id="evt_action_a1",
        reception=make_reception(
            message_view=make_view(
                stage=CaptureStage.MESSAGE_READ_SPAN,
                actor_id="agent_b",
                model_call_id="call_b1",
                tokens=span(),
            )
        ),
    )
    with pytest.raises(ValidationError, match="receiver-side activations"):
        make_trace(config=actor_only, turns=(sender_turn,),
                   links=(receiver_view_link,))


def test_trace_link_integrity() -> None:
    sender = make_turn()
    response = make_turn(
        actor_id="agent_b",
        actor_role="counterpart",
        recipient_id="agent_a",
        recipient_role="potential_deceiver",
        ordinal=1,
        model_call_id="call_b1",
        action_event_id="evt_action_b1",
    )
    good = make_trace(
        turns=(sender, response),
        links=(
            DyadLink(
                dyad_id="dyad_1",
                trial_id="trial_1",
                sender_turn_id=sender.turn_id,
                action_event_id="evt_action_a1",
                reception=make_reception(),
                response_turn_id=response.turn_id,
            ),
        ),
    )
    assert len(good.links) == 1

    with pytest.raises(ValidationError, match="unknown sender turn"):
        make_trace(
            turns=(sender, response),
            links=(
                DyadLink(
                    dyad_id="dyad_1",
                    trial_id="trial_1",
                    sender_turn_id="turn_missing",
                    action_event_id="evt_action_a1",
                    reception=make_reception(),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="contradicts sender"):
        make_trace(
            turns=(sender, response),
            links=(
                DyadLink(
                    dyad_id="dyad_1",
                    trial_id="trial_1",
                    sender_turn_id=sender.turn_id,
                    action_event_id="evt_action_b1",
                    reception=make_reception(
                        source_action_event_id="evt_action_b1"
                    ),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="response actor must be"):
        wrong_responder = make_turn(
            ordinal=2, action_event_id="evt_action_a2", model_call_id="call_a2"
        )
        make_trace(
            turns=(sender, wrong_responder),
            links=(
                DyadLink(
                    dyad_id="dyad_1",
                    trial_id="trial_1",
                    sender_turn_id=sender.turn_id,
                    action_event_id="evt_action_a1",
                    reception=make_reception(),
                    response_turn_id=wrong_responder.turn_id,
                ),
            ),
        )
    with pytest.raises(ValidationError, match="must come after"):
        early_response = make_turn(
            actor_id="agent_b",
            actor_role="counterpart",
            recipient_id="agent_a",
            recipient_role="potential_deceiver",
            ordinal=1,
            model_call_id="call_b0",
            action_event_id="evt_action_b0",
        )
        late_sender = make_turn(
            ordinal=2, action_event_id="evt_action_a1", model_call_id="call_a1"
        )
        make_trace(
            turns=(early_response, late_sender),
            links=(
                DyadLink(
                    dyad_id="dyad_1",
                    trial_id="trial_1",
                    sender_turn_id=late_sender.turn_id,
                    action_event_id="evt_action_a1",
                    reception=make_reception(),
                    response_turn_id=early_response.turn_id,
                ),
            ),
        )
