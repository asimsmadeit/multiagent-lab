"""Contract tests for canonical tri-state label records."""

from __future__ import annotations

import json

import pytest

from interpretability.labels.schema import (
    LABEL_SCHEMA_VERSION,
    BehaviorTarget,
    LabelRecord,
    LabelSource,
    LabelSourcePolicy,
    LabelStatus,
    LabelValue,
    project_label,
)
from interpretability.labels.rules import label_record_from_evaluation


def _record(
    *,
    source: LabelSource = LabelSource.RULE,
    value: LabelValue = LabelValue.TRUE,
    status: LabelStatus = LabelStatus.AVAILABLE,
    error: str | None = None,
) -> LabelRecord:
    return LabelRecord(
        subject_actor_id="seller",
        behavior_target=BehaviorTarget.FACTUAL_DECEPTION,
        value=value,
        status=status,
        source=source,
        target_event_id="event-4",
        evaluation_event_id="event-6",
        evaluator_version="rules-1",
        evidence_event_ids=("fact-1", "event-4"),
        confidence=0.9 if status is LabelStatus.AVAILABLE else None,
        evaluation_error=error,
    )


def test_label_record_is_json_safe_stable_and_round_trips() -> None:
    record = _record()
    serialized = record.to_dict()

    json.dumps(serialized)
    restored = LabelRecord.from_dict(serialized)

    assert restored == record
    assert restored.label_id == record.label_id
    assert record.schema_version == LABEL_SCHEMA_VERSION == "1.2.0"


def test_label_restore_requires_current_schema_and_content_id() -> None:
    payload = _record().to_dict()
    del payload["label_id"]
    payload["metadata"] = {"tampered": True}
    with pytest.raises(ValueError, match="label_id is required"):
        LabelRecord.from_dict(payload)

    payload = _record().to_dict()
    payload["schema_version"] = "999"
    with pytest.raises(ValueError, match="schema version"):
        LabelRecord.from_dict(payload)


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_label_restore_requires_exact_current_fields(mutation: str) -> None:
    payload = _record().to_dict()
    if mutation == "missing":
        del payload["fallback_reason"]
        match = "missing fields: fallback_reason"
    else:
        payload["future_field"] = "not part of label-record/1.2"
        match = "unknown fields: future_field"

    with pytest.raises(ValueError, match=match):
        LabelRecord.from_dict(payload)


def test_label_restore_rejects_non_array_evidence_and_non_mapping_metadata() -> None:
    evidence = _record().to_dict()
    evidence["evidence_event_ids"] = "event-4"
    with pytest.raises(TypeError, match="evidence_event_ids must be an array"):
        LabelRecord.from_dict(evidence)

    metadata = _record().to_dict()
    metadata["metadata"] = []
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        LabelRecord.from_dict(metadata)


def test_label_record_deep_freezes_caller_owned_evidence_and_metadata() -> None:
    evidence = ["event-4", "fact-1"]
    metadata = {
        "judge": {"reasons": ["claim", "omission"]},
        "scores": [0.2, 0.8],
    }
    record = LabelRecord(
        subject_actor_id="seller",
        behavior_target=BehaviorTarget.FACTUAL_DECEPTION,
        value=LabelValue.TRUE,
        status=LabelStatus.AVAILABLE,
        source=LabelSource.RULE,
        target_event_id="event-4",
        evaluation_event_id="event-6",
        evaluator_version="rules-1",
        evidence_event_ids=evidence,
        metadata=metadata,
    )
    original_id = record.label_id
    original_payload = record.to_dict()

    evidence.append("late-event")
    metadata["judge"]["reasons"].append("late-reason")
    metadata["scores"][0] = 99.0

    assert record.label_id == original_id
    assert record.to_dict() == original_payload
    with pytest.raises(TypeError):
        record.metadata["new"] = "mutation"


def test_label_record_rejects_false_honesty_for_unknown_evaluation() -> None:
    with pytest.raises(ValueError, match="non-available"):
        _record(
            value=LabelValue.FALSE,
            status=LabelStatus.UNKNOWN,
            error="judge timed out",
        )

    unknown = _record(
        value=LabelValue.UNKNOWN,
        status=LabelStatus.UNKNOWN,
        error="judge timed out",
    )
    assert unknown.value is LabelValue.UNKNOWN
    assert unknown.evaluation_error == "judge timed out"


def test_actual_behavior_policy_rejects_agent_perception() -> None:
    with pytest.raises(ValueError, match="agent perception"):
        LabelSourcePolicy(
            behavior_target=BehaviorTarget.FACTUAL_DECEPTION,
            source_order=(LabelSource.AGENT_PERCEPTION,),
            policy_version="policy-1",
        )


def test_source_policy_copies_mutable_source_order() -> None:
    source_order = [LabelSource.HUMAN, LabelSource.RULE]
    policy = LabelSourcePolicy(
        behavior_target=BehaviorTarget.FACTUAL_DECEPTION,
        source_order=source_order,
        policy_version="human-then-rule-1",
        allow_fallback=True,
    )

    source_order.reverse()

    assert policy.source_order == (LabelSource.HUMAN, LabelSource.RULE)
    assert isinstance(policy.source_order, tuple)


def test_available_label_requires_known_source_and_target_event() -> None:
    with pytest.raises(ValueError, match="known source"):
        _record(source=LabelSource.UNKNOWN)

    with pytest.raises(ValueError, match="target event"):
        LabelRecord(
            subject_actor_id="seller",
            behavior_target=BehaviorTarget.FACTUAL_DECEPTION,
            value=LabelValue.TRUE,
            status=LabelStatus.AVAILABLE,
            source=LabelSource.RULE,
            target_event_id=None,
            evaluation_event_id="event-6",
            evaluator_version="rules-1",
        )


def test_agent_perception_only_assesses_counterpart_estimate() -> None:
    with pytest.raises(ValueError, match="counterpart deception estimates"):
        _record(source=LabelSource.AGENT_PERCEPTION)

    perception = LabelRecord(
        subject_actor_id="seller",
        behavior_target=BehaviorTarget.COUNTERPART_DECEPTION_ESTIMATE,
        value=LabelValue.FALSE,
        status=LabelStatus.AVAILABLE,
        source=LabelSource.AGENT_PERCEPTION,
        target_event_id="event-4",
        evaluation_event_id="event-6",
        evaluator_version="agent-perception-1",
    )

    assert perception.source is LabelSource.AGENT_PERCEPTION


def test_projection_selects_declared_source_and_retains_record_id() -> None:
    rule = _record(source=LabelSource.RULE, value=LabelValue.TRUE)
    human = _record(source=LabelSource.HUMAN, value=LabelValue.FALSE)
    policy = LabelSourcePolicy(
        behavior_target=BehaviorTarget.FACTUAL_DECEPTION,
        source_order=(LabelSource.HUMAN, LabelSource.RULE),
        policy_version="human-then-rule-1",
        allow_fallback=True,
    )

    projection = project_label(
        [rule, human],
        policy,
        subject_actor_id="seller",
        target_event_id="event-4",
    )

    assert projection.value is LabelValue.FALSE
    assert projection.selected_source is LabelSource.HUMAN
    assert projection.selected_record_id == human.label_id


def test_projection_preserves_same_source_conflict_and_missing_as_unknown() -> None:
    positive = _record(value=LabelValue.TRUE)
    negative = _record(value=LabelValue.FALSE)
    policy = LabelSourcePolicy(
        behavior_target=BehaviorTarget.FACTUAL_DECEPTION,
        source_order=(LabelSource.RULE,),
        policy_version="rule-1",
    )

    conflicted = project_label(
        [positive, negative],
        policy,
        subject_actor_id="seller",
        target_event_id="event-4",
    )
    missing = project_label(
        [],
        policy,
        subject_actor_id="seller",
        target_event_id="event-4",
    )

    assert conflicted.status is LabelStatus.CONFLICTED
    assert conflicted.value is LabelValue.UNKNOWN
    assert missing.status is LabelStatus.UNKNOWN
    assert missing.value is LabelValue.UNKNOWN


def test_successful_rules_fallback_is_available_and_retains_primary_error() -> None:
    record = label_record_from_evaluation(
        {
            "deception_detected": False,
            "deception_score": 0.0,
            "evaluation_method": "rules_fallback",
            "evaluation_succeeded": True,
            "fallback_rule_succeeded": True,
            "fallback_reason": "evaluation_exception",
            "evaluation_error": "judge unavailable",
        },
        subject_actor_id="seller",
        target_event_id="event-4",
        evaluator_version="rules-1",
    )

    assert record.status is LabelStatus.AVAILABLE
    assert record.value is LabelValue.FALSE
    assert record.source is LabelSource.RULE
    assert record.fallback_reason == "evaluation_exception"
    assert record.evaluation_error == "judge unavailable"


def test_failed_or_unconfirmed_rules_fallback_is_unknown() -> None:
    for payload in (
        {
            "evaluation_method": "rules_fallback",
            "evaluation_succeeded": False,
            "fallback_rule_succeeded": True,
            "deception_detected": False,
        },
        {
            "evaluation_method": "rules_fallback",
            "evaluation_succeeded": True,
            "deception_detected": False,
        },
        {
            "evaluation_method": "rules_fallback",
            "fallback_rule_succeeded": True,
            "deception_detected": False,
        },
    ):
        record = label_record_from_evaluation(
            payload,
            subject_actor_id="seller",
            target_event_id="event-4",
            evaluator_version="rules-1",
        )
        assert record.status is LabelStatus.UNKNOWN
        assert record.value is LabelValue.UNKNOWN


def test_missing_or_malformed_evaluation_becomes_unknown_not_false() -> None:
    missing = label_record_from_evaluation(
        None,
        subject_actor_id="seller",
        target_event_id="event-4",
        evaluator_version="rules-1",
    )
    malformed = label_record_from_evaluation(
        {"evaluation_method": "deepeval", "reasoning": "invalid payload"},
        subject_actor_id="seller",
        target_event_id="event-4",
        evaluator_version="deepeval-1",
    )

    assert missing.value is LabelValue.UNKNOWN
    assert missing.status is LabelStatus.UNKNOWN
    assert malformed.value is LabelValue.UNKNOWN
    assert malformed.status is LabelStatus.UNKNOWN
    assert malformed.source is LabelSource.MODEL_JUDGE


def test_failed_judge_boolean_and_unknown_method_fail_closed() -> None:
    failed = label_record_from_evaluation(
        {
            "evaluation_method": "deepeval",
            "evaluation_succeeded": False,
            "deception_detected": False,
            "evaluation_error": "judge timeout",
        },
        subject_actor_id="seller",
        target_event_id="event-4",
        evaluator_version="deepeval-1",
    )
    typo = label_record_from_evaluation(
        {
            "evaluation_method": "deep_eval",
            "evaluation_succeeded": True,
            "deception_detected": False,
        },
        subject_actor_id="seller",
        target_event_id="event-4",
        evaluator_version="unknown-1",
    )

    assert failed.status is LabelStatus.UNKNOWN
    assert failed.value is LabelValue.UNKNOWN
    assert failed.source is LabelSource.MODEL_JUDGE
    assert typo.status is LabelStatus.UNKNOWN
    assert typo.value is LabelValue.UNKNOWN
    assert typo.source is LabelSource.UNKNOWN
    assert "unrecognized" in str(typo.evaluation_error)


def test_non_boolean_success_flag_fails_closed() -> None:
    record = label_record_from_evaluation(
        {
            "evaluation_method": "gm",
            "evaluation_succeeded": "yes",
            "is_deceptive": True,
        },
        subject_actor_id="seller",
        target_event_id="event-4",
        evaluator_version="gm-1",
    )

    assert record.status is LabelStatus.UNKNOWN
    assert record.value is LabelValue.UNKNOWN
