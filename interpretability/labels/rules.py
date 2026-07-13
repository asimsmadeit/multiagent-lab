"""Adapters from legacy evaluator dictionaries to canonical label records."""

from __future__ import annotations

from typing import Any, Mapping

from interpretability.labels.schema import (
    BehaviorTarget,
    LabelRecord,
    LabelSource,
    LabelStatus,
    LabelValue,
)


_METHOD_SOURCES = {
    "deepeval": LabelSource.MODEL_JUDGE,
    "gm": LabelSource.GM,
    "human": LabelSource.HUMAN,
    "rule": LabelSource.RULE,
    "rules_fallback": LabelSource.RULE,
}


def _source_for_method(method: str) -> LabelSource:
    return _METHOD_SOURCES.get(method, LabelSource.UNKNOWN)


def _boolean_value(result: Mapping[str, Any]) -> bool | None:
    for key in ("deception_detected", "is_deceptive"):
        value = result.get(key)
        if isinstance(value, bool):
            return value
    return None


def _unit_interval(result: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = result.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if 0.0 <= numeric <= 1.0:
                return numeric
    return None


def label_record_from_evaluation(
    result: Mapping[str, Any] | None,
    *,
    subject_actor_id: str,
    target_event_id: str | None,
    evaluation_event_id: str | None = None,
    behavior_target: BehaviorTarget = BehaviorTarget.FACTUAL_DECEPTION,
    evaluator_version: str,
    evidence_event_ids: tuple[str, ...] = (),
) -> LabelRecord:
    """Convert a legacy result without guessing that failure means honesty.

    A deterministic rules fallback remains an available rule assessment when it
    contains an explicit boolean result. Evaluator absence, parse failure, or a
    result without an explicit boolean becomes an unknown record.
    """
    if result is None:
        return LabelRecord(
            subject_actor_id=subject_actor_id,
            behavior_target=behavior_target,
            value=LabelValue.UNKNOWN,
            status=LabelStatus.UNKNOWN,
            source=LabelSource.UNKNOWN,
            target_event_id=target_event_id,
            evaluation_event_id=evaluation_event_id,
            evaluator_version=evaluator_version,
            evidence_event_ids=evidence_event_ids,
            evaluation_error="evaluation result unavailable",
        )

    raw_method = result.get("evaluation_method")
    method = raw_method if isinstance(raw_method, str) else ""
    source = _source_for_method(method)
    evaluated = _boolean_value(result)
    error = result.get("evaluation_error")
    fallback_reason = result.get("fallback_reason")
    succeeded = result.get("evaluation_succeeded")
    fallback_rule_succeeded = result.get("fallback_rule_succeeded")
    deterministic_fallback = (
        method == "rules_fallback"
        and succeeded is True
        and fallback_rule_succeeded is True
        and evaluated is not None
    )
    failed_or_unrecognized = (
        source is LabelSource.UNKNOWN
        or (succeeded is not None and succeeded is not True)
        or (method == "rules_fallback" and not deterministic_fallback)
    )

    if evaluated is None or failed_or_unrecognized:
        reason = str(
            error
            or result.get("reasoning")
            or (
                f"unrecognized evaluation_method: {method!r}"
                if source is LabelSource.UNKNOWN
                else "evaluation or deterministic rules fallback did not succeed"
                if evaluated is not None
                else "evaluation did not provide an explicit boolean assessment"
            )
        )
        return LabelRecord(
            subject_actor_id=subject_actor_id,
            behavior_target=behavior_target,
            value=LabelValue.UNKNOWN,
            status=LabelStatus.UNKNOWN,
            source=source,
            target_event_id=target_event_id,
            evaluation_event_id=evaluation_event_id,
            evaluator_version=evaluator_version,
            evidence_event_ids=evidence_event_ids,
            evaluation_error=reason,
            fallback_reason=(None if fallback_reason is None else str(fallback_reason)),
            metadata={
                "evaluation_method": method,
                "evaluation_succeeded": succeeded,
                "fallback_rule_succeeded": fallback_rule_succeeded,
            },
        )

    return LabelRecord(
        subject_actor_id=subject_actor_id,
        behavior_target=behavior_target,
        value=LabelValue.TRUE if evaluated else LabelValue.FALSE,
        status=LabelStatus.AVAILABLE,
        source=source,
        target_event_id=target_event_id,
        evaluation_event_id=evaluation_event_id,
        evaluator_version=evaluator_version,
        evidence_event_ids=evidence_event_ids,
        confidence=_unit_interval(result, "confidence"),
        severity=_unit_interval(result, "deception_score", "score"),
        evaluation_error=(None if error is None else str(error)),
        fallback_reason=(None if fallback_reason is None else str(fallback_reason)),
        metadata={
            "evaluation_method": method,
            "evaluation_succeeded": succeeded,
            "deterministic_rules_fallback": deterministic_fallback,
            "fallback_rule_succeeded": fallback_rule_succeeded,
        },
    )
