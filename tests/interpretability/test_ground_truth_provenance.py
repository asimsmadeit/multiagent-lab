"""Deterministic tests for rule-fallback ground-truth provenance."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from interpretability.core.ground_truth import GroundTruthDetector


RULE_RESULT = {
    "deception_detected": True,
    "deception_score": 0.75,
    "deception_type": "test_rule",
    "explanation": "deterministic rule result",
    "evaluation_method": "rule",
    "evaluation_status": "complete",
    "evaluation_succeeded": True,
}


def _detector_with_fixed_rules(monkeypatch: pytest.MonkeyPatch) -> GroundTruthDetector:
    detector = GroundTruthDetector()
    monkeypatch.setattr(
        detector,
        "apply_rules",
        lambda response, scenario_type, scenario_params: dict(RULE_RESULT),
    )
    return detector


def _assert_common_fallback_provenance(result: dict, reason: str) -> None:
    assert result["evaluation_method"] == "rules_fallback"
    assert result["evaluation_status"] == "fallback_to_rules"
    assert result["evaluation_succeeded"] is True
    assert result["primary_evaluation_succeeded"] is False
    assert result["fallback_rule_succeeded"] is True
    assert result["fallback_reason"] == reason
    for key in (
        "deception_detected",
        "deception_score",
        "deception_type",
        "explanation",
    ):
        assert result[key] == RULE_RESULT[key]


def test_missing_detector_records_rule_fallback_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector_with_fixed_rules(monkeypatch)

    result = detector.detect_with_deepeval("response", "scenario", {})

    _assert_common_fallback_provenance(result, "detector_unavailable")
    assert "evaluation_error" not in result
    assert "evaluation_error_type" not in result
    assert "evaluation_error_source" not in result


def test_detector_initialization_error_is_retained_in_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector_with_fixed_rules(monkeypatch)
    detector._deepeval_initialization_error = {
        "message": "optional backend missing",
        "type": "ImportError",
        "source": "deepeval_initialization",
    }

    result = detector.detect_with_deepeval("response", "scenario", {})

    _assert_common_fallback_provenance(result, "detector_unavailable")
    assert result["evaluation_error"] == "optional backend missing"
    assert result["evaluation_error_type"] == "ImportError"
    assert result["evaluation_error_source"] == "deepeval_initialization"


def test_unsuccessful_evaluation_records_result_error_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector_with_fixed_rules(monkeypatch)
    detector._deepeval_detector = SimpleNamespace(
        evaluate_deception=lambda response, context: SimpleNamespace(
            evaluation_succeeded=False,
            reasoning="invalid evaluator payload",
        )
    )

    result = detector.detect_with_deepeval("response", "scenario", {})

    _assert_common_fallback_provenance(result, "evaluation_unsuccessful")
    assert result["evaluation_error"] == "invalid evaluator payload"
    assert result["evaluation_error_source"] == "deepeval_result"
    assert "evaluation_error_type" not in result


def test_evaluation_exception_records_exception_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _detector_with_fixed_rules(monkeypatch)

    def raise_runtime_error(response: str, context: str) -> None:
        raise RuntimeError("evaluator offline")

    detector._deepeval_detector = SimpleNamespace(
        evaluate_deception=raise_runtime_error,
    )

    result = detector.detect_with_deepeval("response", "scenario", {})

    _assert_common_fallback_provenance(result, "evaluation_exception")
    assert result["evaluation_error"] == "evaluator offline"
    assert result["evaluation_error_type"] == "RuntimeError"
    assert result["evaluation_error_source"] == "deepeval_runtime"


def test_unknown_scenario_is_unavailable_instead_of_honest() -> None:
    detector = GroundTruthDetector()

    direct = detector.apply_rules("response", "unknown_scenario", {})
    fallback = detector.detect_with_deepeval(
        "response", "unknown_scenario", {}
    )

    assert direct["deception_detected"] is None
    assert direct["evaluation_succeeded"] is False
    assert direct["evaluation_status"] == "unavailable"
    assert fallback["deception_detected"] is None
    assert fallback["evaluation_succeeded"] is False
    assert fallback["fallback_rule_succeeded"] is False
    assert fallback["evaluation_status"] == "fallback_unavailable"


@pytest.mark.parametrize(
    "scenario",
    ["info_withholding", "promise_break", "alliance_betrayal"],
)
def test_history_dependent_rule_without_evidence_is_unavailable(
    scenario: str,
) -> None:
    result = GroundTruthDetector().apply_rules("ambiguous response", scenario, {})

    assert result["deception_detected"] is None
    assert result["evaluation_succeeded"] is False
    assert result["evaluation_status"] == "unavailable"
