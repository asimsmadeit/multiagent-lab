"""Release-gate tests for scenario static validation (Plan 1, Phase 7)."""

from __future__ import annotations

import pytest

# Reuse the strict compiler fixture instead of duplicating ~170 lines of
# spec construction; pytest's prepend import mode has this directory on
# sys.path whenever either module is collected.
from test_compiler import _compiler_spec, _copy_spec, _replace_prompt

from interpretability.scenarios.compiler import compile_scenario
from interpretability.scenarios.schema import (
    ActionKind,
    ExtractorReference,
    IncentiveCondition,
)
from interpretability.scenarios.validators import (
    DEFAULT_FORBIDDEN_PROMPT_TERMS,
    ScenarioValidationError,
    ValidationReport,
    validate_or_raise,
    validate_rendered_prompts,
    validate_spec,
)


@pytest.fixture(scope="module")
def spec():
    # The compiler fixture predates the validators' requirement that every
    # action kind has a deterministic extractor; declare one for the
    # terminal walk_away action so the baseline spec is release-clean.
    base = _compiler_spec()
    return _copy_spec(
        base,
        extractors=base.extractors
        + (
            ExtractorReference(
                extractor_name="compiler_terminal",
                extractor_version="extractor/1",
                supported_action_kinds=(ActionKind.WALK_AWAY,),
                deterministic=True,
            ),
        ),
    )


@pytest.fixture(scope="module")
def instance(spec):
    return compile_scenario(spec, "trial-7", 7, IncentiveCondition.MINIMAL)


def _forbidden_variant(spec):
    """Append a forbidden term to one real template, keeping placeholders."""
    template = spec.prompt_templates[0]
    return (
        _replace_prompt(
            spec,
            template.template_id,
            template.template + " Never deceive your counterpart.",
        ),
        template.template_id,
    )


def test_clean_spec_validates_and_gates_pass(spec) -> None:
    report = validate_spec(spec)
    assert isinstance(report, ValidationReport)
    assert report.is_valid
    assert report.errors == ()
    assert validate_or_raise(report) is report
    # Deterministic: a second validation yields the identical canonical report.
    assert validate_spec(spec).report_hash == report.report_hash


def test_rendered_prompt_audit_flags_echoed_braces_deterministically(
    spec, instance
) -> None:
    # The compiler fixture deliberately includes a brace-echoing template,
    # so the rendered audit must flag exactly that and nothing else — and
    # must do so byte-identically on repeated runs.
    report = validate_rendered_prompts(spec, instance)
    assert not report.is_valid
    assert {issue.code for issue in report.errors} == {
        "prompt.unresolved_placeholder"
    }
    again = validate_rendered_prompts(spec, instance)
    assert again.report_hash == report.report_hash
    assert "deceive" in DEFAULT_FORBIDDEN_PROMPT_TERMS


def test_forbidden_term_in_template_blocks_the_spec(spec) -> None:
    mutated, template_id = _forbidden_variant(spec)
    report = validate_spec(mutated)
    assert not report.is_valid
    offending = [
        issue for issue in report.errors if "forbidden" in issue.code
    ]
    assert offending, [issue.code for issue in report.errors]
    del template_id  # location is index-addressed, term is in related_ids
    assert any(
        issue.path.startswith("/prompt_templates/")
        and "deceive" in issue.related_ids
        for issue in offending
    )
    with pytest.raises(ScenarioValidationError) as excinfo:
        validate_or_raise(report)
    assert excinfo.value.report is report
    assert str(len(report.errors)) in str(excinfo.value)


def test_forbidden_term_reaches_rendered_prompt_audit(spec) -> None:
    mutated, _ = _forbidden_variant(spec)
    mutated_instance = compile_scenario(
        mutated, "trial-7", 7, IncentiveCondition.MINIMAL
    )
    report = validate_rendered_prompts(mutated, mutated_instance)
    assert not report.is_valid
    assert any("forbidden" in issue.code for issue in report.errors)


def test_custom_forbidden_terms_override_the_default_list(spec, instance) -> None:
    # A benign word present in the fixture templates becomes forbidden, so
    # the same clean spec/instance pair must now fail the audit.
    text = " ".join(prompt.template for prompt in spec.prompt_templates)
    marker = next(
        word.strip(".,:;!?").lower()
        for word in text.split()
        if len(word.strip(".,:;!?")) >= 4 and word.isalpha()
    )
    report = validate_rendered_prompts(spec, instance, forbidden_terms=(marker,))
    assert not report.is_valid


def test_invalid_capabilities_input_is_a_report_error_not_a_crash(spec) -> None:
    report = validate_spec(spec, capabilities=123)  # type: ignore[arg-type]
    assert not report.is_valid
    assert any(
        issue.code == "validation.invalid_capabilities"
        for issue in report.errors
    )


def test_reports_enforce_canonical_issue_order(spec) -> None:
    mutated, _ = _forbidden_variant(spec)
    report = validate_spec(mutated, capabilities=123)  # type: ignore[arg-type]
    assert len(report.issues) >= 2
    reversed_issues = tuple(reversed(report.issues))
    with pytest.raises(Exception, match="canonical order"):
        ValidationReport(
            subject_hash=report.subject_hash, issues=reversed_issues
        )
