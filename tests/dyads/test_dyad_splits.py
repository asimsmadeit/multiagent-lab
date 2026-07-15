"""Family resolution, stratified assignment, and leakage-audit tests."""

from __future__ import annotations

import pytest

from interpretability.data.splits import SplitAssignment, SplitManifest
from interpretability.dyads.splits import (
    DyadTrialRecord,
    FamilyLeakageAudit,
    assign_partitions,
    audit_family_leakage,
    resolve_families,
)


def make_record(
    trial_id: str,
    *,
    base: str | None = None,
    scenario: str = "ultimatum_bluff",
    incentive: str = "high",
    policy: str = "default",
    orientation: str = "a_deceives",
    pairing: str = "gemma7b_vs_gemma7b",
    paraphrase_of: str | None = None,
    mirror_of: str | None = None,
    rollout_of: str | None = None,
) -> DyadTrialRecord:
    return DyadTrialRecord(
        trial_id=trial_id,
        dyad_id=f"dyad_{trial_id}",
        base_instance_id=base or f"base_{trial_id}",
        scenario_id=scenario,
        incentive_condition=incentive,
        counterpart_policy=policy,
        role_orientation=orientation,
        model_pairing=pairing,
        paraphrase_of=paraphrase_of,
        mirror_of=mirror_of,
        rollout_of=rollout_of,
    )


# ---------------------------------------------------------------------------
# Family resolution
# ---------------------------------------------------------------------------


def test_sibling_edges_merge_transitively_into_one_family() -> None:
    records = [
        make_record("t1"),
        make_record("t2", paraphrase_of="t1"),
        make_record("t3", mirror_of="t1", orientation="b_deceives"),
        make_record("t4", rollout_of="t3", orientation="b_deceives"),
        make_record("t5", base="base_t1"),  # shared base instance
        make_record("t6"),  # unrelated
    ]
    families = resolve_families(records)
    connected = {families[t] for t in ("t1", "t2", "t3", "t4", "t5")}
    assert len(connected) == 1
    assert families["t6"] not in connected


def test_family_ids_are_content_derived_and_order_independent() -> None:
    records = [
        make_record("t1"),
        make_record("t2", paraphrase_of="t1"),
        make_record("t3"),
    ]
    forward = resolve_families(records)
    backward = resolve_families(list(reversed(records)))
    assert forward == backward
    assert forward["t1"].startswith("dyfam_")


def test_family_resolution_fails_closed_on_bad_edges() -> None:
    with pytest.raises(ValueError, match="unknown sibling parent"):
        resolve_families([make_record("t1", paraphrase_of="ghost")])
    with pytest.raises(ValueError, match="reference the trial itself"):
        make_record("t1", mirror_of="t1")
    with pytest.raises(ValueError, match="Duplicate trial_id"):
        resolve_families([make_record("t1"), make_record("t1")])
    with pytest.raises(ValueError, match="at least one trial record"):
        resolve_families([])


# ---------------------------------------------------------------------------
# Stratified assignment
# ---------------------------------------------------------------------------


def _bulk_records() -> list[DyadTrialRecord]:
    records: list[DyadTrialRecord] = []
    for scenario in ("ultimatum_bluff", "alliance_betrayal"):
        for index in range(30):
            base_id = f"{scenario}_{index}"
            records.append(
                make_record(
                    f"{scenario}_t{index}", base=base_id, scenario=scenario
                )
            )
            records.append(
                make_record(
                    f"{scenario}_t{index}_mirror",
                    base=base_id,
                    scenario=scenario,
                    orientation="b_deceives",
                    mirror_of=f"{scenario}_t{index}",
                )
            )
    return records


def test_assignment_is_family_atomic_and_manifest_valid() -> None:
    records = _bulk_records()
    manifest, report = assign_partitions(records, seed=7)
    assert isinstance(manifest, SplitManifest)
    manifest.validate()

    families = resolve_families(records)
    partition_by_trial = {row.trial_id: row.partition for row in
                          manifest.assignments}
    for record in records:
        mirror = record.mirror_of
        if mirror is not None:
            assert partition_by_trial[record.trial_id] == (
                partition_by_trial[mirror]
            ), "mirrored pair crossed a partition boundary"
    for row in manifest.assignments:
        assert row.trial_family_id == families[row.trial_id]
    assert report.total_trials == len(records)
    assert report.total_families == len(set(families.values()))


def test_assignment_is_deterministic_for_a_seed() -> None:
    records = _bulk_records()
    manifest_a, _ = assign_partitions(records, seed=7)
    manifest_b, _ = assign_partitions(list(reversed(records)), seed=7)
    assert manifest_a.manifest_id == manifest_b.manifest_id


def test_assignment_approximates_target_fractions() -> None:
    records = _bulk_records()
    _, report = assign_partitions(records, seed=7, tolerance=0.1)
    assert report.unbalanced_strata == ()
    for balance in report.strata:
        assert balance.balanced
        assert balance.max_deviation <= 0.1
        assert sum(balance.trial_counts.values()) > 0


def test_unbalanceable_stratum_is_reported_not_raised() -> None:
    # One giant family (11 trials) plus one singleton: no allocation can hit
    # 70/15/15 within a 10% tolerance.
    records = [make_record("t0", base="giant")]
    records += [
        make_record(f"t{i}", base="giant", rollout_of="t0")
        for i in range(1, 11)
    ]
    records.append(make_record("solo"))
    manifest, report = assign_partitions(records, seed=3, tolerance=0.1)
    manifest.validate()
    assert len(report.unbalanced_strata) >= 1
    flagged = report.stratum(report.unbalanced_strata[0])
    assert not flagged.balanced


def test_assignment_rejects_hostile_configuration() -> None:
    records = [make_record("t1")]
    with pytest.raises(ValueError, match="exactly the partitions"):
        assign_partitions(records, fractions={"train": 1.0})
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_partitions(
            records,
            fractions={"train": 0.5, "development": 0.4, "test": 0.4},
        )
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        assign_partitions(
            records,
            fractions={"train": 1.5, "development": -0.25, "test": -0.25},
        )
    with pytest.raises(ValueError, match="seed must be an integer"):
        assign_partitions(records, seed="7")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tolerance"):
        assign_partitions(records, tolerance=2.0)


# ---------------------------------------------------------------------------
# Independent leakage audit
# ---------------------------------------------------------------------------


def test_audit_passes_generated_manifests() -> None:
    records = _bulk_records()
    manifest, _ = assign_partitions(records, seed=11)
    audit = audit_family_leakage(manifest, records)
    assert audit == FamilyLeakageAudit(ok=True, violations=())


def test_audit_catches_leakage_hidden_by_wrong_declared_families() -> None:
    # The manifest declares two distinct families for a mirrored pair, so
    # SplitManifest's own validation cannot object — the audit must.
    records = [
        make_record("t1"),
        make_record("t2", mirror_of="t1", orientation="b_deceives"),
    ]
    manifest = SplitManifest(
        assignments=(
            SplitAssignment(
                trial_id="t1",
                trial_family_id="declared_a",
                dyad_id="dyad_t1",
                partition="train",
            ),
            SplitAssignment(
                trial_id="t2",
                trial_family_id="declared_b",
                dyad_id="dyad_t2",
                partition="test",
            ),
        )
    )
    manifest.validate()  # internally consistent, still leaking
    audit = audit_family_leakage(manifest, records)
    assert not audit.ok
    kinds = {violation.kind for violation in audit.violations}
    assert "family_crosses_partitions" in kinds
    crossing = next(
        violation
        for violation in audit.violations
        if violation.kind == "family_crosses_partitions"
    )
    assert crossing.trial_ids == ("t1", "t2")
    assert crossing.partitions == ("test", "train")


def test_audit_reports_missing_and_unknown_trials() -> None:
    records = [make_record("t1"), make_record("t2")]
    manifest = SplitManifest(
        assignments=(
            SplitAssignment(
                trial_id="t1",
                trial_family_id="fam",
                dyad_id="dyad_t1",
                partition="train",
            ),
            SplitAssignment(
                trial_id="t9",
                trial_family_id="fam9",
                dyad_id="dyad_t9",
                partition="test",
            ),
        )
    )
    audit = audit_family_leakage(manifest, records)
    assert not audit.ok
    kinds = sorted(violation.kind for violation in audit.violations)
    assert kinds == ["missing_from_manifest", "unknown_to_records"]
