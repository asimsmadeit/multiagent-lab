"""Family-safe stratified dyadic splits (Plan 4, Phase 3).

The data layer's :class:`~interpretability.data.splits.SplitManifest` already
rejects manifests whose *declared* ``trial_family_id`` or ``dyad_id`` crosses
partitions. This module supplies what it cannot know:

1. **Family resolution** — a base scenario instance, its paraphrase siblings,
   its mirrored role pair, and repeated stochastic rollouts are one leakage
   family, resolved by union-find over declared sibling edges.
2. **Stratified assignment** — whole families are allocated to
   train/development/test, stratified by scenario, incentive, counterpart
   policy, role orientation, and model pairing; strata that family atomicity
   makes unbalanceable are reported, never silently dropped.
3. **Independent audit** — families are re-resolved from sibling edges and
   checked against an existing manifest, catching leakage that manifests with
   incorrect declared family IDs would otherwise hide.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from interpretability.data.splits import (
    PARTITIONS,
    Partition,
    SplitAssignment,
    SplitManifest,
)
from interpretability.scenarios.schema import canonical_json

_FRACTION_SUM_TOLERANCE = 1e-9
DEFAULT_FRACTIONS: Mapping[str, float] = {
    "train": 0.7,
    "development": 0.15,
    "test": 0.15,
}


def _require_identifier(value: Any, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_identifier(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_identifier(value, field_name)


@dataclass(frozen=True)
class DyadTrialRecord:
    """One trial's identity, stratum, and leakage-family sibling edges."""

    trial_id: str
    dyad_id: str
    base_instance_id: str
    scenario_id: str
    incentive_condition: str
    counterpart_policy: str
    role_orientation: str
    model_pairing: str
    paraphrase_of: str | None = None
    mirror_of: str | None = None
    rollout_of: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.trial_id, "trial_id")
        _require_identifier(self.dyad_id, "dyad_id")
        _require_identifier(self.base_instance_id, "base_instance_id")
        _require_identifier(self.scenario_id, "scenario_id")
        _require_identifier(self.incentive_condition, "incentive_condition")
        _require_identifier(self.counterpart_policy, "counterpart_policy")
        _require_identifier(self.role_orientation, "role_orientation")
        _require_identifier(self.model_pairing, "model_pairing")
        for name in ("paraphrase_of", "mirror_of", "rollout_of"):
            parent = _optional_identifier(getattr(self, name), name)
            if parent == self.trial_id:
                raise ValueError(f"{name} must not reference the trial itself")

    def stratum_key(self) -> str:
        """Canonical stratification key for this trial."""
        return canonical_json(
            {
                "scenario_id": self.scenario_id,
                "incentive_condition": self.incentive_condition,
                "counterpart_policy": self.counterpart_policy,
                "role_orientation": self.role_orientation,
                "model_pairing": self.model_pairing,
            }
        )

    def parent_edges(self) -> tuple[str, ...]:
        return tuple(
            parent
            for parent in (self.paraphrase_of, self.mirror_of, self.rollout_of)
            if parent is not None
        )


class _UnionFind:
    def __init__(self, members: Iterable[str]) -> None:
        self._parent = {member: member for member in members}

    def find(self, member: str) -> str:
        root = member
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[member] != root:
            self._parent[member], member = root, self._parent[member]
        return root

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            # Deterministic representative: lexicographically smaller root.
            keep, drop = sorted((root_left, root_right))
            self._parent[drop] = keep


def _validated_records(
    records: Sequence[DyadTrialRecord],
) -> dict[str, DyadTrialRecord]:
    if not records:
        raise ValueError("at least one trial record is required")
    by_id: dict[str, DyadTrialRecord] = {}
    for record in records:
        if not isinstance(record, DyadTrialRecord):
            raise ValueError("records must be DyadTrialRecord instances")
        if record.trial_id in by_id:
            raise ValueError(f"Duplicate trial_id: {record.trial_id}")
        by_id[record.trial_id] = record
    for record in by_id.values():
        for parent in record.parent_edges():
            if parent not in by_id:
                raise ValueError(
                    f"trial {record.trial_id} references unknown sibling "
                    f"parent {parent}"
                )
    return by_id


def resolve_families(
    records: Sequence[DyadTrialRecord],
) -> dict[str, str]:
    """Return ``trial_id -> family_id`` under sibling-edge connectivity.

    Trials are one family when they share a ``base_instance_id`` or are
    connected through paraphrase/mirror/rollout parent edges, transitively.
    Family IDs are content-derived from the sorted member trial IDs, so the
    same record set always yields the same IDs regardless of input order.
    """
    by_id = _validated_records(records)
    uf = _UnionFind(by_id)
    base_first: dict[str, str] = {}
    for trial_id in sorted(by_id):
        record = by_id[trial_id]
        anchor = base_first.setdefault(record.base_instance_id, trial_id)
        uf.union(anchor, trial_id)
        for parent in record.parent_edges():
            uf.union(parent, trial_id)
    components: dict[str, list[str]] = {}
    for trial_id in by_id:
        components.setdefault(uf.find(trial_id), []).append(trial_id)
    families: dict[str, str] = {}
    for members in components.values():
        digest = hashlib.sha256(
            canonical_json(sorted(members)).encode("utf-8")
        ).hexdigest()
        family_id = f"dyfam_{digest[:20]}"
        for trial_id in members:
            families[trial_id] = family_id
    return families


@dataclass(frozen=True)
class StratumBalance:
    """Achieved allocation for one stratum after family-atomic assignment."""

    stratum_key: str
    trial_counts: Mapping[Partition, int]
    family_counts: Mapping[Partition, int]
    target_fractions: Mapping[Partition, float]
    achieved_fractions: Mapping[Partition, float]
    max_deviation: float
    balanced: bool


@dataclass(frozen=True)
class StratificationReport:
    """Per-stratum balance plus the strata family atomicity left unbalanced."""

    strata: tuple[StratumBalance, ...]
    unbalanced_strata: tuple[str, ...]
    tolerance: float
    total_trials: int
    total_families: int

    def stratum(self, stratum_key: str) -> StratumBalance:
        for balance in self.strata:
            if balance.stratum_key == stratum_key:
                return balance
        raise KeyError(stratum_key)


@dataclass(frozen=True)
class FamilyLeakageViolation:
    """One reconstructed family observed in more than one partition."""

    family_id: str
    trial_ids: tuple[str, ...]
    partitions: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class FamilyLeakageAudit:
    """Result of independently re-deriving families and checking a manifest."""

    ok: bool
    violations: tuple[FamilyLeakageViolation, ...] = ()


def _validated_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    if set(fractions) != set(PARTITIONS):
        raise ValueError(
            f"fractions must define exactly the partitions {PARTITIONS}"
        )
    validated: dict[str, float] = {}
    for partition in PARTITIONS:
        value = fractions[partition]
        if type(value) not in (int, float) or isinstance(value, bool):
            raise ValueError("fractions must be numbers")
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError("each fraction must lie in [0, 1]")
        validated[partition] = value
    total = sum(validated.values())
    if abs(total - 1.0) > _FRACTION_SUM_TOLERANCE:
        raise ValueError(f"fractions must sum to 1.0, got {total!r}")
    return validated


def assign_partitions(
    records: Sequence[DyadTrialRecord],
    *,
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
    seed: int = 42,
    tolerance: float = 0.1,
) -> tuple[SplitManifest, StratificationReport]:
    """Allocate whole families to partitions, stratified and deterministic.

    Families never straddle partitions. Within each stratum, families are
    ordered deterministically (seeded shuffle, then largest-first) and each
    is assigned to the partition with the greatest remaining deficit against
    its target fraction. Strata whose achieved fractions deviate from targets
    by more than ``tolerance`` are reported as unbalanced rather than raised,
    because whole-family atomicity can make exact balance impossible.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(tolerance) is not float or not 0.0 <= tolerance <= 1.0:
        raise ValueError("tolerance must be a float in [0, 1]")
    targets = _validated_fractions(fractions)
    by_id = _validated_records(records)
    families = resolve_families(records)

    family_members: dict[str, list[str]] = {}
    for trial_id, family_id in families.items():
        family_members.setdefault(family_id, []).append(trial_id)
    for members in family_members.values():
        members.sort()

    # A family's stratum profile is the multiset of its members' strata, so a
    # mirrored role pair (two role orientations, one family) stratifies as one
    # combined unit instead of leaking through per-trial stratification.
    family_stratum: dict[str, str] = {
        family_id: canonical_json(
            sorted(by_id[trial_id].stratum_key() for trial_id in members)
        )
        for family_id, members in family_members.items()
    }
    strata_families: dict[str, list[str]] = {}
    for family_id in sorted(family_members):
        strata_families.setdefault(family_stratum[family_id], []).append(
            family_id
        )

    assignment_of: dict[str, Partition] = {}
    per_stratum_counts: dict[str, dict[Partition, dict[str, int]]] = {}
    for stratum_key in sorted(strata_families):
        family_ids = strata_families[stratum_key]
        rng = random.Random(
            f"{seed}:{hashlib.sha256(stratum_key.encode('utf-8')).hexdigest()}"
        )
        rng.shuffle(family_ids)
        family_ids.sort(
            key=lambda family_id: -len(family_members[family_id])
        )  # stable: preserves the seeded order among equal sizes
        stratum_total = sum(
            len(family_members[family_id]) for family_id in family_ids
        )
        assigned: dict[Partition, int] = {p: 0 for p in PARTITIONS}
        counts: dict[Partition, dict[str, int]] = {
            p: {"trials": 0, "families": 0} for p in PARTITIONS
        }
        for family_id in family_ids:
            size = len(family_members[family_id])
            deficits = {
                p: targets[p] * stratum_total - assigned[p] for p in PARTITIONS
            }
            best = max(PARTITIONS, key=lambda p: (deficits[p], p))
            assigned[best] += size
            counts[best]["trials"] += size
            counts[best]["families"] += 1
            for trial_id in family_members[family_id]:
                assignment_of[trial_id] = best
        per_stratum_counts[stratum_key] = counts

    assignments = tuple(
        SplitAssignment(
            trial_id=trial_id,
            trial_family_id=families[trial_id],
            dyad_id=by_id[trial_id].dyad_id,
            partition=assignment_of[trial_id],
        )
        for trial_id in sorted(by_id)
    )
    manifest = SplitManifest(assignments=assignments, seed=seed)

    balances: list[StratumBalance] = []
    unbalanced: list[str] = []
    for stratum_key in sorted(per_stratum_counts):
        counts = per_stratum_counts[stratum_key]
        stratum_total = sum(counts[p]["trials"] for p in PARTITIONS)
        achieved = {
            p: (counts[p]["trials"] / stratum_total if stratum_total else 0.0)
            for p in PARTITIONS
        }
        max_deviation = max(
            abs(achieved[p] - targets[p]) for p in PARTITIONS
        )
        balanced = max_deviation <= tolerance
        if not balanced:
            unbalanced.append(stratum_key)
        balances.append(
            StratumBalance(
                stratum_key=stratum_key,
                trial_counts={p: counts[p]["trials"] for p in PARTITIONS},
                family_counts={p: counts[p]["families"] for p in PARTITIONS},
                target_fractions=dict(targets),
                achieved_fractions=achieved,
                max_deviation=max_deviation,
                balanced=balanced,
            )
        )
    report = StratificationReport(
        strata=tuple(balances),
        unbalanced_strata=tuple(unbalanced),
        tolerance=tolerance,
        total_trials=len(by_id),
        total_families=len(family_members),
    )
    return manifest, report


def audit_family_leakage(
    manifest: SplitManifest,
    records: Sequence[DyadTrialRecord],
) -> FamilyLeakageAudit:
    """Re-derive families from sibling edges and check them against a manifest.

    This catches leakage that :class:`SplitManifest` cannot see: a manifest
    whose *declared* ``trial_family_id`` values are wrong (for example, a
    mirrored pair recorded as two distinct families) validates internally but
    still leaks. Missing or unknown manifest trials are violations, not
    silently ignored rows.
    """
    families = resolve_families(records)
    partition_of: dict[str, str] = {
        row.trial_id: row.partition for row in manifest.assignments
    }
    violations: list[FamilyLeakageViolation] = []

    missing = sorted(set(families) - set(partition_of))
    if missing:
        violations.append(
            FamilyLeakageViolation(
                family_id="",
                trial_ids=tuple(missing),
                partitions=(),
                kind="missing_from_manifest",
            )
        )
    unknown = sorted(set(partition_of) - set(families))
    if unknown:
        violations.append(
            FamilyLeakageViolation(
                family_id="",
                trial_ids=tuple(unknown),
                partitions=(),
                kind="unknown_to_records",
            )
        )

    members_by_family: dict[str, list[str]] = {}
    for trial_id, family_id in families.items():
        members_by_family.setdefault(family_id, []).append(trial_id)
    for family_id in sorted(members_by_family):
        members = sorted(members_by_family[family_id])
        partitions = sorted(
            {
                partition_of[trial_id]
                for trial_id in members
                if trial_id in partition_of
            }
        )
        if len(partitions) > 1:
            violations.append(
                FamilyLeakageViolation(
                    family_id=family_id,
                    trial_ids=tuple(members),
                    partitions=tuple(partitions),
                    kind="family_crosses_partitions",
                )
            )
    return FamilyLeakageAudit(
        ok=not violations, violations=tuple(violations)
    )
