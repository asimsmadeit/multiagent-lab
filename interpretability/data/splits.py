"""Versioned, group-safe split and permutation contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np


SPLIT_MANIFEST_VERSION = "1.0"
Partition = Literal["train", "development", "test"]
PARTITIONS = ("train", "development", "test")


def _require_identifier(value: Any, field_name: str) -> str:
    """Return a non-blank identifier without coercing untrusted values."""
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    """Reject schema drift at persisted JSON object boundaries."""
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class SplitAssignment:
    """Partition assignment for one trial and its leakage-control groups."""

    trial_id: str
    trial_family_id: str
    dyad_id: str
    partition: Partition

    def __post_init__(self) -> None:
        _require_identifier(self.trial_id, "trial_id")
        _require_identifier(self.trial_family_id, "trial_family_id")
        _require_identifier(self.dyad_id, "dyad_id")
        if type(self.partition) is not str or self.partition not in PARTITIONS:
            raise ValueError(f"Unknown partition: {self.partition!r}")


@dataclass(frozen=True)
class SplitManifest:
    """Immutable, JSON-serializable trial split manifest."""

    assignments: Tuple[SplitAssignment, ...]
    seed: int = 42
    version: str = SPLIT_MANIFEST_VERSION
    locked: bool = False
    manifest_id: Optional[str] = None

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("split manifest seed must be an integer")
        if type(self.version) is not str:
            raise ValueError("split manifest version must be a string")
        if type(self.locked) is not bool:
            raise ValueError("split manifest locked must be a boolean")
        if self.manifest_id is not None and type(self.manifest_id) is not str:
            raise ValueError("split manifest manifest_id must be a string")
        if not all(isinstance(row, SplitAssignment) for row in self.assignments):
            raise ValueError(
                "split manifest assignments must contain SplitAssignment rows"
            )
        ordered = tuple(sorted(self.assignments, key=lambda row: row.trial_id))
        object.__setattr__(self, "assignments", ordered)
        self._validate_assignments()
        expected_id = self._calculate_manifest_id()
        if self.manifest_id is not None and self.manifest_id != expected_id:
            raise ValueError("manifest_id does not match manifest contents")
        object.__setattr__(self, "manifest_id", expected_id)

    def validate(self) -> None:
        """Verify assignments, locked partitions, and the content digest."""
        self._validate_assignments()
        if self.manifest_id != self._calculate_manifest_id():
            raise ValueError("manifest_id does not match manifest contents")

    def _validate_assignments(self) -> None:
        """Reject duplicate trials and family/dyad leakage across partitions."""
        if self.version != SPLIT_MANIFEST_VERSION:
            raise ValueError(
                f"Unsupported split manifest version: {self.version!r}"
            )
        trial_partitions = {}
        family_partitions = {}
        dyad_partitions = {}
        for row in self.assignments:
            _require_identifier(row.trial_id, "trial_id")
            _require_identifier(row.trial_family_id, "trial_family_id")
            _require_identifier(row.dyad_id, "dyad_id")
            if type(row.partition) is not str or row.partition not in PARTITIONS:
                raise ValueError(f"Unknown partition: {row.partition!r}")
            if row.trial_id in trial_partitions:
                raise ValueError(f"Duplicate trial_id: {row.trial_id}")
            trial_partitions[row.trial_id] = row.partition
            family_partitions.setdefault(row.trial_family_id, set()).add(
                row.partition
            )
            dyad_partitions.setdefault(row.dyad_id, set()).add(row.partition)

        leaking_families = sorted(
            key for key, values in family_partitions.items() if len(values) > 1
        )
        if leaking_families:
            raise ValueError(
                "trial_family_id crosses partitions: "
                + ", ".join(leaking_families)
            )
        leaking_dyads = sorted(
            key for key, values in dyad_partitions.items() if len(values) > 1
        )
        if leaking_dyads:
            raise ValueError(
                "dyad_id crosses partitions: " + ", ".join(leaking_dyads)
            )
        if self.locked:
            assigned_partitions = {row.partition for row in self.assignments}
            missing = [
                partition
                for partition in PARTITIONS
                if partition not in assigned_partitions
            ]
            if missing:
                raise ValueError(
                    "locked split manifest has empty partitions: "
                    + ", ".join(missing)
                )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "locked": self.locked,
            "assignments": [
                {
                    "trial_id": row.trial_id,
                    "trial_family_id": row.trial_family_id,
                    "dyad_id": row.dyad_id,
                    "partition": row.partition,
                }
                for row in self.assignments
            ],
        }

    def _calculate_manifest_id(self) -> str:
        canonical = json.dumps(
            self._identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def build(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        seed: int = 42,
        train_fraction: float = 0.7,
        development_fraction: float = 0.15,
    ) -> "SplitManifest":
        """Deterministically assign connected family/dyad components."""
        if type(seed) is not int:
            raise ValueError("split manifest seed must be an integer")
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 <= development_fraction < 1:
            raise ValueError("development_fraction must be in [0, 1)")
        if train_fraction + development_fraction >= 1:
            raise ValueError("train and development fractions must leave test data")

        normalized_rows = []
        required_fields = ("trial_id", "trial_family_id", "dyad_id")
        for index, row in enumerate(records):
            if not isinstance(row, Mapping):
                raise ValueError(f"split record {index} must be a mapping")
            missing = [field for field in required_fields if field not in row]
            if missing:
                raise ValueError(
                    f"split record {index} is missing fields: "
                    + ", ".join(missing)
                )
            normalized_rows.append(tuple(
                _require_identifier(row[field], field)
                for field in required_fields
            ))
        normalized = sorted(normalized_rows)
        trial_ids = [row[0] for row in normalized]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("records contain duplicate trial_id values")

        parents = list(range(len(normalized)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        family_owner = {}
        dyad_owner = {}
        for index, (_, family_id, dyad_id) in enumerate(normalized):
            if family_id in family_owner:
                union(index, family_owner[family_id])
            else:
                family_owner[family_id] = index
            if dyad_id in dyad_owner:
                union(index, dyad_owner[dyad_id])
            else:
                dyad_owner[dyad_id] = index

        components = {}
        for index in range(len(normalized)):
            components.setdefault(find(index), []).append(index)

        component_rows = []
        for indices in components.values():
            component_trials = tuple(
                sorted(normalized[index][0] for index in indices)
            )
            component_key = "|".join(component_trials)
            tie_break = hashlib.sha256(
                f"{SPLIT_MANIFEST_VERSION}:{seed}:{component_key}".encode("utf-8")
            ).hexdigest()
            component_rows.append((indices, component_trials, tie_break))
        if len(component_rows) < 3:
            raise ValueError(
                "At least three independent family/dyad components are required "
                "for train/development/test splits"
            )

        # Largest components are placed first; seed-derived hashes provide
        # stable tie-breaking and partition choices for equally good fits.
        component_rows.sort(key=lambda row: (-len(row[0]), row[2]))
        test_fraction = 1.0 - train_fraction - development_fraction
        fractions = {
            "train": train_fraction,
            "development": development_fraction,
            "test": test_fraction,
        }
        targets = {
            partition: len(normalized) * fraction
            for partition, fraction in fractions.items()
        }
        allocated = {partition: 0 for partition in PARTITIONS}
        component_partitions = []
        for position, (indices, component_trials, _) in enumerate(component_rows):
            remaining_after = len(component_rows) - position - 1
            empty_partitions = [
                partition for partition in PARTITIONS if allocated[partition] == 0
            ]
            if remaining_after < len(empty_partitions):
                candidates = empty_partitions
            else:
                candidates = list(PARTITIONS)

            weight = len(indices)
            scored_candidates = []
            for partition in candidates:
                projected = dict(allocated)
                projected[partition] += weight
                squared_error = sum(
                    (projected[name] - targets[name]) ** 2 for name in PARTITIONS
                )
                tie_material = (
                    f"{SPLIT_MANIFEST_VERSION}:{seed}:"
                    f"{'|'.join(component_trials)}:{partition}"
                ).encode("utf-8")
                tie_break = hashlib.sha256(tie_material).hexdigest()
                scored_candidates.append((squared_error, tie_break, partition))
            _, _, selected_partition = min(scored_candidates)
            allocated[selected_partition] += weight
            component_partitions.append(selected_partition)

        assignments = []
        for (indices, _, _), partition in zip(
            component_rows, component_partitions
        ):
            assignments.extend(
                SplitAssignment(
                    trial_id=normalized[index][0],
                    trial_family_id=normalized[index][1],
                    dyad_id=normalized[index][2],
                    partition=partition,
                )
                for index in indices
            )
        return cls(assignments=tuple(assignments), seed=seed, locked=True)

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_payload(), "manifest_id": self.manifest_id}

    @property
    def partitions(self) -> dict[str, Tuple[str, ...]]:
        """Return explicit train/development/test trial membership."""
        return {
            partition: tuple(
                row.trial_id
                for row in self.assignments
                if row.partition == partition
            )
            for partition in PARTITIONS
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "SplitManifest":
        if type(payload) is not str:
            raise TypeError("split manifest JSON payload must be a string")
        raw = json.loads(payload)
        if type(raw) is not dict:
            raise ValueError("split manifest JSON root must be an object")
        _require_exact_keys(
            raw,
            {"version", "seed", "locked", "assignments", "manifest_id"},
            context="split manifest JSON",
        )
        if type(raw["version"]) is not str:
            raise ValueError("split manifest version must be a string")
        if type(raw["seed"]) is not int:
            raise ValueError("split manifest seed must be an integer")
        if type(raw["locked"]) is not bool:
            raise ValueError("split manifest locked must be a boolean")
        if type(raw["manifest_id"]) is not str or not raw["manifest_id"]:
            raise ValueError("split manifest manifest_id must be a non-empty string")
        if type(raw["assignments"]) is not list:
            raise ValueError("split manifest assignments must be an array")

        assignment_fields = {
            "trial_id",
            "trial_family_id",
            "dyad_id",
            "partition",
        }
        assignments_list = []
        for index, row in enumerate(raw["assignments"]):
            if type(row) is not dict:
                raise ValueError(
                    f"split manifest assignment {index} must be an object"
                )
            _require_exact_keys(
                row,
                assignment_fields,
                context=f"split manifest assignment {index}",
            )
            assignments_list.append(SplitAssignment(
                trial_id=_require_identifier(row["trial_id"], "trial_id"),
                trial_family_id=_require_identifier(
                    row["trial_family_id"], "trial_family_id"
                ),
                dyad_id=_require_identifier(row["dyad_id"], "dyad_id"),
                partition=row["partition"],
            ))
        return cls(
            assignments=tuple(assignments_list),
            seed=raw["seed"],
            version=raw["version"],
            locked=raw["locked"],
            manifest_id=raw["manifest_id"],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SplitManifest":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def permute_group_blocks(
    values: Sequence[Any] | np.ndarray,
    groups: Optional[Sequence[Any] | np.ndarray],
    rng: np.random.Generator,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Permute complete, equally shaped, contiguous group blocks.

    Returning ``(None, reason)`` makes missing or incompatible grouping
    explicitly unavailable; this function never falls back to row shuffling.
    """
    values_array = np.asarray(values)
    if groups is None:
        return None, "groups are required for block permutation"
    groups_array = np.asarray(groups)
    if values_array.ndim == 0 or len(values_array) != len(groups_array):
        return None, "groups must have one value per label row"

    group_order = []
    indices_by_group = {}
    for index, group in enumerate(groups_array.tolist()):
        try:
            known = group in indices_by_group
        except TypeError:
            return None, "group identifiers must be hashable"
        if not known:
            group_order.append(group)
            indices_by_group[group] = []
        indices_by_group[group].append(index)
    if len(group_order) < 2:
        return None, "at least two groups are required for block permutation"

    block_indices = [np.asarray(indices_by_group[group]) for group in group_order]
    if any(
        len(indices) > 1 and not np.all(np.diff(indices) == 1)
        for indices in block_indices
    ):
        return None, "group rows must form contiguous blocks"
    block_sizes = {len(indices) for indices in block_indices}
    if len(block_sizes) != 1:
        return None, "group blocks must have compatible sizes"

    source_order = rng.permutation(len(block_indices))
    if np.array_equal(source_order, np.arange(len(block_indices))):
        source_order = np.roll(source_order, 1)
    shuffled = np.empty_like(values_array)
    for target_index, source_index in enumerate(source_order):
        shuffled[block_indices[target_index]] = values_array[
            block_indices[source_index]
        ]
    return shuffled, None


def group_disjoint_split_indices(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Any] | np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test indices without allowing a group to cross sides."""
    from sklearn.model_selection import GroupShuffleSplit

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")
    groups_array = np.asarray(groups)
    if len(groups_array) != len(y):
        raise ValueError("groups must have one value per sample")
    if len(np.unique(groups_array)) < 2:
        raise ValueError("group-aware splitting requires at least two groups")
    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=test_size,
        random_state=random_state,
    )
    binary_y = (np.asarray(y) > 0.5).astype(int)
    require_both_classes = len(np.unique(binary_y)) > 1
    first_split = None
    for train_indices, test_indices in splitter.split(X, y, groups=groups_array):
        if first_split is None:
            first_split = (train_indices, test_indices)
        if not require_both_classes:
            return train_indices, test_indices
        if (
            len(np.unique(binary_y[train_indices])) > 1
            and len(np.unique(binary_y[test_indices])) > 1
        ):
            return train_indices, test_indices
    if require_both_classes:
        raise ValueError(
            "Could not create a group-disjoint split containing both label classes"
        )
    if first_split is None:
        raise ValueError("Could not create a group-disjoint split")
    return first_split
