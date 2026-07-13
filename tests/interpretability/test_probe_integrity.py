"""Focused regressions for probe sample and group boundaries."""

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from interpretability.data import (
    SplitAssignment,
    SplitManifest,
    negotiation_sample_mask,
    permute_group_blocks,
)
from interpretability.probes.mech_interp_tools import _probe_split_indices
from interpretability.probes.sanity_checks import (
    sanity_check_random_labels as standalone_random_label_check,
)
from interpretability.probes.train_probes import (
    _exact_binary_target,
    _sample_keep_mask,
    sanity_check_random_labels as training_random_label_check,
)


def _valid_manifest_payload() -> dict:
    records = [
        {
            "trial_id": f"t{index}",
            "trial_family_id": f"f{index}",
            "dyad_id": f"d{index}",
        }
        for index in range(3)
    ]
    return json.loads(SplitManifest.build(records, seed=8).to_json())


def test_counterpart_target_eligibility_is_exact_binary_and_countable():
    values = [0, 1.0, True, None, float("nan"), 0.4, "1"]

    numeric, available = _exact_binary_target(values, expected_rows=len(values))

    assert available.tolist() == [True, True, True, False, False, False, False]
    assert numeric[:3].tolist() == [0.0, 1.0, 1.0]
    assert np.isnan(numeric[3:]).all()


def test_split_manifest_is_stable_immutable_and_json_round_trippable():
    records = [
        {"trial_id": "t3", "trial_family_id": "f2", "dyad_id": "d2"},
        {"trial_id": "t1", "trial_family_id": "f1", "dyad_id": "d1"},
        {"trial_id": "t2", "trial_family_id": "f1", "dyad_id": "d1"},
        {"trial_id": "t4", "trial_family_id": "f3", "dyad_id": "d3"},
    ]

    first = SplitManifest.build(records, seed=19)
    second = SplitManifest.build(reversed(records), seed=19)
    restored = SplitManifest.from_json(first.to_json())

    assert first == second == restored
    assert set(first.partitions) == {"train", "development", "test"}
    assert all(first.partitions.values())
    assert first.locked is True
    assert len(first.manifest_id) == 64
    assert first.to_json() == restored.to_json()
    with pytest.raises(FrozenInstanceError):
        first.assignments[0].partition = "test"


def test_split_manifest_seed_changes_stable_assignment_when_components_allow():
    records = [
        {
            "trial_id": f"trial-{index:02}",
            "trial_family_id": f"family-{index:02}",
            "dyad_id": f"dyad-{index:02}",
        }
        for index in range(12)
    ]

    seed_one = SplitManifest.build(records, seed=1)
    seed_one_replay = SplitManifest.build(list(reversed(records)), seed=1)
    seed_two = SplitManifest.build(records, seed=2)

    assert seed_one == seed_one_replay
    assert seed_one.manifest_id != seed_two.manifest_id
    assert seed_one.assignments != seed_two.assignments
    assert all(seed_one.partitions.values())
    assert all(seed_two.partitions.values())


def test_split_manifest_keeps_every_crossed_variant_with_its_family() -> None:
    records = [
        {
            'trial_id': f'trial-{family}-{variant}',
            'trial_family_id': f'family-{family}',
            'dyad_id': f'dyad-{family}',
        }
        for family in range(3)
        for variant in range(4)
    ]

    manifest = SplitManifest.build(records, seed=17)
    for family in range(3):
        family_partitions = {
            assignment.partition
            for assignment in manifest.assignments
            if assignment.trial_family_id == f'family-{family}'
        }
        assert len(family_partitions) == 1
        assert sum(
            assignment.trial_family_id == f'family-{family}'
            for assignment in manifest.assignments
        ) == 4


def test_split_manifest_rejects_insufficient_connected_components():
    records = [
        {"trial_id": "t1", "trial_family_id": "family", "dyad_id": "d1"},
        {"trial_id": "t2", "trial_family_id": "family", "dyad_id": "d2"},
        {"trial_id": "t3", "trial_family_id": "f3", "dyad_id": "d3"},
    ]

    with pytest.raises(ValueError, match="At least three independent"):
        SplitManifest.build(records)


def test_split_manifest_json_digest_detects_tampering():
    records = [
        {
            "trial_id": f"t{index}",
            "trial_family_id": f"f{index}",
            "dyad_id": f"d{index}",
        }
        for index in range(3)
    ]
    manifest = SplitManifest.build(records, seed=8)
    tampered = json.loads(manifest.to_json())
    tampered["assignments"][0]["trial_id"] = "altered-trial"

    with pytest.raises(ValueError, match="manifest_id does not match"):
        SplitManifest.from_json(json.dumps(tampered))


@pytest.mark.parametrize("root", [None, [], "manifest", 4, True])
def test_split_manifest_json_requires_an_object_root(root):
    with pytest.raises(ValueError, match="root must be an object"):
        SplitManifest.from_json(json.dumps(root))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", True, "seed must be an integer"),
        ("seed", 8.0, "seed must be an integer"),
        ("seed", "8", "seed must be an integer"),
        ("locked", 1, "locked must be a boolean"),
        ("locked", "true", "locked must be a boolean"),
        ("version", 1, "version must be a string"),
        ("manifest_id", True, "manifest_id must be a non-empty string"),
        ("manifest_id", None, "manifest_id must be a non-empty string"),
    ],
)
def test_split_manifest_json_rejects_coerced_root_scalars(
    field, value, message
):
    payload = _valid_manifest_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        SplitManifest.from_json(json.dumps(payload))


@pytest.mark.parametrize("field", [
    "version", "seed", "locked", "assignments", "manifest_id",
])
def test_split_manifest_json_rejects_missing_root_fields(field):
    payload = _valid_manifest_payload()
    del payload[field]

    with pytest.raises(ValueError, match="missing fields"):
        SplitManifest.from_json(json.dumps(payload))


def test_split_manifest_json_rejects_unknown_root_fields():
    payload = _valid_manifest_payload()
    payload["future_field"] = "not-yet-supported"

    with pytest.raises(ValueError, match="unknown fields: future_field"):
        SplitManifest.from_json(json.dumps(payload))


@pytest.mark.parametrize("assignments", [None, {}, "rows", 3, True])
def test_split_manifest_json_requires_an_assignment_array(assignments):
    payload = _valid_manifest_payload()
    payload["assignments"] = assignments

    with pytest.raises(ValueError, match="assignments must be an array"):
        SplitManifest.from_json(json.dumps(payload))


@pytest.mark.parametrize("row", [None, [], "assignment", 3, True])
def test_split_manifest_json_requires_assignment_objects(row):
    payload = _valid_manifest_payload()
    payload["assignments"][0] = row

    with pytest.raises(ValueError, match="assignment 0 must be an object"):
        SplitManifest.from_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("trial_id", 1, "trial_id must be a non-empty string"),
        ("trial_id", True, "trial_id must be a non-empty string"),
        ("trial_family_id", None, "trial_family_id must be a non-empty string"),
        ("dyad_id", [], "dyad_id must be a non-empty string"),
        ("partition", 1, "Unknown partition"),
        ("partition", True, "Unknown partition"),
    ],
)
def test_split_manifest_json_rejects_coerced_assignment_values(
    field, value, message
):
    payload = _valid_manifest_payload()
    payload["assignments"][0][field] = value

    with pytest.raises(ValueError, match=message):
        SplitManifest.from_json(json.dumps(payload))


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_split_manifest_json_requires_exact_assignment_fields(mutation):
    payload = _valid_manifest_payload()
    if mutation == "missing":
        del payload["assignments"][0]["dyad_id"]
        message = "missing fields: dyad_id"
    else:
        payload["assignments"][0]["extra"] = "value"
        message = "unknown fields: extra"

    with pytest.raises(ValueError, match=message):
        SplitManifest.from_json(json.dumps(payload))


@pytest.mark.parametrize("value", [1, True, None, "", "   "])
@pytest.mark.parametrize("field", ["trial_id", "trial_family_id", "dyad_id"])
def test_split_manifest_build_rejects_invalid_identifier_types(field, value):
    records = [
        {
            "trial_id": f"t{index}",
            "trial_family_id": f"f{index}",
            "dyad_id": f"d{index}",
        }
        for index in range(3)
    ]
    records[0][field] = value

    with pytest.raises(ValueError, match=f"{field} must be a non-empty string"):
        SplitManifest.build(records)


@pytest.mark.parametrize("seed", [True, 8.0, "8"])
def test_split_manifest_build_rejects_coerced_seed_values(seed):
    records = [
        {
            "trial_id": f"t{index}",
            "trial_family_id": f"f{index}",
            "dyad_id": f"d{index}",
        }
        for index in range(3)
    ]

    with pytest.raises(ValueError, match="seed must be an integer"):
        SplitManifest.build(records, seed=seed)


def test_locked_manual_manifest_rejects_empty_partitions():
    with pytest.raises(ValueError, match="empty partitions"):
        SplitManifest(
            assignments=(
                SplitAssignment("t1", "f1", "d1", "train"),
                SplitAssignment("t2", "f2", "d2", "test"),
            ),
            locked=True,
        )


def test_split_manifest_rejects_family_and_dyad_partition_leakage():
    with pytest.raises(ValueError, match="trial_family_id crosses partitions"):
        SplitManifest(assignments=(
            SplitAssignment("t1", "family", "dyad-1", "train"),
            SplitAssignment("t2", "family", "dyad-2", "test"),
        ))

    with pytest.raises(ValueError, match="dyad_id crosses partitions"):
        SplitManifest(assignments=(
            SplitAssignment("t1", "family-1", "dyad", "development"),
            SplitAssignment("t2", "family-2", "dyad", "test"),
        ))


def test_block_permutation_preserves_complete_trial_layouts():
    labels = np.arange(8)
    groups = np.repeat(["a", "b", "c", "d"], 2)

    shuffled, reason = permute_group_blocks(
        labels,
        groups,
        np.random.default_rng(7),
    )

    assert reason is None
    assert shuffled is not None
    original_blocks = sorted(
        tuple(labels[index:index + 2]) for index in range(0, len(labels), 2)
    )
    shuffled_blocks = sorted(
        tuple(shuffled[index:index + 2]) for index in range(0, len(labels), 2)
    )
    assert shuffled_blocks == original_blocks
    assert not np.array_equal(shuffled, labels)


@pytest.mark.parametrize(
    "groups",
    [
        None,
        np.array(["only"] * 6),
        np.array(["a", "a", "b", "b", "b", "b"]),
        np.array(["a", "b", "a", "b", "a", "b"]),
    ],
)
def test_incompatible_random_label_layouts_report_unavailable(groups):
    X = np.arange(18, dtype=float).reshape(6, 3)
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    standalone = standalone_random_label_check(X, y, n_shuffles=1, groups=groups)
    training = training_random_label_check(X, y, n_shuffles=1, groups=groups)

    for result in (standalone, training):
        assert result["available"] is False
        assert result["passed"] is None
        assert result["mean_shuffled_r2"] is None
        assert "unavailable" in result["message"].lower()


def test_random_label_checks_accept_compatible_trial_blocks():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 6))
    groups = np.repeat([f"trial-{index}" for index in range(10)], 4)
    patterns = (
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 0.0, 0.0]),
    )
    y = np.concatenate([patterns[index % 2] for index in range(10)])

    standalone = standalone_random_label_check(X, y, n_shuffles=2, groups=groups)
    training = training_random_label_check(X, y, n_shuffles=2, groups=groups)

    assert standalone["available"] is True
    assert training["available"] is True


def test_negotiation_mask_excludes_unknown_targets_with_counts():
    metadata = [
        {"sample_type": "negotiation", "round_num": 0},
        {"sample_type": "negotiation", "round_num": 0},
        {"sample_type": "negotiation", "round_num": 0},
        {"sample_type": "negotiation", "round_num": 0},
        {"sample_type": "pre_verification", "round_num": -1},
        {},
    ]
    labels = [0.0, None, float("nan"), 0.4, 1.0, 0.0]

    mask, counts = negotiation_sample_mask(metadata, labels)

    assert mask.tolist() == [True, False, False, False, False, False]
    assert counts == {
        "total": 6,
        "non_negotiation": 2,
        "invalid_round_or_probe": 0,
        "target_unavailable": 3,
        "included": 1,
    }


def test_probe_filter_combines_central_eligibility_and_qc_counts():
    metadata = [
        {
            "sample_type": "negotiation", "round_num": 0,
            "qc_flags": [], "qc_status": "passed", "qc_version": "response-qc/2",
        },
        {
            "sample_type": "negotiation", "round_num": 0,
            "qc_flags": [], "qc_status": "passed", "qc_version": "response-qc/2",
        },
        {"sample_type": "pre_verification", "round_num": -1},
        {
            "sample_type": "negotiation", "round_num": 1,
            "qc_flags": ["template_echo"], "qc_status": "rejected",
            "qc_version": "response-qc/2",
        },
    ]
    labels = np.array([0.0, np.nan, 1.0, 1.0])

    mask, counts = _sample_keep_mask(metadata, labels)

    assert mask.tolist() == [True, False, False, False]
    assert counts == {
        "total": 4,
        "non_negotiation": 1,
        "invalid_round_or_probe": 0,
        "target_unavailable": 1,
        "included": 1,
        "eligible_before_qc": 2,
        "probe_rounds": 1,
        "qc_failures": 1,
    }


def test_probe_filter_fails_closed_when_qc_cannot_be_reproduced() -> None:
    metadata = [
        {"sample_type": "negotiation", "round_num": 0},
        {
            "sample_type": "negotiation",
            "round_num": 1,
            "full_response": "I can offer seventy dollars for delivery today.",
        },
    ]

    mask, counts = _sample_keep_mask(metadata, np.array([0.0, 1.0]))

    assert mask.tolist() == [False, True]
    assert counts["qc_failures"] == 1
    assert counts["included"] == 1


def test_probe_filter_fails_closed_on_stale_qc_without_response() -> None:
    metadata = [{
        "sample_type": "negotiation",
        "round_num": 0,
        "qc_flags": [],
        "qc_status": "passed",
        "qc_version": "response-qc/1",
    }]

    mask, counts = _sample_keep_mask(metadata, np.array([0.0]))

    assert mask.tolist() == [False]
    assert counts["qc_failures"] == 1
    assert counts["included"] == 0


def test_probe_filter_reproduces_contextual_execution_qc() -> None:
    metadata = [{
        "sample_type": "negotiation",
        "round_num": 1,
        "scenario": "promise_break",
        "semantic_phase": "execution",
        "full_response": "DEFECT",
        "qc_flags": [],
        "qc_status": "passed",
        "qc_version": "response-qc/2",
    }]

    mask, counts = _sample_keep_mask(metadata, np.array([1.0]))

    assert mask.tolist() == [True]
    assert counts["qc_failures"] == 0
    assert counts["included"] == 1


@pytest.mark.parametrize(
    "row",
    [
        {"sample_type": "negotiation", "round_num": -1},
        {
            "sample_type": "negotiation",
            "round_num": 0,
            "is_verification_probe": True,
        },
        {"sample_type": "negotiation"},
    ],
)
def test_negotiation_mask_rejects_probe_or_unproven_rounds(row) -> None:
    mask, counts = negotiation_sample_mask([row], [0.0])

    assert mask.tolist() == [False]
    assert counts["invalid_round_or_probe"] == 1


def test_legacy_negotiation_helper_requires_groups_and_keeps_trials_disjoint():
    X = np.arange(80, dtype=float).reshape(20, 4)
    groups = np.repeat([f"trial-{index}" for index in range(10)], 2)
    y = np.tile([0.0, 1.0], 10)

    with pytest.raises(ValueError, match="require trial/dyad groups"):
        _probe_split_indices(
            X,
            y,
            test_size=0.2,
            random_state=3,
            groups=None,
            data_kind="negotiation",
        )

    train_indices, test_indices = _probe_split_indices(
        X,
        y,
        test_size=0.2,
        random_state=3,
        groups=groups,
        data_kind="negotiation",
    )

    assert set(groups[train_indices]).isdisjoint(groups[test_indices])
