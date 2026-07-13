"""Central sample eligibility rules for negotiation probe training."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np


_UNKNOWN_LABELS = {"", "unknown", "unavailable", "none", "null", "nan"}


def _label_available(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        if value.strip().lower() in _UNKNOWN_LABELS:
            return False
        try:
            value = float(value)
        except ValueError:
            return False
    if isinstance(value, Real):
        return math.isfinite(float(value))
    return False


def _binary_target_available(value: Any) -> bool:
    """Return whether a headline classification target is explicit 0 or 1."""
    if not _label_available(value):
        return False
    return float(value) in {0.0, 1.0}


def negotiation_sample_mask(
    metadata: Sequence[Mapping[str, Any]],
    target_labels: Sequence[Any],
) -> tuple[np.ndarray, dict[str, int]]:
    """Select explicit negotiation rows with an available numeric target."""
    if len(metadata) != len(target_labels):
        raise ValueError("metadata and target_labels must have the same length")

    mask = np.zeros(len(target_labels), dtype=bool)
    counts = {
        "total": len(target_labels),
        "non_negotiation": 0,
        "invalid_round_or_probe": 0,
        "target_unavailable": 0,
        "included": 0,
    }
    for index, (row, label) in enumerate(zip(metadata, target_labels)):
        if row.get("sample_type") != "negotiation":
            counts["non_negotiation"] += 1
            continue
        round_num = row.get("round_num")
        if (
            not isinstance(round_num, Integral)
            or isinstance(round_num, bool)
            or int(round_num) < 0
            or row.get("is_verification_probe") is True
        ):
            counts["invalid_round_or_probe"] += 1
            continue
        if not _binary_target_available(label):
            counts["target_unavailable"] += 1
            continue
        mask[index] = True
        counts["included"] += 1
    return mask, counts
