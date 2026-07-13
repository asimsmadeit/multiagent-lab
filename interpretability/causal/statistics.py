"""Paired family/dyad-clustered uncertainty for causal outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Sequence

import numpy as np


_ROW_LEVEL_UNITS = {"row", "rows", "sample", "samples", "turn", "turns"}


def _cluster_identifier(value: Any) -> str:
    if value is None:
        raise ValueError("cluster IDs must not be missing")
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("cluster IDs must not be missing")
        return value
    if isinstance(value, bool):
        raise ValueError("cluster IDs must be strings or finite numbers")
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValueError("cluster IDs must be finite")
        return str(value)
    raise ValueError("cluster IDs must be strings or finite numbers")


@dataclass(frozen=True)
class ClusteredPairedEstimate:
    """Paired mean effect with cluster-resampled uncertainty."""

    effect: float
    ci_low: float
    ci_high: float
    p_value: float
    n_rows: int
    n_clusters: int
    cluster_unit: str
    n_bootstrap: int
    n_permutations: int
    random_seed: int
    alpha: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect": self.effect,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_value": self.p_value,
            "n_rows": self.n_rows,
            "n_clusters": self.n_clusters,
            "cluster_unit": self.cluster_unit,
            "n_bootstrap": self.n_bootstrap,
            "n_permutations": self.n_permutations,
            "random_seed": self.random_seed,
            "alpha": self.alpha,
            "bootstrap_ci": {
                "low": self.ci_low,
                "high": self.ci_high,
                "confidence_level": 1.0 - self.alpha,
                "n_resamples": self.n_bootstrap,
                "clustered": True,
            },
            "sign_permutation": {
                "p_value": self.p_value,
                "n_permutations": self.n_permutations,
                "clustered": True,
                "two_sided": True,
            },
        }


def paired_clustered_estimate(
    baseline: Sequence[float] | np.ndarray,
    intervention: Sequence[float] | np.ndarray,
    cluster_ids: Sequence[Any] | np.ndarray,
    *,
    cluster_unit: str,
    n_bootstrap: int = 2000,
    n_permutations: int = 10000,
    random_seed: int = 42,
    alpha: float = 0.05,
) -> ClusteredPairedEstimate:
    """Estimate a paired effect without treating repeated rows as independent."""
    try:
        baseline_array = np.asarray(baseline, dtype=float)
        intervention_array = np.asarray(intervention, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("paired outcomes must be numeric") from exc
    raw_clusters = np.asarray(cluster_ids, dtype=object)
    if baseline_array.ndim != 1 or intervention_array.ndim != 1:
        raise ValueError("baseline and intervention must be one-dimensional")
    if raw_clusters.ndim != 1:
        raise ValueError("cluster IDs must be one-dimensional")
    if not (
        len(baseline_array) == len(intervention_array) == len(raw_clusters)
    ):
        raise ValueError("baseline, intervention, and clusters must align")
    if len(baseline_array) == 0:
        raise ValueError("paired outcomes must not be empty")
    if not isinstance(cluster_unit, str) or not cluster_unit.strip():
        raise ValueError("cluster_unit must be named")
    if cluster_unit.strip().lower() in _ROW_LEVEL_UNITS:
        raise ValueError("cluster_unit must not identify row-level units")
    for name, value in (
        ("n_bootstrap", n_bootstrap),
        ("n_permutations", n_permutations),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, Integral)
        or random_seed < 0
    ):
        raise ValueError("random_seed must be a non-negative integer")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not math.isfinite(float(alpha))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise ValueError("alpha must be finite and between 0 and 1")
    if not np.isfinite(baseline_array).all() or not np.isfinite(
        intervention_array
    ).all():
        raise ValueError("paired outcomes must be finite")
    clusters = np.asarray(
        [_cluster_identifier(value) for value in raw_clusters],
        dtype=str,
    )

    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 3:
        raise ValueError("at least three independent clusters are required")
    differences = intervention_array - baseline_array
    cluster_effects = np.asarray([
        differences[clusters == cluster].mean()
        for cluster in unique_clusters
    ])
    effect = float(cluster_effects.mean())
    rng = np.random.default_rng(int(random_seed))

    bootstrap = np.empty(int(n_bootstrap), dtype=float)
    for index in range(int(n_bootstrap)):
        sampled = rng.integers(0, len(cluster_effects), size=len(cluster_effects))
        bootstrap[index] = cluster_effects[sampled].mean()
    alpha_value = float(alpha)
    ci_low, ci_high = np.quantile(
        bootstrap, [alpha_value / 2, 1 - alpha_value / 2]
    )

    extreme = 0
    for _ in range(int(n_permutations)):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(cluster_effects))
        permuted = float((cluster_effects * signs).mean())
        if abs(permuted) >= abs(effect) - 1e-12:
            extreme += 1
    p_value = (extreme + 1) / (int(n_permutations) + 1)

    return ClusteredPairedEstimate(
        effect=effect,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        n_rows=len(differences),
        n_clusters=len(unique_clusters),
        cluster_unit=cluster_unit,
        n_bootstrap=int(n_bootstrap),
        n_permutations=int(n_permutations),
        random_seed=int(random_seed),
        alpha=alpha_value,
    )
