"""Dyadic feature families (Plan 4, Phase 7).

Each family is exposed separately so baselines are comparable before any
fusion: score-level aggregation, train-only linear alignment with relational
features, temporal trajectory features, and the shuffled-pairing control
that keeps feature count constant while destroying cross-agent
correspondence (the parameter-matched control the plan requires).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np

FEATURES_VERSION = "dyad-features/1.0.0"

FusionMethod = Literal["mean", "max", "product"]


def _matrix(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty 2-D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.shape[0] < 1:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def fuse_scores(
    sender_scores: Sequence[float],
    receiver_scores: Sequence[float],
    method: FusionMethod = "mean",
) -> np.ndarray:
    """Deterministic score-level aggregation of the two sides."""
    sender = _vector(sender_scores, "sender_scores")
    receiver = _vector(receiver_scores, "receiver_scores")
    if sender.shape != receiver.shape:
        raise ValueError("sender and receiver scores must align")
    if method == "mean":
        return (sender + receiver) / 2.0
    if method == "max":
        return np.maximum(sender, receiver)
    if method == "product":
        if ((sender < 0) | (sender > 1) | (receiver < 0) | (receiver > 1)).any():
            raise ValueError("product fusion requires scores in [0, 1]")
        return sender * receiver
    raise ValueError(f"unknown fusion method: {method!r}")


@dataclass(frozen=True)
class LinearAlignment:
    """Least-squares map from one representation space into another.

    Fit on the training partition only; applying a map fitted elsewhere is
    exactly the leakage the plan forbids, so the fitted object records how
    many rows it saw and its version for provenance.
    """

    matrix: np.ndarray
    n_fit_rows: int
    version: str = FEATURES_VERSION

    def apply(self, source: Sequence[Sequence[float]]) -> np.ndarray:
        array = _matrix(source, "source")
        if array.shape[1] != self.matrix.shape[0]:
            raise ValueError("source dimensionality does not match the fit")
        return array @ self.matrix


def fit_linear_alignment(
    source_train: Sequence[Sequence[float]],
    target_train: Sequence[Sequence[float]],
) -> LinearAlignment:
    source = _matrix(source_train, "source_train")
    target = _matrix(target_train, "target_train")
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target training rows must align")
    if source.shape[0] < source.shape[1]:
        raise ValueError(
            "alignment requires at least as many rows as source dimensions"
        )
    matrix, *_ = np.linalg.lstsq(source, target, rcond=None)
    return LinearAlignment(matrix=matrix, n_fit_rows=source.shape[0])


def relational_features(
    sender_vectors: Sequence[Sequence[float]],
    receiver_vectors: Sequence[Sequence[float]],
) -> Mapping[str, np.ndarray]:
    """Per-pair cosine, L2 difference, and dot product (aligned spaces)."""
    sender = _matrix(sender_vectors, "sender_vectors")
    receiver = _matrix(receiver_vectors, "receiver_vectors")
    if sender.shape != receiver.shape:
        raise ValueError("relational features require matched shapes")
    sender_norms = np.linalg.norm(sender, axis=1)
    receiver_norms = np.linalg.norm(receiver, axis=1)
    if (sender_norms == 0).any() or (receiver_norms == 0).any():
        raise ValueError("zero vectors have no defined direction")
    dots = np.einsum("ij,ij->i", sender, receiver)
    return {
        "cosine": dots / (sender_norms * receiver_norms),
        "l2_difference": np.linalg.norm(sender - receiver, axis=1),
        "dot": dots,
    }


def shuffled_pairing_control(
    receiver_vectors: Sequence[Sequence[float]],
    dyad_ids: Sequence[str],
    *,
    seed: int = 42,
) -> np.ndarray:
    """Re-pair receivers across dyads, preserving the feature multiset.

    A fusion gain that survives this control is explained by feature width,
    not by genuine cross-agent correspondence.
    """
    receiver = _matrix(receiver_vectors, "receiver_vectors")
    ids = [str(dyad_id) for dyad_id in dyad_ids]
    if len(ids) != receiver.shape[0]:
        raise ValueError("dyad_ids must align with receiver_vectors")
    if len(set(ids)) < 2:
        raise ValueError("shuffled pairing requires at least two dyads")
    rng = np.random.default_rng(seed)
    for _ in range(64):
        permutation = rng.permutation(receiver.shape[0])
        if any(
            ids[original] != ids[int(shuffled)]
            for original, shuffled in enumerate(permutation)
        ):
            return receiver[permutation]
    raise RuntimeError("could not construct a cross-dyad permutation")


def score_slope(scores: Sequence[float]) -> float:
    """Least-squares slope of a per-turn score trajectory."""
    values = _vector(scores, "scores")
    if values.shape[0] < 2:
        raise ValueError("a slope requires at least two turns")
    turns = np.arange(values.shape[0], dtype=float)
    return float(np.polyfit(turns, values, 1)[0])


def change_point(scores: Sequence[float]) -> int:
    """Index of the largest absolute jump between consecutive turns."""
    values = _vector(scores, "scores")
    if values.shape[0] < 2:
        raise ValueError("a change point requires at least two turns")
    return int(np.argmax(np.abs(np.diff(values)))) + 1


def time_to_first_alert(
    scores: Sequence[float],
    threshold: float,
) -> int | None:
    """First turn index whose score reaches the threshold, else ``None``."""
    values = _vector(scores, "scores")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    hits = np.nonzero(values >= threshold)[0]
    return int(hits[0]) if hits.size else None


__all__ = [
    "FEATURES_VERSION",
    "LinearAlignment",
    "change_point",
    "fit_linear_alignment",
    "fuse_scores",
    "relational_features",
    "score_slope",
    "shuffled_pairing_control",
    "time_to_first_alert",
]
