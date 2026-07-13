# MATS Probe Training and Analysis v2
# Enhanced version with:
#   - Per-scenario AUC tracking for cross-scenario generalization
#   - Model scale comparison (9B vs 27B)
#   - Difference-in-means baseline
#   - Better visualization
#
# Usage:
#   python train_probes.py --data ./experiment_outputs/merged_activations_*.pt
#   python train_probes.py --compare results_9b.json results_27b.json

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import torch
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from interpretability.data import (
    SplitManifest,
    load_activation_dataset,
    negotiation_sample_mask,
    permute_group_blocks,
)
from interpretability.probes.metrics import (
    evaluate_nested_grouped_layers,
    resolve_manifest_row_groups,
)


# =============================================================================
# PROBE CLASSES
# =============================================================================

@dataclass
class ProbeResult:
    """Results from training a probe."""
    layer: int
    label_type: str  # "gm" or "agent"
    r2_score: float
    accuracy: float
    auc: float
    train_r2: float
    test_r2: float
    cross_val_scores: List[float]

    def to_dict(self) -> Dict:
        return {
            "layer": int(self.layer),
            "label_type": self.label_type,
            "r2_score": float(self.r2_score),
            "accuracy": float(self.accuracy),
            "auc": float(self.auc),
            "train_r2": float(self.train_r2),
            "test_r2": float(self.test_r2),
            "cross_val_mean": float(np.mean(self.cross_val_scores)) if self.cross_val_scores else 0.0,
            "cross_val_std": float(np.std(self.cross_val_scores)) if self.cross_val_scores else 0.0,
        }


@dataclass
class ThresholdSensitivityResult:
    """Results from threshold sensitivity analysis.

    Helps understand if the default 0.5 binarization threshold is appropriate.
    """
    by_threshold: Dict[float, Dict[str, Optional[float]]]
    """Results for each threshold: {accuracy, f1, precision, recall}"""

    best_threshold: float
    """Threshold with highest F1 score"""

    is_robust: bool
    """True if F1 variance across thresholds < 10%"""

    recommendation: str
    """Human-readable recommendation"""

    def to_dict(self) -> Dict:
        return {
            "by_threshold": self.by_threshold,
            "best_threshold": self.best_threshold,
            "is_robust": self.is_robust,
            "recommendation": self.recommendation,
        }


@dataclass
class LayerAnalysisResult:
    """Results from analyzing probe performance across layers.

    Used to find the optimal layer for probing and verify expected patterns.
    """
    best_layer: int
    """Layer with highest AUC"""

    peak_auc: float
    """AUC at best layer"""

    peak_r2: float
    """R² at best layer"""

    auc_std_across_layers: float
    """Standard deviation of AUC across layers"""

    is_flat_curve: bool
    """Warning: True if AUC std < 0.05 (suspicious)"""

    has_expected_inverted_u: Optional[bool]
    """True if middle layers > early and late (expected pattern)"""

    relative_position: float
    """Position of best layer (0=first, 1=last)"""

    layer_aucs: Dict[int, float]
    """AUC for each layer"""

    warnings: List[str]
    """List of warning messages if patterns are unexpected"""

    def to_dict(self) -> Dict:
        return {
            "best_layer": self.best_layer,
            "peak_auc": self.peak_auc,
            "peak_r2": self.peak_r2,
            "auc_std_across_layers": self.auc_std_across_layers,
            "is_flat_curve": self.is_flat_curve,
            "has_expected_inverted_u": self.has_expected_inverted_u,
            "relative_position": self.relative_position,
            "layer_aucs": self.layer_aucs,
            "warnings": [w for w in self.warnings if w is not None],
        }


@dataclass
class GeneralizationResult:
    """Results from cross-scenario generalization analysis.

    Tests whether the probe generalizes across different scenarios,
    which is critical for validating it captures general deception features.
    """
    by_scenario: Dict[str, Dict[str, Any]]
    """Per-scenario holdout results"""

    average_r2: Optional[float]
    """Mean R² across scenarios (can be negative for poor generalization)"""

    average_auc: Optional[float]
    """Mean AUC across scenarios (>0.7 = good generalization)"""

    std_r2: Optional[float]
    """Standard deviation of R² across scenarios"""

    std_auc: Optional[float]
    """Standard deviation of AUC across scenarios"""

    def to_dict(self) -> Dict:
        return {
            "by_scenario": self.by_scenario,
            "average_r2": self.average_r2,
            "average_auc": self.average_auc,
            "std_r2": self.std_r2,
            "std_auc": self.std_auc,
        }


def _train_test_indices(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return row or group-aware split indices.

    Negotiation turns from the same trial must remain on one side of a split.
    ``groups=None`` preserves the public functions' legacy row-split behavior.
    """
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")
    if len(y) < 2:
        raise ValueError("At least two samples are required for a train/test split")

    indices = np.arange(len(y))
    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit

        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("groups must have one value per sample")
        if len(np.unique(groups)) < 2:
            raise ValueError("group-aware splitting requires at least two groups")
        splitter = GroupShuffleSplit(
            n_splits=100, test_size=0.2, random_state=random_state
        )
        binary_y = (np.asarray(y) > 0.5).astype(int)
        require_both_classes = len(np.unique(binary_y)) > 1
        first_split = None
        for train_idx, test_idx in splitter.split(X, y, groups=groups):
            if first_split is None:
                first_split = (train_idx, test_idx)
            if not require_both_classes:
                return train_idx, test_idx
            if (
                len(np.unique(binary_y[train_idx])) > 1
                and len(np.unique(binary_y[test_idx])) > 1
            ):
                return train_idx, test_idx
        if require_both_classes:
            raise ValueError(
                "Could not create a group-disjoint split containing both label "
                "classes in train and test"
            )
        if first_split is not None:
            return first_split
        raise ValueError("Could not create a group-aware train/test split")

    binary_y = (np.asarray(y) > 0.5).astype(int)
    min_class_count = (
        min(np.bincount(binary_y)) if len(np.unique(binary_y)) > 1 else 0
    )
    stratify_arg = binary_y if min_class_count >= 2 else None
    return train_test_split(
        indices,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify_arg,
    )


def train_ridge_probe(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 100.0,
    use_pca: bool = True,
    n_components: int = 30,
    random_state: int = 42,
    normalize: bool = True,
    groups: Optional[np.ndarray] = None,
) -> Tuple[Any, ProbeResult]:
    """Train a Ridge regression probe.

    Args:
        X: Feature matrix [N, d_model]
        y: Labels [N]
        alpha: Ridge regularization strength
        use_pca: Whether to apply PCA for dimensionality reduction
        n_components: Number of PCA components
        random_state: Random seed for reproducibility (default 42)
        normalize: Whether to apply StandardScaler (fitted on train only)

    Returns:
        Tuple of (trained Ridge probe, ProbeResult with metrics)
    """

    train_idx, test_idx = _train_test_indices(
        X, y, random_state=random_state, groups=groups
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Return the fitted preprocessing and estimator as one object so callers
    # cannot accidentally feed raw activations to a Ridge fitted in PCA space.
    from sklearn.pipeline import Pipeline
    probe_steps = []
    if normalize:
        probe_steps.append(('scaler', StandardScaler()))
    if use_pca:
        n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
        if n_comp >= 1:
            probe_steps.append(('pca', PCA(
                n_components=n_comp, random_state=random_state
            )))
    probe_steps.append(('ridge', Ridge(alpha=alpha)))
    probe = Pipeline(probe_steps)
    probe.fit(X_train, y_train)

    # Evaluate
    train_pred = probe.predict(X_train)
    test_pred = probe.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # Cross-validation (Pipeline ensures scaler/PCA are re-fit per fold, no leakage)
    cv_steps = []
    if normalize:
        cv_steps.append(('scaler', StandardScaler()))
    if use_pca:
        # Account for CV fold size: each fold trains on (cv-1)/cv of the data
        # Two-fold CV is the smallest possible training fold. This conservative
        # bound also keeps PCA valid for small grouped datasets.
        max_pca = max(1, X.shape[0] // 2 - 1)
        cv_steps.append(('pca', PCA(
            n_components=min(n_components, max_pca, X.shape[1]),
            random_state=random_state,
        )))
    cv_steps.append(('ridge', Ridge(alpha=alpha)))
    try:
        if groups is not None:
            from sklearn.model_selection import GroupKFold

            n_splits = min(5, len(np.unique(groups)))
            cv = GroupKFold(n_splits=n_splits)
            cv_scores = cross_val_score(
                Pipeline(cv_steps), X, y, cv=cv, groups=groups, scoring='r2'
            )
        else:
            n_splits = min(5, len(y))
            cv_scores = cross_val_score(
                Pipeline(cv_steps), X, y, cv=n_splits, scoring='r2'
            )
    except ValueError:
        cv_scores = np.array([])

    # Binary metrics
    binary_pred = (test_pred > 0.5).astype(int)
    binary_true = (y_test > 0.5).astype(int)

    accuracy = accuracy_score(binary_true, binary_pred)

    try:
        auc = roc_auc_score(binary_true, test_pred)
    except ValueError:
        auc = 0.5

    result = ProbeResult(
        layer=-1,
        label_type="",
        r2_score=test_r2,
        accuracy=accuracy,
        auc=auc,
        train_r2=train_r2,
        test_r2=test_r2,
        cross_val_scores=cv_scores.tolist(),
    )

    return probe, result


def train_logistic_probe(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    random_state: int = 42,
    normalize: bool = True,
    tune_C: bool = False,
    val_fraction: float = 0.25,
    use_pca: bool = True,
    n_components: int = 100,
    groups: Optional[np.ndarray] = None,
) -> Tuple[Any, ProbeResult]:
    """Train a Logistic Regression probe for binary deception classification.

    This is the recommended probe type for binary deception detection per
    RAPTOR (2026), Apollo Research (2025), and Anthropic (2024).

    Args:
        X: Feature matrix [N, d_model]
        y: Labels [N] (continuous, will be binarized at 0.5)
        C: Inverse regularization strength (lower = more regularization)
        random_state: Random seed for reproducibility
        normalize: Whether to apply StandardScaler (fitted on train only)
        tune_C: If True, sweep C values on a validation set
        val_fraction: Fraction of train to use for validation when tune_C=True
        use_pca: Whether to apply PCA for dimensionality reduction (speeds up training)
        n_components: Number of PCA components

    Returns:
        Tuple of (trained LogisticRegression probe, ProbeResult with metrics)
    """
    # Binarize labels
    y_binary = (np.array(y) > 0.5).astype(int)

    # Check we have both classes
    if len(np.unique(y_binary)) < 2:
        from sklearn.dummy import DummyClassifier

        probe = DummyClassifier(
            strategy='constant', constant=int(y_binary[0])
        ).fit(X, y_binary)
        return probe, ProbeResult(
            layer=-1, label_type="", r2_score=0.0, accuracy=0.5,
            auc=0.5, train_r2=0.0, test_r2=0.0, cross_val_scores=[]
        )

    # Split data
    train_idx, test_idx = _train_test_indices(
        X, y_binary, random_state=random_state, groups=groups
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_binary[train_idx], y_binary[test_idx]
    train_groups = np.asarray(groups)[train_idx] if groups is not None else None

    from sklearn.pipeline import Pipeline

    def build_pipeline(c_value: float, n_fit_rows: int) -> Pipeline:
        steps = []
        if normalize:
            steps.append(('scaler', StandardScaler()))
        if use_pca:
            n_comp = min(n_components, n_fit_rows - 1, X.shape[1])
            if n_comp >= 1:
                steps.append(('pca', PCA(
                    n_components=n_comp, random_state=random_state
                )))
        steps.append(('lr', LogisticRegression(
            C=c_value, max_iter=5000,
            random_state=random_state, solver='lbfgs',
        )))
        return Pipeline(steps)

    # Tune C on validation set if requested
    if tune_C:
        if train_groups is not None:
            from sklearn.model_selection import GroupShuffleSplit
            inner = GroupShuffleSplit(
                n_splits=100, test_size=val_fraction, random_state=random_state
            )
            inner_splits = inner.split(X_train, y_train, groups=train_groups)
            try:
                tr_idx, val_idx = next(
                    (tr, val) for tr, val in inner_splits
                    if len(np.unique(y_train[tr])) > 1
                    and len(np.unique(y_train[val])) > 1
                )
            except StopIteration as exc:
                raise ValueError(
                    "Could not create a class-balanced group validation split"
                ) from exc
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        else:
            min_c = min(np.bincount(y_train)) if len(np.unique(y_train)) > 1 else 0
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=val_fraction,
                random_state=random_state,
                stratify=y_train if min_c >= 2 else None,
            )
        best_C, best_val_auc = C, 0.0
        for c_candidate in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            clf = build_pipeline(c_candidate, len(X_tr))
            clf.fit(X_tr, y_tr)
            try:
                val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
            except ValueError:
                val_auc = 0.5
            if val_auc > best_val_auc:
                best_C, best_val_auc = c_candidate, val_auc
        C = best_C

    # Train the final preprocessing + classifier pipeline on outer-train only.
    probe = build_pipeline(C, len(X_train))
    probe.fit(X_train, y_train)

    # Evaluate
    train_proba = probe.predict_proba(X_train)[:, 1]
    test_proba = probe.predict_proba(X_test)[:, 1]
    test_pred = probe.predict(X_test)

    test_acc = accuracy_score(y_test, test_pred)

    try:
        auc = roc_auc_score(y_test, test_proba)
    except ValueError:
        auc = 0.5

    try:
        train_auc = roc_auc_score(y_train, train_proba)
    except ValueError:
        train_auc = 0.5

    # Cross-validation (Pipeline ensures scaler/PCA are re-fit per fold, no leakage)
    y_all_binary = (np.array(y) > 0.5).astype(int)
    try:
        if groups is not None:
            from sklearn.model_selection import StratifiedGroupKFold
            n_splits = min(5, len(np.unique(groups)))
            cv = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        else:
            min_class_count = min(np.bincount(y_all_binary))
            n_splits = min(5, min_class_count)
            if n_splits < 2:
                raise ValueError("Insufficient class counts for cross-validation")
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        cv_steps = []
        if normalize:
            cv_steps.append(('scaler', StandardScaler()))
        if use_pca:
            max_pca = max(1, X.shape[0] // 2 - 1)
            cv_steps.append(('pca', PCA(
                n_components=min(n_components, max_pca, X.shape[1]),
                random_state=random_state,
            )))
        cv_steps.append(('lr', LogisticRegression(
            C=C, max_iter=5000, solver='lbfgs'
        )))
        cv_scores = cross_val_score(
            Pipeline(cv_steps), X, y_all_binary, cv=cv, scoring='roc_auc',
            groups=groups,
        )
    except ValueError:
        cv_scores = np.array([0.5])

    result = ProbeResult(
        layer=-1,
        label_type="",
        r2_score=float(auc),  # Use AUC as primary metric for logistic
        accuracy=test_acc,
        auc=auc,
        train_r2=float(train_auc),
        test_r2=float(auc),
        cross_val_scores=cv_scores.tolist(),
    )

    return probe, result


def train_probe_multi_seed(
    X: np.ndarray,
    y: np.ndarray,
    probe_type: str = "logistic",
    seeds: List[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Train probes across split seeds and report descriptive sensitivity.

    Split seeds reuse the same observations and therefore are correlated.
    Their spread is a sensitivity diagnostic, not a sampling confidence
    interval. Inferential intervals must instead resample independent connected
    trial-family/dyad units.

    Args:
        X: Feature matrix [N, d_model]
        y: Labels [N]
        probe_type: "logistic", "ridge", or "mass_mean"
        seeds: List of random seeds (default: [42, 123, 456, 789, 1024])
        **kwargs: Additional arguments passed to the probe training function

    Returns:
        Dict with per-seed results and descriptive seed-sensitivity summaries
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    per_seed = []
    for seed in seeds:
        if probe_type == "logistic":
            _, result = train_logistic_probe(X, y, random_state=seed, **kwargs)
        elif probe_type == "ridge":
            _, result = train_ridge_probe(X, y, random_state=seed, **kwargs)
        elif probe_type == "mass_mean":
            _, result = train_mass_mean_probe(X, y, random_state=seed, **kwargs)
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}")

        per_seed.append({
            "seed": seed,
            "auc": result.auc,
            "accuracy": result.accuracy,
            "r2": result.r2_score,
            "train_r2": result.train_r2,
            "test_r2": result.test_r2,
        })

    aucs = [r["auc"] for r in per_seed]
    accs = [r["accuracy"] for r in per_seed]

    n = len(aucs)
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs, ddof=1) if n > 1 else 0.0
    seed_range = (float(np.min(aucs)), float(np.max(aucs)))
    seed_quantiles = tuple(
        float(value) for value in np.quantile(aucs, [0.05, 0.95])
    )

    return {
        "per_seed": per_seed,
        "auc_mean": float(auc_mean),
        "auc_std": float(auc_std),
        "auc_seed_range": seed_range,
        "auc_seed_quantiles_05_95": seed_quantiles,
        "interval_interpretation": (
            "descriptive split-seed sensitivity; not a confidence interval"
        ),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)) if n > 1 else 0.0,
        "n_seeds": n,
        "probe_type": probe_type,
    }


def train_mass_mean_probe(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    random_state: int = 42,
    normalize: bool = True,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ProbeResult]:
    """
    Train a mass-mean direction probe (Marks & Tegmark method).
    Often more robust than Ridge for binary concepts.

    Args:
        X: Feature matrix [N, d_model]
        y: Labels [N]
        threshold: Threshold for binary classification (default 0.5)
        random_state: Random seed for train/test split (default 42)
        normalize: Whether to center activations (subtract training mean)

    Returns:
        Tuple of (direction vector, ProbeResult with metrics)
    """

    binary_y = (y > threshold).astype(bool)

    # Check we have both classes
    if binary_y.sum() == 0 or binary_y.sum() == len(binary_y):
        # All same class - return dummy
        return np.zeros(X.shape[1]), ProbeResult(
            layer=-1, label_type="", r2_score=0.0, accuracy=0.5,
            auc=0.5, train_r2=0.0, test_r2=0.0, cross_val_scores=[]
        )

    # Split FIRST to prevent data leakage in direction computation
    train_idx, test_idx = _train_test_indices(
        X, y, random_state=random_state, groups=groups
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    binary_y_train = (y_train > threshold).astype(bool)

    # Center activations by subtracting mean (Marks & Tegmark method)
    # Fit on train only
    if normalize:
        X_mean = X_train.mean(axis=0)
        X_train = X_train - X_mean
        X_test = X_test - X_mean

    # Compute means on TRAIN set only (prevents test leakage)
    honest_mean = X_train[~binary_y_train].mean(axis=0)
    deceptive_mean = X_train[binary_y_train].mean(axis=0)

    # Direction vector
    direction = deceptive_mean - honest_mean
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        direction = np.zeros_like(direction)
    else:
        direction = direction / norm

    # Project onto direction
    train_proj = X_train @ direction
    test_proj = X_test @ direction

    # Normalize projections using train statistics only
    proj_min, proj_max = train_proj.min(), train_proj.max()
    if proj_max - proj_min < 1e-8:
        test_pred = np.full_like(test_proj, 0.5)
        train_pred = np.full_like(train_proj, 0.5)
    else:
        test_pred = (test_proj - proj_min) / (proj_max - proj_min)
        train_pred = (train_proj - proj_min) / (proj_max - proj_min)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # Binary metrics
    binary_pred = (test_pred > threshold).astype(int)
    binary_true = (y_test > threshold).astype(int)

    accuracy = accuracy_score(binary_true, binary_pred)

    try:
        auc = roc_auc_score(binary_true, test_pred)
    except ValueError:
        auc = 0.5

    result = ProbeResult(
        layer=-1,
        label_type="",
        r2_score=test_r2,
        accuracy=accuracy,
        auc=auc,
        train_r2=train_r2,
        test_r2=test_r2,
        cross_val_scores=[],
    )

    return direction, result


# =============================================================================
# THRESHOLD SENSITIVITY ANALYSIS
# =============================================================================

def threshold_sensitivity_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: List[float] = None,
) -> Dict[str, Any]:
    """Analyze how binary classification metrics vary with threshold choice.

    This function helps understand the impact of the 0.5 binarization threshold
    and whether the probe's performance is robust to threshold changes.

    Args:
        y_true: Ground truth continuous labels [N]
        y_pred: Model predictions [N]
        thresholds: List of thresholds to test (default: [0.3, 0.4, 0.5, 0.6, 0.7])

    Returns:
        Dict with:
        - by_threshold: {threshold: {accuracy, f1, precision, recall}}
        - best_threshold: Threshold with highest F1
        - is_robust: True if performance is similar across thresholds
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = {}
    for thresh in thresholds:
        binary_true = (y_true > thresh).astype(int)
        binary_pred = (y_pred > thresh).astype(int)

        # Skip if only one class present
        if len(np.unique(binary_true)) < 2:
            results[thresh] = {"accuracy": None, "f1": None, "note": "single class"}
            continue

        try:
            results[thresh] = {
                "accuracy": float(accuracy_score(binary_true, binary_pred)),
                "f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
                "precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
                "recall": float(recall_score(binary_true, binary_pred, zero_division=0)),
            }
        except ValueError:
            results[thresh] = {"accuracy": None, "f1": None, "error": "scoring failed"}

    # Find best threshold by F1
    valid_results = {t: r for t, r in results.items() if r.get("f1") is not None}
    if valid_results:
        best_threshold = max(valid_results.keys(), key=lambda t: valid_results[t]["f1"])
        f1_scores = [r["f1"] for r in valid_results.values()]
        is_robust = (max(f1_scores) - min(f1_scores)) < 0.1  # <10% variation
    else:
        best_threshold = 0.5
        is_robust = False

    return {
        "by_threshold": results,
        "best_threshold": float(best_threshold),
        "is_robust": bool(is_robust),
        "recommendation": "Threshold choice robust" if is_robust else "Consider optimizing threshold",
    }


# =============================================================================
# SANITY CHECKS
# =============================================================================

def sanity_check_random_labels(
    X: np.ndarray,
    y: np.ndarray,
    n_shuffles: int = 5,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Sanity check: probes on shuffled labels should give R² near 0."""
    shuffle_r2s = []
    rng = np.random.default_rng(0)

    for seed in range(n_shuffles):
        y_shuffled, unavailable_reason = permute_group_blocks(y, groups, rng)
        if y_shuffled is None:
            return {
                "available": False,
                "mean_shuffled_r2": None,
                "std_shuffled_r2": None,
                "max_shuffled_r2": None,
                "passed": None,
                "message": f"Random-label check unavailable: {unavailable_reason}",
            }
        _, result = train_ridge_probe(
            X, y_shuffled, random_state=seed, groups=groups
        )
        shuffle_r2s.append(result.r2_score)

    return {
        "available": True,
        "mean_shuffled_r2": float(np.mean(shuffle_r2s)),
        "std_shuffled_r2": float(np.std(shuffle_r2s)),
        "max_shuffled_r2": float(np.max(shuffle_r2s)),
        "passed": bool(np.mean(shuffle_r2s) < 0.05),
        "message": "Complete compatible label blocks were permuted",
    }


def sanity_check_train_test_gap(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Check for overfitting."""
    _, result = train_ridge_probe(X, y, groups=groups)

    gap = result.train_r2 - result.test_r2

    return {
        "train_r2": float(result.train_r2),
        "test_r2": float(result.test_r2),
        "gap": float(gap),
        "passed": bool(gap < 0.2),
    }


def sanity_check_label_variance(y: np.ndarray) -> Dict[str, float]:
    """Check that labels have variance."""
    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "n_unique": int(len(np.unique(y))),
        "passed": bool(np.std(y) > 0.1),
    }


# =============================================================================
# PER-LAYER PROBE COMPARISON
# =============================================================================

def train_probes_per_layer(
    activations_by_layer: Dict[int, np.ndarray],
    labels: np.ndarray,
    alpha: float = 100.0,
    use_pca: bool = True,
    n_components: int = 30,
    groups: Optional[np.ndarray] = None,
    partition_labels: Optional[np.ndarray] = None,
    split_manifest: Optional[SplitManifest] = None,
    trial_ids: Optional[np.ndarray] = None,
    diagnostic_non_headline: bool = False,
) -> Dict[int, Dict[str, float]]:
    """
    Train separate probes on each layer to find optimal layer for deception detection.

    This is a critical validation step. Expected pattern from literature:
    - Early layers: Low accuracy (basic features, not semantic)
    - Middle layers: Peak accuracy (high-level concepts encoded)
    - Late layers: Lower accuracy (output formatting)

    If accuracy is flat across layers, something is wrong (probe using surface features).

    Args:
        activations_by_layer: Dict mapping layer_num -> activations [N, d_model]
        labels: Ground truth labels [N]
        alpha: Ridge regularization
        groups: Connected family/dyad identity for each repeated turn
        use_pca: Whether to apply PCA
        n_components: PCA components

    Returns:
        Dict mapping layer_num -> {auc, r2, accuracy, train_r2, test_r2}
    """
    if partition_labels is not None or split_manifest is not None:
        hyperparameters = ({
            "C": max(1e-6, 1.0 / float(alpha)),
            "n_components": n_components if use_pca else min(
                np.asarray(next(iter(activations_by_layer.values()))).shape[1],
                len(labels) - 1,
            ),
        },)
        evaluation = evaluate_nested_grouped_layers(
            activations_by_layer,
            labels,
            partition_labels=partition_labels,
            groups=groups if split_manifest is None else None,
            split_manifest=split_manifest,
            trial_ids=trial_ids,
            hyperparameters=hyperparameters,
        )
        return {
            int(row["layer"]): {
                "auc": row["development_metrics"]["auc"],
                "r2": row["development_metrics"]["r2"],
                "accuracy": row["development_metrics"]["accuracy"],
                "train_r2": float("nan"),
                "test_r2": float("nan"),
                "selection_partition": "development",
                "headline_eligible": True,
            }
            for row in evaluation.selection_table
        }
    if not diagnostic_non_headline:
        raise ValueError(
            "Headline per-layer training requires train/development/test "
            "partitions and connected groups. Set diagnostic_non_headline=True "
            "only for deprecated random-holdout diagnostics."
        )

    results = {}

    for layer_num, X in sorted(activations_by_layer.items()):
        layer_key = int(layer_num)
        # Convert to numpy if tensor
        if hasattr(X, 'numpy'):
            X = X.cpu().numpy() if hasattr(X, 'cpu') else X.numpy()
        if hasattr(labels, 'numpy'):
            y = labels.cpu().numpy() if hasattr(labels, 'cpu') else labels.numpy()
        else:
            y = np.array(labels)

        # Skip if not enough samples
        if len(y) < 10:
            results[layer_key] = {
                'auc': 0.5, 'r2': 0.0, 'accuracy': 0.5,
                'train_r2': 0.0, 'test_r2': 0.0, 'note': 'insufficient samples',
                'selection_partition': 'deprecated_random_holdout',
                'headline_eligible': False,
            }
            continue

        # Check label variance
        if np.std(y) < 0.01:
            results[layer_key] = {
                'auc': 0.5, 'r2': 0.0, 'accuracy': 0.5,
                'train_r2': 0.0, 'test_r2': 0.0, 'note': 'no label variance',
                'selection_partition': 'deprecated_random_holdout',
                'headline_eligible': False,
            }
            continue

        try:
            _, probe_result = train_ridge_probe(
                X, y, alpha=alpha, use_pca=use_pca,
                n_components=n_components, groups=groups,
            )
            results[layer_key] = {
                'auc': probe_result.auc,
                'r2': probe_result.r2_score,
                'accuracy': probe_result.accuracy,
                'train_r2': probe_result.train_r2,
                'test_r2': probe_result.test_r2,
                'selection_partition': 'deprecated_random_holdout',
                'headline_eligible': False,
            }
        except Exception as e:
            results[layer_key] = {
                'auc': 0.5, 'r2': 0.0, 'accuracy': 0.5,
                'train_r2': 0.0, 'test_r2': 0.0, 'error': str(e),
                'selection_partition': 'deprecated_random_holdout',
                'headline_eligible': False,
            }

    return results


def find_best_layer(
    layer_results: Dict[int, Dict[str, float]],
    *,
    diagnostic_non_headline: bool = False,
) -> Dict[str, Any]:
    """
    Find the best layer and analyze the layer accuracy curve.

    Returns:
        Dict with best_layer, peak_auc, curve_analysis, etc.
    """
    if not layer_results:
        return {'best_layer': None, 'error': 'No layer results'}
    headline_eligible = all(
        result.get("selection_partition") == "development"
        and result.get("headline_eligible") is True
        for result in layer_results.values()
    )
    if not headline_eligible and not diagnostic_non_headline:
        raise ValueError(
            "Best-layer selection requires development metrics. Deprecated "
            "random-holdout diagnostics cannot produce a headline layer."
        )

    # Find best by AUC
    best_layer = max(layer_results.keys(), key=lambda l: layer_results[l].get('auc', 0))
    best_auc = layer_results[best_layer]['auc']

    # Analyze curve shape
    layers = sorted(layer_results.keys())
    aucs = [layer_results[l]['auc'] for l in layers]

    # Check for expected inverted-U pattern
    n_layers = len(layers)
    if n_layers >= 3:
        first_third = aucs[:n_layers//3]
        middle_third = aucs[n_layers//3:2*n_layers//3]
        last_third = aucs[2*n_layers//3:]

        first_avg = np.mean(first_third) if first_third else 0
        middle_avg = np.mean(middle_third) if middle_third else 0
        last_avg = np.mean(last_third) if last_third else 0

        # Expected: middle > first and middle > last
        has_expected_shape = middle_avg > first_avg and middle_avg > last_avg
    else:
        has_expected_shape = None
        first_avg = middle_avg = last_avg = None

    # Check for flat curve (red flag)
    auc_std = np.std(aucs)
    is_flat = auc_std < 0.05  # If std < 5%, curve is suspiciously flat

    # Relative position of best layer (0=first, 1=last)
    if len(layers) > 1:
        relative_position = (best_layer - min(layers)) / (max(layers) - min(layers))
    else:
        relative_position = 0.5

    return {
        'best_layer': int(best_layer),
        'peak_auc': float(best_auc),
        'peak_r2': float(layer_results[best_layer].get('r2', 0)),
        'auc_std_across_layers': float(auc_std),
        'is_flat_curve': bool(is_flat),
        'has_expected_inverted_u': has_expected_shape,
        'relative_position': float(relative_position),
        'layer_aucs': {int(l): float(layer_results[l]['auc']) for l in layers},
        'selection_partition': (
            'development' if headline_eligible else 'deprecated_random_holdout'
        ),
        'headline_eligible': headline_eligible,
        'analysis': {
            'first_third_avg_auc': float(first_avg) if first_avg is not None else None,
            'middle_third_avg_auc': float(middle_avg) if middle_avg is not None else None,
            'last_third_avg_auc': float(last_avg) if last_avg is not None else None,
        },
        'warnings': [
            'FLAT CURVE: Probe may be using surface features' if is_flat else None,
            'NO INVERTED-U: Unexpected layer pattern' if has_expected_shape is False else None,
            'EARLY PEAK: Best layer in first third' if relative_position < 0.33 else None,
            'LATE PEAK: Best layer in last third' if relative_position > 0.67 else None,
        ],
    }


def plot_layer_accuracy_curve(
    layer_results: Dict[int, Dict[str, float]],
    output_path: str = 'layer_accuracy_curve.png',
    title: str = 'Probe Accuracy by Layer',
) -> None:
    """
    Generate the standard layer accuracy curve plot.

    This is a key visualization for mechanistic interpretability papers.
    """
    layers = sorted(layer_results.keys())
    aucs = [layer_results[l]['auc'] for l in layers]
    r2s = [layer_results[l]['r2'] for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # AUC plot
    ax1.plot(layers, aucs, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=0.5, color='gray', linestyle='--', label='Random (AUC=0.5)')
    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('AUC by Layer', fontsize=14)
    ax1.set_ylim(0.4, 1.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark best layer
    best_layer = max(layers, key=lambda l: layer_results[l]['auc'])
    best_auc = layer_results[best_layer]['auc']
    ax1.scatter([best_layer], [best_auc], color='red', s=200, zorder=5, marker='*')
    ax1.annotate(f'Best: L{best_layer}\nAUC={best_auc:.3f}',
                 xy=(best_layer, best_auc), xytext=(10, -20),
                 textcoords='offset points', fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='red'))

    # R² plot
    ax2.plot(layers, r2s, 'g-o', linewidth=2, markersize=8)
    ax2.axhline(y=0.0, color='gray', linestyle='--', label='Random (R²=0)')
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('R² by Layer', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved layer accuracy curve to {output_path}")


# =============================================================================
# GENERALIZATION ANALYSIS WITH AUC
# =============================================================================

def compute_generalization_auc(
    X: np.ndarray,
    y: np.ndarray,
    scenarios: List[str],
    groups: np.ndarray,
    alpha: float = 100.0,
    random_state: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Measure leave-one-scenario-out transfer without connected-group leakage.

    Methodology (Leave-One-Scenario-Out Cross-Validation):
        For each scenario S:
        1. Train on ALL samples from scenarios OTHER than S
        2. Test on ALL samples from scenario S
        3. Report R² and AUC for this holdout
        4. Repeat for all scenarios

    Args:
        X: Feature matrix [N, d_model]
        y: Labels [N]
        scenarios: List of scenario names [N], one per sample
        groups: Connected family/dyad identity [N]. Any identity spanning a
            held-out scenario and its training scenarios makes that contrast
            unavailable rather than allowing related examples to leak.
        alpha: Ridge regularization strength
        random_state: Random seed for reproducibility
        verbose: Print per-scenario results

    Returns:
        Dict with:
        - by_scenario: Results for each holdout scenario
        - average_r2: Mean R² across scenarios
        - average_auc: Descriptive mean AUC across available holdouts
        - std_r2/std_auc: Standard deviations
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    scenarios_array = np.asarray(scenarios, dtype=str)
    if groups is None:
        raise ValueError(
            "Cross-scenario transfer requires connected family/dyad groups"
        )
    groups_array = np.asarray(groups, dtype=str)
    if not (
        len(X) == len(y) == len(scenarios_array) == len(groups_array)
    ):
        raise ValueError("X, y, scenarios, and groups must be sample-aligned")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Cross-scenario inputs must be finite")
    unique_scenarios = sorted(set(scenarios_array.tolist()))
    results = {}

    for holdout in unique_scenarios:
        # Split by scenario
        train_mask = scenarios_array != holdout
        test_mask = ~train_mask

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        train_groups = set(groups_array[train_mask].tolist())
        test_groups = set(groups_array[test_mask].tolist())
        crossing_groups = sorted(train_groups & test_groups)
        if crossing_groups:
            results[holdout] = {
                "available": False,
                "train_size": int(train_mask.sum()),
                "test_size": int(test_mask.sum()),
                "test_r2": None,
                "test_auc": None,
                "deception_rate": float(np.mean(y_test)),
                "reason": (
                    "Connected family/dyad groups cross the scenario holdout: "
                    + ", ".join(crossing_groups)
                ),
                "crossing_group_ids": crossing_groups,
            }
            continue

        # Skip if test set has only one class
        if len(np.unique((y_test > 0.5).astype(int))) < 2:
            results[holdout] = {
                "available": False,
                "train_size": int(train_mask.sum()),
                "test_size": int(test_mask.sum()),
                "test_r2": None,
                "test_auc": None,
                "deception_rate": float(np.mean(y_test)),
                "reason": "Single class in test set",
            }
            continue

        if len(np.unique((y_train > 0.5).astype(int))) < 2:
            results[holdout] = {
                "available": False,
                "train_size": int(train_mask.sum()),
                "test_size": int(test_mask.sum()),
                "test_r2": None,
                "test_auc": None,
                "deception_rate": float(np.mean(y_test)),
                "reason": "Single class in training set",
            }
            continue

        # Normalize: fit on train only
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Apply PCA
        n_comp = min(30, X_train_s.shape[0] - 1, X_train_s.shape[1])
        pca = PCA(n_components=n_comp, random_state=random_state)
        X_train_pca = pca.fit_transform(X_train_s)
        X_test_pca = pca.transform(X_test_s)

        # Train Ridge probe
        probe = Ridge(alpha=alpha)
        probe.fit(X_train_pca, y_train)

        test_pred = probe.predict(X_test_pca)
        test_r2 = r2_score(y_test, test_pred)

        # Compute AUC
        binary_true = (y_test > 0.5).astype(int)
        try:
            test_auc = roc_auc_score(binary_true, test_pred)
        except ValueError:
            test_auc = 0.5

        results[holdout] = {
            "available": True,
            "train_size": int(train_mask.sum()),
            "test_size": int(test_mask.sum()),
            "test_r2": float(test_r2),
            "test_auc": float(test_auc),
            "deception_rate": float(np.mean(y_test)),
            "train_group_count": len(train_groups),
            "test_group_count": len(test_groups),
        }

    # Compute averages (excluding None values)
    valid_r2s = [r["test_r2"] for r in results.values() if r["test_r2"] is not None]
    valid_aucs = [r["test_auc"] for r in results.values() if r["test_auc"] is not None]

    return {
        "by_scenario": results,
        "average_r2": float(np.mean(valid_r2s)) if valid_r2s else None,
        "average_auc": float(np.mean(valid_aucs)) if valid_aucs else None,
        "std_r2": float(np.std(valid_r2s)) if valid_r2s else None,
        "std_auc": float(np.std(valid_aucs)) if valid_aucs else None,
        "available_holdouts": len(valid_aucs),
        "total_holdouts": len(unique_scenarios),
        "estimand": "leave-one-scenario-out ranking on connected-group-disjoint rows",
        "interpretation_limit": (
            "AUC measures transfer for these held-out scenarios; it does not "
            "establish a scenario-general deception representation."
        ),
    }


# =============================================================================
# PER-SCENARIO DECEPTION RATES
# =============================================================================

def compute_deception_rates(
    y: np.ndarray,
    scenarios: List[str],
) -> Dict[str, float]:
    """Compute deception rate per scenario."""
    unique_scenarios = list(set(scenarios))
    rates = {}

    for scenario in unique_scenarios:
        mask = np.array([s == scenario for s in scenarios])
        rates[scenario] = float(np.mean(y[mask]))

    return rates


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_model_scales(
    results_9b_path: str,
    results_27b_path: str,
) -> Dict[str, Any]:
    """Compare probe results across model scales."""

    with open(results_9b_path) as f:
        r9b = json.load(f)
    with open(results_27b_path) as f:
        r27b = json.load(f)

    comparison = {
        "9b": {},
        "27b": {},
        "deltas": {},
    }

    # Best probe R²
    comparison["9b"]["best_r2"] = r9b["best_probe"]["r2"]
    comparison["27b"]["best_r2"] = r27b["best_probe"]["r2"]
    comparison["deltas"]["best_r2"] = r27b["best_probe"]["r2"] - r9b["best_probe"]["r2"]

    # Best probe layer
    comparison["9b"]["best_layer"] = r9b["best_probe"]["layer"]
    comparison["27b"]["best_layer"] = r27b["best_probe"]["layer"]

    # Target comparison is valid only when both experiments retained an
    # explicit binary complete-case sample.
    if (
        r9b["gm_vs_agent"].get("available") is True
        and r27b["gm_vs_agent"].get("available") is True
    ):
        gap_9b = (
            r9b["gm_vs_agent"]["gm_ridge_r2"]
            - r9b["gm_vs_agent"]["agent_ridge_r2"]
        )
        gap_27b = (
            r27b["gm_vs_agent"]["gm_ridge_r2"]
            - r27b["gm_vs_agent"]["agent_ridge_r2"]
        )
        comparison["9b"]["gm_agent_gap"] = gap_9b
        comparison["27b"]["gm_agent_gap"] = gap_27b
        comparison["deltas"]["gm_agent_gap"] = gap_27b - gap_9b
    else:
        comparison["gm_agent_comparison_available"] = False

    # Primary actual-deception AUC is independent of counterpart-label
    # availability and remains bound to the locked headline probe.
    comparison["9b"]["gm_auc"] = r9b["best_probe"]["auc"]
    comparison["27b"]["gm_auc"] = r27b["best_probe"]["auc"]
    comparison["deltas"]["gm_auc"] = (
        r27b["best_probe"]["auc"] - r9b["best_probe"]["auc"]
    )

    # Generalization (if available)
    if "generalization" in r9b and "generalization" in r27b:
        if r9b["generalization"].get("average_r2") is not None:
            comparison["9b"]["cross_scenario_r2"] = r9b["generalization"]["average_r2"]
        if r27b["generalization"].get("average_r2") is not None:
            comparison["27b"]["cross_scenario_r2"] = r27b["generalization"]["average_r2"]
        if "cross_scenario_r2" in comparison["9b"] and "cross_scenario_r2" in comparison["27b"]:
            comparison["deltas"]["cross_scenario_r2"] = (
                comparison["27b"]["cross_scenario_r2"] - comparison["9b"]["cross_scenario_r2"]
            )

        if r9b["generalization"].get("average_auc") is not None:
            comparison["9b"]["cross_scenario_auc"] = r9b["generalization"]["average_auc"]
        if r27b["generalization"].get("average_auc") is not None:
            comparison["27b"]["cross_scenario_auc"] = r27b["generalization"]["average_auc"]
        if "cross_scenario_auc" in comparison["9b"] and "cross_scenario_auc" in comparison["27b"]:
            comparison["deltas"]["cross_scenario_auc"] = (
                comparison["27b"]["cross_scenario_auc"] - comparison["9b"]["cross_scenario_auc"]
            )

    return comparison


def print_comparison(comparison: Dict[str, Any]):
    """Pretty-print model comparison."""
    print("\n" + "=" * 60)
    print("MODEL SCALE COMPARISON: Gemma 9B vs 27B")
    print("=" * 60)

    print(f"\n{'Metric':<30} {'9B':<12} {'27B':<12} {'Delta':<12}")
    print("-" * 60)

    metrics = [
        ("Best R²", "best_r2"),
        ("Best Layer", "best_layer"),
        ("GM-Agent Gap", "gm_agent_gap"),
        ("GM AUC", "gm_auc"),
        ("Cross-Scenario R²", "cross_scenario_r2"),
        ("Cross-Scenario AUC", "cross_scenario_auc"),
    ]

    for label, key in metrics:
        val_9b = comparison["9b"].get(key)
        val_27b = comparison["27b"].get(key)
        delta = comparison["deltas"].get(key)

        if val_9b is None and val_27b is None:
            continue

        val_9b_str = f"{val_9b:.3f}" if isinstance(val_9b, float) else str(val_9b) if val_9b else "N/A"
        val_27b_str = f"{val_27b:.3f}" if isinstance(val_27b, float) else str(val_27b) if val_27b else "N/A"
        delta_str = f"{delta:+.3f}" if isinstance(delta, float) else str(delta) if delta else "N/A"

        print(f"{label:<30} {val_9b_str:<12} {val_27b_str:<12} {delta_str:<12}")

    print("=" * 60)


# =============================================================================
# MULTI-AGENT AND CROSS-MODE ANALYSIS FUNCTIONS
# =============================================================================

def group_by_round(
    X: np.ndarray,
    y: np.ndarray,
    round_nums: List[int],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Group activations and labels by round number.

    Args:
        X: Feature matrix [N, d_model]
        y: Labels [N]
        round_nums: Round number for each sample [N]

    Returns:
        Dict mapping round_num -> (X_round, y_round)
    """
    round_nums = np.array(round_nums)
    unique_rounds = sorted(set(round_nums))

    grouped = {}
    for r in unique_rounds:
        mask = round_nums == r
        grouped[int(r)] = (X[mask], y[mask])

    return grouped


def _interpret_trajectory(slopes: Dict[int, float], r2s: Dict[int, float]) -> str:
    """Generate human-readable interpretation of temporal trajectory.

    Args:
        slopes: Dict of layer -> slope values
        r2s: Dict of layer -> R² values for the trajectory fit

    Returns:
        Interpretation string
    """
    avg_slope = np.mean(list(slopes.values()))
    avg_r2 = np.mean(list(r2s.values()))

    return (
        "Observed linear slope of per-round held-out AUC: "
        f"{avg_slope:.3f} (trajectory R²={avg_r2:.3f})."
    )


def compute_cross_mode_transfer(
    X: np.ndarray,
    y: np.ndarray,
    mode_labels: List[str],
    alpha: float = 100.0,
    use_pca: bool = True,
    n_components: int = 30,
    groups: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Measure bidirectional transfer on connected-group-disjoint modes.

    Tests whether deception representations transfer between:
    - Forward: instructed → emergent (explicit to incentive-based)
    - Reverse: emergent → instructed (incentive-based to explicit)

    Args:
        X: Feature matrix [N, d_model]
        y: Labels [N]
        mode_labels: "emergent" or "instructed" for each sample [N]
        alpha: Ridge regularization
        use_pca: Whether to apply PCA
        n_components: PCA components
        groups: Connected family/dyad identity for every row. Transfer is
            unavailable if an identity occurs in both modes.
        random_state: PCA and diagnostic split seed

    Returns:
        Dict with forward_transfer_auc, reverse_transfer_auc, transfer_asymmetry,
        and an explicit statement of the measured estimand and its limits
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mode_labels = np.array(mode_labels)
    if len(mode_labels) != len(y) or len(X) != len(y):
        raise ValueError("X, y, and mode_labels must be sample-aligned")
    if groups is None:
        raise ValueError(
            "Cross-mode transfer requires connected family/dyad groups"
        )
    if len(groups) != len(y):
        raise ValueError("groups must have one value per sample")
    groups_array = np.asarray(groups, dtype=str)
    if not np.isfinite(np.asarray(X, dtype=float)).all():
        raise ValueError("X must contain only finite activations")
    if not np.isfinite(np.asarray(y, dtype=float)).all():
        raise ValueError("y must contain only finite labels")

    # Split by mode
    instructed_mask = mode_labels == "instructed"
    emergent_mask = mode_labels == "emergent"

    n_instructed = instructed_mask.sum()
    n_emergent = emergent_mask.sum()

    if n_instructed < 10 or n_emergent < 10:
        return {
            "error": f"Insufficient samples (instructed={n_instructed}, emergent={n_emergent})",
            "n_instructed": int(n_instructed),
            "n_emergent": int(n_emergent),
            "transfer_auc": None,
            "transfer_r2": None,
        }

    X_instructed = X[instructed_mask]
    y_instructed = y[instructed_mask]
    X_emergent = X[emergent_mask]
    y_emergent = y[emergent_mask]
    instructed_group_ids = set(groups_array[instructed_mask].tolist())
    emergent_group_ids = set(groups_array[emergent_mask].tolist())
    crossing_groups = sorted(instructed_group_ids & emergent_group_ids)
    if crossing_groups:
        return {
            "available": False,
            "error": (
                "Connected family/dyad groups occur in both modes; an "
                "unpaired cross-mode estimand would leak related trials"
            ),
            "crossing_group_ids": crossing_groups,
            "n_instructed": int(n_instructed),
            "n_emergent": int(n_emergent),
        }

    # Use one explicit binary target definition in both directions. Continuous
    # perception estimates belong to a separate regression analysis.
    y_instructed_binary = (y_instructed > 0.5).astype(float)
    y_emergent_binary = (y_emergent > 0.5).astype(float)
    if (
        len(np.unique(y_instructed_binary)) < 2
        or len(np.unique(y_emergent_binary)) < 2
    ):
        return {
            "error": "Both modes require honest and deceptive samples",
            "n_instructed": int(n_instructed),
            "n_emergent": int(n_emergent),
        }

    # Apply PCA on training data (instructed)
    if use_pca:
        n_comp = min(n_components, X_instructed.shape[0] - 1, X_instructed.shape[1])
        pca = PCA(n_components=n_comp, random_state=random_state)
        X_instructed_pca = pca.fit_transform(X_instructed)
        X_emergent_pca = pca.transform(X_emergent)
    else:
        X_instructed_pca = X_instructed
        X_emergent_pca = X_emergent

    # Train on instructed (binary labels for unified scale)
    probe = Ridge(alpha=alpha)
    probe.fit(X_instructed_pca, y_instructed_binary)

    # Test on emergent
    emergent_pred = probe.predict(X_emergent_pca)
    transfer_r2 = r2_score(y_emergent_binary, emergent_pred)

    # Compute AUC for forward transfer (instructed → emergent)
    try:
        forward_transfer_auc = roc_auc_score(y_emergent_binary.astype(int), emergent_pred)
    except ValueError:
        forward_transfer_auc = 0.5

    # === REVERSE TRANSFER: Train on emergent → Test on instructed ===
    # This tests if the relationship is symmetric
    if use_pca:
        n_comp_rev = min(n_components, X_emergent.shape[0] - 1, X_emergent.shape[1])
        pca_rev = PCA(n_components=n_comp_rev, random_state=random_state)
        X_emergent_pca_rev = pca_rev.fit_transform(X_emergent)
        X_instructed_pca_rev = pca_rev.transform(X_instructed)
    else:
        X_emergent_pca_rev = X_emergent
        X_instructed_pca_rev = X_instructed

    # Train on emergent, test on instructed (both on the unified binary scale
    # per 2026-04-21 Finding 7 fix; see comments on the forward direction above)
    probe_reverse = Ridge(alpha=alpha)
    probe_reverse.fit(X_emergent_pca_rev, y_emergent_binary)
    instructed_pred = probe_reverse.predict(X_instructed_pca_rev)
    reverse_transfer_r2 = r2_score(y_instructed_binary, instructed_pred)

    # Compute AUC for reverse transfer (using unified binary labels)
    try:
        reverse_transfer_auc = roc_auc_score(y_instructed_binary.astype(int), instructed_pred)
    except ValueError:
        reverse_transfer_auc = 0.5

    # Also compute within-mode performance for comparison
    # Guard: within-mode datasets may be too small or imbalanced for train/test split
    try:
        instructed_groups = groups_array[instructed_mask]
        _, instructed_result = train_ridge_probe(
            X_instructed, y_instructed_binary, alpha=alpha,
            groups=instructed_groups, random_state=random_state,
        )
    except ValueError:
        instructed_result = None
    try:
        emergent_groups = groups_array[emergent_mask]
        _, emergent_result = train_ridge_probe(
            X_emergent, y_emergent_binary, alpha=alpha,
            groups=emergent_groups, random_state=random_state,
        )
    except ValueError:
        emergent_result = None

    # Compute asymmetry
    transfer_asymmetry = abs(forward_transfer_auc - reverse_transfer_auc)

    return {
        "available": True,
        "n_instructed": int(n_instructed),
        "n_emergent": int(n_emergent),
        # Forward transfer (instructed → emergent); both sides binarized
        "forward_transfer_r2": float(transfer_r2),
        "forward_transfer_auc": float(forward_transfer_auc),
        # Reverse transfer (emergent → instructed); both sides binarized
        "reverse_transfer_r2": float(reverse_transfer_r2),
        "reverse_transfer_auc": float(reverse_transfer_auc),
        "label_scale_unified": True,
        # Legacy field for backwards compatibility
        "transfer_auc": float(forward_transfer_auc),
        "transfer_r2": float(transfer_r2),
        # Asymmetry analysis
        "transfer_asymmetry": float(transfer_asymmetry),
        # Within-mode baselines
        "instructed_within_auc": float(instructed_result.auc) if instructed_result else None,
        "emergent_within_auc": float(emergent_result.auc) if emergent_result else None,
        "instructed_deception_rate": float(np.mean(y_instructed)),
        "emergent_deception_rate": float(np.mean(y_emergent)),
        "estimand": (
            "bidirectional ranking transfer between connected-group-disjoint "
            "instructed and emergent samples"
        ),
        "interpretation": (
            "These AUCs describe out-of-mode ranking on the observed samples."
        ),
        "interpretation_limit": (
            "Transfer magnitude alone does not identify shared or distinct "
            "neural mechanisms."
        ),
    }


def analyze_implicit_encoding(
    X: np.ndarray,
    gm_labels: np.ndarray,
    agent_labels: np.ndarray,
    alpha: float = 100.0,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compare binary targets on one declared complete-case sample.

    ``agent_labels`` are Theory-of-Mind estimates about the counterpart. They
    are not self-reports about the acting agent, so this comparison cannot by
    itself establish hidden self-knowledge or self-awareness.

    Args:
        X: Feature matrix [N, d_model]
        gm_labels: Ground truth deception labels [N]
        agent_labels: Acting agent's estimate of counterpart deception [N]
        alpha: Ridge regularization

    Returns:
        Dict with comparison metrics and significance
    """
    X = np.asarray(X, dtype=float)
    gm_labels = np.asarray(gm_labels, dtype=float)
    agent_labels = np.asarray(agent_labels, dtype=float)
    if not (len(X) == len(gm_labels) == len(agent_labels)):
        raise ValueError("X, gm_labels, and agent_labels must be sample-aligned")
    if groups is None or len(groups) != len(gm_labels):
        raise ValueError(
            "Label-source comparison requires one connected group per row"
        )
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite activations")
    for name, values in (
        ("gm_labels", gm_labels), ("agent_labels", agent_labels)
    ):
        if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
            raise ValueError(
                f"{name} must contain available explicit binary labels"
            )

    # Train probes on each target using identical rows and group identities.
    _, gm_result = train_ridge_probe(
        X, gm_labels, alpha=alpha, groups=groups
    )
    _, agent_result = train_ridge_probe(
        X, agent_labels, alpha=alpha, groups=groups
    )

    # Also train mass-mean probes
    _, gm_mm_result = train_mass_mean_probe(X, gm_labels, groups=groups)
    _, agent_mm_result = train_mass_mean_probe(
        X, agent_labels, groups=groups
    )

    # Compute gap
    auc_gap = gm_result.auc - agent_result.auc
    r2_gap = gm_result.r2_score - agent_result.r2_score

    return {
        "available": True,
        "comparison_n": len(gm_labels),
        "gm_auc": float(gm_result.auc),
        "gm_r2": float(gm_result.r2_score),
        "agent_auc": float(agent_result.auc),
        "agent_r2": float(agent_result.r2_score),
        "gm_mass_mean_auc": float(gm_mm_result.auc),
        "agent_mass_mean_auc": float(agent_mm_result.auc),
        "auc_gap": float(auc_gap),
        "r2_gap": float(r2_gap),
        "gm_wins": bool(auc_gap > 0),
        "supports_self_awareness_claim": False,
        "agent_label_semantics": "perceived counterpart deception, not self-report",
        "estimand": "difference in held-out decodability on complete-case rows",
        "interpretation": (
            f"Observed actual-minus-counterpart AUC difference: {auc_gap:.3f}."
        ),
        "interpretation_limit": (
            "The targets have different semantics; their metric difference is "
            "descriptive and is not evidence of self-awareness."
        ),
    }


def _exact_binary_target(
    values: Any,
    *,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return numeric target values and an exact-binary availability mask."""
    if values is None or len(values) != expected_rows:
        raise ValueError("Target array must have one value per sample")
    numeric = np.full(expected_rows, np.nan, dtype=float)
    available = np.zeros(expected_rows, dtype=bool)
    for index, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            numeric[index] = float(value)
            available[index] = True
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            candidate = float(value)
            if np.isfinite(candidate) and candidate in (0.0, 1.0):
                numeric[index] = candidate
                available[index] = True
    return numeric, available


def analyze_round_trajectory(
    activations_by_layer: Dict[int, np.ndarray],
    y: np.ndarray,
    round_nums: List[int],
    alpha: float = 100.0,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """RQ-MA1: Analyze how deception encoding changes over negotiation rounds.

    Trains probes on each round separately and fits a trajectory to understand
    whether deception representations strengthen, weaken, or stay stable.

    Args:
        activations_by_layer: Dict mapping layer -> activations [N, d_model]
        y: Labels [N]
        round_nums: Round number for each sample [N]
        alpha: Ridge regularization

    Returns:
        Dict with per-round metrics and trajectory analysis
    """
    round_nums = np.array(round_nums)
    if len(round_nums) != len(y):
        raise ValueError("round_nums must have one value per sample")
    if groups is None or len(groups) != len(y):
        raise ValueError(
            "temporal analysis requires one connected group per sample"
        )
    unique_rounds = sorted(set(round_nums))

    if len(unique_rounds) < 3:
        return {
            "error": f"Need >= 3 rounds for trajectory analysis (have {len(unique_rounds)})",
            "n_rounds": len(unique_rounds),
        }

    # Use middle layer for analysis
    layers = sorted(activations_by_layer.keys())
    mid_layer = layers[len(layers) // 2]
    X = activations_by_layer[mid_layer]
    if hasattr(X, 'numpy'):
        X = X.cpu().numpy() if hasattr(X, 'cpu') else X.numpy()

    # Train probe on each round
    per_round_results = {}
    for r in unique_rounds:
        r_key = int(r)
        mask = round_nums == r
        X_r = X[mask]
        y_r = y[mask]

        if len(y_r) < 10 or np.std(y_r) < 0.01:
            per_round_results[r_key] = {
                "available": False,
                "auc": None, "r2": None, "n_samples": int(mask.sum()),
                "deception_rate": float(np.mean(y_r)) if len(y_r) > 0 else 0.0,
                "reason": "insufficient samples or variance"
            }
            continue

        round_groups = np.asarray(groups)[mask]
        try:
            _, result = train_ridge_probe(
                X_r, y_r, alpha=alpha, groups=round_groups
            )
        except ValueError as exc:
            per_round_results[r_key] = {
                "available": False,
                "auc": None,
                "r2": None,
                "n_samples": int(mask.sum()),
                "deception_rate": float(np.mean(y_r)),
                "reason": str(exc),
            }
            continue
        per_round_results[r_key] = {
            "available": True,
            "auc": float(result.auc),
            "r2": float(result.r2_score),
            "n_samples": int(mask.sum()),
            "deception_rate": float(np.mean(y_r)),
        }

    # Fit trajectory (linear regression of AUC over rounds)
    # Filter out rounds with invalid AUC (0.5 default or NaN)
    valid_rounds = [
        r for r in unique_rounds
        if per_round_results[r].get("auc") is not None
        and not np.isnan(per_round_results[r]["auc"])
    ]
    if len(valid_rounds) >= 3:
        rounds_arr = np.array(valid_rounds).reshape(-1, 1)
        aucs_arr = np.array([per_round_results[r]["auc"] for r in valid_rounds])

        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(rounds_arr, aucs_arr)

        slope = float(reg.coef_[0])
        trajectory_r2 = float(reg.score(rounds_arr, aucs_arr))

        interpretation = _interpret_trajectory({mid_layer: slope}, {mid_layer: trajectory_r2})
    else:
        slope = None
        trajectory_r2 = None
        interpretation = "Insufficient valid rounds for trajectory fit"

    return {
        "per_round": per_round_results,
        "n_rounds": len(unique_rounds),
        "trajectory_slope": slope,
        "trajectory_r2": trajectory_r2,
        "interpretation": interpretation,
        "analysis_layer": int(mid_layer),
        "interpretation_limit": (
            "Per-round probe metrics are descriptive and do not establish a "
            "within-trial temporal mechanism."
        ),
    }


def analyze_dyadic_pairs(
    X: np.ndarray,
    y: np.ndarray,
    counterpart_idxs: List[int],
    groups: np.ndarray,
    alpha: float = 100.0,
) -> Dict[str, Any]:
    """RQ-MA2: Analyze deceiver vs victim activation patterns.

    For each agent pair, compare the activation patterns of the deceiver
    (high deception label) vs the victim (low deception label) to understand
    the relational nature of deception.

    Args:
        X: Feature matrix [N, d_model]
        y: Deception labels [N]
        counterpart_idxs: Index of counterpart for each sample [N]
        groups: Connected family/dyad identity for each sample
        alpha: Ridge regularization

    Returns:
        Dict with dyadic analysis metrics
    """
    n_samples = len(y)
    if len(X) != n_samples:
        raise ValueError("X and y must contain the same number of samples")
    if len(counterpart_idxs) != n_samples:
        raise ValueError("counterpart_idxs must have one value per sample")
    if groups is None or len(groups) != n_samples:
        raise ValueError("dyadic analysis requires one connected group per sample")
    groups_array = np.asarray(groups, dtype=str)

    def coerce_index(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            if np.isnan(value):
                return None
        except TypeError:
            pass
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    # Identify valid pairs (both agents have samples). Deduplicate reciprocal
    # links: if sample i has counterpart j and sample j has counterpart i, the
    # pair is the same dyad and must only be counted once, otherwise downstream
    # pair-probe AUC is inflated by train/test leakage across the reciprocal.
    valid_pairs = []
    seen_pairs = set()
    for i in range(n_samples):
        cp_idx = coerce_index(counterpart_idxs[i])
        if cp_idx is None:
            continue
        if 0 <= cp_idx < n_samples and cp_idx != i:
            if coerce_index(counterpart_idxs[cp_idx]) != i:
                continue
            pair_key = (min(i, cp_idx), max(i, cp_idx))
            if pair_key in seen_pairs:
                continue
            if np.isclose(y[i], y[cp_idx]):
                continue
            if groups_array[i] != groups_array[cp_idx]:
                raise ValueError(
                    "Counterpart-linked rows must share a connected group identity"
                )
            seen_pairs.add(pair_key)
            valid_pairs.append(pair_key)

    if len(valid_pairs) < 10:
        return {
            "error": f"Insufficient valid pairs ({len(valid_pairs)})",
            "n_pairs": len(valid_pairs),
        }

    # Compute deception asymmetry per pair
    deceiver_activations = []
    victim_activations = []
    asymmetries = []
    connected_pair_groups = []

    for i, j in valid_pairs:
        # Higher deception label = deceiver
        if y[i] > y[j]:
            deceiver_activations.append(X[i])
            victim_activations.append(X[j])
        else:
            deceiver_activations.append(X[j])
            victim_activations.append(X[i])
        asymmetries.append(abs(y[i] - y[j]))
        connected_pair_groups.append(groups_array[i])

    if len(deceiver_activations) < 10:
        return {
            "error": "Insufficient asymmetric pairs",
            "n_pairs": len(valid_pairs),
        }

    deceiver_X = np.array(deceiver_activations)
    victim_X = np.array(victim_activations)

    # Compare mean activations
    deceiver_mean = deceiver_X.mean(axis=0)
    victim_mean = victim_X.mean(axis=0)

    # Direction from victim to deceiver
    diff_direction = deceiver_mean - victim_mean
    diff_norm = np.linalg.norm(diff_direction)
    if diff_norm > 1e-8:
        diff_direction = diff_direction / diff_norm

    # Project all samples onto this direction
    deceiver_proj = deceiver_X @ diff_direction
    victim_proj = victim_X @ diff_direction

    # Separability (d-prime)
    d_prime = (deceiver_proj.mean() - victim_proj.mean()) / (
        np.sqrt(0.5 * (deceiver_proj.var() + victim_proj.var())) + 1e-8
    )

    # Keep both members of a pair and every related pair from the same
    # family/dyad component on one side of the split.
    pair_X = np.vstack([deceiver_X, victim_X])
    pair_y = np.array([1.0] * len(deceiver_X) + [0.0] * len(victim_X))
    connected_pair_groups_array = np.asarray(connected_pair_groups, dtype=str)
    pair_groups = np.concatenate([
        connected_pair_groups_array,
        connected_pair_groups_array,
    ])

    if len(set(connected_pair_groups)) < 3:
        return {
            "available": False,
            "n_pairs": len(valid_pairs),
            "n_connected_groups": len(set(connected_pair_groups)),
            "pair_probe_auc": None,
            "pair_probe_r2": None,
            "d_prime": float(d_prime),
            "mean_asymmetry": float(np.mean(asymmetries)),
            "reason": "At least three connected groups are required",
        }

    pair_auc, pair_r2 = _train_probe_grouped(
        pair_X, pair_y, pair_groups, alpha=alpha
    )
    if pair_auc is None:
        return {
            "available": False,
            "n_pairs": len(valid_pairs),
            "n_connected_groups": len(set(connected_pair_groups)),
            "pair_probe_auc": None,
            "pair_probe_r2": pair_r2,
            "d_prime": float(d_prime),
            "mean_asymmetry": float(np.mean(asymmetries)),
            "reason": "No class-balanced connected-group holdout was available",
        }

    return {
        "available": True,
        "n_pairs": len(valid_pairs),
        "n_connected_groups": len(set(connected_pair_groups)),
        "pair_probe_auc": float(pair_auc),
        "pair_probe_r2": float(pair_r2) if pair_r2 is not None else None,
        "split_method": "connected_family_dyad_group",
        "d_prime": float(d_prime),
        "mean_asymmetry": float(np.mean(asymmetries)),
        "interpretation": (
            f"Held-out connected-group pair-ranking AUC: {pair_auc:.3f}."
        ),
        "interpretation_limit": (
            "The descriptive full-sample d-prime and held-out AUC do not "
            "identify a victim state or a causal dyadic mechanism."
        ),
    }


def _train_probe_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float = 100.0,
    n_components: int = 30,
    random_state: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Train a ridge probe with a trial-level train/test split.

    Used when the target is defined per-group (e.g., trial outcome) but
    samples are per-row (e.g., per round). A standard train_test_split puts
    the same group in both folds and leaks the backfilled group-level label.
    GroupShuffleSplit guarantees no group appears on both sides.

    Returns (auc, r2). Either may be None if the split is degenerate.
    """
    if len(np.unique(groups)) < 2:
        return None, None

    try:
        train_idx, test_idx = _train_test_indices(
            X, y, random_state=random_state, groups=groups
        )
    except ValueError:
        return None, None

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_comp = min(n_components, X_train_s.shape[0] - 1, X_train_s.shape[1])
    if n_comp >= 1:
        pca = PCA(n_components=n_comp, random_state=random_state)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)
    else:
        X_train_p, X_test_p = X_train_s, X_test_s

    probe = Ridge(alpha=alpha)
    probe.fit(X_train_p, y_train)
    y_pred = probe.predict(X_test_p)

    binary_test = (y_test > 0.5).astype(int)
    if len(np.unique(binary_test)) < 2:
        auc = None
    else:
        try:
            auc = float(roc_auc_score(binary_test, y_pred))
        except ValueError:
            auc = None
    r2 = float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else None
    return auc, r2


def analyze_outcome_prediction(
    X: np.ndarray,
    y: np.ndarray,
    round_nums: List[int],
    group_ids: List[str],
    trial_outcomes: List[bool],
    alpha: float = 100.0,
) -> Dict[str, Any]:
    """RQ-MA3: Predict trial outcome from early-round activations.

    Tests whether deception representations in early rounds (1-2) can predict
    the ultimate trial outcome (success/failure of deception).

    Args:
        X: Feature matrix [N, d_model]
        y: Deception labels [N]
        round_nums: Round number for each sample [N]
        group_ids: Connected family/dyad identity for each sample [N]
        trial_outcomes: Success/failure for each sample [N]
        alpha: Ridge regularization

    Returns:
        Dict with outcome prediction metrics
    """
    if not (
        len(X) == len(y) == len(round_nums) == len(group_ids) == len(trial_outcomes)
    ):
        raise ValueError(
            "X, y, round_nums, group_ids, and trial_outcomes must be sample-aligned"
        )

    round_nums = np.array(round_nums)
    group_ids_arr = np.array([str(value) for value in group_ids])
    success_outcomes = {"agreement", "deal", "success", "accepted", "true", "1"}
    failure_outcomes = {
        "no_agreement", "no deal", "no_deal", "failure", "failed",
        "rejected", "false", "0",
    }

    def normalize_outcome(value: Any) -> Optional[float]:
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isnan(value):
                return None
            if value in (0, 1):
                return float(value)
            return None
        normalized = str(value).strip().lower()
        if normalized in success_outcomes:
            return 1.0
        if normalized in failure_outcomes:
            return 0.0
        return None

    normalized_outcomes = [normalize_outcome(value) for value in trial_outcomes]
    known_outcome_mask = np.array(
        [value is not None for value in normalized_outcomes], dtype=bool
    )
    trial_outcomes = np.array([
        value if value is not None else np.nan for value in normalized_outcomes
    ])

    observed_rounds = sorted(set(round_nums[round_nums >= 0].tolist()))
    early_round_values = observed_rounds[:2]
    late_round_values = observed_rounds[-2:]
    early_mask = np.isin(round_nums, early_round_values) & known_outcome_mask
    n_early = early_mask.sum()

    if n_early < 20:
        return {
            "error": f"Insufficient early-round samples ({n_early})",
            "n_early_samples": int(n_early),
        }

    X_early = X[early_mask]
    outcomes_early = trial_outcomes[early_mask]
    groups_early = group_ids_arr[early_mask]

    # Check outcome variance
    if np.std(outcomes_early) < 0.01:
        return {
            "error": "No outcome variance in early rounds",
            "n_early_samples": int(n_early),
            "outcome_rate": float(np.mean(outcomes_early)),
        }

    # Split on the connected family/dyad unit, never on repeated turn rows.
    early_auc, early_r2 = _train_probe_grouped(
        X_early, outcomes_early, groups_early, alpha=alpha
    )

    # Also try late rounds for comparison
    late_mask = np.isin(round_nums, late_round_values) & known_outcome_mask
    n_late = late_mask.sum()
    groups_late = group_ids_arr[late_mask] if n_late else np.array([])
    if (
        n_late >= 20
        and np.std(trial_outcomes[late_mask]) > 0.01
        and len(np.unique(groups_late)) >= 2
    ):
        late_auc, _ = _train_probe_grouped(
            X[late_mask], trial_outcomes[late_mask], groups_late, alpha=alpha
        )
    else:
        late_auc = None

    return {
        "available": early_auc is not None,
        "n_early_samples": int(n_early),
        "n_early_groups": len(set(groups_early.tolist())),
        "early_round_values": [int(value) for value in early_round_values],
        "late_round_values": [int(value) for value in late_round_values],
        "early_rounds_auc": float(early_auc) if early_auc is not None else None,
        "early_rounds_r2": float(early_r2) if early_r2 is not None else None,
        "late_rounds_auc": float(late_auc) if late_auc is not None else None,
        "outcome_rate": float(np.mean(outcomes_early)),
        "interpretation": (
            "Early connected-group held-out outcome-ranking AUC is "
            + ("unavailable." if early_auc is None else f"{early_auc:.3f}.")
        ),
        "interpretation_limit": (
            "This observational ranking does not establish that early "
            "activations determine outcomes or identify an intervention point."
        ),
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def _sample_keep_mask(
    metadata: List[Dict[str, Any]],
    target_labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build the negotiation/QC mask for an aligned serialized dataset."""
    n_samples = len(target_labels)
    if metadata and len(metadata) != n_samples:
        raise ValueError(
            f"metadata has {len(metadata)} rows, expected {n_samples}"
        )
    if not metadata:
        metadata = [{} for _ in range(n_samples)]

    keep, eligibility_counts = negotiation_sample_mask(metadata, target_labels)
    qc_failures = 0
    from interpretability.core.qc_filter import (
        QC_VERSION,
        classify_sample_response,
    )
    for i, row in enumerate(metadata):
        if not keep[i]:
            continue

        flags = row.get('qc_flags')
        qc_status = row.get('qc_status')
        qc_version = row.get('qc_version')
        if row.get('full_response') is not None:
            recomputed = classify_sample_response(
                row['full_response'],
                scenario=row.get('scenario'),
                semantic_phase=row.get('semantic_phase'),
            )
            if flags is not None and set(flags) != set(recomputed):
                flags = {'qc_metadata_mismatch'}
            else:
                flags = recomputed
            qc_status = 'passed' if not flags else 'rejected'
            qc_version = QC_VERSION
        elif (
            qc_status not in {'passed', 'rejected'}
            or qc_version != QC_VERSION
        ):
            # Headline eligibility fails closed when neither a reviewed QC
            # result nor the response needed to reproduce it is available.
            flags = {'qc_unavailable'}
        elif qc_status == 'passed' and flags:
            flags = set(flags) | {'qc_metadata_mismatch'}
        elif qc_status == 'rejected' and not flags:
            flags = {'qc_metadata_mismatch'}
        if flags:
            keep[i] = False
            qc_failures += 1

    eligibility_counts["eligible_before_qc"] = eligibility_counts["included"]
    eligibility_counts["included"] = int(keep.sum())
    return keep, {
        **eligibility_counts,
        "probe_rounds": (
            eligibility_counts["non_negotiation"]
            + eligibility_counts["invalid_round_or_probe"]
        ),
        "qc_failures": qc_failures,
    }


def _remap_counterpart_indices(
    counterpart_idxs: List[Optional[int]],
    keep_mask: np.ndarray,
) -> List[Optional[int]]:
    """Filter counterpart links and remap surviving old indices."""
    if len(counterpart_idxs) != len(keep_mask):
        raise ValueError("counterpart_idxs must align with keep_mask")
    old_to_new = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(np.flatnonzero(keep_mask))
    }
    remapped = []
    for old_idx in np.flatnonzero(keep_mask):
        counterpart_idx = counterpart_idxs[old_idx]
        if counterpart_idx is None:
            remapped.append(None)
            continue
        try:
            counterpart_idx = int(counterpart_idx)
        except (TypeError, ValueError, OverflowError):
            remapped.append(None)
            continue
        remapped.append(old_to_new.get(counterpart_idx))
    return remapped

def run_full_analysis(
    data_path: str,
    *,
    trusted_legacy: bool = False,
) -> Dict[str, Any]:
    """Run complete probe training and analysis."""

    print(f"\n{'='*60}")
    print("MATS PROBE TRAINING AND ANALYSIS v2")
    print(f"{'='*60}")

    # Safe JSON+NPZ is the default. Pickle-capable .pt input remains available
    # only through the caller's explicit trusted_legacy acknowledgement.
    print(f"\nLoading data from: {data_path}")
    data = load_activation_dataset(
        data_path,
        trusted_legacy=trusted_legacy,
    )

    activations = data["activations"]
    labels = data["labels"]
    config = data.get("config", {})
    metadata = data.get("metadata", [])

    gm_labels = np.array(labels["gm_labels"])
    agent_labels = np.asarray(labels["agent_labels"], dtype=object)
    scenarios = labels["scenario"]

    print(f"Loaded {len(gm_labels)} samples")
    print(f"Layers available: {list(activations.keys())}")

    n_total = len(gm_labels)
    for key in ('agent_labels', 'scenario'):
        if len(labels[key]) != n_total:
            raise ValueError(
                f"Label array '{key}' has {len(labels[key])} rows, expected {n_total}"
            )
    for layer_key, tensor in activations.items():
        if tensor.shape[0] != n_total:
            raise ValueError(
                f"Activation layer {layer_key} has {tensor.shape[0]} rows, "
                f"expected {n_total}"
            )
    for key, values in labels.items():
        if values is not None and hasattr(values, '__len__') and len(values) != n_total:
            raise ValueError(
                f"Label array '{key}' has {len(values)} rows, expected {n_total}"
            )

    keep_mask, filter_counts = _sample_keep_mask(metadata, gm_labels)
    n_dropped = int((~keep_mask).sum())
    original_counterparts = labels.get('counterpart_idxs')
    if n_dropped > 0:
        print(
            f"  Filtering out {filter_counts['probe_rounds']} probe-round and "
            f"{filter_counts['target_unavailable']} unknown-target and "
            f"{filter_counts['qc_failures']} failed-QC samples"
        )
        keep_indices = np.flatnonzero(keep_mask).tolist()
        for layer_key in list(activations.keys()):
            activations[layer_key] = activations[layer_key][keep_indices]
        gm_labels = gm_labels[keep_mask]
        agent_labels = agent_labels[keep_mask]
        scenarios = [s for s, k in zip(scenarios, keep_mask) if k]
        if metadata:
            metadata = [m for m, k in zip(metadata, keep_mask) if k]
        for key, values in list(labels.items()):
            if values is not None and hasattr(values, '__len__'):
                labels[key] = [v for v, k in zip(values, keep_mask) if k]
        if original_counterparts is not None:
            labels['counterpart_idxs'] = _remap_counterpart_indices(
                original_counterparts, keep_mask
            )
        print(f"  Remaining: {len(gm_labels)} negotiation samples")

    if len(gm_labels) == 0:
        raise ValueError("No negotiation samples remain after sample-type/QC filtering")

    labels['gm_labels'] = gm_labels.tolist()
    labels['agent_labels'] = agent_labels.tolist()
    labels['scenario'] = scenarios
    agent_binary, agent_available = _exact_binary_target(
        agent_labels,
        expected_rows=len(gm_labels),
    )

    raw_manifest = data.get("split_manifest")
    headline_manifest = None
    if raw_manifest is not None:
        if isinstance(raw_manifest, str):
            headline_manifest = SplitManifest.from_json(raw_manifest)
        elif isinstance(raw_manifest, dict):
            headline_manifest = SplitManifest.from_json(json.dumps(raw_manifest))
        else:
            raise ValueError("split_manifest must be JSON text or a dictionary")
    headline_partitions = labels.get("split_partitions")
    if headline_partitions is None:
        headline_partitions = labels.get("partitions")
    if headline_partitions is None and metadata:
        headline_partitions = [
            row.get("split_partition", row.get("partition")) for row in metadata
        ]
    headline_groups = labels.get("split_groups")
    if headline_groups is None:
        headline_groups = labels.get("connected_group_ids")
    if headline_groups is None and metadata:
        headline_groups = [
            row.get("split_group_id", row.get("connected_group_id"))
            for row in metadata
        ]
    headline_trial_ids = labels.get("trial_ids")
    if headline_trial_ids is None and metadata:
        headline_trial_ids = [row.get("trial_id") for row in metadata]

    if headline_manifest is not None:
        if (
            headline_trial_ids is None
            or len(headline_trial_ids) != len(gm_labels)
        ):
            raise ValueError(
                "SplitManifest analysis requires one trial_id per retained row"
            )
        _, analysis_groups = resolve_manifest_row_groups(
            headline_manifest, headline_trial_ids
        )
        split_unit = "connected trial-family/dyad component from SplitManifest"
    else:
        if headline_groups is None or len(headline_groups) != len(gm_labels):
            raise ValueError(
                "Analysis requires one connected trial-family/dyad group per row"
            )
        if any(value is None or not str(value).strip() for value in headline_groups):
            raise ValueError("Connected group identities cannot be null or empty")
        analysis_groups = np.asarray(headline_groups, dtype=str)
        split_unit = "connected trial-family/dyad group"

    results = {
        "sanity_checks": {},
        "layer_analysis": {},
        "gm_vs_agent": {},
        "generalization": {},
        "deception_rates": {},
        "best_probe": None,
        "filtering": filter_counts,
        "agent_label_eligibility": {
            "total_gm_eligible": len(gm_labels),
            "binary_available": int(agent_available.sum()),
            "excluded_unavailable_or_nonbinary": int((~agent_available).sum()),
            "policy": "explicit 0/1 counterpart-perception labels only",
        },
        "split_unit": split_unit,
    }

    # Deception rates
    print(f"\n{'='*60}")
    print("DECEPTION RATES BY SCENARIO")
    print(f"{'='*60}")

    deception_rates = compute_deception_rates(gm_labels, scenarios)
    results["deception_rates"] = deception_rates

    for scenario, rate in sorted(deception_rates.items(), key=lambda x: x[1]):
        print(f"  {scenario}: {rate*100:.1f}%")

    # Choose primary layer (mid-layer)
    layers = sorted(
        [key for key in activations if str(key).isdigit()],
        key=lambda key: int(key),
    )
    if not layers:
        raise ValueError("Dataset has no last-token numeric activation layers")
    non_primary_layers = [key for key in activations if key not in layers]
    if non_primary_layers:
        results['supplemental_activation_keys'] = [
            str(key) for key in non_primary_layers
        ]
    mid_layer = layers[len(layers) // 2]
    X_mid = activations[mid_layer].float().numpy()

    print(f"\nPrimary analysis layer: {mid_layer}")
    print(f"Activation shape: {X_mid.shape}")

    # ==========================================================================
    # SANITY CHECKS
    # ==========================================================================
    print(f"\n{'='*60}")
    print("SANITY CHECKS")
    print(f"{'='*60}")

    # Check 1: Label variance
    print("\n1. Label Variance Check")
    variance_check = sanity_check_label_variance(gm_labels)
    results["sanity_checks"]["label_variance"] = variance_check
    status = "PASSED" if variance_check["passed"] else "FAILED"
    print(f"   GM labels - mean: {variance_check['mean']:.3f}, std: {variance_check['std']:.3f}")
    print(f"   {status}")

    # Check 2: Random labels
    print("\n2. Random Labels Check")
    random_check = sanity_check_random_labels(
        X_mid, gm_labels, groups=analysis_groups
    )
    results["sanity_checks"]["random_labels"] = random_check
    if not random_check.get("available", True):
        print(f"   UNAVAILABLE: {random_check['message']}")
    else:
        status = "PASSED" if random_check["passed"] else "FAILED"
        print(f"   Shuffled R²: {random_check['mean_shuffled_r2']:.3f} +/- {random_check['std_shuffled_r2']:.3f}")
        print(f"   {status} (should be near 0)")

    # Check 3: Train-test gap
    print("\n3. Train-Test Gap Check")
    gap_check = sanity_check_train_test_gap(
        X_mid, gm_labels, groups=analysis_groups
    )
    results["sanity_checks"]["train_test_gap"] = gap_check
    status = "PASSED" if gap_check["passed"] else "FAILED"
    print(f"   Train R²: {gap_check['train_r2']:.3f}, Test R²: {gap_check['test_r2']:.3f}, Gap: {gap_check['gap']:.3f}")
    print(f"   {status}")

    # ==========================================================================
    # LAYER COMPARISON
    # ==========================================================================
    print(f"\n{'='*60}")
    print("LAYER ANALYSIS")
    print(f"{'='*60}")

    activation_arrays = {
        int(layer): activations[layer].float().numpy() for layer in layers
    }
    if headline_manifest is not None:
        if (
            headline_trial_ids is None
            or len(headline_trial_ids) != len(gm_labels)
        ):
            raise ValueError(
                "Manifest-backed headline evaluation requires aligned trial_ids"
            )
        nested_kwargs = {
            "split_manifest": headline_manifest,
            "trial_ids": headline_trial_ids,
        }
    else:
        nested_kwargs = {
            "partition_labels": headline_partitions,
            "groups": headline_groups,
        }

    gm_headline = evaluate_nested_grouped_layers(
        activation_arrays,
        gm_labels,
        **nested_kwargs,
    )
    comparison_mask = agent_available.copy()
    agent_headline = None
    gm_comparison_headline = None
    agent_comparison_reason = None
    if (
        int(comparison_mask.sum()) > 0
        and len(np.unique(agent_binary[comparison_mask])) == 2
        and len(np.unique(gm_labels[comparison_mask])) == 2
    ):
        comparison_activations = {
            layer: array[comparison_mask]
            for layer, array in activation_arrays.items()
        }
        if headline_manifest is not None:
            comparison_kwargs = {
                "split_manifest": headline_manifest,
                "trial_ids": np.asarray(headline_trial_ids)[
                    comparison_mask
                ].tolist(),
            }
        else:
            comparison_kwargs = {
                "partition_labels": np.asarray(headline_partitions)[
                    comparison_mask
                ].tolist(),
                "groups": analysis_groups[comparison_mask],
            }
        try:
            gm_comparison_headline = evaluate_nested_grouped_layers(
                comparison_activations,
                gm_labels[comparison_mask],
                **comparison_kwargs,
            )
            agent_headline = evaluate_nested_grouped_layers(
                comparison_activations,
                agent_binary[comparison_mask],
                **comparison_kwargs,
            )
        except ValueError as exc:
            agent_comparison_reason = str(exc)
            gm_comparison_headline = None
            agent_headline = None
    else:
        agent_comparison_reason = (
            "Complete-case actual and counterpart targets require both classes"
        )

    def best_development_rows(evaluation):
        rows_by_layer = {}
        for row in evaluation.selection_table:
            layer = int(row["layer"])
            current = rows_by_layer.get(layer)
            if (
                current is None
                or row["development_metrics"]["auc"]
                > current["development_metrics"]["auc"]
            ):
                rows_by_layer[layer] = row
        return rows_by_layer

    gm_rows = best_development_rows(gm_headline)
    agent_rows = (
        best_development_rows(agent_headline)
        if agent_headline is not None else {}
    )
    layer_results = {}
    for layer in layers:
        gm_metrics = gm_rows[int(layer)]["development_metrics"]
        agent_metrics = (
            agent_rows[int(layer)]["development_metrics"]
            if int(layer) in agent_rows else None
        )
        layer_results[layer] = {
            "gm": {
                "auc": gm_metrics["auc"],
                "r2": gm_metrics["r2"],
                "r2_score": gm_metrics["r2"],
                "accuracy": gm_metrics["accuracy"],
                "selection_partition": "development",
                "headline_eligible": True,
            },
            "agent": (
                {
                    "available": True,
                    "auc": agent_metrics["auc"],
                    "r2": agent_metrics["r2"],
                    "r2_score": agent_metrics["r2"],
                    "accuracy": agent_metrics["accuracy"],
                    "selection_partition": "development",
                    "headline_eligible": True,
                }
                if agent_metrics is not None else {
                    "available": False,
                    "reason": agent_comparison_reason,
                    "headline_eligible": False,
                }
            ),
        }
        print(f"\nLayer {layer} (development selection only):")
        print(f"  GM labels    - R²: {gm_metrics['r2']:.3f}, AUC: {gm_metrics['auc']:.3f}")
        if agent_metrics is not None:
            print(
                f"  Agent labels - R²: {agent_metrics['r2']:.3f}, "
                f"AUC: {agent_metrics['auc']:.3f}"
            )
        else:
            print(f"  Agent labels - unavailable: {agent_comparison_reason}")

    best_layer = gm_headline.selected_layer
    selected_development = gm_headline.split_provenance[
        "selected_development_metrics"
    ]
    best_auc = gm_headline.test_metrics["auc"]
    best_r2 = gm_headline.test_metrics["r2"]
    results["layer_analysis"] = layer_results
    results["nested_grouped_evaluation"] = gm_headline.to_dict()
    results["counterpart_nested_grouped_evaluation"] = (
        agent_headline.to_dict() if agent_headline is not None else {
            "available": False,
            "reason": agent_comparison_reason,
            "comparison_n": int(comparison_mask.sum()),
        }
    )
    results["best_probe"] = {
        "layer": int(best_layer),
        "auc": float(best_auc),
        "r2": float(best_r2),
        "development_auc": float(selected_development["auc"]),
        "selection_partition": "development",
        "evaluation_partition": "locked_test",
        "test_evaluations": 1,
    }

    print(
        f"\nSelected layer: {best_layer} "
        f"(development AUC = {selected_development['auc']:.3f}); "
        f"locked-test AUC = {best_auc:.3f}, R² = {best_r2:.3f}"
    )

    # ==========================================================================
    # GM vs AGENT COMPARISON (Core Metric)
    # ==========================================================================
    print(f"\n{'='*60}")
    print("GM vs AGENT LABEL COMPARISON")
    print(f"{'='*60}")
    print("This compares probe performance on:")
    print("  GM labels:    Ground truth - is agent ACTUALLY deceiving?")
    print("  Agent labels: acting agent's estimate of COUNTERPART deception")
    print("-" * 60)
    print("This comparison does not measure self-report or self-awareness.")
    print("-" * 60)

    X_best = activations[best_layer].float().numpy()

    # Mass-mean probe
    _, gm_mm_result = train_mass_mean_probe(
        X_best, gm_labels, groups=analysis_groups
    )

    print(f"\nGM (Ground Truth) - What the agent IS doing:")
    print(f"  Locked R²:    {gm_headline.test_metrics['r2']:.3f}")
    print(f"  Locked AUC:   {gm_headline.test_metrics['auc']:.3f}")
    print(f"  Mass-Mean R²: {gm_mm_result.r2_score:.3f}")
    print(f"  Mass-Mean AUC:{gm_mm_result.auc:.3f}")

    comparison_groups = analysis_groups[comparison_mask]
    if agent_headline is not None and gm_comparison_headline is not None:
        comparison_X = X_best[comparison_mask]
        try:
            _, comparison_gm_mm = train_mass_mean_probe(
                comparison_X,
                gm_labels[comparison_mask],
                groups=comparison_groups,
            )
            _, comparison_agent_mm = train_mass_mean_probe(
                comparison_X,
                agent_binary[comparison_mask],
                groups=comparison_groups,
            )
        except ValueError:
            comparison_gm_mm = None
            comparison_agent_mm = None
        auc_gap = (
            gm_comparison_headline.test_metrics["auc"]
            - agent_headline.test_metrics["auc"]
        )
        results["gm_vs_agent"] = {
            "available": True,
            "comparison_n": int(comparison_mask.sum()),
            "excluded_n": int((~comparison_mask).sum()),
            "gm_ridge_r2": float(
                gm_comparison_headline.test_metrics["r2"]
            ),
            "agent_ridge_r2": float(agent_headline.test_metrics["r2"]),
            "gm_mass_mean_r2": (
                float(comparison_gm_mm.r2_score)
                if comparison_gm_mm is not None else None
            ),
            "agent_mass_mean_r2": (
                float(comparison_agent_mm.r2_score)
                if comparison_agent_mm is not None else None
            ),
            "gm_auc": float(gm_comparison_headline.test_metrics["auc"]),
            "agent_auc": float(agent_headline.test_metrics["auc"]),
            "auc_gap": float(auc_gap),
            "gm_mass_mean_auc": (
                float(comparison_gm_mm.auc)
                if comparison_gm_mm is not None else None
            ),
            "agent_mass_mean_auc": (
                float(comparison_agent_mm.auc)
                if comparison_agent_mm is not None else None
            ),
            "gm_wins": bool(auc_gap > 0),
            "gm_layer": int(gm_comparison_headline.selected_layer),
            "agent_layer": int(agent_headline.selected_layer),
            "comparison_partition": "locked_test",
            "comparison_policy": "complete-case explicit binary targets",
            "mass_mean_is_non_headline_diagnostic": True,
            "agent_label_semantics": (
                "perceived counterpart deception, not self-report"
            ),
            "supports_self_awareness_claim": False,
            "interpretation_limit": (
                "Different target semantics make the AUC gap descriptive."
            ),
        }
        print(f"\nAgent ToM - perceived COUNTERPART deception:")
        print(f"  Complete-case n: {int(comparison_mask.sum())}")
        print(f"  Locked R²:    {agent_headline.test_metrics['r2']:.3f}")
        print(f"  Locked AUC:   {agent_headline.test_metrics['auc']:.3f}")
        print(f"  Actual-minus-counterpart AUC: {auc_gap:.3f}")
    else:
        results["gm_vs_agent"] = {
            "available": False,
            "comparison_n": int(comparison_mask.sum()),
            "excluded_n": int((~comparison_mask).sum()),
            "reason": agent_comparison_reason,
            "comparison_policy": "complete-case explicit binary targets",
            "agent_label_semantics": (
                "perceived counterpart deception, not self-report"
            ),
            "supports_self_awareness_claim": False,
        }
        print(
            "\nAgent ToM comparison unavailable: "
            f"{agent_comparison_reason}"
        )

    # ==========================================================================
    # GENERALIZATION WITH AUC
    # ==========================================================================
    print(f"\n{'='*60}")
    print("GENERALIZATION ANALYSIS (with AUC)")
    print(f"{'='*60}")

    unique_scenarios = list(set(scenarios))

    if len(unique_scenarios) >= 3:
        gen_results = compute_generalization_auc(
            X_best, gm_labels, scenarios, groups=analysis_groups
        )
        results["generalization"] = gen_results

        for holdout, res in gen_results["by_scenario"].items():
            r2_str = f"{res['test_r2']:.3f}" if res['test_r2'] is not None else "N/A"
            auc_str = f"{res['test_auc']:.3f}" if res['test_auc'] is not None else "N/A"
            rate_str = f"{res['deception_rate']*100:.0f}%"
            print(f"\nHoldout: {holdout} (deception rate: {rate_str})")
            print(f"  R²:  {r2_str}")
            print(f"  AUC: {auc_str}")
            if not res.get("available", False):
                print(f"  Unavailable: {res['reason']}")

        print(f"\n--- AVERAGES ---")
        if gen_results["average_r2"] is not None:
            print(f"Average cross-scenario R²:  {gen_results['average_r2']:.3f} +/- {gen_results['std_r2']:.3f}")
        if gen_results["average_auc"] is not None:
            print(f"Average cross-scenario AUC: {gen_results['average_auc']:.3f} +/- {gen_results['std_auc']:.3f}")

        # Explain R² vs AUC difference
        if gen_results["average_r2"] is not None and gen_results["average_auc"] is not None:
            if gen_results["average_r2"] < 0 and gen_results["average_auc"] > 0.5:
                print(f"\n>> NOTE: Negative R² with positive AUC is expected!")
                print(f"   R² is sensitive to base rate differences between scenarios.")
                print(f"   AUC measures ranking ability, which transfers better.")
    else:
        print("Not enough scenarios for generalization analysis")

    # ==========================================================================
    # RQ1: CROSS-MODE TRANSFER ANALYSIS
    # ==========================================================================
    mode_labels = labels.get("mode_labels", [])
    if mode_labels and len(set(mode_labels)) >= 2:
        print(f"\n{'='*60}")
        print("RQ1: CROSS-MODE TRANSFER ANALYSIS")
        print(f"{'='*60}")
        print("Question: How well does an out-of-mode ranking transfer?")
        print("-" * 60)
        print("Transfer AUC does not by itself identify shared neural mechanisms.")
        print("-" * 60)

        cross_mode_results = compute_cross_mode_transfer(
            X_best, gm_labels, mode_labels, groups=analysis_groups
        )
        results["cross_mode_transfer"] = cross_mode_results

        if "error" not in cross_mode_results:
            print(f"\nSample Counts:")
            print(f"  Instructed: {cross_mode_results['n_instructed']}")
            print(f"  Emergent: {cross_mode_results['n_emergent']}")
            print(f"\nWithin-mode AUC (sanity check - should be >0.6):")
            print(f"  Instructed: {cross_mode_results['instructed_within_auc']:.3f}")
            print(f"  Emergent: {cross_mode_results['emergent_within_auc']:.3f}")
            print(f"\nBidirectional cross-mode transfer:")
            print(f"  Forward (instructed → emergent):")
            print(f"    AUC: {cross_mode_results['forward_transfer_auc']:.3f}")
            print(f"    R²:  {cross_mode_results['forward_transfer_r2']:.3f}")
            print(f"  Reverse (emergent → instructed):")
            print(f"    AUC: {cross_mode_results['reverse_transfer_auc']:.3f}")
            print(f"    R²:  {cross_mode_results['reverse_transfer_r2']:.3f}")
            print(f"  Asymmetry: {cross_mode_results['transfer_asymmetry']:.3f}")
            print(f"\n{cross_mode_results['interpretation']}")
        else:
            print(f"\n{cross_mode_results['error']}")
    else:
        print(f"\n{'='*60}")
        print("RQ1: CROSS-MODE TRANSFER ANALYSIS")
        print(f"{'='*60}")
        print("Skipped - no mode labels or single mode only")
        results["cross_mode_transfer"] = {"skipped": True, "reason": "No mode labels or single mode"}

    # ==========================================================================
    # RQ2: IMPLICIT ENCODING ANALYSIS
    # ==========================================================================
    print(f"\n{'='*60}")
    print("RQ2: LABEL-SOURCE REPRESENTATION ANALYSIS")
    print(f"{'='*60}")
    print("Question: How does decodability differ between acting-agent")
    print("          behavior and its ToM estimate of the counterpart?")
    print("-" * 60)
    print("Interpretation Guide:")
    print("  These are different targets; an AUC gap is label-source divergence.")
    print("  It is not evidence of hidden self-knowledge or self-awareness.")
    print("-" * 60)

    if agent_headline is not None and gm_comparison_headline is not None:
        implicit_results = analyze_implicit_encoding(
            X_best[comparison_mask],
            gm_labels[comparison_mask],
            agent_binary[comparison_mask],
            groups=comparison_groups,
        )
        results["implicit_encoding"] = implicit_results

        print(f"\nComplete-case GM vs counterpart-target decodability:")
        print(f"  GM (Ground Truth) AUC:    {implicit_results['gm_auc']:.3f}")
        print(f"  Counterpart-belief AUC:   {implicit_results['agent_auc']:.3f}")
        print(f"  AUC Gap (GM - Agent):     {implicit_results['auc_gap']:.3f}")
        print(f"\n{implicit_results['interpretation']}")
    else:
        results["implicit_encoding"] = {
            "available": False,
            "reason": agent_comparison_reason,
            "comparison_n": int(comparison_mask.sum()),
            "supports_self_awareness_claim": False,
        }
        print(f"\nUnavailable: {agent_comparison_reason}")

    # ==========================================================================
    # RQ-MA1: TEMPORAL TRAJECTORY ANALYSIS
    # ==========================================================================
    round_nums = labels.get("round_nums", [])
    if round_nums and len(set(round_nums)) >= 3:
        print(f"\n{'='*60}")
        print("RQ-MA1: TEMPORAL TRAJECTORY ANALYSIS")
        print(f"{'='*60}")
        print("Question: Does deception appear suddenly or build up over rounds?")
        print("          This reports a descriptive per-round AUC trajectory.")
        print("-" * 60)
        print("The slope does not establish a temporal neural mechanism.")
        print("-" * 60)

        # Convert activations dict for trajectory analysis
        activations_np = {k: v.float().numpy() if hasattr(v, 'float') else v
                        for k, v in activations.items()}

        trajectory_results = analyze_round_trajectory(
            activations_np, gm_labels, round_nums, groups=analysis_groups
        )
        results["temporal_trajectory"] = trajectory_results

        if "error" not in trajectory_results:
            print(f"\nAnalysis layer: {trajectory_results['analysis_layer']}")
            print(f"Number of rounds: {trajectory_results['n_rounds']}")
            print(f"\nPer-round Deception Detection:")
            for r, res in sorted(trajectory_results["per_round"].items()):
                auc_str = (
                    f"{res['auc']:.3f}"
                    if res.get('auc') is not None else "N/A"
                )
                print(f"  Round {r}: AUC={auc_str}, n={res['n_samples']}, deception_rate={res['deception_rate']*100:.0f}%")
            print(f"\nDescriptive temporal pattern:")
            slope = trajectory_results['trajectory_slope']
            trajectory_r2 = trajectory_results['trajectory_r2']
            print(
                "  Trajectory slope: "
                + ("N/A" if slope is None else f"{slope:.3f}")
            )
            print(
                "  Trajectory R²: "
                + ("N/A" if trajectory_r2 is None else f"{trajectory_r2:.3f}")
            )
            print(f"\n{trajectory_results['interpretation']}")
        else:
            print(f"\n{trajectory_results['error']}")
    else:
        print(f"\n{'='*60}")
        print("RQ-MA1: TEMPORAL TRAJECTORY ANALYSIS")
        print(f"{'='*60}")
        print("Skipped - insufficient round data")
        results["temporal_trajectory"] = {"skipped": True, "reason": "Insufficient round data"}

    # ==========================================================================
    # RQ-MA2: DYADIC PAIR ANALYSIS
    # ==========================================================================
    counterpart_idxs = labels.get("counterpart_idxs", [])
    def has_valid_counterpart_index(values: List[Any]) -> bool:
        for value in values:
            try:
                if value is not None and not np.isnan(value) and int(value) >= 0:
                    return True
            except (TypeError, ValueError, OverflowError):
                continue
        return False

    if counterpart_idxs and has_valid_counterpart_index(counterpart_idxs):
        print(f"\n{'='*60}")
        print("RQ-MA2: DYADIC PAIR ANALYSIS")
        print(f"{'='*60}")
        print("Question: What happens in BOTH agents' activations during")
        print("          differently labeled behavior within counterpart pairs?")
        print("-" * 60)
        print("Metrics are descriptive and do not identify a victim state.")
        print("-" * 60)

        dyadic_results = analyze_dyadic_pairs(
            X_best,
            gm_labels,
            counterpart_idxs,
            groups=analysis_groups,
        )
        results["dyadic_pairs"] = dyadic_results

        if dyadic_results.get("available") is True:
            print(f"\nValid pairs: {dyadic_results['n_pairs']}")
            print(f"\nConnected-group dyadic diagnostics:")
            print(f"  Deceiver/Victim probe AUC: {dyadic_results['pair_probe_auc']:.3f}")
            print(f"  D-prime (separability):    {dyadic_results['d_prime']:.3f}")
            print(f"  Mean asymmetry:            {dyadic_results['mean_asymmetry']:.3f}")
            print(f"\n{dyadic_results['interpretation']}")
            print(f"Limit: {dyadic_results['interpretation_limit']}")
        else:
            print(
                f"\n{dyadic_results.get('error', dyadic_results.get('reason'))}"
            )
    else:
        print(f"\n{'='*60}")
        print("RQ-MA2: DYADIC PAIR ANALYSIS")
        print(f"{'='*60}")
        print("Skipped - no counterpart indices")
        results["dyadic_pairs"] = {"skipped": True, "reason": "No counterpart indices"}

    # ==========================================================================
    # RQ-MA3: OUTCOME PREDICTION ANALYSIS
    # ==========================================================================
    trial_ids = labels.get("trial_ids", [])
    trial_outcomes = labels.get("trial_outcomes", [])
    if round_nums and trial_outcomes and len(trial_outcomes) > 0:
        print(f"\n{'='*60}")
        print("RQ-MA3: OUTCOME PREDICTION ANALYSIS")
        print(f"{'='*60}")
        print("Question: Can early-round activations predict negotiation")
        print("          outcome on connected-group-held-out rows?")
        print("-" * 60)
        print("This observational analysis does not identify an intervention point.")
        print("-" * 60)

        outcome_group_ids = (
            analysis_groups.tolist()
            if analysis_groups is not None
            else trial_ids
        )
        outcome_results = analyze_outcome_prediction(
            X_best, gm_labels, round_nums, outcome_group_ids, trial_outcomes
        )
        results["outcome_prediction"] = outcome_results

        if "error" not in outcome_results:
            print(f"\nSample Counts:")
            print(f"  Early-round samples (rounds 1-2): {outcome_results['n_early_samples']}")
            print(f"  Outcome rate (success): {outcome_results['outcome_rate']*100:.1f}%")
            print(f"\nConnected-group outcome ranking:")
            early_auc = outcome_results['early_rounds_auc']
            print(
                "  Early-round prediction AUC: "
                + ("N/A" if early_auc is None else f"{early_auc:.3f}")
            )
            if outcome_results['late_rounds_auc'] is not None:
                print(f"  Late-round prediction AUC:  {outcome_results['late_rounds_auc']:.3f}")
            print(f"\n{outcome_results['interpretation']}")
            print(f"Limit: {outcome_results['interpretation_limit']}")
        else:
            print(f"\n{outcome_results['error']}")
    else:
        print(f"\n{'='*60}")
        print("RQ-MA3: OUTCOME PREDICTION ANALYSIS")
        print(f"{'='*60}")
        print("Skipped - no outcome data")
        results["outcome_prediction"] = {"skipped": True, "reason": "No outcome data"}

    # ==========================================================================
    # RESEARCH QUESTION SUMMARY
    # ==========================================================================
    print(f"\n{'='*60}")
    print("RESEARCH QUESTION SUMMARY")
    print(f"{'='*60}")
    print("Quick reference for interpreting results:")
    print("-" * 60)

    print(f"\n[TECHNICAL VALIDATION]")
    print(f"  Sanity Checks: ", end="")
    all_passed = all(
        check.get("passed", True)
        for check in results["sanity_checks"].values()
    )
    print(f"{'PASSED' if all_passed else 'SOME FAILED'}")
    print(
        f"  Best Probe: Layer {best_layer}, R²={best_r2:.3f}, "
        f"AUC={gm_headline.test_metrics['auc']:.3f}"
    )

    if results["generalization"].get("average_auc") is not None:
        print(f"  Cross-scenario: R²={results['generalization']['average_r2']:.3f}, AUC={results['generalization']['average_auc']:.3f}")

    print(f"\n" + "-" * 60)
    print("[RESEARCH FINDINGS]")
    print("-" * 60)

    # RQ1: Cross-mode transfer
    if (
        "cross_mode_transfer" in results
        and results["cross_mode_transfer"].get("available") is True
    ):
        xm = results["cross_mode_transfer"]
        print(f"\nRQ1: Instructed vs Emergent Deception")
        print("  Q: How well does bidirectional out-of-mode ranking transfer?")
        transfer_values = [
            value for value in (
                xm.get('forward_transfer_auc'),
                xm.get('reverse_transfer_auc'),
            ) if value is not None
        ]
        transfer_auc = float(np.mean(transfer_values)) if transfer_values else 0.5
        print(f"  Mean bidirectional transfer AUC: {transfer_auc:.3f}")
        print(f"  Limit: {xm['interpretation_limit']}")
    elif (
        "cross_mode_transfer" in results
        and not results["cross_mode_transfer"].get("skipped")
    ):
        print("\nRQ1: Cross-mode transfer unavailable")
        print(f"  {results['cross_mode_transfer'].get('error', 'unknown reason')}")

    # RQ2: Actual behavior vs counterpart-perception labels
    if results.get("implicit_encoding", {}).get("available") is True:
        ie = results["implicit_encoding"]
        print(f"\nRQ2: Label-Source Representation")
        print("  Q: How do actual behavior and counterpart-perception decodability differ?")
        print(f"  GM AUC: {ie['gm_auc']:.3f}, Counterpart-belief AUC: {ie['agent_auc']:.3f}, Gap: {ie['auc_gap']:.3f}")
        print(f"  → {ie['interpretation']}")

    # RQ-MA1: Temporal trajectory
    if "temporal_trajectory" in results and not results["temporal_trajectory"].get("skipped"):
        tt = results["temporal_trajectory"]
        if "error" not in tt and tt.get("trajectory_slope") is not None:
            print(f"\nRQ-MA1: Temporal Emergence")
            print("  Q: What is the descriptive per-round AUC slope?")
            print(f"  Slope: {tt['trajectory_slope']:.3f}")
            print("  Limit: this does not establish a temporal mechanism.")

    # RQ-MA2: Dyadic pairs
    if "dyadic_pairs" in results and not results["dyadic_pairs"].get("skipped"):
        dp = results["dyadic_pairs"]
        if dp.get("available") is True:
            print(f"\nRQ-MA2: Dyadic Signatures")
            print("  Q: Are counterpart-pair roles rank-separable?")
            print(f"  Pair AUC: {dp['pair_probe_auc']:.3f}, D-prime: {dp['d_prime']:.3f}")
            print(f"  Limit: {dp['interpretation_limit']}")

    # RQ-MA3: Outcome prediction
    if "outcome_prediction" in results and not results["outcome_prediction"].get("skipped"):
        op = results["outcome_prediction"]
        if "error" not in op and op.get("early_rounds_auc") is not None:
            print(f"\nRQ-MA3: Outcome Prediction")
            print("  Q: What is the early held-out outcome-ranking AUC?")
            print(f"  Early-round AUC: {op['early_rounds_auc']:.3f}")
            print(f"  Limit: {op['interpretation_limit']}")

    print(f"\n{'='*60}")
    print("END OF ANALYSIS")
    print(f"{'='*60}")

    return results


def plot_results(results: Dict, output_path: str = None):
    """Generate visualization of results."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Layer comparison
    ax1 = axes[0, 0]
    layers = sorted(results["layer_analysis"].keys())
    gm_r2s = [results["layer_analysis"][l]["gm"]["r2_score"] for l in layers]
    agent_r2s = [
        results["layer_analysis"][layer]["agent"].get("r2_score", np.nan)
        for layer in layers
    ]
    gm_aucs = [results["layer_analysis"][l]["gm"]["auc"] for l in layers]

    x = np.arange(len(layers))
    width = 0.25

    ax1.bar(x - width, gm_r2s, width, label='GM R²', color='tab:blue')
    ax1.bar(x, agent_r2s, width, label='Agent R²', color='tab:orange')
    ax1.bar(x + width, gm_aucs, width, label='GM AUC', color='tab:green', alpha=0.7)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Score')
    ax1.set_title('Probe Performance by Layer')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

    # 2. GM vs Agent comparison
    ax2 = axes[0, 1]
    comparison = results["gm_vs_agent"]
    if comparison.get("available") is True:
        methods = ['Ridge\n(GM)', 'Ridge\n(Agent)', 'Mass-Mean\n(GM)']
        r2_values = [
            comparison["gm_ridge_r2"], comparison["agent_ridge_r2"],
            comparison["gm_mass_mean_r2"]
            if comparison["gm_mass_mean_r2"] is not None else np.nan,
        ]
        auc_values = [
            comparison["gm_auc"], comparison["agent_auc"],
            comparison.get("gm_mass_mean_auc")
            if comparison.get("gm_mass_mean_auc") is not None else np.nan,
        ]

        x = np.arange(len(methods))
        width = 0.35

        ax2.bar(x - width/2, r2_values, width, label='R²', color='tab:blue')
        ax2.bar(x + width/2, auc_values, width, label='AUC', color='tab:green')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods)
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    else:
        ax2.text(
            0.5,
            0.5,
            f"Unavailable\n{comparison.get('reason', 'unknown reason')}",
            ha='center',
            va='center',
            wrap=True,
        )
    ax2.set_title('Complete-Case GM vs Counterpart Labels')

    # 3. Generalization with both R² and AUC
    ax3 = axes[1, 0]
    if "generalization" in results and "by_scenario" in results["generalization"]:
        gen = results["generalization"]["by_scenario"]
        scenarios = list(gen.keys())
        r2s = [gen[s]["test_r2"] if gen[s]["test_r2"] is not None else 0 for s in scenarios]
        aucs = [gen[s]["test_auc"] if gen[s]["test_auc"] is not None else 0.5 for s in scenarios]
        rates = [gen[s]["deception_rate"] * 100 for s in scenarios]

        x = np.arange(len(scenarios))
        width = 0.35

        bars1 = ax3.bar(x - width/2, r2s, width, label='R²', color='tab:purple', alpha=0.8)
        bars2 = ax3.bar(x + width/2, aucs, width, label='AUC', color='tab:cyan', alpha=0.8)

        # Add deception rates as text
        for i, (scenario, rate) in enumerate(zip(scenarios, rates)):
            ax3.text(i, max(r2s[i], aucs[i]) + 0.05, f'{rate:.0f}%', ha='center', fontsize=8)

        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='R²=0')
        ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='AUC=0.5 (chance)')

        ax3.set_ylabel('Score')
        ax3.set_title('Cross-Scenario Generalization\n(labels show deception rate)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(scenarios, rotation=45, ha='right')
        ax3.legend(loc='lower right')
    else:
        ax3.text(0.5, 0.5, 'Not enough scenarios', ha='center', va='center')
        ax3.set_title('Generalization')

    # 4. Deception rates by scenario
    ax4 = axes[1, 1]
    if "deception_rates" in results:
        rates = results["deception_rates"]
        scenarios = sorted(rates.keys(), key=lambda x: rates[x])
        values = [rates[s] * 100 for s in scenarios]

        colors = plt.get_cmap("RdYlGn_r")(np.linspace(0.2, 0.8, len(scenarios)))
        bars = ax4.barh(scenarios, values, color=colors)

        ax4.set_xlabel('Deception Rate (%)')
        ax4.set_title('Deception Rate by Scenario')
        ax4.set_xlim(0, 100)

        for bar, val in zip(bars, values):
            ax4.text(val + 2, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}%', va='center', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")

    plt.show()

    return fig


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train probes on captured activations (v2)")

    parser.add_argument("--data", type=str,
                        help=(
                            "Path to a safe activation-dataset JSON manifest; "
                            "reviewed legacy .pt requires --trust-legacy-pt"
                        ))
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save plots")
    parser.add_argument(
        "--trust-legacy-pt",
        action="store_true",
        help="Allow pickle-capable .pt loading only for a reviewed artifact",
    )
    parser.add_argument("--compare", nargs=2, metavar=('9B_RESULTS', '27B_RESULTS'),
                        help="Compare two result files (9B vs 27B)")

    args = parser.parse_args()

    # Model comparison mode
    if args.compare:
        comparison = compare_model_scales(args.compare[0], args.compare[1])
        print_comparison(comparison)

        # Save comparison
        output_path = "model_comparison.json"
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to: {output_path}")
        return

    # Regular analysis mode
    if not args.data:
        parser.error("--data is required for analysis mode")

    # Run analysis
    results = run_full_analysis(
        args.data,
        trusted_legacy=args.trust_legacy_pt,
    )

    # Save results
    if args.output:
        output_path = args.output
    else:
        data_path = Path(args.data)
        output_path = data_path.parent / "probe_results_v2.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate plots
    if args.plot:
        plot_path = Path(output_path).with_suffix(".png")
        plot_results(results, str(plot_path))


if __name__ == "__main__":
    main()
