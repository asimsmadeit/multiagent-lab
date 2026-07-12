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
            probe_steps.append(('pca', PCA(n_components=n_comp)))
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
        cv_steps.append(('pca', PCA(n_components=min(n_components, max_pca, X.shape[1]))))
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
                steps.append(('pca', PCA(n_components=n_comp)))
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
                n_components=min(n_components, max_pca, X.shape[1])
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
    """Train probes across multiple random seeds and aggregate results.

    This provides robust performance estimates with confidence intervals,
    following best practices from RAPTOR (2026) and Truth is Universal (NeurIPS 2024).

    Args:
        X: Feature matrix [N, d_model]
        y: Labels [N]
        probe_type: "logistic", "ridge", or "mass_mean"
        seeds: List of random seeds (default: [42, 123, 456, 789, 1024])
        **kwargs: Additional arguments passed to the probe training function

    Returns:
        Dict with per-seed results, mean/std metrics, and 95% CI
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

    # 95% CI (using t-distribution for small n)
    from scipy import stats
    n = len(aucs)
    auc_mean = np.mean(aucs)
    auc_std = np.std(aucs, ddof=1) if n > 1 else 0.0
    if n > 1:
        t_val = stats.t.ppf(0.975, df=n - 1)
        auc_ci = (auc_mean - t_val * auc_std / np.sqrt(n),
                  auc_mean + t_val * auc_std / np.sqrt(n))
    else:
        auc_ci = (auc_mean, auc_mean)

    return {
        "per_seed": per_seed,
        "auc_mean": float(auc_mean),
        "auc_std": float(auc_std),
        "auc_ci_95": (float(auc_ci[0]), float(auc_ci[1])),
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
) -> Dict[str, float]:
    """Sanity check: probes on shuffled labels should give R² near 0."""
    shuffle_r2s = []
    rng = np.random.default_rng(0)

    for seed in range(n_shuffles):
        y_shuffled = rng.permutation(y)
        _, result = train_ridge_probe(
            X, y_shuffled, random_state=seed, groups=groups
        )
        shuffle_r2s.append(result.r2_score)

    return {
        "mean_shuffled_r2": float(np.mean(shuffle_r2s)),
        "std_shuffled_r2": float(np.std(shuffle_r2s)),
        "max_shuffled_r2": float(np.max(shuffle_r2s)),
        "passed": bool(np.mean(shuffle_r2s) < 0.05),
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
        use_pca: Whether to apply PCA
        n_components: PCA components

    Returns:
        Dict mapping layer_num -> {auc, r2, accuracy, train_r2, test_r2}
    """
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
                'train_r2': 0.0, 'test_r2': 0.0, 'note': 'insufficient samples'
            }
            continue

        # Check label variance
        if np.std(y) < 0.01:
            results[layer_key] = {
                'auc': 0.5, 'r2': 0.0, 'accuracy': 0.5,
                'train_r2': 0.0, 'test_r2': 0.0, 'note': 'no label variance'
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
            }
        except Exception as e:
            results[layer_key] = {
                'auc': 0.5, 'r2': 0.0, 'accuracy': 0.5,
                'train_r2': 0.0, 'test_r2': 0.0, 'error': str(e)
            }

    return results


def find_best_layer(layer_results: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
    """
    Find the best layer and analyze the layer accuracy curve.

    Returns:
        Dict with best_layer, peak_auc, curve_analysis, etc.
    """
    if not layer_results:
        return {'best_layer': None, 'error': 'No layer results'}

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
    alpha: float = 100.0,
    random_state: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute cross-scenario generalization using leave-one-scenario-out (LOSO) evaluation.

    This is a CRITICAL validation for deception probes. A probe that generalizes
    across scenarios is detecting deception-general features, not scenario-specific artifacts.

    Methodology (Leave-One-Scenario-Out Cross-Validation):
        For each scenario S:
        1. Train on ALL samples from scenarios OTHER than S
        2. Test on ALL samples from scenario S
        3. Report R² and AUC for this holdout
        4. Repeat for all scenarios

    Interpretation:
        - High cross-scenario AUC (>0.7): Probe likely captures general deception
        - Low cross-scenario AUC (<0.55): Probe may be learning scenario artifacts
        - Large variance: Some scenarios are outliers (different deception types)

    Args:
        X: Feature matrix [N, d_model]
        y: Labels [N]
        scenarios: List of scenario names [N], one per sample
        alpha: Ridge regularization strength
        random_state: Random seed for reproducibility
        verbose: Print per-scenario results

    Returns:
        Dict with:
        - by_scenario: Results for each holdout scenario
        - average_r2: Mean R² across scenarios
        - average_auc: Mean AUC across scenarios (recommended metric)
        - std_r2/std_auc: Standard deviations
    """
    unique_scenarios = list(set(scenarios))
    results = {}

    for holdout in unique_scenarios:
        # Split by scenario
        train_mask = np.array([s != holdout for s in scenarios])
        test_mask = ~train_mask

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        # Skip if test set has only one class
        if len(np.unique((y_test > 0.5).astype(int))) < 2:
            results[holdout] = {
                "train_size": int(train_mask.sum()),
                "test_size": int(test_mask.sum()),
                "test_r2": None,
                "test_auc": None,
                "deception_rate": float(np.mean(y_test)),
                "note": "Single class in test set",
            }
            continue

        # Normalize: fit on train only
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Apply PCA
        n_comp = min(30, X_train_s.shape[0] - 1, X_train_s.shape[1])
        pca = PCA(n_components=n_comp)
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
            "train_size": int(train_mask.sum()),
            "test_size": int(test_mask.sum()),
            "test_r2": float(test_r2),
            "test_auc": float(test_auc),
            "deception_rate": float(np.mean(y_test)),
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

    # GM vs Agent gap
    gap_9b = r9b["gm_vs_agent"]["gm_ridge_r2"] - r9b["gm_vs_agent"]["agent_ridge_r2"]
    gap_27b = r27b["gm_vs_agent"]["gm_ridge_r2"] - r27b["gm_vs_agent"]["agent_ridge_r2"]
    comparison["9b"]["gm_agent_gap"] = gap_9b
    comparison["27b"]["gm_agent_gap"] = gap_27b
    comparison["deltas"]["gm_agent_gap"] = gap_27b - gap_9b

    # GM AUC
    comparison["9b"]["gm_auc"] = r9b["gm_vs_agent"]["gm_auc"]
    comparison["27b"]["gm_auc"] = r27b["gm_vs_agent"]["gm_auc"]
    comparison["deltas"]["gm_auc"] = r27b["gm_vs_agent"]["gm_auc"] - r9b["gm_vs_agent"]["gm_auc"]

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

    if avg_r2 < 0.1:
        return "No clear temporal pattern (low R²)"

    if avg_slope > 0.05:
        return f"Deception encoding INCREASES over rounds (slope={avg_slope:.3f})"
    elif avg_slope < -0.05:
        return f"Deception encoding DECREASES over rounds (slope={avg_slope:.3f})"
    else:
        return f"Stable deception encoding across rounds (slope={avg_slope:.3f})"


def compute_cross_mode_transfer(
    X: np.ndarray,
    y: np.ndarray,
    mode_labels: List[str],
    alpha: float = 100.0,
    use_pca: bool = True,
    n_components: int = 30,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """RQ1: Compute bidirectional cross-mode transfer between instructed and emergent.

    Tests whether deception representations transfer between:
    - Forward: instructed → emergent (explicit to incentive-based)
    - Reverse: emergent → instructed (incentive-based to explicit)

    Symmetric transfer suggests shared underlying representation.
    Asymmetric transfer suggests one mode's representation is more general.

    Interpretation:
    - High transfer (AUC > 0.65): Same underlying representation
    - Low transfer (AUC ~ 0.5): Different mechanisms
    - Asymmetry > 0.10: One direction transfers significantly better

    Args:
        X: Feature matrix [N, d_model]
        y: Labels [N]
        mode_labels: "emergent" or "instructed" for each sample [N]
        alpha: Ridge regularization
        use_pca: Whether to apply PCA
        n_components: PCA components

    Returns:
        Dict with forward_transfer_auc, reverse_transfer_auc, transfer_asymmetry,
        and interpretation including asymmetry analysis
    """
    mode_labels = np.array(mode_labels)
    if len(mode_labels) != len(y) or len(X) != len(y):
        raise ValueError("X, y, and mode_labels must be sample-aligned")
    if groups is not None and len(groups) != len(y):
        raise ValueError("groups must have one value per sample")

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

    # === 2026-04-21 Finding 7 fix: unify label scale before training ===
    # emergent samples receive binary {0, 1} labels from regex-based
    # emergent_ground_truth; instructed samples receive continuous GM
    # severity scores. The previous implementation trained a ridge probe on
    # continuous instructed labels and evaluated its predictions against
    # binary emergent labels (and vice versa), so the "cross-mode transfer
    # fails" null result was confounded with a label-scale mismatch, not a
    # pure representational distinction. Binarize both y vectors at the 0.5
    # threshold before training so transfer AUC is an apples-to-apples
    # comparison. The continuous R² becomes uninformative after binarization
    # (R² on binary targets is not meaningful) so we report transfer in AUC
    # only, with the continuous R² preserved in a separate _continuous field
    # for backward compatibility.
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
        pca = PCA(n_components=n_comp)
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

    # Also compute the old (label-scale-mismatched) R² for comparison with
    # pre-2026-04-21 results. If this is much worse than the binarized R²,
    # the old "cross-mode transfer fails" null was at least partly a
    # label-scale artifact.
    probe_legacy = Ridge(alpha=alpha)
    probe_legacy.fit(X_instructed_pca, y_instructed)
    emergent_pred_legacy = probe_legacy.predict(X_emergent_pca)
    transfer_r2_legacy_mixed = r2_score(y_emergent, emergent_pred_legacy)

    # Compute AUC for forward transfer (instructed → emergent)
    try:
        forward_transfer_auc = roc_auc_score(y_emergent_binary.astype(int), emergent_pred)
    except ValueError:
        forward_transfer_auc = 0.5

    # === REVERSE TRANSFER: Train on emergent → Test on instructed ===
    # This tests if the relationship is symmetric
    if use_pca:
        n_comp_rev = min(n_components, X_emergent.shape[0] - 1, X_emergent.shape[1])
        pca_rev = PCA(n_components=n_comp_rev)
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

    # Legacy (mismatched-scale) reverse for back-compat reporting
    probe_reverse_legacy = Ridge(alpha=alpha)
    probe_reverse_legacy.fit(X_emergent_pca_rev, y_emergent)
    instructed_pred_legacy = probe_reverse_legacy.predict(X_instructed_pca_rev)
    reverse_transfer_r2_legacy_mixed = r2_score(y_instructed, instructed_pred_legacy)

    # Compute AUC for reverse transfer (using unified binary labels)
    try:
        reverse_transfer_auc = roc_auc_score(y_instructed_binary.astype(int), instructed_pred)
    except ValueError:
        reverse_transfer_auc = 0.5

    # Also compute within-mode performance for comparison
    # Guard: within-mode datasets may be too small or imbalanced for train/test split
    try:
        instructed_groups = np.asarray(groups)[instructed_mask] if groups is not None else None
        _, instructed_result = train_ridge_probe(
            X_instructed, y_instructed, alpha=alpha, groups=instructed_groups
        )
    except ValueError:
        instructed_result = None
    try:
        emergent_groups = np.asarray(groups)[emergent_mask] if groups is not None else None
        _, emergent_result = train_ridge_probe(
            X_emergent, y_emergent, alpha=alpha, groups=emergent_groups
        )
    except ValueError:
        emergent_result = None

    # Compute asymmetry
    transfer_asymmetry = abs(forward_transfer_auc - reverse_transfer_auc)

    # Interpretation based on forward transfer (primary metric)
    if forward_transfer_auc > 0.70:
        interpretation = "STRONG transfer - instructed and emergent share deception representation"
    elif forward_transfer_auc > 0.60:
        interpretation = "MODERATE transfer - partial overlap in representations"
    elif forward_transfer_auc > 0.55:
        interpretation = "WEAK transfer - some shared features but largely different"
    else:
        interpretation = "NO transfer - distinct mechanisms for instructed vs emergent deception"

    # Add asymmetry note if significant
    if transfer_asymmetry > 0.10:
        if forward_transfer_auc > reverse_transfer_auc:
            interpretation += " (ASYMMETRIC: instructed→emergent transfers better)"
        else:
            interpretation += " (ASYMMETRIC: emergent→instructed transfers better)"

    return {
        "n_instructed": int(n_instructed),
        "n_emergent": int(n_emergent),
        # Forward transfer (instructed → emergent); both sides binarized
        "forward_transfer_r2": float(transfer_r2),
        "forward_transfer_auc": float(forward_transfer_auc),
        # Reverse transfer (emergent → instructed); both sides binarized
        "reverse_transfer_r2": float(reverse_transfer_r2),
        "reverse_transfer_auc": float(reverse_transfer_auc),
        # === 2026-04-21 Finding 7 diagnostic fields ===
        # Legacy mixed-scale R²: continuous instructed labels in train,
        # binary emergent labels in test (and vice versa). Included so the
        # delta between unified and legacy is visible in the report.
        "forward_transfer_r2_legacy_mixed": float(transfer_r2_legacy_mixed),
        "reverse_transfer_r2_legacy_mixed": float(reverse_transfer_r2_legacy_mixed),
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
        "interpretation": interpretation,
    }


def analyze_implicit_encoding(
    X: np.ndarray,
    gm_labels: np.ndarray,
    agent_labels: np.ndarray,
    alpha: float = 100.0,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compare actual-deception and perceived-counterpart label predictability.

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
    # Train probes on each label type
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

    # Interpretation
    if auc_gap > 0.10:
        interpretation = "STRONG label-source divergence - actual behavior is more decodable"
    elif auc_gap > 0.05:
        interpretation = "MODERATE label-source divergence - actual behavior is more decodable"
    elif auc_gap > -0.05:
        interpretation = "SIMILAR decodability across actual and counterpart-perception labels"
    else:
        interpretation = "Counterpart-perception labels are more decodable"

    return {
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
        "interpretation": interpretation,
    }


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
    if groups is not None and len(groups) != len(y):
        raise ValueError("groups must have one value per sample")
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
                "auc": 0.5, "r2": 0.0, "n_samples": int(mask.sum()),
                "deception_rate": float(np.mean(y_r)) if len(y_r) > 0 else 0.0,
                "note": "insufficient samples or variance"
            }
            continue

        round_groups = np.asarray(groups)[mask] if groups is not None else None
        _, result = train_ridge_probe(
            X_r, y_r, alpha=alpha, groups=round_groups
        )
        per_round_results[r_key] = {
            "auc": float(result.auc),
            "r2": float(result.r2_score),
            "n_samples": int(mask.sum()),
            "deception_rate": float(np.mean(y_r)),
        }

    # Fit trajectory (linear regression of AUC over rounds)
    # Filter out rounds with invalid AUC (0.5 default or NaN)
    valid_rounds = [r for r in unique_rounds
                    if per_round_results[r].get("auc", 0.5) != 0.5
                    and not np.isnan(per_round_results[r].get("auc", 0.5))]
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
        slope = 0.0
        trajectory_r2 = 0.0
        interpretation = "Insufficient valid rounds for trajectory fit"

    return {
        "per_round": per_round_results,
        "n_rounds": len(unique_rounds),
        "trajectory_slope": slope,
        "trajectory_r2": trajectory_r2,
        "interpretation": interpretation,
        "analysis_layer": int(mid_layer),
    }


def analyze_dyadic_pairs(
    X: np.ndarray,
    y: np.ndarray,
    counterpart_idxs: List[int],
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
        alpha: Ridge regularization

    Returns:
        Dict with dyadic analysis metrics
    """
    n_samples = len(y)
    if len(X) != n_samples:
        raise ValueError("X and y must contain the same number of samples")
    if len(counterpart_idxs) != n_samples:
        raise ValueError("counterpart_idxs must have one value per sample")

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

    for i, j in valid_pairs:
        # Higher deception label = deceiver
        if y[i] > y[j]:
            deceiver_activations.append(X[i])
            victim_activations.append(X[j])
        else:
            deceiver_activations.append(X[j])
            victim_activations.append(X[i])
        asymmetries.append(abs(y[i] - y[j]))

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

    # Train probe to distinguish deceiver from victim.
    #
    # 2026-04-22 fix: use a pair-aware train/test split. deceiver_X[k] and
    # victim_X[k] come from the same dyadic round (same negotiation trial).
    # A random split stacks them and can put deceiver[k] in train and
    # victim[k] in test — so the probe memorizes per-trial activation
    # signatures rather than learning a general deceiver/victim direction.
    # That is what produced the near-1.000 AUCs in the pre-audit results.
    # Group by pair index so both members of a pair are always on the same
    # side of the split.
    pair_X = np.vstack([deceiver_X, victim_X])
    pair_y = np.array([1.0] * len(deceiver_X) + [0.0] * len(victim_X))
    pair_groups = np.concatenate([np.arange(len(deceiver_X)),
                                  np.arange(len(victim_X))])

    pair_auc, pair_r2 = _train_probe_grouped(
        pair_X, pair_y, pair_groups, alpha=alpha
    )
    if pair_auc is None:
        pair_auc = 0.5
    if pair_r2 is None:
        pair_r2 = 0.0

    # Also compute the old (leaky) number so the report surfaces how much
    # of the historic AUC was per-trial memorization vs a real dyadic
    # direction. Keep it under a _legacy_ field for diagnostic only.
    _, pair_result_legacy = train_ridge_probe(pair_X, pair_y, alpha=alpha)

    # Interpretation based on the (post-fix) pair-aware AUC
    if pair_auc > 0.70:
        interpretation = "STRONG dyadic signal - deceiver/victim clearly distinguishable"
    elif pair_auc > 0.60:
        interpretation = "MODERATE dyadic signal - partial separability"
    else:
        interpretation = "WEAK dyadic signal - deceiver/victim activations similar"

    return {
        "n_pairs": len(valid_pairs),
        "pair_probe_auc": float(pair_auc),
        "pair_probe_r2": float(pair_r2),
        "pair_probe_auc_legacy_leaky": float(pair_result_legacy.auc),
        "pair_probe_r2_legacy_leaky": float(pair_result_legacy.r2_score),
        "split_method": "group_aware_by_pair_id",
        "d_prime": float(d_prime),
        "mean_asymmetry": float(np.mean(asymmetries)),
        "interpretation": interpretation,
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
        pca = PCA(n_components=n_comp)
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
    trial_ids: List[str],
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
        trial_ids: Trial ID for each sample [N]
        trial_outcomes: Success/failure for each sample [N]
        alpha: Ridge regularization

    Returns:
        Dict with outcome prediction metrics
    """
    if not (
        len(X) == len(y) == len(round_nums) == len(trial_ids) == len(trial_outcomes)
    ):
        raise ValueError(
            "X, y, round_nums, trial_ids, and trial_outcomes must be sample-aligned"
        )

    round_nums = np.array(round_nums)
    trial_ids_arr = np.array([str(t) for t in trial_ids])
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

    # Filter to early rounds (1-2)
    early_mask = (round_nums >= 0) & (round_nums <= 2) & known_outcome_mask
    n_early = early_mask.sum()

    if n_early < 20:
        return {
            "error": f"Insufficient early-round samples ({n_early})",
            "n_early_samples": int(n_early),
        }

    X_early = X[early_mask]
    outcomes_early = trial_outcomes[early_mask]
    trials_early = trial_ids_arr[early_mask]

    # Check outcome variance
    if np.std(outcomes_early) < 0.01:
        return {
            "error": "No outcome variance in early rounds",
            "n_early_samples": int(n_early),
            "outcome_rate": float(np.mean(outcomes_early)),
        }

    # Group-aware split by trial_id: trial_outcome is a trial-level label
    # backfilled onto every per-round sample. A random per-sample split
    # therefore places the same trial (same label) in both train and test,
    # inflating AUC. Split on trial_id instead.
    early_auc, early_r2 = _train_probe_grouped(
        X_early, outcomes_early, trials_early, alpha=alpha
    )

    # Also try late rounds for comparison
    valid_round_nums = round_nums[known_outcome_mask & (round_nums >= 0)]
    late_mask = (
        known_outcome_mask
        & (round_nums >= 0)
        & (round_nums >= valid_round_nums.max() - 1)
    ) if len(valid_round_nums) else np.zeros(len(round_nums), dtype=bool)
    n_late = late_mask.sum()
    trials_late = trial_ids_arr[late_mask] if n_late else np.array([])
    if (
        n_late >= 20
        and np.std(trial_outcomes[late_mask]) > 0.01
        and len(np.unique(trials_late)) >= 2
    ):
        late_auc, _ = _train_probe_grouped(
            X[late_mask], trial_outcomes[late_mask], trials_late, alpha=alpha
        )
    else:
        late_auc = None

    # Interpretation
    if early_auc is not None and early_auc > 0.70:
        interpretation = "STRONG early prediction - deception outcome predictable from round 1-2"
    elif early_auc is not None and early_auc > 0.60:
        interpretation = "MODERATE early prediction - partial signal in early rounds"
    else:
        interpretation = "WEAK early prediction - outcome not predictable from early activations"

    return {
        "n_early_samples": int(n_early),
        "early_rounds_auc": float(early_auc) if early_auc is not None else None,
        "early_rounds_r2": float(early_r2) if early_r2 is not None else None,
        "late_rounds_auc": float(late_auc) if late_auc is not None else None,
        "outcome_rate": float(np.mean(outcomes_early)),
        "interpretation": interpretation,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def _sample_keep_mask(
    metadata: List[Dict[str, Any]],
    n_samples: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build the negotiation/QC mask for an aligned serialized dataset."""
    if metadata and len(metadata) != n_samples:
        raise ValueError(
            f"metadata has {len(metadata)} rows, expected {n_samples}"
        )
    if not metadata:
        return np.ones(n_samples, dtype=bool), {
            "probe_rounds": 0,
            "qc_failures": 0,
        }

    keep = np.ones(n_samples, dtype=bool)
    probe_rounds = 0
    qc_failures = 0
    for i, row in enumerate(metadata):
        sample_type = row.get('sample_type')
        round_num = row.get('round_num')
        is_negotiation = (
            sample_type == 'negotiation'
            if sample_type is not None
            else round_num is None or round_num >= 0
        )
        if not is_negotiation:
            keep[i] = False
            probe_rounds += 1
            continue

        flags = row.get('qc_flags')
        if flags is None and row.get('full_response') is not None:
            from interpretability.core.qc_filter import classify_response
            flags = classify_response(row['full_response'])
        if flags:
            keep[i] = False
            qc_failures += 1

    return keep, {
        "probe_rounds": probe_rounds,
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

def run_full_analysis(data_path: str) -> Dict[str, Any]:
    """Run complete probe training and analysis."""

    print(f"\n{'='*60}")
    print("MATS PROBE TRAINING AND ANALYSIS v2")
    print(f"{'='*60}")

    # Load data
    # NOTE: weights_only=False needed for dicts with mixed types
    # SECURITY: Only load files YOU generated - pickle can execute arbitrary code
    print(f"\nLoading data from: {data_path}")
    data = torch.load(data_path, weights_only=False)

    activations = data["activations"]
    labels = data["labels"]
    config = data.get("config", {})
    metadata = data.get("metadata", [])

    gm_labels = np.array(labels["gm_labels"])
    agent_labels = np.array(labels["agent_labels"])
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

    keep_mask, filter_counts = _sample_keep_mask(metadata, n_total)
    n_dropped = int((~keep_mask).sum())
    original_counterparts = labels.get('counterpart_idxs')
    if n_dropped > 0:
        print(
            f"  Filtering out {filter_counts['probe_rounds']} probe-round and "
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

    trial_ids_for_split = labels.get('trial_ids', [])
    pod_ids_for_split = labels.get('pod_ids', [])
    analysis_groups = None
    split_unit = "row (legacy dataset without trial_ids)"
    if len(trial_ids_for_split) == len(gm_labels):
        if len(pod_ids_for_split) == len(gm_labels):
            analysis_groups = np.array([
                f"{pod}:{trial}" for pod, trial in zip(
                    pod_ids_for_split, trial_ids_for_split
                )
            ])
            split_unit = "pod_id+trial_id"
        else:
            analysis_groups = np.asarray(trial_ids_for_split).astype(str)
            split_unit = "trial_id"

    results = {
        "sanity_checks": {},
        "layer_analysis": {},
        "gm_vs_agent": {},
        "generalization": {},
        "deception_rates": {},
        "best_probe": None,
        "filtering": filter_counts,
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

    layer_results = {}
    best_layer = None
    best_auc = -1.0

    for layer in layers:
        X = activations[layer].float().numpy()

        # Train probe on GM labels
        _, gm_result = train_ridge_probe(
            X, gm_labels, groups=analysis_groups
        )
        gm_result.layer = layer
        gm_result.label_type = "gm"

        # Train probe on agent labels
        _, agent_result = train_ridge_probe(
            X, agent_labels, groups=analysis_groups
        )
        agent_result.layer = layer
        agent_result.label_type = "agent"

        layer_results[layer] = {
            "gm": gm_result.to_dict(),
            "agent": agent_result.to_dict(),
        }

        print(f"\nLayer {layer}:")
        print(f"  GM labels    - R²: {gm_result.r2_score:.3f}, AUC: {gm_result.auc:.3f}")
        print(f"  Agent labels - R²: {agent_result.r2_score:.3f}, AUC: {agent_result.auc:.3f}")

        # Select best layer by AUC (the primary metric the rest of the pipeline
        # uses), not R². R² can be negative and noisy on small samples, so
        # AUC-based selection is more stable and consistent with downstream use.
        if gm_result.auc > best_auc:
            best_auc = gm_result.auc
            best_layer = layer

    results["layer_analysis"] = layer_results
    if best_layer is None:
        best_layer = layers[len(layers) // 2]
    best_r2 = layer_results[best_layer]["gm"]["r2_score"] if best_layer in layer_results else 0.0
    results["best_probe"] = {
        "layer": int(best_layer),
        "auc": float(best_auc),
        "r2": float(best_r2),
    }

    print(f"\nBest layer: {best_layer} (AUC = {best_auc:.3f}, R² = {best_r2:.3f})")

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

    _, gm_result = train_ridge_probe(
        X_best, gm_labels, groups=analysis_groups
    )
    _, agent_result = train_ridge_probe(
        X_best, agent_labels, groups=analysis_groups
    )

    # Mass-mean probe
    _, gm_mm_result = train_mass_mean_probe(
        X_best, gm_labels, groups=analysis_groups
    )

    results["gm_vs_agent"] = {
        "gm_ridge_r2": float(gm_result.r2_score),
        "agent_ridge_r2": float(agent_result.r2_score),
        "gm_mass_mean_r2": float(gm_mm_result.r2_score),
        "gm_auc": float(gm_result.auc),
        "agent_auc": float(agent_result.auc),
        "gm_mass_mean_auc": float(gm_mm_result.auc),
        "gm_wins": bool(gm_result.r2_score > agent_result.r2_score),
        "agent_label_semantics": "perceived counterpart deception, not self-report",
        "supports_self_awareness_claim": False,
    }

    print(f"\nGM (Ground Truth) - What the agent IS doing:")
    print(f"  Ridge R²:     {gm_result.r2_score:.3f}")
    print(f"  Ridge AUC:    {gm_result.auc:.3f}")
    print(f"  Mass-Mean R²: {gm_mm_result.r2_score:.3f}")
    print(f"  Mass-Mean AUC:{gm_mm_result.auc:.3f}")

    print(f"\nAgent ToM - perceived COUNTERPART deception:")
    print(f"  Ridge R²:     {agent_result.r2_score:.3f}")
    print(f"  Ridge AUC:    {agent_result.auc:.3f}")

    if results["gm_vs_agent"]["gm_wins"]:
        print(f"\n>> Actual-deception labels are more predictable than counterpart-belief labels.")
    else:
        print(f"\n>> Counterpart-belief labels are equally or more predictable.")

    # ==========================================================================
    # GENERALIZATION WITH AUC
    # ==========================================================================
    print(f"\n{'='*60}")
    print("GENERALIZATION ANALYSIS (with AUC)")
    print(f"{'='*60}")

    unique_scenarios = list(set(scenarios))

    if len(unique_scenarios) >= 3:
        gen_results = compute_generalization_auc(X_best, gm_labels, scenarios)
        results["generalization"] = gen_results

        for holdout, res in gen_results["by_scenario"].items():
            r2_str = f"{res['test_r2']:.3f}" if res['test_r2'] is not None else "N/A"
            auc_str = f"{res['test_auc']:.3f}" if res['test_auc'] is not None else "N/A"
            rate_str = f"{res['deception_rate']*100:.0f}%"
            print(f"\nHoldout: {holdout} (deception rate: {rate_str})")
            print(f"  R²:  {r2_str}")
            print(f"  AUC: {auc_str}")

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
        print("Question: Are instructed and emergent deception structurally")
        print("          different in model activations?")
        print("-" * 60)
        print("Interpretation Guide:")
        print("  Transfer AUC ~0.50: DIFFERENT circuits (NOVEL finding)")
        print("  Transfer AUC >0.65: SAME representation")
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
            print(f"\n*** KEY RESULT: Bidirectional Cross-mode Transfer ***")
            print(f"  Forward (instructed → emergent):")
            print(f"    AUC: {cross_mode_results['forward_transfer_auc']:.3f}")
            print(f"    R²:  {cross_mode_results['forward_transfer_r2']:.3f}")
            print(f"  Reverse (emergent → instructed):")
            print(f"    AUC: {cross_mode_results['reverse_transfer_auc']:.3f}")
            print(f"    R²:  {cross_mode_results['reverse_transfer_r2']:.3f}")
            print(f"  Asymmetry: {cross_mode_results['transfer_asymmetry']:.3f}")
            print(f"\n>> ANSWER: {cross_mode_results['interpretation']}")
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

    implicit_results = analyze_implicit_encoding(
        X_best, gm_labels, agent_labels, groups=analysis_groups
    )
    results["implicit_encoding"] = implicit_results

    print(f"\n*** KEY RESULT: GM vs Agent Predictability ***")
    print(f"  GM (Ground Truth) AUC:    {implicit_results['gm_auc']:.3f}")
    print(f"  Counterpart-belief AUC:   {implicit_results['agent_auc']:.3f}")
    print(f"  AUC Gap (GM - Agent):     {implicit_results['auc_gap']:.3f}")
    print(f"\n>> ANSWER: {implicit_results['interpretation']}")

    # ==========================================================================
    # RQ-MA1: TEMPORAL TRAJECTORY ANALYSIS
    # ==========================================================================
    round_nums = labels.get("round_nums", [])
    if round_nums and len(set(round_nums)) >= 3:
        print(f"\n{'='*60}")
        print("RQ-MA1: TEMPORAL TRAJECTORY ANALYSIS")
        print(f"{'='*60}")
        print("Question: Does deception appear suddenly or build up over rounds?")
        print("          (Unique to multi-turn - single-prompt CANNOT see this)")
        print("-" * 60)
        print("Interpretation Guide:")
        print("  Slope > 0.05:  Deception BUILDS over rounds")
        print("  Slope < -0.05: Deception DECREASES over rounds")
        print("  |Slope| < 0.05: STABLE encoding across rounds")
        print("  Look for: Which round shows peak deception?")
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
                auc_str = f"{res['auc']:.3f}" if res.get('auc', 0.5) != 0.5 else "N/A"
                print(f"  Round {r}: AUC={auc_str}, n={res['n_samples']}, deception_rate={res['deception_rate']*100:.0f}%")
            print(f"\n*** KEY RESULT: Temporal Pattern ***")
            print(f"  Trajectory slope: {trajectory_results['trajectory_slope']:.3f}")
            print(f"  Trajectory R²: {trajectory_results['trajectory_r2']:.3f}")
            print(f"\n>> ANSWER: {trajectory_results['interpretation']}")
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
        print("          deception? Can we detect 'being deceived' in victim?")
        print("          (Unique to multi-agent - NO ONE has studied this)")
        print("-" * 60)
        print("Interpretation Guide:")
        print("  Victim AUC > 0.55: Can detect 'being deceived' (NOVEL)")
        print("  D-prime > 1.0: Strong deceiver/victim separation")
        print("  High asymmetry: One agent dominates deception")
        print("-" * 60)

        dyadic_results = analyze_dyadic_pairs(X_best, gm_labels, counterpart_idxs)
        results["dyadic_pairs"] = dyadic_results

        if "error" not in dyadic_results:
            print(f"\nValid pairs: {dyadic_results['n_pairs']}")
            print(f"\n*** KEY RESULT: Dyadic Signatures ***")
            print(f"  Deceiver/Victim probe AUC: {dyadic_results['pair_probe_auc']:.3f}")
            print(f"  D-prime (separability):    {dyadic_results['d_prime']:.3f}")
            print(f"  Mean asymmetry:            {dyadic_results['mean_asymmetry']:.3f}")
            print(f"\n>> ANSWER: {dyadic_results['interpretation']}")
        else:
            print(f"\n{dyadic_results['error']}")
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
        print("          success/failure? Does early deception predict failure?")
        print("-" * 60)
        print("Interpretation Guide:")
        print("  Early AUC > 0.60: Round 1-2 activations predict outcome")
        print("  Early > Late AUC: Outcome determined early (intervention point)")
        print("  Links internal representations to behavioral outcomes")
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
            print(f"\n*** KEY RESULT: Outcome Prediction ***")
            print(f"  Early-round prediction AUC: {outcome_results['early_rounds_auc']:.3f}")
            if outcome_results['late_rounds_auc'] is not None:
                print(f"  Late-round prediction AUC:  {outcome_results['late_rounds_auc']:.3f}")
            print(f"\n>> ANSWER: {outcome_results['interpretation']}")
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
    print(f"  Best Probe: Layer {best_layer}, R²={best_r2:.3f}, AUC={results['gm_vs_agent']['gm_auc']:.3f}")

    if results["generalization"].get("average_auc"):
        print(f"  Cross-scenario: R²={results['generalization']['average_r2']:.3f}, AUC={results['generalization']['average_auc']:.3f}")

    print(f"\n" + "-" * 60)
    print("[RESEARCH FINDINGS]")
    print("-" * 60)

    # RQ1: Cross-mode transfer
    if "cross_mode_transfer" in results and not results["cross_mode_transfer"].get("skipped"):
        xm = results["cross_mode_transfer"]
        print(f"\nRQ1: Instructed vs Emergent Deception")
        print(f"  Q: Are they structurally different?")
        transfer_values = [
            value for value in (
                xm.get('forward_transfer_auc'),
                xm.get('reverse_transfer_auc'),
            ) if value is not None
        ]
        transfer_auc = float(np.mean(transfer_values)) if transfer_values else 0.5
        print(f"  Mean bidirectional transfer AUC: {transfer_auc:.3f}", end="")
        if transfer_auc < 0.55:
            print(" → DIFFERENT circuits (NOVEL)")
        elif transfer_auc > 0.65:
            print(" → SAME representation")
        else:
            print(" → PARTIAL overlap")

    # RQ2: Actual behavior vs counterpart-perception labels
    if "implicit_encoding" in results:
        ie = results["implicit_encoding"]
        print(f"\nRQ2: Label-Source Representation")
        print("  Q: How do actual behavior and counterpart-perception decodability differ?")
        print(f"  GM AUC: {ie['gm_auc']:.3f}, Counterpart-belief AUC: {ie['agent_auc']:.3f}, Gap: {ie['auc_gap']:.3f}")
        print(f"  → {ie['interpretation']}")

    # RQ-MA1: Temporal trajectory
    if "temporal_trajectory" in results and not results["temporal_trajectory"].get("skipped"):
        tt = results["temporal_trajectory"]
        if "error" not in tt:
            print(f"\nRQ-MA1: Temporal Emergence")
            print(f"  Q: Does deception build up or appear suddenly?")
            print(f"  Slope: {tt['trajectory_slope']:.3f}", end="")
            if tt['trajectory_slope'] > 0.05:
                print(" → BUILDS over rounds")
            elif tt['trajectory_slope'] < -0.05:
                print(" → DECREASES over rounds")
            else:
                print(" → STABLE across rounds")

    # RQ-MA2: Dyadic pairs
    if "dyadic_pairs" in results and not results["dyadic_pairs"].get("skipped"):
        dp = results["dyadic_pairs"]
        if "error" not in dp:
            print(f"\nRQ-MA2: Dyadic Signatures")
            print(f"  Q: Can we detect 'being deceived' in victim?")
            print(f"  Pair AUC: {dp['pair_probe_auc']:.3f}, D-prime: {dp['d_prime']:.3f}")
            if dp['pair_probe_auc'] > 0.55:
                print(f"  → YES: Victim signatures detectable (NOVEL)")
            else:
                print(f"  → Signatures not separable")

    # RQ-MA3: Outcome prediction
    if "outcome_prediction" in results and not results["outcome_prediction"].get("skipped"):
        op = results["outcome_prediction"]
        if "error" not in op:
            print(f"\nRQ-MA3: Outcome Prediction")
            print(f"  Q: Can early rounds predict negotiation success?")
            print(f"  Early-round AUC: {op['early_rounds_auc']:.3f}")
            if op['early_rounds_auc'] > 0.60:
                print(f"  → YES: Early intervention possible")
            else:
                print(f"  → Outcome not predictable from early rounds")

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
    agent_r2s = [results["layer_analysis"][l]["agent"]["r2_score"] for l in layers]
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
    methods = ['Ridge\n(GM)', 'Ridge\n(Agent)', 'Mass-Mean\n(GM)']
    r2_values = [comparison["gm_ridge_r2"], comparison["agent_ridge_r2"], comparison["gm_mass_mean_r2"]]
    auc_values = [comparison["gm_auc"], comparison["agent_auc"], comparison.get("gm_mass_mean_auc", 0.5)]

    x = np.arange(len(methods))
    width = 0.35

    ax2.bar(x - width/2, r2_values, width, label='R²', color='tab:blue')
    ax2.bar(x + width/2, auc_values, width, label='AUC', color='tab:green')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel('Score')
    ax2.set_title('GM vs Agent Label Comparison')
    ax2.legend()
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

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
                        help="Path to activations.pt file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save plots")
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
    results = run_full_analysis(args.data)

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
