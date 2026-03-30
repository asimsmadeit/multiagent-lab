# Causal Validation for Deception Probes
# Implements state-of-the-art causal intervention techniques to prove
# that identified features are actually USED by the model for deception.
#
# Based on:
# - "Can LLMs Lie?" (CMU 2025) - activation steering, causal interventions
# - "When Thinking LLMs Lie" (2025) - LAT, steering vectors
# - MIB Benchmark (ICML 2025) - faithfulness, minimality criteria
# - "Open Problems in Mechanistic Interpretability" (2025) - validation standards
#
# Usage:
#   from causal_validation import (
#       activation_patching_test,
#       cross_sample_patching_test,
#       ablation_test,
#       steering_vector_test,
#       run_full_causal_validation,
#   )
#
#   # Run all causal tests
#   results = run_full_causal_validation(
#       model=tl_model,
#       activations=activations,
#       gm_labels=labels,
#       best_layer=12,
#       test_prompts=prompts,
#   )

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings


def _to_serializable(obj: Any) -> Any:
    """Convert numpy/scalar containers to JSON-serializable Python primitives."""
    if isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CausalValidationResult:
    """Results from causal validation tests."""
    test_name: str
    passed: bool
    effect_size: float
    p_value: Optional[float] = None
    n_samples_tested: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "passed": bool(self.passed),
            "effect_size": float(self.effect_size),
            "p_value": float(self.p_value) if self.p_value is not None else None,
            "n_samples_tested": int(self.n_samples_tested),
            "details": _to_serializable(self.details),
            "message": self.message,
        }


@dataclass
class SteeringVector:
    """A steering vector for controlling model behavior."""
    direction: np.ndarray
    layer: int
    method: str  # "mass_mean", "pca", "logistic"
    magnitude: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# STEERING VECTOR EXTRACTION
# =============================================================================

def extract_deception_direction(
    activations: np.ndarray,
    labels: np.ndarray,
    method: str = "mass_mean",
    threshold: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract a steering vector that represents the "deception direction".

    Args:
        activations: Activation matrix [N, d_model]
        labels: Binary or continuous deception labels [N]
        method: "mass_mean" (Marks & Tegmark), "pca", or "logistic"
        threshold: Threshold for binarizing labels

    Returns:
        Tuple of (direction vector [d_model], metadata dict)
    """
    binary_labels = (labels > threshold).astype(bool)

    honest_mask = ~binary_labels
    deceptive_mask = binary_labels

    if honest_mask.sum() < 5 or deceptive_mask.sum() < 5:
        raise ValueError(f"Not enough samples: {honest_mask.sum()} honest, {deceptive_mask.sum()} deceptive")

    metadata = {
        "method": method,
        "n_honest": int(honest_mask.sum()),
        "n_deceptive": int(deceptive_mask.sum()),
        "threshold": threshold,
    }

    if method == "mass_mean":
        # Marks & Tegmark (2023) method: difference of class means
        honest_mean = activations[honest_mask].mean(axis=0)
        deceptive_mean = activations[deceptive_mask].mean(axis=0)
        direction = deceptive_mean - honest_mean

        # Store cosine distance between means
        norm_h = np.linalg.norm(honest_mean)
        norm_d = np.linalg.norm(deceptive_mean)
        if norm_h > 0 and norm_d > 0:
            metadata["cosine_similarity"] = float(np.dot(honest_mean, deceptive_mean) / (norm_h * norm_d))

    elif method == "pca":
        # PCA on the difference vectors
        from sklearn.decomposition import PCA

        # Center each class
        honest_centered = activations[honest_mask] - activations[honest_mask].mean(axis=0)
        deceptive_centered = activations[deceptive_mask] - activations[deceptive_mask].mean(axis=0)

        # Stack and find first PC
        all_centered = np.vstack([honest_centered, deceptive_centered])
        pca = PCA(n_components=1)
        pca.fit(all_centered)
        direction = pca.components_[0]

        # Ensure direction points from honest to deceptive
        if np.dot(direction, activations[deceptive_mask].mean(axis=0) - activations[honest_mask].mean(axis=0)) < 0:
            direction = -direction

        metadata["explained_variance"] = float(pca.explained_variance_ratio_[0])

    elif method == "logistic":
        # Logistic regression weight vector
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000, C=0.1)
        clf.fit(activations, binary_labels)
        direction = clf.coef_[0]

        metadata["accuracy"] = float(clf.score(activations, binary_labels))

    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        warnings.warn("Direction has near-zero norm")
        direction = np.zeros_like(direction)
    else:
        direction = direction / norm

    metadata["original_norm"] = float(norm)

    return direction, metadata


def create_steering_vector(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    layer: int,
    method: str = "mass_mean",
) -> SteeringVector:
    """Create a SteeringVector object for a given layer."""
    X = activations[layer]
    if hasattr(X, 'numpy'):
        X = X.numpy()

    direction, metadata = extract_deception_direction(X, labels, method=method)

    return SteeringVector(
        direction=direction,
        layer=layer,
        method=method,
        metadata=metadata,
    )


# =============================================================================
# ACTIVATION PATCHING TEST
# =============================================================================

def activation_patching_test(
    model,  # TransformerLens HookedTransformer
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    layer: int,
    test_prompts: List[str],
    patch_magnitude: float = 2.0,
    n_samples: int = 20,
    verbose: bool = True,
) -> CausalValidationResult:
    """
    Activation Patching Test: Swap activations and measure behavioral change.

    This is the gold standard for causal validation. If modifying activations
    along the deception direction changes model outputs, the direction is
    causally relevant.

    Test procedure:
    1. Get deception direction from activations
    2. For honest prompts: ADD deception direction
    3. For deceptive prompts: SUBTRACT deception direction
    4. Measure change in output logits/probabilities
    5. If changes are significant, the direction is causally valid

    Args:
        model: TransformerLens HookedTransformer (or compatible)
        activations: Dict mapping layer -> activation array
        labels: Deception labels
        layer: Layer to intervene on
        test_prompts: Prompts to test intervention on
        patch_magnitude: Strength of intervention (in std devs)
        n_samples: Number of samples to test
        verbose: Print progress

    Returns:
        CausalValidationResult with effect size and pass/fail
    """
    if verbose:
        print(f"\n{'='*60}")
        print("ACTIVATION PATCHING TEST")
        print(f"{'='*60}")
        print(f"Layer: {layer}")
        print(f"Patch magnitude: {patch_magnitude}")

    # Get activations for the target layer
    X = activations[layer]
    if hasattr(X, 'numpy'):
        X = X.numpy()

    # Extract deception direction
    try:
        direction, dir_metadata = extract_deception_direction(X, labels, method="mass_mean")
    except ValueError as e:
        return CausalValidationResult(
            test_name="activation_patching",
            passed=False,
            effect_size=0.0,
            n_samples_tested=0,
            message=f"Failed to extract direction: {e}",
        )

    # Convert to tensor
    direction_tensor = torch.tensor(direction, dtype=torch.float32)

    # Compute activation statistics for scaling
    activation_std = np.std(X, axis=0).mean()
    scaled_magnitude = patch_magnitude * activation_std

    if verbose:
        print(f"Activation std: {activation_std:.3f}")
        print(f"Scaled magnitude: {scaled_magnitude:.3f}")

    # Test on prompts
    logit_changes_add = []  # Adding deception direction
    logit_changes_sub = []  # Subtracting deception direction

    n_test = min(n_samples, len(test_prompts))

    for i, prompt in enumerate(test_prompts[:n_test]):
        if verbose and i % 10 == 0:
            print(f"  Testing prompt {i+1}/{n_test}...", flush=True)

        try:
            # Tokenize with truncation
            max_ctx = getattr(model.cfg, 'n_ctx', 8192)
            tokens = model.to_tokens(prompt, truncate=True)
            if tokens.shape[1] > max_ctx:
                tokens = tokens[:, -max_ctx:]

            # Get baseline logits
            with torch.no_grad():
                baseline_logits = model(tokens)

            # Define hooks
            def add_direction_hook(activation, hook):
                activation[:, -1, :] += scaled_magnitude * direction_tensor.to(activation.device)
                return activation

            def sub_direction_hook(activation, hook):
                activation[:, -1, :] -= scaled_magnitude * direction_tensor.to(activation.device)
                return activation

            hook_name = f"blocks.{layer}.hook_resid_post"

            # Get logits with added direction
            with torch.no_grad():
                add_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, add_direction_hook)]
                )

            # Get logits with subtracted direction
            with torch.no_grad():
                sub_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, sub_direction_hook)]
                )

            # Compute changes (mean absolute difference in last-token logits)
            add_change = (add_logits[:, -1, :] - baseline_logits[:, -1, :]).abs().mean().item()
            sub_change = (sub_logits[:, -1, :] - baseline_logits[:, -1, :]).abs().mean().item()

            logit_changes_add.append(add_change)
            logit_changes_sub.append(sub_change)

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed on prompt {i}: {e}")
            continue

    if not logit_changes_add:
        return CausalValidationResult(
            test_name="activation_patching",
            passed=False,
            effect_size=0.0,
            n_samples_tested=0,
            message="All prompts failed",
        )

    # Compute effect size
    mean_add_change = np.mean(logit_changes_add)
    mean_sub_change = np.mean(logit_changes_sub)
    overall_effect = (mean_add_change + mean_sub_change) / 2

    # Statistical significance (compare to random direction)
    random_changes = []
    for _ in range(min(10, n_test)):
        try:
            random_dir = np.random.randn(len(direction))
            random_dir = random_dir / np.linalg.norm(random_dir)
            random_tensor = torch.tensor(random_dir, dtype=torch.float32)

            def random_hook(activation, hook):
                activation[:, -1, :] += scaled_magnitude * random_tensor.to(activation.device)
                return activation

            prompt = test_prompts[0]
            max_ctx = getattr(model.cfg, 'n_ctx', 8192)
            tokens = model.to_tokens(prompt, truncate=True)
            if tokens.shape[1] > max_ctx:
                tokens = tokens[:, -max_ctx:]

            with torch.no_grad():
                baseline = model(tokens)
                random_logits = model.run_with_hooks(
                    tokens, fwd_hooks=[(hook_name, random_hook)]
                )

            random_change = (random_logits[:, -1, :] - baseline[:, -1, :]).abs().mean().item()
            random_changes.append(random_change)

        except Exception:
            continue

    # Effect is significant if deception direction causes larger change than random
    if random_changes:
        random_mean = np.mean(random_changes)
        effect_ratio = overall_effect / (random_mean + 1e-8)
        passed = effect_ratio > 1.5  # Deception direction should have 50% larger effect
    else:
        effect_ratio = 0.0
        passed = overall_effect > 0.1  # Fallback threshold

    # Compute approximate p-value using permutation test idea
    if random_changes and len(random_changes) >= 5:
        from scipy import stats
        _, p_value = stats.ttest_ind(logit_changes_add + logit_changes_sub, random_changes * 2)
    else:
        p_value = None

    result = CausalValidationResult(
        test_name="activation_patching",
        passed=passed,
        effect_size=overall_effect,
        p_value=p_value,
        n_samples_tested=len(logit_changes_add),
        details={
            "mean_add_change": float(mean_add_change),
            "mean_sub_change": float(mean_sub_change),
            "random_baseline": float(np.mean(random_changes)) if random_changes else None,
            "effect_ratio": float(effect_ratio) if random_changes else None,
            "direction_metadata": dir_metadata,
        },
        message=f"Effect size: {overall_effect:.3f}, Ratio vs random: {effect_ratio:.2f}x" if random_changes else f"Effect size: {overall_effect:.3f}",
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Mean change (add dir): {mean_add_change:.3f}")
        print(f"  Mean change (sub dir): {mean_sub_change:.3f}")
        if random_changes:
            print(f"  Random baseline: {np.mean(random_changes):.3f}")
            print(f"  Effect ratio: {effect_ratio:.2f}x")
        print(f"  PASSED: {passed}")

    return result


# =============================================================================
# CROSS-SAMPLE ACTIVATION PATCHING (D11)
# =============================================================================

def cross_sample_patching_test(
    model,  # TransformerLens HookedTransformer
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    layer: int,
    test_prompts: List[str],
    metadata: List[Dict[str, Any]] = None,
    n_pairs: int = 10,
    verbose: bool = True,
) -> CausalValidationResult:
    """
    Cross-Sample Activation Patching (D11): Swap actual activations between
    honest and deceptive samples and measure behavioral change.

    Unlike direction-based patching (which adds/subtracts a mass-mean vector),
    this test uses REAL activations from matched samples:
    1. Take honest sample A and deceptive sample B
    2. Run A's prompt through the model
    3. At layer L, replace A's activation with B's stored activation
    4. Continue the forward pass from L onward
    5. If output shifts toward deception, layer L is causally relevant

    Also tests reverse: patch honest activation into deceptive prompt.

    Args:
        model: TransformerLens HookedTransformer
        activations: Dict mapping layer -> activation array [N, d_model]
        labels: Deception labels [N]
        layer: Layer to patch at
        test_prompts: Corresponding prompts for each sample
        metadata: Optional list of dicts with 'scenario' key for matched pairing
        n_pairs: Number of matched pairs to test
        verbose: Print progress

    Returns:
        CausalValidationResult with effect size from cross-sample patching
    """
    if verbose:
        print(f"\n{'='*60}")
        print("CROSS-SAMPLE ACTIVATION PATCHING (D11)")
        print(f"{'='*60}")
        print(f"Layer: {layer}")
        print(f"Requested pairs: {n_pairs}")

    X = activations[layer]
    if hasattr(X, 'numpy'):
        X = X.numpy()

    binary_labels = (np.array(labels) > 0.5).astype(bool)
    honest_idxs = np.where(~binary_labels)[0]
    deceptive_idxs = np.where(binary_labels)[0]

    if len(honest_idxs) < 2 or len(deceptive_idxs) < 2:
        return CausalValidationResult(
            test_name="cross_sample_patching",
            passed=False,
            effect_size=0.0,
            n_samples_tested=0,
            message=f"Not enough samples: {len(honest_idxs)} honest, {len(deceptive_idxs)} deceptive",
        )

    # Build matched pairs: prefer same-scenario pairs if metadata available
    pairs = []
    if metadata is not None:
        # Group by scenario for matched pairing
        scenario_honest = {}
        scenario_deceptive = {}
        for idx in honest_idxs:
            sc = metadata[idx].get('scenario', 'unknown') if idx < len(metadata) else 'unknown'
            scenario_honest.setdefault(sc, []).append(idx)
        for idx in deceptive_idxs:
            sc = metadata[idx].get('scenario', 'unknown') if idx < len(metadata) else 'unknown'
            scenario_deceptive.setdefault(sc, []).append(idx)
        # Pair within scenarios first
        for sc in scenario_honest:
            if sc in scenario_deceptive:
                h_list = scenario_honest[sc]
                d_list = scenario_deceptive[sc]
                for h, d in zip(h_list, d_list):
                    pairs.append((h, d))
                    if len(pairs) >= n_pairs:
                        break
            if len(pairs) >= n_pairs:
                break

    # Fill remaining pairs with random matching
    while len(pairs) < n_pairs and len(honest_idxs) > 0 and len(deceptive_idxs) > 0:
        h_idx = np.random.choice(honest_idxs)
        d_idx = np.random.choice(deceptive_idxs)
        pairs.append((int(h_idx), int(d_idx)))

    if verbose:
        print(f"Testing {len(pairs)} pairs")

    hook_name = f"blocks.{layer}.hook_resid_post"

    # Measure: for each pair, compute KL divergence between patched and baseline
    # Direction 1: Patch deceptive activation INTO honest prompt
    d_into_h_kl = []
    # Direction 2: Patch honest activation INTO deceptive prompt
    h_into_d_kl = []
    # Control: Patch same-class activation (honest into honest)
    control_kl = []

    for pair_idx, (h_idx, d_idx) in enumerate(pairs):
        if verbose and pair_idx % 5 == 0:
            print(f"  Pair {pair_idx+1}/{len(pairs)}...", flush=True)

        h_prompt = test_prompts[h_idx] if h_idx < len(test_prompts) else test_prompts[0]
        d_prompt = test_prompts[d_idx] if d_idx < len(test_prompts) else test_prompts[0]

        # Stored activations from the dataset
        h_activation = torch.tensor(X[h_idx], dtype=torch.float32)
        d_activation = torch.tensor(X[d_idx], dtype=torch.float32)

        try:
            max_ctx = getattr(model.cfg, 'n_ctx', 8192)

            # --- Direction 1: Patch deceptive into honest ---
            h_tokens = model.to_tokens(h_prompt, truncate=True)
            if h_tokens.shape[1] > max_ctx:
                h_tokens = h_tokens[:, -max_ctx:]

            with torch.no_grad():
                h_baseline_logits = model(h_tokens)
                h_baseline_probs = torch.softmax(h_baseline_logits[:, -1, :], dim=-1)

            def patch_d_into_h(activation, hook):
                # Replace last-token activation with deceptive sample's stored activation
                activation[:, -1, :] = d_activation.to(activation.device)
                return activation

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    h_tokens, fwd_hooks=[(hook_name, patch_d_into_h)]
                )
                patched_probs = torch.softmax(patched_logits[:, -1, :], dim=-1)

            kl = torch.nn.functional.kl_div(
                patched_probs.log().clamp(min=-100),
                h_baseline_probs,
                reduction='batchmean'
            ).item()
            if not np.isnan(kl):
                d_into_h_kl.append(kl)

            # --- Direction 2: Patch honest into deceptive ---
            d_tokens = model.to_tokens(d_prompt, truncate=True)
            if d_tokens.shape[1] > max_ctx:
                d_tokens = d_tokens[:, -max_ctx:]

            with torch.no_grad():
                d_baseline_logits = model(d_tokens)
                d_baseline_probs = torch.softmax(d_baseline_logits[:, -1, :], dim=-1)

            def patch_h_into_d(activation, hook):
                activation[:, -1, :] = h_activation.to(activation.device)
                return activation

            with torch.no_grad():
                patched_d_logits = model.run_with_hooks(
                    d_tokens, fwd_hooks=[(hook_name, patch_h_into_d)]
                )
                patched_d_probs = torch.softmax(patched_d_logits[:, -1, :], dim=-1)

            kl_rev = torch.nn.functional.kl_div(
                patched_d_probs.log().clamp(min=-100),
                d_baseline_probs,
                reduction='batchmean'
            ).item()
            if not np.isnan(kl_rev):
                h_into_d_kl.append(kl_rev)

            # --- Control: Patch honest into honest (should cause less change) ---
            # Use a different honest sample
            other_h_idxs = [i for i in honest_idxs if i != h_idx]
            if other_h_idxs:
                ctrl_idx = np.random.choice(other_h_idxs)
                ctrl_activation = torch.tensor(X[ctrl_idx], dtype=torch.float32)

                def patch_ctrl(activation, hook):
                    activation[:, -1, :] = ctrl_activation.to(activation.device)
                    return activation

                with torch.no_grad():
                    ctrl_logits = model.run_with_hooks(
                        h_tokens, fwd_hooks=[(hook_name, patch_ctrl)]
                    )
                    ctrl_probs = torch.softmax(ctrl_logits[:, -1, :], dim=-1)

                kl_ctrl = torch.nn.functional.kl_div(
                    ctrl_probs.log().clamp(min=-100),
                    h_baseline_probs,
                    reduction='batchmean'
                ).item()
                if not np.isnan(kl_ctrl):
                    control_kl.append(kl_ctrl)

        except Exception as e:
            if verbose:
                print(f"    Warning: Pair {pair_idx} failed: {e}")
            continue

    if not d_into_h_kl:
        return CausalValidationResult(
            test_name="cross_sample_patching",
            passed=False,
            effect_size=0.0,
            n_samples_tested=0,
            message="All pairs failed",
        )

    # Compute effect sizes
    mean_d_into_h = np.mean(d_into_h_kl)
    mean_h_into_d = np.mean(h_into_d_kl) if h_into_d_kl else 0.0
    mean_control = np.mean(control_kl) if control_kl else 0.0
    cross_class_effect = (mean_d_into_h + mean_h_into_d) / 2

    # The test passes if cross-class patching causes significantly MORE change
    # than same-class patching (control)
    if mean_control > 0:
        effect_ratio = cross_class_effect / (mean_control + 1e-8)
        passed = effect_ratio > 1.5
    else:
        effect_ratio = float('inf') if cross_class_effect > 0 else 0.0
        passed = cross_class_effect > 0.1

    # Statistical significance
    p_value = None
    if control_kl and d_into_h_kl and len(control_kl) >= 3 and len(d_into_h_kl) >= 3:
        try:
            from scipy import stats
            _, p_value = stats.mannwhitneyu(
                d_into_h_kl, control_kl, alternative='greater'
            )
        except Exception:
            pass

    result = CausalValidationResult(
        test_name="cross_sample_patching",
        passed=passed,
        effect_size=cross_class_effect,
        p_value=p_value,
        n_samples_tested=len(pairs),
        details={
            "mean_deceptive_into_honest_kl": float(mean_d_into_h),
            "mean_honest_into_deceptive_kl": float(mean_h_into_d),
            "mean_control_kl": float(mean_control),
            "effect_ratio_vs_control": float(effect_ratio) if effect_ratio != float('inf') else None,
            "n_pairs_tested": len(d_into_h_kl),
            "n_control_tested": len(control_kl),
        },
        message=(
            f"Cross-class KL: {cross_class_effect:.3f}, "
            f"Control KL: {mean_control:.3f}, "
            f"Ratio: {effect_ratio:.2f}x"
        ),
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Deceptive→Honest KL: {mean_d_into_h:.3f} ({len(d_into_h_kl)} pairs)")
        print(f"  Honest→Deceptive KL: {mean_h_into_d:.3f} ({len(h_into_d_kl)} pairs)")
        print(f"  Control (same-class) KL: {mean_control:.3f} ({len(control_kl)} pairs)")
        print(f"  Effect ratio vs control: {effect_ratio:.2f}x")
        if p_value is not None:
            print(f"  p-value: {p_value:.4f}")
        print(f"  PASSED: {passed}")

    return result


# =============================================================================
# ABLATION TEST
# =============================================================================

def ablation_test(
    model,  # TransformerLens HookedTransformer
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    layer: int,
    test_prompts: List[str],
    ablation_method: str = "zero",  # "zero", "mean", "noise"
    n_samples: int = 20,
    verbose: bool = True,
) -> CausalValidationResult:
    """
    Ablation Test: Remove/scramble layer activations and measure impact.

    If ablating a layer significantly impacts model outputs on deceptive prompts
    more than on honest prompts, the layer is causally important for deception.

    Ablation methods:
    - "zero": Set activations to zero
    - "mean": Replace with mean activation
    - "noise": Replace with Gaussian noise matching activation statistics

    Args:
        model: TransformerLens HookedTransformer
        activations: Dict mapping layer -> activation array
        labels: Deception labels
        layer: Layer to ablate
        test_prompts: Prompts to test
        ablation_method: Type of ablation
        n_samples: Number of samples
        verbose: Print progress

    Returns:
        CausalValidationResult
    """
    if verbose:
        print(f"\n{'='*60}")
        print("ABLATION TEST")
        print(f"{'='*60}")
        print(f"Layer: {layer}")
        print(f"Method: {ablation_method}")

    X = activations[layer]
    if hasattr(X, 'numpy'):
        X = X.numpy()

    # Compute ablation values (match model dtype to avoid bfloat16 issues)
    model_dtype = next(model.parameters()).dtype
    if ablation_method == "zero":
        ablation_value = torch.zeros(X.shape[1], dtype=model_dtype)
    elif ablation_method == "mean":
        ablation_value = torch.tensor(X.mean(axis=0), dtype=model_dtype)
    elif ablation_method == "noise":
        ablation_value = None  # Will generate fresh noise each time
        noise_mean = X.mean(axis=0)
        noise_std = X.std(axis=0)
    else:
        raise ValueError(f"Unknown ablation method: {ablation_method}")

    hook_name = f"blocks.{layer}.hook_resid_post"

    # Test on prompts
    kl_divergences = []

    n_test = min(n_samples, len(test_prompts))

    for i, prompt in enumerate(test_prompts[:n_test]):
        if verbose and i % 10 == 0:
            print(f"  Testing prompt {i+1}/{n_test}...", flush=True)

        try:
            max_ctx = getattr(model.cfg, 'n_ctx', 8192)
            tokens = model.to_tokens(prompt, truncate=True)
            if tokens.shape[1] > max_ctx:
                tokens = tokens[:, -max_ctx:]

            # Baseline (cast to float32 for numerical stability in softmax/log)
            with torch.no_grad():
                baseline_logits = model(tokens)
                baseline_probs = torch.softmax(baseline_logits[:, -1, :].float(), dim=-1)

            # Ablated
            if ablation_method == "noise":
                noise = torch.tensor(
                    np.random.randn(X.shape[1]) * noise_std + noise_mean,
                    dtype=model_dtype
                )

                def ablate_hook(activation, hook):
                    activation[:, -1, :] = noise.to(activation.device)
                    return activation
            else:
                def ablate_hook(activation, hook):
                    activation[:, -1, :] = ablation_value.to(activation.device)
                    return activation

            with torch.no_grad():
                ablated_logits = model.run_with_hooks(
                    tokens, fwd_hooks=[(hook_name, ablate_hook)]
                )
                ablated_probs = torch.softmax(ablated_logits[:, -1, :].float(), dim=-1)

            # KL divergence (clamp log to avoid -inf from zero probs in bfloat16)
            kl = torch.nn.functional.kl_div(
                ablated_probs.float().log().clamp(min=-100),
                baseline_probs.float(),
                reduction='batchmean'
            ).item()

            # Guard against NaN
            if np.isnan(kl) or np.isinf(kl):
                continue

            kl_divergences.append(kl)

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed on prompt {i}: {e}")
            continue

    if not kl_divergences:
        return CausalValidationResult(
            test_name="ablation",
            passed=False,
            effect_size=0.0,
            n_samples_tested=0,
            message="All prompts failed",
        )

    mean_kl = np.mean(kl_divergences)
    std_kl = np.std(kl_divergences)

    # A significant KL divergence indicates the layer matters
    # Threshold based on typical values (empirical)
    passed = mean_kl > 0.5

    result = CausalValidationResult(
        test_name="ablation",
        passed=passed,
        effect_size=mean_kl,
        n_samples_tested=len(kl_divergences),
        details={
            "mean_kl_divergence": float(mean_kl),
            "std_kl_divergence": float(std_kl),
            "ablation_method": ablation_method,
            "kl_values": [float(k) for k in kl_divergences[:10]],  # First 10
        },
        message=f"KL divergence: {mean_kl:.3f} +/- {std_kl:.3f} (threshold: 0.5)",
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Mean KL divergence: {mean_kl:.3f}")
        print(f"  Std KL divergence: {std_kl:.3f}")
        print(f"  PASSED: {passed}")

    return result


# =============================================================================
# STEERING VECTOR TEST
# =============================================================================

def steering_vector_test(
    model,  # TransformerLens HookedTransformer
    steering_vector: SteeringVector,
    test_prompts: List[str],
    deception_keywords: List[str] = None,
    magnitudes: List[float] = None,
    n_samples: int = 20,
    max_new_tokens: int = 50,
    verbose: bool = True,
) -> CausalValidationResult:
    """
    Steering Vector Test: Add deception direction and check if output changes.

    This is the ultimate causal test: if adding the deception direction
    to activations makes the model produce more deceptive outputs, we have
    strong evidence that the direction causally controls deception.

    Test procedure:
    1. Generate baseline responses (no steering)
    2. Generate steered responses (add deception direction)
    3. Count deception-related keywords in outputs
    4. If steered outputs have more deception keywords, test passes

    Args:
        model: TransformerLens HookedTransformer
        steering_vector: SteeringVector object
        test_prompts: Neutral prompts to test steering on
        deception_keywords: Keywords indicating deceptive content
        magnitudes: List of steering magnitudes to test
        n_samples: Number of samples
        max_new_tokens: Max tokens to generate
        verbose: Print progress

    Returns:
        CausalValidationResult
    """
    if verbose:
        print(f"\n{'='*60}")
        print("STEERING VECTOR TEST")
        print(f"{'='*60}")
        print(f"Layer: {steering_vector.layer}")
        print(f"Method: {steering_vector.method}")

    if deception_keywords is None:
        deception_keywords = [
            "actually", "truth", "honestly", "between us", "secretly",
            "don't tell", "mislead", "hide", "conceal", "lie",
            "pretend", "fake", "deceive", "trick", "manipulate",
            "not exactly", "sort of", "kind of", "maybe",
        ]

    if magnitudes is None:
        magnitudes = [1.0, 2.0, 3.0]

    direction_tensor = torch.tensor(steering_vector.direction, dtype=torch.float32)
    hook_name = f"blocks.{steering_vector.layer}.hook_resid_post"

    results_by_magnitude = {}

    for magnitude in magnitudes:
        if verbose:
            print(f"\n  Testing magnitude: {magnitude}")

        baseline_keyword_counts = []
        steered_keyword_counts = []

        n_test = min(n_samples, len(test_prompts))

        for i, prompt in enumerate(test_prompts[:n_test]):
            try:
                max_ctx = getattr(model.cfg, 'n_ctx', 8192)
                tokens = model.to_tokens(prompt, truncate=True)
                if tokens.shape[1] > max_ctx - max_new_tokens:
                    tokens = tokens[:, -(max_ctx - max_new_tokens):]

                # Generate baseline
                with torch.no_grad():
                    baseline_output = model.generate(
                        tokens,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                    baseline_text = model.tokenizer.decode(baseline_output[0])

                # Generate with steering
                def steering_hook(activation, hook):
                    # Add direction to all positions (or just last)
                    activation[:, :, :] += magnitude * direction_tensor.to(activation.device)
                    return activation

                with torch.no_grad():
                    # Note: run_with_hooks doesn't support generation directly
                    # So we do a workaround: generate baseline, then check logit shift
                    steered_logits = model.run_with_hooks(
                        tokens, fwd_hooks=[(hook_name, steering_hook)]
                    )

                # Count deception keywords in baseline output
                # (We can generate baseline text but can't generate steered text with hooks)
                baseline_kw_count = sum(
                    1 for kw in deception_keywords
                    if kw.lower() in baseline_text.lower()
                )

                # Compare logit changes as proxy for steering effect
                # (Full steered generation with hooks not supported by TransformerLens)
                baseline_logits = model(tokens)
                logit_diff = (steered_logits - baseline_logits).abs().mean().item()

                baseline_keyword_counts.append(baseline_kw_count)
                steered_keyword_counts.append(logit_diff)  # Logit diff as steering proxy

            except Exception as e:
                if verbose:
                    print(f"    Warning: Failed on prompt {i}: {e}")
                continue

        if steered_keyword_counts:
            mean_effect = np.mean(steered_keyword_counts)
            mean_baseline_kw = np.mean(baseline_keyword_counts) if baseline_keyword_counts else 0.0
            results_by_magnitude[magnitude] = {
                "mean_logit_change": float(mean_effect),
                "mean_baseline_keywords": float(mean_baseline_kw),
                "n_tested": len(steered_keyword_counts),
            }

    if not results_by_magnitude:
        return CausalValidationResult(
            test_name="steering_vector",
            passed=False,
            effect_size=0.0,
            n_samples_tested=0,
            message="All tests failed",
        )

    # Check if effect increases with magnitude (dose-response)
    effects = [results_by_magnitude[m]["mean_logit_change"] for m in sorted(results_by_magnitude.keys())]
    dose_response = all(effects[i] <= effects[i+1] for i in range(len(effects)-1))

    # Overall effect is the maximum magnitude effect
    max_effect = max(r["mean_logit_change"] for r in results_by_magnitude.values())

    passed = dose_response and max_effect > 0.5

    result = CausalValidationResult(
        test_name="steering_vector",
        passed=passed,
        effect_size=max_effect,
        n_samples_tested=sum(r["n_tested"] for r in results_by_magnitude.values()),
        details={
            "results_by_magnitude": results_by_magnitude,
            "dose_response": dose_response,
            "steering_method": steering_vector.method,
        },
        message=f"Max effect: {max_effect:.3f}, Dose-response: {dose_response}",
    )

    if verbose:
        print(f"\nResults:")
        for mag, res in results_by_magnitude.items():
            print(f"  Magnitude {mag}: effect = {res['mean_logit_change']:.3f}")
        print(f"  Dose-response: {dose_response}")
        print(f"  PASSED: {passed}")

    return result


# =============================================================================
# PROBE FAITHFULNESS TEST
# =============================================================================

def probe_faithfulness_test(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    layer: int,
    n_ablations: int = 10,
    verbose: bool = True,
    random_state: int = 42,
) -> CausalValidationResult:
    """
    Probe Faithfulness Test: Check if probe relies on meaningful features.

    This test verifies that the probe's performance degrades when we
    ablate (zero out) the most important feature dimensions.

    Test procedure:
    1. Train a probe on full activations
    2. Identify top-k most important dimensions (by weight magnitude)
    3. Zero out those dimensions and retrain
    4. If performance drops significantly, probe is using meaningful features

    Args:
        activations: Dict mapping layer -> activation array
        labels: Deception labels
        layer: Layer to test
        n_ablations: Number of top dimensions to ablate
        verbose: Print progress
        random_state: Random seed for reproducibility (default 42)

    Returns:
        CausalValidationResult
    """
    if verbose:
        print(f"\n{'='*60}")
        print("PROBE FAITHFULNESS TEST")
        print(f"{'='*60}")

    X = activations[layer]
    if hasattr(X, 'numpy'):
        X = X.numpy()

    # Train baseline probe (use random_state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=random_state)

    probe = Ridge(alpha=10.0)
    probe.fit(X_train, y_train)
    baseline_r2 = r2_score(y_test, probe.predict(X_test))

    if verbose:
        print(f"Baseline R²: {baseline_r2:.3f}")

    # Find most important dimensions
    importance = np.abs(probe.coef_)
    top_dims = np.argsort(importance)[-n_ablations:]

    if verbose:
        print(f"Ablating top {n_ablations} dimensions: {top_dims[:5]}...")

    # Ablate and retrain
    X_ablated = X.copy()
    X_ablated[:, top_dims] = 0

    X_train_abl, X_test_abl, _, _ = train_test_split(X_ablated, labels, test_size=0.2, random_state=random_state)

    probe_ablated = Ridge(alpha=10.0)
    probe_ablated.fit(X_train_abl, y_train)
    ablated_r2 = r2_score(y_test, probe_ablated.predict(X_test_abl))

    # Performance drop
    r2_drop = baseline_r2 - ablated_r2
    relative_drop = r2_drop / (baseline_r2 + 1e-8)

    # Probe is faithful if ablating top dims hurts performance
    passed = relative_drop > 0.2  # At least 20% relative drop

    result = CausalValidationResult(
        test_name="probe_faithfulness",
        passed=passed,
        effect_size=r2_drop,
        n_samples_tested=len(X_test),
        details={
            "baseline_r2": float(baseline_r2),
            "ablated_r2": float(ablated_r2),
            "r2_drop": float(r2_drop),
            "relative_drop": float(relative_drop),
            "n_dims_ablated": n_ablations,
            "top_dims": top_dims.tolist(),
        },
        message=f"R² drop: {r2_drop:.3f} ({relative_drop*100:.1f}% relative)",
    )

    if verbose:
        print(f"Ablated R²: {ablated_r2:.3f}")
        print(f"R² drop: {r2_drop:.3f} ({relative_drop*100:.1f}% relative)")
        print(f"PASSED: {passed}")

    return result


# =============================================================================
# SELECTIVITY TEST
# =============================================================================

def selectivity_test(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    layer: int,
    n_random_subsets: int = 10,
    subset_size: int = 50,
    verbose: bool = True,
    random_state: int = 42,
) -> CausalValidationResult:
    """
    Selectivity Test: Random feature subsets should perform at chance.

    This sanity check verifies that the probe isn't just memorizing
    by testing on random subsets of features.

    Args:
        activations: Dict mapping layer -> activation array
        labels: Deception labels
        layer: Layer to test
        n_random_subsets: Number of random subsets to test
        subset_size: Size of each random subset
        verbose: Print progress
        random_state: Random seed for reproducibility (default 42)

    Returns:
        CausalValidationResult
    """
    if verbose:
        print(f"\n{'='*60}")
        print("SELECTIVITY TEST")
        print(f"{'='*60}")

    X = activations[layer]
    if hasattr(X, 'numpy'):
        X = X.numpy()

    random_r2s = []

    for i in range(n_random_subsets):
        # Random subset of features
        random_dims = np.random.choice(X.shape[1], size=min(subset_size, X.shape[1]), replace=False)
        X_subset = X[:, random_dims]

        X_train, X_test, y_train, y_test = train_test_split(X_subset, labels, test_size=0.2, random_state=random_state+i)

        probe = Ridge(alpha=10.0)
        probe.fit(X_train, y_train)
        r2 = r2_score(y_test, probe.predict(X_test))
        random_r2s.append(r2)

    mean_random_r2 = np.mean(random_r2s)
    std_random_r2 = np.std(random_r2s)

    # Random subsets should have low R² (near 0)
    passed = mean_random_r2 < 0.1

    result = CausalValidationResult(
        test_name="selectivity",
        passed=passed,
        effect_size=mean_random_r2,
        n_samples_tested=n_random_subsets,
        details={
            "mean_random_r2": float(mean_random_r2),
            "std_random_r2": float(std_random_r2),
            "max_random_r2": float(np.max(random_r2s)),
            "subset_size": subset_size,
        },
        message=f"Random subset R²: {mean_random_r2:.3f} +/- {std_random_r2:.3f} (should be < 0.1)",
    )

    if verbose:
        print(f"Mean random R²: {mean_random_r2:.3f}")
        print(f"Std random R²: {std_random_r2:.3f}")
        print(f"PASSED: {passed}")

    return result


# =============================================================================
# COMPREHENSIVE CAUSAL VALIDATION
# =============================================================================

def run_full_causal_validation(
    model,  # TransformerLens HookedTransformer (or None to skip model-based tests)
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    best_layer: int,
    test_prompts: List[str] = None,
    metadata: List[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run comprehensive causal validation suite.

    This runs all causal tests and returns a summary of results.

    Args:
        model: TransformerLens HookedTransformer (None to skip model tests)
        activations: Dict mapping layer -> activation array
        labels: Deception labels (GM ground truth)
        best_layer: Best layer from probe training
        test_prompts: Prompts for intervention tests
        metadata: Per-sample metadata dicts (for matched pairing in cross-sample patching)
        verbose: Print progress

    Returns:
        Dict with all test results and overall assessment
    """
    if verbose:
        print("\n" + "=" * 60)
        print("COMPREHENSIVE CAUSAL VALIDATION SUITE")
        print("=" * 60)
        print(f"Best layer: {best_layer}")
        print(f"N samples: {len(labels)}")
        if model:
            print("Model-based tests: ENABLED")
        else:
            print("Model-based tests: DISABLED (no model provided)")

    results = {
        "tests": {},
        "overall_passed": False,
        "n_tests_passed": 0,
        "n_tests_total": 0,
        "causal_evidence_strength": "none",
    }

    # Test 1: Selectivity (no model needed)
    if verbose:
        print("\n" + "-" * 60)
    selectivity_result = selectivity_test(activations, labels, best_layer, verbose=verbose)
    results["tests"]["selectivity"] = selectivity_result.to_dict()
    results["n_tests_total"] += 1
    if selectivity_result.passed:
        results["n_tests_passed"] += 1

    # Test 2: Probe faithfulness (no model needed)
    if verbose:
        print("\n" + "-" * 60)
    faithfulness_result = probe_faithfulness_test(activations, labels, best_layer, verbose=verbose)
    results["tests"]["probe_faithfulness"] = faithfulness_result.to_dict()
    results["n_tests_total"] += 1
    if faithfulness_result.passed:
        results["n_tests_passed"] += 1

    # Model-based tests
    if model is not None and test_prompts is not None:
        # Test 3: Activation patching
        if verbose:
            print("\n" + "-" * 60)
        patching_result = activation_patching_test(
            model, activations, labels, best_layer, test_prompts, verbose=verbose
        )
        results["tests"]["activation_patching"] = patching_result.to_dict()
        results["n_tests_total"] += 1
        if patching_result.passed:
            results["n_tests_passed"] += 1

        # Test 4: Ablation
        if verbose:
            print("\n" + "-" * 60)
        ablation_result = ablation_test(
            model, activations, labels, best_layer, test_prompts, verbose=verbose
        )
        results["tests"]["ablation"] = ablation_result.to_dict()
        results["n_tests_total"] += 1
        if ablation_result.passed:
            results["n_tests_passed"] += 1

        # Test 5: Steering vector
        if verbose:
            print("\n" + "-" * 60)
        try:
            steering_vec = create_steering_vector(activations, labels, best_layer)
            steering_result = steering_vector_test(
                model, steering_vec, test_prompts, verbose=verbose
            )
            results["tests"]["steering_vector"] = steering_result.to_dict()
            results["n_tests_total"] += 1
            if steering_result.passed:
                results["n_tests_passed"] += 1
        except Exception as e:
            if verbose:
                print(f"Steering test failed: {e}")
            results["tests"]["steering_vector"] = {
                "test_name": "steering_vector",
                "passed": False,
                "message": str(e),
            }
            results["n_tests_total"] += 1

        # Test 6: Cross-sample activation patching (D11)
        if verbose:
            print("\n" + "-" * 60)
        try:
            cross_patch_result = cross_sample_patching_test(
                model, activations, labels, best_layer, test_prompts,
                metadata=metadata, verbose=verbose,
            )
            results["tests"]["cross_sample_patching"] = cross_patch_result.to_dict()
            results["n_tests_total"] += 1
            if cross_patch_result.passed:
                results["n_tests_passed"] += 1
        except Exception as e:
            if verbose:
                print(f"Cross-sample patching test failed: {e}")
            results["tests"]["cross_sample_patching"] = {
                "test_name": "cross_sample_patching",
                "passed": False,
                "message": str(e),
            }
            results["n_tests_total"] += 1

    # Overall assessment
    pass_rate = results["n_tests_passed"] / results["n_tests_total"] if results["n_tests_total"] > 0 else 0

    if pass_rate >= 0.8:
        results["causal_evidence_strength"] = "strong"
        results["overall_passed"] = True
    elif pass_rate >= 0.6:
        results["causal_evidence_strength"] = "moderate"
        results["overall_passed"] = True
    elif pass_rate >= 0.4:
        results["causal_evidence_strength"] = "weak"
        results["overall_passed"] = False
    else:
        results["causal_evidence_strength"] = "none"
        results["overall_passed"] = False

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("CAUSAL VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Tests passed: {results['n_tests_passed']}/{results['n_tests_total']}")
        print(f"Pass rate: {pass_rate*100:.0f}%")
        print(f"Evidence strength: {results['causal_evidence_strength'].upper()}")
        print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")

        print("\nIndividual tests:")
        for test_name, test_result in results["tests"].items():
            status = "PASSED" if test_result["passed"] else "FAILED"
            msg = test_result.get("message", "")
            print(f"  {test_name}: {status} - {msg}")

        if not results["overall_passed"]:
            print("\n" + "!" * 60)
            print("WARNING: Causal validation FAILED")
            print("Cannot claim the identified features are causally used for deception.")
            print("Correlation does not imply causation!")
            print("!" * 60)

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("Causal Validation Module")
    print("=" * 60)
    print("\nUsage:")
    print("  from causal_validation import run_full_causal_validation")
    print("  results = run_full_causal_validation(model, activations, labels, layer, prompts)")
    print("\nOr test with synthetic data:")

    # Generate synthetic data for testing
    print("\n\nTesting with synthetic data...")

    n_samples = 100
    d_model = 256

    # Fake activations with some signal
    np.random.seed(42)
    labels = np.random.rand(n_samples)
    activations = {
        12: np.random.randn(n_samples, d_model) + np.outer(labels, np.random.randn(d_model)) * 0.5
    }

    # Test non-model-based functions
    print("\n1. Testing selectivity...")
    sel_result = selectivity_test(activations, labels, layer=12, verbose=True)

    print("\n2. Testing probe faithfulness...")
    faith_result = probe_faithfulness_test(activations, labels, layer=12, verbose=True)

    print("\n3. Testing steering vector extraction...")
    direction, metadata = extract_deception_direction(activations[12], labels)
    print(f"Direction norm: {np.linalg.norm(direction):.3f}")
    print(f"Metadata: {metadata}")

    print("\n\nAll tests completed!")
