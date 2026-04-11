#!/usr/bin/env python3
"""
Confound Control Analyses for Deception Probes
===============================================
Runs two analyses on existing activation data to check whether probes
detect actual deception vs. correlated task features (round number,
incentive condition).

Supports multiple models: Gemma-7B-IT, Llama-3.1-8B, Mistral-7B.

Analysis 1: Matched-State Evaluation
  - Balance honest/deceptive counts within each (round, incentive) group
  - Train probe on balanced subset -> compare AUC to original

Analysis 2: Residualized Probing
  - Regress out confound features (round_num, incentive) from activations
  - Train probe on residuals -> compare AUC to original

Usage:
    python run_confound_controls.py
    python run_confound_controls.py --model meta-llama/Llama-3.1-8B-Instruct --layer 16
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------- Configuration ----------

# Model short names for file path resolution (centralized in config)
from config.experiment import MODEL_SHORT_NAMES, get_model_short_name

# Default best probe layers per model
MODEL_DEFAULT_LAYERS = {
    "google/gemma-7b-it": 14,
    "meta-llama/Llama-3.1-8B-Instruct": 16,
    "mistralai/Mistral-7B-Instruct-v0.1": 16,
    "google/gemma-2b-it": 9,
}

# Gemma-7B original file paths (backwards compatible)
GEMMA_FILES = {
    "UB": "ultimatum_bluff/activations_merged_merged_ub.pt",
    "AB": "alliance_betrayal/activations_merged_20260210_220117.pt",
    "IW": "info_withholding/activations_merged_merged_iw.pt",
}

# Scenario short names to full names
SCENARIO_FULL_NAMES = {
    "UB": "ultimatum_bluff",
    "AB": "alliance_betrayal",
    "IW": "info_withholding",
    "CB": "capability_bluff",
    "HV": "hidden_value",
    "PB": "promise_break",
}

SEEDS = [42, 123, 456, 789, 1024]
PCA_COMPONENTS = 50
TEST_SIZE = 0.2

# HuggingFace fallback
HF_REPO = "sycorpia/ai-control-hackathon"
HF_FILES = {
    "UB": "activations_merged_ultimatum_bluff.pt",
    "AB": "activations_merged_alliance_betrayal.pt",
    "IW": "activations_merged_info_withholding.pt",
}


def ensure_file(data_dir, scenario_short, model_name="google/gemma-7b-it"):
    """Return path to .pt file, downloading from HuggingFace if needed."""
    short = MODEL_SHORT_NAMES.get(model_name, "")
    full_name = SCENARIO_FULL_NAMES.get(scenario_short, scenario_short)

    # 1. Model-specific file: {scenario_full}/activations_{model_short}_{scenario_full}.pt
    if short:
        model_specific = os.path.join(data_dir, full_name, f"activations_{short}_{full_name}.pt")
        if os.path.exists(model_specific):
            return model_specific
        # Flat layout
        model_specific_flat = os.path.join(data_dir, f"activations_{short}_{full_name}.pt")
        if os.path.exists(model_specific_flat):
            return model_specific_flat
        # Glob for timestamped files: activations_{model}_{scenario}_*_.pt
        import glob
        pattern = os.path.join(data_dir, full_name, f"activations_{short}_{full_name}_*.pt")
        matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if matches:
            return matches[0]  # Most recent

    # 2. Gemma original paths (backwards compatible)
    if model_name == "google/gemma-7b-it" and scenario_short in GEMMA_FILES:
        local_path = os.path.join(data_dir, GEMMA_FILES[scenario_short])
        if os.path.exists(local_path):
            return local_path

    # 3. Generic merged
    generic = os.path.join(data_dir, full_name, f"activations_merged_{full_name}.pt")
    if os.path.exists(generic):
        return generic

    # 4. HuggingFace fallback (Gemma only)
    if model_name == "google/gemma-7b-it" and scenario_short in HF_FILES:
        print(f"  Local file not found, downloading from HuggingFace ({HF_REPO})...")
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILES[scenario_short],
            repo_type="dataset",
        )

    raise FileNotFoundError(
        f"No activation file found for model={model_name}, scenario={scenario_short} in {data_dir}"
    )


# ---------- Helpers ----------

def load_data(path, layer):
    """Load .pt file and return activations at given layer, labels, metadata."""
    data = torch.load(path, map_location="cpu", weights_only=False)

    # Find the layer key (may be int or str depending on how it was saved)
    acts = data["activations"]
    if layer in acts:
        X = acts[layer]
    elif str(layer) in acts:
        X = acts[str(layer)]
    else:
        available = sorted(int(k) if str(k).isdigit() else k for k in acts.keys())
        raise ValueError(f"Layer {layer} not found in activations. Available: {available}")

    if isinstance(X, torch.Tensor):
        X = X.float().numpy()
    gm_labels = np.array(data["labels"]["gm_labels"], dtype=float)
    round_nums = np.array(data["labels"]["round_nums"], dtype=int)
    mode_labels = data["labels"]["mode_labels"]  # list of strings
    metadata = data["metadata"]
    return X, gm_labels, round_nums, mode_labels, metadata


def filter_emergent(X, gm_labels, round_nums, mode_labels, metadata):
    """Keep only emergent-mode samples."""
    mask = np.array([m == "emergent" for m in mode_labels])
    return (
        X[mask],
        gm_labels[mask],
        round_nums[mask],
        [metadata[i] for i in range(len(metadata)) if mask[i]],
    )


def binary_labels(gm_labels):
    """Convert continuous gm_labels to binary (>0.5 = deceptive)."""
    return (gm_labels > 0.5).astype(int)


def get_incentive_conditions(metadata):
    """Extract incentive_condition from metadata, return array of strings."""
    return np.array([m.get("incentive_condition", "none") or "none" for m in metadata])


def train_probe(X_train, y_train, X_test, y_test, seed):
    """Train LogisticRegression pipeline, return AUC."""
    n_components = min(PCA_COMPONENTS, X_train.shape[0] - 1, X_train.shape[1])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)


# ---------- Analysis 1: Matched-State Evaluation ----------

def matched_state_analysis(X, gm_labels, round_nums, metadata, seed):
    """
    Balance honest/deceptive counts within each (round, incentive) group,
    then train probe on the balanced subset.
    """
    y = binary_labels(gm_labels)
    incentives = get_incentive_conditions(metadata)

    # Check if incentive has variation
    unique_incentives = set(incentives)
    use_incentive = len(unique_incentives - {"none"}) > 1

    # Group samples by (round, incentive_if_varied)
    groups = defaultdict(lambda: {"honest": [], "deceptive": []})
    for i in range(len(X)):
        if use_incentive:
            key = (round_nums[i], incentives[i])
        else:
            key = (round_nums[i],)
        if y[i] == 0:
            groups[key]["honest"].append(i)
        else:
            groups[key]["deceptive"].append(i)

    # Subsample to balance within each group
    rng = np.random.RandomState(seed)
    matched_indices = []
    for key, g in groups.items():
        n_honest = len(g["honest"])
        n_deceptive = len(g["deceptive"])
        if n_honest == 0 or n_deceptive == 0:
            continue  # skip single-class groups
        n = min(n_honest, n_deceptive)
        matched_indices.extend(rng.choice(g["honest"], n, replace=False).tolist())
        matched_indices.extend(rng.choice(g["deceptive"], n, replace=False).tolist())

    if len(matched_indices) < 10:
        return None, 0  # not enough samples

    matched_indices = np.array(matched_indices)
    X_matched = X[matched_indices]
    y_matched = y[matched_indices]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_matched, y_matched, test_size=TEST_SIZE, stratify=y_matched, random_state=seed
    )

    if len(np.unique(y_test)) < 2:
        return None, len(matched_indices)

    auc = train_probe(X_train, y_train, X_test, y_test, seed)
    return auc, len(matched_indices)


# ---------- Analysis 2: Residualized Probing ----------

def residualized_probing(X, gm_labels, round_nums, metadata, seed):
    """
    Regress out confound features from activations, then train probe.
    """
    y = binary_labels(gm_labels)
    incentives = get_incentive_conditions(metadata)

    # Build confound matrix C
    # Column 1: round_num (int)
    C = round_nums.reshape(-1, 1).astype(float)

    # One-hot encode incentive if it has variation beyond 'none'
    unique_incentives = sorted(set(incentives) - {"none"})
    if len(unique_incentives) > 1:
        for inc in unique_incentives[:-1]:  # drop-last encoding
            col = (incentives == inc).astype(float).reshape(-1, 1)
            C = np.hstack([C, col])

    # Residualize: X_res = X - C @ beta
    reg = LinearRegression()
    reg.fit(C, X)
    X_residual = X - reg.predict(C)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_residual, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )

    if len(np.unique(y_test)) < 2:
        return None

    auc = train_probe(X_train, y_train, X_test, y_test, seed)
    return auc


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Confound control analyses for deception probes")
    parser.add_argument("--data-dir", default="experiment_results",
                        help="Directory containing scenario subdirectories (default: experiment_results)")
    parser.add_argument("--model", default="google/gemma-7b-it",
                        help="HuggingFace model name (default: google/gemma-7b-it). "
                             "Affects activation file lookup and output naming.")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to analyze (default: auto from model, e.g. 14 for Gemma, 16 for Llama/Mistral)")
    parser.add_argument("--original-auc", type=str, default=None,
                        help="Comma-separated original AUC values keyed by scenario short name "
                             "(e.g. 'UB=0.823,AB=0.818,IW=0.831'). If not provided, uses Gemma-7B defaults.")
    parser.add_argument("--scenarios", type=str, default=None,
                        help="Comma-separated scenario short names to run (e.g. 'UB,AB,IW,CB'). "
                             "Default: all scenarios with available data files.")
    args = parser.parse_args()

    # Resolve layer
    layer = args.layer if args.layer is not None else MODEL_DEFAULT_LAYERS.get(args.model, 14)

    # Resolve original AUCs (needed for drop calculation)
    original_aucs = {"UB": 0.823, "AB": 0.818, "IW": 0.831}  # Gemma-7B defaults
    if args.original_auc:
        for pair in args.original_auc.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                original_aucs[k.strip()] = float(v.strip())
            else:
                # Backwards compat: positional UB,AB,IW
                pass

    # Determine which scenarios to run
    if args.scenarios:
        scenario_list = [s.strip() for s in args.scenarios.split(",")]
    else:
        # Default: try all known scenarios, skip those without data
        scenario_list = list(SCENARIO_FULL_NAMES.keys())

    results = {}
    model_short = MODEL_SHORT_NAMES.get(args.model, "unknown")

    print("=" * 70)
    print("CONFOUND CONTROL ANALYSES")
    print("=" * 70)
    print(f"  Model: {args.model} ({model_short})")
    print(f"  Layer: {layer}")
    print(f"  Scenarios: {scenario_list}")

    for scenario_name in scenario_list:
        try:
            pt_path = ensure_file(args.data_dir, scenario_name, args.model)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {scenario_name}: {e}")
            continue

        orig_auc = original_aucs.get(scenario_name, 0.0)

        print(f"\n{'─' * 70}")
        print(f"Scenario: {scenario_name}  |  Original AUC: {orig_auc:.3f}")
        print(f"{'─' * 70}")

        # Load and filter to emergent
        X_all, gm_all, rounds_all, modes_all, meta_all = load_data(pt_path, layer)
        X, gm, rounds, meta = filter_emergent(X_all, gm_all, rounds_all, modes_all, meta_all)

        y = binary_labels(gm)
        n_honest = (y == 0).sum()
        n_deceptive = (y == 1).sum()
        print(f"  Emergent samples: {len(X)} (honest={n_honest}, deceptive={n_deceptive})")

        if n_honest < 5 or n_deceptive < 5:
            print(f"  [SKIP] Not enough samples of both classes")
            continue

        scenario_results = {"original_auc": orig_auc}

        # --- Analysis 1: Matched-State ---
        print(f"\n  Analysis 1: Matched-State Evaluation")
        aucs_matched = []
        for seed in SEEDS:
            auc, n_matched = matched_state_analysis(X, gm, rounds, meta, seed)
            if auc is not None:
                aucs_matched.append(auc)

        if aucs_matched:
            mean_auc = np.mean(aucs_matched)
            std_auc = np.std(aucs_matched)
            drop = orig_auc - mean_auc
            print(f"    Matched AUC: {mean_auc:.3f} ± {std_auc:.3f}  (n_seeds={len(aucs_matched)})")
            print(f"    Drop from original: {drop:+.3f}")
            if drop < 0.05:
                print(f"    → Probe robust to round/incentive balancing (drop < 0.05)")
            else:
                print(f"    → Notable drop — probe may partly rely on confound structure")
            scenario_results["matched_state"] = {
                "mean_auc": round(mean_auc, 4),
                "std_auc": round(std_auc, 4),
                "n_seeds": len(aucs_matched),
                "drop": round(drop, 4),
                "per_seed": [round(a, 4) for a in aucs_matched],
            }
        else:
            print(f"    [FAIL] Could not run — insufficient balanced samples")
            scenario_results["matched_state"] = {"error": "insufficient samples"}

        # --- Analysis 2: Residualized Probing ---
        print(f"\n  Analysis 2: Residualized Probing")
        aucs_resid = []
        for seed in SEEDS:
            auc = residualized_probing(X, gm, rounds, meta, seed)
            if auc is not None:
                aucs_resid.append(auc)

        if aucs_resid:
            mean_auc = np.mean(aucs_resid)
            std_auc = np.std(aucs_resid)
            drop = orig_auc - mean_auc
            print(f"    Residualized AUC: {mean_auc:.3f} ± {std_auc:.3f}  (n_seeds={len(aucs_resid)})")
            print(f"    Drop from original: {drop:+.3f}")
            if drop < 0.05:
                print(f"    → Probe signal survives confound removal (drop < 0.05)")
            else:
                print(f"    → Notable drop — confounds explain part of probe signal")
            scenario_results["residualized"] = {
                "mean_auc": round(mean_auc, 4),
                "std_auc": round(std_auc, 4),
                "n_seeds": len(aucs_resid),
                "drop": round(drop, 4),
                "per_seed": [round(a, 4) for a in aucs_resid],
            }
        else:
            print(f"    [FAIL] Could not run — single class in test split")
            scenario_results["residualized"] = {"error": "single class"}

        results[scenario_name] = scenario_results

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Scenario':<8} {'Original':>10} {'Matched':>14} {'Residualized':>14}")
    print(f"{'─' * 50}")
    for sc, r in results.items():
        orig = f"{r['original_auc']:.3f}"
        ms = r.get("matched_state", {})
        ms_str = f"{ms['mean_auc']:.3f}±{ms['std_auc']:.3f}" if "mean_auc" in ms else "N/A"
        rs = r.get("residualized", {})
        rs_str = f"{rs['mean_auc']:.3f}±{rs['std_auc']:.3f}" if "mean_auc" in rs else "N/A"
        print(f"{sc:<8} {orig:>10} {ms_str:>14} {rs_str:>14}")

    # Save results
    if args.model == "google/gemma-7b-it":
        out_path = os.path.join(args.data_dir, "confound_control_results.json")
    else:
        out_path = os.path.join(args.data_dir, f"confound_control_results_{model_short}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
