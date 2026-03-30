#!/opt/homebrew/bin/python3.11
"""
Confound Control Analyses for Deception Probes
===============================================
Runs two analyses on existing activation data to check whether probes
detect actual deception vs. correlated task features (round number,
incentive condition).

Analysis 1: Matched-State Evaluation
  - Balance honest/deceptive counts within each (round, incentive) group
  - Train probe on balanced subset -> compare AUC to original

Analysis 2: Residualized Probing
  - Regress out confound features (round_num, incentive) from activations
  - Train probe on residuals -> compare AUC to original
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

SCENARIOS = {
    "UB": {
        "file": "ultimatum_bluff/activations_merged_merged_ub.pt",
        "original_auc": 0.823,
    },
    "AB": {
        "file": "alliance_betrayal/activations_merged_20260210_220117.pt",
        "original_auc": 0.818,
    },
    "IW": {
        "file": "info_withholding/activations_merged_merged_iw.pt",
        "original_auc": 0.831,
    },
}

LAYER = 14
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


def ensure_file(data_dir, scenario_name, local_file):
    """Return path to .pt file, downloading from HuggingFace if needed."""
    local_path = os.path.join(data_dir, local_file)
    if os.path.exists(local_path):
        return local_path
    if scenario_name in HF_FILES:
        print(f"  Local file not found: {local_path}")
        print(f"  Downloading from HuggingFace ({HF_REPO})...")
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILES[scenario_name],
            repo_type="dataset",
        )
    return local_path  # will fail later with file not found


# ---------- Helpers ----------

def load_data(path):
    """Load .pt file and return activations (layer 14), labels, metadata."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    X = data["activations"][LAYER]
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
    args = parser.parse_args()

    results = {}

    print("=" * 70)
    print("CONFOUND CONTROL ANALYSES")
    print("=" * 70)

    for scenario_name, cfg in SCENARIOS.items():
        pt_path = ensure_file(args.data_dir, scenario_name, cfg["file"])
        if not os.path.exists(pt_path):
            print(f"\n[SKIP] {scenario_name}: file not found at {pt_path}")
            continue

        print(f"\n{'─' * 70}")
        print(f"Scenario: {scenario_name}  |  Original AUC: {cfg['original_auc']:.3f}")
        print(f"{'─' * 70}")

        # Load and filter to emergent
        X_all, gm_all, rounds_all, modes_all, meta_all = load_data(pt_path)
        X, gm, rounds, meta = filter_emergent(X_all, gm_all, rounds_all, modes_all, meta_all)

        y = binary_labels(gm)
        n_honest = (y == 0).sum()
        n_deceptive = (y == 1).sum()
        print(f"  Emergent samples: {len(X)} (honest={n_honest}, deceptive={n_deceptive})")

        if n_honest < 5 or n_deceptive < 5:
            print(f"  [SKIP] Not enough samples of both classes")
            continue

        scenario_results = {"original_auc": cfg["original_auc"]}

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
            drop = cfg["original_auc"] - mean_auc
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
            drop = cfg["original_auc"] - mean_auc
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
    out_path = os.path.join(args.data_dir, "confound_control_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
