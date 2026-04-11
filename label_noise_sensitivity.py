#!/usr/bin/env python3
"""
Label noise sensitivity analysis (optimized).

Precomputes PCA/scaling once, then only retrains ridge per noise level.
"""

import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent / "experiment_results"
FILES = {
    "UB": BASE / "ultimatum_bluff" / "activations_merged_merged_ub.pt",
    "AB": BASE / "alliance_betrayal" / "activations_merged_alliance_betrayal.pt",
    "IW": BASE / "info_withholding" / "activations_merged_merged_iw.pt",
}

LAYER = 14
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.09, 0.10, 0.15, 0.20]
N_SEEDS = 30


def load_data(filepath, layer=14):
    import torch
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    activations = data["activations"][layer].float().numpy()
    gm_labels = data["labels"]["gm_labels"]
    mode_labels = data["labels"]["mode_labels"]

    if hasattr(gm_labels, "numpy"):
        gm_labels = gm_labels.float().numpy()
    elif isinstance(gm_labels, list):
        gm_labels = np.array(gm_labels, dtype=np.float32)

    if hasattr(mode_labels, "numpy"):
        mode_labels = mode_labels.numpy()
    elif isinstance(mode_labels, list):
        mode_labels = np.array(mode_labels)

    # Filter emergent only
    mask = np.array([(m == 0 or str(m) == "0" or m == "emergent")
                     for m in mode_labels])
    activations = activations[mask]
    gm_labels = gm_labels[mask]
    labels = (gm_labels >= 0.5).astype(np.int32)
    return activations, labels


def run_scenario(name, filepath):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    import time

    print(f"\n{'='*60}")
    print(f"Loading {name}...")
    t0 = time.time()
    X, y = load_data(filepath)
    print(f"  Loaded: n={len(y)}, deception_rate={y.mean():.3f} ({time.time()-t0:.1f}s)")

    # Fixed train/test split on clean labels
    X_train, X_test, y_train_clean, y_test_clean = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Precompute PCA + scaling ONCE
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=50, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)

    # Clean mass-mean direction (in original space)
    clean_dir = X[y == 1].mean(0) - X[y == 0].mean(0)
    clean_dir /= (np.linalg.norm(clean_dir) + 1e-10)

    results = []

    for noise in NOISE_LEVELS:
        aucs = []
        cosines = []
        t1 = time.time()

        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed + 10000)

            # Flip training labels only (test stays clean)
            noisy_train = y_train_clean.copy()
            flip = rng.random(len(noisy_train)) < noise
            noisy_train[flip] = 1 - noisy_train[flip]

            # Train ridge on noisy labels
            clf = RidgeClassifier(alpha=10.0)
            clf.fit(X_train_p, noisy_train)

            # Evaluate on CLEAN test labels
            scores = clf.decision_function(X_test_p)
            try:
                auc = roc_auc_score(y_test_clean, scores)
            except ValueError:
                auc = 0.5
            aucs.append(auc)

            # Direction stability (on full data with noisy labels)
            noisy_y = y.copy()
            flip_all = rng.random(len(y)) < noise
            noisy_y[flip_all] = 1 - noisy_y[flip_all]
            d1 = X[noisy_y == 1].mean(0) - X[noisy_y == 0].mean(0)
            d1 /= (np.linalg.norm(d1) + 1e-10)
            cosines.append(float(np.dot(clean_dir, d1)))

        m_auc, s_auc = np.mean(aucs), np.std(aucs)
        m_cos, s_cos = np.mean(cosines), np.std(cosines)
        dt = time.time() - t1
        print(f"  Noise {noise:5.1%}: AUC={m_auc:.4f}±{s_auc:.4f}  "
              f"cos_sim={m_cos:.4f}±{s_cos:.4f}  ({dt:.1f}s)")

        results.append({
            "noise_rate": noise, "auc_mean": float(m_auc),
            "auc_std": float(s_auc), "direction_cosine_mean": float(m_cos),
            "direction_cosine_std": float(s_cos),
        })

    return {"scenario": name, "n_samples": len(y),
            "deception_rate": float(y.mean()), "noise_levels": results}


def make_figure(all_results, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"UB": "#2196F3", "AB": "#FF9800", "IW": "#4CAF50"}
    markers = {"UB": "o", "AB": "s", "IW": "^"}

    for sc, res in all_results.items():
        ns = [r["noise_rate"] * 100 for r in res["noise_levels"]]
        am = [r["auc_mean"] for r in res["noise_levels"]]
        ae = [r["auc_std"] for r in res["noise_levels"]]
        cm = [r["direction_cosine_mean"] for r in res["noise_levels"]]
        ce = [r["direction_cosine_std"] for r in res["noise_levels"]]

        ax1.errorbar(ns, am, yerr=ae, marker=markers[sc], color=colors[sc],
                     label=sc, capsize=3, linewidth=2, markersize=6)
        ax2.errorbar(ns, cm, yerr=ce, marker=markers[sc], color=colors[sc],
                     label=sc, capsize=3, linewidth=2, markersize=6)

    for ax in [ax1, ax2]:
        ax.axvline(x=9, color="red", linestyle="--", alpha=0.7, linewidth=1)
        ax.grid(True, alpha=0.3)

    ax1.annotate("our label\nnoise (9%)", xy=(9, ax1.get_ylim()[0]),
                 xytext=(12, 0.55), fontsize=8, color="red",
                 arrowprops=dict(arrowstyle="->", color="red", alpha=0.7))

    ax1.set_xlabel("Label Noise Rate (%)")
    ax1.set_ylabel("Probe AUROC (evaluated on clean labels)")
    ax1.set_title("(a) AUC Degradation Under Label Noise")
    ax1.legend(loc="lower left")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Label Noise Rate (%)")
    ax2.set_ylabel("Cosine Sim. to Clean Direction")
    ax2.set_title("(b) Mass-Mean Direction Stability")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"\nFigure: {output_path} and {pdf_path}")


def main():
    all_results = {}
    for sc, fp in FILES.items():
        if not fp.exists():
            print(f"SKIP {sc}: not found")
            continue
        all_results[sc] = run_scenario(sc, fp)

    out = str(BASE / "label_noise_sensitivity.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON: {out}")

    if all_results:
        fig = str(Path(__file__).parent.parent / "latex_paper" / "figures" /
                  "fig7_label_noise.png")
        make_figure(all_results, fig)


if __name__ == "__main__":
    main()
