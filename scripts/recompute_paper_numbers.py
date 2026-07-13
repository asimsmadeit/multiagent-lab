#!/usr/bin/env python3
"""Recompute the headline paper numbers on post-audit data.

Created 2026-04-25 for the AAMAS NEXUS + AI4CNI camera-ready preparations.
Refreshes every quantitative claim from the original COLM/CAIS draft so
the new submission cites post-audit numbers consistently.

Outputs (to experiment_results/paper_numbers/):
  cosine_similarity_matrix.json  — commission/omission/emergent/instructed
  cross_scenario_transfer.json   — LOSO transfer matrix per model
  probe_faithfulness.json        — mass-mean ablation on headline scenarios
  per_round_auc.json             — temporal section, with bootstrap CIs
  baselines.json                 — shuffled labels, random features, etc.

CPU only. Uses cached HF activations from previous runs.
"""

from __future__ import annotations

# NumPy exposes RandomState at runtime; Pylint cannot infer the lazy namespace.
# pylint: disable=no-member

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from interpretability.script_artifacts import (
    add_legacy_trust_argument,
    download_activation_input,
    load_activation_input,
)

HF_REPO = 'sycorpia/multiagent-lab-data'
ALPHA = 100.0
N_COMP = 30
SEED = 42

# Activation files we'll use, organized by model and scenario.
RUNS = {
    # Gemma-7B no-ToM (cleanest data, headline)
    ('gemma7b', 'no_tom', 'ultimatum_bluff'):
        'gemma7b/ultimatum_bluff/no_tom_20260412_042958/activations.pt',
    ('gemma7b', 'no_tom', 'alliance_betrayal'):
        'gemma7b/alliance_betrayal/no_tom_20260412_043012/activations.pt',
    ('gemma7b', 'no_tom', 'info_withholding'):
        'gemma7b/info_withholding/no_tom_20260412_043021/activations.pt',
    # Gemma-7B with-ToM (the paper's original data slice)
    ('gemma7b', 'tom', 'ultimatum_bluff'):
        'gemma7b/ultimatum_bluff/tom_20260211/activations.pt',
    ('gemma7b', 'tom', 'alliance_betrayal'):
        'gemma7b/alliance_betrayal/tom_20260210/activations.pt',
    ('gemma7b', 'tom', 'info_withholding'):
        'gemma7b/info_withholding/tom_20260211/activations.pt',
    # Llama-3.1-8B
    ('llama31_8b', 'tom', 'ultimatum_bluff'):
        'llama31_8b/ultimatum_bluff/tom_20260414_192608/activations.pt',
    ('llama31_8b', 'tom', 'alliance_betrayal'):
        'llama31_8b/alliance_betrayal/tom_20260412_043955/activations.pt',
    ('llama31_8b', 'tom', 'info_withholding'):
        'llama31_8b/info_withholding/tom_20260412_051758/activations.pt',
    # Mistral-7B
    ('mistral7b', 'tom', 'ultimatum_bluff'):
        'ultimatum_bluff/activations_mistral7b_ultimatum_bluff_both_20260412_194428.pt',
    ('mistral7b', 'tom', 'alliance_betrayal'):
        'alliance_betrayal/activations_mistral7b_alliance_betrayal_both_20260413_152248.pt',
    ('mistral7b', 'tom', 'info_withholding'):
        'info_withholding/activations_mistral7b_info_withholding_both_20260413_103456.pt',
}


def _is_negotiation(md: Dict[str, Any]) -> bool:
    st = md.get('sample_type')
    if st is not None:
        return st == 'negotiation'
    rn = md.get('round_num')
    return rn is None or rn >= 0


def load_run(
    key,
    layer: Optional[int] = None,
    *,
    trust_legacy_pt: bool = False,
):
    path = RUNS[key]
    print(f"  Loading {path}", flush=True)
    fp = download_activation_input(HF_REPO, path)
    data = load_activation_input(fp, trust_legacy_pt=trust_legacy_pt)
    md = data.get('metadata', [])
    labels = data['labels']
    acts = data['activations']
    y_all = np.array(labels['gm_labels'])
    n = len(y_all)
    keep = np.array([_is_negotiation(md[i]) if i < len(md) else True
                     for i in range(n)], dtype=bool)
    y = y_all[keep]
    if layer is None:
        # Default: layer 14 if available (paper convention), else mid layer
        layer_keys = [k for k in acts if isinstance(k, int)]
        layer = 14 if 14 in layer_keys else sorted(layer_keys)[len(layer_keys) // 2]
    X = acts[layer].float().numpy()[keep]
    mode = labels.get('mode_labels', None)
    if mode:
        mode = [mode[i] for i in range(n) if keep[i]]
    return dict(X=X, y=y, mode=mode, layer=layer)


def mass_mean_direction(X, y_binary):
    """Compute mass-mean direction: mean(deceptive) - mean(honest), normalized."""
    if y_binary.sum() < 2 or (~y_binary).sum() < 2:
        return None
    dec_mean = X[y_binary].mean(axis=0)
    hon_mean = X[~y_binary].mean(axis=0)
    direction = dec_mean - hon_mean
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return None
    return direction / norm


# ============================================================
# Section 1: Commission/omission cosine similarity matrices
# ============================================================

def section1_cosine_similarities(*, trust_legacy_pt: bool = False):
    print("\n=== Section 1: Cosine similarity matrices (bipolar deception) ===")
    results = {}
    for model in ('gemma7b', 'llama31_8b', 'mistral7b'):
        # For Gemma we have both no_tom and tom; report both
        configs = ['tom', 'no_tom'] if model == 'gemma7b' else ['tom']
        for config in configs:
            print(f"\n  Model: {model} ({config})")
            directions = {}
            for scenario in ('ultimatum_bluff', 'alliance_betrayal', 'info_withholding'):
                key = (model, config, scenario)
                if key not in RUNS:
                    continue
                d = load_run(key, trust_legacy_pt=trust_legacy_pt)
                X = d['X']
                y_binary = (d['y'] > 0.5)
                if y_binary.sum() < 2 or (~y_binary).sum() < 2:
                    print(f"    {scenario}: insufficient samples, skipping")
                    continue
                # Emergent direction (we only have one mode per file currently)
                dir_em = mass_mean_direction(X, y_binary)
                if dir_em is not None:
                    directions[(scenario, 'em')] = dir_em
                    print(f"    {scenario}/em: dir computed (n_dec={int(y_binary.sum())}, n_hon={int((~y_binary).sum())})")
                # Try instructed if present
                if d.get('mode') is not None:
                    mode_arr = np.array(d['mode'])
                    instr_mask = mode_arr == 'instructed'
                    if instr_mask.sum() >= 10:
                        X_i = X[instr_mask]
                        y_i = y_binary[instr_mask]
                        if y_i.sum() >= 2 and (~y_i).sum() >= 2:
                            dir_in = mass_mean_direction(X_i, y_i)
                            if dir_in is not None:
                                directions[(scenario, 'in')] = dir_in
                                print(f"    {scenario}/in: dir computed (n={int(instr_mask.sum())})")

            # Compute cosine similarity matrix among these directions
            keys = sorted(directions.keys())
            n = len(keys)
            cos_mat = np.zeros((n, n))
            for i, ki in enumerate(keys):
                for j, kj in enumerate(keys):
                    cos_mat[i, j] = float(np.dot(directions[ki], directions[kj]))
            print(f"\n    Cosine similarity matrix ({len(keys)} directions):")
            label_strs = [f"{s[:2]}/{m}" for s, m in keys]
            print("           " + "  ".join(f"{l:>7s}" for l in label_strs))
            for i, k in enumerate(keys):
                row = "    " + f"{label_strs[i]:>7s}"
                for j in range(n):
                    row += f"  {cos_mat[i, j]:>+.3f}"
                print(row)

            results[f'{model}_{config}'] = {
                'directions': [{'scenario': s, 'mode': m} for s, m in keys],
                'cosine_matrix': cos_mat.tolist(),
            }
    return results


# ============================================================
# Section 2: Cross-scenario transfer (LOSO)
# ============================================================

def section2_cross_scenario_transfer(*, trust_legacy_pt: bool = False):
    print("\n=== Section 2: Cross-scenario transfer ===")
    results = {}
    for model in ('gemma7b', 'llama31_8b', 'mistral7b'):
        configs = ['tom', 'no_tom'] if model == 'gemma7b' else ['tom']
        for config in configs:
            print(f"\n  Model: {model} ({config})")
            scenarios = ['ultimatum_bluff', 'alliance_betrayal', 'info_withholding']
            data_per_sc = {}
            for sc in scenarios:
                key = (model, config, sc)
                if key not in RUNS:
                    continue
                d = load_run(key, trust_legacy_pt=trust_legacy_pt)
                if d['X'].shape[1] != 4096 and d['X'].shape[1] != 3072:
                    # Layer dim should match; skip if oddly shaped
                    pass
                data_per_sc[sc] = d
            transfer_mat = {}
            for tr_sc in scenarios:
                for te_sc in scenarios:
                    if tr_sc not in data_per_sc or te_sc not in data_per_sc:
                        continue
                    Xtr = data_per_sc[tr_sc]['X']
                    ytr = data_per_sc[tr_sc]['y']
                    Xte = data_per_sc[te_sc]['X']
                    yte = data_per_sc[te_sc]['y']
                    if Xtr.shape[1] != Xte.shape[1]:
                        continue
                    if (ytr > 0.5).sum() < 2 or (yte > 0.5).sum() < 2:
                        continue
                    sc_ = StandardScaler()
                    Xtr_s = sc_.fit_transform(Xtr)
                    Xte_s = sc_.transform(Xte)
                    n_c = min(N_COMP, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
                    pca = PCA(n_components=n_c)
                    Xtr_p = pca.fit_transform(Xtr_s)
                    Xte_p = pca.transform(Xte_s)
                    probe = Ridge(alpha=ALPHA)
                    probe.fit(Xtr_p, ytr)
                    pred = probe.predict(Xte_p)
                    try:
                        auc = roc_auc_score((yte > 0.5).astype(int), pred)
                    except ValueError:
                        auc = 0.5
                    transfer_mat[f'{tr_sc}->{te_sc}'] = float(auc)
            print(f"  Transfer AUCs:")
            for k, v in transfer_mat.items():
                print(f"    {k}: {v:.3f}")
            results[f'{model}_{config}'] = transfer_mat
    return results


# ============================================================
# Section 3: Probe faithfulness — mass-mean ablation
# ============================================================

def section3_probe_faithfulness(*, trust_legacy_pt: bool = False):
    print("\n=== Section 3: Probe faithfulness (mass-mean ablation) ===")
    targets = [
        ('gemma7b', 'no_tom', 'info_withholding', 'headline'),
        ('gemma7b', 'no_tom', 'ultimatum_bluff', 'mid'),
        ('gemma7b', 'no_tom', 'alliance_betrayal', 'mid'),
        ('llama31_8b', 'tom', 'info_withholding', 'cross-model'),
        ('mistral7b', 'tom', 'alliance_betrayal', 'cross-model'),
    ]
    results = {}
    for model, config, scenario, label in targets:
        key = (model, config, scenario)
        if key not in RUNS:
            continue
        print(f"\n  {model}/{config}/{scenario} ({label})")
        d = load_run(key, trust_legacy_pt=trust_legacy_pt)
        X, y = d['X'], d['y']
        y_binary = (y > 0.5)
        if y_binary.sum() < 2 or (~y_binary).sum() < 2:
            print("    insufficient class balance, skipping")
            continue
        # Baseline probe AUC
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                               random_state=SEED, stratify=y_binary.astype(int))
        sc_ = StandardScaler(); Xtr_s = sc_.fit_transform(Xtr); Xte_s = sc_.transform(Xte)
        pca = PCA(n_components=min(N_COMP, Xtr_s.shape[0] - 1, Xtr_s.shape[1]))
        Xtr_p = pca.fit_transform(Xtr_s); Xte_p = pca.transform(Xte_s)
        probe = Ridge(alpha=ALPHA); probe.fit(Xtr_p, ytr)
        pred = probe.predict(Xte_p)
        try:
            baseline_auc = roc_auc_score((yte > 0.5).astype(int), pred)
        except: baseline_auc = 0.5

        # Mass-mean ablation: project out the mean-difference direction
        direction = mass_mean_direction(Xtr, (ytr > 0.5))
        if direction is None:
            continue
        proj_tr = (Xtr @ direction)[:, None] * direction[None, :]
        Xtr_ab = Xtr - proj_tr
        proj_te = (Xte @ direction)[:, None] * direction[None, :]
        Xte_ab = Xte - proj_te

        sc_a = StandardScaler(); Xtr_as = sc_a.fit_transform(Xtr_ab); Xte_as = sc_a.transform(Xte_ab)
        pca_a = PCA(n_components=min(N_COMP, Xtr_as.shape[0] - 1, Xtr_as.shape[1]))
        Xtr_ap = pca_a.fit_transform(Xtr_as); Xte_ap = pca_a.transform(Xte_as)
        probe_a = Ridge(alpha=ALPHA); probe_a.fit(Xtr_ap, ytr)
        pred_a = probe_a.predict(Xte_ap)
        try:
            ablated_auc = roc_auc_score((yte > 0.5).astype(int), pred_a)
        except: ablated_auc = 0.5

        delta = baseline_auc - ablated_auc
        print(f"    baseline AUC: {baseline_auc:.3f}")
        print(f"    after mass-mean ablation: {ablated_auc:.3f}")
        print(f"    delta: {delta:+.3f}")
        results[f'{model}_{config}_{scenario}'] = {
            'baseline_auc': float(baseline_auc),
            'ablated_auc': float(ablated_auc),
            'delta': float(delta),
            'label': label,
        }
    return results


# ============================================================
# Section 4: Per-round AUC + bootstrap CIs
# ============================================================

def section4_per_round_auc(*, trust_legacy_pt: bool = False):
    print("\n=== Section 4: Per-round AUC + bootstrap CIs ===")
    targets = [
        ('gemma7b', 'no_tom', 'ultimatum_bluff'),
        ('gemma7b', 'no_tom', 'info_withholding'),
        ('gemma7b', 'tom', 'ultimatum_bluff'),
        ('gemma7b', 'tom', 'info_withholding'),
    ]
    results = {}
    for model, config, scenario in targets:
        key = (model, config, scenario)
        if key not in RUNS:
            continue
        print(f"\n  {model}/{config}/{scenario}")
        path = RUNS[key]
        fp = download_activation_input(HF_REPO, path)
        data = load_activation_input(fp, trust_legacy_pt=trust_legacy_pt)
        md = data.get('metadata', [])
        labels = data['labels']
        acts = data['activations']
        n = len(labels['gm_labels'])
        round_nums = [md[i].get('round_num') if i < len(md) else None for i in range(n)]
        keep = np.array([_is_negotiation(md[i]) if i < len(md) else True
                         for i in range(n)], dtype=bool)
        round_nums_kept = np.array([r for i, r in enumerate(round_nums) if keep[i]])
        y = np.array(labels['gm_labels'])[keep]
        layer_keys = [k for k in acts if isinstance(k, int)]
        layer = 14 if 14 in layer_keys else sorted(layer_keys)[len(layer_keys) // 2]
        X = acts[layer].float().numpy()[keep]

        per_round = {}
        for rn in sorted(set(round_nums_kept.tolist())):
            mask = round_nums_kept == rn
            if mask.sum() < 30:
                continue
            X_r = X[mask]; y_r = y[mask]
            yb = (y_r > 0.5)
            if yb.sum() < 5 or (~yb).sum() < 5:
                continue
            # Compute AUC + bootstrap CI
            point_aucs = []
            for s in range(200):
                rng = np.random.RandomState(SEED + s)
                idx = rng.choice(len(y_r), size=len(y_r), replace=True)
                Xb = X_r[idx]; yb_r = y_r[idx]; yb_bin = (yb_r > 0.5)
                if yb_bin.sum() < 2 or (~yb_bin).sum() < 2:
                    continue
                try:
                    Xtr, Xte, ytr, yte = train_test_split(
                        Xb, yb_r, test_size=0.2, random_state=SEED+s,
                        stratify=yb_bin.astype(int))
                    sc_ = StandardScaler(); Xtr_s = sc_.fit_transform(Xtr); Xte_s = sc_.transform(Xte)
                    pca = PCA(n_components=min(N_COMP, Xtr_s.shape[0]-1, Xtr_s.shape[1]))
                    Xtr_p = pca.fit_transform(Xtr_s); Xte_p = pca.transform(Xte_s)
                    probe = Ridge(alpha=ALPHA); probe.fit(Xtr_p, ytr)
                    pred = probe.predict(Xte_p)
                    auc = roc_auc_score((yte > 0.5).astype(int), pred)
                    point_aucs.append(auc)
                except Exception:
                    continue
            if len(point_aucs) < 20:
                continue
            arr = np.array(point_aucs)
            per_round[int(rn)] = {
                'mean_auc': float(arr.mean()),
                'ci_low': float(np.percentile(arr, 2.5)),
                'ci_high': float(np.percentile(arr, 97.5)),
                'n_samples': int(mask.sum()),
                'deception_rate': float(yb.mean()),
            }
            print(f"    round {rn}: AUC = {arr.mean():.3f} "
                  f"[{np.percentile(arr, 2.5):.3f}, {np.percentile(arr, 97.5):.3f}], "
                  f"n={int(mask.sum())}, dec_rate={float(yb.mean()):.2f}")
        results[f'{model}_{config}_{scenario}'] = per_round
    return results


# ============================================================
# Section 5: Baselines (shuffled labels, random features)
# ============================================================

def section5_baselines(*, trust_legacy_pt: bool = False):
    print("\n=== Section 5: Baselines (sanity controls) ===")
    target = ('gemma7b', 'no_tom', 'ultimatum_bluff')
    d = load_run(target, trust_legacy_pt=trust_legacy_pt)
    X, y = d['X'], d['y']
    yb = (y > 0.5).astype(int)

    # Real probe AUC
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=yb)
    sc_ = StandardScaler(); Xtr_s = sc_.fit_transform(Xtr); Xte_s = sc_.transform(Xte)
    pca = PCA(n_components=min(N_COMP, Xtr_s.shape[0]-1, Xtr_s.shape[1]))
    Xtr_p = pca.fit_transform(Xtr_s); Xte_p = pca.transform(Xte_s)
    probe = Ridge(alpha=ALPHA); probe.fit(Xtr_p, ytr)
    pred = probe.predict(Xte_p)
    real_auc = roc_auc_score((yte > 0.5).astype(int), pred)

    # Shuffled labels (100 perms)
    rng = np.random.RandomState(SEED)
    shuffled = []
    for s in range(100):
        ytr_p = rng.permutation(ytr)
        probe_s = Ridge(alpha=ALPHA); probe_s.fit(Xtr_p, ytr_p)
        pred_s = probe_s.predict(Xte_p)
        try:
            shuffled.append(roc_auc_score((yte > 0.5).astype(int), pred_s))
        except: continue
    shuffled = np.array(shuffled)

    # Random features
    rand_X = rng.randn(*X.shape) * X.std()
    Xtr_r, Xte_r, _, _ = train_test_split(rand_X, y, test_size=0.2, random_state=SEED, stratify=yb)
    sc_r = StandardScaler(); Xtr_rs = sc_r.fit_transform(Xtr_r); Xte_rs = sc_r.transform(Xte_r)
    pca_r = PCA(n_components=min(N_COMP, Xtr_rs.shape[0]-1, Xtr_rs.shape[1]))
    Xtr_rp = pca_r.fit_transform(Xtr_rs); Xte_rp = pca_r.transform(Xte_rs)
    probe_r = Ridge(alpha=ALPHA); probe_r.fit(Xtr_rp, ytr)
    pred_r = probe_r.predict(Xte_rp)
    rand_auc = roc_auc_score((yte > 0.5).astype(int), pred_r)

    # Random directions ceiling (1000 unit vectors, project, threshold)
    rand_dir_aucs = []
    for s in range(1000):
        rng2 = np.random.RandomState(s)
        v = rng2.randn(X.shape[1])
        v = v / np.linalg.norm(v)
        proj = X @ v
        try:
            auc = roc_auc_score(yb, proj)
            auc = max(auc, 1.0 - auc)  # symmetric
            rand_dir_aucs.append(auc)
        except: continue
    rand_dir_99 = float(np.percentile(rand_dir_aucs, 99))

    # Majority class
    maj = max(yb.mean(), 1 - yb.mean())

    result = {
        'real_probe_auc': float(real_auc),
        'shuffled_mean': float(shuffled.mean()),
        'shuffled_max': float(shuffled.max()),
        'shuffled_std': float(shuffled.std()),
        'random_features_auc': float(rand_auc),
        'random_directions_99pct': rand_dir_99,
        'majority_class': float(maj),
    }
    print(f"  real probe AUC:     {real_auc:.3f}")
    print(f"  shuffled labels:    {shuffled.mean():.3f} ± {shuffled.std():.3f} (max: {shuffled.max():.3f})")
    print(f"  random features:    {rand_auc:.3f}")
    print(f"  random dirs 99%ile: {rand_dir_99:.3f}")
    print(f"  majority class:     {maj:.3f}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_legacy_trust_argument(parser)
    args = parser.parse_args(argv)
    out_dir = Path('experiment_results/paper_numbers')
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_results['cosine_similarity'] = section1_cosine_similarities(
        trust_legacy_pt=args.trust_legacy_pt
    )
    all_results['cross_scenario_transfer'] = section2_cross_scenario_transfer(
        trust_legacy_pt=args.trust_legacy_pt
    )
    all_results['probe_faithfulness'] = section3_probe_faithfulness(
        trust_legacy_pt=args.trust_legacy_pt
    )
    all_results['per_round_auc'] = section4_per_round_auc(
        trust_legacy_pt=args.trust_legacy_pt
    )
    all_results['baselines'] = section5_baselines(
        trust_legacy_pt=args.trust_legacy_pt
    )

    out_path = out_dir / 'paper_numbers.json'
    with out_path.open('w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nFull results saved to {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
