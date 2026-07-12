#!/usr/bin/env python3
"""Finish remaining Tier A sections (cross-arch transfer + bootstrap CIs).

Created 2026-04-25 to replace the slow run that bottlenecked on
per-layer search + 500 bootstraps. This version uses the best-layer
table from the prior delta CSV and runs 200 bootstraps.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HF_REPO = 'sycorpia/multiagent-lab-data'
ALPHA = 100.0
N_COMP = 30
SEED = 42
N_BOOT = 200

# Best layer per (model, config, scenario) — taken from audit_qc_delta.csv
BEST_LAYERS = {
    ('gemma7b', 'tom', 'ultimatum_bluff'): 27,
    ('gemma7b', 'tom', 'alliance_betrayal'): 14,
    ('gemma7b', 'tom', 'info_withholding'): 14,
    ('gemma7b', 'no_tom', 'ultimatum_bluff'): 20,
    ('gemma7b', 'no_tom', 'alliance_betrayal'): 27,
    ('gemma7b', 'no_tom', 'info_withholding'): 27,
    ('llama31_8b', 'tom', 'ultimatum_bluff'): 4,
    ('llama31_8b', 'tom', 'alliance_betrayal'): 31,
    ('llama31_8b', 'tom', 'info_withholding'): 31,
    ('mistral7b', 'tom', 'ultimatum_bluff'): 31,
    ('mistral7b', 'tom', 'alliance_betrayal'): 31,
    ('mistral7b', 'tom', 'info_withholding'): 28,
}

RUNS = {
    ('gemma7b', 'tom', 'ultimatum_bluff'):
        'gemma7b/ultimatum_bluff/tom_20260211/activations.pt',
    ('gemma7b', 'tom', 'alliance_betrayal'):
        'gemma7b/alliance_betrayal/tom_20260210/activations.pt',
    ('gemma7b', 'tom', 'info_withholding'):
        'gemma7b/info_withholding/tom_20260211/activations.pt',
    ('gemma7b', 'no_tom', 'ultimatum_bluff'):
        'gemma7b/ultimatum_bluff/no_tom_20260412_042958/activations.pt',
    ('gemma7b', 'no_tom', 'alliance_betrayal'):
        'gemma7b/alliance_betrayal/no_tom_20260412_043012/activations.pt',
    ('gemma7b', 'no_tom', 'info_withholding'):
        'gemma7b/info_withholding/no_tom_20260412_043021/activations.pt',
    ('llama31_8b', 'tom', 'ultimatum_bluff'):
        'llama31_8b/ultimatum_bluff/tom_20260414_192608/activations.pt',
    ('llama31_8b', 'tom', 'alliance_betrayal'):
        'llama31_8b/alliance_betrayal/tom_20260412_043955/activations.pt',
    ('llama31_8b', 'tom', 'info_withholding'):
        'llama31_8b/info_withholding/tom_20260412_051758/activations.pt',
    ('mistral7b', 'tom', 'ultimatum_bluff'):
        'ultimatum_bluff/activations_mistral7b_ultimatum_bluff_both_20260412_194428.pt',
    ('mistral7b', 'tom', 'alliance_betrayal'):
        'alliance_betrayal/activations_mistral7b_alliance_betrayal_both_20260413_152248.pt',
    ('mistral7b', 'tom', 'info_withholding'):
        'info_withholding/activations_mistral7b_info_withholding_both_20260413_103456.pt',
}

OUT_DIR = Path('experiment_results/paper_numbers')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _is_negotiation(md):
    st = md.get('sample_type')
    if st is not None:
        return st == 'negotiation'
    rn = md.get('round_num')
    return rn is None or rn >= 0


def load_layer(model: str, config: str, scenario: str, layer: int):
    """Load only the requested layer for speed."""
    key = (model, config, scenario)
    if key not in RUNS:
        return None
    path = RUNS[key]
    fp = hf_hub_download(HF_REPO, path, repo_type='dataset')
    data = torch.load(fp, weights_only=False)
    md = data.get('metadata', [])
    labels = data['labels']
    acts = data['activations']
    y_all = np.array(labels['gm_labels'])
    n = len(y_all)
    keep = np.array([_is_negotiation(md[i]) if i < len(md) else True
                     for i in range(n)], dtype=bool)
    if layer not in acts:
        layer_keys = sorted([k for k in acts if isinstance(k, int)])
        layer = layer_keys[len(layer_keys) // 2] if layer_keys else None
        if layer is None:
            return None
    X = acts[layer].float().numpy()[keep]
    y = y_all[keep]
    return X, y, layer


def fit_probe_auc_fast(X, y, seed=SEED):
    yb = (y > 0.5).astype(int)
    if yb.sum() < 2 or (~yb.astype(bool)).sum() < 2:
        return None
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=yb
        )
    except Exception:
        return None
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    n_comp = min(N_COMP, Xtr_s.shape[0]-1, Xtr_s.shape[1])
    if n_comp < 1:
        return None
    pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=seed)
    Xtr_p = pca.fit_transform(Xtr_s); Xte_p = pca.transform(Xte_s)
    probe = Ridge(alpha=ALPHA); probe.fit(Xtr_p, ytr)
    pred = probe.predict(Xte_p)
    yte_b = (yte > 0.5).astype(int)
    if len(np.unique(yte_b)) < 2:
        return None
    try:
        return float(roc_auc_score(yte_b, pred))
    except Exception:
        return None


def mass_mean_direction(X, y_binary):
    if y_binary.sum() < 2 or (~y_binary).sum() < 2:
        return None
    d = X[y_binary].mean(0) - X[~y_binary].mean(0)
    n = np.linalg.norm(d)
    if n < 1e-8:
        return None
    return d / n


# --- Section 2: cross-arch direction transfer (info_withholding) ---

def cross_arch_transfer():
    print("\n=== Cross-architecture probe-direction transfer ===")
    scenario = 'info_withholding'
    sources = []
    best_per_config = {
        ('gemma7b', 'tom'): 14,
        ('gemma7b', 'no_tom'): 27,
        ('llama31_8b', 'tom'): 31,
        ('mistral7b', 'tom'): 28,
    }
    for (model, config), layer in best_per_config.items():
        print(f"  loading {model}/{config}/{scenario} L{layer}")
        loaded = load_layer(model, config, scenario, layer)
        if loaded is None:
            continue
        X, y, layer_used = loaded
        yb = (y > 0.5)
        d = mass_mean_direction(X, yb)
        if d is None:
            continue
        sources.append({'tag': f'{model}/{config}', 'X': X, 'y': y,
                        'direction': d, 'd_dim': X.shape[1]})

    matrix = {}
    for src in sources:
        for tgt in sources:
            key = f"{src['tag']}->{tgt['tag']}"
            if src['d_dim'] != tgt['d_dim']:
                matrix[key] = None
                continue
            proj = tgt['X'] @ src['direction']
            yb = (tgt['y'] > 0.5).astype(int)
            try:
                auc = roc_auc_score(yb, proj)
                auc = max(auc, 1.0 - auc)
                matrix[key] = float(auc)
            except Exception:
                matrix[key] = None

    print(f"\n  Cross-arch direction-transfer AUC (scenario={scenario}, symmetric):")
    print("  src \\ tgt".ljust(28) + "".join(s['tag'].rjust(20) for s in sources))
    for src in sources:
        row = "  " + src['tag'].ljust(26)
        for tgt in sources:
            v = matrix.get(f"{src['tag']}->{tgt['tag']}")
            if v is None:
                row += "n/a".rjust(20)
            else:
                row += f"{v:.3f}".rjust(20)
        print(row)

    out = OUT_DIR / 'cross_arch_direction_transfer.json'
    out.write_text(json.dumps({'scenario': scenario, 'matrix': matrix,
                               'tags': [s['tag'] for s in sources]},
                              indent=2, default=str))
    print(f"  Saved: {out}")
    return matrix


# --- Section 3: bootstrap CIs ---

def bootstrap_cis():
    print(f"\n=== Bootstrap 95% CIs (n_boot={N_BOOT}) ===")
    results = {}
    targets = list(RUNS.keys())
    for i, key in enumerate(targets):
        model, config, scenario = key
        layer = BEST_LAYERS.get(key)
        tag = f'{model}_{config}_{scenario}'
        print(f"\n  [{i+1}/{len(targets)}] {tag} L{layer}")
        loaded = load_layer(model, config, scenario, layer)
        if loaded is None:
            continue
        X, y, layer_used = loaded
        # Point AUC
        point_auc = fit_probe_auc_fast(X, y)
        if point_auc is None:
            continue
        print(f"    point AUC: {point_auc:.3f}")
        rng = np.random.RandomState(SEED)
        aucs = []
        t0 = time.time()
        for s in range(N_BOOT):
            idx = rng.choice(len(y), size=len(y), replace=True)
            auc = fit_probe_auc_fast(X[idx], y[idx], seed=SEED + s)
            if auc is not None:
                aucs.append(auc)
            if s % 50 == 49:
                elapsed = time.time() - t0
                rate = (s + 1) / elapsed
                remaining = (N_BOOT - s - 1) / rate
                print(f"    bootstrap {s+1}/{N_BOOT} ({elapsed:.0f}s, eta {remaining:.0f}s)")
        if not aucs:
            continue
        arr = np.array(aucs)
        results[tag] = {
            'best_layer': layer_used,
            'point_auc': float(point_auc),
            'bootstrap_mean': float(arr.mean()),
            'bootstrap_ci_low': float(np.percentile(arr, 2.5)),
            'bootstrap_ci_high': float(np.percentile(arr, 97.5)),
            'n_boot_succeeded': int(len(arr)),
            'wall_time_sec': float(time.time() - t0),
        }
        print(f"    bootstrap mean={arr.mean():.3f} "
              f"95% CI [{np.percentile(arr, 2.5):.3f}, {np.percentile(arr, 97.5):.3f}] "
              f"({time.time()-t0:.0f}s)")
    out = OUT_DIR / 'bootstrap_cis.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Saved: {out}")
    return results


def main():
    t0 = time.time()
    try:
        cross_arch_transfer()
    except Exception as e:
        import traceback; traceback.print_exc()
    try:
        bootstrap_cis()
    except Exception as e:
        import traceback; traceback.print_exc()
    print(f"\nTotal: {time.time()-t0:.0f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
