#!/usr/bin/env python3
"""Dyadic-leakage investigation — phases I through VI.

Created 2026-04-22 following the plan in DATA_QUALITY_FIX_PLAN.md and the
investigation plan shared with the user. Goal: determine whether dyadic
AUC >> d-prime ceiling is real multi-dimensional signal or residual
train/test leakage, and if leakage, identify the mechanism.

The primary target is Llama-3.1-8B info_withholding (d-prime 0.95,
AUC 1.000). We then replicate findings on the other 1.000-AUC runs.

Usage:
    python scripts/investigate_dyadic_leakage.py
    python scripts/investigate_dyadic_leakage.py --target llama_iw --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

HF_REPO = 'sycorpia/multiagent-lab-data'
ALPHA = 100.0
N_COMP = 30
SEED = 42

TARGETS = {
    'llama_iw': 'llama31_8b/info_withholding/tom_20260412_051758/activations.pt',
    'llama_ab': 'llama31_8b/alliance_betrayal/tom_20260412_043955/activations.pt',
    'llama_ub': 'llama31_8b/ultimatum_bluff/tom_20260414_192608/activations.pt',
    'gemma_notom_ub': 'gemma7b/ultimatum_bluff/no_tom_20260412_042958/activations.pt',
    'mistral_ab': 'alliance_betrayal/activations_mistral7b_alliance_betrayal_both_20260413_152248.pt',
}


# ---------- shared helpers ----------

def _is_negotiation(md_entry: Dict[str, Any]) -> bool:
    st = md_entry.get('sample_type')
    if st is not None:
        return st == 'negotiation'
    rn = md_entry.get('round_num')
    if rn is None:
        return True
    return rn >= 0


def load_target(alias: str, layer: Optional[int] = None) -> Dict[str, Any]:
    """Load an activations.pt, filter to negotiation samples, and return the
    pieces we need for the dyadic analysis (best-layer X, y, counterpart_idxs,
    trial_ids, scenario params/metadata)."""
    path = TARGETS[alias]
    print(f"Loading {path} ...")
    local = hf_hub_download(HF_REPO, path, repo_type='dataset')
    data = torch.load(local, weights_only=False)

    metadata = data.get('metadata', [])
    labels = data['labels']
    acts = data['activations']
    y_all = np.array(labels['gm_labels'])
    cp_all = labels.get('counterpart_idxs') or []
    trial_all = labels.get('trial_ids') or []

    n = len(y_all)
    keep = np.array([_is_negotiation(metadata[i]) if i < len(metadata) else True
                     for i in range(n)], dtype=bool)
    y = y_all[keep]
    cp = [cp_all[i] for i in range(n) if keep[i]] if cp_all else [None]*int(keep.sum())
    trial = [trial_all[i] for i in range(n) if keep[i]] if trial_all else [0]*int(keep.sum())
    meta = [metadata[i] for i in range(n) if keep[i]] if metadata else []

    layer_keys = [k for k in acts if isinstance(k, int) or (isinstance(k, str) and not k.endswith('_mean'))]
    if layer is None:
        # pick last layer (mid-to-late typically best for deception)
        layer = sorted(layer_keys, key=lambda x: int(x) if str(x).isdigit() else -1)[-1]
    X = acts[layer].float().numpy()[keep]

    print(f"  After filter: N={len(y)}, d={X.shape[1]}, layer={layer}")
    return dict(X=X, y=y, cp=cp, trial=trial, meta=meta, layer=layer, path=path)


def build_pairs(cp: List[Any], y: np.ndarray):
    """Dedup pairs the same way analyze_dyadic_pairs does."""
    n = len(y)
    pairs = []
    seen = set()
    for i in range(n):
        c = cp[i]
        if c is None:
            continue
        try:
            c_int = int(c)
        except (TypeError, ValueError):
            continue
        if c_int == i or not (0 <= c_int < n):
            continue
        key = (min(i, c_int), max(i, c_int))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((i, c_int))
    return pairs


def stack_deceiver_victim(X, y, pairs):
    dec_idx, vic_idx, asym = [], [], []
    for i, j in pairs:
        if y[i] > y[j]:
            dec_idx.append(i); vic_idx.append(j)
        else:
            dec_idx.append(j); vic_idx.append(i)
        asym.append(abs(y[i] - y[j]))
    return np.asarray(dec_idx), np.asarray(vic_idx), np.asarray(asym)


def compute_d_prime(X_dec, X_vic):
    dec_mean = X_dec.mean(axis=0)
    vic_mean = X_vic.mean(axis=0)
    direction = dec_mean - vic_mean
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return 0.0, direction
    d = direction / norm
    dec_proj = X_dec @ d
    vic_proj = X_vic @ d
    dp = (dec_proj.mean() - vic_proj.mean()) / (
        np.sqrt(0.5 * (dec_proj.var() + vic_proj.var())) + 1e-8
    )
    return float(dp), d


def fit_probe_group_split(pair_X, pair_y, groups, alpha=ALPHA, n_comp=N_COMP, seed=SEED):
    """Group-aware train/test split, StandardScaler + PCA + Ridge."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(pair_X, pair_y, groups=groups))
    Xtr, Xte = pair_X[tr], pair_X[te]
    ytr, yte = pair_y[tr], pair_y[te]

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    nc = min(n_comp, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
    pca = PCA(n_components=nc)
    Xtr_p = pca.fit_transform(Xtr_s)
    Xte_p = pca.transform(Xte_s)

    probe = Ridge(alpha=alpha)
    probe.fit(Xtr_p, ytr)
    pred = probe.predict(Xte_p)

    try:
        auc = roc_auc_score((yte > 0.5).astype(int), pred)
    except ValueError:
        auc = 0.5
    return {
        'auc': float(auc),
        'probe': probe,
        'pca': pca,
        'scaler': sc,
        'train_idx': tr,
        'test_idx': te,
        'n_train': int(len(tr)),
        'n_test': int(len(te)),
    }


# ---------- phases ----------

def phase1_reproduce(d):
    """Reproduce the current pair-aware AUC twice to confirm stability."""
    print("\n=== Phase I: reproducibility ===")
    pairs = build_pairs(d['cp'], d['y'])
    dec_idx, vic_idx, asym = stack_deceiver_victim(d['X'], d['y'], pairs)
    X_dec, X_vic = d['X'][dec_idx], d['X'][vic_idx]
    pair_X = np.vstack([X_dec, X_vic])
    pair_y = np.array([1.0]*len(X_dec) + [0.0]*len(X_vic))
    pair_groups = np.concatenate([np.arange(len(X_dec)), np.arange(len(X_vic))])

    r1 = fit_probe_group_split(pair_X, pair_y, pair_groups, seed=SEED)
    r2 = fit_probe_group_split(pair_X, pair_y, pair_groups, seed=SEED)
    print(f"  Run 1 AUC: {r1['auc']:.4f}")
    print(f"  Run 2 AUC: {r2['auc']:.4f}")
    print(f"  Reproducible: {r1['auc'] == r2['auc']}")

    d_prime, direction = compute_d_prime(X_dec, X_vic)
    print(f"  d-prime: {d_prime:.3f}")
    print(f"  1-D ceiling AUC (from d-prime): "
          f"~{0.5 + (d_prime / np.sqrt(2*np.pi)) / 2:.3f} (rough)")

    return dict(
        pair_X=pair_X, pair_y=pair_y, pair_groups=pair_groups,
        X_dec=X_dec, X_vic=X_vic, dec_idx=dec_idx, vic_idx=vic_idx,
        pairs=pairs, d_prime=d_prime, direction_1d=direction,
        auc=r1['auc'],
    )


def phase2_controls(state, d):
    """Null + random-features + 1-D ceiling."""
    print("\n=== Phase II: null controls and 1-D ceiling ===")
    pair_X, pair_y, pair_groups = state['pair_X'], state['pair_y'], state['pair_groups']
    rng = np.random.RandomState(SEED)

    # Control 1: shuffled labels
    shuffled_aucs = []
    for i in range(50):
        perm = rng.permutation(len(pair_y))
        r = fit_probe_group_split(pair_X, pair_y[perm], pair_groups, seed=SEED+i)
        shuffled_aucs.append(r['auc'])
    shuffled_aucs = np.array(shuffled_aucs)
    print(f"  Label-shuffle null: mean AUC = {shuffled_aucs.mean():.3f}, "
          f"95% CI [{np.percentile(shuffled_aucs, 2.5):.3f}, "
          f"{np.percentile(shuffled_aucs, 97.5):.3f}]")

    # Control 2: random features
    random_X = rng.randn(*pair_X.shape) * pair_X.std()
    r = fit_probe_group_split(random_X, pair_y, pair_groups, seed=SEED)
    print(f"  Random-features AUC: {r['auc']:.3f}")

    # Control 3: 1-D projection ceiling (use the mean-diff direction, fit a
    # logistic on the single dimension)
    proj = pair_X @ state['direction_1d']
    # use group-aware split too
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(pair_X, pair_y, groups=pair_groups))
    proj_tr = proj[tr].reshape(-1, 1)
    proj_te = proj[te].reshape(-1, 1)
    probe = Ridge(alpha=ALPHA)
    probe.fit(proj_tr, pair_y[tr])
    pred = probe.predict(proj_te)
    try:
        auc_1d = roc_auc_score((pair_y[te] > 0.5).astype(int), pred)
    except ValueError:
        auc_1d = 0.5
    print(f"  1-D projection AUC: {auc_1d:.3f}")

    gap = state['auc'] - auc_1d
    print(f"  Gap (full - 1D): {gap:+.3f} "
          f"— {'large, multi-D structure' if gap > 0.05 else 'small, 1-D explains it'}")

    return dict(null_mean=float(shuffled_aucs.mean()),
                null_ci=(float(np.percentile(shuffled_aucs, 2.5)),
                         float(np.percentile(shuffled_aucs, 97.5))),
                random_features_auc=float(r['auc']),
                auc_1d=float(auc_1d),
                multi_d_gap=float(gap))


def phase3_hypotheses(state, d):
    """Test H1 (trial_id split), H5 (PCA leak via global scope), H3 (cp/idx shortcut),
    H4 (round-num), H6 (agent-id). We skip H2 scenario-param unless we have
    the per-sample params available (they are in metadata on some runs)."""
    print("\n=== Phase III: leakage hypotheses ===")
    pair_X, pair_y = state['pair_X'], state['pair_y']
    dec_idx, vic_idx = state['dec_idx'], state['vic_idx']

    # H1: group by trial_id instead of pair_id
    trial = d['trial']
    pair_trial_ids = np.array([trial[i] for i in np.concatenate([dec_idx, vic_idx])])
    if len(np.unique(pair_trial_ids)) >= 2:
        r = fit_probe_group_split(pair_X, pair_y, pair_trial_ids, seed=SEED)
        print(f"  H1 (trial_id split): AUC = {r['auc']:.3f}  "
              f"(pair_id was {state['auc']:.3f})")
        h1_auc = r['auc']
    else:
        print(f"  H1 (trial_id split): skipped (only {len(np.unique(pair_trial_ids))} unique trials)")
        h1_auc = None

    # H3: counterpart-index shortcut. Check joint distribution of
    # (cp > self) vs deceiver flag.
    cp_greater = []
    for i, j in state['pairs']:
        # pair is (i, j); deceiver is whichever has higher y
        cp_i_greater = d['cp'][i] is not None and int(d['cp'][i]) > i
        # The "deceiver flag" convention: 1 if index i is the deceiver
        deceiver_is_i = d['y'][i] > d['y'][j]
        cp_greater.append((cp_i_greater, deceiver_is_i))
    arr = np.array(cp_greater)
    if len(arr):
        p_match = (arr[:, 0] == arr[:, 1]).mean()
        print(f"  H3 (cp-index shortcut): P(cp_greater == deceiver_is_i) = {p_match:.2f} "
              f"(0.5 = no shortcut)")
    else:
        p_match = None

    # H4: round-num signature. Within one round only, is the probe still
    # near-perfect?
    round_nums_per_sample = np.array(
        [d['meta'][i].get('round_num', 0) if i < len(d['meta']) else 0
         for i in np.concatenate([dec_idx, vic_idx])]
    )
    by_round_aucs = {}
    for r_num in sorted(set(round_nums_per_sample.tolist())):
        mask = round_nums_per_sample == r_num
        if mask.sum() < 20:
            continue
        sub_X = pair_X[mask]; sub_y = pair_y[mask]
        sub_groups = np.concatenate([np.arange(len(dec_idx)),
                                     np.arange(len(vic_idx))])[mask]
        if len(np.unique(sub_groups)) < 4 or len(np.unique(sub_y)) < 2:
            continue
        try:
            res = fit_probe_group_split(sub_X, sub_y, sub_groups, seed=SEED)
            by_round_aucs[int(r_num)] = res['auc']
        except Exception:
            continue
    print(f"  H4 (within-round AUC): {by_round_aucs}")

    # H6: agent identity. Scramble pair deceiver/victim within each trial
    # and recompute. If identity was the shortcut, AUC collapses.
    rng = np.random.RandomState(SEED)
    scrambled_dec = dec_idx.copy(); scrambled_vic = vic_idx.copy()
    for k in range(len(scrambled_dec)):
        if rng.random() < 0.5:
            scrambled_dec[k], scrambled_vic[k] = scrambled_vic[k], scrambled_dec[k]
    scrambled_pair_X = np.vstack([d['X'][scrambled_dec], d['X'][scrambled_vic]])
    scrambled_pair_y = pair_y  # same labels — but now 50% of "deceivers" are randomly assigned
    # This is effectively a permutation control; should collapse to null.
    scrambled_groups = np.concatenate([np.arange(len(scrambled_dec)),
                                       np.arange(len(scrambled_vic))])
    r = fit_probe_group_split(scrambled_pair_X, scrambled_pair_y, scrambled_groups, seed=SEED)
    print(f"  H6 (scrambled deceiver/victim within-pair): AUC = {r['auc']:.3f} "
          f"(should be ~0.5 if labels are carrying the signal)")

    return dict(h1_trial_id_auc=h1_auc,
                h3_cp_shortcut_rate=float(p_match) if p_match is not None else None,
                h4_within_round_aucs=by_round_aucs,
                h6_scrambled_auc=float(r['auc']))


def phase4_introspect(state, d):
    """What does the probe actually use?"""
    print("\n=== Phase IV: probe introspection ===")
    pair_X, pair_y, pair_groups = state['pair_X'], state['pair_y'], state['pair_groups']

    # Fit once more to get the full probe + pca
    res = fit_probe_group_split(pair_X, pair_y, pair_groups, seed=SEED)
    probe, pca, sc = res['probe'], res['pca'], res['scaler']

    # Reconstruct w_act (coefficient in original activation space, ignoring
    # scaling — this is fine for direction comparison)
    w_pca = probe.coef_
    w_scaled = pca.components_.T @ w_pca  # (d,)
    # undo scaling
    w_act = w_scaled / (sc.scale_ + 1e-12)
    w_act = w_act / (np.linalg.norm(w_act) + 1e-12)

    # Cosine similarity with the mean-diff direction
    cos = float(np.dot(w_act, state['direction_1d']))
    print(f"  cos(probe_direction, mean_diff_direction) = {cos:.3f}")
    print(f"  -> {'aligns with d-prime direction (1-D signal)' if abs(cos) > 0.7 else 'uses other dimensions (multi-D or noise)'}")

    # Seed stability: fit across 5 seeds, check AUC range and direction drift
    aucs = []
    directions = []
    for s in range(5):
        r = fit_probe_group_split(pair_X, pair_y, pair_groups, seed=SEED + s)
        aucs.append(r['auc'])
        wp = r['probe'].coef_
        ws = r['pca'].components_.T @ wp
        wa = ws / (r['scaler'].scale_ + 1e-12)
        wa = wa / (np.linalg.norm(wa) + 1e-12)
        directions.append(wa)
    aucs = np.array(aucs)
    # Pairwise cos between directions across seeds
    cos_mat = np.array([[abs(float(np.dot(directions[i], directions[j])))
                         for j in range(len(directions))]
                        for i in range(len(directions))])
    stability = float(cos_mat[np.triu_indices(len(directions), k=1)].mean())
    print(f"  AUC across 5 seeds: min={aucs.min():.3f}, max={aucs.max():.3f}, "
          f"std={aucs.std():.3f}")
    print(f"  Direction stability (avg |cos|): {stability:.3f}  "
          f"— {'stable (likely real)' if stability > 0.7 else 'unstable (likely overfitting)'}")

    return dict(cos_probe_vs_meandiff=cos,
                auc_seed_min=float(aucs.min()),
                auc_seed_max=float(aucs.max()),
                auc_seed_std=float(aucs.std()),
                direction_stability=stability)


def phase5_cross_scenario(target_alias, all_targets, layer=None):
    """Train dyadic probe on target, test on other scenarios (same model)."""
    print("\n=== Phase V: cross-scenario transfer ===")
    # Determine model family
    model_family = target_alias.split('_')[0]

    # Find sibling scenarios for the same model
    sibling_aliases = [
        a for a in all_targets.keys()
        if a != target_alias and a.split('_')[0] == model_family
    ]
    if not sibling_aliases:
        print(f"  No sibling scenarios for {target_alias}; skipping.")
        return {}

    src = load_target(target_alias, layer=layer)
    src_pairs = build_pairs(src['cp'], src['y'])
    if len(src_pairs) < 10:
        print(f"  Too few pairs on source; skipping.")
        return {}
    src_dec_idx, src_vic_idx, _ = stack_deceiver_victim(src['X'], src['y'], src_pairs)
    src_pair_X = np.vstack([src['X'][src_dec_idx], src['X'][src_vic_idx]])
    src_pair_y = np.array([1.0]*len(src_dec_idx) + [0.0]*len(src_vic_idx))

    # Fit on full source
    sc = StandardScaler(); X_s = sc.fit_transform(src_pair_X)
    pca = PCA(n_components=min(N_COMP, X_s.shape[0]-1, X_s.shape[1]))
    X_p = pca.fit_transform(X_s)
    probe = Ridge(alpha=ALPHA); probe.fit(X_p, src_pair_y)

    transfer_aucs = {}
    for tgt_alias in sibling_aliases:
        tgt = load_target(tgt_alias, layer=layer)
        tgt_pairs = build_pairs(tgt['cp'], tgt['y'])
        if len(tgt_pairs) < 10:
            transfer_aucs[tgt_alias] = None
            continue
        t_dec, t_vic, _ = stack_deceiver_victim(tgt['X'], tgt['y'], tgt_pairs)
        t_X = np.vstack([tgt['X'][t_dec], tgt['X'][t_vic]])
        t_y = np.array([1.0]*len(t_dec) + [0.0]*len(t_vic))
        # Match dimensionality
        if t_X.shape[1] != src_pair_X.shape[1]:
            print(f"  Dim mismatch {t_X.shape[1]} != {src_pair_X.shape[1]}; skip")
            transfer_aucs[tgt_alias] = None
            continue
        t_Xs = sc.transform(t_X); t_Xp = pca.transform(t_Xs)
        pred = probe.predict(t_Xp)
        try:
            auc = roc_auc_score((t_y > 0.5).astype(int), pred)
        except ValueError:
            auc = 0.5
        transfer_aucs[tgt_alias] = float(auc)
        print(f"  {target_alias} -> {tgt_alias}: AUC = {auc:.3f}  "
              f"— {'real signal' if auc > 0.65 else 'likely scenario-specific'}")
    return transfer_aucs


def phase6_bootstrap(state):
    """Bootstrap CI on the dyadic AUC."""
    print("\n=== Phase VI: bootstrap CI ===")
    pair_X, pair_y, pair_groups = state['pair_X'], state['pair_y'], state['pair_groups']
    rng = np.random.RandomState(SEED)
    aucs = []
    for i in range(200):
        # Resample pairs with replacement
        n_pairs = len(pair_y) // 2
        sample_pair_ids = rng.choice(n_pairs, size=n_pairs, replace=True)
        # Build resampled pair_X and pair_y — half deceiver, half victim
        dec_mask = sample_pair_ids
        vic_mask = sample_pair_ids + n_pairs
        resampled_X = np.concatenate([pair_X[dec_mask], pair_X[vic_mask]])
        resampled_y = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)])
        resampled_groups = np.concatenate([sample_pair_ids, sample_pair_ids])
        try:
            r = fit_probe_group_split(resampled_X, resampled_y, resampled_groups, seed=SEED+i)
            aucs.append(r['auc'])
        except Exception:
            continue
    aucs = np.array(aucs)
    print(f"  Bootstrap AUC: mean={aucs.mean():.3f}, 95% CI "
          f"[{np.percentile(aucs, 2.5):.3f}, {np.percentile(aucs, 97.5):.3f}]")
    return dict(mean=float(aucs.mean()),
                ci=(float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))))


def run_investigation(target_alias: str, verbose: bool = True) -> Dict[str, Any]:
    d = load_target(target_alias)
    report = {'target': target_alias, 'path': d['path'], 'layer': d['layer'],
              'n_samples': len(d['y'])}

    state = phase1_reproduce(d)
    report['phase1'] = {'auc': state['auc'], 'd_prime': state['d_prime'],
                        'n_pairs': len(state['pairs'])}

    report['phase2'] = phase2_controls(state, d)
    report['phase3'] = phase3_hypotheses(state, d)
    report['phase4'] = phase4_introspect(state, d)
    report['phase6'] = phase6_bootstrap(state)
    report['phase5_cross_scenario'] = phase5_cross_scenario(target_alias, TARGETS, layer=d['layer'])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='llama_iw',
                        choices=list(TARGETS.keys()) + ['all_llama', 'all_1000'])
    parser.add_argument('--out', default='experiment_results/dyadic_investigation.json')
    args = parser.parse_args()

    if args.target == 'all_llama':
        targets = [a for a in TARGETS if a.startswith('llama')]
    elif args.target == 'all_1000':
        targets = ['llama_iw', 'llama_ab', 'llama_ub', 'gemma_notom_ub', 'mistral_ab']
    else:
        targets = [args.target]

    reports = {}
    for t in targets:
        print(f"\n{'='*70}\nINVESTIGATING: {t}\n{'='*70}")
        try:
            reports[t] = run_investigation(t)
        except Exception as e:
            import traceback; traceback.print_exc()
            reports[t] = {'error': f'{type(e).__name__}: {e}'}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        json.dump(reports, f, indent=2, default=str)
    print(f"\nReport saved to {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
