#!/usr/bin/env python3
"""Tier A camera-ready experiments for AAMAS NEXUS / AI4CNI.

Created 2026-04-25 to produce four pieces of paper material:
  1. Per-layer x per-round probe AUC heatmaps (figure data + plot)
  2. Cross-architecture probe-direction transfer matrix
  3. Bootstrap 95% CIs on every headline AUC
  4. Gram-Schmidt orthogonalized TTPD decomposition (cleans up the
     >100% variance artifact)

CPU-only. Uses cached HF activations. Each section saves output
independently so a partial failure does not kill the others.
"""

from __future__ import annotations

# NumPy exposes RandomState at runtime; Pylint cannot infer the lazy namespace.
# pylint: disable=no-member

import argparse
import json
import sys
import time
import traceback
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


def load_target(
    model: str,
    config: str,
    scenario: str,
    *,
    trust_legacy_pt: bool = False,
) -> Optional[Dict[str, Any]]:
    key = (model, config, scenario)
    if key not in RUNS:
        return None
    path = RUNS[key]
    try:
        fp = download_activation_input(HF_REPO, path)
    except Exception as e:
        print(f"  ERROR loading {path}: {e}")
        return None
    data = load_activation_input(fp, trust_legacy_pt=trust_legacy_pt)
    md = data.get('metadata', [])
    labels = data['labels']
    acts = data['activations']
    y_all = np.array(labels['gm_labels'])
    n = len(y_all)
    keep = np.array([_is_negotiation(md[i]) if i < len(md) else True
                     for i in range(n)], dtype=bool)
    y = y_all[keep]
    round_nums = np.array([md[i].get('round_num', 0) if i < len(md) else 0
                           for i in range(n)])[keep]
    layer_keys = sorted([k for k in acts if isinstance(k, int)])
    X_per_layer = {l: acts[l].float().numpy()[keep] for l in layer_keys}
    return {
        'model': model, 'config': config, 'scenario': scenario,
        'y': y, 'round_nums': round_nums,
        'X_per_layer': X_per_layer, 'layers': layer_keys,
        'n': len(y),
    }


def fit_probe_auc(X, y, seed=SEED):
    """Standard probe pipeline; returns AUC or None."""
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
    try:
        Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    except Exception:
        return None
    n_comp = min(N_COMP, Xtr_s.shape[0]-1, Xtr_s.shape[1])
    if n_comp < 1:
        return None
    pca = PCA(n_components=n_comp)
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


# ============================================================
# Section 1: per-layer x per-round probe AUC heatmap
# ============================================================

def section1_heatmap(*, trust_legacy_pt: bool = False):
    print("\n=== Section 1: per-layer x per-round probe AUC ===")
    targets = [
        ('gemma7b', 'tom', 'info_withholding'),
        ('gemma7b', 'no_tom', 'info_withholding'),
        ('llama31_8b', 'tom', 'info_withholding'),
        ('mistral7b', 'tom', 'info_withholding'),
    ]
    results = {}
    for model, config, scenario in targets:
        tag = f'{model}_{config}_{scenario}'
        print(f"\n  {tag}")
        d = load_target(
            model,
            config,
            scenario,
            trust_legacy_pt=trust_legacy_pt,
        )
        if d is None:
            continue
        layers = d['layers']
        round_nums = d['round_nums']
        unique_rounds = sorted(np.unique(round_nums).tolist())
        unique_rounds = [r for r in unique_rounds if r >= 0][:6]
        print(f"    layers: {layers}, rounds: {unique_rounds}")
        matrix = {}
        for layer in layers:
            X = d['X_per_layer'][layer]
            for r in unique_rounds:
                mask = round_nums == r
                if mask.sum() < 30:
                    continue
                X_r = X[mask]; y_r = d['y'][mask]
                yb = (y_r > 0.5)
                if yb.sum() < 2 or (~yb).sum() < 2:
                    continue
                auc = fit_probe_auc(X_r, y_r)
                if auc is not None:
                    matrix[f'L{layer}_R{r}'] = float(auc)
                    print(f"      L{layer} R{r}: AUC = {auc:.3f} (n={mask.sum()})")
        results[tag] = {
            'layers': layers,
            'rounds': unique_rounds,
            'matrix': matrix,
        }
    out = OUT_DIR / 'per_layer_per_round_auc.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Saved: {out}")

    # Try plotting
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (tag, res) in zip(axes.flat, results.items()):
            layers = res['layers']
            rounds = res['rounds']
            grid = np.full((len(layers), len(rounds)), np.nan)
            for i, l in enumerate(layers):
                for j, r in enumerate(rounds):
                    v = res['matrix'].get(f'L{l}_R{r}')
                    if v is not None:
                        grid[i, j] = v
            im = ax.imshow(grid, aspect='auto', cmap='viridis', vmin=0.5, vmax=1.0)
            ax.set_title(tag.replace('_', ' '), fontsize=10)
            ax.set_xticks(range(len(rounds)))
            ax.set_xticklabels([f'R{r}' for r in rounds], fontsize=8)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([f'L{l}' for l in layers], fontsize=8)
            ax.set_xlabel('round', fontsize=9)
            ax.set_ylabel('layer', fontsize=9)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    v = grid[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                                color='white' if v < 0.75 else 'black', fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.04)
        fig.suptitle('Probe AUC by layer × round (info_withholding scenario)', fontsize=11)
        fig.tight_layout()
        png = OUT_DIR / 'per_layer_per_round_heatmap.png'
        fig.savefig(png, dpi=140)
        plt.close(fig)
        print(f"  Plot: {png}")
    except ImportError:
        print("  matplotlib not available; skipped plot")
    except Exception as e:
        print(f"  Plot error: {e}")
    return results


# ============================================================
# Section 2: cross-architecture probe direction transfer
# ============================================================

def section2_cross_arch_transfer(*, trust_legacy_pt: bool = False):
    print("\n=== Section 2: cross-architecture probe-direction transfer ===")
    # Best layer per (model, config) — from earlier audit
    best_layers = {
        ('gemma7b', 'tom'): 14,
        ('gemma7b', 'no_tom'): 27,
        ('llama31_8b', 'tom'): 31,
        ('mistral7b', 'tom'): 31,
    }
    # Pick one scenario for the matrix (info_withholding has strongest signal)
    scenario = 'info_withholding'
    sources = []
    for (model, config), layer in best_layers.items():
        d = load_target(
            model,
            config,
            scenario,
            trust_legacy_pt=trust_legacy_pt,
        )
        if d is None:
            continue
        if layer not in d['layers']:
            layer = sorted(d['layers'])[len(d['layers']) // 2]
        X = d['X_per_layer'][layer]
        y = d['y']
        yb = (y > 0.5)
        direction = mass_mean_direction(X, yb)
        if direction is None:
            continue
        sources.append({'tag': f'{model}/{config}', 'X': X, 'y': y,
                        'direction': direction, 'd_dim': X.shape[1]})

    matrix = {}
    for src in sources:
        for tgt in sources:
            if src['d_dim'] != tgt['d_dim']:
                # Different residual-stream dimension; not directly transferable.
                # Skip rather than zero-pad.
                matrix[f"{src['tag']}->{tgt['tag']}"] = None
                continue
            proj = tgt['X'] @ src['direction']
            yb = (tgt['y'] > 0.5).astype(int)
            try:
                auc = roc_auc_score(yb, proj)
                auc_sym = max(auc, 1.0 - auc)
            except Exception:
                auc_sym = None
            matrix[f"{src['tag']}->{tgt['tag']}"] = auc_sym
    print(f"\n  Cross-arch direction-transfer AUC (scenario={scenario}, symmetric):")
    header_label = 'src vs tgt'
    print(f"    {header_label:<22s}", end="")
    for tgt in sources:
        print(f"  {tgt['tag']:>18s}", end="")
    print()
    for src in sources:
        print(f"    {src['tag']:<22s}", end="")
        for tgt in sources:
            v = matrix[f"{src['tag']}->{tgt['tag']}"]
            print(f"  {v:>18}", end="") if v is None else print(f"  {v:>18.3f}", end="")
        print()
    out = OUT_DIR / 'cross_arch_direction_transfer.json'
    out.write_text(json.dumps({'matrix': matrix, 'scenario': scenario,
                               'tags': [s['tag'] for s in sources],
                               'best_layers': {f'{k[0]}_{k[1]}': v for k, v in best_layers.items()}},
                              indent=2, default=str))
    print(f"  Saved: {out}")
    return matrix


# ============================================================
# Section 3: bootstrap 95% CIs on every headline AUC
# ============================================================

def section3_bootstrap_cis(
    n_boot: int = 500,
    *,
    trust_legacy_pt: bool = False,
):
    print(f"\n=== Section 3: bootstrap 95% CIs (n_boot={n_boot}) ===")
    results = {}
    targets = list(RUNS.keys())
    for i, (model, config, scenario) in enumerate(targets):
        tag = f'{model}_{config}_{scenario}'
        print(f"\n  [{i+1}/{len(targets)}] {tag}")
        d = load_target(
            model,
            config,
            scenario,
            trust_legacy_pt=trust_legacy_pt,
        )
        if d is None:
            continue
        # Best layer = highest AUC at full-data probe
        best_layer = None; best_auc = 0.0
        for layer in d['layers']:
            X = d['X_per_layer'][layer]
            auc = fit_probe_auc(X, d['y'])
            if auc is not None and auc > best_auc:
                best_auc = auc; best_layer = layer
        if best_layer is None:
            print(f"    skipped (no valid probe)")
            continue
        print(f"    best layer: {best_layer}, point AUC: {best_auc:.3f}")
        # Bootstrap on best layer
        X = d['X_per_layer'][best_layer]
        y = d['y']
        rng = np.random.RandomState(SEED)
        aucs = []
        t0 = time.time()
        for s in range(n_boot):
            idx = rng.choice(len(y), size=len(y), replace=True)
            Xb = X[idx]; yb = y[idx]
            auc = fit_probe_auc(Xb, yb, seed=SEED + s)
            if auc is not None:
                aucs.append(auc)
        if not aucs:
            continue
        arr = np.array(aucs)
        results[tag] = {
            'best_layer': int(best_layer),
            'point_auc': float(best_auc),
            'bootstrap_mean': float(arr.mean()),
            'bootstrap_ci_low': float(np.percentile(arr, 2.5)),
            'bootstrap_ci_high': float(np.percentile(arr, 97.5)),
            'n_boot_succeeded': int(len(arr)),
            'wall_time_sec': float(time.time() - t0),
        }
        print(f"    bootstrap: mean={arr.mean():.3f} "
              f"95% CI [{np.percentile(arr, 2.5):.3f}, {np.percentile(arr, 97.5):.3f}] "
              f"({time.time() - t0:.0f}s)")
    out = OUT_DIR / 'bootstrap_cis.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Saved: {out}")
    return results


# ============================================================
# Section 4: Gram-Schmidt orthogonalized TTPD
# ============================================================

def section4_ttpd_orthogonalized(*, trust_legacy_pt: bool = False):
    print("\n=== Section 4: Gram-Schmidt orthogonalized TTPD decomposition ===")
    COMMISSION = ('ultimatum_bluff', 'alliance_betrayal')
    OMISSION = ('info_withholding',)
    targets = [
        ('gemma7b', 'tom'), ('gemma7b', 'no_tom'),
        ('llama31_8b', 'tom'), ('mistral7b', 'tom'),
    ]
    results = {}
    for model, config in targets:
        tag = f'{model}_{config}'
        print(f"\n  {tag}")
        directions = {}
        for scenario in ('ultimatum_bluff', 'alliance_betrayal', 'info_withholding'):
            d = load_target(
                model,
                config,
                scenario,
                trust_legacy_pt=trust_legacy_pt,
            )
            if d is None:
                continue
            layer = 14 if 14 in d['layers'] else sorted(d['layers'])[len(d['layers']) // 2]
            X = d['X_per_layer'][layer]
            y = d['y']
            yb = (y > 0.5)
            direction = mass_mean_direction(X, yb)
            if direction is not None:
                directions[(scenario, 'em')] = direction
            # Try instructed if labels distinguish modes
            full_path = RUNS[(model, config, scenario)]
            try:
                fp = download_activation_input(HF_REPO, full_path)
                data_full = load_activation_input(
                    fp,
                    trust_legacy_pt=trust_legacy_pt,
                )
                ml = data_full['labels'].get('mode_labels')
                md = data_full.get('metadata', [])
                if ml is not None:
                    n = len(data_full['labels']['gm_labels'])
                    keep = np.array([_is_negotiation(md[i]) if i < len(md) else True for i in range(n)], dtype=bool)
                    mode_arr = np.array([ml[i] for i in range(n) if keep[i]])
                    in_mask = mode_arr == 'instructed'
                    if in_mask.sum() >= 10:
                        d_full = data_full['activations'][layer].float().numpy()[keep]
                        X_in = d_full[in_mask]
                        y_in = (data_full['labels']['gm_labels'])
                        y_in_arr = np.array(y_in)[keep][in_mask]
                        yb_in = (y_in_arr > 0.5)
                        d_in = mass_mean_direction(X_in, yb_in)
                        if d_in is not None:
                            directions[(scenario, 'in')] = d_in
            except Exception:
                pass

        if len(directions) < 3:
            print(f"    fewer than 3 directions; skipping")
            continue
        keys = sorted(directions.keys())
        D = np.array([directions[k] for k in keys])

        d_g = D.mean(0)
        d_g = d_g / (np.linalg.norm(d_g) + 1e-12)

        comm_dirs = [directions[k] for k in keys if k[0] in COMMISSION]
        omis_dirs = [directions[k] for k in keys if k[0] in OMISSION]
        if not comm_dirs or not omis_dirs:
            print(f"    missing commission or omission; skipping")
            continue
        d_p_raw = np.mean(comm_dirs, 0) - np.mean(omis_dirs, 0)

        # Gram-Schmidt: orthogonalize d_p against d_g
        d_p_orth = d_p_raw - (np.dot(d_p_raw, d_g)) * d_g
        n_p = np.linalg.norm(d_p_orth)
        if n_p < 1e-8:
            print(f"    polarity collapses to general after orthogonalization; "
                  "1-D structure (no second axis)")
            d_p_orth = None
        else:
            d_p_orth = d_p_orth / n_p

        # Project each direction onto orthogonal basis
        projections = {}
        for k in keys:
            d_i = directions[k]
            a = float(np.dot(d_i, d_g))
            b = float(np.dot(d_i, d_p_orth)) if d_p_orth is not None else None
            res = d_i - a * d_g - (b * d_p_orth if d_p_orth is not None else 0.0)
            res_norm = float(np.linalg.norm(res))
            projections[f'{k[0]}/{k[1]}'] = {
                'general_a': a,
                'polarity_b_orth': b,
                'residual_norm': res_norm,
                'is_commission': k[0] in COMMISSION,
            }

        # Variance explained — now proper because basis is orthonormal
        var_1d = sum(p['general_a'] ** 2 for p in projections.values())
        var_2d = sum(
            p['general_a'] ** 2 + (p['polarity_b_orth'] ** 2 if p['polarity_b_orth'] is not None else 0.0)
            for p in projections.values()
        )
        n_dirs = len(keys)
        # cos(d_g, d_p_raw) before orthogonalization (the original metric)
        cos_g_p_raw = float(np.dot(d_g, d_p_raw / (np.linalg.norm(d_p_raw) + 1e-12)))

        results[tag] = {
            'n_directions': n_dirs,
            'cos_general_polarity_raw': cos_g_p_raw,
            'pct_variance_1d_general': float(var_1d / n_dirs),
            'pct_variance_2d_orthogonal': float(var_2d / n_dirs),
            'orthogonal_polarity_axis_exists': d_p_orth is not None,
            'projections': projections,
        }
        print(f"    cos(general, polarity_raw): {cos_g_p_raw:+.3f}")
        print(f"    1-D variance explained: {100 * var_1d / n_dirs:.1f}%")
        if d_p_orth is not None:
            print(f"    2-D (orthogonalized) variance explained: {100 * var_2d / n_dirs:.1f}%")
            print(f"    polarity orthogonal axis adds: "
                  f"{100 * (var_2d - var_1d) / n_dirs:.1f} pp")
        else:
            print(f"    no orthogonal polarity axis (polarity collapses into general)")
        for label, p in projections.items():
            kind = 'COMM' if p['is_commission'] else 'OMIS'
            b_str = f"{p['polarity_b_orth']:+.3f}" if p['polarity_b_orth'] is not None else "n/a"
            print(f"    {label:<25s} [{kind}]  a={p['general_a']:+.3f}  b_orth={b_str}  "
                  f"residual={p['residual_norm']:.3f}")

    out = OUT_DIR / 'ttpd_orthogonalized.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Saved: {out}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_legacy_trust_argument(parser)
    args = parser.parse_args(argv)
    overall_start = time.time()
    sections = [
        ('section1_heatmap', section1_heatmap),
        ('section2_cross_arch_transfer', section2_cross_arch_transfer),
        ('section4_ttpd_orthogonalized', section4_ttpd_orthogonalized),
        ('section3_bootstrap_cis', section3_bootstrap_cis),
    ]
    summary = {}
    for name, fn in sections:
        t0 = time.time()
        try:
            summary[name] = fn(trust_legacy_pt=args.trust_legacy_pt)
            summary[f'{name}_wall_time_sec'] = time.time() - t0
        except Exception as e:
            print(f"\nERROR in {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            summary[name] = {'error': f'{type(e).__name__}: {e}'}
    summary['_total_wall_time_sec'] = time.time() - overall_start
    out = OUT_DIR / 'tier_a_summary.json'
    out.write_text(json.dumps({k: v for k, v in summary.items()
                               if not isinstance(v, (dict, list)) or 'error' in (v if isinstance(v, dict) else {})},
                              indent=2, default=str))
    print(f"\nTotal wall time: {summary['_total_wall_time_sec']:.1f}s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
