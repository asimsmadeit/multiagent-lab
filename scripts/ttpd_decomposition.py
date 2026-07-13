#!/usr/bin/env python3
"""TTPD-analog decomposition of commission/omission deception directions.

Created 2026-04-25. Tests whether the ToM-induced bipolar deception axis
(commission vs omission anti-correlated) is structurally analogous to the
"Truth Is Universal" 2-D subspace in Bürger et al. (NeurIPS 2024 / 2025).

Bürger framework (briefly):
  Truth statements split into affirmative (e.g. "Paris is in France") and
  negated ("Paris is not in Spain") forms. Two directions explain most
  variance:
    t_g = general truth direction (positive class mean − negative class mean)
    t_p = polarity direction (separates affirmative-true from negated-true)
  Bürger showed both directions transfer across models, and t_p is roughly
  orthogonal to t_g.

Our analog (no affirmative/negated pairs available, so this is the closest
matched comparison):
  Commission deception (active lying — UB, AB) and omission deception
  (information withholding — IW) split the deception class. Define:
    d_g = "general deception" = mean across all six (scenario, mode)
          mass-mean directions, normalized
    d_p = "polarity" = mean(commission directions) − mean(omission
          directions), normalized
  Predictions if our bipolar finding is structurally analogous:
    1. d_g and d_p are nearly orthogonal (|cos| < 0.3)
    2. Commission directions project positively onto d_p, omission negatively
    3. The 2-D {d_g, d_p} basis explains most variance among the six
       directions (>~80%)
    4. The same decomposition on no-ToM / Llama / Mistral data does NOT
       show this structure (variance is captured by 1-D, not 2-D)

Outputs JSON with the decomposition for each (model, config) plus a
side-by-side comparison table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from interpretability.script_artifacts import (
    add_legacy_trust_argument,
    download_activation_input,
    load_activation_input,
)

HF_REPO = 'sycorpia/multiagent-lab-data'

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

# Group scenarios by deception type
COMMISSION_SCENARIOS = ('ultimatum_bluff', 'alliance_betrayal')
OMISSION_SCENARIOS = ('info_withholding',)


def _is_negotiation(md):
    st = md.get('sample_type')
    if st is not None:
        return st == 'negotiation'
    rn = md.get('round_num')
    return rn is None or rn >= 0


def load_directions(
    model: str,
    config: str,
    *,
    trust_legacy_pt: bool = False,
) -> Dict[Tuple[str, str], np.ndarray]:
    """Load mass-mean deception directions for a given (model, config) over all scenarios.

    Returns dict {(scenario, mode): unit_norm_direction}.
    """
    directions = {}
    for scenario in ('ultimatum_bluff', 'alliance_betrayal', 'info_withholding'):
        key = (model, config, scenario)
        if key not in RUNS:
            continue
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
        layer_keys = [k for k in acts if isinstance(k, int)]
        layer = 14 if 14 in layer_keys else sorted(layer_keys)[len(layer_keys) // 2]
        X = acts[layer].float().numpy()[keep]
        mode_arr = labels.get('mode_labels')
        if mode_arr:
            mode_arr = np.array([mode_arr[i] for i in range(n) if keep[i]])

        # Emergent direction
        em_mask = (mode_arr == 'emergent') if mode_arr is not None else np.ones(len(y), dtype=bool)
        X_em = X[em_mask]
        y_em = (y[em_mask] > 0.5)
        if y_em.sum() >= 2 and (~y_em).sum() >= 2:
            d = X_em[y_em].mean(0) - X_em[~y_em].mean(0)
            n_ = np.linalg.norm(d)
            if n_ > 1e-8:
                directions[(scenario, 'em')] = d / n_

        # Instructed direction (if present)
        if mode_arr is not None:
            in_mask = mode_arr == 'instructed'
            if in_mask.sum() >= 10:
                X_in = X[in_mask]
                y_in = (y[in_mask] > 0.5)
                if y_in.sum() >= 2 and (~y_in).sum() >= 2:
                    d = X_in[y_in].mean(0) - X_in[~y_in].mean(0)
                    n_ = np.linalg.norm(d)
                    if n_ > 1e-8:
                        directions[(scenario, 'in')] = d / n_
    return directions


def decompose(directions: Dict[Tuple[str, str], np.ndarray]) -> Dict[str, Any]:
    """Compute the TTPD-analog decomposition.

    Returns a dict with general direction, polarity direction, projections,
    orthogonality, variance explained at 1-D and 2-D, and predictions.
    """
    keys = sorted(directions.keys())
    if len(keys) < 3:
        return {'error': f'too few directions ({len(keys)})'}

    # Stack into matrix D [n_directions, d_model]
    D = np.array([directions[k] for k in keys])
    n_dirs, d_model = D.shape

    # General direction: mean of all unit-normalized directions, normalized
    d_g = D.mean(0)
    norm_g = np.linalg.norm(d_g)
    if norm_g < 1e-8:
        # All directions cancel out — strongly bipolar. Use median-based fallback.
        d_g = np.median(D, axis=0)
        norm_g = np.linalg.norm(d_g)
    d_g = d_g / (norm_g + 1e-12)

    # Polarity direction: mean(commission) − mean(omission)
    comm_dirs = [directions[k] for k in keys if k[0] in COMMISSION_SCENARIOS]
    omis_dirs = [directions[k] for k in keys if k[0] in OMISSION_SCENARIOS]
    if not comm_dirs or not omis_dirs:
        d_p = None
        comm_mean = None; omis_mean = None
    else:
        comm_mean = np.mean(comm_dirs, axis=0)
        omis_mean = np.mean(omis_dirs, axis=0)
        d_p = comm_mean - omis_mean
        norm_p = np.linalg.norm(d_p)
        if norm_p < 1e-8:
            d_p = None
        else:
            d_p = d_p / norm_p

    # Orthogonality of d_g and d_p
    cos_g_p = float(np.dot(d_g, d_p)) if d_p is not None else None

    # Project each direction onto (d_g, d_p) basis
    projections = {}
    for k in keys:
        d_i = directions[k]
        a = float(np.dot(d_i, d_g))
        b = float(np.dot(d_i, d_p)) if d_p is not None else None
        residual = d_i - a * d_g - (b * d_p if d_p is not None else 0.0)
        residual_norm = float(np.linalg.norm(residual))
        projections[f'{k[0]}/{k[1]}'] = {
            'general_coef_a': a,
            'polarity_coef_b': b,
            'residual_norm': residual_norm,
            'is_commission': k[0] in COMMISSION_SCENARIOS,
        }

    # Variance explained at 1-D (general only) and 2-D (general + polarity)
    # Each direction is unit-norm so total variance is n_dirs.
    var_1d = sum(p['general_coef_a']**2 for p in projections.values())
    var_2d = sum(
        p['general_coef_a']**2 + (p['polarity_coef_b']**2 if p['polarity_coef_b'] is not None else 0.0)
        for p in projections.values()
    )
    total_var = float(n_dirs)

    # Polarity prediction: commission directions positive on d_p, omission negative
    polarity_signs_correct = None
    if d_p is not None:
        comm_b = [p['polarity_coef_b'] for p in projections.values() if p['is_commission']]
        omis_b = [p['polarity_coef_b'] for p in projections.values() if not p['is_commission']]
        polarity_signs_correct = int(
            all(b > 0 for b in comm_b if b is not None) and
            all(b < 0 for b in omis_b if b is not None)
        )

    return {
        'n_directions': int(n_dirs),
        'cos_general_polarity': cos_g_p,  # 0 = perfect orthogonality
        'pct_variance_1d_general_only': float(var_1d / total_var),
        'pct_variance_2d_general_plus_polarity': float(var_2d / total_var),
        'projections': projections,
        'polarity_predictions_satisfied': polarity_signs_correct,
    }


def cross_compare(d_g_a: np.ndarray, d_p_a: Optional[np.ndarray],
                   d_g_b: np.ndarray, d_p_b: Optional[np.ndarray]) -> Dict[str, float]:
    """Compute alignment between two (general, polarity) bases."""
    cos_gg = float(np.dot(d_g_a, d_g_b))
    out = {'cos_general_general': cos_gg}
    if d_p_a is not None and d_p_b is not None:
        out['cos_polarity_polarity'] = float(np.dot(d_p_a, d_p_b))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_legacy_trust_argument(parser)
    args = parser.parse_args(argv)
    out_dir = Path('experiment_results/paper_numbers')
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ('gemma7b', 'tom'),
        ('gemma7b', 'no_tom'),
        ('llama31_8b', 'tom'),
        ('mistral7b', 'tom'),
    ]

    full_report: Dict[str, Any] = {'targets': {}}
    raw_directions: Dict[str, Dict] = {}

    for model, config in targets:
        print(f"\n=== {model} ({config}) ===")
        dirs = load_directions(
            model,
            config,
            trust_legacy_pt=args.trust_legacy_pt,
        )
        if not dirs:
            print(f"  no directions loaded")
            continue
        raw_directions[f'{model}_{config}'] = dirs
        decomp = decompose(dirs)
        full_report['targets'][f'{model}_{config}'] = decomp

        if 'error' in decomp:
            print(f"  {decomp['error']}")
            continue

        print(f"  n_directions: {decomp['n_directions']}")
        print(f"  cos(general, polarity): {decomp['cos_general_polarity']:.3f}  "
              f"(<0.3 = orthogonal-ish, supports 2-D structure)")
        print(f"  variance explained:")
        print(f"    1-D (general only):           {decomp['pct_variance_1d_general_only']*100:.1f}%")
        print(f"    2-D (general + polarity):     {decomp['pct_variance_2d_general_plus_polarity']*100:.1f}%")
        print(f"  polarity sign prediction satisfied: {decomp['polarity_predictions_satisfied']}")
        print(f"  per-direction projections (a = general loading, b = polarity loading):")
        for label, p in decomp['projections'].items():
            kind = 'COMM' if p['is_commission'] else 'OMIS'
            b_str = f"{p['polarity_coef_b']:+.3f}" if p['polarity_coef_b'] is not None else "n/a"
            print(f"    {label:<25s} [{kind}]  a={p['general_coef_a']:+.3f}  b={b_str}  "
                  f"residual={p['residual_norm']:.3f}")

    # Cross-config comparison: do d_g and d_p transfer across (model, config)?
    print("\n=== Cross-config comparison ===")
    keys_with_d_p = []
    for model, config in targets:
        dirs = raw_directions.get(f'{model}_{config}')
        if not dirs:
            continue
        ks = sorted(dirs.keys())
        D = np.array([dirs[k] for k in ks])
        d_g = D.mean(0); d_g = d_g / (np.linalg.norm(d_g) + 1e-12)
        comm_dirs = [dirs[k] for k in ks if k[0] in COMMISSION_SCENARIOS]
        omis_dirs = [dirs[k] for k in ks if k[0] in OMISSION_SCENARIOS]
        d_p = None
        if comm_dirs and omis_dirs:
            d_p = np.mean(comm_dirs, axis=0) - np.mean(omis_dirs, axis=0)
            n = np.linalg.norm(d_p)
            if n < 1e-8:
                d_p = None
            else:
                d_p = d_p / n
        keys_with_d_p.append((f'{model}_{config}', d_g, d_p))

    cross = {}
    for i, (n_a, dg_a, dp_a) in enumerate(keys_with_d_p):
        for j, (n_b, dg_b, dp_b) in enumerate(keys_with_d_p):
            if i >= j:
                continue
            # Need same dim — only compare same-dim pairs
            if dg_a.shape != dg_b.shape:
                continue
            comp = cross_compare(dg_a, dp_a, dg_b, dp_b)
            cross[f'{n_a} vs {n_b}'] = comp
            cross_str = ", ".join(f"{k}={v:+.3f}" for k, v in comp.items())
            print(f"  {n_a:<22s} vs {n_b:<22s}: {cross_str}")
    full_report['cross_config'] = cross

    out_path = out_dir / 'ttpd_decomposition.json'
    with out_path.open('w') as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"\n\nSaved: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
