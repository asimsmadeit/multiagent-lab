#!/usr/bin/env python3
"""PCA on stacked deception directions — formal proof of 1-D vs 2-D structure.

Created 2026-04-25 to complement the TTPD orthogonalised analysis.
Stacks the 6 mass-mean direction vectors per (model, config) and
runs PCA. Reports explained-variance ratio of the first 2 components.

Predictions:
  - Gemma-7B no-ToM: PC1 captures >95% (1-D)
  - Gemma-7B with-ToM: PC1 + PC2 capture ~95% (2-D; PC1 alone <80%)
  - Llama / Mistral with-ToM: somewhere in between
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA

from interpretability.script_artifacts import (
    add_legacy_trust_argument,
    download_activation_input,
    load_activation_input,
)

HF_REPO = 'sycorpia/multiagent-lab-data'
RUNS = {
    ('gemma7b', 'tom', 'ultimatum_bluff'):   'gemma7b/ultimatum_bluff/tom_20260211/activations.pt',
    ('gemma7b', 'tom', 'alliance_betrayal'): 'gemma7b/alliance_betrayal/tom_20260210/activations.pt',
    ('gemma7b', 'tom', 'info_withholding'):  'gemma7b/info_withholding/tom_20260211/activations.pt',
    ('gemma7b', 'no_tom', 'ultimatum_bluff'):   'gemma7b/ultimatum_bluff/no_tom_20260412_042958/activations.pt',
    ('gemma7b', 'no_tom', 'alliance_betrayal'): 'gemma7b/alliance_betrayal/no_tom_20260412_043012/activations.pt',
    ('gemma7b', 'no_tom', 'info_withholding'):  'gemma7b/info_withholding/no_tom_20260412_043021/activations.pt',
    ('llama31_8b', 'tom', 'ultimatum_bluff'):   'llama31_8b/ultimatum_bluff/tom_20260414_192608/activations.pt',
    ('llama31_8b', 'tom', 'alliance_betrayal'): 'llama31_8b/alliance_betrayal/tom_20260412_043955/activations.pt',
    ('llama31_8b', 'tom', 'info_withholding'):  'llama31_8b/info_withholding/tom_20260412_051758/activations.pt',
    ('mistral7b', 'tom', 'ultimatum_bluff'):   'ultimatum_bluff/activations_mistral7b_ultimatum_bluff_both_20260412_194428.pt',
    ('mistral7b', 'tom', 'alliance_betrayal'): 'alliance_betrayal/activations_mistral7b_alliance_betrayal_both_20260413_152248.pt',
    ('mistral7b', 'tom', 'info_withholding'):  'info_withholding/activations_mistral7b_info_withholding_both_20260413_103456.pt',
}


def _is_negotiation(md):
    st = md.get('sample_type')
    if st is not None:
        return st == 'negotiation'
    rn = md.get('round_num')
    return rn is None or rn >= 0


def mass_mean(X, y_binary):
    if y_binary.sum() < 2 or (~y_binary).sum() < 2:
        return None
    d = X[y_binary].mean(0) - X[~y_binary].mean(0)
    n = np.linalg.norm(d)
    return None if n < 1e-8 else d / n


def directions_for(model: str, config: str, *, trust_legacy_pt: bool = False):
    out = {}
    for scenario in ('ultimatum_bluff', 'alliance_betrayal', 'info_withholding'):
        key = (model, config, scenario)
        if key not in RUNS:
            continue
        fp = download_activation_input(HF_REPO, RUNS[key])
        data = load_activation_input(fp, trust_legacy_pt=trust_legacy_pt)
        md = data.get('metadata', [])
        labels = data['labels']
        acts = data['activations']
        y_all = np.array(labels['gm_labels'])
        n = len(y_all)
        keep = np.array([_is_negotiation(md[i]) if i < len(md) else True
                         for i in range(n)], dtype=bool)
        layer_keys = sorted([k for k in acts if isinstance(k, int)])
        layer = 14 if 14 in layer_keys else layer_keys[len(layer_keys)//2]
        X = acts[layer].float().numpy()[keep]
        y = (y_all[keep] > 0.5)
        d = mass_mean(X, y)
        if d is not None:
            out[(scenario, 'em')] = d
        # Try instructed
        ml = labels.get('mode_labels')
        if ml:
            mode_arr = np.array([ml[i] for i in range(n) if keep[i]])
            in_mask = mode_arr == 'instructed'
            if in_mask.sum() >= 10:
                X_in = X[in_mask]; y_in = y[in_mask]
                d_in = mass_mean(X_in, y_in)
                if d_in is not None:
                    out[(scenario, 'in')] = d_in
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_legacy_trust_argument(parser)
    args = parser.parse_args(argv)
    out_dir = Path('experiment_results/paper_numbers')
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [('gemma7b', 'tom'), ('gemma7b', 'no_tom'),
               ('llama31_8b', 'tom'), ('mistral7b', 'tom')]
    results = {}
    for model, config in targets:
        tag = f'{model}_{config}'
        print(f"\n=== {tag} ===")
        dirs = directions_for(
            model,
            config,
            trust_legacy_pt=args.trust_legacy_pt,
        )
        if len(dirs) < 2:
            print(f"  fewer than 2 directions; skipping")
            continue
        keys = sorted(dirs.keys())
        D = np.array([dirs[k] for k in keys])
        n_comp = min(D.shape[0], D.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(D)
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        results[tag] = {
            'n_directions': int(D.shape[0]),
            'd_model': int(D.shape[1]),
            'explained_variance_ratio': [float(v) for v in evr],
            'cumulative': [float(v) for v in cum],
            'pc1_pct': float(evr[0] * 100),
            'pc1_pc2_pct': float(cum[1] * 100) if len(evr) > 1 else float(cum[0] * 100),
        }
        print(f"  n_dirs={D.shape[0]}, d_model={D.shape[1]}")
        print(f"  PC1: {evr[0]*100:.1f}%")
        if len(evr) > 1:
            print(f"  PC1+PC2: {cum[1]*100:.1f}%")
        if len(evr) > 2:
            print(f"  PC1+PC2+PC3: {cum[2]*100:.1f}%")
    out = out_dir / 'pca_on_directions.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
