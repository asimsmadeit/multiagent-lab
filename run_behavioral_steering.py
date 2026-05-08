#!/usr/bin/env python3
"""Run behavioral steering across (model, scenario) cells.

Loads each model with TransformerLens, computes the mass-mean deception
direction at the model's best layer from cached activations on
HuggingFace, then calls steering_behavioral_test to generate text under
steering at magnitudes -3..+3 and score the generated text with the
rule-based deception evaluator. Saves per-cell results as JSON.

Usage:
    python run_behavioral_steering.py
    python run_behavioral_steering.py --only gemma_iw,llama_ab
    python run_behavioral_steering.py --n-samples 30 --magnitudes -3,-1,0,1,3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download

HF_REPO = "sycorpia/multiagent-lab-data"
OUT_DIR = Path("experiment_results/behavioral_steering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (cell_id, model_name, scenario, activations_path, best_layer)
CELLS = [
    ("gemma_ub", "google/gemma-7b-it", "ultimatum_bluff",
     "gemma7b/ultimatum_bluff/no_tom_20260412_042958/activations.pt", 20),
    ("gemma_ab", "google/gemma-7b-it", "alliance_betrayal",
     "gemma7b/alliance_betrayal/no_tom_20260412_043012/activations.pt", 27),
    ("gemma_iw", "google/gemma-7b-it", "info_withholding",
     "gemma7b/info_withholding/no_tom_20260412_043021/activations.pt", 27),

    ("llama_ub", "meta-llama/Llama-3.1-8B-Instruct", "ultimatum_bluff",
     "llama31_8b/ultimatum_bluff/tom_20260414_192608/activations.pt", 4),
    ("llama_ab", "meta-llama/Llama-3.1-8B-Instruct", "alliance_betrayal",
     "llama31_8b/alliance_betrayal/tom_20260412_043955/activations.pt", 31),
    ("llama_iw", "meta-llama/Llama-3.1-8B-Instruct", "info_withholding",
     "llama31_8b/info_withholding/tom_20260412_051758/activations.pt", 31),

    ("mistral_ub", "mistralai/Mistral-7B-Instruct-v0.1", "ultimatum_bluff",
     "ultimatum_bluff/activations_mistral7b_ultimatum_bluff_both_20260412_194428.pt", 31),
    ("mistral_ab", "mistralai/Mistral-7B-Instruct-v0.1", "alliance_betrayal",
     "alliance_betrayal/activations_mistral7b_alliance_betrayal_both_20260413_152248.pt", 31),
    ("mistral_iw", "mistralai/Mistral-7B-Instruct-v0.1", "info_withholding",
     "info_withholding/activations_mistral7b_info_withholding_both_20260413_103456.pt", 28),
]


def is_negotiation(md: Dict[str, Any]) -> bool:
    st = md.get("sample_type")
    if st is not None:
        return st == "negotiation"
    rn = md.get("round_num")
    return rn is None or rn >= 0


def load_direction(activations_path: str, layer: int) -> Tuple[np.ndarray, int]:
    """Load activations from HF, filter to negotiation samples, return
    mass-mean direction at the requested layer."""
    print(f"  loading {activations_path}", flush=True)
    fp = hf_hub_download(HF_REPO, activations_path, repo_type="dataset")
    data = torch.load(fp, weights_only=False)
    md = data.get("metadata", [])
    labels = data["labels"]
    acts = data["activations"]
    y_all = np.array(labels["gm_labels"])
    n = len(y_all)
    keep = np.array(
        [is_negotiation(md[i]) if i < len(md) else True for i in range(n)],
        dtype=bool,
    )
    if layer not in acts:
        layer_keys = sorted([k for k in acts if isinstance(k, int)])
        layer = layer_keys[len(layer_keys) // 2]
        print(f"  requested layer not present; falling back to {layer}", flush=True)
    X = acts[layer].float().numpy()[keep]
    y = y_all[keep]
    yb = (y > 0.5)
    if yb.sum() < 2 or (~yb).sum() < 2:
        return None, layer
    direction = X[yb].mean(0) - X[~yb].mean(0)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return None, layer
    return direction / norm, layer


def build_test_prompts(scenario: str, n_prompts: int = 20):
    """Reuse the project's scenario prompt generator if available; otherwise
    fall back to a minimal hand-coded prompt."""
    try:
        from interpretability.scenarios.emergent_prompts import (
            generate_scenario_params,
            ULTIMATUM_HIGH,
            ALLIANCE_HIGH,
            INFO_WITHHOLD_HIGH,
        )
        prompts = []
        params_list = []
        for i in range(n_prompts):
            params = generate_scenario_params(scenario, trial_id=i)
            if scenario == "ultimatum_bluff":
                prompt = ULTIMATUM_HIGH.format(true_walkaway=params["true_walkaway"])
            elif scenario == "alliance_betrayal":
                prompt = ALLIANCE_HIGH
            elif scenario == "info_withholding":
                prompt = INFO_WITHHOLD_HIGH.format(
                    defect=params["defect"],
                    fair_price=params["fair_price"],
                )
            else:
                prompt = f"Scenario: {scenario}. Make your move."
            prompts.append(prompt)
            params_list.append(params)
        return prompts, params_list
    except Exception as e:
        print(f"  could not load scenario prompts: {e}; using minimal fallback")
        prompts = [f"Scenario: {scenario}. Make your move." for _ in range(n_prompts)]
        params_list = [{} for _ in range(n_prompts)]
        return prompts, params_list


def run_cell(cell_id, model_name, scenario, activations_path, best_layer,
             magnitudes, n_samples, max_new_tokens, dtype, device, force):
    out_path = OUT_DIR / f"{cell_id}.json"
    if out_path.exists() and not force:
        print(f"[{cell_id}] already done at {out_path}; skipping (use --force to rerun)")
        return

    print(f"\n=== {cell_id} ({model_name} / {scenario}) ===", flush=True)
    t0 = time.time()

    direction, layer = load_direction(activations_path, best_layer)
    if direction is None:
        print(f"  could not compute direction; skipping")
        return
    print(f"  direction: layer={layer}, dim={direction.shape[0]}, norm=1.0", flush=True)

    print(f"  loading model {model_name} on {device} ({dtype})", flush=True)
    from transformer_lens import HookedTransformer
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch_dtype,
    )

    # Match dimensionality
    d_model = model.cfg.d_model
    if direction.shape[0] != d_model:
        print(f"  ERROR: direction dim {direction.shape[0]} != model d_model {d_model}; skipping")
        return

    test_prompts, scenario_params_list = build_test_prompts(scenario, n_prompts=n_samples)

    from interpretability.causal.causal_validation import (
        SteeringVector,
        steering_behavioral_test,
    )
    sv = SteeringVector(direction=direction, layer=layer, magnitude=1.0,
                        method="mass_mean")

    print(f"  running steering_behavioral_test (n={n_samples}, magnitudes={magnitudes}, "
          f"max_new_tokens={max_new_tokens}, control=on)", flush=True)
    result = steering_behavioral_test(
        model=model,
        steering_vector=sv,
        test_prompts=test_prompts,
        scenario=scenario,
        scenario_params_list=scenario_params_list,
        magnitudes=magnitudes,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        random_direction_control=True,
        verbose=True,
    )

    payload = {
        "cell_id": cell_id,
        "model_name": model_name,
        "scenario": scenario,
        "layer": layer,
        "n_samples": n_samples,
        "magnitudes": magnitudes,
        "max_new_tokens": max_new_tokens,
        "wall_time_sec": time.time() - t0,
        "result": result.to_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  saved {out_path}", flush=True)

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="comma-separated cell ids to run")
    parser.add_argument("--magnitudes", default="-3,-2,-1,0,1,2,3",
                        help="comma-separated steering magnitudes")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="prompts per magnitude")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true",
                        help="rerun cells whose output already exists")
    args = parser.parse_args()

    magnitudes = [float(x) for x in args.magnitudes.split(",")]
    selected = set(args.only.split(",")) if args.only else None

    overall_t0 = time.time()
    for cell in CELLS:
        cell_id = cell[0]
        if selected and cell_id not in selected:
            continue
        try:
            run_cell(*cell,
                     magnitudes=magnitudes,
                     n_samples=args.n_samples,
                     max_new_tokens=args.max_new_tokens,
                     dtype=args.dtype,
                     device=args.device,
                     force=args.force)
        except Exception as e:
            print(f"[{cell_id}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
    print(f"\ntotal wall time: {time.time() - overall_t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
