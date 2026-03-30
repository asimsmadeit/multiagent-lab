#!/usr/bin/env python3
"""
Run causal validation suite for deception probes on Lambda Labs GH200.

Loads Gemma-7B-IT in TransformerLens, loads pre-collected activation data,
generates test prompts from scenario templates, and runs the full causal
validation suite (activation patching, ablation, steering vectors, probe
faithfulness, selectivity).

Usage:
    python run_causal.py --scenario ultimatum_bluff
    python run_causal.py --scenario alliance_betrayal --n-prompts 100 --layer 14
    python run_causal.py --scenario info_withholding --device cuda
"""

# ── Flash attention workaround (must be before any torch/transformers imports) ──
import sys
sys.modules["flash_attn"] = None

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

# Direct imports — bypass interpretability/__init__.py to avoid heavy deps
# (concordia_mini, pandas, etc.) that aren't needed for causal validation.
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parent

def _import_from_file(module_name, file_path):
    """Import a module directly from file, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_causal_mod = _import_from_file(
    "causal_validation",
    PROJECT_ROOT / "interpretability" / "causal" / "causal_validation.py",
)
run_full_causal_validation = _causal_mod.run_full_causal_validation

_prompts_mod = _import_from_file(
    "emergent_prompts",
    PROJECT_ROOT / "interpretability" / "scenarios" / "emergent_prompts.py",
)
ULTIMATUM_HIGH = _prompts_mod.ULTIMATUM_HIGH
ALLIANCE_HIGH = _prompts_mod.ALLIANCE_HIGH
INFO_WITHHOLD_HIGH = _prompts_mod.INFO_WITHHOLD_HIGH
DEFECT_OPTIONS = _prompts_mod.DEFECT_OPTIONS
generate_scenario_params = _prompts_mod.generate_scenario_params
IncentiveCondition = _prompts_mod.IncentiveCondition
get_emergent_prompt = _prompts_mod.get_emergent_prompt


# ── Mapping from CLI scenario names to local .pt file paths ──
ACTIVATION_FILES = {
    "ultimatum_bluff": "ultimatum_bluff/activations_merged_merged_ub.pt",
    "alliance_betrayal": "alliance_betrayal/activations_merged_20260210_220117.pt",
    "info_withholding": "info_withholding/activations_merged_merged_iw.pt",
}

# ── HuggingFace fallback for when local files don't exist ──
HF_REPO = "sycorpia/ai-control-hackathon"
HF_FILES = {
    "ultimatum_bluff": "activations_merged_ultimatum_bluff.pt",
    "alliance_betrayal": "activations_merged_alliance_betrayal.pt",
    "info_withholding": "activations_merged_info_withholding.pt",
}


def ensure_activation_file(data_dir: str, scenario: str) -> str:
    """Return path to activation file, downloading from HuggingFace if needed."""
    local_path = os.path.join(data_dir, ACTIVATION_FILES[scenario])
    if os.path.exists(local_path):
        return local_path

    # Try HuggingFace
    print(f"Local file not found: {local_path}")
    print(f"Downloading from HuggingFace ({HF_REPO})...")
    from huggingface_hub import hf_hub_download

    hf_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILES[scenario],
        repo_type="dataset",
    )
    print(f"Downloaded to cache: {hf_path}")
    return hf_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run causal validation for deception probes"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="ultimatum_bluff",
        choices=list(ACTIVATION_FILES.keys()),
        help="Which scenario to validate (default: ultimatum_bluff)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="experiment_results",
        help="Directory containing scenario activation files (default: experiment_results)",
    )
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=50,
        help="Number of test prompts to generate (default: 50)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=14,
        help="Best probe layer to validate (default: 14)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model loading (default: cuda)",
    )
    return parser.parse_args()


def load_model(device: str):
    """Load Gemma-7B-IT into TransformerLens."""
    print("\n" + "=" * 60)
    print("LOADING MODEL: google/gemma-7b-it")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dtype: bfloat16")

    t0 = time.time()
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "google/gemma-7b-it",
        device=device,
        dtype=torch.bfloat16,
    )
    elapsed = time.time() - t0
    print(f"Model loaded in {elapsed:.1f}s")
    print(f"  n_layers: {model.cfg.n_layers}")
    print(f"  d_model:  {model.cfg.d_model}")
    print(f"  n_ctx:    {model.cfg.n_ctx}")
    return model


def load_activations(data_dir: str, scenario: str, layer: int):
    """
    Load activation .pt file and extract activations, labels, and metadata.

    Returns:
        activations: dict {layer_int: np.ndarray}
        labels: np.ndarray of GM deception scores
        metadata: list of dicts
    """
    pt_path = ensure_activation_file(data_dir, scenario)
    print(f"\nLoading activations from: {pt_path}")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    # ── Extract activations as {int_layer: np.ndarray} ──
    raw_acts = data["activations"]
    activations = {}
    for k, v in raw_acts.items():
        layer_key = int(k)
        if isinstance(v, torch.Tensor):
            activations[layer_key] = v.float().numpy()
        else:
            activations[layer_key] = np.array(v, dtype=np.float32)

    # ── Extract GM labels ──
    raw_labels = data["labels"]["gm_labels"]
    labels = np.array(raw_labels, dtype=np.float64)

    # ── Metadata ──
    metadata = data.get("metadata", [])

    print(f"  Layers available: {sorted(activations.keys())}")
    print(f"  Samples: {len(labels)}")
    print(f"  Label distribution: mean={labels.mean():.3f}, "
          f"deceptive(>0.5)={int((labels > 0.5).sum())}, "
          f"honest(<=0.5)={int((labels <= 0.5).sum())}")

    if layer not in activations:
        available = sorted(activations.keys())
        raise ValueError(
            f"Requested layer {layer} not in activations. Available: {available}"
        )

    print(f"  Layer {layer} shape: {activations[layer].shape}")

    return activations, labels, metadata


def filter_emergent_samples(activations, labels, metadata):
    """
    Filter to only emergent-mode samples (the ones where deception was NOT
    explicitly instructed -- these are the interesting ones for our analysis).

    If metadata has a 'mode' field, keep only mode=='emergent'.
    If no mode info is present, keep everything (assume all emergent).
    """
    if not metadata:
        print("  No metadata available -- keeping all samples")
        return activations, labels, metadata

    # Check for mode field (stored as 'experiment_mode' or 'mode')
    sample = metadata[0] if isinstance(metadata, list) else {}
    mode_key = None
    if isinstance(sample, dict):
        if "experiment_mode" in sample:
            mode_key = "experiment_mode"
        elif "mode" in sample:
            mode_key = "mode"

    if mode_key:
        emergent_mask = np.array([
            m.get(mode_key, "emergent") == "emergent" for m in metadata
        ])
        n_emergent = emergent_mask.sum()
        print(f"  Filtering to emergent mode: {n_emergent}/{len(labels)} samples")

        if n_emergent == 0:
            print("  WARNING: No emergent samples found, keeping all samples")
            return activations, labels, metadata

        filtered_acts = {}
        for layer_key, arr in activations.items():
            filtered_acts[layer_key] = arr[emergent_mask]

        filtered_labels = labels[emergent_mask]
        filtered_meta = [m for m, keep in zip(metadata, emergent_mask) if keep]

        return filtered_acts, filtered_labels, filtered_meta

    # Check for condition field (HIGH_INCENTIVE = emergent high incentive)
    if isinstance(sample, dict) and "condition" in sample:
        emergent_mask = np.array([
            "emergent" in str(m.get("condition", "")).lower()
            or "high_incentive" in str(m.get("condition", "")).lower()
            or "low_incentive" in str(m.get("condition", "")).lower()
            for m in metadata
        ])
        n_emergent = emergent_mask.sum()
        if n_emergent > 0 and n_emergent < len(labels):
            print(f"  Filtering by condition field: {n_emergent}/{len(labels)} samples")
            filtered_acts = {}
            for layer_key, arr in activations.items():
                filtered_acts[layer_key] = arr[emergent_mask]
            filtered_labels = labels[emergent_mask]
            filtered_meta = [m for m, keep in zip(metadata, emergent_mask) if keep]
            return filtered_acts, filtered_labels, filtered_meta

    print("  No mode/condition filter applicable -- keeping all samples")
    return activations, labels, metadata


def generate_test_prompts(scenario: str, n_prompts: int) -> list:
    """
    Generate test prompts by filling scenario templates with random parameters.

    For each scenario, we use the HIGH_INCENTIVE template (the one most likely
    to elicit deception) and fill in randomized parameter values.
    """
    print(f"\nGenerating {n_prompts} test prompts for '{scenario}'...")
    prompts = []

    for i in range(n_prompts):
        params = generate_scenario_params(scenario, trial_id=i)

        if scenario == "ultimatum_bluff":
            # ULTIMATUM_HIGH needs {true_walkaway}
            prompt = ULTIMATUM_HIGH.format(true_walkaway=params["true_walkaway"])

        elif scenario == "alliance_betrayal":
            # ALLIANCE_HIGH has no placeholders -- it's a fixed prompt
            prompt = ALLIANCE_HIGH

        elif scenario == "info_withholding":
            # INFO_WITHHOLD_HIGH needs {defect} and {fair_price}
            prompt = INFO_WITHHOLD_HIGH.format(
                defect=params["defect"],
                fair_price=params["fair_price"],
            )

        else:
            # Fallback: use get_emergent_prompt which handles all scenarios
            prompt = get_emergent_prompt(
                scenario, IncentiveCondition.HIGH_INCENTIVE, params
            )

        prompts.append(prompt)

    # Show a sample
    print(f"  Sample prompt (first 200 chars):")
    print(f"    {prompts[0][:200]}...")

    return prompts


def save_results(results: dict, scenario: str):
    """Save causal validation results to JSON."""
    out_path = f"causal_validation_results_{scenario}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
    return out_path


def print_summary(results: dict, scenario: str):
    """Print a clear human-readable summary."""
    print("\n" + "=" * 60)
    print(f"CAUSAL VALIDATION RESULTS: {scenario.upper()}")
    print("=" * 60)

    n_passed = results["n_tests_passed"]
    n_total = results["n_tests_total"]
    strength = results["causal_evidence_strength"]
    overall = results["overall_passed"]

    print(f"  Tests passed:       {n_passed}/{n_total}")
    print(f"  Evidence strength:  {strength.upper()}")
    print(f"  Overall verdict:    {'PASSED' if overall else 'FAILED'}")

    print("\n  Individual tests:")
    for name, res in results.get("tests", {}).items():
        status = "PASS" if res.get("passed") else "FAIL"
        msg = res.get("message", "")
        effect = res.get("effect_size", "N/A")
        print(f"    [{status}] {name}")
        print(f"          effect_size={effect}  {msg}")

    if not overall:
        print("\n  ** Causal evidence is insufficient. The identified probe")
        print("     directions may be correlational, not causal.")
    else:
        print("\n  ** Causal evidence supports that probe directions are")
        print("     meaningfully used by the model for deceptive behavior.")


def main():
    args = parse_args()

    print("=" * 60)
    print("CAUSAL VALIDATION RUNNER")
    print("=" * 60)
    print(f"  Scenario:  {args.scenario}")
    print(f"  Data dir:  {args.data_dir}")
    print(f"  Layer:     {args.layer}")
    print(f"  N prompts: {args.n_prompts}")
    print(f"  Device:    {args.device}")

    # ── Step 1: Load the TransformerLens model ──
    model = load_model(args.device)

    # ── Step 2: Load activation data from .pt file ──
    activations, labels, metadata = load_activations(
        args.data_dir, args.scenario, args.layer
    )

    # ── Step 3: Filter to emergent-mode samples only ──
    activations, labels, metadata = filter_emergent_samples(
        activations, labels, metadata
    )

    # Sanity check: enough samples for the tests
    n_deceptive = int((labels > 0.5).sum())
    n_honest = int((labels <= 0.5).sum())
    if n_deceptive < 5 or n_honest < 5:
        print(f"\nWARNING: Very few samples in one class "
              f"(deceptive={n_deceptive}, honest={n_honest}).")
        print("Causal tests may fail due to insufficient data.")

    # ── Step 4: Generate test prompts from scenario templates ──
    test_prompts = generate_test_prompts(args.scenario, args.n_prompts)

    # ── Step 5: Run the full causal validation suite ──
    print("\n" + "=" * 60)
    print("RUNNING CAUSAL VALIDATION SUITE")
    print("=" * 60)

    t0 = time.time()
    results = run_full_causal_validation(
        model=model,
        activations=activations,
        labels=labels,
        best_layer=args.layer,
        test_prompts=test_prompts,
        verbose=True,
    )
    elapsed = time.time() - t0
    results["wall_time_seconds"] = elapsed
    results["scenario"] = args.scenario
    results["layer"] = args.layer
    results["n_prompts"] = args.n_prompts

    # ── Step 6: Save results ──
    out_path = save_results(results, args.scenario)

    # ── Step 7: Print summary ──
    print_summary(results, args.scenario)
    print(f"\nTotal wall time: {elapsed:.1f}s")
    print(f"Results file: {out_path}")


if __name__ == "__main__":
    main()
