#!/usr/bin/env python3
"""
Smoke test: Verify TransformerLens works with each model before full experiment runs.

Tests:
1. Model loads successfully
2. Tokenization works
3. Forward pass with cache captures activations at expected layers
4. Activation shapes are correct (d_model matches expected)
5. Generation produces non-empty output
6. Full single-trial negotiation completes (optional, --full flag)

Usage:
    # Quick test (just model + activations, ~2 min per model)
    python smoke_test_models.py --model meta-llama/Llama-3.1-8B-Instruct

    # Test all models
    python smoke_test_models.py --all

    # Full integration test (1 trial negotiation, ~10 min per model)
    python smoke_test_models.py --model meta-llama/Llama-3.1-8B-Instruct --full
"""

import sys
sys.modules["flash_attn"] = None

import argparse
import time
import traceback

from config.experiment import MODEL_PRESETS

# Build test config from central presets
_TARGET_MODELS = [
    "google/gemma-7b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
]

# Dense layer configs per model (9 layers each)
_DENSE_LAYERS = {
    "google/gemma-7b-it": [0, 4, 8, 11, 14, 17, 20, 24, 27],
    "meta-llama/Llama-3.1-8B-Instruct": [0, 4, 8, 12, 16, 20, 24, 28, 31],
    "mistralai/Mistral-7B-Instruct-v0.1": [0, 4, 8, 12, 16, 20, 24, 28, 31],
}

MODELS = {}
for _m in _TARGET_MODELS:
    _preset = MODEL_PRESETS[_m]
    _n = _preset["n_layers"]
    MODELS[_m] = {
        "n_layers": _n,
        "d_model": _preset["d_model"],
        "test_layers": [0, _n // 2, _n - 1],
        "dense_layers": _DENSE_LAYERS.get(_m, list(range(0, _n, 4)) + [_n - 1]),
    }

TEST_PROMPT = """You are negotiating a split of 100 points with another party.
Your true minimum acceptable amount is 30 points.
The other party offers you 45 points. What do you say?"""


def test_model(model_name: str, device: str = "cuda", run_full: bool = False):
    """Run smoke tests for a single model."""
    import torch
    from transformer_lens import HookedTransformer

    expected = MODELS[model_name]
    results = {}

    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {model_name}")
    print(f"{'='*60}")

    # Test 1: Load model
    print(f"\n[1/5] Loading model...")
    t0 = time.time()
    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=torch.bfloat16,
        )
        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.1f}s")
        results["load"] = "PASS"
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        results["load"] = f"FAIL: {e}"
        return results

    # Test 2: Verify architecture
    print(f"\n[2/5] Verifying architecture...")
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_ctx = model.cfg.n_ctx

    if n_layers != expected["n_layers"]:
        print(f"  WARNING: Expected {expected['n_layers']} layers, got {n_layers}")
        results["arch"] = f"WARN: {n_layers} layers (expected {expected['n_layers']})"
    else:
        print(f"  n_layers: {n_layers} (expected {expected['n_layers']})")
        results["arch"] = "PASS"

    if d_model != expected["d_model"]:
        print(f"  WARNING: Expected d_model={expected['d_model']}, got {d_model}")
    else:
        print(f"  d_model:  {d_model} (expected {expected['d_model']})")
    print(f"  n_ctx:    {n_ctx}")

    # Test 3: Tokenization
    print(f"\n[3/5] Testing tokenization...")
    try:
        tokens = model.to_tokens(TEST_PROMPT, truncate=True)
        print(f"  Token shape: {tokens.shape}")
        assert tokens.shape[0] == 1, "Batch dim should be 1"
        assert tokens.shape[1] > 10, f"Too few tokens: {tokens.shape[1]}"
        results["tokenize"] = "PASS"
    except Exception as e:
        print(f"  FAILED: {e}")
        results["tokenize"] = f"FAIL: {e}"
        return results

    # Test 4: Forward pass + activation capture
    print(f"\n[4/5] Testing activation capture at layers {expected['test_layers']}...")
    hook_names = [f"blocks.{l}.hook_resid_post" for l in expected["test_layers"]]
    try:
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name in hook_names
            )

        for hook_name in hook_names:
            if hook_name not in cache:
                print(f"  MISSING: {hook_name}")
                results["activations"] = f"FAIL: {hook_name} missing"
                return results

            act = cache[hook_name]
            last_token = act[0, -1, :]
            print(f"  {hook_name}: shape={act.shape}, last_token={last_token.shape}")
            assert act.shape[0] == 1, "Batch dim should be 1"
            assert act.shape[2] == d_model, f"d_model mismatch: {act.shape[2]} vs {d_model}"
            assert last_token.shape[0] == d_model

        results["activations"] = "PASS"
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        results["activations"] = f"FAIL: {e}"
        return results

    # Test 5: Generation
    print(f"\n[5/5] Testing generation...")
    try:
        t0 = time.time()
        generated = model.generate(
            tokens,
            max_new_tokens=50,
            temperature=0.7,
            stop_at_eos=True,
        )
        gen_time = time.time() - t0
        response_tokens = generated[0, tokens.shape[1]:]
        response = model.to_string(response_tokens)
        print(f"  Generated {len(response_tokens)} tokens in {gen_time:.1f}s")
        print(f"  Response: {response[:200]}...")
        assert len(response.strip()) > 0, "Empty response"
        results["generation"] = "PASS"
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        results["generation"] = f"FAIL: {e}"

    # Test 6 (optional): Full integration — 1 trial negotiation
    if run_full:
        print(f"\n[6/6] Running full integration test (1 trial)...")
        try:
            t0 = time.time()
            sys.path.insert(0, ".")
            from interpretability import InterpretabilityRunner

            layers = expected["test_layers"]
            runner = InterpretabilityRunner(
                model_name=model_name,
                device=device,
                layers_to_capture=layers,
                torch_dtype=torch.bfloat16,
                max_tokens=128,
                evaluator_type="rule",  # Skip DeepEval
            )

            result = runner.run_single_negotiation(
                scenario_type="ultimatum_bluff",
                max_rounds=2,
            )

            n_samples = result.get("samples_collected", 0)
            full_time = time.time() - t0
            print(f"  Completed: {n_samples} samples in {full_time:.1f}s")

            if n_samples > 0:
                # Verify activations are captured
                sample = runner.activation_samples[0]
                for hook_name in [f"blocks.{l}.hook_resid_post" for l in layers]:
                    if hook_name in sample.activations:
                        act = sample.activations[hook_name]
                        print(f"  {hook_name}: {act.shape}")
                        assert act.shape[0] == d_model, f"Wrong d_model in sample: {act.shape}"
                results["full_integration"] = f"PASS ({n_samples} samples)"
            else:
                results["full_integration"] = "WARN: 0 samples"
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results["full_integration"] = f"FAIL: {e}"

    # Cleanup
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Smoke test TransformerLens models")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to test (default: all)")
    parser.add_argument("--all", action="store_true",
                        help="Test all models")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--full", action="store_true",
                        help="Run full integration test (1 trial negotiation)")
    args = parser.parse_args()

    if args.all:
        models_to_test = list(MODELS.keys())
    elif args.model:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            print(f"Available: {list(MODELS.keys())}")
            sys.exit(1)
        models_to_test = [args.model]
    else:
        # Default: test the two new models
        models_to_test = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.1",
        ]

    all_results = {}
    for model_name in models_to_test:
        all_results[model_name] = test_model(model_name, args.device, args.full)

    # Summary
    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    for model_name, results in all_results.items():
        all_pass = all(v == "PASS" or v.startswith("PASS") for v in results.values())
        status = "ALL PASS" if all_pass else "ISSUES"
        print(f"\n  {model_name}: {status}")
        for test, result in results.items():
            marker = "✓" if result == "PASS" or result.startswith("PASS") else "✗"
            print(f"    {marker} {test}: {result}")


if __name__ == "__main__":
    main()
