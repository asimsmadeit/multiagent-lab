#!/usr/bin/env python3
"""Data-quality smoke test for generation pipeline.

Created 2026-04-21 as the acceptance gate for the Phase A fixes in
DATA_QUALITY_FIX_PLAN.md. Runs 5 trials of a scenario on a target model,
applies the QC filter, and asserts:

  - pct_clean >= 60% on negotiation turns
  - zero EOT token leaks (</s>, <|eot|>, <|eot_id|>, <|end_of_text|>)
  - zero third-person narration
  - zero repetition loops

Exit code is non-zero if any assertion fails, so this can be wired into
CI or a pre-run check before any 50-trial production run is allowed.

Usage:
    python smoke_test_quality.py --model google/gemma-7b-it
    python smoke_test_quality.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --scenario ultimatum_bluff --trials 5

This script is CPU-capable but much faster on GPU. It does not save
activations or results; it only checks that the generation pipeline
produces clean dialogue.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List


DEFAULT_SCENARIOS = ('ultimatum_bluff', 'alliance_betrayal', 'info_withholding')


def run_smoke(model_name: str, scenarios: List[str], trials: int,
              device: str, dtype: str,
              min_pct_clean: float,
              forbidden_flags: tuple) -> Dict[str, Any]:
    """Run the smoke test and return a summary dict."""
    from interpretability.evaluation import InterpretabilityRunner
    from interpretability.core.qc_filter import qc_report, classify_response

    print(f"\n{'='*70}")
    print(f"Smoke test: {model_name}")
    print(f"Scenarios: {scenarios}")
    print(f"Trials per scenario: {trials}")
    print(f"{'='*70}")

    runner = InterpretabilityRunner(
        model_name=model_name,
        device=device,
        torch_dtype=dtype,
        evaluator_type='rule',
    )

    all_passed = True
    per_scenario: Dict[str, Dict[str, Any]] = {}
    for scenario in scenarios:
        print(f"\n--- {scenario} ---")
        # Run trials; activation samples accumulate on the runner.
        initial_n = len(runner.activation_samples)
        try:
            for _ in range(trials):
                runner.run_emergent_trial(
                    scenario=scenario,
                    max_rounds=3,
                )
        except Exception as e:
            print(f"  RUN FAILED: {type(e).__name__}: {e}")
            per_scenario[scenario] = {'error': str(e), 'passed': False}
            all_passed = False
            continue

        new_samples = runner.activation_samples[initial_n:]
        report = qc_report(new_samples)
        pct_clean = report['pct_clean']
        flags = report['flag_counts']

        n_forbidden = sum(flags.get(f, 0) for f in forbidden_flags)

        scenario_passed = pct_clean >= min_pct_clean and n_forbidden == 0
        status = 'PASS' if scenario_passed else 'FAIL'
        print(f"  n_negotiation={report['n_negotiation']}, n_clean={report['n_clean']}, "
              f"pct_clean={pct_clean:.1%}")
        print(f"  flag_counts={flags}")
        print(f"  forbidden_flag_count={n_forbidden}")
        print(f"  {status} (threshold: pct_clean >= {min_pct_clean:.0%}, "
              f"forbidden flags: {forbidden_flags})")

        per_scenario[scenario] = {
            'report': report,
            'forbidden_count': n_forbidden,
            'passed': scenario_passed,
        }
        if not scenario_passed:
            all_passed = False

    return {
        'model': model_name,
        'passed': all_passed,
        'per_scenario': per_scenario,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='HF model name (e.g., google/gemma-7b-it)')
    parser.add_argument('--scenario', action='append',
                        help='Scenario to test (can be repeated). Default: all three.')
    parser.add_argument('--trials', type=int, default=5, help='Trials per scenario (default: 5)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--min-pct-clean', type=float, default=0.60,
                        help='Minimum pct_clean to pass (default: 0.60)')
    parser.add_argument('--forbid', action='append',
                        help='QC flag that fails the gate on any occurrence (can be repeated). '
                             'Default: eot_token_leak, narrating_not_speaking, repetition_loop')
    args = parser.parse_args()

    scenarios = args.scenario or list(DEFAULT_SCENARIOS)
    forbidden = tuple(args.forbid) if args.forbid else (
        'eot_token_leak', 'narrating_not_speaking', 'repetition_loop'
    )

    result = run_smoke(
        model_name=args.model,
        scenarios=scenarios,
        trials=args.trials,
        device=args.device,
        dtype=args.dtype,
        min_pct_clean=args.min_pct_clean,
        forbidden_flags=forbidden,
    )

    print(f"\n{'='*70}")
    print(f"Overall: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"{'='*70}")
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
