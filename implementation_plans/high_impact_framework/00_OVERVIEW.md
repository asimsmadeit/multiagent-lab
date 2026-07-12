# High-Impact Framework Implementation Roadmap

## Purpose

This directory turns the framework upgrades rated **Very High** or **High (long term)** into standalone build specifications. A new implementation agent should read this file, then the plan it is assigned, plus the repository-root `AGENTS.md`. No prior conversation is required.

The target system is a research framework for emergent deception in multi-agent negotiation. `negotiation/` constructs Concordia agents and their ordered cognitive components. `interpretability/` generates scenarios, executes dyadic negotiations, labels behavior, captures model activations, trains probes, and performs causal interventions. Preserve the distinction between the acting agent's `actual_deception` label and its estimate of another agent's behavior, `perceived_deception`.

## Recommended Build Order

| Order | Plan | Priority | Result |
|---:|---|---|---|
| 1 | [Verifiable Scenario DSL](01_VERIFIABLE_SCENARIO_DSL.md) | Very High | Typed, executable scenario truth and label contracts replace scattered prompt/evaluator logic. |
| 2 | [Event-Sourced Execution](02_EVENT_SOURCED_EXECUTION.md) | Very High | Every fact, observation, action, model call, activation, label, and intervention becomes replayable and attributable. |
| 3 | [Theory of Mind 2.0](03_THEORY_OF_MIND_2.md) | Very High | ToM becomes an evidence-updated belief model whose predictions and policy effects can be measured. |
| 4 | [Bilateral Instrumentation](04_BILATERAL_INSTRUMENTATION.md) | Very High | Activations and belief state are aligned across both agents and across time, without role or split leakage. |
| 5 | [Stronger Causal Testing](05_STRONGER_CAUSAL_TESTING.md) | Very High | Probe claims are tested with held-out, controlled, dose-response interventions and clustered inference. |
| 6 | [Adaptive Adversaries](06_ADAPTIVE_ADVERSARIES.md) | Very High | Monitors are evaluated against agents that know about or optimize against detection, including transfer attacks. |
| 7 | [Benchmark Packaging](07_BENCHMARK_PACKAGING.md) | High, long term | Stable tasks, schemas, baselines, audit artifacts, and release governance become usable by outside researchers. |

The order is a dependency graph, not just a preference. The DSL defines truth; the event layer records it; ToM consumes typed evidence; bilateral capture joins both agents' traces; causal tests intervene on those defined variables; adaptive evaluation attacks the resulting monitors; packaging freezes only contracts that survived those tests.

## Cross-Plan Rules

1. Do not explicitly instruct deception in emergent conditions. Encode private facts, incentives, and action consequences, then observe behavior.
2. Treat rule, model-judge, human, and agent-belief labels as separate sources. Never silently collapse disagreement into a single truth value.
3. Capture activations from the same acting-model call that generated the stored response. Give component-analysis and evaluator calls different purposes and identifiers.
4. Split by trial and dyad. Role, scenario template, paraphrase family, and counterpart policy must not leak across train and test.
5. Randomize or counterbalance actor role and turn order. Report safety metrics with task utility, calibration, cost, latency, and fixed low false-positive-rate operating points.
6. Select layers, directions, thresholds, and hyperparameters using development data only. A locked test set is evaluated once per declared experiment.
7. Null findings still pass an implementation milestone when the preregistered test is valid, reproducible, and fully reported.
8. Preserve backward-compatible loaders during migration, but mark legacy records with explicit schema and provenance warnings.

## Shared Delivery Protocol

For each plan, implement one numbered phase per pull request when practical. Every pull request must identify the hypothesis, schema change, label source, split unit, model/layers, and verification commands. Add deterministic unit tests before model-backed experiments. Keep generated data, secrets, and large checkpoints out of Git.

Before starting a downstream plan, run its dependency gate listed in that document. When all seven plans are complete, the minimum release gate is:

```bash
python -m pip install -e '.[dev]'
pytest tests/ -v
python -m interpretability.cli --help
```

Also require a fresh, versioned smoke run that replays to the same transcript and labels; produces valid paired-agent records; passes grouped-split leakage checks; runs causal controls; reports adaptive-monitor safety/usefulness curves; and builds the public benchmark artifact from a clean checkout.

## Expected Research Payoff

Together these upgrades support a defensible paper question that the current components alone cannot answer: **when deception emerges during negotiation, do dynamically updated beliefs and cross-agent internal states provide a causal, robust detection signal under distribution shift and adaptive evasion?** The framework's distinctive value is the joint availability of controlled private information, recursive ToM state, multi-round agent interaction, ground-truth action consequences, and white-box activations. The roadmap turns those ingredients into auditable experimental variables rather than loosely aligned metadata.
