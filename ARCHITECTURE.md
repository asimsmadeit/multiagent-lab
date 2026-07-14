# Architecture and Research Contracts

## System Map

The repository has two primary packages. `negotiation/` defines actors, reasoning components, protocol state, and game-master adjudication. `interpretability/` compiles scenarios, executes trials, captures model activations, builds datasets, trains probes, and runs causal interventions. `config/` contains strict experiment models and a packaged offline reference configuration. `concordia_mini/` is a temporary compatibility surface; live runtime code imports the pinned `gdm-concordia==2.4.0` package.

## Negotiation Runtime

`negotiation/base_negotiator.py` constructs a Concordia entity from ordered context components. `ConcatActComponent` concatenates each component's `pre_act` output before generation, so component order affects both behavior and captured activations. `advanced_negotiator.py` adds optional reasoning modules such as theory of mind, uncertainty awareness, and strategy evolution. Module names must stay aligned with `ModuleType` and component lookup keys.

Agent cognition is separate from environment authority. Code under `negotiation/game_master/` owns turn order, protocol state, validation, event resolution, and committed outcomes. The typed schemas and rules in `negotiation/domain/` define offers, agreements, promises, private values, and scenario-specific adjudication. Advice produced by an actor component must never be treated as a committed GM event.

## Interpretability Pipeline

1. `interpretability/scenarios/` deterministically compiles trial parameters, private facts, prompts, and rule-evaluation inputs.
2. `interpretability/runtime/` executes alternating, simultaneous, or solo protocols and records committed events.
3. `TransformerLensWrapper` or `HybridLanguageModel` generates text and captures selected residual-stream activations from the same acting-model action.
4. A `GenerationRecord` binds requested and effective sampling, retained tokens, capture positions, model revisions, and response text. `ActivationSample` links that record to trial, actor, round, scenario, outcome, and label provenance.
5. `interpretability/data/` writes safe JSON manifests and non-pickled NPZ arrays. Split manifests keep related trials, dyads, role reversals, policies, and prompt variants in one partition.
6. `interpretability/probes/` trains and evaluates held-out probes. `interpretability/causal/` applies preregistered directions through patching, ablation, or steering and records every application disposition.

## Scientific Invariants

- `actual_deception` describes the acting agent's behavior according to the selected GM, rule, or human source. `perceived_deception` describes its estimate of the counterpart. They are not interchangeable or self-reports.
- Emergent scenarios encode incentives and private information without explicitly instructing deception. Generated parameters must agree across prompts, transcripts, evaluators, and metadata.
- Primary behavior probes train only on `sample_type == "negotiation"`. Verification and plausibility calls are separate interventions.
- The deterministic QC filter excludes malformed, narrated, template-echoed, or special-token-leaking responses.
- Trial families and counterpart dyads must not cross train, development, and test boundaries. Layer and hyperparameter selection uses development groups; the locked test set is evaluated once.
- Activation rows must identify whether they came from the literal generation pass or response-aligned replay. Missing, conflicted, and failed labels remain unavailable rather than defaulting to honesty.
- New public artifacts use JSON and NPZ with pickle disabled. Legacy `.pt` files require explicit trusted input.

## Verification

Use Python 3.12 or newer and install with `python -m pip install -e '.[dev]'`. Run deterministic regression coverage with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/matplotlib \
  python -m pytest tests/ -v
```

Model-backed capture, SAE, judging, and behavioral causal checks require suitable checkpoints, credentials, and hardware. They supplement rather than replace offline regression coverage.
