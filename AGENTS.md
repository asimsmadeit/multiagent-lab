# Repository Guidelines

## Working Scope

Concentrate changes in `negotiation/` and `interpretability/`. The former defines agent reasoning and interaction; the latter produces deception experiments, activation datasets, probes, and causal tests. Change `concordia_mini/` or `config/` only when required by these interfaces.

## Negotiation Architecture

`base_negotiator.py` builds a Concordia entity from ordered context components: observations, memory, instructions, strategy, and reasoning questions. `ConcatActComponent` concatenates their `pre_act` context before generation; order therefore changes behavior and captured activations.

`advanced_negotiator.py` injects optional `ContextComponent`s. Keep names aligned with `ModuleType` and component lookup keys. These modules advise the actor; `negotiation/game_master/` separately tracks protocol, state, validation, turn order, event resolution, and GM modules. Do not confuse agent cognition with GM adjudication.

## Interpretability Pipeline

The pipeline is:

1. `scenarios/` generates deterministic trial parameters, private facts, prompts, and scenario-specific deception rules.
2. `InterpretabilityRunner` builds the potential deceiver and counterpart, runs rounds, and evaluates responses.
3. `TransformerLensWrapper` or `HybridLanguageModel` generates text and captures selected residual-stream activations from the same acting-model call.
4. `ActivationSample` binds activations to text, trial/round, scenario, modules, counterpart, outcome, and labels.
5. `save_dataset()` serializes aligned tensors, labels, and metadata. `probes/train_probes.py` filters, splits, trains probes, and reports AUC/R2, transfer, temporal, dyadic, and outcome analyses.
6. `causal/causal_validation.py` tests directions with patching, ablation, steering, selectivity, and behavioral/logit sensitivity.

## Scientific Invariants

- `actual_deception`/GM labels describe what the acting agent did. `perceived_deception`/agent labels describe its estimate of the counterpart; never present them as self-report or substitute one for the other.
- Emergent scenarios must encode incentives and private facts without explicitly instructing deception. Keep generated parameters stable across prompt, rule evaluator, transcript, and metadata.
- Train only on `sample_type == "negotiation"`. Verification (`round_num == -1`) and plausibility (`-2`) probes are separate interventions.
- Apply `core/qc_filter.py`; malformed, narrated, template-echoed, or special-token-leaking responses do not represent valid negotiation decisions.
- Preserve trial-grouped and dyad-aware splits. Turns or counterpart pairs from one trial must not cross train/test boundaries.
- Select layers by held-out AUC, retain random-label and train/test-gap controls, and load `.pt` files only from trusted sources.

## Development and Review

Use Python 3.12+, matching the pinned Concordia 2.4 runtime. Use four-space indentation, type hints, `snake_case` functions, and `PascalCase` classes. Install with `python -m pip install -e '.[dev]'`. Run `pytest tests/test_imports.py tests/test_probes.py tests/test_unified_analysis.py -v`, then `pytest tests/ -v`. Model-backed smoke tests require credentials/hardware and cannot be the only regression coverage.

Commits use short imperative subjects. Pull requests must state the hypothesis or behavior changed, scenarios/modules affected, label source, split unit, model/layers, verification commands, and whether results are newly generated or reproduced. Never commit API tokens, checkpoints, private transcripts, or unreviewed generated datasets.
