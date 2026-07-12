# Interpretability Audit

## Scope and Status

This audit covers the experiment path from scenario construction through activation capture, labeling, serialization, probe training, and causal intervention. The reviewed production code is under `interpretability/`; regression coverage is in `tests/interpretability/` and `tests/interpretability_runner/`.

The pipeline is operational under mocked/local tests, and the main alignment and leakage defects found during review have been repaired. It is not yet publication-ready: several remaining issues affect the construct validity of "deception" and the strength of causal claims. No GPU-backed generation, external evaluator, or human-label validation was performed as part of this audit.

## Pipeline Contract

1. `scenarios/` creates stable private facts, prompts, incentives, and deterministic label rules.
2. `InterpretabilityRunner` constructs Concordia agents, runs interactions, and emits one `ActivationSample` for each acting-model call.
3. `TransformerLensWrapper` or `HybridLanguageModel` generates text and captures the residual stream corresponding to the retained response.
4. `save_dataset()` aligns tensors with response text, labels, trial/round identifiers, dyadic links, ToM state, outcomes, prompts, and sampling metadata.
5. `probes/train_probes.py` filters and splits data, fits preprocessing plus probes, and reports transfer, temporal, dyadic, and outcome analyses.
6. `causal/causal_validation.py` applies direction extraction, patching, ablation, selectivity, and steering tests.

`actual_deception` is an external/rule assessment of the acting agent. `perceived_deception` is that agent's estimate of a counterpart. The schema now states this distinction explicitly (`interpretability/evaluation.py:169-179`, `interpretability/evaluation.py:3622-3625`).

## Repaired and Verified Behavior

### Activation and Runtime Alignment

- Generation forwards `temperature`, `top_p`, `top_k`, and seed controls. Capture replays the exact retained token sequence and records the final retained response token, excluding terminator/EOS suffixes (`interpretability/evaluation.py:498-639`).
- Steering is replayed over positions corresponding to the generation process, and hooks are cleared in `finally` (`interpretability/evaluation.py:480-496`, `interpretability/evaluation.py:621-647`).
- Counterpart calls use `FastModelWrapper`, which disables both capture and steering, preventing them from overwriting acting-agent activations (`interpretability/evaluation.py:1060-1074`).
- GM memory permits duplicate events, while actor memory retains normal deduplication (`interpretability/evaluation.py:1323-1352`, `interpretability/evaluation.py:1845`).

### Labels and Scenarios

- Emergent labels are per acting turn rather than cumulative across the transcript, so an earlier deceptive turn no longer contaminates later honest activations (`interpretability/evaluation.py:2679-2690`).
- The general `GroundTruthDetector` falls back to deterministic rules when DeepEval fails and records fallback provenance (`interpretability/core/ground_truth.py:95-139`).
- Instructed promise/alliance rules require an observed promise and a subsequent violation; absent evidence remains honest (`interpretability/scenarios/deception_scenarios.py:299-310`, `interpretability/scenarios/deception_scenarios.py:476-487`).
- Scenario parameters use local, replayable RNG instances rather than changing process-global state (`interpretability/scenarios/emergent_prompts.py:909-925`; `interpretability/scenarios/deception_scenarios.py:522-550`).
- Prompt generation fails on missing parameters, and regression tests check that emergent prompts do not explicitly instruct deception.

### Dataset and Probe Integrity

- QC rejects non-text, short, special-token-leaking, repetitive, narrated, template-echoed, and tutorial-style responses; unknown flag names fail closed (`interpretability/core/qc_filter.py:49-74`, `interpretability/core/qc_filter.py:97-132`, `interpretability/core/qc_filter.py:150-181`).
- Verification and plausibility calls are explicitly typed and excluded from negotiation training (`interpretability/evaluation.py:253-265`, `interpretability/core/qc_filter.py:135-147`).
- Serialization validates identical activation-layer sets, preserves sparse SAE row alignment, and records schema/runtime versions, exact prompts, sampling controls, label semantics, and QC flags (`interpretability/evaluation.py:3451-3577`, `interpretability/evaluation.py:3601-3637`).
- Ridge and logistic probes return fitted preprocessing pipelines. Scaling/PCA are fit inside train/CV folds, and trial-aware splits keep groups disjoint (`interpretability/probes/train_probes.py:164-224`, `interpretability/probes/train_probes.py:252-305`, `interpretability/probes/train_probes.py:378-493`).
- Unknown probe rounds/outcomes are excluded, and dyadic indices are reciprocal rather than self-links.

### Causal Data Contract

- Causal loading excludes verification/plausibility samples and constructs composite `pod_id:trial_id` groups (`interpretability/causal/causal_validation.py:39-134`).
- Probe-based causal controls use group-disjoint, seeded splits (`interpretability/causal/causal_validation.py:137-170`).
- Cross-sample patching requires prompt/activation alignment and no longer substitutes an unrelated prompt (`interpretability/causal/causal_validation.py:639-657`, `interpretability/causal/causal_validation.py:2003-2024`).
- Behavioral steering includes a magnitude sweep, permutation statistic, and matched random-direction control (`interpretability/causal/causal_validation.py:1284-1564`).

## Remaining Findings

| Severity | Finding and Impact | Required Resolution |
| --- | --- | --- |
| **Critical** | `hidden_value` calls any asking price over 1.2 times the seller's private value deceptive (`interpretability/scenarios/emergent_prompts.py:1077-1097`). An asking price is not a factual claim about value; this labels normal bargaining as lying and invalidates the target construct. | Separate reservation price, market value, asserted factual value, and offer. Label only a contradicted factual assertion, with an event-level evidence record. |
| **Critical** | Promise/alliance regexes infer deception from commitment and betrayal words in one utterance (`interpretability/scenarios/emergent_prompts.py:1124-1160`). They cannot establish intent when promising or an executed later betrayal. The contest promise game instead labels ex-post commitment violation (`interpretability/scenarios/contest_scenarios.py:1128-1206`, `interpretability/scenarios/contest_scenarios.py:1272-1282`). | Model promise, latent/declared plan, executable action, and outcome as distinct events. Report commitment violation separately from deceptive intent. |
| **High** | The runner builds a GM but runs the turn loop itself. Label extraction reaches into `_context_components` and module-private histories instead of consuming adjudicated GM events (`interpretability/evaluation.py:1793-1835`, `interpretability/evaluation.py:1839-1857`, `interpretability/evaluation.py:1966-2095`). | Make the GM own action validation and event resolution, then derive labels from an immutable event log through public component APIs. |
| **High** | Runner-local DeepEval failure/no-detector paths return zero-valued labels, making unavailable evaluation indistinguishable from observed honesty (`interpretability/evaluation.py:1278-1321`, `interpretability/evaluation.py:1772-1780`). | Return `label_status`, `label_source`, confidence, and error provenance. Exclude unknown labels rather than coercing them to zero. |
| **High** | Information withholding is evaluated against only the current response (`interpretability/scenarios/emergent_prompts.py:1100-1121`). A later terse offer can be marked deceptive after an earlier disclosure; conversely, omission may be labeled before disclosure was required. | Define a disclosure obligation and evaluate it over event history at the relevant decision boundary. |
| **High** | Best layer is selected by AUC on each layer's test partition, then the same split design is reused for reported best-layer results (`interpretability/probes/train_probes.py:2278-2352`). This creates selection optimism. | Use nested trial-grouped CV or a train/validation/test design: validation chooses layer/hyperparameters; untouched test estimates final performance. |
| **High** | `run_full_causal_validation()` does not accept scenario context. It reads `scenario` from a newly created result dict, so behavioral steering normally falls back to keyword scoring (`interpretability/causal/causal_validation.py:1972-1988`). Keyword fallback itself is allowed at `interpretability/causal/causal_validation.py:1353-1379`. | Require per-prompt scenario parameters or a versioned scorer; skip behavioral claims when they are absent. Persist baseline and steered completions for blinded review. |
| **High** | Overall causal evidence is an uncalibrated fraction of heterogeneous pass/fail heuristics; 60 percent is called "moderate" and 80 percent "strong" (`interpretability/causal/causal_validation.py:2037-2051`). Correlated proxy tests can therefore produce an overstated claim. | Pre-register a hierarchy of necessary tests, require behavioral selectivity and random controls, report effect sizes/CIs, and do not collapse results to an evidence adjective without calibration. |
| **High** | Roles and action order are deterministic in the runner (`interpretability/evaluation.py:1970-1993`). Probe performance may encode agent identity, first-mover position, prompt template, or counterpart policy rather than deception. | Randomize/counterbalance role, order, agent name, counterpart type, and scenario surface form; group splits by complete dyad/trial. |
| **Medium** | Activation capture is a teacher-forced replay after generation, not an immutable generation event, and wrapper state such as `_last_prompt` and `_current_activations` is mutable (`interpretability/evaluation.py:526-535`, `interpretability/evaluation.py:576-643`). Concurrent calls can misassociate data. | Return a per-call generation record containing tokens, prompt, response, cache, sampling config, and call ID; guard or prohibit concurrent wrapper use. |
| **Medium** | ToM `trust_level` is currently a constant 0.5 placeholder, yet it is serialized as a sample label (`interpretability/evaluation.py:1390-1400`). | Emit real ToM trust state or mark the field unavailable and exclude it from analysis. |
| **Medium** | Random-label sanity checks permute individual rows even when trial groups are supplied (`interpretability/probes/sanity_checks.py:54-85`). This breaks within-trial label structure and produces an inappropriate null. | Permute labels at the complete trial/dyad group level. |
| **Medium** | Legacy helpers still use row-level `train_test_split` (`interpretability/probes/mech_interp_tools.py:414-455`, `interpretability/probes/mech_interp_tools.py:624-673`). Accidental use can reintroduce leakage. | Delegate these APIs to the grouped probe implementation or deprecate them with a hard error for negotiation data. |
| **Low** | Scenario config dictionaries are returned by reference (`interpretability/scenarios/deception_scenarios.py:500-519`), `load_dotenv()` mutates environment state at import (`interpretability/core/deepeval_detector.py:21-24`), and several loaders use `torch.load(weights_only=False)` (`interpretability/core/dataset_builder.py:249-259`). | Return immutable/deep-copied configs, move environment loading to the CLI, and treat `.pt` artifacts as trusted-only or migrate metadata to a non-executable format. |
| **Low** | `LLMAgentRunner` seeds Python's module-global RNG (`interpretability/llm_evaluation.py:384-386`). | Use a runner-owned `random.Random` and pass it to scenario/agent operations. |

## Verification

Focused regression command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/matplotlib \
  python -m pytest \
  tests/interpretability/test_core_scenario_probe_regressions.py \
  tests/interpretability_runner -q
```

Result: **32 passed** (17 core/scenario/probe regressions and 15 runner/causal/serialization regressions). The broader focused interpretability suite was also exercised with project import, probe, unified-analysis, configuration, and Concordia compatibility tests as part of the repository audit. Model-backed activation equivalence and causal-behavior tests remain required before claiming empirical validity.

## Recommended Refactor Order

1. Define versioned `InteractionEvent`, `GenerationRecord`, and `LabelRecord` contracts with explicit subject, source, status, evidence, and provenance.
2. Put scenario rules and GM adjudication on the event log; repair hidden-value, withholding, promise, and alliance constructs before generating new data.
3. Make generation/capture return immutable per-call records and remove reliance on wrapper "last call" state.
4. Centralize dataset validation and enforce QC, negotiation-only filtering, label availability, trial/dyad groups, and schema versions at every analysis entry point.
5. Implement nested grouped evaluation and retire row-split probe helpers.
6. Pass exact prompts plus scenario parameters into causal tests; require behavioral scoring and matched controls for causal claims.
7. Counterbalance roles/order and validate labels against blinded human annotations before a benchmark or conference submission.
