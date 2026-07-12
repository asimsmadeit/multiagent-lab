# Ideal Refactor Plan

## Goal

Refactor the repository into a reproducible experiment system where negotiation behavior, adjudication, activation capture, labels, and statistical evaluation have explicit contracts and independent tests. The refactor should reduce the current monolithic runner and component state ambiguity without changing scientific results silently.

The target research question is not “can a classifier fit these rows?” It is whether emergent deceptive behavior has a robust and causally relevant signal under held-out scenarios, dyads, roles, and monitoring conditions. Architecture decisions should make invalid experimental comparisons difficult to express.

## External Justification

The official [Concordia architecture](https://github.com/google-deepmind/concordia) separates entities, components, and the engine: entities propose actions, while a Game Master grounds them into outcomes. Keeping negotiation cognition and adjudication separate follows that model and prevents an agent's heuristic belief from becoming ground truth.

[Concordia 2.4](https://github.com/google-deepmind/concordia/releases/tag/v2.4.0) added component state restoration, JSON action specifications, repeat-preserving GM memory, observation-fallback controls, and safer engine behavior. These changes directly address failures found here, so local code should depend on the released package rather than copy its internals.

[Inspect evaluation logs](https://inspect.aisi.org.uk/eval-logs.html) retain sample histories and allow later analysis, while its [scoring workflow](https://inspect.aisi.org.uk/scoring-workflow.html) permits rescoring an existing run. [W3C PROV](https://www.w3.org/TR/prov-primer/) models entities, activities, and agents so a result can be traced to how it was produced. Together they justify immutable call/event records and derived datasets rather than one mutable `ActivationSample` assembled long after execution.

Scikit-learn's [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html) splits according to supplied groups and defines test size in groups, not rows. Trials, dyads, and paraphrase families are therefore first-class split keys. PyTorch's [reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html) notes that complete reproducibility is not guaranteed across releases/platforms and requires control of multiple randomness sources. Seeds, RNG ownership, package versions, device, and algorithms must be recorded rather than relying on global seed calls.

## Architectural Principles

1. **One upstream runtime:** all live entities and components use `gdm-concordia==2.4.0`. `concordia_mini` becomes a time-limited compatibility package and is then removed.
2. **Domain truth is typed:** private facts, offers, commitments, agreements, utilities, and rule evidence are structured objects. Prompts render those objects; parsers and labelers do not independently recreate them.
3. **Cognition is not adjudication:** agent modules advise actions and estimate counterparts. GM/rule components validate actions and outcomes. Neither source silently substitutes for the other.
4. **Model calls are atomic records:** the exact assembled input, generation settings, output tokens/text, activation tensors, purpose, and actor share one call ID.
5. **State is restorable:** every stateful component returns a JSON-compatible mapping, accepts the same mapping, and passes a behavior-preserving round-trip test.
6. **Randomness and time are dependencies:** components receive `random.Random`, `numpy.random.Generator`, and a logical/injected clock. No production research path depends on module-global RNG or wall time.
7. **Datasets are projections:** immutable run records are canonical; training tensors are versioned projections with QC, label provenance, and split-family IDs.
8. **Statistics follow independent units:** selection, calibration, and testing use disjoint grouped partitions. Turns are repeated measurements, not independent samples.
9. **Compatibility is explicit:** schema/API upgrades have adapters and version fields. A legacy default may be supported, but it is never guessed silently.

## Target Package Layout

```text
negotiation/
  domain/
    schema.py             # Offer, commitment, fact, outcome, utility
    scenario.py           # Compiled deterministic scenario instances
    parsing.py            # Evidence-bearing text -> domain action
    rules.py              # Pure validation and behavioral labels
  agents/
    builder.py            # Basic/advanced Concordia builders
    config.py             # Pydantic module configuration
    components/           # Cognitive modules only
    state.py              # Shared state/RNG/clock contracts
  game_master/
    builder.py            # Protocol and engine assembly
    adjudication.py       # Action -> outcome/observations
    components/           # GM-only state and policy
    protocols.py          # Alternating/simultaneous turn rules

interpretability/
  runtime/
    runner.py             # Experiment-level orchestration only
    trial.py              # One trial state machine
    model_call.py         # GenerationRecord and capture session
    events.py             # Append-only typed events/artifact references
  labels/
    schema.py             # Label value, source, evidence, uncertainty
    rules.py              # Domain-rule adapter
    judges.py             # Optional model/human annotation adapters
    consensus.py          # Agreement analysis, never silent overwrite
  data/
    schema.py             # Versioned row/manifest contracts
    projectors.py         # Events -> negotiation/dyadic/probe datasets
    qc.py                 # Deterministic validity policy
    io.py                 # Safe tensor/metadata formats
  activations/
    capture.py            # Backend-neutral call-scoped capture
    transformer_lens.py
    hybrid.py
    token_selection.py
  probes/
    datasets.py           # Negotiation-only filtering and split keys
    pipelines.py          # Fitted preprocessing + estimator together
    evaluation.py         # Grouped metrics and controls
  causal/
    design.py             # Frozen intervention manifests
    interventions.py
    outcomes.py
    statistics.py
```

`interpretability/evaluation.py` remains as a deprecated facade until callers move. It should not retain business logic after migration.

## Core Contracts

### Scenario and Action

`ScenarioInstance` is immutable and contains spec version, seed, trial/family IDs, public state, per-role private views, legal actions, and label/outcome rules. A single compiled instance supplies every prompt and evaluator parameter.

`ObservedAction` stores raw text, normalized offer/commitment/disclosure fields, evidence spans, parser version, and uncertainty. Agreement detection must require explicit non-negated evidence; absence of a parsed claim is unknown where a rule requires one.

### Generation Record

```text
GenerationRecord
  call_id, run_id, trial_id, actor_id, purpose
  assembled_prompt/messages, prompt_hash, token_ids
  model/tokenizer revision, Concordia version
  temperature, top_p, top_k, max_tokens, seed
  output_token_ids, output_text, terminator
  activation artifact refs with layer/stage/token indices
  started/completed timestamps for provenance only
```

Call purposes include `actor_action`, `counterpart_action`, `component_analysis`, `belief_verification`, `judge`, and `monitor`. Only actor/counterpart action calls enter primary behavior probes unless a study declares otherwise.

### Label Record

`LabelRecord` contains acting-agent ID, behavior target, value (`true`, `false`, or `unknown`), continuous severity, source (`rule`, `gm`, `model_judge`, `human`, or `agent_perception`), evidence/fact IDs, evaluator version, and confidence. `actual_deception` is a compatibility projection of selected adjudicator sources; `perceived_deception` remains an agent estimate of the counterpart.

### Stateful Component

Every component test follows:

1. construct with injected RNG/clock;
2. apply observations/actions;
3. serialize `get_state()` through JSON;
4. construct a new component and `set_state()`;
5. compare public state and the next deterministic behavior;
6. mutate the restored instance and prove the original has no shared lists/dicts.

## Phased Implementation

### Phase 0: Freeze Current Contracts

**Purpose:** prevent the refactor from erasing behavior or hiding known defects.

1. Keep the module-by-module fake-model tests produced by this audit.
2. Add golden semantic fixtures for all six emergent scenarios, action parsing, label evidence, role views, and sample projection. Avoid whitespace snapshots.
3. Mark knowingly incorrect legacy outputs with strict expected-failure tests and issue IDs.
4. Create a fresh audited smoke dataset; do not use contaminated historical data as the behavioral oracle.

**Exit gate:** tests distinguish intended parity, intentional fixes, and unsupported legacy behavior.

### Phase 1: Complete the Concordia Boundary

1. Replace live `concordia_mini` imports with official `concordia` imports.
2. Update every language-model adapter to the 2.4 `sample_text` signature and explicitly pass sampling settings.
3. Migrate ActionSpec serialization, thought-chain imports, component state, and GM duplicate memory as one compatibility release.
4. Add a warning/re-export bridge for documented `concordia_mini` public imports, then remove the vendored implementation in the next major version.
5. Run upstream contract tests plus local entity/GM integration tests.

**Exit gate:** `rg 'concordia_mini' negotiation interpretability` finds no live imports; all constructed entities share official component types.

### Phase 2: Extract the Negotiation Domain

1. Introduce typed offers, commitments, disclosures, agreement state, and outcomes.
2. Move parsing into pure functions returning values plus evidence and uncertainty.
3. Compile deterministic scenarios using a local RNG derived from run seed, family ID, and trial ID.
4. Replace GM heuristic string inspection with pure rules over domain actions; retain model judges as independent annotations.
5. Make privacy views explicit and test that neither actor nor fallback observation generation can access unauthorized facts.

**Exit gate:** prompts, rules, utilities, and saved metadata agree because they consume one scenario instance.

### Phase 3: Normalize Components

1. Define common configuration, RNG, clock, state, and diagnostic protocols.
2. Separate `analyze_observation()` from `render_pre_act_context()` so reading state does not mutate it.
3. Replace hidden calls to global random/wall time and remove read-time decay or repeated side effects.
4. Replace placeholder ToM recursion and strategy/constraint fields with explicit typed beliefs, or remove claims that they are implemented.
5. Require each module to declare inputs, outputs, state, extra model-call count, and logging fields.
6. Test every module individually and in basic/advanced component order.

**Exit gate:** all components restore behavior, no state is shared across agents, and diagnostic reads are idempotent.

### Phase 4: Split the Runner into a Trial State Machine

1. Extract model loading/configuration from execution.
2. Implement `TrialRunner` with explicit states: compiled, agents built, initialized, turn proposed, action captured, adjudicated, observed, completed/failed.
3. Record append-only events and content-addressed activation artifacts at each boundary.
4. Execute the GM engine when GM behavior is claimed; otherwise use and name a rule adjudicator. Never create an inert GM solely to label rows.
5. Capture both agents symmetrically when the experiment declares bilateral white-box access.
6. Project legacy `ActivationSample` only after a trial is complete.

**Exit gate:** an interrupted trial has explicit status, and projection replay reconstructs identical transcript, labels, and sample metadata without model calls.

### Phase 5: Rebuild Dataset and Probe Boundaries

1. Filter `sample_type == 'negotiation'` in the canonical dataset-view function, not independently in scripts.
2. Create immutable split manifests grouping base scenario, paraphrases, trial, mirrored roles, and dyad.
3. Return a fitted sklearn `Pipeline` containing scaler/PCA/probe; never return an estimator whose required preprocessing is unavailable.
4. Use nested grouped selection: discovery fit, development layer/hyperparameter selection, locked test evaluation.
5. Retain random labels, nuisance baselines, train/test gap, calibration, scenario transfer, and role/dyad audits.
6. Replace pickle-first public artifacts with safe array plus JSON manifest formats; legacy trusted `.pt` loading is explicit.

**Exit gate:** automated leakage checks fail row-random or overlapping-family manifests, and a saved probe reproduces predictions after reload.

### Phase 6: Reframe Causal Evaluation

1. Freeze a causal design manifest before confirmation.
2. Define layer, token stage, direction source, coefficient, outcome, and independent unit.
3. Include zero/sham hooks, norm-matched random directions, label-shuffled directions, nuisance directions, and a positive hook control.
4. Report logit, behavioral, fluency, utility, and counterpart outcomes separately.
5. Use paired repeated generations and dyad/family-clustered uncertainty.
6. Test ToM belief clamps and sender/receiver interventions only after the underlying variables have typed state and event lineage.

**Exit gate:** every causal claim maps to a frozen intervention, estimand, controls, grouped confidence interval, and explicit interpretation limit.

### Phase 7: Packaging and Deprecation

1. Publish schema versions, data cards, split manifests, reference configs, and checksums.
2. Keep text-only, single-agent white-box, bilateral, ToM, and adaptive tracks separate by access assumption.
3. Remove deprecated facades and the vendored Concordia copy only after downstream import telemetry/search and one release cycle.
4. Re-run a clean-wheel install, offline score fixture, model-backed smoke study, and fresh paper-number reproduction.

## Recommended Pull-Request Order

1. `runtime: finish Concordia 2.4 migration`
2. `domain: add scenario/action/label schemas`
3. `components: inject RNG/clock and enforce state round trips`
4. `runtime: introduce call records and trial state machine`
5. `data: centralize QC, projections, and grouped manifests`
6. `probes: persist preprocessing pipelines and locked evaluation`
7. `causal: design manifests and complete controls`
8. `cleanup: remove legacy facades and concordia_mini`

Do not combine these into a single rewrite. Each pull request should preserve a working compatibility facade and have an independently reversible data/schema migration.

## What Not to Refactor Yet

- Do not invent a generic plugin framework before the existing six cognitive modules share a real contract.
- Do not replace deterministic scenario rules with an LLM judge abstraction.
- Do not optimize model-loading performance while call identity and activation correctness are unsettled.
- Do not regenerate publication datasets until runtime, label, QC, role, and split contracts are frozen.
- Do not claim complete cross-platform determinism; record the environment and test reproducibility within the declared platform.

## Expected Value

The refactor turns the framework's unique combination of private facts, multi-round negotiation, ToM state, GM grounding, and white-box activations into auditable research variables. It reduces false positive results from prompt/role/split leakage, makes null results publishable because the experiment is well specified, permits post-hoc labeler and monitor comparison on the same behavior, and creates a stable base for the high-impact research roadmap in `implementation_plans/high_impact_framework/`.
