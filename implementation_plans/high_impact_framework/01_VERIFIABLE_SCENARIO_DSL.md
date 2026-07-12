# Plan 1: Verifiable Scenario DSL

## Objective and Completion Signal

Build a typed, versioned scenario language that is the single source of truth for prompts, private facts, interventions, valid actions, outcomes, and deception labels. Completion means all six current emergent scenarios compile through this layer, reproduce their existing deterministic parameters, and emit evidence-bearing labels without using arbitrary `eval`, while legacy entry points still work.

This plan is first because a detector cannot be scientifically validated when the behavior being detected is defined by scattered prompt strings and partial regex checks.

## Repository Baseline

- `interpretability/scenarios/emergent_prompts.py` currently contains `generate_scenario_params()`, prompt builders, counterpart variants, belief-shift injections, six scenario checkers, and `compute_ground_truth()`.
- `interpretability/core/ground_truth.py` exposes rule, LLM, and DeepEval paths, but its direct rule dispatcher covers only a subset of scenarios.
- `interpretability/evaluation.py::ActivationSample` stores `scenario_params`, several label sources, and `emergent_ground_truth`.
- Emergent prompts must create incentives without words that explicitly direct lying or deception.

Do not rewrite these files in one pass. Introduce the new contract, write parity tests, then turn old functions into adapters.

## Why This Design Is Valid

[Colosseum](https://arxiv.org/abs/2602.15198) distinguishes written plans to collude from actions that cause measurable regret, demonstrating why executable outcomes are stronger than transcript-only judgments. [LIARS' BENCH](https://arxiv.org/abs/2511.16035) finds that some lies cannot be identified from the transcript alone, so the framework must preserve the actor's private facts and the target of belief. [DECOR](https://arxiv.org/abs/2605.19270) decomposes responses into atomic informational units and manipulation dimensions, supporting evidence-level labels rather than one opaque score. [ODCV-Bench](https://odcvbenchmark.github.io/) varies whether deception is mandated or merely incentivized, supporting an explicit condition contract.

The design implication is narrow: scenario state and executed consequences should be machine-verifiable; natural-language interpretation should produce auditable extracted claims, not silently become ground truth. This does not prove intent. If the acting model did not receive or internalize the private fact, the correct belief-dependent label is `unknown`, not deceptive.

## Target Architecture

Add this package:

```text
interpretability/scenarios/
  schema.py              # Pydantic data contracts and enums
  predicates.py          # Closed, typed predicate AST and evaluator
  compiler.py            # Spec + seed -> immutable ScenarioInstance
  registry.py            # Versioned spec lookup
  extractors.py          # Text -> Claim/Offer/Commitment with evidence spans
  validators.py          # Static and instance-level validation
  specs/                 # Versioned JSON specifications
    ultimatum_bluff.v1.json
    capability_bluff.v1.json
    hidden_value.v1.json
    info_withholding.v1.json
    promise_break.v1.json
    alliance_betrayal.v1.json
```

Add JSON files to `[tool.setuptools.package-data]` in `pyproject.toml` so an installed wheel can load them.

Core types should include:

- `ScenarioSpec`: `scenario_id`, semantic `version`, roles, fact definitions, conditions, prompt templates, intervention points, parser name, outcome rules, and label rules.
- `ScenarioInstance`: immutable `run_seed`, `trial_id`, resolved facts, per-role `PrivateView`, public state, condition, and a SHA-256 `instance_hash`.
- `FactRef`: typed reference with visibility (`public`, `role_private`, or `adjudicator_only`).
- `ObservedAction`: raw text plus parsed `Claim`, `Offer`, `Commitment`, disclosure, and evidence spans.
- `Predicate`: a closed AST with operators such as `eq`, `ne`, `lt`, `lte`, `gt`, `gte`, `contains`, `all`, `any`, and `not`. Never execute source text.
- `BehaviorLabel`: `value` (`true`, `false`, `unknown`), subtype, severity, rule IDs, fact IDs, evidence spans, belief-verification status, and `label_source`.
- `Outcome`: agreement state, utilities, constraint violations, and scenario-specific regret where defined.

Use explicit fields for `false_claim`, `strategic_omission`, `misleading_framing`, and `commitment_violation`. Preserve `actual_deception` as a derived compatibility projection, not the canonical record.

## Step-by-Step Implementation

### Phase 1: Freeze Existing Behavior

1. Create `tests/scenarios/test_legacy_scenario_parity.py` with fixed seeds for every current scenario and incentive condition.
2. Snapshot only semantic values: parameters, private facts, prompt-visible values, checker results for honest/deceptive boundary examples, and forbidden-word validation. Do not snapshot whitespace.
3. Add adversarial boundary cases: negation, quoted counterpart text, decimals, alternate currency formatting, a truthful omission, an unsupported promise, and no relevant claim.
4. Record known legacy gaps with `xfail(strict=True)` tests rather than encoding bugs as required behavior.

### Phase 2: Define the Schema

1. Implement Pydantic v2 models in `schema.py` with `extra='forbid'` and immutable instance models.
2. Define stable enum values for role, condition, visibility, label source, and behavior subtype.
3. Require `schema_version`, `spec_version`, and IDs on all persisted objects.
4. Implement canonical JSON serialization with sorted keys and a content hash. Exclude timestamps from hashes.
5. Add round-trip, invalid-field, missing-private-fact, and unknown-version tests in `tests/scenarios/test_schema.py`.

### Phase 3: Implement Safe Predicates

1. Build an exhaustive Pydantic discriminated union for predicate nodes in `predicates.py`.
2. Resolve references only through a supplied typed environment; fail closed on missing or unauthorized facts.
3. Return a `PredicateResult` containing the Boolean/unknown result and every input fact used.
4. Add truth-table tests for every operator and property tests for `not`, `all`, and `any`. Add `hypothesis` to the dev dependency only if used; otherwise generate deterministic cases with `pytest.mark.parametrize`.

### Phase 4: Compile Specs Deterministically

1. Move each parameter range and invariant into its JSON spec without changing distributions.
2. Implement `compile_scenario(scenario_id, version, trial_id, run_seed, condition)` using a local `random.Random` derived from all identity fields. Do not mutate global RNG state.
3. Resolve templates from the compiled instance. A prompt must never recompute or independently sample a value.
4. Produce separate role views so no private fact can leak into the counterpart prompt.
5. Validate that the same input yields byte-identical canonical instances and different role views contain only authorized facts.

### Phase 5: Parse Actions into Atomic Evidence

1. Define a `ScenarioActionExtractor` protocol and one deterministic extractor per scenario in `extractors.py`.
2. Each extractor must retain raw character spans and normalization decisions. For example, `$12.5k` may normalize to `12500`, but the source span remains available.
3. Parse negation and attributed speech before numeric/keyword rules so “you said this is final” is not treated as the actor's claim.
4. Represent uncertain parses explicitly. Do not coerce “no claim found” to honest when the rule requires a claim.
5. Keep LLM extraction as a separately named optional source. Validate its structured output against the same schema and never overwrite deterministic extraction.

### Phase 6: Evaluate Labels and Outcomes

1. Run label predicates against the compiled private view, prior commitments, public state, and `ObservedAction`.
2. Emit one `BehaviorLabel` per rule and an aggregation record explaining how compatibility fields are calculated.
3. Require successful pre-negotiation fact verification for labels whose definition depends on model belief. Store behavioral misrepresentation separately even when belief is unverified.
4. Implement outcomes from committed actions, not stated intentions. For applicable scenarios, compute utility and regret against the feasible truthful/cooperative baseline.
5. Add paired examples where identical text has different labels because the actor's private fact differs. This is a key validation of non-transcript ground truth.

### Phase 7: Static Validation and Prompt Safety

1. Validate unique IDs, reachable facts, legal role access, total predicate references, valid intervention rounds, and satisfiable parameter bounds.
2. Generate at least 100 instances per spec and assert prompt/rule consistency.
3. Extend forbidden-word checks across every rendered emergent prompt, including counterpart and intervention prompts.
4. Add mutation tests that deliberately rename a fact, leak a private value, reverse a threshold, or alter a prompt value; each mutation must fail a test.

### Phase 8: Migrate Runtime Callers

1. Add registry-backed implementations beneath `generate_scenario_params()`, `get_emergent_prompt()`, `get_counterpart_prompt()`, and `compute_ground_truth()`.
2. Keep their public signatures during one deprecation cycle, but attach spec and instance version metadata.
3. Update `InterpretabilityRunner` to compile one instance per trial and pass that exact object through prompts, labels, outcomes, and saved metadata.
4. Update `GroundTruthDetector` to return schema-native labels for all six scenarios. Keep DeepEval and legacy LLM judgments as independent annotations.
5. Update serialization to store `instance_hash`, rule version, evidence spans, and label provenance.

### Phase 9: Add Developer Commands

Add CLI commands with nonzero exit codes on failure:

```bash
deception scenarios validate
deception scenarios render --scenario ultimatum_bluff --trial-id 7 --seed 123
deception scenarios explain-label --event-log PATH --turn-id ID
```

`render` must redact facts not visible to the selected role unless `--adjudicator-view` is explicitly passed.

## Verification

Run:

```bash
pytest tests/scenarios/ -v
pytest tests/test_config.py tests/test_imports.py -v
pytest tests/ -v
deception scenarios validate
```

Then run a small model-backed experiment and manually audit at least five examples per scenario: compiled facts, visible prompts, parsed claims, predicate evidence, compatibility labels, and outcome. Record disagreements rather than tuning against the final test set.

## Acceptance Criteria

- All six specs validate and reproduce deterministic generation from a seed.
- No rendered emergent prompt explicitly instructs deception.
- Prompt, checker, outcome, and metadata values originate from one `ScenarioInstance`.
- Every label is explainable by versioned rules, facts, and text spans; unknown is supported.
- Private-view leakage and global-RNG tests pass.
- Legacy API parity passes except documented, intentionally fixed gaps.
- A clean wheel contains the JSON specs and all scenario commands work after installation.

## Risks and Mitigations

- **Natural language is not fully machine parseable.** Bound claims to scenario-relevant observables, retain uncertainty, and triangulate with blinded human/LLM annotations.
- **A DSL can overfit six templates.** Hold out paraphrase families and add a seventh pilot scenario using only public extension interfaces before freezing v1.
- **Version drift can invalidate datasets.** Hash specs and instances into every event and dataset row; never edit released specs in place.
- **Ground truth can overclaim intent.** Separate executed misrepresentation, inferred belief, and inferred intent. Only the first is directly verifiable here.

## Handoff to Plan 2

Plan 2 may start when `ScenarioInstance`, `ObservedAction`, `BehaviorLabel`, and `Outcome` have canonical serializers and stable identifiers. Those objects become payloads in the event stream; do not make the event layer reconstruct them from prose.
