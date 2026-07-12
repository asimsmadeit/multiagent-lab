# Plan 5: Stronger Causal Testing

## Objective and Completion Signal

Turn probe directions from correlational diagnostics into explicitly defined causal experiments. Completion means each causal result names its intervention, comparison, outcome, estimand, selection data, independent evaluation data, controls, uncertainty method, and scope of inference. A valid null result is acceptable; an under-controlled positive result is not.

## Repository Baseline

`interpretability/causal/causal_validation.py` already implements direction extraction, activation and cross-sample patching, ablation, logit sensitivity, behavioral steering, probe faithfulness, selectivity, and a full-suite runner. `InterpretabilityRunner` also exposes steering and belief-stability studies. These are valuable primitives.

The upgrade is experimental architecture: isolate discovery from confirmation, standardize matched contexts and token stages, add random/sham/content controls, use dyad-clustered statistics, intervene on structured ToM variables, and require behavioral plus utility outcomes. Keep the existing public functions as compatibility wrappers while extracting reusable internals.

## Why This Design Is Valid

[Towards Reliable Evaluation of Behavior Steering](https://arxiv.org/abs/2410.17245) recommends downstream-similar contexts, model likelihoods, standardized comparisons, and baseline controls, and finds some interventions weaker than prior demonstrations suggested. [Pressure-Testing Deception Probes](https://arxiv.org/abs/2605.27958) reports collapse under stylistic distribution shift and recovery from style augmentation, so clean in-distribution effects are insufficient. [Faithfulness as Information Flow](https://arxiv.org/abs/2605.24286) frames sufficiency, completeness, and necessity as distinct properties. [NARCBench](https://arxiv.org/abs/2604.01151) finds multiple aggregation strategies are needed across collusion types, warning against a universal single direction.

The logical consequence is to test a family of precise causal claims rather than label one direction “the deception feature.” Activation intervention can establish that a representation is causally relevant to a measured output under a specific context; it does not by itself prove the model represents deception, intent, or a natural mechanism.

## Target Causal Claims and Estimands

Predeclare claims at increasing strength:

1. **Sensitivity:** changing activation along direction `d` changes a prespecified action-token likelihood.
2. **Behavioral effect:** the intervention changes the probability of a verifiably misleading action under Plan 1 rules.
3. **Selectivity:** the effect is larger for deception-relevant outcomes than matched task, style, fluency, or utility controls.
4. **Necessity/sufficiency:** removing or adding the signal changes behavior in the predicted direction under held-out contexts.
5. **Mediation:** randomized evidence changes ToM belief, which changes activation, which changes action; each link is measured and not merely narrated.
6. **Dyadic effect:** sender or receiver intervention changes the partner's later belief/action or the joint outcome.

Primary estimand example:

```text
ATE(alpha, layer, stage) =
  E[Y(action under +alpha*d) - Y(action under sham)]
```

Here `Y` is defined before the run, such as a structured action log-odds contrast, Plan 1 deception indicator over repeated generations, or counterpart utility. Keep logit, behavioral, and outcome estimands separate.

## Target Architecture

Refactor toward:

```text
interpretability/causal/
  schema.py        # Intervention, estimand, design, and result contracts
  design.py        # Frozen manifests, pairing, randomization, split checks
  directions.py    # Train-only direction extraction and alignment
  hooks.py         # Layer/stage/token-scoped interventions
  outcomes.py      # Logit, action, ToM, and dyadic outcome measurement
  controls.py      # Sham, random, content/style, and positive controls
  statistics.py    # Cluster bootstrap, randomization tests, correction
  runner.py        # Resumable execution over Plan 2 events
  reporting.py     # Tables, plots, claim-bound summaries
```

Keep `causal_validation.py` as a facade during migration.

## Step-by-Step Implementation

### Phase 1: Freeze a Causal Design Contract

1. Define `CausalDesign` with hypothesis ID, unit of randomization, scenario/spec versions, eligible sample types, discovery/development/confirmatory split IDs, model revision, hook/stage/token selector, direction method, coefficient grid, outcomes, controls, seed, stopping rule, and statistical method.
2. Serialize the design and hash it before confirmatory execution. The event stream must reference this hash.
3. Define the independent unit as scenario-instance family/dyad, not turn. Paired interventions within a prompt remain linked to that unit.
4. Reject designs that select a layer/direction/alpha on the confirmatory split or mix verification/plausibility samples into negotiation training.
5. Add `deception causal validate-design CONFIG` with actionable errors.

### Phase 2: Construct Matched Experimental Sets

1. Use Plan 1 to generate matched pairs that differ in one causal factor: private fact, incentive, available evidence, partner policy, or belief-shift intervention.
2. Record surface differences and create paraphrase families. Do not call two prompts “matched” if the intervention necessarily changes task content without accounting for it.
3. Use Plan 3 controlled partners for belief experiments and Plan 4 role-mirrored dyads for bilateral experiments.
4. Allocate entire base-instance/paraphrase/role families to discovery, development, or confirmation before direction fitting.
5. Balance labels and relevant nuisance variables within each partition; report variables that cannot be balanced.

### Phase 3: Separate Direction Discovery from Testing

1. Fit mass-mean, logistic, PCA/subspace, and ToM-specific directions only on discovery trials.
2. Standardize/whiten using discovery statistics. Store means, covariance approximations, direction norms, fit sample IDs, and random seed.
3. Select layer, token stage, rank, and regularization on development trials using a preregistered criterion.
4. Freeze one primary direction and a small declared secondary set before confirmation. Do not choose the best confirmatory result after the fact.
5. Include label-shuffled directions and directions trained on matched style/content targets.

### Phase 4: Implement Precise Hook Semantics

1. Hook by model revision, hook name, layer, call purpose, capture stage, and token selector. Fail if the actual tensor shape or token set differs from the design.
2. Support additive steering, projection removal, mean replacement, cross-sample patching, and subspace resampling as distinct intervention types.
3. Define coefficient units: multiples of discovery-set activation standard deviation after direction normalization.
4. Apply interventions independently during prefill, partner-message reading, and generation. Never aggregate these into an unspecified “layer intervention.”
5. Log pre/post tensor checksums, norm change, selected token indices, and intervention event IDs without persisting unneeded full hidden states.

### Phase 5: Add Mandatory Controls

Every primary intervention must include:

1. **Sham hook:** identical hook path with zero coefficient.
2. **Norm-matched random directions:** several seeded directions, orthogonalized to the target where appropriate.
3. **Label-shuffled direction:** identical extraction pipeline with permuted family-level labels.
4. **Nuisance directions:** response length, numerical offer amount, sentiment/style, scenario, and role when estimable.
5. **Positive control:** an intervention with a known measurable target in the same backend, such as an offer-token/logit direction or a synthetic feature fixture.
6. **Untargeted outcomes:** perplexity/fluency, legal action rate, agreement, utility, and non-deception semantic features.

Controls must use the same coefficient grid and sampling budget as the target direction.

### Phase 6: Measure Dose Response and Behavior

1. Use a symmetric coefficient grid selected on development data, for example `[-4, -2, -1, 0, 1, 2, 4]` standard-deviation units.
2. First measure prespecified logit contrasts at decision-relevant tokens using identical prefixes.
3. Then sample multiple actions per condition with paired generation seeds/configuration where supported.
4. Score actions through Plan 1 rule/evidence labels, separately retained model/human annotations, QC, and negotiation outcomes.
5. Plot directionality, monotonicity, saturation, and collateral degradation. A large effect caused by incoherent output is a failed selectivity result.
6. Record noncompliance such as invalid generations or hook failures and analyze it rather than dropping silently.

### Phase 7: Test ToM Information Flow

Use a randomized chain rather than correlational mediation language:

1. Randomize diagnostic partner evidence or a typed belief-shift event.
2. Verify that the structured posterior changes (first-stage relevance).
3. Clamp or shuffle the posterior at the advisor boundary while holding transcript constant.
4. Intervene on the candidate activation conditional on the belief state.
5. Measure predicted action, actual action, receiver belief update, and outcome.
6. Estimate evidence total effect, belief-controlled direct effect, activation-controlled effect, and interaction with partner type. Present formal mediation only when identification assumptions are stated and plausible.

Include `tom_text_only`, frozen, oracle, and shuffled conditions from Plan 3 to distinguish useful structured state from persuasive context wording.

### Phase 8: Test Bilateral Causality

1. Patch the sender during production and measure receiver processing, receiver posterior, and response.
2. Independently patch the receiver only while it reads the sender message, leaving the sent text fixed.
3. Compare sender-only, receiver-only, both, and sham factorial conditions.
4. Test temporal specificity by intervening one turn earlier/later and at matched irrelevant spans.
5. Report whether a receiver-side detector is causal to a control decision separately from whether it merely predicts sender behavior.

### Phase 9: Statistical Inference

1. Estimate uncertainty with cluster bootstrap or randomization inference at the randomized family/dyad unit.
2. Use paired estimates for within-prompt interventions. Report effect size and confidence interval, not only p-values.
3. Correct declared secondary layer/stage/outcome comparisons with Holm or false-discovery-rate control.
4. Predefine the smallest effect size of interest and use equivalence bounds for meaningful null claims.
5. Report all planned conditions, failures, exclusions, and stopping decisions in a machine-readable results manifest.

### Phase 10: Reporting and Compatibility

1. Wrap current `activation_patching_test()`, `ablation_test()`, and steering functions around the new schemas where possible.
2. Mark older results lacking split/control metadata as exploratory.
3. Generate a claim table with columns: claim, design hash, intervention, outcome, effect/CI, controls passed, transfer scope, and interpretation limit.
4. Add:

```bash
deception causal validate-design CONFIG
deception causal run CONFIG --run-dir PATH
deception causal report RUN_DIR
```

## Verification

```bash
pytest tests/causal/ -v
pytest tests/events/ tests/dyads/ tests/negotiation/tom/ -v
pytest tests/ -v
```

Before expensive runs, use a tiny deterministic hooked model or synthetic residual fixture in which one known dimension controls a known token. The suite must recover that effect, reject random directions, and preserve pairing.

## Acceptance Criteria

- Every confirmatory result references a frozen design hash and disjoint family-grouped splits.
- Layer, stage, direction, and coefficient selection never use confirmatory outcomes.
- Sham, random, shuffled-label, nuisance, and positive controls execute through the same hook path.
- Logit effects, generated behavior, QC, task utility, and dyadic consequences are separately reported.
- Statistics cluster at the unit of randomization and account for declared multiplicity.
- ToM and bilateral causal claims use randomized/clamped variables with exact event lineage.
- Null and adverse findings appear in the report rather than being filtered by `passed=True`.

## Risks and Mitigations

- **Intervention leaves the model manifold:** use small dose ranges, resampling/patching alternatives, fluency and norm diagnostics, and avoid interpreting extreme coefficients.
- **Direction is a nuisance proxy:** match contexts, residualize/evaluate nuisance directions, and require transfer plus selectivity.
- **Researcher degrees of freedom:** frozen manifests, locked confirmation, and complete condition reporting.
- **Behavioral stochasticity:** paired repeated sampling and dyad-clustered intervals, with a fixed budget.
- **Causal overstatement:** bind conclusions to the manipulated model, layer, token stage, scenario family, and measured outcome.

## Handoff to Plan 6

Plan 6 may start after one detector/monitor has a locked feature pipeline, threshold calibration split, and causal control report. Adaptive adversaries must attack a frozen monitor and must not receive confirmatory labels or hidden monitor parameters.
