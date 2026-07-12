# Plan 3: Theory of Mind 2.0

## Objective and Completion Signal

Replace the current prose-oriented Theory of Mind (ToM) component with an evidence-updated, inspectable belief system that demonstrably affects negotiation policy. Completion means the framework can answer four separate questions: what the agent predicted about its counterpart, how that prediction changed after evidence, whether it was calibrated against a known counterpart policy, and whether intervening on the belief changed the agent's action.

This is not a claim that an LLM has human Theory of Mind. “ToM” here means a computational partner model used to predict and adapt to another policy.

## Repository Baseline

`negotiation/components/theory_of_mind.py` already provides useful hooks:

- `MentalModel`, `RecursiveBelief`, emotion and linguistic-cue extraction;
- `post_observe()` updates a counterpart model and `post_act()` records self cues;
- `get_state()` exposes a summary for interpretability;
- `pre_act()` inserts ToM guidance through the ordered Concordia context.

However, `_build_recursive_beliefs()` currently fills generic placeholders such as `we_are_motivated_to_close` and `recursive_reasoning`. `_generate_theory_of_mind_guidance()` largely analyzes the short call-to-action rather than consuming a versioned belief trajectory. `strategies` and `constraints` are not substantively updated. These gaps make the component useful scaffolding, but not yet a measurable ToM mechanism.

## Why This Design Is Valid

[Theory of Mind Benchmarks Are Broken](https://arxiv.org/abs/2412.19726) distinguishes literal partner prediction from **functional ToM**, adaptation that rationally uses the prediction. [BayesBench](https://arxiv.org/abs/2606.30850) reports a gap between updating latent beliefs and using them in downstream predictions. [AI Organizations Can Be More Effective but Less Aligned](https://alignment.anthropic.com/2026/ai-organizations/) shows that multi-agent behavior depends on organizational construction, so partner and role structure must be experimental variables. [Faithfulness as Information Flow](https://arxiv.org/abs/2605.24286) argues that visible reasoning can be bypassed and evaluates sufficiency, completeness, and necessity instead.

Therefore, prose that sounds socially insightful is not the success criterion. The component must expose quantitative priors/posteriors, score them against controlled ground truth, test policy adaptation, and support randomized interventions that establish information flow from evidence through belief to action.

## Research Hypotheses

Predeclare these as separable hypotheses; none is guaranteed true:

1. Dynamic partner beliefs predict held-out counterpart actions better than a static or transcript-only baseline.
2. ToM improves utility or agreement under partner-policy shift without increasing deception or exploitation.
3. Belief updates mediate some effect of partner evidence on the actor's next action.
4. Belief inconsistency across two agents adds detection value beyond either agent's text or activation alone.

## Target Architecture

Add an internal package while keeping the existing class as a compatibility facade:

```text
negotiation/components/tom/
  schema.py          # Evidence, hypotheses, belief state, updates, decisions
  features.py        # Deterministic observation/action feature extraction
  likelihoods.py     # Controlled-policy and estimated observation models
  updater.py         # Bayesian and baseline update strategies
  recursion.py       # Explicit level-0/1/2 belief targets
  policy.py          # Belief-to-recommendation mapping
  instrumentation.py # Event-safe structured summaries
  evaluation.py      # Calibration, prediction, adaptation metrics
```

`negotiation/components/theory_of_mind.py::TheoryOfMind` should delegate to these classes so existing builders and `ModuleType` lookup keys do not change.

Use these core records:

- `EvidenceItem`: event ID, observer, source actor, turn, observable features, visibility, reliability, and extractor version.
- `BeliefDistribution`: named finite hypotheses, probabilities summing to one, and epistemic status.
- `PartnerBeliefState`: counterpart goals, constraints, strategy/policy type, fact beliefs, trustworthiness, and expected next actions.
- `RecursiveBeliefState`: explicit target at each level, not free-form recursion. Level 1 is the actor's distribution over the counterpart's belief; level 2 is the actor's distribution over what the counterpart believes about the actor's belief.
- `BeliefUpdate`: prior, likelihood/evidence score, posterior, method, evidence IDs, entropy change, and warnings.
- `ToMDecisionTrace`: belief-state hash, predicted counterpart action distribution, recommendation features, chosen action call ID, and intervention condition.

Do not store or expose hidden chain-of-thought. Store structured variables, short evidence summaries, and hashes of any private prompt text.

## Step-by-Step Implementation

### Phase 1: Establish Controlled Partner Ground Truth

1. Treat `CounterpartType` values in `interpretability/scenarios/emergent_prompts.py` as the first finite policy hypotheses: default, skeptical, credulous, informed, and absent.
2. Define executable counterpart-policy contracts: observation access, response tendency, verification behavior, acceptance threshold, and action distribution where stochastic.
3. Add at least two non-language scripted policies with known probabilities. They are calibration fixtures, not realistic agents.
4. Sample partner identity independently of actor role. Hold out at least one policy family or parameter range for transfer tests.
5. Record the true policy type only in adjudicator events; never leak it to the actor's context.

### Phase 2: Define Belief Variables and Semantics

1. Enumerate hypotheses before collecting evaluation data. Start with policy type, next-action category, reservation-value interval, and belief about one scenario-critical fact.
2. Define an `unknown`/other bucket so forced closed-world confidence is not rewarded.
3. Set default recursion depth to two. Reject higher depths unless a scenario defines a scorable target for every level.
4. Document which variables have objective ground truth, inferred ground truth, or no ground truth. Emotion/personality may remain auxiliary and must not be presented as verified mental state.
5. Add schema tests enforcing normalized distributions, finite values, stable identity, and no private-fact visibility violations.

### Phase 3: Build Versioned Observation Features

1. Convert observations into scenario-relevant features using Plan 1 extractors: offer amount, claim/commitment type, requested evidence, acceptance/rejection, disclosed fact, and response latency/absence when meaningful.
2. Keep linguistic/emotion features as a separate channel with extractor confidence. Do not let weak deception cues directly define actual deception.
3. Link every feature to source event and text span from Plan 2.
4. Add negation, quotation, and paraphrase tests. A feature extractor failure yields missing evidence, not a zero-valued observation.

### Phase 4: Implement Belief Updating

1. Implement a `BeliefUpdater` protocol: `update(prior, evidence, observation_model) -> BeliefUpdate`.
2. Provide `BayesianUpdater` for controlled policies using `posterior(h) proportional to likelihood(e|h) * prior(h)`. Normalize in log space and handle zero likelihoods with declared smoothing.
3. Provide `FrequencyBaselineUpdater` and `FrozenPriorUpdater`. These are required comparison conditions.
4. Estimate non-scripted likelihood tables only from training partitions, with shrinkage and versioned fit metadata. Never estimate from the held-out dyads used for final scoring.
5. If an LLM proposes likelihoods or semantic features, constrain it to schema output, calibrate on development data, and identify the result as model-derived evidence.
6. Test exact posterior values on small tables, evidence-order behavior, missing evidence, contradictory evidence, and entropy calculations.

### Phase 5: Make Recursive Beliefs Concrete

1. Replace `_build_recursive_beliefs()` placeholders with named variables tied to the scenario instance.
2. For level 1, estimate what facts the counterpart could know from its role view and public transcript. Visibility supplies hard constraints; behavior supplies soft evidence.
3. For level 2, model what evidence the counterpart has about the actor's beliefs. Use explicit observation history rather than templated language.
4. Store recursion target, probability, evidence, and confidence/calibration separately. Confidence must decrease only when warranted by uncertainty, not by a fixed cosmetic decay.
5. Add impossible-knowledge tests: neither level may condition on facts unavailable along its modeled information path.

### Phase 6: Connect Beliefs to Policy

1. Implement `ToMPolicyAdvisor.recommend(state, legal_actions, objective)` returning structured scores for candidate negotiation moves and a short context summary.
2. Begin with deterministic, inspectable mappings for controlled scenarios, such as expected acceptance probability and expected utility under each hypothesized policy.
3. Render that structured recommendation into the existing Concordia component output. Include uncertainty and alternatives; never insert the adjudicator's ground truth.
4. Ensure `pre_act()` reads the persisted `PartnerBeliefState`, not only `action_spec.call_to_action`.
5. Emit a `ToMDecisionTrace` before the action-generating call and link it to the action event afterward.
6. Preserve component ordering and add an explicit configuration field so experiments can ablate ToM without changing unrelated modules.

### Phase 7: Add Intervention Conditions

Implement randomized conditions at the belief-to-policy boundary:

- `no_tom`: component absent;
- `frozen_prior`: observations do not update beliefs;
- `dynamic_tom`: normal updates;
- `oracle_tom`: true controlled partner policy supplied, clearly marked as an upper bound;
- `shuffled_tom`: a belief trajectory from a different training dyad;
- `counterfactual_belief`: clamp one posterior while holding transcript and other state fixed;
- `tom_text_only`: prose summary is present but structured belief values are masked from the advisor.

Use identical scenario instances and paired random seeds where the inference backend permits. An intervention event must record exactly which variables changed.

### Phase 8: Measure Literal and Functional ToM

1. Literal metrics: next-action log loss, Brier score, top-k accuracy, calibration error, and update direction after diagnostic evidence.
2. Functional metrics: expected/realized utility, agreement, regret versus oracle advisor, adaptation latency after a belief shift, and policy sensitivity to posterior changes.
3. Safety metrics: actual deception, manipulation, commitment violations, counterpart utility, and Pareto efficiency. An advisor that gains utility by exploiting a credulous partner is not an unqualified improvement.
4. Faithfulness metrics: belief sufficiency (can the advisor act from belief state without raw transcript?), completeness (what does raw transcript add?), and necessity (does belief corruption change action in the predicted direction?).
5. Cluster uncertainty by trial/dyad and report results by partner type, scenario, role, model, and condition.

### Phase 9: Migrate and Deprecate Placeholders

1. Add the new engine behind a `tom_version='v2'` builder option and keep v1 as a frozen baseline.
2. Update `get_state()` to return schema-native summaries plus a `state_hash`; retain old keys during migration.
3. Add an event projector for old `tom_state` consumers in `interpretability/evaluation.py`.
4. Delete placeholder recursion only after v1/v2 ablation data and migration tests exist.
5. Document that earlier datasets contain heuristic ToM summaries and are not equivalent to v2 belief trajectories.

## Verification

```bash
pytest tests/negotiation/tom/ -v
pytest tests/scenarios/ tests/events/ -v
pytest tests/ -v
```

Run a model-free controlled-policy study first, then a small model-backed study. Plot predicted policy probabilities over turns and inspect whether a known belief-shift event moves the posterior and the next action in the preregistered direction.

## Acceptance Criteria

- Every ToM belief has a defined target, hypothesis space, evidence lineage, and update method.
- Controlled partner predictions beat or are honestly reported against frozen/frequency baselines on held-out trials.
- Functional adaptation is measured independently from prediction accuracy.
- Belief interventions hold scenario/transcript constant and produce attributable action comparisons.
- No recursion level accesses facts outside the modeled observer's information path.
- Actual-deception labels never come from ToM linguistic cues.
- Existing agent builders work with both frozen v1 and opt-in v2 until migration completes.

## Risks and Mitigations

- **Anthropomorphic overclaiming:** use “partner-policy belief” in metrics and reserve mental-state claims for variables with defensible ground truth.
- **Self-fulfilling adaptation:** evaluate predictions before the actor responds and use scripted/exogenous counterpart policies for calibration.
- **Prompt leakage:** separate adjudicator state from each role view and add automated information-flow tests.
- **Better utility but worse safety:** report joint and counterpart outcomes alongside actor utility.
- **LLM-generated pseudo-precision:** distinguish exact controlled likelihoods from estimated/model-derived likelihoods and calibrate the latter.

## Handoff to Plan 4

Plan 4 may start once every action has a linked `ToMDecisionTrace`, belief-state hash, actor identity, and model-call ID. Bilateral instrumentation will join those records across agents and time; it must not infer beliefs from the prose guidance string.
