# Plan 4: Bilateral Multi-Agent Instrumentation

## Objective and Completion Signal

Capture aligned behavioral, belief, and activation traces for **both** negotiation agents across the full interaction. Completion means every dyadic turn can be queried as: what actor A knew and believed before acting, which exact model call produced A's move, how B processed that move, how B's beliefs changed, and which call produced B's response.

The result must support fair comparisons among text-only, one-agent white-box, two-agent white-box, ToM-only, and fused detectors without role or dyad leakage.

## Repository Baseline

`InterpretabilityRunner._run_single_emergent_trial()` currently builds the main negotiator with `self.model` and a simpler counterpart with `self.fast_model`; comments explicitly note that the fast counterpart skips activation capture. `ActivationSample` has `counterpart_idx` and `counterpart_name`, but the runtime primarily creates main-agent samples. The wrapper exposes mutable “current activations,” so internal component calls can make call-to-sample alignment ambiguous. Prompt-only and hybrid capture paths also select different token positions.

Plans 1-3 provide typed scenario state, call-scoped events, and structured ToM traces. Do not implement bilateral capture by merely adding more fields to the current monolithic sample.

## Why This Design Is Valid

[NARCBench](https://arxiv.org/abs/2604.01151) shows that aggregating activations across agents can detect multi-agent collusion under environment shift, reports token-localized signals, and finds no single aggregation method dominates across collusion types. [AI Organizations Can Be More Effective but Less Aligned](https://alignment.anthropic.com/2026/ai-organizations/) finds that collective behavior differs from single-agent behavior and depends on system construction. [LIARS' BENCH](https://arxiv.org/abs/2511.16035) shows that transcript-only evidence can be insufficient when belief and private context matter.

This framework can extend those results in a distinct direction: emergent rather than instructed behavior, sequential negotiation rather than final group classification, controlled private facts, belief shifts, and explicit ToM state. Bilateral instrumentation is valuable only if it preserves these causal and temporal relationships; concatenating two unaligned final vectors would largely duplicate existing work.

## Research Questions

1. Does the receiver's internal response to a deceptive message add held-out detection signal beyond the sender's activation and transcript?
2. Does cross-agent belief disagreement predict later deception, failed agreement, or exploitation before either outcome is visible in text?
3. Are signals localized to the sender's production, the receiver's processing, or a temporal interaction between them?
4. Does bilateral signal transfer across roles, partner policies, scenarios, models, and adaptive monitor awareness?

## Target Architecture

Add:

```text
interpretability/dyads/
  schema.py          # ActorTurn, Reception, DyadLink, role assignment
  capture.py         # Call-scoped symmetric capture sessions
  alignment.py       # Token, message, turn, and belief-state alignment
  projections.py     # Event stream -> dyadic examples
  features.py        # Single, paired, relational, and temporal features
  splits.py          # Dyad/trial/template-safe partitions
  evaluation.py      # Baselines, transfer matrices, early detection
```

Canonical records should be normalized by actor-turn rather than “main” versus “counterpart”:

- `ActorTurn`: dyad, trial, round, turn ordinal, actor/recipient identities and roles, action call ID, pre-action belief hash, action event, and outcome references.
- `Reception`: recipient, delivered observation event, processing/component call IDs, post-observation belief hash, and next-action call ID.
- `ActivationView`: call ID, model/revision, hook/layer, capture stage, token indices/text spans, aggregation, and artifact reference.
- `DyadLink`: links a sent action to its reception and response, including causal parent event IDs.

Do not duplicate tensors in pair records; reference content-addressed artifacts from Plan 2.

## Step-by-Step Implementation

### Phase 1: Define Capture Modes and Threat Model

1. Add explicit modes: `both_white_box`, `actor_white_box`, `receiver_white_box`, and `text_only`.
2. Record whether agents share weights, tokenizer, architecture, and layer count. “Both white-box” does not imply a common representation space.
3. Separate agent identity, role, policy class, module configuration, and model instance in the schema.
4. Declare which mode each research claim requires. A detector requiring both activation streams is not deployable when counterpart internals are unavailable.

### Phase 2: Make Agent Construction Symmetric

1. Refactor trial construction into `build_dyad(role_assignment, actor_config, counterpart_config, capture_mode)`.
2. Permit the same advanced component set on either side; do not hard-code all cognitive modules to “Negotiator.”
3. Create separate call-scoped model adapters even when weights are shared. Each adapter must emit actor ID and call purpose into Plan 2 events.
4. Preserve an asymmetric lightweight-counterpart condition as an experimental baseline, not the default bilateral condition.
5. Test that swapping identities and roles swaps only declared private views/configuration, not labels or instrumentation availability.

### Phase 3: Randomize and Counterbalance Roles

1. For each compiled scenario instance, generate paired role assignments when logically valid: model/configuration A as potential deceiver then B as potential deceiver.
2. Randomize first mover independently where the scenario permits; otherwise record it as a fixed scenario property.
3. Stratify by scenario, incentive, counterpart policy, role, and model pairing before assigning train/development/test dyads.
4. Ensure a base instance, paraphrase sibling, mirrored role pair, or repeated stochastic rollout cannot cross a split boundary.
5. Add a leakage test that reconstructs family IDs and fails if any family appears in multiple partitions.

### Phase 4: Standardize Call-Scoped Activation Capture

1. Use the `GenerationRecord`/`model_call_id` boundary from Plan 2. Remove dyadic code paths that infer freshness from `call_count` or read a global current tensor.
2. Define capture stages consistently across backends:
   - `prefill_last`: last non-padding prompt token before generation;
   - `generated_last`: last generated token;
   - `generated_mean`: mean over generated tokens;
   - `evidence_span`: tokens aligned to a parsed claim/commitment from Plan 1;
   - `message_read_span`: receiver-side tokens corresponding to the partner message.
3. Store tokenizer offset mapping and exact token indices. If a backend cannot provide a stage, mark it unavailable; never substitute a differently defined tensor under the same name.
4. Tag component inference, verifier, judge, and monitor activations separately. Only `actor_action` and `counterpart_action` calls enter the primary negotiation detector.
5. Add synthetic tests with distinct sentinel prompts/calls to prove no wrapper overwrites or cross-actor contamination.

### Phase 5: Record Reception and Belief Transitions

1. When A's action is delivered to B, emit `ObservationDelivered` with A's action event as a parent.
2. Capture B's pre-observation and post-observation ToM state hashes, plus structured `BeliefUpdate` IDs.
3. Link B's next action to that reception even if it occurs in the next round or after an intervention.
4. Represent absence, timeout, refusal, and invalid output as typed events rather than missing rows.
5. Verify that receiver state never contains sender-private facts except through public communication or declared interventions.

### Phase 6: Build Dyadic Projections

Produce separate, versioned example tables:

1. `sender_turn`: predict the sender's current behavioral label.
2. `receiver_processing`: predict the label of the message B just received.
3. `send_receive_pair`: join A's production with B's processing of that exact message.
4. `temporal_window`: prior K paired turns used for early/longitudinal detection.
5. `dyad_outcome`: whole-trajectory classification/regression for agreement, regret, or exploitation.

Each row must contain family/split IDs, capture availability, label provenance, QC status, and the source event IDs. Filter primary probe training to `sample_type == 'negotiation'`.

### Phase 7: Implement Feature Families and Baselines

Evaluate these independently before fusion:

- transcript-only and scenario-metadata baselines;
- sender activation only;
- receiver activation only;
- sender and receiver score aggregation (mean, max, product, learned stacker);
- aligned concatenation when representation dimensions are compatible;
- relational features: cosine similarity after train-only alignment, difference/norm, canonical correlation, and belief divergence;
- temporal features: score slope, lagged receiver response, change-point, and time-to-first-detection;
- structured ToM only and ToM-plus-activation fusion.

Fit representation alignment and fusion models only on the training partition. Include parameter-matched controls so gains are not merely due to doubling feature count.

### Phase 8: Evaluate Generalization and Timing

1. Report within-distribution performance and a matrix holding out scenario, counterpart policy, role orientation, model pairing, and prompt-framing family.
2. Use AUROC and AUPRC, plus TPR at 1% and 5% FPR. For sequential use, report false alarms per dyad and detection lead time.
3. Compare final-turn, current-turn, and one-turn-ahead detection so post-outcome information is not mistaken for early warning.
4. Bootstrap confidence intervals by independent dyad/family, not by turn.
5. Run ablations for token stage, layer, sender/receiver side, ToM, and temporal history with multiple-comparison correction.

### Phase 9: Integrate with Existing Dataset and Analysis Code

1. Keep `ActivationSample.counterpart_idx` as a compatibility projection but make event IDs the canonical join key.
2. Extend `save_dataset()`/new event export to include dyadic manifests and artifact references.
3. Update probe loaders to reject actor-turn random splits when a dyad projection is selected.
4. Add a summary command:

```bash
deception dyads audit RUN_DIR
deception dyads build RUN_DIR --view send_receive_pair
deception dyads evaluate DATASET_MANIFEST --config CONFIG
```

The audit must report missing sides, duplicate receptions, role imbalance, label prevalence, per-backend capture stages, and split overlap.

## Verification

```bash
pytest tests/dyads/ -v
pytest tests/events/ tests/negotiation/tom/ -v
pytest tests/test_probes.py tests/test_unified_analysis.py -v
pytest tests/ -v
```

Run a two-round scripted-model fixture where every prompt and activation is uniquely identifiable, followed by a small open-weight model smoke study with both agents captured. Manually trace at least ten action-reception-response chains.

## Acceptance Criteria

- Every primary dyadic row links the sender call, receiver observation, receiver belief transition, and next response without index guessing.
- Both sides use identical capture-stage definitions or explicit missingness.
- Role-mirrored and repeated-instance families never cross splits.
- Role, first-mover, and capture-mode distributions are reported and counterbalanced where possible.
- Sender-only, receiver-only, text-only, and parameter-matched baselines accompany every fusion result.
- Early-detection results use only information available by the declared time.
- A black-box-partner mode remains usable and its deployment assumptions are explicit.

## Risks and Mitigations

- **Twice the capture cost:** support selected-layer pilots, streamed artifacts, and asymmetric ablations before full runs.
- **Incompatible model spaces:** prefer score-level aggregation and learned train-only alignment; never compare raw neuron coordinates across unrelated architectures as if aligned.
- **Role confounding:** paired role swaps, randomized assignment, and family-grouped splits are mandatory.
- **Receiver signal is just message content:** compare receiver activations to transcript and token-matched controls, and use causal reception interventions in Plan 5.
- **Duplication of NARCBench:** center the contribution on sequential belief dynamics, emergent incentives, receiver processing, causal testing, and adaptive evaluation.

## Handoff to Plan 5

Plan 5 may start when dyadic joins, capture-stage semantics, role/family splits, and structured belief interventions pass their audits. Causal tests must operate on these stable units rather than rediscovering alignment from row order.
