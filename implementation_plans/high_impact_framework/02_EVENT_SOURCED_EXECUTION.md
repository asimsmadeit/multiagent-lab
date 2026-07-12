# Plan 2: Event-Sourced Execution and Provenance

## Objective and Completion Signal

Make an append-only event stream the canonical record of every experiment. A completed trial must be reconstructable into its transcript, agent-visible state, labels, outcomes, ToM state, monitor scores, and activation references without rerunning a model. Existing `ActivationSample` datasets remain supported as derived projections.

Completion means a smoke trial can be interrupted, resumed safely, replayed twice to byte-equivalent semantic projections, and traced from a reported metric back to the exact scenario version and model call that produced it.

## Repository Baseline

`interpretability/evaluation.py` currently performs orchestration and stores mutable wrapper state such as `_current_activations`. It later assembles `ActivationSample` records and writes a `.pt` dataset. The sample contains rich metadata, but the final row is not a complete execution history: component calls, full assembled context, exact call identity, private-view delivery, label derivation, and counterpart activations can be lost or ambiguously aligned.

Retain `InterpretabilityRunner` as the initial orchestrator. This plan changes its recording boundary before attempting a broader runner refactor.

## Why This Design Is Valid

[Event Sourcing](https://www.martinfowler.com/eaaDev/EventSourcing.html) records state changes as events, enabling complete rebuilds, temporal queries, and replay. [W3C PROV](https://www.w3.org/TR/prov-primer/) models entities, activities, and agents so a result's origin can be inspected, trusted, and reproduced. [Inspect evaluation logs](https://inspect.aisi.org.uk/eval-logs.html) retain sample histories and support later analysis, while its [scoring workflow](https://inspect.aisi.org.uk/scoring-workflow.html) permits rescoring completed runs.

For this repository, the key benefit is scientific rather than operational: a probe row must be attributable to the exact action-generating call, and new labelers or monitors should be evaluated post hoc on the same immutable behavior. Replay here means replaying recorded events into projections; it does **not** claim that stochastic model inference will regenerate identical tokens.

## Target Architecture

Add:

```text
interpretability/events/
  schema.py          # Event envelope, payload unions, schema upgrades
  writer.py          # Append-only JSONL writer and trial transactions
  reader.py          # Validation, streaming, filtering
  artifacts.py       # Content-addressed activation/text artifact storage
  projectors.py      # Transcript, trial, ActivationSample, and metrics views
  replay.py          # Deterministic projection runner and resume inspection
  provenance.py      # Lineage queries and W3C-PROV-inspired export
```

Use one JSONL stream per worker/pod and content-addressed artifact files. Merge streams only after validating identities and trial ownership. JSON event payloads should stay inspectable; large activations belong in compressed `.npz` files with dtype, shape, checksum, and hook metadata, avoiding pickle for new artifacts.

The event envelope must contain:

```text
schema_version, event_id, event_type, run_id, pod_id, trial_id,
dyad_id, sequence_num, recorded_at, actor_id, actor_role,
model_call_id, parent_event_ids, previous_event_hash,
payload, payload_schema_version, content_hash
```

`recorded_at` is provenance only and must be excluded from semantic-replay equality.

Initial event types:

- Run: `RunStarted`, `RunConfigFrozen`, `RunCompleted`, `RunFailed`.
- Scenario: `ScenarioInstantiated`, `PrivateViewAssigned`, `InterventionScheduled`.
- Agent: `AgentBuilt`, `ObservationDelivered`, `ComponentContextProduced`, `ToMStateUpdated`.
- Inference: `ModelCallStarted`, `ModelCallCompleted`, `ActivationCaptured`, `ModelCallFailed`.
- Interaction: `ActionProposed`, `ActionCommitted`, `TurnAdvanced`.
- Evaluation: `BehaviorLabeled`, `MonitorScored`, `OutcomeResolved`, `QualityControlApplied`.
- Intervention: `BeliefIntervened`, `ActivationIntervened`, `ProtocolDecisionApplied`.

Keep event payloads typed with Pydantic discriminated unions. Unknown future event types may be preserved by the reader but cannot be projected until an upgrader exists.

## Step-by-Step Implementation

### Phase 1: Specify Identity and Ordering

1. Define UUID-based `run_id`, deterministic `trial_id`, `dyad_id`, and unique `model_call_id`. A model call ID must be allocated before inference and passed into activation capture.
2. Define per-trial monotonic `sequence_num`. Do not infer scientific ordering from wall-clock timestamps.
3. Define actor identity separately from role; role can be randomized while identity remains stable.
4. Write `docs` inside the module docstring describing which events are facts, activities, and derived annotations.
5. Add tests for duplicate IDs, missing parents, sequence gaps, and illegal actor/role combinations.

### Phase 2: Implement Typed Events and Integrity

1. Create an immutable `EventEnvelope` and typed payload models in `schema.py`.
2. Canonicalize semantic JSON using sorted keys and compact separators. Hash `previous_event_hash + canonical_event_without_hashes` with SHA-256.
3. Validate the hash chain on read. A corrupted or truncated stream must fail with an event ID and byte/line location.
4. Add pure version-upgrade functions, for example `upgrade_behavior_labeled_v1_to_v2()`. Never mutate a released event schema in place.
5. Test serialization round trips, tampering, forward preservation, and deterministic hashing.

### Phase 3: Build the Store and Artifact Layer

1. Implement `EventWriter.append(event)` with a process-local lock, flush, and optional `fsync` at trial boundaries.
2. Implement a trial transaction convention: `TrialStarted` opens a trial and `TrialCompleted` seals it. On resume, an unsealed trial is either continued from its last legal event or marked failed; never silently duplicate it.
3. Store activation arrays by content hash under `artifacts/sha256/<prefix>/<hash>.npz`. Write to a temporary path, checksum, then atomically rename.
4. Store artifact metadata in `ActivationCaptured`: hook name, layer, token selection, aggregation, shape, dtype, tokenizer ID, model revision, artifact hash, and source `model_call_id`.
5. Support optional external artifact URIs but require a checksum. Event logs must remain valid when large artifacts are unavailable, while activation projections report the missing dependency.

### Phase 4: Define the Model-Call Boundary

1. Introduce a result object such as `GenerationRecord(text, model_call_id, prompt_messages, token_ids, activations, generation_config, usage)`.
2. Replace implicit reads of `_current_activations` with an explicit capture returned for that call or delivered through a call-scoped recorder.
3. Tag every call with a purpose: `actor_action`, `counterpart_action`, `tom_inference`, `component_analysis`, `belief_verification`, `plausibility_probe`, `judge`, or `monitor`.
4. Store the full assembled input or a content-addressed reference, not only `action_spec.call_to_action`. Redact secrets only in export projections, never in the canonical research record.
5. Assert that an action event and activation event share the same `model_call_id`. Reject activation-bearing negotiation samples that do not satisfy this invariant.

### Phase 5: Instrument One Trial End to End

1. At run start, freeze configuration: code commit/dirty flag, Python/package versions, model/tokenizer revisions, seeds, device/dtype, selected layers, scenario spec hashes, module order, and CLI arguments.
2. Emit the compiled `ScenarioInstance` from Plan 1 and one `PrivateViewAssigned` event per role.
3. Emit observation delivery and component context events in actual `ConcatActComponent` order. Store component names and hashes even when full text capture is disabled.
4. Wrap each generation call with started/completed/failed events and associate activations before committing the action.
5. Emit labels separately by source, followed by QC and outcome events. Do not modify an earlier label event when an additional judge finishes.
6. Seal the trial only after required artifacts pass checksum validation.

### Phase 6: Build Deterministic Projections

Implement pure projectors that consume an event iterator:

1. `TranscriptProjector`: public transcript plus actor/role/turn IDs.
2. `AgentViewProjector`: exactly what one actor had observed by a given sequence.
3. `TrialStateProjector`: commitments, interventions, outcome, and current turn.
4. `ActivationSampleProjector`: backward-compatible `ActivationSample` rows with explicit warnings/defaults for unavailable legacy fields.
5. `DyadProjector`: paired records needed by Plan 4.
6. `MetricInputProjector`: labels and outcome values, without computing experiment-wide statistics inside the event package.

Projectors must not call models, read global config, or use current time. Test idempotence and semantic byte equality.

### Phase 7: Add Resume and Post-Hoc Evaluation

1. On startup, scan streams and report completed, resumable, invalid, and failed trials.
2. Resume only at declared safe boundaries. Because in-memory Concordia state may not yet be fully serializable, the first release may restart the current unsealed trial while preserving the failure record; document this limitation.
3. Add a post-hoc command that appends a new annotation stream referencing existing event IDs. Never rewrite the original run to add a new monitor or labeler.
4. Record annotation code/model versions and source event hashes so rescoring remains attributable.

### Phase 8: Migrate Saving and Merging

1. Make event logging opt-in, compare projected rows against current direct `ActivationSample` creation, then switch it to default after parity.
2. Refactor `save_dataset()` to write a projection manifest plus tensors; retain the old `.pt` export behind an explicit compatibility command.
3. Update `interpretability/merge_parallel_results.py` to merge validated event streams by `(run_id, pod_id, trial_id)`, detecting overlapping trial ownership.
4. Add import tooling for legacy `.pt` datasets. Imported rows receive `LegacySampleImported` events and `provenance_complete=false`; do not fabricate model-call lineage.

### Phase 9: Add CLI and Lineage Queries

Provide:

```bash
deception events validate RUN_DIR
deception events replay RUN_DIR --projection transcript
deception events trace RUN_DIR --event-id UUID
deception events export-dataset RUN_DIR --format npz
deception events rescore RUN_DIR --labeler CONFIG
```

`trace` should show parent calls, scenario instance, actor view, activation artifact, label derivation, and downstream sample IDs.

## Verification

```bash
pytest tests/events/ -v
pytest tests/test_imports.py tests/test_unified_analysis.py -v
pytest tests/ -v
deception events validate path/to/smoke_run
deception events replay path/to/smoke_run --projection transcript
```

Also test deliberate process termination after each event type, duplicate worker output, a missing artifact, a modified JSON line, and an interrupted artifact write.

## Acceptance Criteria

- A recorded action, its input, output tokens, and activations share one call ID.
- Both complete and failed trials have explicit terminal status.
- Hash validation catches mutation and truncation.
- Two replays produce semantically identical transcript, label, outcome, and sample projections.
- New annotations can be appended without modifying original events.
- Legacy `.pt` export remains possible and is marked with its provenance limitations.
- Parallel merges reject duplicate trials and preserve per-trial order.

## Risks and Mitigations

- **Storage growth:** default to selected layers/token summaries, compress arrays, and make retention policy part of frozen config.
- **Sensitive private facts:** separate canonical access-controlled artifacts from redacted exports; never log credentials or API keys.
- **Event schema churn:** version payloads independently and require pure upgrade tests.
- **False determinism:** call the feature “projection replay,” not model replay, unless deterministic inference is independently demonstrated.
- **Runner complexity:** instrument existing boundaries first; do not combine this work with a full rewrite of `evaluation.py`.

## Handoff to Plan 3

Plan 3 may start when observation delivery, component order, model-call purpose, and typed state-update events are stable. ToM must publish structured belief updates into these events rather than only a prose component string.
