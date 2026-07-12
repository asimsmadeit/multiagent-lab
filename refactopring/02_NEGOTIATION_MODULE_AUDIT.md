# Negotiation Module Audit

## Scope and Status

This audit covers the negotiation agents, all optional reasoning components, the game-master (GM) builder, and all negotiation-specific GM components. The code now imports the pinned `gdm-concordia==2.4.0` API and its offline contract suite passes. That establishes construction, lifecycle, state, and deterministic-behavior coverage; it does **not** establish that every GM domain method participates in Concordia's live event loop.

## Component Inventory

- Agent assembly: `base_negotiator.py`, `advanced_negotiator.py`, and `constants.py`.
- Core components: instructions, memory, and basic strategy.
- Optional components: cultural adaptation, temporal strategy, swarm intelligence, uncertainty awareness, strategy evolution, and theory of mind (ToM).
- Adjudication: `game_master/negotiation.py`, state tracking, validation, plus cultural, social, temporal, uncertainty, collective-intelligence, and strategy-evolution GM modules.
- Shared parsing: `utils/parsing.py`.

## Corrected and Verified

| Area | Result |
| --- | --- |
| Concordia compatibility | Agent and GM entities build and act against Concordia 2.4; custom instructions use the current state contract and the first GM action no longer miswires state as associative memory. |
| Configuration | Unknown modules, malformed module configuration, invalid strategies, invalid participant sets, and unsupported swarm agents fail clearly. Agent component names remain aligned with `ModuleType` and GM auto-detection. |
| State integrity | All nine agent components, six optional GM modules, the state tracker, and validator have JSON-safe state round trips. Component instances no longer share mutable runtime state. |
| Reproducibility | Uncertainty sampling and strategy evolution use injectable, locally seeded RNGs; memory, temporal state, and GM state accept injectable clocks. RNG state is serialized. |
| Negotiation logic | Offer parsing prefers the actual offer over unrelated monetary values; negated acceptance is rejected; numeric parser outputs are bounded; only an offer's recipient may accept or reject it. |
| Strategy behavior | Multi-turn actions stay in one strategy episode, transition bookkeeping is ordered correctly, crossover weights are normalized, and a zero reservation value still produces a usable initial target. |
| GM bookkeeping | Module state is isolated and serializable; offer/agreement restoration is complete; event and participant accounting is deduplicated; auto-detected modules are added before observation/event-resolution components are assembled. |

## Open Findings

### High: GM domain logic is not connected to resolved events

The six specialized modules expose `validate_action()` and `update_state()`, but their inherited Concordia lifecycle hooks return without invoking either method (`negotiation/game_master/components/gm_modules.py:145-165`). The GM only supplies these modules as prompt context (`negotiation/game_master/negotiation.py:324-343`). Direct API tests pass, while normal entity actions can leave module state unchanged. Add a typed event adapter that builds one `ModuleContext`, calls validators by priority before resolution, then dispatches the committed event to `update_state()` exactly once.

### High: Protocol state is not driven by the live negotiation

`NegotiationStateTracker` provides sound direct APIs for start, offer, accept, reject, round advance, and termination (`gm_state.py:109-285`), but the GM builder never starts a negotiation. Its `post_act()` explicitly does not parse actions (`gm_state.py:333-354`). Consequently, rendered GM context can report no active negotiation even while agents are acting. Initialize state during GM construction and update it from structured resolved events, not free-text keyword matching.

### High: Validation is advisory rather than an adjudication gate

`NegotiationValidator` appears in the event-resolution prompt, but no live path calls `validate_offer()` or blocks invalid agreements. Contract feasibility is also a placeholder (`gm_validation.py:254-268`), and enforcement only toggles an internal flag (`gm_validation.py:303-319`). Introduce domain schemas, explicit rejection outcomes, and integration tests proving invalid actions cannot mutate state.

### Medium: ToM outputs are largely template-driven

Recursive beliefs contain hard-coded propositions (`components/theory_of_mind.py:347-384`), while `pre_act()` analyzes the generic call-to-action instead of the stored counterpart model (`theory_of_mind.py:461-519`). `_detect_deception()` accepts `baseline_patterns` but never uses it (`theory_of_mind.py:272-345`). Replace templates with evidence-linked belief records, calibrated uncertainty, and explicit counterpart identifiers; test that changed observations change beliefs and advice.

### Medium: Configuration and tracked metrics are partially inert

- `SwarmIntelligence` validates and stores `consensus_threshold` and `max_iterations`, but decision construction performs one weighted vote without using either (`components/swarm_intelligence.py:277-298,377-427`).
- `TemporalStrategy.observe()` creates counterpart records but does not populate interaction count, concessions, outcomes, or exchanged value (`components/temporal_strategy.py:367-394`).
- GM cultural auto-detection assigns every detected participant `western_business` rather than reading component state (`game_master/negotiation.py:357-371`).
- Social-emotion and strategy classifiers remain keyword heuristics (`gm_social_intelligence.py:67-127`; `gm_strategy_evolution.py:117-167`), so their outputs need calibration or an explicit heuristic label in research metadata.

## Verification Evidence

`python -m pytest tests/negotiation -q` passes **20 tests**. Coverage includes base and advanced agent construction, all optional modules in one acting entity, GM auto-detection and first action, every component lifecycle, JSON state restoration, seeded behavior, parser edge cases, module isolation, participant authorization, validator persistence, and every GM module's direct validate/update/report interface.

Model-backed quality, protocol-level state transitions, and enforcement through Concordia's event loop remain unverified because the high-severity integration paths above do not yet exist.

## Recommended Order

1. Define typed negotiation actions and resolved events.
2. Wire one GM orchestration component to validate, resolve, and dispatch them.
3. Start and advance `NegotiationStateTracker` exclusively from those events.
4. Replace inert configuration and placeholder metrics with tested behavior.
5. Rebuild ToM around evidence-linked counterpart beliefs, then add calibration and behavioral-ablation tests.
