# Concordia 2.4 Compatibility Audit

## Scope and Pins

The official source is [`google-deepmind/concordia`](https://github.com/google-deepmind/concordia). The migration targets the latest stable release, [`v2.4.0`](https://github.com/google-deepmind/concordia/releases/tag/v2.4.0), commit `702998f57da71f87bf4e607abc1325ee51cca21f`, released March 6, 2026. The latest audited default-branch commit was [`361ae4192b0701e6608963a5a968fd1e2006d3a5`](https://github.com/google-deepmind/concordia/commit/361ae4192b0701e6608963a5a968fd1e2006d3a5), dated July 7, 2026. It contains unreleased changes while `setup.py` still reports version 2.4.0, so it is evidence for future work, not the dependency target.

Normalized source comparison established that `concordia_mini` is based on official `v2.1.0`, commit `bc2b1fa1479529bad05ec693c68733c407dbdb15`, with a Python 3.10 `StrEnum` fallback and a corrected error message among the inspected core files. It was not compatible with the current API despite having no explicit upstream version marker.

## Applied Decisions

| Decision | Status | Reason |
|---|---|---|
| Pin `gdm-concordia==2.4.0` | Fixed | Uses an immutable released API instead of silently extending an old fork. |
| Require Python 3.12+ | Fixed | Concordia 2.4 declares Python `>=3.12`; pretending to support 3.10 would leave an untestable dependency contract. |
| Store tag/main commit metadata | Fixed | `config/concordia_runtime.py` makes provenance and runtime drift testable. |
| Test upstream ActionSpec, sampling, and repeated GM memory | Fixed | These are the most consequential changed contracts for this repository. |
| Move project imports to `concordia` | Fixed | `rg 'concordia_mini' negotiation interpretability` finds no live imports, so constructed entities share the official type system. |
| Retain `concordia_mini` temporarily | Deferred | Existing external users may import it; removal requires a deprecation release and import inventory. |

## Breaking and Behavioral Differences

### Runtime and Language Models

The official [`LanguageModel.sample_text`](https://github.com/google-deepmind/concordia/blob/v2.4.0/concordia/language_model/language_model.py) adds `top_p` and `top_k`; `InteractiveDocument` forwards them. Default temperature changed from `0.5` to `1.0`. Every local TransformerLens, hybrid, Ollama, and evaluator adapter must accept the full signature. Experiments must record explicit temperature, `top_p`, `top_k`, seed, model revision, and Concordia version because the default change affects both generated behavior and captured activation distributions.

Provider adapters moved from `concordia.language_model.*` into `concordia.contrib.language_models.*`. Local wrapper imports must either use those supported locations or implement the official abstract interface.

### Action Specifications and Acting Context

`ActionSpec.to_dict()` now emits JSON-safe output-type strings and option lists, and the engine serializes action specs as JSON while preserving a legacy parser. The type, parser, next-action prompt, and switch component must move together; partial migration can make the game master request or parse the wrong action.

`ConcatActComponent` replaces only the `{name}` placeholder rather than calling unrestricted `str.format`, persists its component order and options, exposes that order, and returns `nan` for invalid float output instead of silently manufacturing `0.0`. The last change prevents failed parsing from becoming a valid scientific measurement.

### Memory, State, and Lifecycle

`AssociativeMemoryBank` adds `allow_duplicates` and batched writes. Agent memories normally deduplicate; game-master memories must use `allow_duplicates=True` because identical offers can legitimately recur across rounds. Otherwise a later action can disappear and event resolution can read stale state.

Concordia 2.4 persists meaningful state for constant, observation, question, concatenation, agent, and game-master components. Custom negotiation components need mapping-based, JSON-compatible round-trip state. Wall-clock timestamps are not reproducible simulation state and must be replaced by an injected clock or logical turn index.

Entity `act()` and `observe()` reset the phase after exceptions. This prevents one component failure from leaving the agent permanently stuck outside `READY`.

### Game Master and Privacy

Thought-chain helpers moved into `concordia.components.game_master.event_resolution`. Event resolution filters putative events by active player before selecting the newest event, preventing one player's stale action from being resolved for another.

`MakeObservation` uses a thread-safe shareable queue and supports disabling LLM fallback. Negotiation experiments should pass `allow_llm_fallback=False`; otherwise an unconstrained game-master call may invent or repeat private facts and contaminate information-asymmetry experiments.

The sequential engine rejects unknown actor names, ignores empty observations, and increments skipped setup steps. These are correctness changes, not UI-only differences.

## Local Findings and Disposition

1. **Deferred:** the interpretability runner constructs a negotiation game master but does not execute the normal GM act/observe loop. Reading untouched custom module state cannot provide adjudicated labels. Execute the GM or name the rule evaluator as the oracle.
2. **Fixed:** the runner now creates GM memory with `allow_duplicates=True`, independently of actor memory.
3. **Partially fixed:** `NegotiationStateTracker` now accepts an injected clock and round-trips complete mapping state, but the live GM still does not initialize or advance it from resolved negotiation events.
4. **Fixed:** automatic GM-module selection now occurs before observation and event-resolution component lists are assembled.
5. **Fixed:** custom mediator instructions use Concordia 2.4's mapping state contract, and `MakeObservation` disables LLM fallback to prevent fabricated private context.
6. **Active data-compatibility risk:** changed sampling defaults invalidate direct comparison with old activation datasets. Upgraded experiments require fresh data or an explicit legacy condition.

## Upstream Verification

The pristine Concordia 2.4 source passed 34 selected upstream tests covering engine, agent components, event resolution, and sequential execution under Python 3.13 during this audit. Local contract tests are in `tests/test_concordia_compat.py`.

The long-term architecture should depend on the pinned official package and keep only domain-specific negotiation/interpretability code locally. If vendoring is retained for a deployment constraint, it needs an automated normalized-diff test against the tagged source and preserved Apache headers.
