# Plan 7: Benchmark Packaging and External Research Interface

## Objective and Completion Signal

Package the mature framework as a versioned benchmark that outside researchers can install, validate, run, score, and audit from a clean checkout. Completion means a release has stable task contracts, checksummed data/artifacts, leakage-resistant splits, reference baselines, reproducible event logs, documentation, and an explicit governance process for new versions.

This is deliberately last. A benchmark freezes assumptions and creates incentives; publishing unstable labels or confounded splits would amplify errors rather than research value.

## Repository Baseline

The project already exposes the `deception` CLI, deterministic scenario functions, activation datasets, probe training, causal tools, and analysis scripts. There is also an in-progress ESR benchmark surface: `tests/test_esr_bench.py` expects an `esr_bench` package, and `examples/kaggle/` contains notebook templates. At the time of this plan, that package is not present in the tracked source tree and the tests describe an 18-item-per-task pilot.

Preserve that work, but do not present 18 multiple-choice items as the full multi-agent deception benchmark. Treat them as a smoke/pilot track while the dynamic, white-box, bilateral, and adaptive tracks are built from Plans 1-6.

## Why This Design Is Valid

[Datasheets for Datasets](https://arxiv.org/abs/1803.09010) recommends documenting motivation, composition, collection, uses, and limitations to improve transparency and accountability. [Inspect](https://inspect.aisi.org.uk/tasks.html) separates tasks, datasets, solvers, and scorers, while [evaluation logs](https://inspect.aisi.org.uk/eval-logs.html) retain auditable execution histories. [ControlArena](https://control-arena.aisi.org.uk/) separates settings, policies, monitors, protocols, and scorers for control evaluation. [SLEIGHT-Bench](https://alignment.anthropic.com/2026/sleight-bench/) demonstrates paired benign calibration, low-FPR reporting, a no-training canary, and controlled handling of dual-use benchmark data.

The benchmark should therefore expose modular tasks and scorers, preserve provenance-rich logs, document intended/non-intended uses, and protect final/adaptive evaluation material from casual training contamination.

## Benchmark Tracks

Version tracks independently; a method may enter only the tracks matching its access assumptions.

1. **ESR Static Pilot:** multiple-choice epistemic/social-state items from the existing draft. Purpose: installation, schema, and black-box reasoning smoke test.
2. **Behavior Verification:** transcript plus authorized scenario context; predict evidence-bearing behavior labels and outcomes.
3. **Sequential Detection:** score each turn using only information available by that turn; measure lead time and false alarms per dyad.
4. **Single-Agent White Box:** sender activations plus text and typed metadata.
5. **Bilateral White Box:** sender/receiver activations and structured belief transitions.
6. **Functional ToM:** predict counterpart policy/action and demonstrate adaptive policy use under held-out partners.
7. **Causal Validation:** run frozen interventions and controls against compatible open-weight models.
8. **Adaptive Control:** frozen monitor and protocol evaluation against bounded monitor-aware policies; restricted data where necessary.

Do not collapse these into one leaderboard number. Publish a scorecard by capability/access assumption.

## Target Package and Artifact Layout

```text
esr_bench/
  __init__.py
  schema.py             # Submission, prediction, task, score contracts
  registry.py           # Versioned task/track registry
  data.py               # Verified loaders; supports current pilot API
  tasks.py              # Prompt construction and deterministic grading
  kaggle_tasks.py       # Current notebook-compatible record builders
  scorers/
  adapters/             # Native runner, Inspect, optional external formats
  baselines/
  cli.py

benchmark/
  cards/                # Dataset/track/metric/threat-model documentation
  manifests/            # Version, split IDs, URLs, hashes, licenses
  configs/              # Frozen baseline and reference-run configs
  schemas/              # Published JSON Schemas
  reports/              # Generated reference results, not raw large data
  canary.txt
```

Add `esr_bench*` to setuptools discovery and a console entry such as `esr-bench = "esr_bench.cli:cli"`. Keep heavy evaluation dependencies optional; basic data loading and static scoring should not require TransformerLens.

Large event logs/activations belong in a versioned artifact host, not Git. Manifests in Git must include SHA-256, byte size, compression, media type, schema version, source run IDs, and access/license terms.

## Step-by-Step Implementation

### Phase 1: Write the Benchmark Contract Before Data Export

1. Define intended users, claims, threat models, access modes, and prohibited claims for each track.
2. Define the independent evaluation unit: scenario-instance family/dyad for dynamic tracks and item family for static tracks.
3. Freeze label terminology from Plan 1. Keep behavioral action labels, model belief verification, agent perception, model-judge annotations, and intent inference separate.
4. Define prediction schemas before building scorers: sample/event ID, score, optional subtype, coverage, method version, runtime/cost, and declared data access.
5. Define a valid submission as predictions plus a method card and environment manifest; hidden reasoning is never required.

### Phase 2: Stabilize the ESR Static Pilot

1. Inventory the existing notebook templates and `tests/test_esr_bench.py` expectations. Do not overwrite untracked user work.
2. Implement the minimal `esr_bench.data`, `tasks`, and `kaggle_tasks` APIs expected by those tests or adapt them through a reviewed migration.
3. Assign immutable item IDs and family IDs, validate option uniqueness and one correct answer, and keep item IDs out of prompts.
4. Separate development examples from held-out evaluation; 18 items per task are insufficient for strong ranking claims, so label the first release `pilot` and report binomial uncertainty.
5. Add contamination and memorization warnings. Static results must not be used as evidence that a method detects emergent multi-agent deception.

### Phase 3: Define Versioning and Manifests

1. Use semantic versions for code interfaces and independent date/semantic versions for dataset releases. A label or split change requires a new dataset version.
2. Create `BenchmarkManifest` with track versions, scenario spec hashes, event schema, model/tokenizer revisions, capture stages, label sources, QC policy, family splits, canary, licenses, and artifact checksums.
3. Sign release manifests where project infrastructure permits and archive immutable copies.
4. Implement `esr-bench verify MANIFEST`, which validates schema, hashes, row counts, unique IDs, split disjointness, and artifact compatibility before loading tensors.
5. Never silently download or execute pickle artifacts. New tensors use non-pickle formats; legacy trusted `.pt` files require an explicit flag and warning.

### Phase 4: Build Release-Quality Data

1. Generate fresh runs from Plans 1-4. Do not promote datasets created before label, sample-type, or role-confound fixes.
2. Apply QC before split assignment and retain exclusion reason counts. Keep raw events immutable and create release projections by hash.
3. Deduplicate by compiled base instance, paraphrase family, transcript similarity, and model-call lineage.
4. Create train/development/public-test/private-test partitions grouped by family/dyad and stratified by scenario, label, role, partner policy, and model where feasible.
5. Manually audit a statistically meaningful stratified sample with blinded reviewers. Report inter-annotator agreement and adjudication without overwriting deterministic label provenance.
6. Validate that private facts and hidden labels do not leak into participant inputs or public-test features.

### Phase 5: Implement Task and Scorer APIs

1. Define a `BenchmarkTask` protocol with dataset iterator, allowed input view, prediction schema, scorer set, and resource metadata.
2. Implement pure scorers that operate on prediction files and hidden/public labels without loading models.
3. For detection, report AUROC, AUPRC, calibration, TPR at 1%/5% FPR, and coverage. For sequential tasks, add false alarms per dyad and detection lead time.
4. For ToM, report log loss, Brier/calibration, partner-action prediction, regret, and safety outcomes. For control, report safety-usefulness frontiers, audit burden, latency, and cost.
5. Bootstrap confidence intervals at family/dyad level and publish per-scenario/role/model slices. Reject turn-level IID intervals.
6. Add a machine-readable `ScoreReport` that contains every metric, inclusion count, threshold source, and benchmark version.

### Phase 6: Add Native and Inspect Adapters

1. Build a native offline adapter first so the benchmark does not depend on an external framework.
2. Add an optional Inspect adapter mapping benchmark samples, solver outputs, event references, and scorers to Inspect tasks/logs.
3. Preserve native IDs and source hashes through all adapters; adapter output must round-trip to the same score report.
4. For control tracks, optionally expose a ControlArena-compatible monitor/protocol wrapper without making ControlArena a core dependency.
5. Add contract tests using tiny fixtures rather than network/model calls.

### Phase 7: Establish Reference Baselines

Run and publish frozen configurations for:

- majority/scenario/length/style baselines;
- transcript lexical and embedding classifiers;
- model-judge baseline with model and prompt version;
- sender linear probe, receiver probe, bilateral score fusion;
- ToM-only and ToM-plus-activation detectors;
- causal random/sham/target directions;
- static, awareness-only, and adaptive monitor conditions.

Every baseline report includes training data, split unit, layer/stage, threshold calibration data, compute, runtime, seeds, and uncertainty. Do not tune baseline choices on private test results.

### Phase 8: Documentation and Researcher Experience

Write concise cards for each track:

1. motivation and research questions;
2. composition and collection process;
3. scenario/agent/model coverage;
4. label construction and uncertainty;
5. splits, known leakage risks, and QC;
6. allowed inputs and deployment assumptions;
7. metrics and interpretation;
8. ethical/dual-use considerations;
9. intended uses, prohibited overclaims, and known limitations;
10. citation, license, contact, and change log.

Provide a quickstart that finishes without GPU for static data and scoring, plus a separate white-box guide with explicit hardware/model requirements.

### Phase 9: CI, Reproduction, and Release

1. CI runs schemas, pilot loaders, deterministic scorers, split audits, manifest verification, and a tiny event replay.
2. A release candidate is built in a clean environment and installed from its wheel.
3. Reproduce at least one reference score from downloaded artifacts and compare within a declared tolerance.
4. Test optional adapters independently and ensure the core package imports without them.
5. Publish checksums, cards, score reports, code tag, environment lock/constraints, and a known-issues list together.

Recommended commands:

```bash
esr-bench list
esr-bench info TRACK@VERSION
esr-bench verify benchmark/manifests/RELEASE.json
esr-bench evaluate TRACK@VERSION predictions.jsonl
esr-bench reproduce BASELINE_CONFIG
```

### Phase 10: Governance and Hidden-Test Policy

1. Establish maintainers for schemas, data, metrics, and security review.
2. Require proposed scenarios to pass Plan 1 validation, event provenance, family deduplication, role counterbalancing, and label audit.
3. Rotate/add private-test families on a published schedule while preserving old versions for longitudinal comparison.
4. Rate-limit or batch private evaluation to reduce test-set hill-climbing. Return aggregate/sliced scores without item-level hidden labels.
5. Document contamination reports, corrections, withdrawals, and version deprecation. Never rewrite historical scores under a changed dataset.
6. Keep adaptive artifacts encrypted or controlled when release would materially improve evasion capability; publish enough aggregate evidence and benign fixtures to audit the protocol.

## Verification

```bash
python -m pip install -e '.[dev]'
pytest tests/test_esr_bench.py tests/benchmark/ -v
pytest tests/ -v
python -m build
```

Install the built wheel in a clean environment, run the static quickstart offline, verify one downloaded manifest, score a fixture submission, and replay one dynamic sample lineage from source events.

## Acceptance Criteria

- The current ESR pilot API works and is labeled as a pilot with appropriate uncertainty.
- Each mature track has a versioned task contract, allowed-view declaration, data/metric card, and grouped split audit.
- All public artifacts are checksummed, licensed, provenance-linked, and safe to load.
- Reference baselines reproduce from frozen configs without private-test tuning.
- Low-FPR, calibration, sequential, utility, and cost metrics are reported where applicable; no single composite score hides tradeoffs.
- Clean-wheel, offline-scoring, optional-adapter, and event-lineage tests pass.
- Canaries, no-training notices, private-test policy, dual-use review, and correction/version governance are documented.

## Risks and Mitigations

- **Premature leaderboard:** release static and small dynamic data as pilots until power, diversity, and external audit gates pass.
- **Benchmark overfitting:** private family-grouped tests, limited feedback, version rotation, and transfer tracks.
- **Label controversy:** evidence-bearing multi-source records and separate scoring for verifiable behavior versus inferred belief/intent.
- **Access inequity:** maintain black-box/text tracks and downloadable precomputed activation tracks alongside expensive live white-box tasks.
- **Artifact misuse or contamination:** canaries, access tiers, non-training terms, checksums, and deliberate release review.

## Final Program Gate

The benchmark is ready for a conference-paper evaluation only when Plans 1-6 have produced at least one fresh preregistered study with audited labels, replayable lineage, paired roles, grouped splits, causal controls, adaptive transfer, and complete null/adverse reporting. The paper should lead with the scientific result; the benchmark is the reusable artifact that makes that result testable by others.
