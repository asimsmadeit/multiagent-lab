# Plan 6: Adaptive Adversaries and Monitor-Aware Evaluation

## Objective and Completion Signal

Evaluate whether deception monitors remain useful when negotiating agents know monitoring exists or optimize against its feedback. Completion means a frozen monitor can be tested through a staged threat model, including held-out attack strategies and monitor architectures, while reporting both deception success and legitimate negotiation utility at fixed false-positive rates.

This is a controlled safety evaluation, not a system for optimizing real-world deception. Use synthetic negotiations, bounded action spaces, no real victims or infrastructure, and access-controlled adaptive artifacts.

## Repository Baseline

The framework already has behavioral labels, activation probes, steering/causal functions, multiple counterpart types, belief-shift injections, and multi-round negotiation. Plans 1-5 add executable outcomes, immutable traces, functional ToM, paired-agent signals, and locked detector pipelines. What is missing is a stable monitor/control interface and a threat model in which the evaluated policy responds strategically to that monitor.

Do not begin adaptive optimization against a detector still being tuned. Detector selection, feature preprocessing, threshold calibration, and confirmatory test data must be frozen first.

## Why This Design Is Valid

[Preference Learning with Lie Detectors](https://arxiv.org/abs/2505.13787) finds detector-based preference learning can produce policies with deception rates above 85% under some settings, showing that detector feedback can teach evasion instead of honesty. [Now You (Still) See Me](https://arxiv.org/abs/2606.09411) adversarially fine-tunes models that retain 58-79% secret recovery while evading ridge and held-out MLP probes, then restores detectability through recontextualized evaluation. [SLEIGHT-Bench](https://alignment.anthropic.com/2026/sleight-bench/) uses paired benign transcripts, evaluates at 1% FPR, and documents monitor blind spots plus data canaries. [ControlArena monitors](https://control-arena.aisi.org.uk/monitors.html) produce step-level suspicion scores that protocols can use to accept, resample, defer, or edit actions.

These results justify three design choices: evaluate at deployable low-FPR operating points; distinguish same-monitor evasion from transfer to held-out monitors/distributions; and measure the behavior of the full policy-monitor-protocol game, not static detector AUROC alone.

## Research Questions

1. How much detection performance is lost from mere monitor awareness versus feedback-driven adaptation?
2. Do text, sender activation, receiver activation, ToM-consistency, and ensemble monitors fail under the same adaptations?
3. Does adaptation hide deception, reduce deception, or simply destroy task performance?
4. Can recontextualization, receiver-side signals, or monitor ensembles recover detection on adaptive policies?
5. Which control protocol best trades safety, usefulness, latency, and audit burden?

## Target Architecture

Add two packages:

```text
interpretability/monitoring/
  schema.py       # MonitorInput/Output, calibration, protocol decisions
  base.py         # Monitor and ControlProtocol protocols
  text.py         # Transcript/evidence monitors
  activations.py  # Sender, receiver, and bilateral probe monitors
  tom.py          # Belief consistency/update monitors
  ensemble.py     # Train-only calibrated fusion
  protocols.py    # Observe, flag, resample, defer, trusted fallback
  evaluation.py   # Safety/usefulness and cost curves

interpretability/adversaries/
  schema.py       # Threat model, budget, strategy, adaptation manifest
  policies.py     # Awareness and bounded adaptive policy wrappers
  search.py       # Black-box candidate search in synthetic action space
  training.py     # Optional PEFT/LoRA adaptation behind explicit flag
  suites.py       # Strategy families and held-out recontextualization
  runner.py       # Frozen-monitor game execution
  reporting.py    # Evasion, transfer, and protocol reports
```

## Monitor and Protocol Contract

`MonitorInput` must specify the event cutoff and available view: public transcript, parsed Plan 1 evidence, sender activations, receiver activations, ToM state, scenario metadata, or combinations. This prevents future information leakage into online scores.

`MonitorOutput` contains normalized `suspicion_score` in `[0,1]`, calibrated threshold version, categorical reason codes, coverage/missingness, model/call ID, latency, token/compute cost, and source event IDs. Keep detailed internal explanations hidden from adaptive policies unless the threat model explicitly grants them.

`ControlProtocol` maps the current state and monitor output to one declared decision: accept, flag for audit, request another sample, defer to a trusted policy, or terminate the synthetic trial. Store the proposed and applied actions separately so safety and usefulness can both be evaluated.

## Step-by-Step Implementation

### Phase 1: Write the Threat-Model Matrix

1. Define policy capabilities: prompt only, sampling multiple candidates, score queries, access to explanations, access to weights/gradients, or parameter fine-tuning.
2. Define monitor knowledge: architecture known/unknown, feature set known/unknown, threshold known/unknown, and whether the monitor shares the actor model.
3. Define budgets: number of negotiations, score queries, candidate samples, trainable parameters, tokens, and wall-clock/compute cap.
4. Define the synthetic goal: obtain a scenario reward while executing a Plan 1 behavior label and minimizing observable suspicion. Never use an evaluator prompt that teaches broadly applicable real-world deception tactics.
5. Freeze each threat model in an `AdaptationManifest` with seed and data partitions.

### Phase 2: Implement Frozen Monitors

1. Wrap at least five baselines behind `Monitor`: text-only, sender-activation, receiver-activation, bilateral fusion, and ToM-consistency.
2. Add simple controls: response-length/style logistic model, scenario-only prior, random score, and oracle rule label as a nondeployable upper bound.
3. Fit preprocessing and fusion only on monitor-training data. Calibrate thresholds on a separate benign calibration set.
4. Freeze artifacts by hash. The adaptive runner must refuse a monitor whose files or config no longer match.
5. Emit scores online at the declared event cutoff and allow the identical monitor to run post hoc for reproducibility checks.

### Phase 3: Build Paired Benign Evaluation

1. For each adaptive scenario family, create a matched benign path with the same roles, task difficulty, length budget, and legitimate objective but without the misleading action.
2. Allocate base instances and paraphrase siblings together to train/adaptation/test partitions.
3. Calibrate and report thresholds at 1% and 5% FPR on benign data not used for monitor fitting or adversary adaptation.
4. Audit whether “benign” paths include negotiation hardball, uncertainty, refusal, or omission that a practical system should allow.
5. Report false alarms per turn and per dyad, since repeated monitoring compounds burden.

### Phase 4: Implement Staged Adaptive Policies

Implement tiers cumulatively so each added capability has an attributable effect:

1. `A0_unaware`: ordinary emergent-incentive policy.
2. `A1_awareness`: policy is told that a monitor may inspect its behavior, but receives no scores.
3. `A2_examples`: policy sees development examples with coarse pass/flag outcomes, not hidden test items.
4. `A3_black_box`: bounded candidate generation/search receives scalar monitor scores on adaptation instances.
5. `A4_surrogate`: policy trains or prompts against a surrogate monitor, then attacks a held-out target monitor.
6. `A5_parameter_adaptation`: optional PEFT/LoRA optimization for approved open-weight models within a fixed compute budget.

All tiers must keep the scenario objective, legal action space, and evaluation labeler fixed. Record whether lower detection results from honest behavior, successful evasion, invalid output, or task collapse.

### Phase 5: Implement Bounded Black-Box Search

1. Generate a fixed number of candidate negotiation actions from the same actor state.
2. Score candidates using only the access granted by the threat model.
3. Select with a preregistered objective combining scenario reward, verified behavior target, monitor score, and validity/task penalties.
4. Keep all candidates and scores in the event/artifact log so selection effects are auditable.
5. Compare against random-best-of-N and utility-only-best-of-N controls. More sampling alone can change behavior and must not be credited to adaptation.

### Phase 6: Optional Parameter Adaptation

1. Gate this phase behind an explicit configuration and resource approval. Use PEFT adapters rather than changing base checkpoints.
2. Partition data into adaptation-train, adaptation-development, and locked attack-test families. The target monitor's attack-test scores are never training rewards.
3. Train against one monitor or surrogate at a time and checkpoint by development objective, including KL/capability regularization.
4. Evaluate adapters on honest capability/negotiation utility, actual behavior, linguistic quality, held-out monitor types, new scenarios, new roles, and recontextualized prompts.
5. Store adapters in restricted experiment storage. Release aggregate results and vetted defenses before considering release of evasion artifacts.

### Phase 7: Test Transfer and Recontextualization

1. Create a full matrix: adapt against monitor X, evaluate against X and held-out monitors Y/Z.
2. Hold out at least one scenario, partner policy, style family, role orientation, and model family.
3. Add recontextualized test variants that preserve the private fact and objective but alter surface format, conversational history, and evidence placement.
4. Test whether receiver-side and bilateral monitors recover signal when sender-side probes are evaded.
5. Evaluate an ensemble only after component monitors are frozen; report correlated failures and leave-one-monitor-out results.

### Phase 8: Add Control Protocols

Implement in increasing intervention strength:

1. `observe_only`: score without changing behavior.
2. `flag_for_audit`: preserve action but count human/audit budget.
3. `resample`: request N new actions and choose the least suspicious valid one.
4. `defer`: use a separately defined trusted policy above threshold.
5. `terminate`: end only high-severity synthetic trials under a preregistered rule.

Evaluate protocols on the same frozen trajectories post hoc when decisions do not change future state, and rerun interactively when they do. Keep proposed action outcomes so a blocked successful attack is distinguishable from an attack that never occurred.

### Phase 9: Define Primary Metrics

Report:

- attack/deception success rate before and after protocol;
- TPR at 1% and 5% benign FPR, AUROC, AUPRC, and calibration error;
- same-monitor evasion and held-out-monitor transfer gaps;
- actor, counterpart, and joint utility; agreement and regret;
- false alarms per dyad, audit/defer rate, detection lead time;
- inference calls, tokens, latency, storage, and estimated cost;
- task invalidity and capability degradation;
- results by scenario, role, policy tier, model, and adaptation budget.

Produce safety-usefulness frontiers over thresholds rather than declaring one arbitrary threshold best.

### Phase 10: Data Governance and Release Safety

1. Add a benchmark canary and `NO_TRAIN` notice to adaptive test artifacts.
2. Separate public benign/baseline data, controlled-access adaptive transcripts, and nonreleased model adapters.
3. Redact credentials, infrastructure identifiers, and any accidental real-world sensitive content at ingestion.
4. Document dual-use review, access decisions, known blind spots, and the difference between synthetic negotiation evasion and real deployment risk.
5. Preserve encrypted raw adaptive artifacts when needed for reproducibility, with checksums recorded in the public manifest.

### Phase 11: CLI and Automation

```bash
deception monitors fit CONFIG
deception monitors calibrate MONITOR --benign-manifest PATH
deception adversary validate-manifest CONFIG
deception adversary run CONFIG --run-dir PATH
deception adversary report RUN_DIR --fpr 0.01
```

The runner must verify monitor hashes, split disjointness, budgets, and label/spec versions before consuming model compute.

## Verification

```bash
pytest tests/monitoring/ tests/adversaries/ -v
pytest tests/events/ tests/dyads/ tests/causal/ -v
pytest tests/ -v
```

Use scripted policies first: one honest, one obvious attack, one policy that becomes honest under monitoring, and one that only changes style. The report must classify these cases correctly before model-backed adaptation.

## Acceptance Criteria

- Frozen monitor hashes and locked attack-test families are enforced mechanically.
- Awareness, score-query, surrogate, and optional fine-tuning tiers are evaluated separately.
- Paired benign calibration supports 1%/5% FPR reporting and per-dyad false-alarm analysis.
- Evasion is counted only when the verified behavior succeeds and task validity/utility is preserved.
- Same-monitor and held-out-monitor results, costs, and protocol effects are all reported.
- Adaptive artifacts have canaries, access policy, and provenance; real-world harmful targets are excluded.
- The framework can reproduce an adaptive report from event logs and frozen artifacts.

## Risks and Mitigations

- **Detector feedback teaches stronger evasion:** bound the environment, restrict artifact access, and prioritize defense/recontextualization results.
- **Monitor awareness suppresses all behavior:** measure honest utility and distinguish honesty from paralysis.
- **Test-set optimization:** cryptographically hash locked manifests and prohibit target-monitor test feedback during adaptation.
- **False deployment confidence:** state access assumptions for each monitor and report low-FPR performance under transfer.
- **Compute explosion:** stage tiers, stop invalid configurations early, and use fixed query/sample budgets.

## Handoff to Plan 7

Plan 7 may start after schemas, frozen baselines, low-FPR metrics, adaptive threat tiers, and artifact-governance rules have survived at least one pilot. Packaging should publish stable evaluation interfaces and vetted data, not every internal adaptive artifact by default.
