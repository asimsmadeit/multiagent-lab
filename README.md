# Multiagent Emergent Deception

This repository studies emergent deceptive behavior in multi-agent negotiation and the internal model representations associated with it. Negotiation agents and game-master components live under `negotiation/`; scenario generation, activation capture, datasets, probes, and causal interventions live under `interpretability/`.

The runtime is pinned to `gdm-concordia==2.4.0` and requires Python 3.12 or newer. The local `concordia_mini/` package remains only as a temporary compatibility surface; live negotiation and interpretability code use the official Concordia package.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the offline suite with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/matplotlib \
  python -m pytest tests/ -v
```

Model-backed experiments require suitable Hugging Face weights, accelerator memory, or provider credentials. Never commit credentials, activation tensors, model checkpoints, private transcripts, or generated datasets. Treat legacy `.pt` inputs as trusted-only artifacts.

## Refactor Entry Points

Read `AGENTS.md` before changing the system. It defines the negotiation architecture, interpretability pipeline, scientific invariants, and validation commands.

The current audit is indexed at `refactopring/README.md`. The dependency-ordered target design is in `refactopring/05_IDEAL_REFACTOR_PLAN.md`, while detailed high-impact research implementation plans are under `implementation_plans/high_impact_framework/`.

Core refactor work should preserve trial/dyad grouped splits, keep actual-deception labels distinct from counterpart-perception labels, and ensure activations come from the same acting-model call as the stored response.
