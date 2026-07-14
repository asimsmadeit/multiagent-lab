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

## Recovery Artifacts

Public experiment commands write `activation-recovery/3` snapshots for audit and manual salvage. They do not implement exact aggregate-schedule resume: completed-trial snapshots omit both an active runtime envelope and the full completed-result accumulator. Restarting either CLI starts a new run; flags such as `--resume` or `--resume-from` are intentionally rejected. Recovery collections can be inspected or restored into a runner, but must not be presented as a continued experiment or a publishable activation dataset.

## Architecture and Research Contracts

See `ARCHITECTURE.md` for the negotiation component boundaries, game-master responsibilities, interpretability data flow, and scientific invariants. The short version is that agent cognition and GM adjudication are separate, activations must be bound to the exact retained response, and evaluation splits must remain trial- and dyad-disjoint.

The packaged offline reference configuration is `config/reference_offline.json`. Dataset scope, labels, split policy, security, and external validation requirements are documented in `interpretability/data/DATA_CARD.txt`; release hashes are retained in `config/release_checksums.json`.
