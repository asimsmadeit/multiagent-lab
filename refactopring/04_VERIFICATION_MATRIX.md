# Verification Matrix

This matrix records the reproducible checks completed during the Concordia 2.4 compatibility and refactor audit. Results marked **verified** were observed in the stated environment; aggregate counts combine disjoint suites rather than implying one monolithic run.

## Environment

| Check | Command | Observed result | Status |
| --- | --- | --- | --- |
| Python and Concordia runtime | `python --version && python -c "import concordia, importlib.metadata as m; print(m.version('gdm-concordia')); print(concordia.__file__)"` | Python 3.13.3; `gdm-concordia` 2.4.0 imported from the active virtual environment | Verified |
| Editable development install | `python -m pip install -e '.[dev]'` | Project and pinned Concordia 2.4.0 dependencies installed successfully | Verified |
| Pristine upstream tag | `python -m pytest tests -q` in a separate DeepMind Concordia v2.4.0 checkout | 34 passed | Verified upstream, not a local-project test |

## Local Automated Checks

| Area | Exact command | Observed result | Status |
| --- | --- | --- | --- |
| Concordia API contracts | `python -m pytest tests/test_concordia_compat.py -q` | 4 passed | Verified |
| Negotiation agents, components, and game master | `python -m pytest tests/negotiation -q` | 20 passed | Verified |
| Interpretability, runner, probes, analysis, config, and compatibility | `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/matplotlib python -m pytest tests/interpretability tests/interpretability_runner tests/test_imports.py tests/test_probes.py tests/test_unified_analysis.py tests/test_config.py tests/test_concordia_compat.py -q` | 85 passed | Verified |
| Complete focused audit suite | `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/matplotlib python -m pytest tests/negotiation tests/interpretability tests/interpretability_runner tests/test_imports.py tests/test_probes.py tests/test_unified_analysis.py tests/test_config.py tests/test_concordia_compat.py -q` | 105 passed, 0 failed | Verified |
| Complete offline repository suite | `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/matplotlib python -m pytest tests/ -q` | 105 passed, 1 optional ESR-Bench skip | Verified |
| Patch whitespace | `git diff --check` | Exit code 0; no whitespace errors | Verified after audit edits |
| Source compilation | `python -m compileall -q negotiation interpretability config tests/negotiation tests/interpretability tests/interpretability_runner` | Exit code 0 | Verified after final code edits |
| Static error scan | `env PYLINTHOME=/tmp/pylint python -m pylint negotiation interpretability tests/negotiation tests/interpretability tests/interpretability_runner --errors-only --score=no` | Exit code 0 | Verified after final code edits |

The focused tests exercise every configured negotiation agent module and game-master module, state serialization, label/QC logic, grouped probe splits, activation alignment, and causal-data contracts. Mocked language models validate control flow and data contracts; they do not establish model-level scientific validity.

## Environment-Dependent Checks

`tests/test_esr_bench.py` describes an in-progress optional benchmark package and separate gold data that are not yet present. The module now skips cleanly when `esr_bench` is unavailable, so it no longer prevents collection of the core repository suite. Its implementation remains part of the benchmark-packaging plan.

Model-backed end-to-end experiments were not run. They require one or more of: remote-provider credentials, a running Ollama service, downloadable Hugging Face/TransformerLens weights, sufficient GPU or accelerator memory, and substantial runtime. This includes full activation collection, DeepEval-backed judgments, SAE extraction, behavioral steering/patching, and paper-scale causal studies. Those runs must record model revision, device/dtype, seeds, generation parameters, dataset hash, and output schema version before results are used as research evidence.

## Remaining Release Gate

A small credentialed model smoke test must confirm that text generation and residual-stream capture originate from the same acting-model response before release or research-data regeneration. ESR-Bench needs its package and reviewed gold data before that optional benchmark can be enabled.
