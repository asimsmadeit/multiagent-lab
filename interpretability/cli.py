#!/usr/bin/env python3
"""
Click-based CLI for Deception Detection Experiments.

This module provides a modern CLI interface using Click for running
deception detection experiments with Concordia agents.

Usage:
    # Run emergent experiment (default)
    deception run --mode emergent --trials 5

    # Train probes on existing data
    deception train --data activations.json

    # Run with specific scenario
    deception run --scenario-name ultimatum_bluff

    # Show help
    deception --help
    deception run --help
"""

import json
import logging
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, TYPE_CHECKING

import click

from interpretability.tracks import (
    ExperimentTrack,
    get_track_spec,
)
from interpretability.launch import (
    PUBLIC_RESUME_LIMITATION,
    resolve_config_execution_surface,
    resolve_execution_surface,
)
from interpretability.scenarios.compiled import (
    ExecutionProtocol,
    SUPPORTED_SURFACE_VARIANTS,
)

# Lazy imports - only load heavy dependencies when actually running commands
# This allows --help to work without loading PyTorch/TransformerLens
if TYPE_CHECKING:
    import torch
    import numpy as np
    from interpretability import InterpretabilityRunner

_IMPORTS_LOADED = False


def _configure_verbosity(verbose: bool) -> None:
    """Apply the public verbosity flag to package diagnostics."""
    logging.getLogger("interpretability").setLevel(
        logging.DEBUG if verbose else logging.WARNING
    )


def _lazy_import():
    """Import heavy dependencies lazily."""
    global _IMPORTS_LOADED
    global torch, np
    global InterpretabilityRunner, EMERGENT_SCENARIOS, IncentiveCondition
    global get_emergent_scenarios, generate_scenario_params, compute_emergent_ground_truth
    global INSTRUCTED_SCENARIOS, Condition, ExperimentMode, get_instructed_scenarios
    global run_full_analysis, train_ridge_probe, compute_generalization_auc
    global run_all_sanity_checks, print_limitations
    global run_full_causal_validation, activation_patching_test, ablation_test
    global filter_causal_samples

    if _IMPORTS_LOADED:
        return

    import torch as _torch
    import numpy as _np
    torch = _torch
    np = _np

    from interpretability import (
        InterpretabilityRunner as _InterpretabilityRunner,
        EMERGENT_SCENARIOS as _EMERGENT_SCENARIOS,
        IncentiveCondition as _IncentiveCondition,
        get_emergent_scenarios as _get_emergent_scenarios,
        generate_scenario_params as _generate_scenario_params,
        compute_emergent_ground_truth as _compute_emergent_ground_truth,
        INSTRUCTED_SCENARIOS as _INSTRUCTED_SCENARIOS,
        Condition as _Condition,
        ExperimentMode as _ExperimentMode,
        get_instructed_scenarios as _get_instructed_scenarios,
        run_full_analysis as _run_full_analysis,
        train_ridge_probe as _train_ridge_probe,
        compute_generalization_auc as _compute_generalization_auc,
        run_all_sanity_checks as _run_all_sanity_checks,
        print_limitations as _print_limitations,
        run_full_causal_validation as _run_full_causal_validation,
        activation_patching_test as _activation_patching_test,
        ablation_test as _ablation_test,
    )
    from interpretability.causal.causal_validation import (
        filter_causal_samples as _filter_causal_samples,
    )

    InterpretabilityRunner = _InterpretabilityRunner
    EMERGENT_SCENARIOS = _EMERGENT_SCENARIOS
    IncentiveCondition = _IncentiveCondition
    get_emergent_scenarios = _get_emergent_scenarios
    generate_scenario_params = _generate_scenario_params
    compute_emergent_ground_truth = _compute_emergent_ground_truth
    INSTRUCTED_SCENARIOS = _INSTRUCTED_SCENARIOS
    Condition = _Condition
    ExperimentMode = _ExperimentMode
    get_instructed_scenarios = _get_instructed_scenarios
    run_full_analysis = _run_full_analysis
    train_ridge_probe = _train_ridge_probe
    compute_generalization_auc = _compute_generalization_auc
    run_all_sanity_checks = _run_all_sanity_checks
    print_limitations = _print_limitations
    run_full_causal_validation = _run_full_causal_validation
    activation_patching_test = _activation_patching_test
    ablation_test = _ablation_test
    filter_causal_samples = _filter_causal_samples
    _IMPORTS_LOADED = True


# Shared options as decorators
def common_options(f):
    """Common options shared across commands."""
    f = click.option('--output', '-o', default='./experiment_output',
                     help='Output directory')(f)
    f = click.option('--verbose', '-v', is_flag=True,
                     help='Enable verbose output')(f)
    return f


def model_options(f):
    """Model configuration options."""
    f = click.option('--model', '-m', default='google/gemma-2-27b-it',
                     help='HuggingFace model name (default: gemma-2-27b-it)')(f)
    f = click.option('--device', '-d', default=None,
                     help='Device (cuda/cpu/mps, auto-detected if not specified)')(f)
    f = click.option('--dtype', type=click.Choice(['float32', 'float16', 'bfloat16']),
                     default='bfloat16', help='Model dtype')(f)
    return f


@click.group()
@click.version_option(version='1.0.0', prog_name='deception')
def cli():
    """Deception Detection Experiment CLI.

    Run mechanistic interpretability experiments to detect deception
    in LLM negotiation agents.

    Examples:

        # Quick test with 5 trials
        deception run --trials 5

        # Full experiment with specific model
        deception run --model google/gemma-2-27b-it --trials 40

        # Train probes on existing data
        deception train --data activations.json

        # Run causal validation
        deception run --causal --causal-samples 20
    """
    pass


@cli.command()
@model_options
@common_options
@click.option(
    '--config',
    'config_path',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help='Authoritative ExperimentConfig JSON; cannot be mixed with run overrides',
)
@click.option('--mode', type=click.Choice(['emergent', 'instructed', 'both']),
              default='emergent', show_default=True,
              help='Emergent headline run, or explicit legacy instructed compatibility')
@click.option('--scenarios', type=int, default=4,
              help='Number of scenarios to run (default: 4 for balanced deception rates)')
@click.option('--scenario-name', multiple=True,
              type=click.Choice(['ultimatum_bluff', 'capability_bluff', 'hidden_value',
                               'info_withholding', 'promise_break', 'alliance_betrayal']),
              help='Run specific scenario(s). Default: alliance_betrayal, promise_break, capability_bluff, info_withholding')
@click.option('--trials', '-t', type=int, default=40,
              help=('Semantic families per scenario per condition; '
                    'counterbalancing expands every family'))
@click.option('--max-rounds', type=int, default=5,
              help='Max negotiation rounds per trial (default: 5 for temporal analysis)')
@click.option('--max-tokens', type=int, default=128,
              help='Max tokens per LLM response')
@click.option('--layers', help='Comma-separated list of layers to capture')
@click.option('--fast', is_flag=True,
              help='Fast mode: disable ToM module (~3x speedup)')
@click.option('--ultrafast', is_flag=True,
              help='Ultrafast mode: minimal agents (~5x additional speedup)')
@click.option('--hybrid', is_flag=True,
              help='Hybrid mode: HuggingFace + TransformerLens (~20x speedup)')
@click.option('--sae', is_flag=True,
              help='Enable Gemma Scope SAE feature extraction (requires --hybrid)')
@click.option('--sae-layer', type=int, default=31,
              help='Layer for SAE feature extraction (default: 31 for 27B model)')
@click.option('--evaluator', type=click.Choice(['local']),
              default='local', help='Model for ground truth extraction (local uses Gemma-2B)')
@click.option(
    '--evaluator-type',
    type=click.Choice(['deepeval', 'rule']),
    default='deepeval',
    show_default=True,
    help='Ground-truth detector; independent of the local structured extractor',
)
@click.option(
    '--skip-api-eval',
    is_flag=True,
    help='Disable DeepEval/API ground truth and use deterministic rules',
)
@click.option(
    '--checkpoint-dir',
    help=(
        'Write audit/manual-salvage snapshots; these artifacts cannot resume '
        'the public experiment schedule'
    ),
)
@click.option('--causal', is_flag=True,
              help='Run causal validation after probe training')
@click.option('--causal-samples', type=int, default=20,
              help='Number of samples for causal validation')
@click.option('--high-only', is_flag=True,
              help='Run only HIGH_INCENTIVE condition (skip LOW_INCENTIVE)')
@click.option(
    '--experiment-track',
    type=click.Choice([track.value for track in ExperimentTrack]),
    default=None,
    help='Activation-access track; inferred from enabled modules when omitted',
)
@click.option(
    '--protocol',
    type=click.Choice([protocol.value for protocol in ExecutionProtocol]),
    default=ExecutionProtocol.ALTERNATING.value,
    show_default=True,
    help='Turn scheduler; protocol identity is persisted with every trial',
)
@click.option(
    '--counterpart-type',
    'counterpart_types',
    multiple=True,
    type=click.Choice(['default', 'skeptical', 'credulous', 'informed', 'absent']),
    help='Counterpart policy to include; repeat for a crossed policy design',
)
@click.option(
    '--counterbalance/--no-counterbalance',
    default=True,
    show_default=True,
    help='Balance physical roles, order, policy, and prompt surface',
)
@click.option(
    '--counterbalance-seed',
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help='Stable presentation-order seed for the counterbalance schedule',
)
@click.option(
    '--surface-variant',
    'surface_variants',
    multiple=True,
    type=click.Choice(list(SUPPORTED_SURFACE_VARIANTS)),
    help='Prompt surface to include; repeat to select multiple variants',
)
@click.option(
    '--probes',
    'probe_mode',
    type=click.Choice(['auto', 'on', 'off']),
    default='auto',
    show_default=True,
    help='Typed verification/plausibility probes (auto enables alternating only)',
)
def run(mode, model, device, dtype, scenarios, scenario_name, trials,
        max_rounds, max_tokens, layers, fast, ultrafast, hybrid, sae,
        sae_layer, evaluator, evaluator_type, skip_api_eval, checkpoint_dir,
        causal, causal_samples,
        high_only, experiment_track, protocol, counterpart_types,
        counterbalance, counterbalance_seed, surface_variants, probe_mode,
        config_path, output, verbose):
    """Run a deception detection experiment.

    This command runs the complete pipeline:
    1. Run negotiation scenarios through Concordia agents
    2. Capture activations via TransformerLens
    3. Get ground truth labels
    4. Train linear probes
    5. Validate with sanity checks
    6. (Optional) Run causal validation

    Examples:

        # Quick test
        deception run --trials 5

        # Single scenario for parallel pods
        deception run --scenario-name ultimatum_bluff

        # With GPU and SAE
        deception run --device cuda --hybrid --sae --sae-layer 21
    """
    launch_config = None
    execution_plan = None
    publish_activations = True
    family_seed_start = 0
    evaluator_model_name = 'google/gemma-2b-it'
    evaluator_max_tokens = 64
    if config_path is not None:
        context = click.get_current_context()
        controlled = {
            'mode', 'model', 'device', 'dtype', 'scenarios', 'scenario_name',
            'trials', 'max_rounds', 'max_tokens', 'layers', 'fast',
            'ultrafast', 'hybrid',
            'sae', 'sae_layer', 'evaluator', 'evaluator_type',
            'skip_api_eval', 'checkpoint_dir', 'causal',
            'causal_samples', 'high_only', 'experiment_track', 'protocol',
            'counterpart_types', 'counterbalance', 'counterbalance_seed',
            'surface_variants', 'probe_mode', 'output', 'verbose',
        }
        conflicts = sorted(
            name.replace('_', '-')
            for name in controlled
            if context.get_parameter_source(name)
            is click.core.ParameterSource.COMMANDLINE
        )
        if conflicts:
            raise click.UsageError(
                '--config is authoritative and cannot be mixed with: '
                + ', '.join(f'--{name}' for name in conflicts)
            )
        try:
            from config.experiment import ExperimentConfig
            launch_config = ExperimentConfig.load_json(str(config_path))
            execution_plan = resolve_config_execution_surface(launch_config)
        except (OSError, TypeError, ValueError) as error:
            raise click.UsageError(f'invalid --config: {error}') from error

        mode = launch_config.scenarios.mode
        model = launch_config.model.name
        device = launch_config.model.device
        dtype = launch_config.model.dtype
        scenario_name = tuple(launch_config.scenarios.scenarios)
        trials = launch_config.scenarios.num_trials
        max_rounds = launch_config.scenarios.max_rounds
        layers = (
            None
            if execution_plan.experiment_track is ExperimentTrack.TEXT_ONLY
            else ','.join(map(str, launch_config.probes.layers_to_probe))
        )
        fast = False
        ultrafast = False
        hybrid = False
        sae = launch_config.model.use_sae
        sae_layer = launch_config.model.sae_layer or sae_layer
        evaluator = 'local' if launch_config.evaluator.enabled else None
        evaluator_model_name = launch_config.evaluator.model
        evaluator_max_tokens = launch_config.evaluator.max_tokens
        evaluator_type = launch_config.evaluator.ground_truth_method
        skip_api_eval = evaluator_type == 'rule'
        checkpoint_dir = launch_config.checkpoint_dir
        causal = launch_config.causal.enabled
        causal_samples = launch_config.causal.num_samples
        high_only = False
        output = launch_config.output_dir
        verbose = launch_config.verbose
        family_seed_start = launch_config.random_seed
        publish_activations = launch_config.save_activations

    if skip_api_eval:
        evaluator_type = 'rule'
    _configure_verbosity(verbose)

    if mode in {'instructed', 'both'}:
        click.echo(
            click.style(
                'LEGACY/NON-HEADLINE: instructed rows are unclassified and '
                'excluded from negotiation probe training. Mode both trains '
                'on emergent rows only; it is not a cross-mode headline result.',
                fg='yellow',
            )
        )
    if causal and mode == 'instructed':
        raise click.UsageError(
            '--causal cannot run on instructed-only legacy/non-headline rows'
        )

    if execution_plan is None:
        try:
            execution_plan = resolve_execution_surface(
                experiment_track=experiment_track,
                protocol=protocol,
                counterpart_types=counterpart_types,
                counterbalance=counterbalance,
                counterbalance_seed=counterbalance_seed,
                surface_variants=surface_variants,
                fast=fast,
                ultrafast=ultrafast,
                mode=mode,
                probe_mode=probe_mode,
            )
        except (TypeError, ValueError) as error:
            raise click.UsageError(str(error)) from error
    if execution_plan.experiment_track is ExperimentTrack.TEXT_ONLY:
        incompatible = []
        if causal:
            incompatible.append('--causal')
        if sae:
            incompatible.append('--sae')
        if hybrid:
            incompatible.append('--hybrid')
        if layers:
            incompatible.append('--layers')
        if incompatible:
            raise click.UsageError(
                'text_only cannot use white-box options: '
                + ', '.join(incompatible)
            )
    if sae and not hybrid:
        raise click.UsageError('--sae requires --hybrid')

    # Load heavy dependencies
    _lazy_import()

    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle checkpoint directory
    checkpoint_path = None
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        click.echo(f"Recovery snapshots will be written to: {checkpoint_path}")
        click.echo(PUBLIC_RESUME_LIMITATION)

    # Get scenarios
    all_emergent = get_emergent_scenarios()
    all_instructed = get_instructed_scenarios()

    if scenario_name:
        emergent_scenarios = list(scenario_name)
        instructed_scenarios = list(scenario_name)
    else:
        default_scenarios = ["ultimatum_bluff", "hidden_value", "alliance_betrayal"]
        n_scenarios = min(scenarios, 6)
        if n_scenarios <= 3:
            emergent_scenarios = default_scenarios[:n_scenarios]
            instructed_scenarios = default_scenarios[:n_scenarios]
        else:
            emergent_scenarios = all_emergent[:n_scenarios]
            instructed_scenarios = all_instructed[:n_scenarios]

    # Parse layers
    layers_list = None
    if layers:
        layers_list = [int(l.strip()) for l in layers.split(",")]

    # Parse dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype]

    # Print configuration
    click.echo(click.style("\n" + "=" * 60, fg='blue'))
    click.echo(click.style("DECEPTION DETECTION EXPERIMENT", fg='blue', bold=True))
    click.echo(click.style("=" * 60, fg='blue'))
    click.echo(f"Mode: {mode}")
    click.echo(f"Model: {model}")
    click.echo(f"Device: {device}")
    click.echo(f"Dtype: {dtype}")
    click.echo(f"Scenarios: {emergent_scenarios}")
    click.echo(f"Semantic families per condition: {trials}")
    click.echo(f"Max rounds: {max_rounds}")
    click.echo(f"Fast mode: {fast}")
    click.echo(f"Ultrafast mode: {ultrafast}")
    click.echo(f"Hybrid mode: {hybrid}")
    click.echo(f"SAE enabled: {sae}")
    if sae:
        click.echo(f"SAE layer: {sae_layer}")
    click.echo(f"Local structured extractor: {evaluator}")
    click.echo(f"Ground-truth detector requested: {evaluator_type}")
    click.echo(f"Output directory: {output_dir}")

    agent_modules = list(execution_plan.agent_modules)
    selected_track = execution_plan.experiment_track
    participant_ids = execution_plan.participant_ids
    captured_actor_ids = execution_plan.captured_actor_ids
    emergent_condition_count = 1 if high_only else 2
    estimated_emergent_executions = (
        len(emergent_scenarios)
        * trials
        * emergent_condition_count
        * execution_plan.executions_per_family
        if mode in {'emergent', 'both'}
        else 0
    )
    estimated_instructed_executions = (
        len(instructed_scenarios) * trials * 2
        if mode in {'instructed', 'both'}
        else 0
    )
    track_manifest = {
        'experiment_track': selected_track.value,
        'activation_access': get_track_spec(selected_track).activation_access,
        'participant_ids': list(participant_ids),
        'captured_actor_ids': list(captured_actor_ids),
        'headline_capture_policy': execution_plan.per_trial_capture_policy,
        'headline_captured_actor_count_per_trial': (
            execution_plan.per_trial_captured_actor_count
        ),
        'enabled_modules': list(agent_modules),
        'allows_online_adaptation': (
            selected_track is ExperimentTrack.ADAPTIVE
        ),
        'protocol': execution_plan.protocol.value,
        'counterpart_policies': list(execution_plan.counterpart_types),
        'counterbalance': execution_plan.counterbalance,
        'counterbalance_seed': execution_plan.counterbalance_seed,
        'surface_variants': list(execution_plan.surface_variants),
        'execution_design_scope': execution_plan.execution_design_scope,
        'config_source': str(config_path) if config_path is not None else None,
        'experiment_name': (
            launch_config.experiment_name if launch_config is not None else None
        ),
        'activation_publication': {
            'enabled': publish_activations,
            'scientific_status': (
                'activation_dataset_requested'
                if publish_activations
                else 'recovery_evidence_only_no_dataset'
            ),
        },
        'recovery_contract': {
            'writes_snapshots': checkpoint_path is not None,
            'schedule_resume_supported': False,
            'purpose': 'audit_and_manual_salvage_only',
            'limitation': PUBLIC_RESUME_LIMITATION,
        },
        'logging_contract': {
            'verbose': verbose,
            'log_to_file': False,
            'destination': 'stderr_and_cli_stdout',
        },
        'seed_design': {
            'family_seed_start': family_seed_start,
            'use_multi_seed': False,
            'configured_random_seeds': (
                list(launch_config.random_seeds)
                if launch_config is not None else None
            ),
            'configured_random_seeds_status': (
                'inactive_reserved'
                if launch_config is not None else 'not_configured'
            ),
            'source': 'experiment_config' if launch_config is not None else 'cli_default',
        },
        'evaluator_design': {
            'local_structured_extractor': {
                'enabled': evaluator is not None,
                'model': evaluator_model_name,
                'max_tokens': evaluator_max_tokens,
            },
            'ground_truth_detector': {
                'requested': evaluator_type,
                'skip_api_eval': skip_api_eval,
                'effective': None,
                'deepeval_available': None,
            },
        },
        'legacy_instructed_execution_contract': (
            {
                'protocol': 'legacy_run_study_alternating',
                'counterpart_policy_cross': 'unsupported',
                'surface_cross': 'unsupported',
            }
            if mode in {'instructed', 'both'}
            else None
        ),
        'semantic_families_per_scenario_condition': trials,
        'executions_per_family': execution_plan.executions_per_family,
        'estimated_emergent_executions': estimated_emergent_executions,
        'estimated_instructed_legacy_executions': (
            estimated_instructed_executions
        ),
        'estimated_total_physical_executions': (
            estimated_emergent_executions + estimated_instructed_executions
        ),
        'probes': {
            'mode': execution_plan.probe_mode,
            'enabled': (
                execution_plan.protocol is ExecutionProtocol.ALTERNATING
                if execution_plan.run_probes is None
                else execution_plan.run_probes
            ),
        },
        'experiment_mode': mode,
        'headline_probe_rows': (
            'emergent_only'
            if mode == 'both'
            else ('emergent' if mode == 'emergent' else 'none')
        ),
        'instructed_scientific_status': (
            'legacy_non_headline_excluded_from_probes'
            if mode in {'instructed', 'both'}
            else 'not_requested'
        ),
    }
    track_manifest_path = output_dir / 'experiment_track_manifest.json'
    track_manifest_path.write_text(
        json.dumps(track_manifest, sort_keys=True, indent=2) + '\n',
        encoding='utf-8',
    )
    click.echo(f"Experiment track: {selected_track.value}")
    click.echo(f"Execution protocol: {execution_plan.protocol.value}")
    click.echo(
        "Execution budget: "
        f"{trials} semantic families/scenario/condition × "
        f"{execution_plan.executions_per_family} assignments/family; "
        f"estimated total={estimated_emergent_executions + estimated_instructed_executions}"
    )
    click.echo(f"Track manifest: {track_manifest_path}")

    # Initialize runner
    click.echo("\nInitializing InterpretabilityRunner...")
    start_time = time.time()

    runner = InterpretabilityRunner(
        model_name=model,
        device=device,
        torch_dtype=torch_dtype,
        layers_to_capture=layers_list,
        max_tokens=max_tokens,
        use_hybrid=hybrid,
        use_sae=sae,
        sae_layer=sae_layer,
        evaluator_api=evaluator,
        evaluator_model_name=evaluator_model_name,
        evaluator_max_tokens=evaluator_max_tokens,
        evaluator_type=evaluator_type,
        trial_id_offset=family_seed_start,
        experiment_track=selected_track,
        captured_actor_ids=captured_actor_ids,
    )

    detector_manifest = track_manifest['evaluator_design'][
        'ground_truth_detector'
    ]
    detector_manifest['effective'] = getattr(
        runner, 'evaluator_type', evaluator_type
    )
    detector_manifest['deepeval_available'] = bool(
        getattr(runner, '_deepeval_detector', None)
    )
    track_manifest_path.write_text(
        json.dumps(track_manifest, sort_keys=True, indent=2) + '\n',
        encoding='utf-8',
    )

    init_time = time.time() - start_time
    click.echo(f"Initialization complete in {init_time:.1f}s")

    # Run experiments
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    activations_path = None
    text_evidence_path = None
    unpublished_activation_recovery_path = None
    instructed_recovery_path = None

    if mode in ["emergent", "both"]:
        results = _run_emergent_experiment(
            runner=runner,
            scenarios=emergent_scenarios,
            trials_per_scenario=trials,
            max_rounds=max_rounds,
            agent_modules=agent_modules,
            ultrafast=ultrafast,
            checkpoint_dir=str(checkpoint_path) if checkpoint_path else None,
            high_only=high_only,
            counterpart_types=execution_plan.counterpart_types,
            protocol=execution_plan.protocol,
            counterbalance=execution_plan.counterbalance,
            counterbalance_seed=execution_plan.counterbalance_seed,
            surface_variants=execution_plan.surface_variants,
            run_probes=execution_plan.run_probes,
            executions_per_family=execution_plan.executions_per_family,
        )
        all_results["emergent"] = results

        if selected_track is ExperimentTrack.TEXT_ONLY:
            text_evidence_path = runner.write_activation_recovery(
                output_dir / f"text_only_evidence_{timestamp}.json",
                reason=(
                    "text-only runs intentionally contain no activation rows; "
                    "canonical generations and interaction events are retained"
                ),
                experiment_progress={
                    "experiment_mode": mode,
                    "scientific_status": "text_only_evidence",
                    "protocol": execution_plan.protocol.value,
                    "counterpart_policies": list(
                        execution_plan.counterpart_types
                    ),
                    "counterbalance": execution_plan.counterbalance,
                    "counterbalance_seed": execution_plan.counterbalance_seed,
                    "surface_variants": list(
                        execution_plan.surface_variants
                    ),
                    "semantic_families_per_scenario_condition": trials,
                    "executions_per_family": (
                        execution_plan.executions_per_family
                    ),
                    "estimated_total_physical_executions": (
                        estimated_emergent_executions
                    ),
                    "probes": {
                        "mode": execution_plan.probe_mode,
                        "enabled": (
                            execution_plan.protocol
                            is ExecutionProtocol.ALTERNATING
                            if execution_plan.run_probes is None
                            else execution_plan.run_probes
                        ),
                    },
                },
            )
            click.echo(f"Text-only evidence artifact: {text_evidence_path}")

        # In mixed compatibility runs, freeze the publishable emergent dataset
        # before unclassified instructed rows are appended to the runner.
        if mode == "both" and selected_track is not ExperimentTrack.TEXT_ONLY:
            if publish_activations:
                activations_path = runner.save_dataset(
                    str(
                        output_dir
                        / f"activations_emergent_headline_{timestamp}.json"
                    ),
                    experiment_track=selected_track.value,
                    captured_actor_ids=captured_actor_ids,
                )
            else:
                unpublished_activation_recovery_path = (
                    runner.write_activation_recovery(
                        output_dir
                        / f"emergent_recovery_only_{timestamp}.json",
                        reason=(
                            "ExperimentConfig.save_activations=False; no final "
                            "activation dataset was published"
                        ),
                        experiment_progress={
                            "experiment_mode": mode,
                            "scientific_status": "recovery_evidence_only_no_dataset",
                        },
                    )
                )

    if mode in ["instructed", "both"]:
        instructed_offsets = {
            "sample_start": len(runner.activation_samples),
            "generation_start": len(runner.generation_records),
            "label_start": len(runner.label_records),
            "event_start": len(runner.interaction_events),
        }
        results = _run_instructed_experiment(
            runner=runner,
            scenarios=instructed_scenarios,
            trials_per_scenario=trials,
        )
        all_results["instructed"] = results
        instructed_recovery_path = runner.write_activation_recovery(
            output_dir / f"instructed_legacy_recovery_{timestamp}.json",
            **instructed_offsets,
            reason=(
                "legacy instructed rows are unclassified, non-headline, and "
                "ineligible for activation_dataset/4.1.0 publication"
            ),
            experiment_progress={
                "experiment_mode": mode,
                "scientific_status": "legacy_non_headline",
            },
        )
        click.echo(
            f"Instructed recovery artifact (non-publishable): "
            f"{instructed_recovery_path}"
        )

    if (
        mode == "emergent"
        and selected_track is not ExperimentTrack.TEXT_ONLY
    ):
        if publish_activations:
            activations_path = runner.save_dataset(
                str(output_dir / f"activations_emergent_{timestamp}.json"),
                experiment_track=selected_track.value,
                captured_actor_ids=captured_actor_ids,
            )
        else:
            unpublished_activation_recovery_path = runner.write_activation_recovery(
                output_dir / f"emergent_recovery_only_{timestamp}.json",
                reason=(
                    "ExperimentConfig.save_activations=False; no final activation "
                    "dataset was published"
                ),
                experiment_progress={
                    "experiment_mode": mode,
                    "scientific_status": "recovery_evidence_only_no_dataset",
                },
            )
    if activations_path is not None:
        click.echo(f"\nHeadline activations saved to: {activations_path}")

    results_path = _write_experiment_results(
        output_dir / f"experiment_results_{timestamp}.json",
        all_results,
    )
    click.echo(f"Experiment results saved to: {results_path}")

    # Train probes
    click.echo(click.style("\n" + "=" * 60, fg='green'))
    click.echo(click.style("POST-EXPERIMENT ANALYSIS", fg='green', bold=True))
    click.echo(click.style("=" * 60, fg='green'))

    # The activation artifact was written by this command in this process.
    if selected_track is ExperimentTrack.TEXT_ONLY:
        click.echo(
            'Probe training unavailable: text-only runs contain no activations.'
        )
        probe_results = {
            'best_probe': None,
            'scientific_status': 'unavailable_text_only',
        }
    elif mode == 'instructed':
        click.echo(
            'Headline probe training unavailable: instructed-only rows are '
            'legacy/non-headline and fail the negotiation eligibility contract.'
        )
        probe_results = {
            'best_probe': None,
            'scientific_status': 'unavailable_instructed_only',
        }
    elif not publish_activations:
        click.echo(
            'Probe training unavailable: ExperimentConfig.save_activations=False '
            'disabled activation-dataset publication.'
        )
        probe_results = {
            'best_probe': None,
            'scientific_status': 'unavailable_activation_publication_disabled',
        }
    else:
        if mode == 'both':
            click.echo(
                'Probe eligibility: training on emergent negotiation rows only; '
                'instructed compatibility rows are excluded.'
            )
        probe_results = _train_probes_on_data(
            # Both-mode analysis intentionally receives the frozen emergent
            # artifact rather than the runner's mixed in-memory collection.
            str(activations_path),
            str(output_dir),
            trusted_legacy=False,
        )

    # Causal validation
    causal_validated = False
    causal_results = None

    if causal and probe_results.get("best_probe"):
        causal_results = _run_causal_validation(
            runner, activations_path, probe_results, causal_samples, output_dir
        )
        causal_validated = causal_results.get("causal_claim_ready", False) if causal_results else False

    # Print summary
    summary_artifact = (
        activations_path
        or text_evidence_path
        or unpublished_activation_recovery_path
        or instructed_recovery_path
    )
    _print_summary(runner, probe_results, causal_results, causal_validated,
                   summary_artifact, output_dir, model, start_time)


@cli.command()
@common_options
@click.option('--data', '-d', required=True, type=click.Path(exists=True),
              help='Path to a safe activation .json manifest or legacy .pt file')
@click.option(
    '--trust-legacy-pt',
    is_flag=True,
    help='Allow pickle-capable .pt loading only for a reviewed artifact',
)
def train(data, trust_legacy_pt, output, verbose):
    """Train probes on existing activation data.

    This command trains linear probes on previously captured
    activation data without running new experiments.

    Example:

        deception train --data activations.json -o ./results
    """
    _configure_verbosity(verbose)

    # Load heavy dependencies
    _lazy_import()

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(click.style("\n" + "=" * 60, fg='green'))
    click.echo(click.style("PROBE TRAINING", fg='green', bold=True))
    click.echo(click.style("=" * 60, fg='green'))
    click.echo(f"Loading data from: {data}")

    results = _train_probes_on_data(
        data,
        str(output_dir),
        trusted_legacy=trust_legacy_pt,
    )

    best_probe = results.get("best_probe")
    if isinstance(best_probe, dict):
        click.echo(click.style("\nBest probe:", fg='green', bold=True))
        click.echo(f"  Layer: {best_probe['layer']}")
        click.echo(f"  R²: {best_probe['r2']:.3f}")


@cli.command()
def scenarios():
    """List available deception scenarios.

    Shows all available scenarios for emergent and instructed
    deception experiments.
    """
    # Load heavy dependencies
    _lazy_import()

    click.echo(click.style("\nEmergent Scenarios:", fg='blue', bold=True))
    click.echo("-" * 40)
    for name in get_emergent_scenarios():
        click.echo(f"  - {name}")

    click.echo(click.style("\nInstructed Scenarios:", fg='blue', bold=True))
    click.echo("-" * 40)
    for name in get_instructed_scenarios():
        click.echo(f"  - {name}")


@cli.group()
def events():
    """Inspect, replay, and trace event-sourced experiment runs.

    These commands operate purely on recorded event streams (Plan 2) and
    do not load models, so they run without GPU or provider credentials.
    """


@events.command('validate')
@click.argument('stream', type=click.Path(exists=True, dir_okay=False))
def events_validate(stream):
    """Validate one event stream and report its trials.

    Exits non-zero when the stream fails integrity validation.
    """
    from interpretability.events.replay import inspect_stream

    inspection = inspect_stream(stream)
    if not inspection.valid:
        click.echo(click.style(
            f"INVALID: {inspection.error_type}: {inspection.error_reason}",
            fg='red', bold=True,
        ))
        raise SystemExit(1)
    click.echo(click.style(f"VALID: {inspection.source_name}", fg='green'))
    for trial in inspection.trials:
        classification = getattr(
            trial.classification, 'value', trial.classification
        )
        click.echo(
            f"  trial {trial.trial_id}: {classification} "
            f"(terminal={trial.terminal_event_id or '-'})"
        )
    click.echo(f"{len(inspection.trials)} trial(s)")


@events.command('replay')
@click.argument('stream', type=click.Path(exists=True, dir_okay=False))
@click.option('--projection', 'projection_name', default='transcript',
              help='Projection to replay (default: transcript).')
@click.option('--trial-id', required=True, help='Trial ID to project.')
def events_replay(stream, projection_name, trial_id):
    """Replay one deterministic projection for one trial."""
    from interpretability.events.replay import (
        ProjectionRequest,
        replay_projection,
    )

    request = ProjectionRequest(projection_name, trial_id)
    result = replay_projection(stream, request)
    manifest = result.manifest
    click.echo(click.style(
        f"{manifest.projection_name} @ trial {manifest.trial_id}",
        fg='blue', bold=True,
    ))
    click.echo(f"  semantic hash: {manifest.projection_semantic_hash}")


@events.command('trace')
@click.argument('stream', type=click.Path(exists=True, dir_okay=False))
@click.option('--event-id', required=True, help='Event UUID to trace.')
def events_trace(stream, event_id):
    """Trace one event's lineage: ancestors, descendants, artifacts."""
    from interpretability.events.provenance import trace_event
    from interpretability.events.reader import EventReader

    envelopes = [located.event for located in EventReader(stream).iter_events()]
    lineage = trace_event(envelopes, event_id)
    click.echo(click.style(f"lineage of {event_id}", fg='blue', bold=True))
    click.echo(f"  ancestors:   {len(lineage.ancestor_event_ids)}")
    click.echo(f"  descendants: {len(lineage.descendant_event_ids)}")
    click.echo(f"  same call:   {len(lineage.same_call_event_ids)}")
    click.echo(f"  artifacts:   {len(lineage.artifact_hashes)}")
    if lineage.missing_parent_ids:
        click.echo(click.style(
            f"  missing parents: {len(lineage.missing_parent_ids)}",
            fg='yellow',
        ))


# Helper functions (private)

def _run_emergent_experiment(
    runner,
    scenarios,
    trials_per_scenario,
    max_rounds,
    agent_modules,
    ultrafast,
    checkpoint_dir,
    high_only=False,
    counterpart_types=(),
    protocol=ExecutionProtocol.ALTERNATING,
    counterbalance=True,
    counterbalance_seed=0,
    surface_variants=(),
    run_probes=None,
    executions_per_family=1,
):
    """Run emergent deception experiment."""
    if high_only:
        conditions = [IncentiveCondition.HIGH_INCENTIVE]
    else:
        conditions = [IncentiveCondition.HIGH_INCENTIVE, IncentiveCondition.LOW_INCENTIVE]

    click.echo(click.style("\n" + "=" * 60, fg='cyan'))
    click.echo(click.style("EMERGENT DECEPTION EXPERIMENT", fg='cyan', bold=True))
    click.echo(click.style("=" * 60, fg='cyan'))
    click.echo(f"Scenarios: {scenarios}")
    click.echo(f"Conditions: {[c.value for c in conditions]}")
    click.echo(f"Semantic families per condition: {trials_per_scenario}")
    click.echo(f"Executions per family: {executions_per_family}")
    click.echo(f"Max rounds: {max_rounds}")
    click.echo(f"Agent modules: {agent_modules}")
    click.echo(f"Ultrafast mode: {ultrafast}")
    click.echo(
        "Total physical executions: "
        f"{len(scenarios) * len(conditions) * trials_per_scenario * executions_per_family}"
    )

    return runner.run_all_emergent_scenarios(
        scenarios=scenarios,
        trials_per_scenario=trials_per_scenario,
        conditions=conditions,
        max_rounds=max_rounds,
        agent_modules=agent_modules,
        ultrafast=ultrafast,
        checkpoint_dir=checkpoint_dir,
        counterpart_types=counterpart_types,
        protocol=protocol,
        counterbalance=counterbalance,
        counterbalance_seed=counterbalance_seed,
        surface_variants=surface_variants,
        run_probes=run_probes,
    )


def _write_experiment_results(path: str | Path, results: Dict[str, Any]) -> Path:
    """Persist aggregate study results as a separate, auditable JSON artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def json_default(value):
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        to_dict = getattr(value, 'to_dict', None)
        if callable(to_dict):
            return to_dict()
        item = getattr(value, 'item', None)
        if callable(item):
            try:
                return item()
            except (TypeError, ValueError):
                pass
        to_list = getattr(value, 'tolist', None)
        if callable(to_list):
            return to_list()
        if isinstance(value, (Path, set, tuple)):
            return list(value) if not isinstance(value, Path) else str(value)
        enum_value = getattr(value, 'value', None)
        if enum_value is not None:
            return enum_value
        raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

    destination.write_text(
        json.dumps(results, sort_keys=True, indent=2, default=json_default) + '\n',
        encoding='utf-8',
    )
    return destination


def _run_instructed_experiment(runner, scenarios, trials_per_scenario):
    """Run instructed deception experiment."""
    conditions = [Condition.DECEPTIVE, Condition.HONEST]

    click.echo(click.style("\n" + "=" * 60, fg='cyan'))
    click.echo(click.style("INSTRUCTED DECEPTION EXPERIMENT", fg='cyan', bold=True))
    click.echo(click.style("=" * 60, fg='cyan'))
    click.echo(f"Scenarios: {scenarios}")
    click.echo(f"Conditions: {[c.value for c in conditions]}")
    click.echo(f"Trials per condition: {trials_per_scenario}")

    all_samples = []
    for scenario in scenarios:
        for condition in conditions:
            click.echo(f"\nRunning {scenario} / {condition.value}...")
            result = runner.run_study(
                scenario=scenario,
                num_trials=trials_per_scenario,
                condition=condition.value,
                use_gm=True,
            )
            all_samples.extend(result.activation_samples)

    return {"samples": all_samples, "mode": "instructed"}


def _train_probes_on_data(
    data_path,
    output_dir,
    *,
    trusted_legacy=False,
):
    """Train probes on captured data."""
    results = run_full_analysis(
        data_path,
        trusted_legacy=trusted_legacy,
    )

    if output_dir:
        output_path = Path(output_dir) / "probe_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results saved to: {output_path}")

    return results


def _run_causal_validation(runner, activations_path, probe_results, n_samples, output_dir):
    """Run causal validation tests."""
    click.echo(click.style("\n" + "=" * 60, fg='yellow'))
    click.echo(click.style("CAUSAL VALIDATION", fg='yellow', bold=True))
    click.echo(click.style("=" * 60, fg='yellow'))

    try:
        from interpretability.data import load_activation_dataset
        data = load_activation_dataset(activations_path)
        activations = {
            k: v.float().numpy() if hasattr(v, 'numpy') else v
            for k, v in data["activations"].items()
        }
        label_data = data["labels"]
        activations, gm_labels, sample_metadata, group_ids = filter_causal_samples(
            activations,
            label_data["gm_labels"],
            data.get("metadata"),
            round_nums=label_data.get("round_nums"),
            trial_ids=label_data.get("trial_ids"),
            pod_ids=label_data.get("pod_ids"),
        )
        best_probe = probe_results.get("best_probe")
        if not isinstance(best_probe, dict):
            raise ValueError("Probe results do not contain a valid best probe")
        best_layer = best_probe["layer"]

        test_prompts = [
            "In this negotiation, I will be completely honest about my position.",
            "I need to think carefully about what to reveal in this deal.",
            "The other party is asking about my bottom line.",
            "I should consider whether to share all the information I have.",
            "This offer seems reasonable, but I'm not sure if I should accept.",
        ] * (n_samples // 5 + 1)

        # Get TransformerLens model
        tl_model = getattr(runner, 'tl_model', None)
        if tl_model is None:
            tl_model = getattr(runner.model, 'tl_model', None)

        if tl_model is None:
            raise RuntimeError(
                "requested causal validation cannot access a TransformerLens model"
            )

        sample_prompts = [
            row.get("full_prompt", "") for row in sample_metadata
        ] if sample_metadata else []
        if not all(sample_prompts):
            sample_prompts = None

        results = run_full_causal_validation(
            model=tl_model,
            activations=activations,
            labels=gm_labels,
            best_layer=best_layer,
            test_prompts=test_prompts[:n_samples],
            sample_prompts=sample_prompts,
            metadata=sample_metadata,
            group_ids=group_ids,
            verbose=True,
        )

        # Save results
        def convert_numpy(obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        results_path = output_dir / "causal_validation_results.json"
        with open(results_path, "w") as f:
            json.dump(convert_numpy(results), f, indent=2)
        click.echo(f"Causal results saved to: {results_path}")

        return results

    except Exception as e:
        click.echo(click.style(f"Causal validation failed: {e}", fg='red'))
        import traceback
        traceback.print_exc()
        raise click.ClickException(
            f"requested causal validation failed: {e}"
        ) from e


def _print_summary(runner, probe_results, causal_results, causal_validated,
                   activations_path, output_dir, model, start_time):
    """Print experiment summary."""
    click.echo(click.style("\n" + "=" * 60, fg='green'))
    click.echo(click.style("EXPERIMENT COMPLETE", fg='green', bold=True))
    click.echo(click.style("=" * 60, fg='green'))
    click.echo(f"Total samples: {len(runner.activation_samples)}")
    click.echo(f"Primary experiment artifact: {activations_path}")
    click.echo(f"Output directory: {output_dir}")

    best_probe = probe_results.get("best_probe")
    if isinstance(best_probe, dict):
        click.echo(click.style("\nBest probe performance:", bold=True))
        click.echo(f"  Layer: {best_probe['layer']}")
        click.echo(f"  R²: {best_probe['r2']:.3f}")

    if probe_results.get("gm_vs_agent"):
        gm_vs_agent = probe_results["gm_vs_agent"]
        click.echo(click.style("\nGM vs Agent comparison:", bold=True))
        if gm_vs_agent.get("available", True):
            click.echo(f"  GM R²: {gm_vs_agent['gm_ridge_r2']:.3f}")
            click.echo(f"  Agent R²: {gm_vs_agent['agent_ridge_r2']:.3f}")
            if gm_vs_agent["gm_wins"]:
                click.echo(
                    "  >> Actual-deception labels are more predictable than "
                    "counterpart-belief labels"
                )
        else:
            reason = gm_vs_agent.get(
                "reason", "no valid complete-case counterpart target"
            )
            click.echo(f"  Unavailable: {reason}")

    if causal_results:
        click.echo(click.style("\nCausal validation:", bold=True))
        click.echo(f"  Tests passed: {causal_results['n_tests_passed']}/{causal_results['n_tests_total']}")
        click.echo("  Aggregate evidence strength: unavailable (not calibrated)")
        if causal_validated:
            click.echo(click.style("  >> PREREGISTERED CAUSAL GATE PASSED", fg='green', bold=True))
        else:
            click.echo(click.style("  >> Review individual estimands and controls", fg='yellow'))

    print_limitations(
        n_samples=len(runner.activation_samples),
        model_name=model,
        causal_validated=causal_validated,
    )

    click.echo(f"\nTotal experiment time: {(time.time() - start_time):.1f}s")


# Entry point for backward compatibility
def main():
    """Entry point for backward compatibility with argparse version."""
    cli()


if __name__ == "__main__":
    cli()
