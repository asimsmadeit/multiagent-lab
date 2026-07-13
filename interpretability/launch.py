"""Shared, fail-closed normalization for public experiment entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from interpretability.scenarios.compiled import (
    CounterpartPolicy,
    ExecutionProtocol,
    SUPPORTED_COUNTERPART_POLICIES,
    SUPPORTED_SURFACE_VARIANTS,
    validate_counterpart_policy,
    validate_execution_protocol,
    validate_surface_variant,
)
from interpretability.tracks import ExperimentTrack, validate_track_assignment


PUBLIC_PARTICIPANTS = ("Negotiator", "Counterpart")
SOLO_PARTICIPANTS = ("Negotiator", "AbsentCounterpart")
PUBLIC_AGENT_MODULES = (
    "theory_of_mind",
    "cultural_adaptation",
    "temporal_strategy",
    "swarm_intelligence",
    "uncertainty_aware",
    "strategy_evolution",
)
PUBLIC_RESUME_LIMITATION = (
    "Public schedule resume is unsupported: aggregate recovery snapshots do "
    "not retain the complete result accumulator required for exact continuation."
)

_SUPPORTED_PROBE_DEFAULTS = {
    "train_ratio": 0.8,
    "regularization": 1.0,
    "max_iter": 1000,
    "token_position": "last",
    "min_accuracy": 0.6,
    "run_sanity_checks": True,
    "run_cross_scenario_validation": True,
    "binary_threshold": 0.5,
    "run_threshold_sensitivity": True,
}
_SUPPORTED_CAUSAL_DEFAULTS = {
    "num_samples": 30,
    "run_activation_patching": True,
    "run_ablation": True,
    "run_steering": True,
    "min_effect_size": 0.1,
}


@dataclass(frozen=True)
class ExecutionSurfacePlan:
    """Normalized public execution design resolved before model construction."""

    experiment_track: ExperimentTrack
    protocol: ExecutionProtocol
    agent_modules: tuple[str, ...]
    participant_ids: tuple[str, str]
    captured_actor_ids: tuple[str, ...]
    per_trial_capture_policy: str
    per_trial_captured_actor_count: int
    counterpart_types: tuple[str, ...]
    counterbalance: bool
    counterbalance_seed: int
    surface_variants: tuple[str, ...]
    executions_per_family: int
    probe_mode: str
    run_probes: bool | None
    execution_design_scope: str


def resolve_execution_surface(
    *,
    experiment_track: ExperimentTrack | str | None,
    protocol: ExecutionProtocol | str = ExecutionProtocol.ALTERNATING,
    counterpart_types: Sequence[str] | None = None,
    counterbalance: bool = True,
    counterbalance_seed: int = 0,
    surface_variants: Sequence[str] | None = None,
    fast: bool = False,
    ultrafast: bool = False,
    no_tom: bool = False,
    mode: str = "emergent",
    probe_mode: str = "auto",
    agent_modules: Sequence[str] | None = None,
) -> ExecutionSurfacePlan:
    """Resolve tracks, modules, capture access, and protocol design exactly once."""
    if type(counterbalance) is not bool:
        raise TypeError("counterbalance must be a boolean")
    if type(counterbalance_seed) is not int or counterbalance_seed < 0:
        raise ValueError("counterbalance_seed must be a non-negative integer")
    if mode not in {"emergent", "instructed", "both"}:
        raise ValueError(f"Unsupported experiment mode: {mode!r}")

    disabled_tom = bool(fast or ultrafast or no_tom)
    selected_track = ExperimentTrack(
        experiment_track
        or (
            ExperimentTrack.SINGLE_AGENT_WHITE_BOX
            if disabled_tom
            else ExperimentTrack.THEORY_OF_MIND
        )
    )
    execution_protocol = validate_execution_protocol(protocol)
    if (
        mode != "emergent"
        and execution_protocol is not ExecutionProtocol.ALTERNATING
    ):
        raise ValueError(
            "legacy instructed execution supports protocol=alternating only"
        )
    if probe_mode not in {"auto", "on", "off"}:
        raise ValueError("probe_mode must be one of: auto, on, off")
    run_probes = {"auto": None, "on": True, "off": False}[probe_mode]
    if (
        run_probes is True
        and execution_protocol is not ExecutionProtocol.ALTERNATING
    ):
        raise ValueError("typed probes currently require protocol=alternating")

    if agent_modules is not None:
        normalized_modules = tuple(agent_modules)
        if any(
            not isinstance(module, str) or not module
            for module in normalized_modules
        ):
            raise ValueError("agent modules must be non-empty strings")
        if len(set(normalized_modules)) != len(normalized_modules):
            raise ValueError("agent modules must be unique")
        unknown_modules = set(normalized_modules).difference(PUBLIC_AGENT_MODULES)
        if unknown_modules:
            raise ValueError(
                "unknown agent modules: " + ", ".join(sorted(unknown_modules))
            )
        if fast or ultrafast or no_tom:
            raise ValueError(
                "explicit agent_modules cannot be combined with fast/no-tom/ultrafast"
            )
    elif selected_track is ExperimentTrack.THEORY_OF_MIND:
        if disabled_tom:
            raise ValueError(
                "theory_of_mind track is incompatible with fast/no-tom/ultrafast"
            )
        normalized_modules = ("theory_of_mind",)
    elif selected_track is ExperimentTrack.ADAPTIVE:
        if fast or ultrafast:
            raise ValueError("adaptive track is incompatible with fast/ultrafast")
        normalized_modules = ("strategy_evolution",)
    else:
        normalized_modules = () if disabled_tom else ("theory_of_mind",)

    if mode != "emergent" and selected_track in {
        ExperimentTrack.TEXT_ONLY,
        ExperimentTrack.BILATERAL_WHITE_BOX,
        ExperimentTrack.ADAPTIVE,
    }:
        raise ValueError(
            f"{selected_track.value} currently supports emergent mode only"
        )
    if (
        execution_protocol is ExecutionProtocol.SOLO_NO_RESPONSE
        and selected_track is ExperimentTrack.BILATERAL_WHITE_BOX
    ):
        raise ValueError(
            "solo_no_response cannot use the bilateral_white_box track"
        )

    participant_ids = (
        SOLO_PARTICIPANTS
        if execution_protocol is ExecutionProtocol.SOLO_NO_RESPONSE
        else PUBLIC_PARTICIPANTS
    )
    if selected_track is ExperimentTrack.TEXT_ONLY:
        captured_actor_ids: tuple[str, ...] = ()
        representative_capture = ()
        capture_policy = "none"
        capture_count = 0
    elif selected_track is ExperimentTrack.BILATERAL_WHITE_BOX:
        captured_actor_ids = participant_ids
        representative_capture = participant_ids
        capture_policy = "all_participants_per_trial"
        capture_count = len(participant_ids)
    else:
        captured_actor_ids = (
            (participant_ids[0],)
            if (
                execution_protocol is ExecutionProtocol.SOLO_NO_RESPONSE
                or not counterbalance
            )
            else participant_ids
        )
        representative_capture = (participant_ids[0],)
        capture_policy = "one_logical_actor_per_trial"
        capture_count = 1

    validate_track_assignment(
        selected_track,
        participant_ids=participant_ids,
        captured_actor_ids=representative_capture,
        enabled_modules=normalized_modules,
    )

    raw_policies = tuple(counterpart_types or ())
    if any(not isinstance(policy, str) or not policy for policy in raw_policies):
        raise ValueError("counterpart policies must be non-empty strings")
    if len(set(raw_policies)) != len(raw_policies):
        raise ValueError("counterpart policies must be unique")
    if execution_protocol is ExecutionProtocol.SOLO_NO_RESPONSE:
        normalized_policies = raw_policies or ("absent",)
        if normalized_policies != ("absent",):
            raise ValueError(
                "solo_no_response requires exactly counterpart_type='absent'"
            )
        if counterbalance:
            raise ValueError("solo_no_response requires counterbalance=False")
    else:
        if "absent" in raw_policies:
            raise ValueError(
                "counterpart_type='absent' requires protocol='solo_no_response'"
            )
        normalized_policies = raw_policies or (
            tuple(SUPPORTED_COUNTERPART_POLICIES)
            if counterbalance
            else (CounterpartPolicy.DEFAULT.value,)
        )
        normalized_policies = tuple(
            validate_counterpart_policy(policy).value
            for policy in normalized_policies
        )
        if not counterbalance and len(normalized_policies) != 1:
            raise ValueError(
                "counterbalance=False requires exactly one counterpart policy"
            )

    selected_surfaces = tuple(surface_variants or ())
    if len(set(selected_surfaces)) != len(selected_surfaces):
        raise ValueError("surface variants must be unique")
    for variant in selected_surfaces:
        validate_surface_variant(variant)
    if execution_protocol is ExecutionProtocol.SOLO_NO_RESPONSE:
        if selected_surfaces and selected_surfaces != ("default",):
            raise ValueError("solo_no_response supports only the default surface")
        selected_surfaces = ("default",)
    elif not counterbalance:
        if selected_surfaces and selected_surfaces != ("default",):
            raise ValueError("counterbalance=False supports only the default surface")
        selected_surfaces = ("default",)
    elif not selected_surfaces:
        selected_surfaces = tuple(SUPPORTED_SURFACE_VARIANTS)

    if mode == "instructed":
        default_policies = tuple(SUPPORTED_COUNTERPART_POLICIES)
        default_surfaces = tuple(SUPPORTED_SURFACE_VARIANTS)
        if raw_policies and normalized_policies != default_policies:
            raise ValueError(
                "instructed-only legacy runs do not support custom counterpart policies"
            )
        if surface_variants and selected_surfaces != default_surfaces:
            raise ValueError(
                "instructed-only legacy runs do not support custom prompt surfaces"
            )
        if not counterbalance:
            raise ValueError(
                "instructed-only legacy runs require the default counterbalance setting"
            )
        if probe_mode != "auto":
            raise ValueError(
                "instructed-only legacy runs do not support typed probe controls"
            )
        # The legacy instructed runner has no policy/surface scheduler. Publish
        # only its effective one-execution contract, never the emergent cross.
        normalized_policies = (CounterpartPolicy.DEFAULT.value,)
        selected_surfaces = ("default",)
        counterbalance = False
        run_probes = False

    executions_per_family = (
        4 * len(normalized_policies) * len(selected_surfaces)
        if counterbalance
        else 1
    )

    return ExecutionSurfacePlan(
        experiment_track=selected_track,
        protocol=execution_protocol,
        agent_modules=normalized_modules,
        participant_ids=participant_ids,
        captured_actor_ids=captured_actor_ids,
        per_trial_capture_policy=capture_policy,
        per_trial_captured_actor_count=capture_count,
        counterpart_types=normalized_policies,
        counterbalance=counterbalance,
        counterbalance_seed=counterbalance_seed,
        surface_variants=selected_surfaces,
        executions_per_family=executions_per_family,
        probe_mode=probe_mode,
        run_probes=run_probes,
        execution_design_scope=(
            "emergent_headline"
            if mode == "emergent"
            else (
                "emergent_headline_only"
                if mode == "both"
                else "legacy_instructed_not_parameterized"
            )
        ),
    )


def resolve_config_execution_surface(config: object) -> ExecutionSurfacePlan:
    """Normalize a validated ExperimentConfig through the public CLI contract."""
    scenarios = getattr(config, "scenarios", None)
    if scenarios is None:
        raise TypeError("config must expose a scenarios section")
    if bool(getattr(config, "use_multi_seed", False)):
        raise ValueError(
            "public CLI config launching does not yet support use_multi_seed=True"
        )
    if not bool(getattr(config, "verbose", False)):
        raise ValueError(
            "public CLI config launching requires verbose=True; quiet output is "
            "not implemented"
        )
    if bool(getattr(config, "log_to_file", False)):
        raise ValueError(
            "public CLI config launching requires log_to_file=False; no config-"
            "controlled file logger is implemented"
        )
    if bool(getattr(config, "save_activations", False)):
        raise ValueError(
            "public CLI config launching does not yet support "
            "save_activations=True; use direct CLI publication options"
        )
    plan = resolve_execution_surface(
        experiment_track=getattr(config, "experiment_track", None),
        protocol=getattr(scenarios, "protocol", ExecutionProtocol.ALTERNATING),
        counterpart_types=getattr(scenarios, "counterpart_policies", None),
        counterbalance=getattr(scenarios, "counterbalance", True),
        counterbalance_seed=getattr(scenarios, "counterbalance_seed", 0),
        surface_variants=getattr(scenarios, "surface_variants", None),
        mode=getattr(scenarios, "mode", "emergent"),
        probe_mode=getattr(scenarios, "probes", "auto"),
        agent_modules=getattr(scenarios, "agent_modules", None),
    )
    model = getattr(config, "model", None)
    if model is None:
        raise TypeError("config must expose a model section")
    if bool(getattr(model, "use_sae", False)):
        raise ValueError(
            "public CLI config launching does not yet support model.use_sae=True; "
            "the configured SAE release/id are not wired to the runtime backend"
        )
    if not bool(getattr(model, "auto_configure", True)):
        raise ValueError(
            "public CLI config launching requires model.auto_configure=True; "
            "manual backend configuration is not wired"
        )
    if (
        plan.experiment_track is not ExperimentTrack.TEXT_ONLY
        and not bool(getattr(model, "use_transformerlens", False))
    ):
        raise ValueError(
            "white-box config tracks require model.use_transformerlens=True"
        )
    if (
        plan.experiment_track is ExperimentTrack.TEXT_ONLY
        and bool(getattr(model, "use_transformerlens", False))
    ):
        raise ValueError(
            "text_only config requires model.use_transformerlens=False"
        )
    if (
        plan.experiment_track is not ExperimentTrack.TEXT_ONLY
        and not bool(getattr(model, "cache_activations", False))
    ):
        raise ValueError(
            "white-box config tracks require model.cache_activations=True"
        )

    probes = getattr(config, "probes", None)
    if probes is None:
        raise TypeError("config must expose a probes section")
    unsupported_probe_values = {
        name: getattr(probes, name, None)
        for name, expected in _SUPPORTED_PROBE_DEFAULTS.items()
        if getattr(probes, name, None) != expected
    }
    if unsupported_probe_values:
        names = ", ".join(sorted(unsupported_probe_values))
        raise ValueError(
            "public CLI config launching supports fixed probe defaults only; "
            f"unsupported override(s): {names}"
        )

    causal = getattr(config, "causal", None)
    if causal is None:
        raise TypeError("config must expose a causal section")
    if bool(getattr(causal, "enabled", False)):
        raise ValueError(
            "public CLI config launching does not yet support causal.enabled=True; "
            "the full CausalConfig contract is not wired"
        )
    causal_defaults = dict(_SUPPORTED_CAUSAL_DEFAULTS)
    if plan.experiment_track is ExperimentTrack.TEXT_ONLY:
        causal_defaults.update(
            run_activation_patching=False,
            run_ablation=False,
            run_steering=False,
        )
    unsupported_causal_values = {
        name: getattr(causal, name, None)
        for name, expected in causal_defaults.items()
        if getattr(causal, name, None) != expected
    }
    if unsupported_causal_values:
        names = ", ".join(sorted(unsupported_causal_values))
        raise ValueError(
            "public CLI config launching supports fixed inactive causal defaults "
            f"only; unsupported override(s): {names}"
        )
    return plan


__all__ = [
    "ExecutionSurfacePlan",
    "PUBLIC_AGENT_MODULES",
    "PUBLIC_PARTICIPANTS",
    "PUBLIC_RESUME_LIMITATION",
    "SOLO_PARTICIPANTS",
    "resolve_config_execution_surface",
    "resolve_execution_surface",
]
