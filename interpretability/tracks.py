"""Explicit experiment tracks separated by model-state access assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExperimentTrack(str, Enum):
    """Supported access regimes that must not be pooled silently."""

    TEXT_ONLY = "text_only"
    SINGLE_AGENT_WHITE_BOX = "single_agent_white_box"
    BILATERAL_WHITE_BOX = "bilateral_white_box"
    THEORY_OF_MIND = "theory_of_mind"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class TrackSpec:
    """Auditable access and module requirements for one experiment track."""

    track: ExperimentTrack
    activation_access: str
    captured_actor_count: int | None
    required_modules: tuple[str, ...] = ()
    allows_online_adaptation: bool = False


_TRACK_SPECS = {
    ExperimentTrack.TEXT_ONLY: TrackSpec(
        ExperimentTrack.TEXT_ONLY, "none", 0
    ),
    ExperimentTrack.SINGLE_AGENT_WHITE_BOX: TrackSpec(
        ExperimentTrack.SINGLE_AGENT_WHITE_BOX, "one_declared_actor", 1
    ),
    ExperimentTrack.BILATERAL_WHITE_BOX: TrackSpec(
        ExperimentTrack.BILATERAL_WHITE_BOX, "all_actors", None
    ),
    ExperimentTrack.THEORY_OF_MIND: TrackSpec(
        ExperimentTrack.THEORY_OF_MIND,
        "one_declared_actor",
        1,
        required_modules=("theory_of_mind",),
    ),
    ExperimentTrack.ADAPTIVE: TrackSpec(
        ExperimentTrack.ADAPTIVE,
        "declared_by_manifest",
        None,
        required_modules=("strategy_evolution",),
        allows_online_adaptation=True,
    ),
}


def get_track_spec(track: ExperimentTrack | str) -> TrackSpec:
    """Resolve one versioned access regime without fallback guessing."""
    return _TRACK_SPECS[ExperimentTrack(track)]


def validate_track_assignment(
    track: ExperimentTrack | str,
    *,
    participant_ids: tuple[str, ...],
    captured_actor_ids: tuple[str, ...],
    enabled_modules: tuple[str, ...] = (),
) -> TrackSpec:
    """Reject access/module assignments inconsistent with the declared track."""
    spec = get_track_spec(track)
    participant_ids = tuple(map(str, participant_ids))
    captured_actor_ids = tuple(map(str, captured_actor_ids))
    enabled_modules = tuple(map(str, enabled_modules))
    if len(participant_ids) < 2 or len(set(participant_ids)) != len(participant_ids):
        raise ValueError("tracks require at least two unique participants")
    if any(not participant for participant in participant_ids):
        raise ValueError("track participant IDs must be non-empty")
    if len(set(captured_actor_ids)) != len(captured_actor_ids):
        raise ValueError("captured actors cannot contain duplicates")
    if len(set(enabled_modules)) != len(enabled_modules):
        raise ValueError("enabled modules cannot contain duplicates")
    if not set(captured_actor_ids).issubset(participant_ids):
        raise ValueError("captured actors must be trial participants")
    if spec.captured_actor_count is not None:
        if len(captured_actor_ids) != spec.captured_actor_count:
            raise ValueError(
                f"{spec.track.value} requires {spec.captured_actor_count} "
                "captured actor(s)"
            )
    elif spec.track is ExperimentTrack.BILATERAL_WHITE_BOX:
        if set(captured_actor_ids) != set(participant_ids):
            raise ValueError("bilateral_white_box requires every actor to be captured")
    missing_modules = set(spec.required_modules).difference(enabled_modules)
    if missing_modules:
        raise ValueError(f"track requires modules: {sorted(missing_modules)}")
    adaptive_modules = {"strategy_evolution"}
    enabled_adaptation = adaptive_modules.intersection(enabled_modules)
    if enabled_adaptation and not spec.allows_online_adaptation:
        raise ValueError(
            f"{spec.track.value} does not allow online adaptation modules: "
            f"{sorted(enabled_adaptation)}"
        )
    if spec.allows_online_adaptation and not captured_actor_ids:
        raise ValueError("adaptive tracks require at least one captured actor")
    return spec


__all__ = [
    "ExperimentTrack",
    "TrackSpec",
    "get_track_spec",
    "validate_track_assignment",
]
