"""Versioned agent-profile identities used in trial provenance."""

from enum import StrEnum


class AgentProfile(StrEnum):
    """Executable cognition profiles that must not be pooled silently."""

    ADVANCED = "advanced_negotiator/1"
    ULTRAFAST_MINIMAL = "ultrafast_minimal/1"


def validate_agent_profile(value: AgentProfile | str) -> AgentProfile:
    """Resolve one supported profile without accepting unversioned aliases."""
    if isinstance(value, AgentProfile):
        return value
    if not isinstance(value, str):
        raise TypeError("agent profile must be a string or AgentProfile")
    try:
        return AgentProfile(value)
    except ValueError as error:
        raise ValueError(f"unsupported agent profile: {value!r}") from error


__all__ = ["AgentProfile", "validate_agent_profile"]
