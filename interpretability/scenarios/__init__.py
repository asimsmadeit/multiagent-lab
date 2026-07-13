"""Compiled scenario contracts and prompt-policy variants."""

from interpretability.scenarios.compiled import (
    COUNTERPART_KNOWLEDGE_GRANT_VERSION,
    EMERGENT_SCENARIO_SPEC_VERSION,
    SUPPORTED_COUNTERPART_POLICIES,
    CounterpartPolicy,
    ExecutionProtocol,
    compile_emergent_scenario,
    render_actor_prompt,
    render_counterpart_prompt,
    validate_counterpart_policy,
    validate_counterpart_policy_contract,
    validate_execution_protocol,
)

__all__ = [
    "COUNTERPART_KNOWLEDGE_GRANT_VERSION",
    "EMERGENT_SCENARIO_SPEC_VERSION",
    "SUPPORTED_COUNTERPART_POLICIES",
    "CounterpartPolicy",
    "ExecutionProtocol",
    "compile_emergent_scenario",
    "render_actor_prompt",
    "render_counterpart_prompt",
    "validate_counterpart_policy",
    "validate_counterpart_policy_contract",
    "validate_execution_protocol",
]
