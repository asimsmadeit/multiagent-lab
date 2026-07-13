"""Canonical schemas shared by activation producers and serializers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class ActivationSample:
    """One activation row with acting-agent and counterpart-belief labels.

    The required fields are the intersection used by both the experiment
    runner and :class:`DatasetBuilder`. Remaining defaults preserve legacy
    keyword construction while allowing lightweight offline datasets.
    """

    # Row identity and activation payload.
    trial_id: int | str
    round_num: int
    agent_name: str
    activations: Dict[str, torch.Tensor]

    # Existing scalar label semantics are intentionally unchanged.
    actual_deception: Optional[float]
    perceived_deception: Optional[float]

    # Generated context and auxiliary scalar labels.
    prompt: str = ""
    response: str = ""
    emotion_intensity: float = 0.0
    trust_level: Optional[float] = None
    cooperation_intent: float = 0.0
    commitment_violation: Optional[float] = 0.0
    manipulation_score: Optional[float] = 0.0
    consistency_score: Optional[float] = 1.0

    # Negotiation and execution context.
    scenario_type: Optional[str] = None
    modules_enabled: List[str] = field(default_factory=list)
    pod_id: int = 0
    gm_modules_enabled: List[str] = field(default_factory=list)
    counterpart_idx: Optional[int] = None
    counterpart_name: Optional[str] = None
    trial_outcome: Optional[str] = None
    joint_value: Optional[float] = None
    agent_utility: Optional[float] = None
    condition_id: Optional[str] = None

    # Optional intervention/capture payloads. None retains "not captured".
    followup_activations: Optional[Dict[str, torch.Tensor]] = None
    emergent_scenario: Optional[str] = None
    incentive_condition: Optional[str] = None
    scenario_params: Dict[str, Any] = field(default_factory=dict)
    emergent_ground_truth: Optional[bool] = None
    sae_features: Optional[Dict[int, float]] = None
    sae_top_features: Optional[List[int]] = None
    sae_sparsity: Optional[float] = None

    # General provenance and research annotations.
    # Wall-clock time is operational metadata, not part of a scientific row.
    # Producers that need it must inject a persisted value explicitly.
    timestamp: Optional[str] = None
    experiment_mode: Optional[str] = None
    experiment_track: Optional[str] = None
    execution_protocol: Optional[str] = None
    tom_state: Optional[Dict[str, Any]] = None
    deepeval_false_claims: Optional[float] = None
    deepeval_omission: Optional[float] = None
    deepeval_framing: Optional[float] = None
    deepeval_commitment: Optional[float] = None
    deepeval_confidence: Optional[float] = None
    deepeval_reasoning: Optional[str] = None
    ground_truth_evaluation_succeeded: Optional[bool] = None
    ground_truth_evaluation_method: Optional[str] = None
    ground_truth_evaluation_error: Optional[str] = None
    component_reasoning: Optional[Dict[str, str]] = None
    dialogue_history: Optional[List[str]] = None
    is_verification_probe: bool = False
    plausibility_response: Optional[str] = None
    # Rows are ineligible until a producer explicitly establishes that they
    # are an adjudicated negotiation decision and records a reproducible QC
    # result. This avoids legacy/default construction becoming training data.
    sample_type: str = "unclassified"
    semantic_phase: Optional[str] = None
    qc_flags: List[str] = field(default_factory=list)
    qc_status: str = "unreviewed"
    qc_version: Optional[str] = None
    counterpart_type: Optional[str] = None
    belief_shift_injected: bool = False
    belief_shift_type: Optional[str] = None
    belief_shift_round: Optional[int] = None
    framing_variant: Optional[str] = None
    activation_position: str = "last_response_token"
    sampling_config: Dict[str, Any] = field(default_factory=dict)

    # Parallel evaluator projections retained during the schema migration.
    gt_regex: Optional[float] = None
    gt_llm_rules: Optional[float] = None
    gt_deepeval: Optional[float] = None

    # Canonical references introduced without replacing legacy scalar labels.
    generation_record_id: Optional[str] = None
    interaction_event_id: Optional[str] = None
    label_record_ids: List[str] = field(default_factory=list)
    actual_deception_projection: Optional[float] = None
    trial_family_id: Optional[str] = None
    scenario_instance_id: Optional[str] = None
    role_assignment_id: Optional[str] = None
    order_assignment_id: Optional[str] = None
    counterpart_assignment_id: Optional[str] = None
    surface_assignment_id: Optional[str] = None
    counterbalance_id: Optional[str] = None
    first_mover_id: Optional[str] = None
    role_assignment: Dict[str, str] = field(default_factory=dict)
    surface_assignment: Dict[str, str] = field(default_factory=dict)
    actor_profile: Optional[str] = None
    counterpart_profile: Optional[str] = None
    intervention_design_id: Optional[str] = None
    intervention_application_receipt_ids: List[str] = field(default_factory=list)
