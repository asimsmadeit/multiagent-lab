"""Published schema-version registry for run and artifact provenance."""

from __future__ import annotations

import hashlib
import json
from typing import Dict

from interpretability.causal.design import CAUSAL_DESIGN_SCHEMA_VERSION
from interpretability.causal.execution import (
    CAUSAL_APPLICATION_RECEIPT_SCHEMA_VERSION,
)
from interpretability.core.qc_filter import QC_VERSION
from interpretability.data.activation_dataset import (
    ACTIVATION_DATASET_SCHEMA_VERSION,
)
from interpretability.data.activation_recovery import (
    ACTIVATION_RECOVERY_SCHEMA_VERSION,
)
from interpretability.data.io import ARRAY_BUNDLE_SCHEMA_VERSION
from interpretability.data.splits import SPLIT_MANIFEST_VERSION
from interpretability.labels.schema import LABEL_SCHEMA_VERSION
from interpretability.probes.artifacts import HEADLINE_PROBE_SCHEMA_VERSION
from interpretability.runtime.model_call import GENERATION_SCHEMA_VERSION
from interpretability.runtime.interventions import (
    INTERVENTION_APPLICATION_LOG_SCHEMA_VERSION,
    INTERVENTION_APPLICATION_SCHEMA_VERSION,
    INTERVENTION_DESIGN_SCHEMA_VERSION,
    INTERVENTION_SCHEDULE_SCHEMA_VERSION,
    PROBE_INTERVENTION_SCHEMA_VERSION,
    SCRIPTED_OBSERVATION_SCHEMA_VERSION,
)
from interpretability.runtime.protocols import SOLO_NO_RESPONSE_PROTOCOL_VERSION
from interpretability.runtime.runner import RUNTIME_EXECUTOR_VERSION
from interpretability.runtime.trial import TRIAL_RUNTIME_SCHEMA_VERSION
from interpretability.scenarios.compiled import (
    COUNTERPART_KNOWLEDGE_GRANT_VERSION,
    EMERGENT_SCENARIO_SPEC_VERSION,
)
from negotiation.domain.schema import SCHEMA_VERSION as NEGOTIATION_DOMAIN_VERSION
from negotiation.game_master.adjudication import (
    ADJUDICATION_VERSION,
    SIMULTANEOUS_BATCH_VERSION,
)


SCHEMA_REGISTRY_VERSION = "2.2.0"


def schema_registry() -> Dict[str, str]:
    """Return public persisted artifact and nested-record contract versions."""
    return {
        "adjudication": ADJUDICATION_VERSION,
        "activation_dataset": ACTIVATION_DATASET_SCHEMA_VERSION,
        "activation_recovery_checkpoint": ACTIVATION_RECOVERY_SCHEMA_VERSION,
        "array_bundle": ARRAY_BUNDLE_SCHEMA_VERSION,
        "causal_application_receipt": (
            CAUSAL_APPLICATION_RECEIPT_SCHEMA_VERSION
        ),
        "causal_design": CAUSAL_DESIGN_SCHEMA_VERSION,
        "counterpart_knowledge_grant": COUNTERPART_KNOWLEDGE_GRANT_VERSION,
        "emergent_scenario_spec": EMERGENT_SCENARIO_SPEC_VERSION,
        "generation_record": GENERATION_SCHEMA_VERSION,
        "headline_probe": HEADLINE_PROBE_SCHEMA_VERSION,
        "intervention_application": INTERVENTION_APPLICATION_SCHEMA_VERSION,
        "intervention_application_log": (
            INTERVENTION_APPLICATION_LOG_SCHEMA_VERSION
        ),
        "intervention_design": INTERVENTION_DESIGN_SCHEMA_VERSION,
        "intervention_schedule": INTERVENTION_SCHEDULE_SCHEMA_VERSION,
        "label_record": LABEL_SCHEMA_VERSION,
        "negotiation_domain": NEGOTIATION_DOMAIN_VERSION,
        "response_qc": QC_VERSION,
        "runtime_executor": RUNTIME_EXECUTOR_VERSION,
        "schema_registry": SCHEMA_REGISTRY_VERSION,
        "split_manifest": SPLIT_MANIFEST_VERSION,
        "simultaneous_batch": SIMULTANEOUS_BATCH_VERSION,
        "solo_no_response_protocol": SOLO_NO_RESPONSE_PROTOCOL_VERSION,
        "probe_intervention": PROBE_INTERVENTION_SCHEMA_VERSION,
        "scripted_observation": SCRIPTED_OBSERVATION_SCHEMA_VERSION,
        "trial_runtime": TRIAL_RUNTIME_SCHEMA_VERSION,
    }


def schema_registry_checksum() -> str:
    """Hash the canonical registry for inclusion in run manifests."""
    encoded = json.dumps(
        schema_registry(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SCHEMA_REGISTRY_VERSION",
    "schema_registry",
    "schema_registry_checksum",
]
