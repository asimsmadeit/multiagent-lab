"""Canonical data contracts for interpretability experiments."""

from .schema import ActivationSample
from .activation_dataset import (
    ACTIVATION_DATASET_SCHEMA_VERSION,
    load_activation_dataset,
    save_activation_dataset,
)
from .activation_recovery import (
    ACTIVATION_RECOVERY_SCHEMA_VERSION,
    load_activation_recovery_checkpoint,
    save_activation_recovery_checkpoint,
)
from .io import (
    ARRAY_BUNDLE_SCHEMA_VERSION,
    load_array_bundle,
    load_trusted_legacy_torch,
    save_array_bundle,
)
from .selection import negotiation_sample_mask
from .splits import (
    SPLIT_MANIFEST_VERSION,
    SplitAssignment,
    SplitManifest,
    group_disjoint_split_indices,
    permute_group_blocks,
)

__all__ = [
    "ActivationSample",
    "ACTIVATION_DATASET_SCHEMA_VERSION",
    "ACTIVATION_RECOVERY_SCHEMA_VERSION",
    "ARRAY_BUNDLE_SCHEMA_VERSION",
    "SPLIT_MANIFEST_VERSION",
    "SplitAssignment",
    "SplitManifest",
    "group_disjoint_split_indices",
    "negotiation_sample_mask",
    "permute_group_blocks",
    "load_array_bundle",
    "load_activation_dataset",
    "load_activation_recovery_checkpoint",
    "load_trusted_legacy_torch",
    "save_array_bundle",
    "save_activation_dataset",
    "save_activation_recovery_checkpoint",
]
