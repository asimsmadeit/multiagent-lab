"""Dataset builder for activation probe training.

This module handles the collection and serialization of activation samples
for training linear probes to detect deception.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ActivationSample:
    """Single activation sample with metadata.

    This is the core data structure for collecting activations and labels
    during experiment runs.
    """
    # Identifiers
    trial_id: str
    round_num: int
    agent_name: str

    # Activations: Dict[hook_name, Tensor]
    activations: Dict[str, torch.Tensor]

    # Labels
    actual_deception: float  # GM ground truth
    perceived_deception: float  # Agent self-report

    # Scenario info
    scenario_type: Optional[str] = None
    emergent_scenario: Optional[str] = None
    emergent_ground_truth: Optional[bool] = None
    incentive_condition: Optional[str] = None

    # SAE features (optional)
    sae_features: Optional[Dict[int, float]] = None
    sae_top_features: Optional[List[int]] = None


class DatasetBuilder:
    """Builds and saves activation datasets for probe training.

    This class collects activation samples during experiment runs and
    saves them in a format compatible with train_probes.py.

    Example:
        builder = DatasetBuilder()

        # During experiment
        for trial in trials:
            sample = ActivationSample(...)
            builder.add_sample(sample)

        # After experiment
        builder.save("activations.pt")
    """

    def __init__(self):
        """Initialize empty dataset builder."""
        self.samples: List[ActivationSample] = []

    def add_sample(self, sample: ActivationSample) -> None:
        """Add an activation sample to the dataset."""
        self.samples.append(sample)

    def clear(self) -> None:
        """Clear all samples."""
        self.samples = []

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def save(self, filepath: str, model_name: str = "unknown") -> None:
        """Save activation dataset in format compatible with train_probes.py.

        Args:
            filepath: Path to save the .pt file
            model_name: Name of the model (for metadata)
        """
        if not self.samples:
            logger.warning("No samples to save!")
            return

        # Collect data by layer
        activations_by_layer: Dict[int, List[torch.Tensor]] = {}
        all_gm_deception: List[float] = []
        all_agent_deception: List[float] = []
        all_scenarios: List[str] = []
        all_sae_features: List[Dict[int, float]] = []
        all_sae_top_features: List[List[int]] = []
        metadata: List[Dict[str, Any]] = []

        for sample in self.samples:
            # Organize activations by layer
            for layer_name, activation in sample.activations.items():
                # Extract layer number from hook name
                # e.g., "blocks.21.hook_resid_post" -> 21
                try:
                    layer_num = int(layer_name.split('.')[1])
                except (IndexError, ValueError):
                    layer_num = layer_name

                if layer_num not in activations_by_layer:
                    activations_by_layer[layer_num] = []
                activations_by_layer[layer_num].append(activation)

            # GM ground truth label
            if sample.emergent_ground_truth is not None:
                gm_label = 1.0 if sample.emergent_ground_truth else 0.0
            else:
                gm_label = sample.actual_deception
            all_gm_deception.append(gm_label)

            # Agent self-report label
            all_agent_deception.append(sample.perceived_deception)

            # Scenario name
            scenario = sample.emergent_scenario or sample.scenario_type
            all_scenarios.append(scenario)

            # Metadata for each sample
            metadata.append({
                'trial_id': sample.trial_id,
                'round_num': sample.round_num,
                'agent_name': sample.agent_name,
                'scenario': scenario,
                'incentive_condition': sample.incentive_condition,
                'emergent_ground_truth': sample.emergent_ground_truth,
                'actual_deception': sample.actual_deception,
                'perceived_deception': sample.perceived_deception,
            })

            # SAE features (if available)
            if sample.sae_features is not None:
                all_sae_features.append(sample.sae_features)
                all_sae_top_features.append(sample.sae_top_features or [])

        # Stack activations by layer: Dict[layer_num, Tensor[N, d_model]]
        stacked_activations = {}
        for layer_num, acts in activations_by_layer.items():
            stacked_activations[layer_num] = torch.stack(acts)

        # Format expected by train_probes.py
        dataset = {
            'activations': stacked_activations,
            'labels': {
                'gm_labels': all_gm_deception,
                'agent_labels': all_agent_deception,
                'scenario': all_scenarios,
            },
            'config': {
                'model': model_name,
                'layers': list(stacked_activations.keys()),
                'n_samples': len(all_gm_deception),
                'has_sae': len(all_sae_features) > 0,
            },
            'metadata': metadata,
        }

        # Add SAE features if available
        if all_sae_features:
            try:
                max_idx = max(max(f.keys()) for f in all_sae_features if f)
                sae_dim = max_idx + 1

                sae_tensor = torch.zeros(len(all_sae_features), sae_dim)
                for i, features in enumerate(all_sae_features):
                    if features:
                        for idx, val in features.items():
                            sae_tensor[i, idx] = val

                dataset['sae_features'] = sae_tensor
                dataset['sae_top_features'] = all_sae_top_features
                dataset['config']['sae_dim'] = sae_dim
            except Exception as e:
                logger.warning("Could not save SAE features: %s", e)

        torch.save(dataset, filepath)
        self._log_summary(filepath, stacked_activations, all_gm_deception,
                         all_scenarios, all_sae_features, dataset)

    def _log_summary(self, filepath: str, stacked_activations: Dict,
                     all_gm_deception: List[float], all_scenarios: List[str],
                     all_sae_features: List, dataset: Dict) -> None:
        """Log summary of saved dataset."""
        n_samples = len(all_gm_deception)
        layers = sorted(stacked_activations.keys())
        d_model = stacked_activations[layers[0]].shape[1] if layers else 0

        logger.info("Saved %d samples to %s", n_samples, filepath)
        logger.info("Layers: %s", layers)
        logger.info("Activation dim: %d", d_model)
        logger.info("GM deception rate: %.1f%%", np.mean(all_gm_deception) * 100)

        if all_sae_features:
            logger.info("SAE features: %d samples, dim=%s",
                       len(all_sae_features), dataset['config'].get('sae_dim', 'N/A'))
        else:
            logger.info("SAE features: None (not captured or SAE disabled)")

        # Per-scenario breakdown
        unique_scenarios = set(all_scenarios)
        if len(unique_scenarios) > 1:
            logger.info("Per-scenario deception rates:")
            for scenario in sorted(unique_scenarios):
                mask = [s == scenario for s in all_scenarios]
                rate = np.mean([all_gm_deception[i] for i, m in enumerate(mask) if m])
                count = sum(mask)
                logger.info("  %s: %.1f%% (%d samples)", scenario, rate * 100, count)

    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """Load a saved dataset.

        Args:
            filepath: Path to the .pt file

        Returns:
            Dataset dictionary with activations, labels, config, metadata
        """
        return torch.load(filepath, weights_only=False)
