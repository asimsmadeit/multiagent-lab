"""Core experiment configuration for mechanistic interpretability research.

This is the central configuration hub for all deception detection experiments.
Uses Pydantic for validation and serialization.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# MODEL PRESETS - Auto-configuration for supported models
# =============================================================================

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # GEMMA FAMILY (Primary - has SAE support via Gemma Scope)
    # =========================================================================
    # Gemma 2B - Fast, lightweight (TransformerLens compatible)
    "google/gemma-2b-it": {
        "n_layers": 18,
        "d_model": 2048,
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_layer": 12,  # ~67% depth (good for high-level features)
        "sae_id": "layer_12/width_16k/canonical",
        "layers_to_probe": [4, 9, 12, 16],  # 25%, 50%, 67%, 89%
        "vram_gb": 4,
    },
    # Gemma 7B - Better quality, more VRAM (TransformerLens compatible)
    "google/gemma-7b-it": {
        "n_layers": 28,
        "d_model": 3072,
        "sae_release": "gemma-scope-7b-pt-res-canonical",
        "sae_layer": 20,  # ~71% depth
        "sae_id": "layer_20/width_16k/canonical",
        "layers_to_probe": [7, 14, 20, 25],  # 25%, 50%, 71%, 89%
        "vram_gb": 16,
    },
    # Gemma 2 27B - Research quality model (PRIMARY for deception research)
    "google/gemma-2-27b-it": {
        "n_layers": 46,
        "d_model": 4608,
        "sae_release": "gemma-scope-27b-pt-res-canonical",
        "sae_layer": 31,  # ~67% depth (optimal for high-level features)
        "sae_id": "layer_31/width_16k/canonical",
        "layers_to_probe": [12, 23, 31, 40],  # 25%, 50%, 67%, 87%
        "vram_gb": 55,
    },
    # =========================================================================
    # LLAMA FAMILY (No SAE, but good for cross-model comparison)
    # =========================================================================
    # Llama 2 7B - Legacy but stable
    "meta-llama/Llama-2-7b-chat-hf": {
        "n_layers": 32,
        "d_model": 4096,
        "sae_release": None,  # No SAE available
        "sae_layer": None,
        "sae_id": None,
        "layers_to_probe": [8, 16, 22, 28],  # 25%, 50%, 69%, 88%
        "vram_gb": 14,
    },
    # Llama 3 8B - Current generation, excellent quality
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "n_layers": 32,
        "d_model": 4096,
        "sae_release": None,
        "sae_layer": None,
        "sae_id": None,
        "layers_to_probe": [8, 16, 22, 28],  # 25%, 50%, 69%, 88%
        "vram_gb": 16,
    },
    # Llama 3.1 8B - Latest Llama
    "meta-llama/Llama-3.1-8B-Instruct": {
        "n_layers": 32,
        "d_model": 4096,
        "sae_release": None,
        "sae_layer": None,
        "sae_id": None,
        "layers_to_probe": [8, 16, 22, 28],
        "vram_gb": 16,
    },
    # Llama 3.1 70B - Large scale comparison
    "meta-llama/Llama-3.1-70B-Instruct": {
        "n_layers": 80,
        "d_model": 8192,
        "sae_release": None,
        "sae_layer": None,
        "sae_id": None,
        "layers_to_probe": [20, 40, 54, 70],  # 25%, 50%, 67%, 88%
        "vram_gb": 140,
    },
    # =========================================================================
    # MISTRAL FAMILY (No SAE, efficient architecture)
    # =========================================================================
    # Mistral 7B - Efficient, high quality
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "n_layers": 32,
        "d_model": 4096,
        "sae_release": None,
        "sae_layer": None,
        "sae_id": None,
        "layers_to_probe": [8, 16, 22, 28],
        "vram_gb": 14,
    },
    # Mixtral 8x7B - MoE architecture (different interpretation patterns)
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "n_layers": 32,
        "d_model": 4096,
        "sae_release": None,
        "sae_layer": None,
        "sae_id": None,
        "layers_to_probe": [8, 16, 22, 28],
        "vram_gb": 90,  # MoE requires more memory
    },
    # =========================================================================
    # QWEN FAMILY (No SAE, strong multilingual)
    # =========================================================================
    # Qwen2 7B - Strong alternative
    "Qwen/Qwen2-7B-Instruct": {
        "n_layers": 28,
        "d_model": 3584,
        "sae_release": None,
        "sae_layer": None,
        "sae_id": None,
        "layers_to_probe": [7, 14, 19, 25],  # 25%, 50%, 68%, 89%
        "vram_gb": 15,
    },
    # Qwen2.5 7B - Latest Qwen
    "Qwen/Qwen2.5-7B-Instruct": {
        "n_layers": 28,
        "d_model": 3584,
        "sae_release": None,
        "sae_layer": None,
        "sae_id": None,
        "layers_to_probe": [7, 14, 19, 25],
        "vram_gb": 15,
    },
    # =========================================================================
    # PHI FAMILY (No SAE, compact/efficient)
    # =========================================================================
    # Phi-3 Mini - Very efficient
    "microsoft/Phi-3-mini-4k-instruct": {
        "n_layers": 32,
        "d_model": 3072,
        "sae_release": None,
        "sae_layer": None,
        "sae_id": None,
        "layers_to_probe": [8, 16, 22, 28],
        "vram_gb": 8,
    },
    # Phi-4 - Latest, very capable
    "microsoft/phi-4": {
        "n_layers": 40,
        "d_model": 5120,
        "sae_release": None,
        "sae_layer": None,
        "sae_id": None,
        "layers_to_probe": [10, 20, 27, 35],  # 25%, 50%, 67%, 88%
        "vram_gb": 28,
    },
}


def get_model_preset(model_name: str) -> Optional[Dict[str, Any]]:
    """Get preset configuration for a model, or None if not found."""
    return MODEL_PRESETS.get(model_name)


class ModelConfig(BaseModel):
    """LLM and interpretability model configuration.

    Auto-configuration: When `auto_configure=True` (default), SAE settings
    and probe layers are automatically set based on the model name.
    Supported models: gemma-2b-it, gemma-7b-it, gemma-7b-it, Llama-2-7b.

    Example:
        # Just set the model name - everything else auto-configures
        config = ModelConfig(name="google/gemma-7b-it")
        # sae_release, sae_layer, sae_id all set automatically
    """

    # Model selection
    name: str = Field(
        default="google/gemma-2-27b-it",
        description="""HuggingFace model name (must be TransformerLens compatible). Tested models:
        - google/gemma-2b-it (fast, ~4GB VRAM)
        - google/gemma-7b-it (better, ~16GB VRAM)
        - google/gemma-2-27b-it (research quality, ~55GB VRAM) [DEFAULT]
        - meta-llama/Llama-2-7b-chat-hf (~14GB VRAM)
        """
    )

    # Auto-configuration
    auto_configure: bool = Field(
        default=True,
        description="Auto-set SAE and probe settings based on model name"
    )

    # Hardware
    device: Literal["cuda", "cpu", "mps"] = Field(
        default="cuda",
        description="Device to run model on"
    )
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="bfloat16",
        description="Model dtype for memory/speed tradeoff"
    )

    # TransformerLens settings
    use_transformerlens: bool = Field(
        default=True,
        description="Enable activation capture via TransformerLens"
    )
    cache_activations: bool = Field(
        default=True,
        description="Cache activations for probe training"
    )

    # SAE settings (auto-configured if auto_configure=True)
    use_sae: bool = Field(
        default=True,
        description="Enable Sparse Autoencoder analysis"
    )
    sae_release: Optional[str] = Field(
        default=None,
        description="SAE Lens release name (auto-set from model if None)"
    )
    sae_layer: Optional[int] = Field(
        default=None,
        ge=0,
        description="Which layer to analyze with SAE (auto-set from model if None)"
    )
    sae_id: Optional[str] = Field(
        default=None,
        description="Specific SAE ID within the release (auto-set from model if None)"
    )

    @model_validator(mode='after')
    def apply_model_preset(self) -> 'ModelConfig':
        """Auto-configure SAE settings based on model name."""
        if not self.auto_configure:
            return self

        preset = get_model_preset(self.name)
        if preset is None:
            # Unknown model - use defaults or warn
            if self.sae_release is None:
                self.use_sae = False  # Disable SAE for unknown models
            return self

        # Apply preset values only if not explicitly set
        if self.sae_release is None:
            self.sae_release = preset.get("sae_release")
        if self.sae_layer is None:
            self.sae_layer = preset.get("sae_layer")
        if self.sae_id is None:
            self.sae_id = preset.get("sae_id")

        # Disable SAE if no release available (e.g., Llama)
        if self.sae_release is None:
            self.use_sae = False

        return self

    def get_recommended_probe_layers(self) -> List[int]:
        """Get recommended layers to probe for this model."""
        preset = get_model_preset(self.name)
        if preset:
            return preset.get("layers_to_probe", [10, 15, 20, 25])
        return [10, 15, 20, 25]  # Default fallback

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture info."""
        preset = get_model_preset(self.name)
        if preset:
            return {
                "n_layers": preset.get("n_layers"),
                "d_model": preset.get("d_model"),
                "vram_gb": preset.get("vram_gb"),
                "sae_available": preset.get("sae_release") is not None,
            }
        return {"n_layers": None, "d_model": None, "vram_gb": None, "sae_available": False}


class EvaluatorConfig(BaseModel):
    """Lightweight local evaluator model for ground truth extraction.

    The evaluator is a separate (smaller) model that judges whether
    the subject model's responses are deceptive. This separation ensures:
    1. The subject model is tested without self-evaluation bias
    2. Faster evaluation with a lightweight model
    3. Consistent evaluation across different subject models
    """

    enabled: bool = Field(default=True, description="Enable separate evaluator model")
    model: str = Field(
        default="google/gemma-2b-it",
        description="Lightweight model for evaluation (~4GB VRAM)"
    )
    max_tokens: int = Field(
        default=64,
        gt=0,
        description="Max tokens for evaluator responses (short = faster)"
    )


class ProbeConfig(BaseModel):
    """Linear probe training configuration.

    Token Position for Probing:
        This implementation uses LAST TOKEN activations for probing, following
        the standard approach in mechanistic interpretability research:

        - Autoregressive models encode the "answer" in the last token position
        - The last token contains the most decision-relevant information
        - This matches Apollo Research's methodology for deception detection

        Alternative approaches (not currently implemented):
        - Mean pooling: Average across all token positions
        - First token: For encoder models (not applicable to decoder-only)
        - Specific positions: e.g., after key phrases

        Reference: Marks et al. 2023 "The Geometry of Truth" uses last token.

    Note: `layers_to_probe` can be auto-configured using `ProbeConfig.for_model()`.
    """

    # Training
    train_ratio: float = Field(
        default=0.8,
        gt=0,
        lt=1,
        description="Train/test split ratio"
    )
    regularization: float = Field(
        default=1.0,
        gt=0,
        description="L2 regularization strength"
    )
    max_iter: int = Field(
        default=1000,
        gt=0,
        description="Maximum training iterations"
    )

    # Layers to probe (use for_model() to auto-set based on model)
    layers_to_probe: List[int] = Field(
        default=[12, 23, 31, 40],  # Default for 27B model (46 layers): 25%, 50%, 67%, 87%
        description="Which layers to train probes on. Use ProbeConfig.for_model() to auto-configure."
    )

    @classmethod
    def for_model(cls, model_name: str, **kwargs) -> "ProbeConfig":
        """Create ProbeConfig with layers optimized for a specific model.

        Example:
            probes = ProbeConfig.for_model("google/gemma-7b-it")
            # layers_to_probe = [10, 21, 31, 38] (auto-set for 9B)
        """
        preset = get_model_preset(model_name)
        if preset and "layers_to_probe" in preset:
            kwargs.setdefault("layers_to_probe", preset["layers_to_probe"])
        return cls(**kwargs)

    # Token position for activation capture
    token_position: Literal["last", "mean", "all"] = Field(
        default="last",
        description="""Which token position(s) to use for probing:
        - 'last': Use only the last token activation (default, recommended)
        - 'mean': Average activations across all token positions
        - 'all': Concatenate all token positions (increases dimensionality)

        Last token is standard for autoregressive models as it contains
        the most decision-relevant information for the model's next action.
        """
    )

    # Validation
    min_accuracy: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Minimum accuracy threshold for valid probe"
    )
    run_sanity_checks: bool = Field(
        default=True,
        description="Run probe sanity checks (shuffled labels, random features)"
    )
    run_cross_scenario_validation: bool = Field(
        default=True,
        description="""Run leave-one-scenario-out (LOSO) cross-validation.

        This tests whether the probe generalizes across different scenarios.
        A probe that only works within-scenario may be learning scenario-specific
        artifacts rather than general deception features.

        Interpretation of results:
        - Cross-scenario AUC > 0.7: Probe captures general deception
        - Cross-scenario AUC < 0.55: Probe may be scenario-specific
        - Large variance: Some scenarios have different deception patterns
        """
    )
    binary_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="""Threshold for binarizing continuous labels to binary deceptive/honest.

        Default 0.5 assumes labels are in [0, 1] range with 0.5 as the natural midpoint.
        For probabilistic labels, 0.5 represents "more likely deceptive than not".

        IMPORTANT: Always report R² (continuous metric) alongside binary metrics
        (accuracy, AUC) to avoid information loss from binarization.

        Use threshold_sensitivity_analysis() in train_probes.py to check robustness.
        """
    )
    run_threshold_sensitivity: bool = Field(
        default=True,
        description="""Run threshold sensitivity analysis to verify 0.5 threshold is appropriate.
        Results show how metrics change with different thresholds (0.3, 0.4, 0.5, 0.6, 0.7).
        """
    )

    @field_validator('layers_to_probe')
    @classmethod
    def validate_layers(cls, v):
        """Ensure layers_to_probe is not empty."""
        if not v:
            raise ValueError("layers_to_probe cannot be empty")
        return v


class CausalConfig(BaseModel):
    """Causal validation configuration (activation patching, ablation)."""

    enabled: bool = Field(default=True, description="Enable causal validation")
    num_samples: int = Field(
        default=30,
        gt=0,
        description="Number of samples for causal interventions"
    )

    # Intervention types
    run_activation_patching: bool = Field(
        default=True,
        description="Patch activations between deceptive/honest examples"
    )
    run_ablation: bool = Field(
        default=True,
        description="Zero-ablate identified deception features"
    )
    run_steering: bool = Field(
        default=True,
        description="Test steering vectors for behavior modification"
    )

    # Thresholds
    min_effect_size: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Minimum effect size to consider significant"
    )


class ScenarioConfig(BaseModel):
    """Deception scenario configuration."""

    mode: Literal["emergent", "instructed", "both"] = Field(
        default="both",
        description="""Scenario mode:
        - emergent: Incentive-based deception (no explicit instructions)
        - instructed: Explicit deception instructions (Apollo-style)
        - both: Run BOTH emergent and instructed modes [DEFAULT]
        - contest: Game-theoretic scenarios (Fishery, Treaty, Gameshow)
        """
    )
    scenarios: List[str] = Field(
        default=[
            "alliance_betrayal",
            "promise_break",
            "capability_bluff",
            "info_withholding",
        ],
        description="""Which scenarios to run (4 selected for balanced deception rates).
        Available emergent scenarios:
        - alliance_betrayal: 3-player alliance with betrayal incentive [DEFAULT]
        - promise_break: Prisoner's dilemma trust violations [DEFAULT]
        - capability_bluff: Inflated timeline estimates [DEFAULT]
        - info_withholding: Concealing product defects [DEFAULT]
        - ultimatum_bluff: Bluffing about walking away
        - hidden_value: Hiding true preferences
        """
    )
    num_trials: int = Field(
        default=50,
        gt=0,
        description="Number of trials per scenario"
    )
    max_rounds: int = Field(
        default=5,
        gt=0,
        description="Maximum conversation rounds per trial (5 for multi-agent temporal analysis)"
    )

    @field_validator('scenarios')
    @classmethod
    def validate_scenarios(cls, v):
        """Ensure scenarios list is not empty."""
        if not v:
            raise ValueError("scenarios list cannot be empty")
        return v


class ExperimentConfig(BaseModel):
    """Main experiment configuration combining all settings."""

    # Sub-configs
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    probes: ProbeConfig = Field(default_factory=ProbeConfig)
    causal: CausalConfig = Field(default_factory=CausalConfig)
    scenarios: ScenarioConfig = Field(default_factory=ScenarioConfig)

    # Output
    output_dir: str = Field(
        default="./results",
        description="Directory for experiment outputs"
    )
    checkpoint_dir: str = Field(
        default="./checkpoints",
        description="Directory for checkpoints (resume interrupted experiments)"
    )
    save_activations: bool = Field(
        default=False,
        description="Save raw activations (large files)"
    )

    # Experiment metadata
    experiment_name: Optional[str] = Field(
        default=None,
        description="Optional experiment name for organizing results"
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility. Used when random_seeds is empty."
    )
    random_seeds: List[int] = Field(
        default=[42, 123, 456, 789, 1337],
        description="""List of random seeds for multi-seed experiments.
        Running with multiple seeds ensures results are robust and not seed-dependent.
        Results should be reported as mean ± std across seeds.
        """
    )
    use_multi_seed: bool = Field(
        default=False,
        description="""If True, run experiments with all seeds in random_seeds.
        If False, use only random_seed for faster single-run experiments.
        """
    )

    # Logging
    verbose: bool = Field(default=True, description="Enable verbose output")
    log_to_file: bool = Field(default=True, description="Log output to file")

    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls.model_validate(data)

    def save_json(self, path: str) -> None:
        """Save config to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON file."""
        import json
        with open(path, 'r') as f:
            return cls.model_validate(json.load(f))

    @classmethod
    def for_model(
        cls,
        model_name: str,
        num_trials: int = 50,
        scenarios: Optional[List[str]] = None,
        **kwargs
    ) -> "ExperimentConfig":
        """Create fully auto-configured experiment for a specific model.

        This is the easiest way to switch between models - everything
        (SAE settings, probe layers, etc.) auto-configures based on model.

        Example:
            # Switch between models with one line:
            config = ExperimentConfig.for_model("google/gemma-2b-it")
            config = ExperimentConfig.for_model("google/gemma-7b-it")

            # With custom settings:
            config = ExperimentConfig.for_model(
                "google/gemma-7b-it",
                num_trials=100,
                scenarios=["ultimatum_bluff", "alliance_betrayal"],
            )

        Args:
            model_name: HuggingFace model name
            num_trials: Number of trials per scenario
            scenarios: List of scenarios (None = all 6 emergent scenarios)
            **kwargs: Additional ExperimentConfig parameters

        Returns:
            Fully configured ExperimentConfig
        """
        # Create model config (auto-configures SAE)
        model_config = ModelConfig(name=model_name)

        # Create probe config with model-appropriate layers
        probe_config = ProbeConfig.for_model(model_name)

        # Create scenario config
        scenario_kwargs = {"num_trials": num_trials}
        if scenarios is not None:
            scenario_kwargs["scenarios"] = scenarios
        scenario_config = ScenarioConfig(**scenario_kwargs)

        return cls(
            model=model_config,
            probes=probe_config,
            scenarios=scenario_config,
            **kwargs
        )

    def print_config_summary(self) -> None:
        """Print a summary of the current configuration."""
        info = self.model.get_model_info()
        print(f"Model: {self.model.name}")
        print(f"  Layers: {info.get('n_layers', '?')}, d_model: {info.get('d_model', '?')}")
        print(f"  VRAM: ~{info.get('vram_gb', '?')}GB")
        print(f"  SAE: {'enabled' if self.model.use_sae else 'disabled'}", end="")
        if self.model.use_sae:
            print(f" (layer {self.model.sae_layer})")
        else:
            print()
        print(f"Probes: layers {self.probes.layers_to_probe}")
        print(f"Scenarios: {len(self.scenarios.scenarios)} scenarios, {self.scenarios.num_trials} trials each")


# Default configurations for common use cases
QUICK_TEST = ExperimentConfig(
    model=ModelConfig(name="google/gemma-2b-it"),
    scenarios=ScenarioConfig(scenarios=["alliance_betrayal"], num_trials=1, max_rounds=3),
    causal=CausalConfig(num_samples=10),
)

FULL_EXPERIMENT = ExperimentConfig(
    model=ModelConfig(name="google/gemma-2-27b-it"),
    scenarios=ScenarioConfig(
        mode="both",
        scenarios=["alliance_betrayal", "promise_break", "capability_bluff", "info_withholding"],
        num_trials=50,
        max_rounds=5,
    ),
    causal=CausalConfig(num_samples=30),
)

FAST_ITERATION = ExperimentConfig(
    model=ModelConfig(name="google/gemma-2b-it", use_sae=False),
    scenarios=ScenarioConfig(scenarios=["alliance_betrayal"], num_trials=5, max_rounds=3),
    causal=CausalConfig(enabled=False),
)
