"""MODULE: config.py
PURPOSE: Configuration for Constitutional AI training
KEY COMPONENTS:
- ConstitutionalTrainingConfig: Training configuration for CAI
- Principle configurations and weights
- RLAIF-specific settings
DEPENDENCIES: dataclass, typing
SPECIAL NOTES: Provides flexible configuration for constitutional AI training
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ConstitutionalTrainingConfig:
    """
    Configuration for training models with Constitutional AI.

    This config extends standard training parameters with constitutional
    AI specific settings for principle-based evaluation and RLAIF training.
    """

    # ========== Standard Training Parameters ==========
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    batch_size: int = 8
    eval_interval: int = 1000
    save_interval: int = 5000
    log_dir: str = "logs/constitutional_training"

    # ========== Constitutional AI Parameters ==========

    # Whether to use constitutional AI evaluation
    use_constitutional_ai: bool = True

    # Whether to use RLAIF (Reinforcement Learning from AI Feedback)
    use_rlaif: bool = False

    # Weight for constitutional loss (0-1, higher = more emphasis on constitutional compliance)
    constitutional_weight: float = 0.5

    # Number of response candidates per prompt (for RLAIF)
    num_responses_per_prompt: int = 5

    # Number of samples for constitutional evaluation
    eval_samples: int = 100

    # ========== Constitutional Principles Configuration ==========

    # Which principles to enable (if None, uses all)
    enabled_principles: List[str] | None = None

    # Custom principle weights (principle_name: weight)
    principle_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "harm_prevention": 2.0,  # Higher weight for harm prevention
            "truthfulness": 1.5,
            "fairness": 1.0,
            "autonomy_respect": 1.0,
        }
    )

    # ========== Model Configuration ==========

    # Model name or path
    model_name: str = "gpt2"

    # Model parameters
    vocab_size: int | None = None
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_seq_length: int = 512

    # Whether to use a separate critique model
    use_critique_model: bool = False
    critique_model_name: str | None = None

    # ========== Data Configuration ==========

    # Training data path
    train_data_path: str | None = None

    # Validation data path
    val_data_path: str | None = None

    # Training prompts for RLAIF (if not using data files)
    training_prompts: List[str] | None = None

    # Validation prompts
    validation_prompts: List[str] | None = None

    # ========== Evaluation Configuration ==========

    # Sensitivity level for safety evaluation ('low', 'medium', 'high')
    safety_sensitivity: str = "medium"

    # Whether to use self-critique in evaluation
    use_self_critique: bool = False

    # Whether to track evaluation history
    track_evaluation_history: bool = True

    # ========== Advanced Settings ==========

    # Temperature for response generation
    temperature: float = 1.0

    # Whether to apply constitutional filtering to outputs
    apply_constitutional_filtering: bool = True

    # Strict mode for filtering (more aggressive)
    strict_filtering: bool = False

    # Random seed for reproducibility
    seed: int = 42

    # Device ('cpu', 'cuda', 'mps', or None for auto)
    device: str | None = None

    def __post_init__(self):
        """Validate and process configuration."""
        # Validate constitutional weight
        if not 0.0 <= self.constitutional_weight <= 1.0:
            raise ValueError(
                f"constitutional_weight must be between 0 and 1, got {self.constitutional_weight}"
            )

        # Validate safety sensitivity
        valid_sensitivities = ["low", "medium", "high"]
        if self.safety_sensitivity not in valid_sensitivities:
            raise ValueError(
                f"safety_sensitivity must be one of {valid_sensitivities}, got {self.safety_sensitivity}"
            )

        # Set default enabled principles if not specified
        if self.enabled_principles is None:
            self.enabled_principles = list(self.principle_weights.keys())

        # Ensure principle weights exist for all enabled principles
        for principle in self.enabled_principles:
            if principle not in self.principle_weights:
                self.principle_weights[principle] = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "eval_interval": self.eval_interval,
            "save_interval": self.save_interval,
            "log_dir": self.log_dir,
            "use_constitutional_ai": self.use_constitutional_ai,
            "use_rlaif": self.use_rlaif,
            "constitutional_weight": self.constitutional_weight,
            "num_responses_per_prompt": self.num_responses_per_prompt,
            "eval_samples": self.eval_samples,
            "enabled_principles": self.enabled_principles,
            "principle_weights": self.principle_weights,
            "model_name": self.model_name,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "max_seq_length": self.max_seq_length,
            "use_critique_model": self.use_critique_model,
            "critique_model_name": self.critique_model_name,
            "train_data_path": self.train_data_path,
            "val_data_path": self.val_data_path,
            "training_prompts": self.training_prompts,
            "validation_prompts": self.validation_prompts,
            "safety_sensitivity": self.safety_sensitivity,
            "use_self_critique": self.use_self_critique,
            "track_evaluation_history": self.track_evaluation_history,
            "temperature": self.temperature,
            "apply_constitutional_filtering": self.apply_constitutional_filtering,
            "strict_filtering": self.strict_filtering,
            "seed": self.seed,
            "device": self.device,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ConstitutionalTrainingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)


# ========== Predefined Configurations ==========


def get_default_config() -> ConstitutionalTrainingConfig:
    """Get default constitutional training configuration."""
    return ConstitutionalTrainingConfig()


def get_strict_config() -> ConstitutionalTrainingConfig:
    """Get strict constitutional training configuration (high safety emphasis)."""
    return ConstitutionalTrainingConfig(
        constitutional_weight=0.8,
        principle_weights={
            "harm_prevention": 3.0,
            "truthfulness": 2.0,
            "fairness": 1.5,
            "autonomy_respect": 1.5,
        },
        safety_sensitivity="high",
        strict_filtering=True,
        use_self_critique=True,
    )


def get_rlaif_config() -> ConstitutionalTrainingConfig:
    """Get RLAIF-focused configuration."""
    return ConstitutionalTrainingConfig(
        use_rlaif=True,
        use_critique_model=False,  # Use same model for critique
        num_responses_per_prompt=10,  # More candidates for better selection
        constitutional_weight=0.7,
        num_epochs=5,
        eval_samples=200,
    )


def get_lightweight_config() -> ConstitutionalTrainingConfig:
    """Get lightweight configuration for testing/debugging."""
    return ConstitutionalTrainingConfig(
        num_epochs=1,
        batch_size=4,
        eval_interval=500,
        save_interval=2000,
        eval_samples=20,
        warmup_steps=100,
        constitutional_weight=0.3,
        use_rlaif=False,
    )


def get_harm_focused_config() -> ConstitutionalTrainingConfig:
    """Get configuration focused primarily on harm prevention."""
    return ConstitutionalTrainingConfig(
        enabled_principles=["harm_prevention", "truthfulness"],
        principle_weights={
            "harm_prevention": 5.0,
            "truthfulness": 1.5,
        },
        constitutional_weight=0.9,
        safety_sensitivity="high",
        strict_filtering=True,
    )
