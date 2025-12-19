"""
Constitutional AI - Research Implementation

A research implementation of Constitutional AI methodology for educational
and experimental purposes.

This package provides tools for evaluating and training language models
according to constitutional principles (harm prevention, truthfulness,
fairness, and autonomy respect).

⚠️ Status: Experimental research code - not production-ready

Public API:
    - ConstitutionalFramework: Manage constitutional principles
    - ConstitutionalPrinciple: Define custom principles
    - setup_default_framework: Quick setup with default principles
    - Evaluation functions: evaluate_harm_potential, evaluate_truthfulness, etc.
    - Training: critique_revision_pipeline, supervised_finetune, etc.
"""

__version__ = "0.1.0-alpha"

# Core framework
# Configuration
from .config import (
    ConstitutionalTrainingConfig,
    get_default_config,
    get_harm_focused_config,
    get_lightweight_config,
    get_rlaif_config,
    get_strict_config,
)

# Phase 1: Critique-Revision
from .critique_revision import (
    ConstitutionalDataset,
    critique_revision_pipeline,
    generate_critique,
    generate_revision,
    supervised_finetune,
)

# Data utilities
from .data_utils import create_default_prompts

# Safety components
from .evaluator import ConstitutionalSafetyEvaluator
from .filter import ConstitutionalSafetyFilter
from .framework import ConstitutionalFramework, ConstitutionalPrinciple

# Model utilities
from .model_utils import GenerationConfig, generate_text, load_model

# End-to-end pipeline
from .pipeline import ConstitutionalPipeline

# Phase 2c: PPO Training
from .ppo_trainer import PPOTrainer

# Phase 2a: Preference Comparison
from .preference_comparison import (
    extract_preference,
    generate_comparison,
    generate_preference_pairs,
)

# Principle evaluators
from .principles import (
    evaluate_autonomy_respect,
    evaluate_fairness,
    evaluate_harm_potential,
    evaluate_truthfulness,
    setup_default_framework,
)

# Phase 2b: Reward Model
from .reward_model import RewardModel, RewardModelTrainer, compute_reward_loss, train_reward_model

# RLAIF Trainer
from .trainer import RLAIFTrainer

# HuggingFace API Evaluator
try:
    from .hf_api_evaluator import HuggingFaceAPIEvaluator
except ImportError:
    # Optional dependency
    HuggingFaceAPIEvaluator = None  # type: ignore[assignment, misc]


__all__ = [
    # Framework
    "ConstitutionalFramework",
    "ConstitutionalPrinciple",
    # Principles
    "evaluate_harm_potential",
    "evaluate_truthfulness",
    "evaluate_fairness",
    "evaluate_autonomy_respect",
    "setup_default_framework",
    # Phase 1
    "generate_critique",
    "generate_revision",
    "critique_revision_pipeline",
    "supervised_finetune",
    "ConstitutionalDataset",
    # Phase 2a
    "generate_comparison",
    "extract_preference",
    "generate_preference_pairs",
    # Phase 2b
    "RewardModel",
    "train_reward_model",
    "compute_reward_loss",
    "RewardModelTrainer",
    # Phase 2c
    "PPOTrainer",
    # RLAIF
    "RLAIFTrainer",
    # Pipeline
    "ConstitutionalPipeline",
    # Safety
    "ConstitutionalSafetyEvaluator",
    "ConstitutionalSafetyFilter",
    # Configuration
    "ConstitutionalTrainingConfig",
    "get_default_config",
    "get_strict_config",
    "get_rlaif_config",
    "get_lightweight_config",
    "get_harm_focused_config",
    # Utils
    "load_model",
    "generate_text",
    "GenerationConfig",
    "create_default_prompts",
    # API
    "HuggingFaceAPIEvaluator",
]
