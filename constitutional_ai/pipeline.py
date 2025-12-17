"""MODULE: pipeline.py
PURPOSE: End-to-end Constitutional AI training pipeline
KEY COMPONENTS:
- ConstitutionalPipeline: Orchestrates Phase 1 (SL) and Phase 2 (RLAIF)
- Phase 1: Critique → Revision → Supervised Fine-tuning
- Phase 2: Preference Generation → Reward Model Training → PPO Optimization
- Checkpoint management for resuming training between phases
DEPENDENCIES: torch, critique_revision, preference_comparison, reward_model, ppo_trainer
SPECIAL NOTES: Implements full Anthropic Constitutional AI methodology
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

import logging


# Module logger
logger = logging.getLogger(__name__)
import os
from pathlib import Path

from tqdm import tqdm

from .critique_revision import critique_revision_pipeline, supervised_finetune
from .framework import ConstitutionalFramework
from .ppo_trainer import PPOTrainer
from .preference_comparison import generate_preference_pairs
from .principles import setup_default_framework
from .reward_model import RewardModel, RewardModelTrainer


class ConstitutionalPipeline:
    """
    End-to-end Constitutional AI training pipeline.

    Implements the complete two-phase Constitutional AI methodology:

    Phase 1 (Supervised Learning):
    1. Generate critiques of model outputs using constitutional principles
    2. Generate revisions based on critiques
    3. Supervised fine-tune on revised outputs

    Phase 2 (RLAIF - Reinforcement Learning from AI Feedback):
    1. Generate preference pairs using constitutional principles
    2. Train reward model on preference data
    3. Optimize policy with PPO using reward model

    This pipeline orchestrates all components to provide a complete
    Constitutional AI training system.
    """

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer: Any,
        device: torch.device | None = None,
        constitutional_framework: ConstitutionalFramework | None = None,
        value_model: nn.Module | None = None,
        phase1_learning_rate: float = 5e-5,
        phase2_learning_rate: float = 1e-6,
        reward_model_learning_rate: float = 1e-5,
        temperature: float = 1.0,
        ppo_epsilon: float = 0.2,
        ppo_value_coef: float = 0.5,
        ppo_entropy_coef: float = 0.01,
        kl_penalty_coef: float = 0.02,
    ):
        """
        Initialize the Constitutional AI pipeline.

        Args:
            base_model: Base language model to train
            tokenizer: Tokenizer for the model
            device: Device for training (defaults to cuda if available)
            constitutional_framework: Framework with constitutional principles
            value_model: Optional separate value model for PPO (shares policy if None)
            phase1_learning_rate: Learning rate for Phase 1 supervised training
            phase2_learning_rate: Learning rate for Phase 2 PPO training
            reward_model_learning_rate: Learning rate for reward model training
            temperature: Sampling temperature for generation
            ppo_epsilon: PPO clipping parameter
            ppo_value_coef: PPO value loss coefficient
            ppo_entropy_coef: PPO entropy bonus coefficient
            kl_penalty_coef: KL divergence penalty coefficient
        """
        self.base_model = base_model
        self.tokenizer = tokenizer

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.base_model.to(self.device)

        # Constitutional framework
        self.constitutional_framework = (
            constitutional_framework
            if constitutional_framework is not None
            else setup_default_framework()
        )

        # Training parameters
        self.phase1_learning_rate = phase1_learning_rate
        self.phase2_learning_rate = phase2_learning_rate
        self.reward_model_learning_rate = reward_model_learning_rate
        self.temperature = temperature

        # PPO parameters
        self.ppo_epsilon = ppo_epsilon
        self.ppo_value_coef = ppo_value_coef
        self.ppo_entropy_coef = ppo_entropy_coef
        self.kl_penalty_coef = kl_penalty_coef

        # Value model
        self.value_model = value_model
        if self.value_model is not None:
            self.value_model.to(self.device)

        # Reward model (initialized in Phase 2)
        self.reward_model: RewardModel | None = None

        # Training state
        self.phase1_complete = False
        self.phase2_complete = False
        self.training_history = {"phase1": {}, "phase2": {}}

        # Statistics
        self.stats = {
            "phase1_samples_processed": 0,
            "phase1_revisions_generated": 0,
            "phase2_preference_pairs": 0,
            "phase2_ppo_steps": 0,
            "total_training_time": 0.0,
        }

    def train(
        self,
        training_prompts: List[str],
        phase1_epochs: int = 3,
        phase1_num_revisions: int = 2,
        phase1_batch_size: int = 8,
        phase2_epochs: int = 3,
        phase2_responses_per_prompt: int = 4,
        phase2_reward_model_epochs: int = 5,
        phase2_ppo_steps: int = 100,
        phase2_ppo_batch_size: int = 16,
        phase2_ppo_epochs_per_batch: int = 4,
        validation_prompts: List[str] | None = None,
        save_dir: str | None = None,
        resume_from_phase1: bool = False,
    ) -> Dict[str, Any]:
        """
        Train the model using the complete Constitutional AI pipeline.

        Args:
            training_prompts: Prompts for training
            phase1_epochs: Number of epochs for Phase 1 supervised training
            phase1_num_revisions: Number of revision iterations per prompt
            phase1_batch_size: Batch size for Phase 1 training
            phase2_epochs: Number of epochs for Phase 2 RLAIF
            phase2_responses_per_prompt: Number of responses per prompt for preferences
            phase2_reward_model_epochs: Epochs for reward model training
            phase2_ppo_steps: Number of PPO optimization steps
            phase2_ppo_batch_size: Batch size for PPO training
            phase2_ppo_epochs_per_batch: PPO epochs per batch
            validation_prompts: Optional validation prompts
            save_dir: Directory to save checkpoints
            resume_from_phase1: If True, skip Phase 1 and load from checkpoint

        Returns:
            Dictionary with training history and final statistics
        """
        logger.info("=" * 80)
        logger.info("CONSTITUTIONAL AI TRAINING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Training prompts: {len(training_prompts)}")
        logger.info(f"Constitutional principles: {len(self.constitutional_framework.principles)}")
        logger.info(f"Device: {self.device}")
        logger.info("")

        # Create save directory if needed
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Phase 1: Supervised Learning with Critique and Revision
        if not resume_from_phase1:
            logger.info("=" * 80)
            logger.info("PHASE 1: SUPERVISED LEARNING (Critique → Revision → Fine-tuning)")
            logger.info("=" * 80)

            phase1_results = self._run_phase1(
                prompts=training_prompts,
                num_epochs=phase1_epochs,
                num_revisions=phase1_num_revisions,
                batch_size=phase1_batch_size,
                validation_prompts=validation_prompts,
            )

            self.training_history["phase1"] = phase1_results
            self.phase1_complete = True

            # Save Phase 1 checkpoint
            if save_dir is not None:
                checkpoint_path = os.path.join(save_dir, "phase1_checkpoint.pt")
                self._save_phase1_checkpoint(checkpoint_path)
                logger.info(f"\nPhase 1 checkpoint saved to: {checkpoint_path}")
        else:
            logger.info("=" * 80)
            logger.info("PHASE 1: LOADING FROM CHECKPOINT")
            logger.info("=" * 80)

            if save_dir is None:
                raise ValueError("save_dir required when resume_from_phase1=True")

            checkpoint_path = os.path.join(save_dir, "phase1_checkpoint.pt")
            self._load_phase1_checkpoint(checkpoint_path)
            logger.info(f"Phase 1 checkpoint loaded from: {checkpoint_path}")

        # Phase 2: RLAIF (Reinforcement Learning from AI Feedback)
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: RLAIF (Preferences → Reward Model → PPO)")
        logger.info("=" * 80)

        phase2_results = self._run_phase2(
            prompts=training_prompts,
            num_epochs=phase2_epochs,
            responses_per_prompt=phase2_responses_per_prompt,
            reward_model_epochs=phase2_reward_model_epochs,
            ppo_steps=phase2_ppo_steps,
            ppo_batch_size=phase2_ppo_batch_size,
            ppo_epochs_per_batch=phase2_ppo_epochs_per_batch,
            validation_prompts=validation_prompts,
        )

        self.training_history["phase2"] = phase2_results
        self.phase2_complete = True

        # Save Phase 2 checkpoint
        if save_dir is not None:
            checkpoint_path = os.path.join(save_dir, "phase2_checkpoint.pt")
            self._save_phase2_checkpoint(checkpoint_path)
            logger.info(f"\nPhase 2 checkpoint saved to: {checkpoint_path}")

        # Final evaluation
        if validation_prompts is not None:
            logger.info("\n" + "=" * 80)
            logger.info("FINAL EVALUATION")
            logger.info("=" * 80)

            final_eval = self.evaluate_constitutional_compliance(
                validation_prompts, self.base_model
            )

            logger.info(f"Final Constitutional Compliance Score: {final_eval['avg_score']:.4f}")
            logger.info(f"Violation Rate: {final_eval['violation_rate']:.2%}")

        # Compile results
        results = {
            "phase1_complete": self.phase1_complete,
            "phase2_complete": self.phase2_complete,
            "training_history": self.training_history,
            "statistics": self.stats,
        }

        if validation_prompts is not None:
            results["final_evaluation"] = final_eval

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)

        return results

    def _run_phase1(
        self,
        prompts: List[str],
        num_epochs: int,
        num_revisions: int,
        batch_size: int,
        validation_prompts: List[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Run Phase 1: Critique-Revision-Supervised Learning.

        Args:
            prompts: Training prompts
            num_epochs: Number of training epochs
            num_revisions: Number of revision iterations
            batch_size: Training batch size
            validation_prompts: Optional validation prompts

        Returns:
            Phase 1 training results
        """
        logger.info("\nStep 1: Generating critiques and revisions...")
        logger.info(f"Processing {len(prompts)} prompts with {num_revisions} revisions each")

        # Generate training data with critiques and revisions
        training_data = critique_revision_pipeline(
            prompts=prompts,
            model=self.base_model,
            tokenizer=self.tokenizer,
            framework=self.constitutional_framework,
            device=self.device,
            num_revisions=num_revisions,
        )

        self.stats["phase1_samples_processed"] = len(training_data)
        self.stats["phase1_revisions_generated"] = len(training_data) * num_revisions

        logger.info(f"Generated {len(training_data)} training examples")
        logger.info(f"Total revisions: {self.stats['phase1_revisions_generated']}")

        # Supervised fine-tuning on revised outputs
        logger.info("\nStep 2: Supervised fine-tuning on revised outputs...")

        sft_results = supervised_finetune(
            model=self.base_model,
            tokenizer=self.tokenizer,
            training_data=training_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=self.phase1_learning_rate,
            device=self.device,
        )

        # Validation
        validation_results = {}
        if validation_prompts is not None:
            logger.info("\nStep 3: Validating Phase 1 model...")
            validation_results = self.evaluate_constitutional_compliance(
                validation_prompts, self.base_model
            )
            logger.info(f"Validation Score: {validation_results['avg_score']:.4f}")
            logger.info(f"Violation Rate: {validation_results['violation_rate']:.2%}")

        # Mark Phase 1 as complete
        self.phase1_complete = True

        return {
            "training_data_size": len(training_data),
            "sft_results": sft_results,
            "validation_results": validation_results,
        }

    def _run_phase2(
        self,
        prompts: List[str],
        num_epochs: int,
        responses_per_prompt: int,
        reward_model_epochs: int,
        ppo_steps: int,
        ppo_batch_size: int,
        ppo_epochs_per_batch: int,
        validation_prompts: List[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Run Phase 2: RLAIF with Preference Learning and PPO.

        Args:
            prompts: Training prompts
            num_epochs: Number of epochs
            responses_per_prompt: Responses per prompt for preference generation
            reward_model_epochs: Epochs for reward model training
            ppo_steps: Number of PPO optimization steps
            ppo_batch_size: PPO batch size
            ppo_epochs_per_batch: PPO epochs per batch
            validation_prompts: Optional validation prompts

        Returns:
            Phase 2 training results
        """
        logger.info("\nStep 1: Generating preference pairs...")
        logger.info(f"Processing {len(prompts)} prompts with {responses_per_prompt} responses each")

        # Generate preference pairs
        preference_data = generate_preference_pairs(
            prompts=prompts,
            model=self.base_model,
            tokenizer=self.tokenizer,
            framework=self.constitutional_framework,
            device=self.device,
            responses_per_prompt=responses_per_prompt,
        )

        self.stats["phase2_preference_pairs"] = len(preference_data)
        logger.info(f"Generated {len(preference_data)} preference pairs")

        # Initialize reward model
        logger.info("\nStep 2: Training reward model...")

        # Get model hidden size
        if hasattr(self.base_model, "config"):
            # GPT-2 uses 'n_embd', most others use 'hidden_size'
            if (
                hasattr(self.base_model.config, "n_embd")
                and self.base_model.config.n_embd is not None
            ):
                hidden_size = self.base_model.config.n_embd  # GPT-2 style
            elif (
                hasattr(self.base_model.config, "hidden_size")
                and self.base_model.config.hidden_size is not None
            ):
                hidden_size = self.base_model.config.hidden_size  # Modern models
            else:
                hidden_size = 768  # Fallback
        else:
            # No config available
            hidden_size = 768

        self.reward_model = RewardModel(base_model=self.base_model, hidden_size=hidden_size)
        self.reward_model.to(self.device)

        # Train reward model
        reward_trainer = RewardModelTrainer(
            reward_model=self.reward_model,
            tokenizer=self.tokenizer,
            learning_rate=self.reward_model_learning_rate,
            device=self.device,
            batch_size=ppo_batch_size,
        )

        reward_results = reward_trainer.train(
            training_data=preference_data, num_epochs=reward_model_epochs
        )

        logger.info("Reward model training complete")
        logger.info(f"Final loss: {reward_results['final_loss']:.4f}")
        logger.info(f"Final accuracy: {reward_results['final_accuracy']:.2%}")

        # PPO training
        logger.info("\nStep 3: PPO optimization with reward model...")

        ppo_trainer = PPOTrainer(
            policy_model=self.base_model,
            value_model=self.value_model,
            reward_model=self.reward_model,
            tokenizer=self.tokenizer,
            device=self.device,
            learning_rate=self.phase2_learning_rate,
            clip_epsilon=self.ppo_epsilon,
            value_loss_coef=self.ppo_value_coef,
            kl_penalty=self.kl_penalty_coef,
        )

        ppo_results = ppo_trainer.train(
            prompts=prompts,
            num_steps=ppo_steps,
            batch_size=ppo_batch_size,
            num_epochs_per_batch=ppo_epochs_per_batch,
            max_length=150,
            temperature=self.temperature,
        )

        self.stats["phase2_ppo_steps"] = ppo_steps

        logger.info("PPO training complete")
        logger.info(f"Final reward: {ppo_results['final_avg_reward']:.4f}")
        logger.info(f"Final KL divergence: {ppo_results['final_kl_divergence']:.4f}")

        # Validation
        validation_results = {}
        if validation_prompts is not None:
            logger.info("\nStep 4: Validating Phase 2 model...")
            validation_results = self.evaluate_constitutional_compliance(
                validation_prompts, self.base_model
            )
            logger.info(f"Validation Score: {validation_results['avg_score']:.4f}")
            logger.info(f"Violation Rate: {validation_results['violation_rate']:.2%}")

        return {
            "preference_pairs": len(preference_data),
            "reward_model_results": reward_results,
            "ppo_results": ppo_results,
            "validation_results": validation_results,
        }

    def evaluate_constitutional_compliance(
        self, test_prompts: List[str], model: nn.Module | None = None
    ) -> Dict[str, Any]:
        """
        Evaluate model's constitutional compliance on test prompts.

        Args:
            test_prompts: Prompts for evaluation
            model: Model to evaluate (uses base_model if None)

        Returns:
            Evaluation results with scores and violation statistics
        """
        if model is None:
            model = self.base_model

        model.eval()

        from .evaluator import ConstitutionalSafetyEvaluator
        from .model_utils import GenerationConfig, generate_text

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.constitutional_framework, critique_model=model, use_self_critique=True
        )

        scores = []
        violations = []

        logger.info(f"Evaluating {len(test_prompts)} test prompts...")

        with torch.no_grad():
            for prompt in tqdm(test_prompts, desc="Evaluation"):
                # Generate response
                config = GenerationConfig(
                    max_length=150, temperature=self.temperature, do_sample=True
                )

                response = generate_text(model, self.tokenizer, prompt, config, device=self.device)

                # Evaluate
                evaluation = evaluator.evaluate(response)

                # Extract score and violations
                direct_eval = evaluation.get("direct_evaluation", {})
                score = direct_eval.get("weighted_score", 0.0)
                flagged = evaluation.get("flagged", False)

                scores.append(score)
                violations.append(1 if flagged else 0)

        return {
            "avg_score": float(sum(scores) / len(scores)) if scores else 0.0,
            "violation_rate": float(sum(violations) / len(violations)) if violations else 0.0,
            "num_violations": sum(violations),
            "total_evaluated": len(test_prompts),
            "scores": scores,
        }

    def _save_phase1_checkpoint(self, path: str) -> None:
        """Save Phase 1 checkpoint."""
        checkpoint = {
            "model_state_dict": self.base_model.state_dict(),
            "phase1_complete": self.phase1_complete,
            "training_history": self.training_history,
            "stats": self.stats,
        }
        torch.save(checkpoint, path)

    def _load_phase1_checkpoint(self, path: str) -> None:
        """Load Phase 1 checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.base_model.load_state_dict(checkpoint["model_state_dict"])
        self.phase1_complete = checkpoint["phase1_complete"]
        self.training_history = checkpoint["training_history"]
        self.stats = checkpoint["stats"]

    def _save_phase2_checkpoint(self, path: str) -> None:
        """Save Phase 2 checkpoint."""
        checkpoint = {
            "model_state_dict": self.base_model.state_dict(),
            "reward_model_state_dict": (
                self.reward_model.state_dict() if self.reward_model else None
            ),
            "value_model_state_dict": self.value_model.state_dict() if self.value_model else None,
            "phase1_complete": self.phase1_complete,
            "phase2_complete": self.phase2_complete,
            "training_history": self.training_history,
            "stats": self.stats,
        }
        torch.save(checkpoint, path)

    def _load_phase2_checkpoint(self, path: str) -> None:
        """Load Phase 2 checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.base_model.load_state_dict(checkpoint["model_state_dict"])

        if checkpoint["reward_model_state_dict"] is not None and self.reward_model is not None:
            self.reward_model.load_state_dict(checkpoint["reward_model_state_dict"])

        if checkpoint["value_model_state_dict"] is not None and self.value_model is not None:
            self.value_model.load_state_dict(checkpoint["value_model_state_dict"])

        self.phase1_complete = checkpoint["phase1_complete"]
        self.phase2_complete = checkpoint["phase2_complete"]
        self.training_history = checkpoint["training_history"]
        self.stats = checkpoint["stats"]

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "pipeline_stats": self.stats,
            "phase1_complete": self.phase1_complete,
            "phase2_complete": self.phase2_complete,
            "training_history": self.training_history,
        }
