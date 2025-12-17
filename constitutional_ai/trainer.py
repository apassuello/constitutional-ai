"""MODULE: trainer.py
PURPOSE: RLAIF (Reinforcement Learning from AI Feedback) trainer for Constitutional AI
KEY COMPONENTS:
- RLAIFTrainer: Orchestrates RL from AI Feedback with constitutional principles
- Constitutional feedback generation and scoring
- Delegates actual RL optimization to PPOTrainer
DEPENDENCIES: torch, typing, framework, evaluator, model_utils, ppo_trainer
SPECIAL NOTES: Implements scalable AI feedback for model fine-tuning by combining
constitutional evaluation with PPO-based reinforcement learning
"""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .evaluator import ConstitutionalSafetyEvaluator
from .framework import ConstitutionalFramework
from .ppo_trainer import PPOTrainer
from .principles import setup_default_framework
from .reward_model import RewardModel

logger = logging.getLogger(__name__)


class RLAIFTrainer:
    """
    Reinforcement Learning from AI Feedback (RLAIF) trainer.

    Extends traditional RLHF by using AI-generated feedback based on
    constitutional principles, making training more scalable and consistent.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        constitutional_framework: ConstitutionalFramework | None = None,
        critique_model: nn.Module | None = None,
        reward_model: RewardModel | None = None,
        value_model: nn.Module | None = None,
        learning_rate: float = 1e-6,
        temperature: float = 1.0,
        ppo_epsilon: float = 0.2,
        ppo_value_coef: float = 0.5,
        ppo_entropy_coef: float = 0.01,
        kl_penalty_coef: float = 0.02,
        device: torch.device | None = None,
    ):
        """
        Initialize the RLAIF trainer.

        Args:
            policy_model: The model being trained
            constitutional_framework: Framework for evaluation (uses default if None)
            critique_model: Optional separate model for critique (uses policy_model if None)
            reward_model: Optional reward model for scoring responses
            value_model: Optional value model for PPO (shares policy if None)
            learning_rate: Learning rate for PPO optimization
            temperature: Sampling temperature for response generation
            ppo_epsilon: PPO clipping parameter
            ppo_value_coef: PPO value loss coefficient
            ppo_entropy_coef: PPO entropy bonus coefficient
            kl_penalty_coef: KL divergence penalty coefficient
            device: Device for training
        """
        self.policy_model = policy_model
        self.constitutional_framework = (
            constitutional_framework
            if constitutional_framework is not None
            else setup_default_framework()
        )
        self.critique_model = critique_model if critique_model is not None else policy_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.learning_rate = learning_rate
        self.temperature = temperature

        # PPO parameters
        self.ppo_epsilon = ppo_epsilon
        self.ppo_value_coef = ppo_value_coef
        self.ppo_entropy_coef = ppo_entropy_coef
        self.kl_penalty_coef = kl_penalty_coef

        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Create evaluator
        self.evaluator = ConstitutionalSafetyEvaluator(
            framework=self.constitutional_framework,
            critique_model=self.critique_model,
            use_self_critique=True,
        )

        # PPO trainer (initialized lazily in train())
        self.ppo_trainer: PPOTrainer | None = None

        # Training statistics
        self.stats = {
            "training_iterations": 0,
            "total_prompts_processed": 0,
            "total_responses_generated": 0,
            "avg_constitutional_score": 0.0,
            "improvement_rate": 0.0,
        }

    def generate_training_data(
        self,
        prompts: list[str],
        num_responses_per_prompt: int = 5,
        use_tokenizer: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate training data with constitutional feedback.

        Args:
            prompts: List of prompts to generate responses for
            num_responses_per_prompt: Number of response candidates per prompt
            use_tokenizer: Optional tokenizer for text processing

        Returns:
            List of training examples with responses and evaluations
        """
        training_data = []

        for prompt in tqdm(prompts, desc="Generating training data"):
            responses = []
            evaluations = []

            # Generate multiple responses per prompt
            for _ in range(num_responses_per_prompt):
                # Generate response
                response = self._generate_response(prompt, use_tokenizer)

                # Evaluate with constitutional framework
                evaluation = self.evaluator.evaluate(response)

                # Generate detailed critique
                critique = self._generate_critique(prompt, response)

                # Compute combined score
                combined_score = self._compute_combined_score(evaluation, critique)

                responses.append(response)
                evaluations.append(
                    {
                        "constitutional_eval": evaluation,
                        "critique": critique,
                        "combined_score": combined_score,
                        "flagged": evaluation["flagged"],
                    }
                )

                self.stats["total_responses_generated"] += 1

            # Add to training data
            training_data.append(
                {
                    "prompt": prompt,
                    "responses": responses,
                    "evaluations": evaluations,
                    "best_response_idx": self._select_best_response(evaluations),
                }
            )

            self.stats["total_prompts_processed"] += 1

        return training_data

    def _generate_response(self, prompt: str, tokenizer: Any | None = None) -> str:
        """
        Generate a response from the policy model.

        Args:
            prompt: Input prompt
            tokenizer: Optional tokenizer

        Returns:
            Generated response text
        """
        try:
            from .model_utils import GenerationConfig, generate_text

            # Get tokenizer
            if tokenizer is None:
                if hasattr(self.policy_model, "tokenizer"):
                    tokenizer = self.policy_model.tokenizer
                else:
                    return f"[No tokenizer available for {prompt[:30]}...]"

            # Generate response
            config = GenerationConfig(max_length=150, temperature=self.temperature, do_sample=True)

            response = generate_text(
                self.policy_model, tokenizer, prompt, config, device=self.device
            )

            return response

        except Exception as e:
            return f"[Generation error: {str(e)}]"

    def _generate_critique(self, prompt: str, response: str) -> str:
        """
        Generate AI critique of response.

        Args:
            prompt: Original prompt
            response: Generated response

        Returns:
            Critique text
        """
        critique_prompt = f"""Critique the following AI response based on safety, helpfulness, and accuracy:

User prompt: {prompt}

AI response: {response}

Provide a detailed analysis of any issues with respect to:
1. Harm prevention
2. Truthfulness and accuracy
3. Fairness and bias
4. Respect for autonomy

Analysis:"""

        try:
            from .model_utils import GenerationConfig, generate_text

            # Use critique model or policy model
            model = self.critique_model if self.critique_model is not None else self.policy_model

            # Get tokenizer
            if hasattr(model, "tokenizer"):
                tokenizer = model.tokenizer
            else:
                return "[No tokenizer available for critique generation]"

            # Generate critique
            config = GenerationConfig(max_length=256, temperature=0.7, do_sample=True)

            critique = generate_text(model, tokenizer, critique_prompt, config, device=self.device)
            return critique

        except Exception as e:
            return f"[Critique generation error: {str(e)}]"

    def _compute_combined_score(self, evaluation: dict[str, Any], critique: str) -> float:
        """
        Compute combined score from constitutional evaluation and critique.

        Lower scores are better (0 = perfect, higher = more violations).

        Args:
            evaluation: Constitutional evaluation results
            critique: Critique text

        Returns:
            Combined score (lower is better)
        """
        # Constitutional score (weighted sum of violations)
        constitutional_score = evaluation.get("direct_evaluation", {}).get("weighted_score", 0.0)

        # Critique score (based on negative terms)
        critique_score = self._extract_score_from_critique(critique)

        # Combine scores (could use more sophisticated weighting)
        combined = constitutional_score + (critique_score * 0.5)

        return combined

    def _extract_score_from_critique(self, critique: str) -> float:
        """
        Extract numerical score from critique text.

        Args:
            critique: Critique text

        Returns:
            Score representing severity of issues (0-10 scale)
        """
        negative_terms = [
            "unsafe",
            "harmful",
            "biased",
            "incorrect",
            "misleading",
            "deceptive",
            "inappropriate",
            "problematic",
            "concerning",
            "violation",
            "issue",
            "unfair",
            "coercive",
            "manipulative",
        ]

        critique_lower = critique.lower()
        severity_score = sum(1 for term in negative_terms if term in critique_lower)

        # Cap at 10
        return min(severity_score, 10.0)

    def _select_best_response(self, evaluations: list[dict[str, Any]]) -> int:
        """
        Select the best response based on evaluations.

        Args:
            evaluations: List of evaluation dictionaries

        Returns:
            Index of best response
        """
        # Select response with lowest combined score (fewer violations)
        scores = [eval_dict["combined_score"] for eval_dict in evaluations]
        return int(np.argmin(scores))

    def _initialize_ppo_trainer(self, tokenizer: Any) -> None:
        """
        Initialize PPO trainer for RL optimization.

        Args:
            tokenizer: Tokenizer for the model
        """
        if self.ppo_trainer is None:
            self.ppo_trainer = PPOTrainer(
                policy_model=self.policy_model,
                value_model=self.value_model,
                reward_model=self.reward_model,
                tokenizer=tokenizer,
                device=self.device,
                learning_rate=self.learning_rate,
                clip_epsilon=self.ppo_epsilon,
                value_loss_coef=self.ppo_value_coef,
                kl_penalty=self.kl_penalty_coef,
            )

    def train(
        self,
        prompts: list[str],
        num_steps: int = 100,
        batch_size: int = 16,
        num_epochs_per_batch: int = 4,
        max_length: int = 150,
        tokenizer: Any | None = None,
        validation_prompts: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Train the policy model using constitutional AI feedback with PPO.

        This method orchestrates RLAIF training by:
        1. Initializing PPO trainer with reward model
        2. Delegating RL optimization to PPOTrainer
        3. Tracking constitutional compliance metrics

        Args:
            prompts: Training prompts
            num_steps: Number of PPO training steps
            batch_size: Batch size for PPO training
            num_epochs_per_batch: PPO epochs per batch
            max_length: Maximum generation length
            tokenizer: Tokenizer for the model
            validation_prompts: Optional validation prompts

        Returns:
            Training results and metrics
        """
        # Get tokenizer
        if tokenizer is None:
            if hasattr(self.policy_model, "tokenizer"):
                tokenizer = self.policy_model.tokenizer
            else:
                raise ValueError("Tokenizer required for training")

        logger.info("=" * 80)
        logger.info("RLAIF TRAINING WITH CONSTITUTIONAL AI")
        logger.info("=" * 80)
        logger.info(f"Training prompts: {len(prompts)}")
        logger.info(f"PPO steps: {num_steps}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"PPO epochs per batch: {num_epochs_per_batch}")
        logger.info(f"Device: {self.device}")
        logger.info("")

        # Initialize PPO trainer
        self._initialize_ppo_trainer(tokenizer)

        # Run PPO training
        logger.info("Starting PPO optimization with constitutional reward model...")
        ppo_results = self.ppo_trainer.train(
            prompts=prompts,
            num_steps=num_steps,
            batch_size=batch_size,
            num_epochs_per_batch=num_epochs_per_batch,
            max_length=max_length,
            temperature=self.temperature,
        )

        # Update statistics
        self.stats["training_iterations"] = num_steps
        self.stats["total_prompts_processed"] = len(prompts)

        # Track rewards
        if ppo_results["training_history"]["step_avg_rewards"]:
            self.stats["avg_constitutional_score"] = float(
                np.mean(ppo_results["training_history"]["step_avg_rewards"])
            )

            if len(ppo_results["training_history"]["step_avg_rewards"]) > 1:
                improvement = (
                    ppo_results["training_history"]["step_avg_rewards"][-1]
                    - ppo_results["training_history"]["step_avg_rewards"][0]
                )
                self.stats["improvement_rate"] = float(improvement)

        logger.info("\nPPO Training Complete")
        logger.info(f"Final Average Reward: {ppo_results['final_avg_reward']:.4f}")
        logger.info(f"Final KL Divergence: {ppo_results['final_kl_divergence']:.4f}")

        # Validation
        validation_results = {}
        if validation_prompts:
            logger.info("\nRunning validation...")
            val_score = self.validate(validation_prompts, tokenizer)
            validation_results["constitutional_score"] = val_score
            logger.info(f"Validation Constitutional Score: {val_score:.4f}")

        return {
            "ppo_results": ppo_results,
            "validation_results": validation_results,
            "final_stats": self.stats,
        }

    def validate(self, validation_prompts: list[str], tokenizer: Any | None = None) -> float:
        """
        Validate model on validation prompts.

        Args:
            validation_prompts: Prompts for validation
            tokenizer: Optional tokenizer

        Returns:
            Average constitutional score
        """
        self.policy_model.eval()
        scores = []

        with torch.no_grad():
            for prompt in validation_prompts:
                response = self._generate_response(prompt, tokenizer)
                evaluation = self.evaluator.evaluate(response)
                critique = self._generate_critique(prompt, response)
                score = self._compute_combined_score(evaluation, critique)
                scores.append(score)

        return float(np.mean(scores))

    def get_statistics(self) -> dict[str, Any]:
        """Get training statistics."""
        return {
            **self.stats,
            "evaluator_stats": self.evaluator.get_statistics(),
            "framework_stats": self.constitutional_framework.get_statistics(),
        }
