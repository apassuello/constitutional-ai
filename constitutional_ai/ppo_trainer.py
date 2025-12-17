"""MODULE: ppo_trainer.py
PURPOSE: Proximal Policy Optimization (PPO) for Constitutional AI
KEY COMPONENTS:
- PPOTrainer: Main PPO training class
- compute_gae: Generalized Advantage Estimation
- compute_kl_divergence: KL penalty computation
- Clipped PPO objective for stable policy updates
DEPENDENCIES: torch, transformers, typing
SPECIAL NOTES: Implements full PPO algorithm for RLAIF Phase 2c
"""

import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
import copy

import numpy as np
from tqdm import tqdm

from .model_utils import GenerationConfig, generate_text


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for Constitutional AI.

    Implements the PPO algorithm with:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - KL divergence penalty
    - Value function training
    """

    def __init__(
        self,
        policy_model: nn.Module,
        value_model: nn.Module,
        reward_model: nn.Module,
        tokenizer,
        device: torch.device,
        learning_rate: float = 1e-5,
        clip_epsilon: float = 0.2,
        kl_penalty: float = 0.1,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 1.0,
    ):
        """
        Initialize PPO trainer.

        Args:
            policy_model: Model being trained (generates responses)
            value_model: Value function estimator
            reward_model: Reward model for feedback (from Component 2)
            tokenizer: Tokenizer for text processing
            device: Computation device
            learning_rate: Learning rate for optimization
            clip_epsilon: PPO clipping parameter (typically 0.1-0.3)
            kl_penalty: KL divergence penalty coefficient
            gamma: Discount factor for rewards
            gae_lambda: GAE lambda parameter for advantage estimation
            value_loss_coef: Coefficient for value loss
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.policy_model = policy_model.to(device)
        self.value_model = value_model.to(device) if value_model is not None else None
        self.reward_model = reward_model.to(device)

        # Create frozen reference model for KL penalty
        self.reference_model = copy.deepcopy(policy_model).to(device)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.tokenizer = tokenizer
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.kl_penalty = kl_penalty
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
        self.value_optimizer = (
            torch.optim.AdamW(value_model.parameters(), lr=learning_rate)
            if value_model is not None
            else None
        )

        # Training statistics
        self.stats = {
            "total_steps": 0,
            "policy_losses": [],
            "value_losses": [],
            "kl_divergences": [],
            "mean_rewards": [],
            "mean_advantages": [],
        }

    def compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        GAE provides a bias-variance tradeoff for advantage estimation.
        It computes advantages by looking ahead and using TD(λ).

        Args:
            rewards: Rewards at each timestep [batch_size, seq_len]
            values: Value estimates [batch_size, seq_len]
            dones: Episode termination flags [batch_size, seq_len]

        Returns:
            Tuple of (advantages, returns)
            - advantages: GAE advantages [batch_size, seq_len]
            - returns: Discounted returns for value training [batch_size, seq_len]
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Last GAE advantage
        last_gae = 0

        # Compute advantages backwards through time
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[:, t + 1]

            # TD residual: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]

            # GAE: A_t = δ_t + γ * λ * A_{t+1}
            advantages[:, t] = delta + self.gamma * self.gae_lambda * (1 - dones[:, t]) * last_gae
            last_gae = advantages[:, t]

        # Compute returns: R_t = A_t + V(s_t)
        returns = advantages + values

        return advantages, returns

    def compute_kl_divergence(
        self, current_logprobs: torch.Tensor, reference_logprobs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between current and reference policy.

        KL penalty prevents the policy from deviating too far from
        the reference model, avoiding catastrophic forgetting.

        Args:
            current_logprobs: Log probabilities from current policy [batch_size, seq_len]
            reference_logprobs: Log probabilities from reference policy [batch_size, seq_len]

        Returns:
            kl_div: Mean KL divergence (scalar)
        """
        # KL(current || reference) = E[log(current) - log(reference)]
        kl_div = (current_logprobs - reference_logprobs).mean()
        return kl_div

    def compute_ppo_loss(
        self, old_logprobs: torch.Tensor, new_logprobs: torch.Tensor, advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute clipped PPO objective.

        PPO clips the probability ratio to prevent too-large policy updates,
        providing stability compared to vanilla policy gradient.

        Args:
            old_logprobs: Log probs from old policy [batch_size, seq_len]
            new_logprobs: Log probs from current policy [batch_size, seq_len]
            advantages: Advantage estimates [batch_size, seq_len]

        Returns:
            loss: PPO clipped loss (scalar)
        """
        # Probability ratio: π_new / π_old
        ratio = torch.exp(new_logprobs - old_logprobs)

        # Unclipped objective: L = ratio * A
        surr1 = ratio * advantages

        # Clipped objective: L_clip = clip(ratio, 1-ε, 1+ε) * A
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

        # Take minimum (pessimistic bound)
        # This prevents too-large policy updates
        policy_loss = -torch.min(surr1, surr2).mean()

        return policy_loss

    def generate_responses(
        self, prompts: List[str], max_length: int = 150, temperature: float = 1.0
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Generate responses from current policy and compute log probabilities.

        Args:
            prompts: List of prompts
            max_length: Maximum response length
            temperature: Sampling temperature

        Returns:
            Tuple of (responses, log_probs)
            - responses: Generated text responses
            - log_probs: Log probabilities of generated tokens [batch_size, seq_len]
        """
        self.policy_model.eval()

        responses = []
        all_logprobs = []

        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                prompt_len = inputs["input_ids"].shape[1]

                # Generate response
                config = GenerationConfig(
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                response = generate_text(
                    self.policy_model, self.tokenizer, prompt, config, self.device
                )
                responses.append(response)

                # Compute log probabilities for the generated response
                full_text = prompt + response
                full_inputs = self.tokenizer(
                    full_text, return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}

                # Get model outputs
                outputs = self.policy_model(**full_inputs)
                logits = outputs.logits

                # Compute log probabilities for generated tokens
                # Shift by 1 to align logits with targets
                shift_logits = logits[:, prompt_len - 1 : -1, :]
                shift_labels = full_inputs["input_ids"][:, prompt_len:]

                # Compute log softmax
                log_probs = F.log_softmax(shift_logits, dim=-1)

                # Gather log probs for actual tokens
                token_log_probs = torch.gather(
                    log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                all_logprobs.append(token_log_probs)

        # Pad log probs to same length
        max_len = max(lp.shape[1] for lp in all_logprobs)
        padded_logprobs = []
        for lp in all_logprobs:
            pad_len = max_len - lp.shape[1]
            if pad_len > 0:
                padding = torch.zeros(lp.shape[0], pad_len, device=self.device)
                lp = torch.cat([lp, padding], dim=1)
            padded_logprobs.append(lp)

        log_probs_tensor = torch.cat(padded_logprobs, dim=0)

        return responses, log_probs_tensor

    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """
        Compute rewards using the reward model.

        Args:
            prompts: List of prompts
            responses: List of generated responses

        Returns:
            rewards: Reward scores [batch_size, seq_len]
        """
        self.reward_model.eval()

        rewards_list = []

        with torch.no_grad():
            for prompt, response in zip(prompts, responses, strict=False):
                # Tokenize prompt and response separately then concatenate
                # This ensures exact boundary alignment
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
                response_tokens = self.tokenizer(
                    response, return_tensors="pt", add_special_tokens=False
                )

                prompt_len = prompt_tokens["input_ids"].shape[1]
                response_len = response_tokens["input_ids"].shape[1]

                # Concatenate token IDs
                input_ids = torch.cat(
                    [prompt_tokens["input_ids"], response_tokens["input_ids"]], dim=1
                )
                attention_mask = torch.ones_like(input_ids)

                # Truncate if needed
                if input_ids.shape[1] > 512:
                    input_ids = input_ids[:, :512]
                    attention_mask = attention_mask[:, :512]
                    # Adjust response_len if truncated
                    if prompt_len < 512:
                        response_len = 512 - prompt_len
                    else:
                        response_len = 0

                inputs = {
                    "input_ids": input_ids.to(self.device),
                    "attention_mask": attention_mask.to(self.device),
                }

                # Get reward from reward model
                reward = self.reward_model(inputs["input_ids"], inputs["attention_mask"])

                # Expand reward to response length
                reward_seq = reward.unsqueeze(1).expand(-1, response_len)

                rewards_list.append(reward_seq)

        # Pad to same length
        max_len = max(r.shape[1] for r in rewards_list)
        padded_rewards = []
        for r in rewards_list:
            pad_len = max_len - r.shape[1]
            if pad_len > 0:
                padding = torch.zeros(r.shape[0], pad_len, device=self.device)
                r = torch.cat([r, padding], dim=1)
            padded_rewards.append(r)

        rewards_tensor = torch.cat(padded_rewards, dim=0)

        return rewards_tensor

    def _compute_values_with_grad(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """
        Compute value estimates WITH gradients for training.

        Args:
            prompts: List of prompts
            responses: List of generated responses

        Returns:
            values: Value estimates [batch_size, seq_len]
        """
        values_list = []

        for prompt, response in zip(prompts, responses, strict=False):
            # Tokenize prompt and response separately then concatenate
            # This ensures exact boundary alignment
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            response_tokens = self.tokenizer(
                response, return_tensors="pt", add_special_tokens=False
            )

            prompt_len = prompt_tokens["input_ids"].shape[1]
            response_len = response_tokens["input_ids"].shape[1]

            # Concatenate token IDs
            input_ids = torch.cat([prompt_tokens["input_ids"], response_tokens["input_ids"]], dim=1)
            attention_mask = torch.ones_like(input_ids)

            # Truncate if needed
            if input_ids.shape[1] > 512:
                input_ids = input_ids[:, :512]
                attention_mask = attention_mask[:, :512]
                # Adjust response_len if truncated
                if prompt_len < 512:
                    response_len = 512 - prompt_len
                else:
                    response_len = 0

            inputs = {
                "input_ids": input_ids.to(self.device),
                "attention_mask": attention_mask.to(self.device),
            }

            # Get value estimate WITH gradients
            value = self.value_model(inputs["input_ids"], inputs["attention_mask"])

            # Repeat value to response length (use repeat not expand for gradient flow)
            value_seq = value.unsqueeze(1).repeat(1, response_len)

            values_list.append(value_seq)

        # Pad to same length (preserve gradients)
        max_len = max(v.shape[1] for v in values_list)
        padded_values = []
        for v in values_list:
            pad_len = max_len - v.shape[1]
            if pad_len > 0:
                # Create padding with requires_grad to preserve gradient flow
                padding = torch.zeros(v.shape[0], pad_len, device=self.device, requires_grad=True)
                v = torch.cat([v, padding], dim=1)
            padded_values.append(v)

        values_tensor = torch.cat(padded_values, dim=0)

        return values_tensor

    def compute_values(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """
        Compute value estimates using the value model (without gradients).

        Args:
            prompts: List of prompts
            responses: List of generated responses

        Returns:
            values: Value estimates [batch_size, seq_len]
        """
        # If no value model, use rewards as values (common fallback)
        if self.value_model is None:
            return self.compute_rewards(prompts, responses)

        self.value_model.eval()

        values_list = []

        with torch.no_grad():
            for prompt, response in zip(prompts, responses, strict=False):
                # Tokenize prompt to get prompt length
                prompt_inputs = self.tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                )
                prompt_len = prompt_inputs["input_ids"].shape[1]

                # Tokenize prompt + response
                text = prompt + response
                inputs = self.tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True, max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get value estimate (assuming value_model has same interface as reward_model)
                # Value model should output scalar value for each state
                value = self.value_model(inputs["input_ids"], inputs["attention_mask"])

                # Expand value to RESPONSE length only (not full sequence)
                response_len = inputs["input_ids"].shape[1] - prompt_len
                value_seq = value.unsqueeze(1).expand(-1, response_len)

                values_list.append(value_seq)

        # Pad to same length
        max_len = max(v.shape[1] for v in values_list)
        padded_values = []
        for v in values_list:
            pad_len = max_len - v.shape[1]
            if pad_len > 0:
                padding = torch.zeros(v.shape[0], pad_len, device=self.device)
                v = torch.cat([v, padding], dim=1)
            padded_values.append(v)

        values_tensor = torch.cat(padded_values, dim=0)

        return values_tensor

    def _get_logprobs_with_grad(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """
        Compute log probabilities WITH gradients for policy training.

        This method computes logprobs for policy updates and backpropagation.
        Use get_logprobs() for evaluation without gradients.

        Args:
            prompts: List of prompts
            responses: List of responses

        Returns:
            log_probs: Log probabilities [batch_size, response_len] WITH gradients
        """
        self.policy_model.train()

        all_logprobs = []

        for prompt, response in zip(prompts, responses, strict=False):
            # Tokenize prompt and response separately then concatenate
            # This ensures exact boundary alignment
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            response_tokens = self.tokenizer(
                response, return_tensors="pt", add_special_tokens=False
            )

            prompt_len = prompt_tokens["input_ids"].shape[1]

            # Concatenate token IDs
            input_ids = torch.cat([prompt_tokens["input_ids"], response_tokens["input_ids"]], dim=1)
            attention_mask = torch.ones_like(input_ids)

            # Truncate if needed
            if input_ids.shape[1] > 512:
                input_ids = input_ids[:, :512]
                attention_mask = attention_mask[:, :512]

            inputs = {
                "input_ids": input_ids.to(self.device),
                "attention_mask": attention_mask.to(self.device),
            }

            # Get model outputs WITH gradients (no torch.no_grad())
            outputs = self.policy_model(**inputs)
            logits = outputs.logits

            # Compute log probabilities for response tokens
            shift_logits = logits[:, prompt_len - 1 : -1, :]
            shift_labels = inputs["input_ids"][:, prompt_len:]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            all_logprobs.append(token_log_probs)

        # Pad to same length (preserve gradients)
        max_len = max(lp.shape[1] for lp in all_logprobs)
        padded_logprobs = []
        for lp in all_logprobs:
            pad_len = max_len - lp.shape[1]
            if pad_len > 0:
                # Create padding with requires_grad to preserve gradient flow
                padding = torch.zeros(lp.shape[0], pad_len, device=self.device, requires_grad=True)
                lp = torch.cat([lp, padding], dim=1)
            padded_logprobs.append(lp)

        log_probs_tensor = torch.cat(padded_logprobs, dim=0)

        return log_probs_tensor

    def get_logprobs(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """
        Compute log probabilities for given responses under current policy.

        Args:
            prompts: List of prompts
            responses: List of responses

        Returns:
            log_probs: Log probabilities [batch_size, seq_len]
        """
        self.policy_model.eval()

        all_logprobs = []

        with torch.no_grad():
            for prompt, response in zip(prompts, responses, strict=False):
                # Tokenize prompt and response separately then concatenate
                # This ensures exact boundary alignment
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
                response_tokens = self.tokenizer(
                    response, return_tensors="pt", add_special_tokens=False
                )

                prompt_len = prompt_tokens["input_ids"].shape[1]

                # Concatenate token IDs
                input_ids = torch.cat(
                    [prompt_tokens["input_ids"], response_tokens["input_ids"]], dim=1
                )
                attention_mask = torch.ones_like(input_ids)

                # Truncate if needed
                if input_ids.shape[1] > 512:
                    input_ids = input_ids[:, :512]
                    attention_mask = attention_mask[:, :512]

                inputs = {
                    "input_ids": input_ids.to(self.device),
                    "attention_mask": attention_mask.to(self.device),
                }

                # Get model outputs
                outputs = self.policy_model(**inputs)
                logits = outputs.logits

                # Compute log probabilities for response tokens
                # shift by 1 for autoregressive prediction
                shift_logits = logits[:, prompt_len - 1 : -1, :]
                shift_labels = inputs["input_ids"][:, prompt_len:]

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                all_logprobs.append(token_log_probs)

        # Pad to same length
        max_len = max(lp.shape[1] for lp in all_logprobs)
        padded_logprobs = []
        for lp in all_logprobs:
            pad_len = max_len - lp.shape[1]
            if pad_len > 0:
                padding = torch.zeros(lp.shape[0], pad_len, device=self.device)
                lp = torch.cat([lp, padding], dim=1)
            padded_logprobs.append(lp)

        log_probs_tensor = torch.cat(padded_logprobs, dim=0)

        return log_probs_tensor

    def get_reference_logprobs(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """
        Compute log probabilities for given responses under reference policy.

        Args:
            prompts: List of prompts
            responses: List of responses

        Returns:
            log_probs: Log probabilities from reference model [batch_size, seq_len]
        """
        self.reference_model.eval()

        all_logprobs = []

        with torch.no_grad():
            for prompt, response in zip(prompts, responses, strict=False):
                # Tokenize prompt and response separately then concatenate
                # This ensures exact boundary alignment
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
                response_tokens = self.tokenizer(
                    response, return_tensors="pt", add_special_tokens=False
                )

                prompt_len = prompt_tokens["input_ids"].shape[1]

                # Concatenate token IDs
                input_ids = torch.cat(
                    [prompt_tokens["input_ids"], response_tokens["input_ids"]], dim=1
                )
                attention_mask = torch.ones_like(input_ids)

                # Truncate if needed
                if input_ids.shape[1] > 512:
                    input_ids = input_ids[:, :512]
                    attention_mask = attention_mask[:, :512]

                inputs = {
                    "input_ids": input_ids.to(self.device),
                    "attention_mask": attention_mask.to(self.device),
                }

                # Get reference model outputs
                outputs = self.reference_model(**inputs)
                logits = outputs.logits

                # Compute log probabilities for response tokens
                shift_logits = logits[:, prompt_len - 1 : -1, :]
                shift_labels = inputs["input_ids"][:, prompt_len:]

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                all_logprobs.append(token_log_probs)

        # Pad to same length
        max_len = max(lp.shape[1] for lp in all_logprobs)
        padded_logprobs = []
        for lp in all_logprobs:
            pad_len = max_len - lp.shape[1]
            if pad_len > 0:
                padding = torch.zeros(lp.shape[0], pad_len, device=self.device)
                lp = torch.cat([lp, padding], dim=1)
            padded_logprobs.append(lp)

        log_probs_tensor = torch.cat(padded_logprobs, dim=0)

        return log_probs_tensor

    def train_step(
        self,
        prompts: List[str],
        num_epochs_per_batch: int = 4,
        max_length: int = 150,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """
        Single PPO training step.

        This implements the full PPO algorithm:
        1. Generate responses with current policy
        2. Compute rewards using reward model
        3. Compute values and advantages using GAE
        4. Multiple epochs of policy optimization with clipping
        5. Train value function

        Args:
            prompts: Batch of prompts
            num_epochs_per_batch: Number of optimization epochs per batch
            max_length: Maximum response length
            temperature: Sampling temperature

        Returns:
            Training metrics dictionary
        """
        metrics = {}

        # Step 1: Generate responses with current policy
        responses, old_logprobs = self.generate_responses(
            prompts, max_length=max_length, temperature=temperature
        )

        # Step 2: Compute rewards using reward model
        rewards = self.compute_rewards(prompts, responses)

        # Step 3: Compute values
        values = self.compute_values(prompts, responses)

        # Step 4: Compute advantages using GAE
        dones = torch.zeros_like(rewards)  # No early termination in text generation
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Normalize advantages (improves stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Detach to prevent gradients flowing through old computations
        old_logprobs = old_logprobs.detach()
        advantages = advantages.detach()
        returns = returns.detach()

        # Step 5: Multiple epochs of optimization
        policy_losses = []
        value_losses = []
        kl_divs = []

        for _epoch in range(num_epochs_per_batch):
            # Get current log probabilities WITH gradients for training
            new_logprobs = self._get_logprobs_with_grad(prompts, responses)

            # Compute PPO loss
            ppo_loss = self.compute_ppo_loss(old_logprobs, new_logprobs, advantages)

            # Compute KL divergence for monitoring and penalty
            reference_logprobs = self.get_reference_logprobs(prompts, responses)
            kl_div = self.compute_kl_divergence(new_logprobs, reference_logprobs)

            # Total policy loss: PPO loss + KL penalty
            policy_loss = ppo_loss + self.kl_penalty * kl_div

            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()

            # Compute value function loss (only if we have a value model)
            if self.value_model is not None:
                # Compute values WITH gradients for training
                self.value_model.train()
                new_values = self._compute_values_with_grad(prompts, responses)
                value_loss = F.mse_loss(new_values, returns)

                # Update value function
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
            else:
                value_loss = torch.tensor(0.0)

            # Track metrics
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            kl_divs.append(kl_div.item())

        # Aggregate metrics
        metrics["policy_loss"] = np.mean(policy_losses)
        metrics["value_loss"] = np.mean(value_losses)
        metrics["kl_divergence"] = np.mean(kl_divs)
        metrics["mean_reward"] = rewards.mean().item()
        metrics["mean_advantage"] = advantages.mean().item()

        # Update statistics
        self.stats["total_steps"] += 1
        self.stats["policy_losses"].append(metrics["policy_loss"])
        self.stats["value_losses"].append(metrics["value_loss"])
        self.stats["kl_divergences"].append(metrics["kl_divergence"])
        self.stats["mean_rewards"].append(metrics["mean_reward"])
        self.stats["mean_advantages"].append(metrics["mean_advantage"])

        return metrics

    def train(
        self,
        prompts: List[str],
        num_steps: int = 100,
        batch_size: int = 4,
        num_epochs_per_batch: int = 4,
        max_length: int = 150,
        temperature: float = 1.0,
        checkpoint_dir: str | None = None,
        checkpoint_freq: int = 10,
    ) -> Dict[str, Any]:
        """
        Full PPO training loop.

        Args:
            prompts: Training prompts
            num_steps: Number of training steps
            batch_size: Number of prompts per batch
            num_epochs_per_batch: Optimization epochs per batch
            max_length: Maximum response length
            temperature: Sampling temperature
            checkpoint_dir: Directory to save checkpoints
            checkpoint_freq: Checkpoint frequency (steps)

        Returns:
            Training results and metrics
        """
        logger.info(f"Starting PPO training for {num_steps} steps")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Epochs per batch: {num_epochs_per_batch}")
        logger.info(f"Total prompts: {len(prompts)}")

        training_history = {
            "policy_losses": [],
            "value_losses": [],
            "kl_divergences": [],
            "step_avg_rewards": [],
            "mean_rewards": [],  # Alias for step_avg_rewards for backwards compatibility
        }

        for step in tqdm(range(num_steps), desc="PPO Training"):
            # Sample batch of prompts
            batch_indices = np.random.choice(
                len(prompts), size=min(batch_size, len(prompts)), replace=False
            )
            batch_prompts = [prompts[i] for i in batch_indices]

            # Training step
            metrics = self.train_step(
                batch_prompts,
                num_epochs_per_batch=num_epochs_per_batch,
                max_length=max_length,
                temperature=temperature,
            )

            # Log metrics
            training_history["policy_losses"].append(metrics["policy_loss"])
            training_history["value_losses"].append(metrics["value_loss"])
            training_history["kl_divergences"].append(metrics["kl_divergence"])
            training_history["step_avg_rewards"].append(metrics["mean_reward"])
            training_history["mean_rewards"].append(
                metrics["mean_reward"]
            )  # Alias for backwards compatibility

            # Print progress
            if (step + 1) % 10 == 0:
                logger.info(f"\nStep {step + 1}/{num_steps}")
                logger.info(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                logger.info(f"  Value Loss: {metrics['value_loss']:.4f}")
                logger.info(f"  KL Div: {metrics['kl_divergence']:.4f}")
                logger.info(f"  Mean Reward: {metrics['mean_reward']:.4f}")

            # Checkpoint
            if checkpoint_dir and (step + 1) % checkpoint_freq == 0:
                self.save_checkpoint(checkpoint_dir, step + 1)

        return {
            "training_history": training_history,
            "final_stats": self.stats,
            "final_avg_reward": (
                training_history["step_avg_rewards"][-1]
                if training_history["step_avg_rewards"]
                else 0.0
            ),
            "final_kl_divergence": (
                training_history["kl_divergences"][-1]
                if training_history["kl_divergences"]
                else 0.0
            ),
        }

    def save_checkpoint(self, checkpoint_dir: str, step: int):
        """
        Save training checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint
            step: Current training step
        """
        import os

        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"ppo_checkpoint_step_{step}.pt")

        checkpoint = {
            "step": step,
            "policy_model_state_dict": self.policy_model.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "stats": self.stats,
        }

        if self.value_model is not None:
            checkpoint["value_model_state_dict"] = self.value_model.state_dict()
            checkpoint["value_optimizer_state_dict"] = self.value_optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        # PyTorch 2.6+ requires weights_only=False for checkpoints with numpy objects
        # This is safe for checkpoints we create ourselves
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])

        if self.value_model is not None and "value_model_state_dict" in checkpoint:
            self.value_model.load_state_dict(checkpoint["value_model_state_dict"])
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])

        self.stats = checkpoint["stats"]

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from step {checkpoint['step']}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            **self.stats,
            "avg_policy_loss": (
                np.mean(self.stats["policy_losses"]) if self.stats["policy_losses"] else 0.0
            ),
            "avg_value_loss": (
                np.mean(self.stats["value_losses"]) if self.stats["value_losses"] else 0.0
            ),
            "avg_kl_divergence": (
                np.mean(self.stats["kl_divergences"]) if self.stats["kl_divergences"] else 0.0
            ),
            "avg_reward": (
                np.mean(self.stats["mean_rewards"]) if self.stats["mean_rewards"] else 0.0
            ),
        }
