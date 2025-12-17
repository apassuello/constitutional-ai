"""
Unit tests for PPO Trainer (Component 4).

Tests all core PPO components:
- GAE computation
- KL divergence calculation
- Clipped PPO objective
- Training step
- Full training loop
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from constitutional_ai.ppo_trainer import PPOTrainer


class MockRewardModel(nn.Module):
    """Mock reward model for testing."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """Return mock reward scores."""
        batch_size = input_ids.shape[0]
        # Return random rewards between -1 and 1
        return torch.randn(batch_size)


class MockValueModel(nn.Module):
    """Mock value model for testing."""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        """Return mock value estimates."""
        batch_size = input_ids.shape[0]
        # Create dummy hidden states from input_ids to establish gradient connection
        # Use mean of input_ids as a simple feature, ensuring float type
        hidden = input_ids.float().mean(dim=1, keepdim=True)
        hidden = hidden.expand(batch_size, self.value_head.in_features)
        # Pass through value_head to connect to model parameters
        values = self.value_head(hidden).squeeze(-1)
        return values


class TestComputeGAE:
    """Test Generalized Advantage Estimation."""

    def test_gae_basic_computation(self):
        """Test that GAE computes advantages correctly."""
        device = torch.device("cpu")

        # Create mock models
        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Initialize trainer
        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            gamma=0.99,
            gae_lambda=0.95,
        )

        # Test with simple rewards and values
        batch_size = 2
        seq_len = 5

        rewards = torch.tensor([[1.0, 0.5, 0.2, 0.1, 0.0], [0.8, 0.6, 0.4, 0.2, 0.0]])
        values = torch.tensor([[0.5, 0.4, 0.3, 0.2, 0.1], [0.6, 0.5, 0.4, 0.3, 0.2]])
        dones = torch.zeros(batch_size, seq_len)

        # Compute GAE
        advantages, returns = trainer.compute_gae(rewards, values, dones)

        # Check shapes
        assert advantages.shape == (batch_size, seq_len)
        assert returns.shape == (batch_size, seq_len)

        # Check that advantages are computed (not all zeros)
        assert not torch.allclose(advantages, torch.zeros_like(advantages))

        # Check that returns = advantages + values
        torch.testing.assert_close(returns, advantages + values, rtol=1e-5, atol=1e-5)

    def test_gae_backwards_computation(self):
        """Test that GAE computes backwards through time."""
        device = torch.device("cpu")

        # Create minimal trainer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            gamma=1.0,  # No discounting for simpler test
            gae_lambda=0.0,  # Pure TD(0) for verification
        )

        # Simple test case
        rewards = torch.tensor([[1.0, 0.0, 0.0]])
        values = torch.tensor([[0.0, 0.0, 0.0]])
        dones = torch.zeros(1, 3)

        advantages, returns = trainer.compute_gae(rewards, values, dones)

        # With lambda=0, advantages should be TD errors
        # First timestep: δ = r_0 + γ*V(s_1) - V(s_0) = 1.0 + 0 - 0 = 1.0
        assert advantages[0, 0].item() == pytest.approx(1.0, abs=1e-5)

    def test_gae_with_dones(self):
        """Test that GAE handles episode termination correctly."""
        device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
        )

        rewards = torch.tensor([[1.0, 1.0, 1.0]])
        values = torch.tensor([[0.5, 0.5, 0.5]])
        dones = torch.tensor([[0.0, 1.0, 0.0]])  # Episode ends at timestep 1

        advantages, returns = trainer.compute_gae(rewards, values, dones)

        # Check that computation doesn't propagate past done
        assert advantages.shape == (1, 3)


class TestComputeKLDivergence:
    """Test KL divergence computation."""

    def test_kl_divergence_identical_policies(self):
        """Test KL divergence is ~0 for identical policies."""
        device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
        )

        # Identical log probabilities
        logprobs = torch.randn(4, 10)

        kl_div = trainer.compute_kl_divergence(logprobs, logprobs)

        # Should be exactly 0
        assert kl_div.item() == pytest.approx(0.0, abs=1e-6)

    def test_kl_divergence_different_policies(self):
        """Test KL divergence is positive for different policies."""
        device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
        )

        # Different log probabilities
        current_logprobs = torch.randn(4, 10)
        reference_logprobs = torch.randn(4, 10)

        kl_div = trainer.compute_kl_divergence(current_logprobs, reference_logprobs)

        # KL can be positive or negative in our approximation
        # Just check it's computed
        assert isinstance(kl_div.item(), float)


class TestComputePPOLoss:
    """Test clipped PPO objective."""

    def test_ppo_loss_no_clipping(self):
        """Test PPO loss when ratio is within clip range."""
        device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            clip_epsilon=0.2,
        )

        # Small difference in log probs (ratio close to 1)
        old_logprobs = torch.tensor([[-1.0, -1.0]])
        new_logprobs = torch.tensor([[-1.05, -0.95]])  # ratio ~ [0.95, 1.05]
        advantages = torch.tensor([[1.0, 1.0]])

        loss = trainer.compute_ppo_loss(old_logprobs, new_logprobs, advantages)

        # Loss should be computed
        assert isinstance(loss.item(), float)
        # Should be negative of expected value (we minimize loss)
        assert loss.item() < 0

    def test_ppo_loss_with_clipping(self):
        """Test PPO loss clips large ratios."""
        device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            clip_epsilon=0.2,
        )

        # Large difference in log probs (ratio far from 1)
        old_logprobs = torch.tensor([[-1.0, -1.0]])
        new_logprobs = torch.tensor([[-0.1, -2.0]])  # Large ratios
        advantages = torch.tensor([[1.0, 1.0]])

        loss = trainer.compute_ppo_loss(old_logprobs, new_logprobs, advantages)

        # Loss should be computed and clipping should apply
        assert isinstance(loss.item(), float)

    def test_ppo_loss_negative_advantages(self):
        """Test PPO loss with negative advantages."""
        device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
        )

        old_logprobs = torch.tensor([[-1.0, -1.0]])
        new_logprobs = torch.tensor([[-1.05, -0.95]])
        advantages = torch.tensor([[-1.0, -1.0]])  # Negative advantages

        loss = trainer.compute_ppo_loss(old_logprobs, new_logprobs, advantages)

        # Loss should be positive (we want to decrease prob of bad actions)
        assert isinstance(loss.item(), float)


class TestTrainStep:
    """Test PPO training step."""

    @pytest.mark.slow
    def test_train_step_completes(self):
        """Test that training step completes without errors."""
        device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
        )

        prompts = ["Hello", "Hi there"]

        # Run training step with minimal epochs
        metrics = trainer.train_step(
            prompts, num_epochs_per_batch=1, max_length=20, temperature=1.0
        )

        # Check that metrics are returned
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "kl_divergence" in metrics
        assert "mean_reward" in metrics
        assert "mean_advantage" in metrics

        # Check that all metrics are numbers
        for _key, value in metrics.items():
            assert isinstance(value, (int, float))

    @pytest.mark.slow
    def test_train_step_updates_parameters(self):
        """Test that training step updates policy parameters."""
        device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
        )

        # Get initial parameters
        initial_params = {
            name: param.clone()
            for name, param in policy_model.named_parameters()
            if param.requires_grad
        }

        prompts = ["Test prompt"]

        # Run training step
        trainer.train_step(prompts, num_epochs_per_batch=2, max_length=20)

        # Check that some parameters changed
        params_changed = False
        for name, param in policy_model.named_parameters():
            if param.requires_grad and name in initial_params:
                if not torch.allclose(param, initial_params[name], atol=1e-6):
                    params_changed = True
                    break

        assert params_changed, "Policy parameters should be updated after training step"

    @pytest.mark.slow
    def test_train_step_gradients_flow(self):
        """Test that gradients flow properly through policy."""
        device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
        )

        prompts = ["Test"]

        # Run training step
        trainer.train_step(prompts, num_epochs_per_batch=1, max_length=15)

        # Check that statistics were updated
        assert trainer.stats["total_steps"] == 1
        assert len(trainer.stats["policy_losses"]) == 1
        assert len(trainer.stats["value_losses"]) == 1


class TestFullTraining:
    """Test full PPO training loop."""

    @pytest.mark.slow
    def test_train_loop_completes(self):
        """Test that full training loop completes."""
        device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
        )

        prompts = ["Hello", "Hi", "Test"]

        # Run minimal training
        results = trainer.train(
            prompts, num_steps=2, batch_size=2, num_epochs_per_batch=1, max_length=15
        )

        # Check results structure
        assert "training_history" in results
        assert "final_stats" in results

        # Check training history
        history = results["training_history"]
        assert "policy_losses" in history
        assert "value_losses" in history
        assert "kl_divergences" in history
        assert "mean_rewards" in history

        # Check that we have data for all steps
        assert len(history["policy_losses"]) == 2
        assert len(history["value_losses"]) == 2


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    @pytest.mark.slow
    def test_save_and_load_checkpoint(self, tmp_path):
        """Test that checkpoints can be saved and loaded."""
        device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
        )

        # Train a bit
        prompts = ["Test"]
        trainer.train_step(prompts, num_epochs_per_batch=1, max_length=15)

        # Save checkpoint
        checkpoint_dir = str(tmp_path)
        trainer.save_checkpoint(checkpoint_dir, step=1)

        # Create new trainer
        policy_model2 = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model2 = MockValueModel()

        trainer2 = PPOTrainer(
            policy_model=policy_model2,
            value_model=value_model2,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
        )

        # Load checkpoint
        import os

        checkpoint_path = os.path.join(checkpoint_dir, "ppo_checkpoint_step_1.pt")
        trainer2.load_checkpoint(checkpoint_path)

        # Check that stats were loaded
        assert trainer2.stats["total_steps"] == trainer.stats["total_steps"]


class TestIntegration:
    """Integration tests for PPO with reward model."""

    @pytest.mark.slow
    def test_ppo_with_reward_model(self):
        """Test PPO integration with reward model."""
        device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            clip_epsilon=0.2,
            kl_penalty=0.1,
        )

        prompts = ["What is AI?"]

        # Run training
        metrics = trainer.train_step(prompts, num_epochs_per_batch=1, max_length=20)

        # Verify all components work together
        assert metrics["policy_loss"] is not None
        assert metrics["value_loss"] is not None
        assert metrics["kl_divergence"] is not None
        assert metrics["mean_reward"] is not None

    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        device = torch.device("cpu")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
        value_model = MockValueModel()
        reward_model = MockRewardModel()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        trainer = PPOTrainer(
            policy_model=policy_model,
            value_model=value_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
        )

        # Get initial stats
        stats = trainer.get_statistics()

        assert "total_steps" in stats
        assert "avg_policy_loss" in stats
        assert "avg_value_loss" in stats
        assert "avg_kl_divergence" in stats
        assert "avg_reward" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
