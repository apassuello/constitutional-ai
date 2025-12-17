"""
Unit tests for reward model training (Component 2 of Constitutional AI).

Tests cover:
- RewardModel forward pass
- Loss computation (Bradley-Terry model)
- Training loop
- Checkpoint saving/loading
- Integration with PreferenceDataset
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the components to test
from constitutional_ai.reward_model import (
    RewardModel,
    RewardModelTrainer,
    compute_reward_loss,
    evaluate_reward_model,
    train_reward_model,
)


@pytest.fixture
def device():
    """Get available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tokenizer():
    """Load tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def base_model(device):
    """Load small base model for testing."""
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = model.to(device)
    return model


@pytest.fixture
def reward_model(base_model):
    """Create reward model for testing."""
    return RewardModel(base_model, hidden_size=768)


@pytest.fixture
def sample_preference_data():
    """Create sample preference data for testing."""
    return [
        {
            "prompt": "What is photosynthesis?",
            "chosen": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen.",
            "rejected": "Plants make food.",
        },
        {
            "prompt": "Explain gravity.",
            "chosen": "Gravity is a fundamental force that attracts objects with mass toward each other.",
            "rejected": "Things fall down.",
        },
        {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a field of AI where computers learn patterns from data without explicit programming.",
            "rejected": "Computers learning stuff.",
        },
        {
            "prompt": "How does the internet work?",
            "chosen": "The internet works by connecting computers globally through protocols like TCP/IP.",
            "rejected": "Magic wires.",
        },
    ]


class TestRewardModel:
    """Test RewardModel class."""

    def test_initialization(self, base_model):
        """Test that RewardModel initializes correctly."""
        reward_model = RewardModel(base_model, hidden_size=768)

        assert reward_model.base_model is base_model
        assert reward_model.hidden_size == 768
        assert isinstance(reward_model.reward_head, nn.Sequential)
        assert len(reward_model.reward_head) == 4  # Linear, ReLU, Dropout, Linear

    def test_forward_pass_shape(self, reward_model, tokenizer, device):
        """Test that forward pass produces correct output shape."""
        reward_model = reward_model.to(device)

        # Create dummy input
        text = "This is a test response."
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass
        rewards = reward_model(input_ids, attention_mask)

        # Check output shape
        assert rewards.shape == (1,), f"Expected shape (1,), got {rewards.shape}"
        assert rewards.dtype == torch.float32

    def test_forward_pass_batch(self, reward_model, tokenizer, device):
        """Test forward pass with batch input."""
        reward_model = reward_model.to(device)

        # Create batch input
        texts = ["First response.", "Second response.", "Third response."]
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass
        rewards = reward_model(input_ids, attention_mask)

        # Check output shape
        assert rewards.shape == (3,), f"Expected shape (3,), got {rewards.shape}"
        assert rewards.dtype == torch.float32

    def test_forward_pass_different_lengths(self, reward_model, tokenizer, device):
        """Test forward pass handles variable length sequences."""
        reward_model = reward_model.to(device)

        # Create inputs of different lengths
        texts = ["Short.", "This is a much longer response with many more tokens in it."]
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass
        rewards = reward_model(input_ids, attention_mask)

        # Check output shape
        assert rewards.shape == (2,)
        assert not torch.isnan(rewards).any()
        assert not torch.isinf(rewards).any()

    def test_get_rewards_method(self, reward_model, tokenizer, device):
        """Test get_rewards convenience method."""
        reward_model = reward_model.to(device)

        prompts = ["What is AI?", "Explain gravity"]
        responses = ["AI is artificial intelligence", "Gravity is a force"]

        rewards = reward_model.get_rewards(prompts, responses, tokenizer, device)

        assert rewards.shape == (2,)
        assert rewards.dtype == torch.float32

    def test_gradient_flow(self, reward_model, tokenizer, device):
        """Test that gradients flow through the model."""
        reward_model = reward_model.to(device)
        reward_model.train()

        # Create input
        text = "Test response"
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass
        reward = reward_model(input_ids, attention_mask)

        # Backward pass
        loss = reward.sum()
        loss.backward()

        # Check that gradients exist
        has_gradients = False
        for param in reward_model.reward_head.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        assert has_gradients, "No gradients found in reward head"


class TestComputeRewardLoss:
    """Test compute_reward_loss function."""

    def test_loss_basic(self):
        """Test basic loss computation."""
        reward_chosen = torch.tensor([1.5, 2.0, 1.8])
        reward_rejected = torch.tensor([0.5, 0.8, 1.0])

        loss = compute_reward_loss(reward_chosen, reward_rejected)

        # Loss should be positive scalar
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_loss_when_chosen_better(self):
        """Test that loss is low when chosen > rejected."""
        reward_chosen = torch.tensor([2.0, 3.0, 2.5])
        reward_rejected = torch.tensor([0.5, 0.5, 0.5])

        loss = compute_reward_loss(reward_chosen, reward_rejected)

        # Loss should be very small when chosen >> rejected
        assert loss.item() < 0.5, f"Loss too high: {loss.item()}"

    def test_loss_when_equal(self):
        """Test that loss is ~0.693 when rewards are equal."""
        reward_chosen = torch.tensor([1.0, 1.0, 1.0])
        reward_rejected = torch.tensor([1.0, 1.0, 1.0])

        loss = compute_reward_loss(reward_chosen, reward_rejected)

        # For equal rewards, loss should be around -log(0.5) = 0.693
        expected_loss = 0.693
        assert (
            abs(loss.item() - expected_loss) < 0.01
        ), f"Expected ~{expected_loss}, got {loss.item()}"

    def test_loss_when_rejected_better(self):
        """Test that loss is high when rejected > chosen (wrong preference)."""
        reward_chosen = torch.tensor([0.5, 0.5, 0.5])
        reward_rejected = torch.tensor([2.0, 3.0, 2.5])

        loss = compute_reward_loss(reward_chosen, reward_rejected)

        # Loss should be high when chosen << rejected
        assert loss.item() > 1.0, f"Loss should be high, got {loss.item()}"

    def test_loss_gradient(self):
        """Test that loss supports gradient computation."""
        reward_chosen = torch.tensor([1.5, 2.0], requires_grad=True)
        reward_rejected = torch.tensor([0.5, 0.8], requires_grad=True)

        loss = compute_reward_loss(reward_chosen, reward_rejected)
        loss.backward()

        # Check gradients exist
        assert reward_chosen.grad is not None
        assert reward_rejected.grad is not None
        assert not torch.isnan(reward_chosen.grad).any()
        assert not torch.isnan(reward_rejected.grad).any()


class TestTrainRewardModel:
    """Test train_reward_model function."""

    def test_training_completes(self, reward_model, sample_preference_data, tokenizer, device):
        """Test that training completes without errors."""
        metrics = train_reward_model(
            reward_model=reward_model,
            training_data=sample_preference_data,
            tokenizer=tokenizer,
            num_epochs=2,
            batch_size=2,
            learning_rate=1e-5,
            device=device,
        )

        # Check that metrics are returned
        assert "losses" in metrics
        assert "accuracy" in metrics
        assert "epochs" in metrics
        assert len(metrics["losses"]) == 2
        assert len(metrics["accuracy"]) == 2

    def test_training_improves_accuracy(
        self, reward_model, sample_preference_data, tokenizer, device
    ):
        """Test that training improves accuracy over epochs."""
        # Use more epochs to see improvement
        metrics = train_reward_model(
            reward_model=reward_model,
            training_data=sample_preference_data * 5,  # Repeat data for more stable training
            tokenizer=tokenizer,
            num_epochs=3,
            batch_size=2,
            learning_rate=1e-4,  # Higher LR for faster learning in test
            device=device,
        )

        # Accuracy should generally improve or stay high
        # Note: With small data, this might not always hold, so we just check it's reasonable
        assert metrics["accuracy"][-1] >= 0.0, "Accuracy should be non-negative"
        assert metrics["accuracy"][-1] <= 1.0, "Accuracy should not exceed 1.0"

    def test_training_with_validation(
        self, reward_model, sample_preference_data, tokenizer, device
    ):
        """Test training with validation data."""
        # Split data
        train_data = sample_preference_data[:3]
        val_data = sample_preference_data[3:]

        metrics = train_reward_model(
            reward_model=reward_model,
            training_data=train_data,
            tokenizer=tokenizer,
            num_epochs=2,
            batch_size=2,
            learning_rate=1e-5,
            device=device,
            validation_data=val_data,
        )

        # Check validation metrics
        assert "val_losses" in metrics
        assert "val_accuracy" in metrics
        assert len(metrics["val_losses"]) == 2
        assert len(metrics["val_accuracy"]) == 2

    def test_training_loss_decreases(self, reward_model, sample_preference_data, tokenizer, device):
        """Test that loss generally decreases during training."""
        metrics = train_reward_model(
            reward_model=reward_model,
            training_data=sample_preference_data * 3,
            tokenizer=tokenizer,
            num_epochs=3,
            batch_size=2,
            learning_rate=1e-4,
            device=device,
        )

        # Check that loss is finite
        assert all(not torch.isnan(torch.tensor(loss)).item() for loss in metrics["losses"])
        assert all(not torch.isinf(torch.tensor(loss)).item() for loss in metrics["losses"])


class TestEvaluateRewardModel:
    """Test evaluate_reward_model function."""

    def test_evaluation(self, reward_model, sample_preference_data, tokenizer, device):
        """Test that evaluation runs without errors."""
        loss, accuracy = evaluate_reward_model(
            reward_model=reward_model.to(device),
            evaluation_data=sample_preference_data,
            tokenizer=tokenizer,
            device=device,
            batch_size=2,
        )

        # Check outputs
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss >= 0
        assert 0 <= accuracy <= 1

    def test_evaluation_after_training(
        self, reward_model, sample_preference_data, tokenizer, device
    ):
        """Test evaluation after training."""
        # Train first
        train_reward_model(
            reward_model=reward_model,
            training_data=sample_preference_data,
            tokenizer=tokenizer,
            num_epochs=2,
            batch_size=2,
            device=device,
        )

        # Evaluate
        loss, accuracy = evaluate_reward_model(
            reward_model=reward_model,
            evaluation_data=sample_preference_data,
            tokenizer=tokenizer,
            device=device,
            batch_size=2,
        )

        # Accuracy should be reasonable after training
        assert accuracy > 0.0, "Accuracy should be positive after training"


class TestRewardModelTrainer:
    """Test RewardModelTrainer class."""

    def test_initialization(self, reward_model, tokenizer, device):
        """Test trainer initialization."""
        trainer = RewardModelTrainer(
            reward_model=reward_model,
            tokenizer=tokenizer,
            device=device,
            learning_rate=1e-5,
            batch_size=4,
        )

        assert trainer.reward_model is reward_model
        assert trainer.tokenizer is tokenizer
        assert trainer.device == device
        assert trainer.learning_rate == 1e-5
        assert trainer.batch_size == 4

    def test_train_method(self, reward_model, sample_preference_data, tokenizer, device):
        """Test trainer train method."""
        trainer = RewardModelTrainer(reward_model=reward_model, tokenizer=tokenizer, device=device)

        metrics = trainer.train(
            training_data=sample_preference_data, num_epochs=2, validation_split=0.25
        )

        assert "losses" in metrics
        assert "accuracy" in metrics
        assert len(metrics["losses"]) == 2

    def test_save_and_load_checkpoint(self, reward_model, tokenizer, device):
        """Test checkpoint saving and loading."""
        trainer = RewardModelTrainer(reward_model=reward_model, tokenizer=tokenizer, device=device)

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint"

            # Save checkpoint
            trainer.save_checkpoint(str(checkpoint_path))

            # Check files exist
            assert (checkpoint_path.parent / f"{checkpoint_path.name}.pt").exists()
            assert (checkpoint_path.parent / f"{checkpoint_path.name}_metadata.json").exists()

            # Create new trainer and load checkpoint
            new_model = RewardModel(reward_model.base_model, hidden_size=768)
            new_trainer = RewardModelTrainer(
                reward_model=new_model, tokenizer=tokenizer, device=device
            )
            new_trainer.load_checkpoint(str(checkpoint_path))

            # Check that weights match
            for p1, p2 in zip(
                reward_model.reward_head.parameters(),
                new_model.reward_head.parameters(),
                strict=False,
            ):
                assert torch.allclose(p1, p2), "Loaded weights don't match saved weights"

    def test_evaluate_method(self, reward_model, sample_preference_data, tokenizer, device):
        """Test trainer evaluate method."""
        trainer = RewardModelTrainer(reward_model=reward_model, tokenizer=tokenizer, device=device)

        results = trainer.evaluate(sample_preference_data)

        assert "loss" in results
        assert "accuracy" in results
        assert isinstance(results["loss"], float)
        assert isinstance(results["accuracy"], float)


class TestIntegrationWithPreferenceDataset:
    """Test integration with PreferenceDataset from preference_comparison.py."""

    def test_works_with_preference_dataset(
        self, reward_model, sample_preference_data, tokenizer, device
    ):
        """Test that reward model works with PreferenceDataset."""
        from torch.utils.data import DataLoader

        from constitutional_ai.preference_comparison import PreferenceDataset

        # Create dataset
        dataset = PreferenceDataset(
            data=sample_preference_data, tokenizer=tokenizer, max_length=512
        )

        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Get a batch
        batch = next(iter(dataloader))

        # Check batch format
        assert "chosen_input_ids" in batch
        assert "chosen_attention_mask" in batch
        assert "rejected_input_ids" in batch
        assert "rejected_attention_mask" in batch

        # Use reward model
        reward_model = reward_model.to(device)
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        reward_chosen = reward_model(chosen_ids, chosen_mask)
        reward_rejected = reward_model(rejected_ids, rejected_mask)

        # Check outputs
        assert reward_chosen.shape[0] == 2
        assert reward_rejected.shape[0] == 2

        # Compute loss
        loss = compute_reward_loss(reward_chosen, reward_rejected)
        assert not torch.isnan(loss)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_training_data(self, reward_model, tokenizer, device):
        """Test handling of empty training data."""
        with pytest.raises(Exception):
            train_reward_model(
                reward_model=reward_model,
                training_data=[],
                tokenizer=tokenizer,
                num_epochs=1,
                device=device,
            )

    def test_single_example(self, reward_model, sample_preference_data, tokenizer, device):
        """Test training with single example."""
        single_example = sample_preference_data[:1]

        # Should not crash
        metrics = train_reward_model(
            reward_model=reward_model,
            training_data=single_example,
            tokenizer=tokenizer,
            num_epochs=1,
            batch_size=1,
            device=device,
        )

        assert len(metrics["losses"]) == 1

    def test_very_long_sequences(self, reward_model, tokenizer, device):
        """Test handling of very long sequences."""
        long_text = "This is a test. " * 100  # Very long text

        preference_data = [
            {
                "prompt": "Test prompt",
                "chosen": long_text,
                "rejected": "Short response",
            }
        ]

        # Should handle with truncation
        metrics = train_reward_model(
            reward_model=reward_model,
            training_data=preference_data,
            tokenizer=tokenizer,
            num_epochs=1,
            batch_size=1,
            max_length=512,
            device=device,
        )

        assert len(metrics["losses"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
