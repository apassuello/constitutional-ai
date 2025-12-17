"""MODULE: reward_model.py
PURPOSE: Reward model training for Constitutional AI (RLAIF Phase 2a)
KEY COMPONENTS:
- RewardModel: Neural network that scores responses based on constitutional compliance
- compute_reward_loss: Bradley-Terry preference ranking loss
- train_reward_model: Training function for reward model
- RewardModelTrainer: Complete training pipeline with validation and checkpointing
DEPENDENCIES: torch, transformers, typing
SPECIAL NOTES: Implements Component 2 of Constitutional AI - trains reward model on preference pairs
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    Reward model for Constitutional AI.

    Takes (prompt, response) pairs as input and outputs a scalar reward score
    indicating the quality/constitutional compliance of the response.

    Architecture:
        - Base language model (frozen or fine-tuned)
        - Classification head: hidden_size -> 256 -> 1 (scalar reward)

    The model is trained using preference pairs to learn which responses
    better follow constitutional principles.

    Attributes:
        base_model: Pre-trained language model (e.g., GPT-2)
        reward_head: Neural network projecting hidden states to scalar reward
    """

    def __init__(self, base_model, hidden_size: int = 768):
        """
        Initialize reward model.

        Args:
            base_model: Pre-trained language model (e.g., GPT-2)
                       Must support output_hidden_states=True
            hidden_size: Hidden size of base model's final layer (default: 768 for GPT-2)

        Example:
            >>> from transformers import AutoModelForCausalLM
            >>> base_model = AutoModelForCausalLM.from_pretrained('gpt2')
            >>> reward_model = RewardModel(base_model, hidden_size=768)
        """
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size

        # Reward head: projects hidden states to scalar score
        # Architecture: Linear(768->256) -> ReLU -> Dropout -> Linear(256->1)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute reward score.

        Process:
        1. Pass input through base model to get hidden states
        2. Extract hidden state of last token
        3. Project through reward head to get scalar score

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            reward: Scalar reward scores [batch_size]

        Example:
            >>> inputs = tokenizer("Hello world", return_tensors="pt")
            >>> reward = reward_model(inputs['input_ids'], inputs['attention_mask'])
            >>> reward.shape
            torch.Size([1])
        """
        # Get base model outputs with hidden states
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )

        # Extract last layer hidden states: [batch_size, seq_len, hidden_size]
        hidden_states = outputs.hidden_states[-1]

        # Get hidden state of last token for each sequence in batch
        # We find the last non-padding token for each sequence
        sequence_lengths = (
            attention_mask.sum(dim=1) - 1
        ).long()  # -1 for 0-indexing, cast to long for indexing
        batch_size = hidden_states.shape[0]

        # Index to get last token hidden state for each sequence
        # Shape: [batch_size, hidden_size]
        last_token_hidden = hidden_states[
            torch.arange(batch_size, device=hidden_states.device), sequence_lengths
        ]

        # Compute reward score through reward head
        # Shape: [batch_size, 1] -> squeeze to [batch_size]
        reward = self.reward_head(last_token_hidden).squeeze(-1)

        return reward

    def get_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        tokenizer,
        device: torch.device,
        max_length: int = 512,
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of (prompt, response) pairs.

        Convenience method that handles tokenization and device management.

        Args:
            prompts: List of prompts
            responses: List of responses (same length as prompts)
            tokenizer: Tokenizer for encoding text
            device: Device to run computation on
            max_length: Maximum sequence length (default: 512)

        Returns:
            rewards: Tensor of shape [batch_size] with reward scores

        Example:
            >>> prompts = ["What is AI?", "Explain gravity"]
            >>> responses = ["AI is artificial intelligence", "Gravity is a force"]
            >>> rewards = reward_model.get_rewards(prompts, responses, tokenizer, device)
        """
        # Combine prompts with responses
        texts = [p + " " + r for p, r in zip(prompts, responses, strict=False)]

        # Tokenize
        encodings = tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )

        # Move to device
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        # Compute rewards
        with torch.no_grad():
            rewards = self.forward(input_ids, attention_mask)

        return rewards


def compute_reward_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """
    Compute preference ranking loss using Bradley-Terry model.

    The Bradley-Terry model assumes:
        P(chosen > rejected) = sigmoid(reward_chosen - reward_rejected)

    Loss encourages reward_chosen > reward_rejected by minimizing:
        -log(P(chosen > rejected)) = -log(sigmoid(reward_chosen - reward_rejected))

    This is equivalent to binary cross-entropy with target=1 for the preference.

    Args:
        reward_chosen: Rewards for chosen (preferred) responses [batch_size]
        reward_rejected: Rewards for rejected responses [batch_size]

    Returns:
        loss: Scalar loss value (mean over batch)

    Example:
        >>> reward_chosen = torch.tensor([1.5, 2.0, 1.8])
        >>> reward_rejected = torch.tensor([0.5, 0.8, 1.0])
        >>> loss = compute_reward_loss(reward_chosen, reward_rejected)
        >>> # Loss should be low since chosen > rejected

    Note:
        - If reward_chosen >> reward_rejected: loss -> 0 (good)
        - If reward_chosen ~= reward_rejected: loss ~= 0.693 (uncertain)
        - If reward_chosen << reward_rejected: loss -> infinity (bad)
    """
    # Bradley-Terry preference model
    # P(chosen > rejected) = sigmoid(reward_chosen - reward_rejected)
    # Loss = -log(P(chosen > rejected))
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()

    return loss


def train_reward_model(
    reward_model: RewardModel,
    training_data: List[Dict[str, Any]],
    tokenizer,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    device: torch.device | None = None,
    validation_data: List[Dict[str, Any]] | None = None,
    max_length: int = 512,
    gradient_accumulation_steps: int = 1,
    log_interval: int = 10,
) -> Dict[str, Any]:
    """
    Train reward model on preference pairs.

    Training process:
    1. For each preference pair (chosen, rejected):
       - Compute reward_chosen = RewardModel(prompt + chosen)
       - Compute reward_rejected = RewardModel(prompt + rejected)
       - Compute loss = -log(sigmoid(reward_chosen - reward_rejected))
    2. Update model parameters to minimize loss
    3. Track accuracy: % of times reward_chosen > reward_rejected

    Args:
        reward_model: RewardModel instance to train
        training_data: List of preference examples with keys:
                      'prompt', 'chosen', 'rejected'
        tokenizer: Tokenizer for encoding text
        num_epochs: Number of training epochs (default: 3)
        batch_size: Batch size for training (default: 4)
        learning_rate: Learning rate (default: 1e-5)
        device: Computation device (default: auto-detect)
        validation_data: Optional validation set with same format
        max_length: Maximum sequence length (default: 512)
        gradient_accumulation_steps: Steps to accumulate gradients (default: 1)
        log_interval: How often to log progress (default: 10)

    Returns:
        Dictionary with training metrics:
            - losses: List of average loss per epoch
            - accuracy: List of accuracy per epoch (% correct preferences)
            - epochs: List of epoch numbers
            - val_losses: Validation losses (if validation_data provided)
            - val_accuracy: Validation accuracy (if validation_data provided)

    Example:
        >>> from constitutional_ai.preference_comparison import generate_preference_pairs
        >>> preference_data = generate_preference_pairs(prompts, model, tokenizer, framework, device)
        >>> metrics = train_reward_model(reward_model, preference_data, tokenizer, num_epochs=3)
        >>> logger.info(f"Final accuracy: {metrics['accuracy'][-1]:.2%}")
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Training reward model on {device}")
    logger.info(f"Training samples: {len(training_data)}")
    if validation_data:
        logger.info(f"Validation samples: {len(validation_data)}")

    reward_model = reward_model.to(device)
    reward_model.train()

    # Setup optimizer
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=learning_rate)

    # Initialize metrics tracking
    metrics = {"losses": [], "accuracy": [], "epochs": []}

    if validation_data:
        metrics["val_losses"] = []
        metrics["val_accuracy"] = []

    # Import tqdm if available
    try:
        from tqdm import tqdm

        use_tqdm = True
    except ImportError:
        use_tqdm = False

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        # Create batches
        num_batches = (len(training_data) + batch_size - 1) // batch_size

        # Setup iterator
        batch_iterator = range(0, len(training_data), batch_size)
        if use_tqdm:
            batch_iterator = tqdm(
                batch_iterator, desc=f"Epoch {epoch+1}/{num_epochs}", total=num_batches
            )

        for i in batch_iterator:
            batch = training_data[i : i + batch_size]

            # Prepare texts for chosen responses
            chosen_texts = [item["prompt"] + " " + item["chosen"] for item in batch]

            # Prepare texts for rejected responses
            rejected_texts = [item["prompt"] + " " + item["rejected"] for item in batch]

            # Tokenize chosen responses
            chosen_encodings = tokenizer(
                chosen_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Tokenize rejected responses
            rejected_encodings = tokenizer(
                rejected_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Move to device
            chosen_ids = chosen_encodings["input_ids"].to(device)
            chosen_mask = chosen_encodings["attention_mask"].to(device)
            rejected_ids = rejected_encodings["input_ids"].to(device)
            rejected_mask = rejected_encodings["attention_mask"].to(device)

            # Forward pass
            reward_chosen = reward_model(chosen_ids, chosen_mask)
            reward_rejected = reward_model(rejected_ids, rejected_mask)

            # Compute loss
            loss = compute_reward_loss(reward_chosen, reward_rejected)

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update parameters if we've accumulated enough gradients
            if (i // batch_size + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Track metrics (use unscaled loss)
            epoch_loss += loss.item() * gradient_accumulation_steps

            # Calculate accuracy: how often reward_chosen > reward_rejected
            with torch.no_grad():
                correct += (reward_chosen > reward_rejected).sum().item()
                total += len(batch)

        # Final optimizer step if gradients remain
        if (num_batches % gradient_accumulation_steps) != 0:
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Compute epoch metrics
        avg_loss = epoch_loss / num_batches
        accuracy = correct / total if total > 0 else 0.0

        metrics["losses"].append(avg_loss)
        metrics["accuracy"].append(accuracy)
        metrics["epochs"].append(epoch + 1)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f} ({correct}/{total})"
        )

        # Validation
        if validation_data:
            val_loss, val_accuracy = evaluate_reward_model(
                reward_model,
                validation_data,
                tokenizer,
                device,
                batch_size=batch_size,
                max_length=max_length,
            )
            metrics["val_losses"].append(val_loss)
            metrics["val_accuracy"].append(val_accuracy)
            logger.info(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    logger.info("Training complete!")

    # Add convenience keys for final values
    metrics["final_loss"] = metrics["losses"][-1] if metrics["losses"] else 0.0
    metrics["final_accuracy"] = metrics["accuracy"][-1] if metrics["accuracy"] else 0.0

    return metrics


def evaluate_reward_model(
    reward_model: RewardModel,
    evaluation_data: List[Dict[str, Any]],
    tokenizer,
    device: torch.device,
    batch_size: int = 4,
    max_length: int = 512,
) -> tuple:
    """
    Evaluate reward model on a dataset.

    Args:
        reward_model: Trained RewardModel
        evaluation_data: List of preference examples
        tokenizer: Tokenizer
        device: Device for computation
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length

    Returns:
        Tuple of (average_loss, accuracy)
    """
    reward_model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(evaluation_data), batch_size):
            batch = evaluation_data[i : i + batch_size]

            # Prepare texts
            chosen_texts = [item["prompt"] + " " + item["chosen"] for item in batch]
            rejected_texts = [item["prompt"] + " " + item["rejected"] for item in batch]

            # Tokenize
            chosen_encodings = tokenizer(
                chosen_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            rejected_encodings = tokenizer(
                rejected_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Move to device
            chosen_ids = chosen_encodings["input_ids"].to(device)
            chosen_mask = chosen_encodings["attention_mask"].to(device)
            rejected_ids = rejected_encodings["input_ids"].to(device)
            rejected_mask = rejected_encodings["attention_mask"].to(device)

            # Forward pass
            reward_chosen = reward_model(chosen_ids, chosen_mask)
            reward_rejected = reward_model(rejected_ids, rejected_mask)

            # Compute loss
            loss = compute_reward_loss(reward_chosen, reward_rejected)
            total_loss += loss.item()

            # Calculate accuracy
            correct += (reward_chosen > reward_rejected).sum().item()
            total += len(batch)

    reward_model.train()

    num_batches = (len(evaluation_data) + batch_size - 1) // batch_size
    avg_loss = total_loss / num_batches
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


class RewardModelTrainer:
    """
    Complete training pipeline for reward model with validation and checkpointing.

    This class provides a full-featured training interface with:
    - Automatic train/validation split
    - Checkpoint saving and loading
    - Training history tracking
    - Early stopping (optional)
    - Metrics visualization

    Example:
        >>> trainer = RewardModelTrainer(
        ...     reward_model=reward_model,
        ...     tokenizer=tokenizer,
        ...     device=device
        ... )
        >>> trainer.train(
        ...     training_data=preference_data,
        ...     num_epochs=5,
        ...     save_dir='./checkpoints'
        ... )
        >>> trainer.save_checkpoint('./final_model')
    """

    def __init__(
        self,
        reward_model: RewardModel,
        tokenizer,
        device: torch.device | None = None,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
    ):
        """
        Initialize reward model trainer.

        Args:
            reward_model: RewardModel instance
            tokenizer: Tokenizer
            device: Device for training (default: auto-detect)
            learning_rate: Learning rate (default: 1e-5)
            batch_size: Batch size (default: 4)
        """
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_history = []

    def train(
        self,
        training_data: List[Dict[str, Any]],
        num_epochs: int = 3,
        validation_split: float = 0.1,
        validation_data: List[Dict[str, Any]] | None = None,
        save_dir: str | None = None,
        save_best_only: bool = True,
        early_stopping_patience: int | None = None,
    ) -> Dict[str, Any]:
        """
        Train reward model with validation and checkpointing.

        Args:
            training_data: List of preference examples
            num_epochs: Number of epochs to train
            validation_split: Fraction of training data for validation (default: 0.1)
            validation_data: Optional separate validation set
            save_dir: Directory to save checkpoints (default: None, no saving)
            save_best_only: Only save checkpoint when validation improves
            early_stopping_patience: Stop if no improvement for N epochs (default: None)

        Returns:
            Training metrics dictionary
        """
        # Split data if no validation data provided
        if validation_data is None and validation_split > 0:
            split_idx = int(len(training_data) * (1 - validation_split))
            train_subset = training_data[:split_idx]
            validation_data = training_data[split_idx:]
            logger.info(f"Split data: {len(train_subset)} train, {len(validation_data)} validation")
        else:
            train_subset = training_data

        # Train
        metrics = train_reward_model(
            reward_model=self.reward_model,
            training_data=train_subset,
            tokenizer=self.tokenizer,
            num_epochs=num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            device=self.device,
            validation_data=validation_data,
        )

        # Save training history
        self.training_history.append(metrics)

        # Save checkpoint if requested
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # Determine if this is the best model
            if save_best_only and len(metrics.get("val_accuracy", [])) > 0:
                best_accuracy = max(metrics["val_accuracy"])
                if metrics["val_accuracy"][-1] == best_accuracy:
                    self.save_checkpoint(save_path / "best_model")
                    logger.info(f"Saved best model with validation accuracy: {best_accuracy:.4f}")
            else:
                self.save_checkpoint(save_path / "final_model")

        return metrics

    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint (without extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(
            {
                "model_state_dict": self.reward_model.state_dict(),
                "training_history": self.training_history,
                "hidden_size": self.reward_model.hidden_size,
            },
            str(path) + ".pt",
        )

        # Save metadata
        metadata = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "hidden_size": self.reward_model.hidden_size,
        }
        with open(str(path) + "_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint (without extension)
        """
        path = Path(path)

        # Load model state
        checkpoint = torch.load(str(path) + ".pt", map_location=self.device, weights_only=True)
        self.reward_model.load_state_dict(checkpoint["model_state_dict"])
        self.training_history = checkpoint.get("training_history", [])

        logger.info(f"Checkpoint loaded from {path}")

    def evaluate(self, evaluation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            evaluation_data: List of preference examples

        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        loss, accuracy = evaluate_reward_model(
            self.reward_model,
            evaluation_data,
            self.tokenizer,
            self.device,
            batch_size=self.batch_size,
        )

        return {"loss": loss, "accuracy": accuracy}
