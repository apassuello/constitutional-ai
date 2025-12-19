"""
Model Manager for Constitutional AI Demo.

Handles loading, state management, and lifecycle of models.
Supports both baseline models and pre-trained constitutional models.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class ModelStatus(Enum):
    """Model loading status."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    TRAINING = "training"
    ERROR = "error"


class ModelManager:
    """
    Manages model loading and state for the demo.

    Supports two modes:
    - Baseline: Load untrained base models (gpt2, distilgpt2, etc.)
    - Constitutional: Load pre-trained constitutional AI models
    """

    def __init__(self):
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.model_name: Optional[str] = None
        self.model_type: Optional[str] = None  # 'baseline' or 'constitutional'
        self.status = ModelStatus.NOT_LOADED
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.error_message: Optional[str] = None

    def load_baseline(self, model_name: str) -> Tuple[bool, str]:
        """
        Load a baseline (untrained) model.

        Args:
            model_name: HuggingFace model name (e.g., 'gpt2', 'distilgpt2')

        Returns:
            Tuple of (success, message)
        """
        try:
            self.status = ModelStatus.LOADING
            self.error_message = None

            from constitutional_ai import load_model

            self.model, self.tokenizer = load_model(model_name)
            self.model_name = model_name
            self.model_type = "baseline"
            self.status = ModelStatus.READY

            return True, f"✅ Loaded baseline model: {model_name}"

        except Exception as e:
            self.status = ModelStatus.ERROR
            self.error_message = str(e)
            return False, f"❌ Error loading model: {e}"

    def load_constitutional(self, checkpoint_path: str) -> Tuple[bool, str]:
        """
        Load a pre-trained constitutional AI model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Tuple of (success, message)
        """
        try:
            self.status = ModelStatus.LOADING
            self.error_message = None

            checkpoint_dir = Path(checkpoint_path)
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            # Load model from checkpoint
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

            # Set padding token if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Move to device
            self.model = self.model.to(self.device)

            self.model_name = checkpoint_dir.name
            self.model_type = "constitutional"
            self.status = ModelStatus.READY

            return True, f"✅ Loaded constitutional model: {self.model_name}"

        except Exception as e:
            self.status = ModelStatus.ERROR
            self.error_message = str(e)
            return False, f"❌ Error loading checkpoint: {e}"

    def unload(self):
        """Unload the current model and free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()

        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.model_type = None
        self.status = ModelStatus.NOT_LOADED
        self.error_message = None

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.status == ModelStatus.READY and self.model is not None

    def get_status_info(self) -> dict:
        """
        Get current status information.

        Returns:
            Dictionary with status details
        """
        return {
            "status": self.status.value,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": str(self.device),
            "is_loaded": self.is_loaded(),
            "error": self.error_message,
        }

    def get_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Get the current model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            ValueError: If no model is loaded
        """
        if not self.is_loaded():
            raise ValueError("No model loaded. Please load a model first.")

        return self.model, self.tokenizer

    def set_training_mode(self):
        """Set status to training."""
        if self.is_loaded():
            self.status = ModelStatus.TRAINING

    def set_ready_mode(self):
        """Set status back to ready after training."""
        if self.model is not None:
            self.status = ModelStatus.READY
