"""Mocks for transformers library components."""

from unittest.mock import Mock

import torch


def create_mock_tokenizer(pad_token="<pad>", eos_token="<eos>"):
    """
    Create a mock tokenizer with standard behavior.

    Args:
        pad_token: Padding token string
        eos_token: End-of-sequence token string

    Returns:
        Mock tokenizer with standard attributes and methods

    Usage:
        tokenizer = create_mock_tokenizer()
        tokenizer.pad_token  # "<pad>"
        tokenizer.decode(ids)  # "generated text"
    """
    tokenizer = Mock()
    tokenizer.pad_token = pad_token
    tokenizer.eos_token = eos_token
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    # Mock encoding behavior
    def call_side_effect(text, **kwargs):
        """Mock tokenizer call returns standard encoded dict."""
        return {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

    tokenizer.side_effect = call_side_effect
    tokenizer.return_value = call_side_effect(None)

    # Mock decode behavior
    tokenizer.decode.return_value = "generated text"

    # Mock from_pretrained for patching
    tokenizer.from_pretrained = Mock(return_value=tokenizer)

    return tokenizer


def create_mock_model(hidden_size=768, config_style="modern"):
    """
    Create a mock language model.

    Args:
        hidden_size: Model hidden size
        config_style: Config attribute style
            - "modern": Uses config.hidden_size
            - "gpt2": Uses config.n_embd
            - "no_config": No config attribute (will raise AttributeError)

    Returns:
        Mock model with config, parameters, generate, etc.

    Usage:
        # Modern model (BERT, GPT-3 style)
        model = create_mock_model(hidden_size=768, config_style="modern")

        # GPT-2 style model
        model = create_mock_model(hidden_size=1024, config_style="gpt2")

        # Model without config
        model = create_mock_model(config_style="no_config")
    """
    model = Mock()

    # Config setup based on style
    if config_style == "modern":
        model.config = Mock()
        model.config.hidden_size = hidden_size
        model.config.n_embd = None
    elif config_style == "gpt2":
        model.config = Mock()
        model.config.n_embd = hidden_size
        model.config.hidden_size = None
    elif config_style == "no_config":
        # Delete config attribute to simulate models without config
        del model.config

    # Standard model methods
    def parameters_side_effect():
        """Mock parameters returns fresh generator each time."""
        yield torch.randn(10, 10)
        yield torch.randn(5, 5)

    model.parameters.side_effect = lambda: parameters_side_effect()

    # Mock generate returning token IDs
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    # Mock device methods
    model.to.return_value = model
    model.train.return_value = None
    model.eval.return_value = None

    # Mock state dict for checkpointing
    model.state_dict.return_value = {"weights": torch.randn(10, 10)}
    model.load_state_dict.return_value = None

    # Mock from_pretrained for patching
    model.from_pretrained = Mock(return_value=model)

    return model
