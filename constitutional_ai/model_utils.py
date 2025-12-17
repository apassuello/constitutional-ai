"""MODULE: model_utils.py
PURPOSE: Utilities for loading and using models with Constitutional AI
KEY COMPONENTS:
- load_model: Load pretrained models (GPT-2, etc.)
- generate_text: Generate text from models with tokenization
- batch_generate: Batch text generation
DEPENDENCIES: transformers, torch
SPECIAL NOTES: Provides model integration for constitutional training
"""

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 100  # FIX: Use max_new_tokens instead of max_length
    max_length: int | None = None  # Deprecated, kept for compatibility
    temperature: float = 1.0
    top_p: float = 1.0  # FIX: Disable top_p filtering (1.0 = no filtering)
    top_k: int = 0  # FIX: Disable top_k filtering (0 = no filtering)
    num_return_sequences: int = 1
    do_sample: bool = True
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    min_new_tokens: int | None = None  # Minimum tokens to generate


def load_model(
    model_name: str = "gpt2", device: torch.device | None = None, load_in_8bit: bool = False
):
    """
    Load a pretrained language model and tokenizer.

    Args:
        model_name: Name or path of model (e.g., 'gpt2', 'gpt2-medium')
        device: Device to load model on
        load_in_8bit: Whether to load in 8-bit mode for memory efficiency

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers library required. Install with: pip install transformers"
        ) from None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {}
    if load_in_8bit and device.type == "cuda":
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if not load_in_8bit:
        model = model.to(device)

    logger.info("Model loaded successfully")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    generation_config: GenerationConfig | None = None,
    device: torch.device | None = None,
) -> str:
    """
    Generate text from a prompt using the model.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        generation_config: Generation configuration
        device: Device for generation

    Returns:
        Generated text (without prompt)
    """
    if generation_config is None:
        generation_config = GenerationConfig()

    if device is None:
        device = next(model.parameters()).device

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get pad token id
    pad_token_id = generation_config.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id

    # Generate
    # FIX: Build generation kwargs, only passing parameters that are actually needed
    gen_kwargs = {
        "temperature": generation_config.temperature,
        "num_return_sequences": generation_config.num_return_sequences,
        "do_sample": generation_config.do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": generation_config.eos_token_id or tokenizer.eos_token_id,
        # FIX: Disable cache to avoid DynamicCache errors with Phi-3 and similar models
        "use_cache": False,
    }

    # Only pass top_p if it's actually doing filtering (< 1.0)
    if generation_config.top_p < 1.0:
        gen_kwargs["top_p"] = generation_config.top_p

    # Only pass top_k if it's actually doing filtering (> 0)
    # CRITICAL: Don't pass top_k=0 as some models interpret it as "use 0 tokens"!
    if generation_config.top_k > 0:
        gen_kwargs["top_k"] = generation_config.top_k

    # Use max_new_tokens (preferred) or fall back to max_length for compatibility
    if generation_config.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = generation_config.max_new_tokens
    elif generation_config.max_length is not None:
        gen_kwargs["max_length"] = generation_config.max_length
    else:
        gen_kwargs["max_new_tokens"] = 100  # Default

    if generation_config.min_new_tokens is not None:
        gen_kwargs["min_new_tokens"] = generation_config.min_new_tokens

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # Decode output
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_length:]  # Remove prompt
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return generated_text


def batch_generate(
    model,
    tokenizer,
    prompts: list[str],
    generation_config: GenerationConfig | None = None,
    batch_size: int = 4,
    device: torch.device | None = None,
    show_progress: bool = True,
) -> list[str]:
    """
    Generate text for multiple prompts in batches.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of input prompts
        generation_config: Generation configuration
        batch_size: Number of prompts per batch
        device: Device for generation
        show_progress: Whether to show progress bar

    Returns:
        List of generated texts
    """
    if generation_config is None:
        generation_config = GenerationConfig()

    if device is None:
        device = next(model.parameters()).device

    results = []

    # Import tqdm if available
    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Generating")
        except ImportError:
            pass

    for i in iterator:
        batch_prompts = prompts[i : i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get pad token id
        pad_token_id = generation_config.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id

        # Generate
        # FIX: Build generation kwargs, only passing parameters that are actually needed
        gen_kwargs = {
            "temperature": generation_config.temperature,
            "do_sample": generation_config.do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": generation_config.eos_token_id or tokenizer.eos_token_id,
            # FIX: Disable cache to avoid DynamicCache errors with Phi-3 and similar models
            "use_cache": False,
        }

        # Only pass top_p if it's actually doing filtering (< 1.0)
        if generation_config.top_p < 1.0:
            gen_kwargs["top_p"] = generation_config.top_p

        # Only pass top_k if it's actually doing filtering (> 0)
        # CRITICAL: Don't pass top_k=0 as some models interpret it as "use 0 tokens"!
        if generation_config.top_k > 0:
            gen_kwargs["top_k"] = generation_config.top_k

        if generation_config.max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = generation_config.max_new_tokens
        elif generation_config.max_length is not None:
            gen_kwargs["max_length"] = generation_config.max_length
        else:
            gen_kwargs["max_new_tokens"] = 100

        if generation_config.min_new_tokens is not None:
            gen_kwargs["min_new_tokens"] = generation_config.min_new_tokens

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode outputs
        prompt_lengths = inputs["input_ids"].ne(pad_token_id).sum(dim=1)
        for j, output in enumerate(outputs):
            prompt_len = prompt_lengths[j].item()
            generated_ids = output[prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(generated_text)

    return results


def prepare_model_for_training(model, learning_rate: float = 5e-5, weight_decay: float = 0.01):
    """
    Prepare model for training with appropriate optimizer.

    Args:
        model: Model to train
        learning_rate: Learning rate
        weight_decay: Weight decay

    Returns:
        Optimizer
    """
    # Enable gradient computation
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    return optimizer


def get_model_device(model) -> torch.device:
    """Get the device a model is on."""
    return next(model.parameters()).device
