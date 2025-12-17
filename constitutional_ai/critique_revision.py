"""MODULE: critique_revision.py
PURPOSE: Critique-Revision Cycle for Constitutional AI (Phase 1: Supervised Learning)
KEY COMPONENTS:
- generate_critique: Generate constitutional critique of responses
- generate_revision: Generate improved responses addressing critiques
- critique_revision_pipeline: Complete pipeline for dataset generation
- ConstitutionalDataset: PyTorch dataset for supervised fine-tuning
- supervised_finetune: Train model on critique-revised responses
DEPENDENCIES: torch, transformers, tqdm, typing, framework, model_utils
SPECIAL NOTES: Implements Phase 1 of Constitutional AI methodology from Anthropic (2022)
"""

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from .framework import ConstitutionalFramework
from .model_utils import GenerationConfig, generate_text
from .principles import get_eval_debug_level, set_eval_debug_level

# Module logger (prefixed with _ to avoid shadowing the logger parameter in functions)
_logger = logging.getLogger(__name__)


def _get_model_name(model: PreTrainedModel) -> str:
    """Extract model name from model config if available."""
    try:
        if hasattr(model, "config") and hasattr(model.config, "name_or_path"):
            name = model.config.name_or_path
            # Shorten common model names
            if "phi-2" in name.lower():
                return "Phi-2"
            if "phi-3" in name.lower():
                return "Phi-3"
            if "gpt2" in name.lower():
                return "GPT-2"
            if "qwen" in name.lower():
                return "Qwen"
            if "mistral" in name.lower():
                return "Mistral"
            if "tinyllama" in name.lower():
                return "TinyLlama"
            return name.split("/")[-1][:20]  # Last part, truncated
    except Exception:
        pass
    return "Unknown"


def _log_section_header(title: str) -> None:
    """Print a section header."""
    _logger.info("─" * 70)
    _logger.info(f"│ {title}")
    _logger.info(f"{'─' * 70}")


def _print_content(label: str, content: str, role: str = "") -> None:
    """Print content with label and optional role attribution."""
    _log_section_header(f"{label} [{role}]" if role else label)
    _logger.info(content)


def _print_evaluation_box(
    text_preview: str,
    violations: list[str],
    score: float,
    phase: str = "INITIAL",
    eval_model_name: str = "Evaluation Model",
    method: str = "",
) -> None:
    """
    Print evaluation interaction in a clear boxed format.
    Shows the input to the evaluation model and its output.
    """
    _logger.info(f"{'┌' + '─' * 68 + '┐'}")
    _logger.info(f"│ EVALUATION ({phase}) → {eval_model_name:<40} │")
    _logger.info(f"{'├' + '─' * 68 + '┤'}")

    # Show what we're evaluating (truncated)
    preview = text_preview[:60] + "..." if len(text_preview) > 60 else text_preview
    preview = preview.replace("\n", " ")
    _logger.info(f'│ Input:  "{preview}"')
    _logger.info(f"{'├' + '─' * 68 + '┤'}")

    # Show result
    if not violations:
        _logger.info(f"│ Result: ✓ CLEAN (Score: {score:.2f})")
    else:
        _logger.info(f"│ Result: ⚠ {len(violations)} VIOLATION(S) (Score: {score:.2f})")
        _logger.info(f"│         → {', '.join(violations)}")

    if method:
        _logger.info(f"│ Method: {method}")

    _logger.info(f"{'└' + '─' * 68 + '┘'}")


# Prompt templates from Anthropic's Constitutional AI paper
CRITIQUE_TEMPLATE = """Consider the following conversation between a human and an assistant:
Human: {prompt}

Assistant: {response}

Identify any ways in which the assistant's response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal.

Specifically, evaluate the response against these principles:
{principles_text}

Critique:
"""

REVISION_TEMPLATE = """Consider the following conversation:

Human: {prompt}

Assistant: {response}

A critique of this response identified the following issues:
{critique}

Please rewrite the assistant's response to address these issues while still being helpful to the human.

Revised Response:
"""


def generate_critique(
    prompt: str,
    response: str,
    principles: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    logger=None,  # type: ignore
) -> str:
    """
    Generate constitutional critique of a response.

    Args:
        prompt: Original user prompt
        response: Model's response to critique
        principles: List of constitutional principles to evaluate
        model: Language model for generation
        tokenizer: Model tokenizer
        device: Computation device
        logger: Optional ContentLogger for pipeline visibility

    Returns:
        Critique text identifying violations
    """
    # Format principles
    principles_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(principles)])

    # Build critique prompt
    critique_prompt = CRITIQUE_TEMPLATE.format(
        prompt=prompt, response=response, principles_text=principles_text
    )

    # Store in logger silently (don't display - content is displayed by caller)
    if logger:
        logger.log_stage("CRITIQUE-PROMPT", critique_prompt, truncate=400, silent=True)

    # Generate critique using model
    # FIX: Use max_new_tokens to avoid probability tensor errors
    config = GenerationConfig(max_new_tokens=256, temperature=0.7, do_sample=True)

    try:
        # PERFORMANCE: Use torch.no_grad() for inference-only operations
        # Expected speedup: 10-30%, memory reduction: ~50%
        with torch.no_grad():
            critique = generate_text(model, tokenizer, critique_prompt, config, device)
        # Handle empty responses
        if not critique or critique.strip() == "":
            critique = "No specific issues identified."

        if logger:
            logger.log_stage("CRITIQUE-GENERATION", critique, silent=True)

        return critique
    except Exception as e:
        # Catch all exceptions to ensure graceful degradation
        if logger:
            logger.log_stage("CRITIQUE-ERROR", f"Critique generation failed: {e}")
        _logger.info(f"Warning: Critique generation failed: {e}")
        return "Error generating critique."


def generate_revision(
    prompt: str,
    response: str,
    critique: str,
    principles: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    logger=None,  # type: ignore
) -> str:
    """
    Generate revised response based on critique.

    Args:
        prompt: Original user prompt
        response: Original response that was critiqued
        critique: Critique identifying issues
        principles: Constitutional principles
        model: Language model for generation
        tokenizer: Model tokenizer
        device: Computation device
        logger: Optional ContentLogger for pipeline visibility

    Returns:
        Revised response addressing critique
    """
    revision_prompt = REVISION_TEMPLATE.format(prompt=prompt, response=response, critique=critique)

    if logger:
        logger.log_stage("REVISION-PROMPT", revision_prompt, truncate=400, silent=True)

    # FIX: Use max_new_tokens to avoid probability tensor errors
    config = GenerationConfig(max_new_tokens=256, temperature=0.7, do_sample=True)

    try:
        # PERFORMANCE: Use torch.no_grad() for inference-only operations
        # Expected speedup: 10-30%, memory reduction: ~50%
        with torch.no_grad():
            revision = generate_text(model, tokenizer, revision_prompt, config, device)
        # Handle empty responses - fall back to original
        if not revision or revision.strip() == "":
            revision = response

        if logger:
            logger.log_stage("REVISION-GENERATION", revision, silent=True)

        return revision
    except Exception as e:
        # Catch all exceptions to ensure graceful degradation
        if logger:
            logger.log_stage("REVISION-ERROR", f"Revision generation failed: {e}, using original")
        _logger.info(f"Warning: Revision generation failed: {e}")
        return response  # Fall back to original


def critique_revision_pipeline(
    prompts: list[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    framework: ConstitutionalFramework,
    device: torch.device,
    num_revisions: int = 1,
    logger=None,  # type: ignore
    collect_preference_pairs: bool = True,
) -> dict[str, Any]:
    """
    Complete critique-revision pipeline for dataset generation.

    Args:
        prompts: List of prompts to generate data for
        model: Language model
        tokenizer: Tokenizer
        framework: Constitutional framework with principles
        device: Computation device
        num_revisions: Number of critique-revision iterations
        logger: Optional ContentLogger for pipeline visibility
        collect_preference_pairs: If True, also collect preference pairs for RLAIF

    Returns:
        Dictionary containing:
        - training_data: List of training examples for SFT (Phase 1)
        - preference_pairs: List of (chosen, rejected) pairs for reward model (Phase 2)
        - stats: Pipeline statistics
    """
    training_data = []
    preference_pairs = []  # NEW: For RLAIF Phase 2
    principles = [p.description for p in framework.principles.values()]
    gen_model_name = _get_model_name(model)
    eval_model_name = _get_model_name(framework.model) if framework.model else "Regex"

    # Suppress verbose evaluation debug output during pipeline (we have our own display)
    original_debug_level = get_eval_debug_level()
    set_eval_debug_level(0)

    # Print pipeline configuration
    _logger.info(f"{'═' * 70}")
    _logger.info("  CONSTITUTIONAL AI TRAINING PIPELINE")
    _logger.info(f"{'═' * 70}")
    _logger.info(f"  Generation Model: {gen_model_name}")
    _logger.info(f"  Evaluation Model: {eval_model_name}")
    _logger.info(f"  Prompts: {len(prompts)} | Revisions per prompt: {num_revisions}")
    _logger.info(f"{'═' * 70}\n")

    for idx, prompt in enumerate(tqdm(prompts, desc="Generating revised responses")):
        # Print example header
        _logger.info(f"{'━' * 70}")
        _logger.info(f"  EXAMPLE {idx + 1}/{len(prompts)}")
        _logger.info(f"{'━' * 70}")

        # Print full prompt
        _print_content("PROMPT", prompt)

        # Store silently (already displayed above)
        if logger:
            logger.log_stage(
                f"TRAINING-EXAMPLE {idx + 1}/{len(prompts)}", f"Prompt: {prompt}", silent=True
            )

        config = GenerationConfig(max_new_tokens=150, temperature=1.0, do_sample=True)

        try:
            with torch.no_grad():
                response = generate_text(model, tokenizer, prompt, config, device)

            # Print full initial response
            _print_content("1. INITIAL RESPONSE", response, role="Generation Model")
            if logger:
                logger.log_stage("INITIAL-GENERATION", response, silent=True)

            # Store initial response for preference pairs BEFORE revision
            initial_response = response

            # Evaluate initial response (with clear evaluation model display)
            initial_score = framework.evaluate_text(response)
            violations = initial_score.get("flagged_principles", [])
            weighted_score = initial_score.get("weighted_score", 0.0)
            eval_method = initial_score.get("evaluation_method", "unknown")
            _print_evaluation_box(
                response,
                violations,
                weighted_score,
                phase="INITIAL",
                eval_model_name=eval_model_name,
                method=eval_method,
            )

            if logger:
                logger.log_stage(
                    "INITIAL-EVALUATION",
                    f"Violations: {violations}\nWeighted score: {weighted_score:.2f}",
                    metadata={"violations": violations, "score": weighted_score},
                    silent=True,
                )

            # Iterative critique and revision
            for _iteration in range(num_revisions):
                critique = generate_critique(
                    prompt, response, principles, model, tokenizer, device, logger=logger
                )
                # Print full critique
                _print_content("2. CRITIQUE", critique, role="Generation Model")

                response = generate_revision(
                    prompt, response, critique, principles, model, tokenizer, device, logger=logger
                )
                # Print full revised response
                _print_content("3. REVISED RESPONSE", response, role="Generation Model")

            # Evaluate revised response (with clear evaluation model display)
            revised_score = framework.evaluate_text(response)
            initial_weighted_score = initial_score.get("weighted_score", 0.0)
            revised_weighted_score = revised_score.get("weighted_score", 0.0)
            improvement = initial_weighted_score - revised_weighted_score
            revised_violations = revised_score.get("flagged_principles", [])
            revised_method = revised_score.get("evaluation_method", "unknown")

            _print_evaluation_box(
                response,
                revised_violations,
                revised_weighted_score,
                phase="REVISED",
                eval_model_name=eval_model_name,
                method=revised_method,
            )

            if logger:
                logger.log_stage(
                    "REVISION-EVALUATION",
                    f"Violations: {revised_violations if revised_violations else 'NONE'}\n"
                    f"Weighted score: {revised_weighted_score:.2f}",
                    metadata={
                        "violations": revised_violations,
                        "score": revised_weighted_score,
                        "improvement": improvement,
                    },
                    silent=True,
                )

            # Print improvement summary
            if improvement > 0:
                _logger.info(
                    f"  ✓ IMPROVEMENT: {initial_weighted_score:.2f} → {revised_weighted_score:.2f} ({improvement:+.2f})"
                )
                _logger.info("  → Added to training set")
                training_data.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "num_revisions": num_revisions,
                        "improvement": improvement,
                    }
                )

                # NEW: Collect preference pairs for RLAIF Phase 2
                # The revised response is "chosen", initial response is "rejected"
                if collect_preference_pairs:
                    preference_pairs.append(
                        {
                            "prompt": prompt,
                            "chosen": response,  # Revised (better)
                            "rejected": initial_response,  # Initial (worse)
                            "chosen_score": revised_weighted_score,
                            "rejected_score": initial_weighted_score,
                            "margin": improvement,
                        }
                    )

                if logger:
                    logger.log_stage(
                        "TRAINING-PAIR-CREATED",
                        f"✓ Training example added\n"
                        f"  Improvement: {initial_weighted_score:.2f} → "
                        f"{revised_weighted_score:.2f} ({improvement:.2f} reduction)",
                        silent=True,
                    )
            else:
                _logger.info(
                    f"  ✗ NO IMPROVEMENT: {initial_weighted_score:.2f} → {revised_weighted_score:.2f} ({improvement:+.2f})"
                )
                _logger.info("  → Skipped")
                if logger:
                    logger.log_stage(
                        "TRAINING-PAIR-SKIPPED",
                        f"✗ Training example skipped (improvement: {improvement:+.2f})\n"
                        f"  Score: {initial_weighted_score:.2f} → {revised_weighted_score:.2f}",
                        silent=True,
                    )
        except Exception as e:
            # Catch all exceptions to ensure graceful degradation
            if logger:
                logger.log_stage("TRAINING-EXAMPLE-ERROR", f"Failed: {e}")
            _logger.info(f"Warning: Failed to process prompt '{prompt[:50]}...': {e}")
            continue

    # Print final summary
    _logger.info(f"{'═' * 70}")
    _logger.info("  PIPELINE SUMMARY")
    _logger.info(f"{'═' * 70}")
    _logger.info(f"  Total prompts processed: {len(prompts)}")
    _logger.info(f"  Training examples generated: {len(training_data)}")
    _logger.info(f"  Preference pairs collected: {len(preference_pairs)}")
    _logger.info(f"  Examples skipped: {len(prompts) - len(training_data)}")
    if training_data:
        avg_improvement = sum(d["improvement"] for d in training_data) / len(training_data)
        _logger.info(f"  Average improvement: {avg_improvement:.2f}")
    if preference_pairs:
        avg_margin = sum(p["margin"] for p in preference_pairs) / len(preference_pairs)
        _logger.info(f"  Average preference margin: {avg_margin:.2f}")
    _logger.info(f"{'═' * 70}\n")

    # Restore original debug level
    set_eval_debug_level(original_debug_level)

    # Build stats
    stats = {
        "total_prompts": len(prompts),
        "training_examples": len(training_data),
        "preference_pairs": len(preference_pairs),
        "skipped": len(prompts) - len(training_data),
        "avg_improvement": (
            sum(d["improvement"] for d in training_data) / len(training_data)
            if training_data
            else 0.0
        ),
        "avg_margin": (
            sum(p["margin"] for p in preference_pairs) / len(preference_pairs)
            if preference_pairs
            else 0.0
        ),
    }

    return {"training_data": training_data, "preference_pairs": preference_pairs, "stats": stats}


class ConstitutionalDataset(Dataset):
    """
    PyTorch dataset for Constitutional AI supervised fine-tuning.

    Wraps critique-revised training data for use with PyTorch DataLoader.
    """

    def __init__(self, data: list[dict[str, Any]], tokenizer, max_length: int = 512):
        """
        Initialize dataset.

        Args:
            data: List of training examples from critique_revision_pipeline
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single training example.

        Args:
            idx: Index of example

        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        item = self.data[idx]
        text = item["prompt"] + item["response"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }


def supervised_finetune(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    training_data: list[dict[str, Any]],
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    device: torch.device = None,
    use_amp: bool = True,
) -> dict[str, Any]:
    """
    Fine-tune model on critique-revised responses.

    Args:
        model: Base model to fine-tune
        tokenizer: Tokenizer
        training_data: Data from critique-revision pipeline
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Computation device
        use_amp: Use automatic mixed precision for faster training (default: True)

    Returns:
        Training metrics and fine-tuned model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate training data
    if not training_data or len(training_data) == 0:
        raise ValueError("Training data is empty. Cannot train model.")

    # Filter out invalid training examples
    valid_data = []
    for idx, item in enumerate(training_data):
        # Check if required fields exist and are non-empty
        # Support both 'response' and 'revised_response' for backward compatibility
        if "prompt" not in item:
            _logger.info(f"Warning: Skipping training example {idx}: missing prompt")
            continue

        # Accept either 'response' or 'revised_response'
        response_key = None
        if "response" in item:
            response_key = "response"
        elif "revised_response" in item:
            response_key = "revised_response"
        else:
            _logger.info(
                f"Warning: Skipping training example {idx}: missing response or revised_response"
            )
            continue

        prompt = item.get("prompt", "").strip()
        response = item.get(response_key, "").strip()

        if not prompt or not response:
            _logger.info(f"Warning: Skipping training example {idx}: empty prompt or response")
            continue

        # Check for NaN or None values
        if prompt == "nan" or response == "nan" or prompt == "None" or response == "None":
            _logger.info(f"Warning: Skipping training example {idx}: NaN or None value detected")
            continue

        # Normalize to 'response' key for consistency
        if response_key == "revised_response":
            item["response"] = item["revised_response"]

        valid_data.append(item)

    if not valid_data:
        raise ValueError(f"All {len(training_data)} training examples are invalid. Cannot train.")

    _logger.info(f"Using {len(valid_data)}/{len(training_data)} valid training examples")

    model = model.to(device)
    model.train()

    # Create dataset and dataloader
    dataset = ConstitutionalDataset(valid_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # PERFORMANCE: Initialize GradScaler for Automatic Mixed Precision (AMP)
    # Expected speedup: 2-3x on compatible GPUs, reduced memory usage
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    # Training loop
    metrics = {"losses": [], "epochs": []}

    for epoch in range(num_epochs):
        epoch_loss = 0
        batch_count = 0
        nan_batches = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Check for NaN in input tensors
                if (
                    torch.isnan(input_ids.float()).any()
                    or torch.isnan(attention_mask.float()).any()
                ):
                    _logger.info(
                        f"Warning: NaN detected in batch {batch_idx} input tensors, skipping"
                    )
                    nan_batches += 1
                    continue

                # PERFORMANCE: Use automatic mixed precision for forward pass
                # autocast() automatically handles float16/float32 conversions
                with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
                    )
                    loss = outputs.loss

                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    _logger.info(f"Warning: NaN/Inf loss detected in batch {batch_idx}, skipping")
                    nan_batches += 1
                    continue

                # PERFORMANCE: Use GradScaler for backward pass with AMP
                # Scales loss to prevent gradient underflow in float16
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # Check for NaN gradients before clipping
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and (
                        torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                    ):
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    _logger.info(
                        f"Warning: NaN/Inf gradient detected in batch {batch_idx}, skipping"
                    )
                    nan_batches += 1
                    optimizer.zero_grad()
                    continue

                # Gradient clipping to prevent explosion (unscales gradients first)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update weights with scaler
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                batch_count += 1

            except (RuntimeError, ValueError, TypeError) as e:
                _logger.info(f"Warning: Error processing batch {batch_idx}: {e}")
                nan_batches += 1
                continue

        if batch_count == 0:
            _logger.info(f"ERROR: Epoch {epoch+1} - All batches were invalid or produced NaN")
            # Still record the epoch with 0 loss
            metrics["losses"].append(0.0)
            metrics["epochs"].append(epoch + 1)
        else:
            avg_loss = epoch_loss / batch_count
            metrics["losses"].append(avg_loss)
            metrics["epochs"].append(epoch + 1)
            _logger.info(
                f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f} ({batch_count} batches, {nan_batches} skipped)"
            )

    return {"model": model, "metrics": metrics}
