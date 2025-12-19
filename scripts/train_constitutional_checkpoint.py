#!/usr/bin/env python3
"""
Train a Constitutional AI model and save checkpoint for demo.

This script trains a model with Constitutional AI and saves it as a checkpoint
that can be loaded in the Gradio demo for before/after comparisons.

Usage:
    # Quick training (2 epochs, ~15-20 min)
    python scripts/train_constitutional_checkpoint.py --model distilgpt2 --mode quick

    # Standard training (5 epochs, ~45-60 min)
    python scripts/train_constitutional_checkpoint.py --model gpt2 --mode standard

    # Custom configuration
    python scripts/train_constitutional_checkpoint.py \
        --model gpt2 \
        --epochs 3 \
        --num-prompts 20 \
        --output demos/checkpoints/gpt2-constitutional-3epochs
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch

from constitutional_ai import (
    load_model,
    setup_default_framework,
)
from constitutional_ai.critique_revision import critique_revision_pipeline
from constitutional_ai.trainer import supervised_finetune
from constitutional_ai.config import get_default_config, get_lightweight_config


def get_training_prompts(mode: str) -> list[str]:
    """Get training prompts based on mode."""
    # Quick mode prompts (10 prompts)
    quick_prompts = [
        "How can I help someone who is feeling sad?",
        "What are the steps to resolve a conflict peacefully?",
        "What is the current scientific consensus on climate change?",
        "Explain how vaccines work.",
        "Describe the contributions of various cultures to mathematics.",
        "What factors contribute to success in a career?",
        "What are some options for career planning?",
        "How might someone make an informed decision about their health?",
        "What are the ethical considerations in AI development?",
        "How should we balance individual rights with collective welfare?",
    ]

    # Standard mode prompts (30 prompts)
    standard_prompts = quick_prompts + [
        "What are the best practices for conflict resolution?",
        "How do I ensure my work is scientifically rigorous?",
        "What strategies promote fairness in hiring?",
        "How can I respect others' personal boundaries?",
        "What are the principles of ethical AI development?",
        "How do we create inclusive communities?",
        "What factors influence good decision-making?",
        "How can I communicate effectively across cultures?",
        "What are the fundamentals of critical thinking?",
        "How do I build trust in professional relationships?",
        "What are the challenges in balancing privacy and security?",
        "How do we address historical injustices?",
        "What role does context play in ethical judgments?",
        "How can technology be used responsibly?",
        "What are the trade-offs in policy decisions?",
        "How can I support someone going through a difficult time?",
        "What are effective approaches to learning new skills?",
        "How do different cultures approach problem-solving?",
        "What makes communication clear and effective?",
        "How can we promote equity in education?",
    ]

    return quick_prompts if mode == "quick" else standard_prompts


def main():
    parser = argparse.ArgumentParser(
        description="Train Constitutional AI model and save checkpoint"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        choices=["distilgpt2", "gpt2", "gpt2-medium"],
        help="Base model to train",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "standard", "custom"],
        help="Training mode (quick=2 epochs/10 prompts, standard=5 epochs/30 prompts)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs (for custom mode)"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=None, help="Number of prompts (for custom mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for checkpoint (default: demos/checkpoints/{model}-constitutional)",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")

    args = parser.parse_args()

    # Determine configuration
    if args.mode == "custom":
        if args.epochs is None or args.num_prompts is None:
            parser.error("--epochs and --num-prompts required for custom mode")
        num_epochs = args.epochs
        prompts = get_training_prompts("standard")[: args.num_prompts]
    elif args.mode == "quick":
        num_epochs = 2
        prompts = get_training_prompts("quick")
    else:  # standard
        num_epochs = 5
        prompts = get_training_prompts("standard")

    # Determine output path
    if args.output:
        output_dir = Path(args.output)
    else:
        suffix = f"-{args.mode}" if args.mode != "custom" else f"-{num_epochs}epochs"
        output_dir = Path(f"demos/checkpoints/{args.model}-constitutional{suffix}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONSTITUTIONAL AI CHECKPOINT TRAINING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Base model: {args.model}")
    print(f"  Mode: {args.mode}")
    print(f"  Training prompts: {len(prompts)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output: {output_dir}")
    print(f"\nEstimated time: ", end="")
    if args.mode == "quick":
        print("15-20 minutes")
    elif args.mode == "standard":
        print("45-60 minutes")
    else:
        print(f"~{len(prompts) * num_epochs * 0.5} minutes")

    input("\nPress Enter to start training (or Ctrl+C to cancel)...")

    start_time = time.time()

    # Step 1: Load base model
    print("\n[1/4] Loading base model...")
    model, tokenizer = load_model(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Step 2: Setup framework
    print("\n[2/4] Setting up Constitutional AI framework...")
    framework = setup_default_framework()
    print(f"  Principles: {len(framework.principles)}")

    # Step 3: Generate critique-revision training data
    print(f"\n[3/4] Generating critique-revision pairs ({len(prompts)} prompts)...")
    print("  This phase generates critiques and improved responses for each prompt.")

    training_data = critique_revision_pipeline(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        framework=framework,
        num_epochs=1,  # One pass to generate data
        device=device,
    )

    print(f"  ✓ Generated {len(training_data)} training examples")

    # Step 4: Supervised fine-tuning
    print(f"\n[4/4] Supervised fine-tuning ({num_epochs} epochs)...")
    print("  Training model on constitutional critique-revised responses...")

    train_prompts = [item["prompt"] for item in training_data]
    train_responses = [item["revised_response"] for item in training_data]

    metrics = supervised_finetune(
        model=model,
        tokenizer=tokenizer,
        prompts=train_prompts,
        responses=train_responses,
        num_epochs=num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
    )

    training_time = time.time() - start_time

    print(f"\n  ✓ Training complete!")
    print(f"  Final loss: {metrics.get('final_loss', 'N/A')}")
    print(f"  Training time: {training_time:.1f}s ({training_time/60:.1f} min)")

    # Step 5: Save checkpoint
    print(f"\n[5/5] Saving checkpoint to {output_dir}...")

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training info
    training_info = {
        "base_model": args.model,
        "mode": args.mode,
        "num_epochs": num_epochs,
        "num_prompts": len(prompts),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "training_time_seconds": training_time,
        "final_loss": metrics.get("final_loss"),
        "date": datetime.now().isoformat(),
        "device": str(device),
    }

    with open(output_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    print("  ✓ Checkpoint saved")

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Constitutional AI checkpoint saved to: {output_dir}")
    print(f"\n📊 Training Summary:")
    print(f"   - Base model: {args.model}")
    print(f"   - Training examples: {len(training_data)}")
    print(f"   - Epochs: {num_epochs}")
    print(f"   - Final loss: {metrics.get('final_loss', 'N/A'):.4f}")
    print(f"   - Training time: {training_time/60:.1f} minutes")
    print(f"\n🎯 Next Steps:")
    print(f"   1. Test the checkpoint:")
    print(f"      python examples/quickstart.py --checkpoint {output_dir}")
    print(f"   2. Use in Gradio demo:")
    print(f"      - Run: python demos/gradio_demo.py")
    print(f"      - Go to 'Comparison' tab")
    print(f"      - Load this checkpoint for before/after comparison")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
