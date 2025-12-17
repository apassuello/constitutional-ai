"""
Constitutional AI Real Training Demo

This demo demonstrates ACTUAL model fine-tuning with Constitutional AI:
1. Load a real pretrained model (GPT-2 or similar)
2. Generate baseline responses using the model
3. Fine-tune the model using RLAIF with constitutional feedback
4. Generate improved responses and compare
5. Visualize real training improvements

Usage:
    # Quick demo with GPT-2 (1 epoch, few samples)
    python demos/constitutional_ai_real_training_demo.py --quick_demo

    # Full training
    python demos/constitutional_ai_real_training_demo.py --model gpt2 --num_epochs 3 --num_prompts 20

    # Using smaller model for faster training
    python demos/constitutional_ai_real_training_demo.py --model distilgpt2 --quick_demo
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import json
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
# Path append removed - using installed package)))

# Import project modules
from constitutional_ai import (
    setup_default_framework,
    ConstitutionalSafetyEvaluator,
    RLAIFTrainer,
    load_model,
    generate_text,
    GenerationConfig,
)
from constitutional_ai.data_utils import create_default_prompts
from constitutional_ai.config import (
    get_default_config,
    get_lightweight_config,
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_training_prompts(num_prompts: int = 15) -> List[str]:
    """Get prompts for training."""
    all_prompts = create_default_prompts()

    # Add more diverse prompts
    additional_prompts = [
        "Explain the importance of honesty in relationships.",
        "What are effective ways to manage stress?",
        "How can technology improve education?",
        "Describe the role of empathy in communication.",
        "What are sustainable practices for daily life?",
    ]

    all_prompts.extend(additional_prompts)
    return all_prompts[:num_prompts]


def evaluate_model_responses(
    model,
    tokenizer,
    prompts: List[str],
    evaluator: ConstitutionalSafetyEvaluator,
    generation_config: GenerationConfig,
    device: torch.device,
    desc: str = "Evaluating"
) -> Dict[str, Any]:
    """
    Generate responses from model and evaluate them.

    Args:
        model: The model to generate responses
        tokenizer: Tokenizer for the model
        prompts: List of prompts
        evaluator: Constitutional evaluator
        generation_config: Generation configuration
        device: Device for generation
        desc: Description for progress bar

    Returns:
        Dictionary with responses, evaluations, and metrics
    """
    print(f"\n{desc}...")
    model.eval()

    responses = []
    evaluations = []

    with torch.no_grad():
        for prompt in tqdm(prompts, desc=desc):
            # Generate response
            response = generate_text(
                model,
                tokenizer,
                prompt,
                generation_config,
                device=device
            )

            # Remove prompt from response (model might include it)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            # Evaluate response
            eval_result = evaluator.evaluate(response)

            responses.append(response)
            evaluations.append(eval_result)

    # Compute metrics
    num_flagged = sum(1 for ev in evaluations if ev['flagged'])
    compliance_rate = 1.0 - (num_flagged / len(evaluations)) if evaluations else 1.0

    # Count violations by principle
    principle_violations = {}
    for ev in evaluations:
        flagged_principles = ev.get('direct_evaluation', {}).get('flagged_principles', [])
        for principle in flagged_principles:
            principle_violations[principle] = principle_violations.get(principle, 0) + 1

    print(f"\nResults:")
    print(f"  Compliance Rate: {compliance_rate:.1%}")
    print(f"  Flagged: {num_flagged}/{len(evaluations)}")
    if principle_violations:
        print(f"  Violations by principle:")
        for principle, count in principle_violations.items():
            print(f"    - {principle}: {count}")

    return {
        "responses": responses,
        "evaluations": evaluations,
        "num_flagged": num_flagged,
        "compliance_rate": compliance_rate,
        "principle_violations": principle_violations,
    }


def visualize_comparison(
    baseline_metrics: Dict[str, Any],
    finetuned_metrics: Dict[str, Any],
    training_history: Dict[str, Any],
    save_path: str = "constitutional_real_training_comparison.png"
):
    """
    Visualize training results.

    Args:
        baseline_metrics: Baseline model metrics
        finetuned_metrics: Fine-tuned model metrics
        training_history: Training history from trainer
        save_path: Path to save visualization
    """
    print(f"\nCreating visualization...")

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Compliance Rate Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Baseline', 'Fine-tuned']
    compliance_rates = [
        baseline_metrics['compliance_rate'],
        finetuned_metrics['compliance_rate']
    ]

    bars = ax1.bar(models, compliance_rates, color=['#ff6b6b', '#51cf66'])
    ax1.set_ylabel('Compliance Rate', fontsize=12)
    ax1.set_title('Constitutional Compliance', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.0])
    ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Target')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 2: Violations by Principle
    ax2 = fig.add_subplot(gs[0, 1])
    all_principles = set(
        list(baseline_metrics.get('principle_violations', {}).keys()) +
        list(finetuned_metrics.get('principle_violations', {}).keys())
    )

    if all_principles:
        principles = sorted(list(all_principles))
        baseline_violations = [baseline_metrics.get('principle_violations', {}).get(p, 0) for p in principles]
        finetuned_violations = [finetuned_metrics.get('principle_violations', {}).get(p, 0) for p in principles]

        x = np.arange(len(principles))
        width = 0.35

        ax2.bar(x - width/2, baseline_violations, width, label='Baseline', color='#ff6b6b', alpha=0.8)
        ax2.bar(x + width/2, finetuned_violations, width, label='Fine-tuned', color='#51cf66', alpha=0.8)

        ax2.set_xlabel('Principle', fontsize=12)
        ax2.set_ylabel('Violations', fontsize=12)
        ax2.set_title('Violations by Principle', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([p.replace('_', '\n') for p in principles], fontsize=9)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Training Loss
    ax3 = fig.add_subplot(gs[0, 2])
    if 'epoch_losses' in training_history and training_history['epoch_losses']:
        epochs = range(1, len(training_history['epoch_losses']) + 1)
        ax3.plot(epochs, training_history['epoch_losses'],
                marker='o', linewidth=2, markersize=8, color='#339af0')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss', fontsize=12)
        ax3.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax3.grid(alpha=0.3)

    # Plot 4: Training Rewards
    ax4 = fig.add_subplot(gs[1, 0])
    if 'epoch_rewards' in training_history and training_history['epoch_rewards']:
        epochs = range(1, len(training_history['epoch_rewards']) + 1)
        ax4.plot(epochs, training_history['epoch_rewards'],
                marker='s', linewidth=2, markersize=8, color='#51cf66')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Avg Reward', fontsize=12)
        ax4.set_title('Constitutional Rewards (higher is better)', fontsize=14, fontweight='bold')
        ax4.grid(alpha=0.3)

    # Plot 5: Flagged Responses Count
    ax5 = fig.add_subplot(gs[1, 1])
    models = ['Baseline', 'Fine-tuned']
    flagged_counts = [
        baseline_metrics['num_flagged'],
        finetuned_metrics['num_flagged']
    ]

    bars = ax5.bar(models, flagged_counts, color=['#ff6b6b', '#51cf66'])
    ax5.set_ylabel('Flagged Responses', fontsize=12)
    ax5.set_title('Safety Violations Count', fontsize=14, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 6: Improvement Summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    improvement = finetuned_metrics['compliance_rate'] - baseline_metrics['compliance_rate']
    reduction = baseline_metrics['num_flagged'] - finetuned_metrics['num_flagged']

    summary_text = f"""
    Training Summary

    Baseline:
      • Compliance: {baseline_metrics['compliance_rate']:.1%}
      • Violations: {baseline_metrics['num_flagged']}

    Fine-tuned:
      • Compliance: {finetuned_metrics['compliance_rate']:.1%}
      • Violations: {finetuned_metrics['num_flagged']}

    Improvement:
      • Compliance: {improvement:+.1%}
      • Reduction: {reduction:+d} violations
    """

    ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Constitutional AI Real Training Demo')
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Model to use (gpt2, distilgpt2, etc.)')
    parser.add_argument('--num_epochs', type=int, default=2,
                       help='Number of training epochs')
    parser.add_argument('--num_prompts', type=int, default=15,
                       help='Number of training prompts')
    parser.add_argument('--num_responses_per_prompt', type=int, default=3,
                       help='Response candidates per prompt for RLAIF')
    parser.add_argument('--quick_demo', action='store_true',
                       help='Run quick demo (1 epoch, 5 prompts)')
    parser.add_argument('--output_dir', type=str, default='output/constitutional_real_training',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--load_in_8bit', action='store_true',
                       help='Load model in 8-bit mode (requires bitsandbytes)')

    args = parser.parse_args()

    # Adjust for quick demo
    if args.quick_demo:
        args.num_epochs = 1
        args.num_prompts = 5
        args.num_responses_per_prompt = 2
        print("\n🚀 Quick demo mode: 1 epoch, 5 prompts, 2 responses per prompt\n")

    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("Constitutional AI - REAL MODEL TRAINING DEMO")
    print("="*70)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Step 1: Load Model
    print(f"\n[Step 1] Loading pretrained model: {args.model}")
    print("  (This may take a moment...)")

    try:
        model, tokenizer = load_model(
            model_name=args.model,
            device=device,
            load_in_8bit=args.load_in_8bit
        )
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nTroubleshooting:")
        print("  - Install transformers: pip install transformers")
        print("  - For 8-bit loading: pip install bitsandbytes")
        print("  - Try a smaller model: --model distilgpt2")
        return

    # Step 2: Setup Constitutional Framework
    print("\n[Step 2] Setting up Constitutional AI framework...")
    framework = setup_default_framework()
    evaluator = ConstitutionalSafetyEvaluator(
        framework=framework,
        critique_model=None,  # Use direct evaluation only for speed
        use_self_critique=False
    )
    print(f"  Active principles: {framework.get_active_principles()}")

    # Step 3: Get Training Prompts
    print(f"\n[Step 3] Loading training prompts...")
    training_prompts = get_training_prompts(args.num_prompts)
    print(f"  Loaded {len(training_prompts)} prompts")

    # Step 4: Baseline Evaluation
    print("\n[Step 4] Baseline Evaluation")
    print("-" * 70)

    generation_config = GenerationConfig(
        max_length=150,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )

    baseline_metrics = evaluate_model_responses(
        model,
        tokenizer,
        training_prompts,
        evaluator,
        generation_config,
        device,
        desc="Generating baseline responses"
    )

    # Step 5: Constitutional AI Training
    print("\n[Step 5] Constitutional AI Training with RLAIF")
    print("-" * 70)
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Responses per prompt: {args.num_responses_per_prompt}")
    print(f"  Training prompts: {len(training_prompts)}")
    print()

    # Create RLAIF trainer
    trainer = RLAIFTrainer(
        policy_model=model,
        constitutional_framework=framework,
        critique_model=None,  # Use same model for critique
        learning_rate=1e-5,
        temperature=generation_config.temperature,
        device=device
    )

    # Train
    training_result = trainer.train(
        prompts=training_prompts,
        num_epochs=args.num_epochs,
        num_responses_per_prompt=args.num_responses_per_prompt,
        tokenizer=tokenizer
    )

    print("\n✓ Training complete!")

    # Step 6: Post-Training Evaluation
    print("\n[Step 6] Post-Training Evaluation")
    print("-" * 70)

    finetuned_metrics = evaluate_model_responses(
        model,
        tokenizer,
        training_prompts,
        evaluator,
        generation_config,
        device,
        desc="Generating fine-tuned responses"
    )

    # Step 7: Analysis and Visualization
    print("\n[Step 7] Results Analysis")
    print("-" * 70)

    improvement = finetuned_metrics['compliance_rate'] - baseline_metrics['compliance_rate']
    reduction = baseline_metrics['num_flagged'] - finetuned_metrics['num_flagged']

    print(f"\n📊 RESULTS:")
    print(f"  Baseline Compliance:    {baseline_metrics['compliance_rate']:.1%}")
    print(f"  Fine-tuned Compliance:  {finetuned_metrics['compliance_rate']:.1%}")
    print(f"  Improvement:            {improvement:+.1%}")
    print(f"  Violations Reduced:     {reduction:+d}")

    # Visualize
    viz_path = os.path.join(args.output_dir, "training_results.png")
    visualize_comparison(
        baseline_metrics,
        finetuned_metrics,
        training_result['training_history'],
        viz_path
    )

    # Step 8: Save Results
    print("\n[Step 8] Saving results...")

    results = {
        "config": {
            "model": args.model,
            "num_epochs": args.num_epochs,
            "num_prompts": args.num_prompts,
            "num_responses_per_prompt": args.num_responses_per_prompt,
        },
        "baseline": {
            "compliance_rate": baseline_metrics['compliance_rate'],
            "num_flagged": baseline_metrics['num_flagged'],
            "principle_violations": baseline_metrics['principle_violations'],
        },
        "finetuned": {
            "compliance_rate": finetuned_metrics['compliance_rate'],
            "num_flagged": finetuned_metrics['num_flagged'],
            "principle_violations": finetuned_metrics['principle_violations'],
        },
        "improvement": {
            "compliance_rate_delta": improvement,
            "violations_reduced": reduction,
        },
        "training_history": training_result['training_history'],
        "training_stats": training_result['final_stats'],
    }

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Results saved to: {results_path}")

    # Save sample responses
    samples_path = os.path.join(args.output_dir, "sample_responses.txt")
    with open(samples_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SAMPLE RESPONSES COMPARISON\n")
        f.write("="*70 + "\n\n")

        for i in range(min(3, len(training_prompts))):
            f.write(f"\n{'='*70}\n")
            f.write(f"PROMPT {i+1}:\n{training_prompts[i]}\n")
            f.write(f"{'='*70}\n\n")

            f.write("BASELINE RESPONSE:\n")
            f.write(baseline_metrics['responses'][i] + "\n\n")

            f.write("FINE-TUNED RESPONSE:\n")
            f.write(finetuned_metrics['responses'][i] + "\n\n")

    print(f"  ✓ Sample responses saved to: {samples_path}")

    # Final Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"✓ Model: {args.model}")
    print(f"✓ Constitutional compliance improved: {improvement:+.1%}")
    print(f"✓ Safety violations reduced: {reduction:+d}")
    print(f"✓ Training epochs: {args.num_epochs}")
    print(f"✓ Results saved to: {args.output_dir}")
    print("="*70)

    print("\n📚 Next Steps:")
    print("  1. Review sample responses in sample_responses.txt")
    print("  2. Check training visualization in training_results.png")
    print("  3. Adjust hyperparameters for better results")
    print("  4. Try with your own custom prompts and principles")
    print("  5. Deploy the fine-tuned model in production")


if __name__ == "__main__":
    main()
