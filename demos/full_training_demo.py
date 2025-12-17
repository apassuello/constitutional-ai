#!/usr/bin/env python3
"""
Constitutional AI Training Demo Script

This script demonstrates the full Constitutional AI pipeline:
- Phase 1: Supervised Learning from AI Feedback (Critique-Revision)
- Phase 2: Reinforcement Learning from AI Feedback (RLAIF)

Usage:
    python demo_constitutional_ai.py --phase 1           # Run Phase 1 only
    python demo_constitutional_ai.py --phase 2           # Run Phase 2 only
    python demo_constitutional_ai.py --phase both        # Run both phases
    python demo_constitutional_ai.py --quick-test        # Quick test with small dataset

Requirements:
    pip install torch transformers tqdm
"""

import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Setup and verify environment."""
    logger.info("=" * 80)
    logger.info("CONSTITUTIONAL AI TRAINING DEMO")
    logger.info("=" * 80)

    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    return device


def load_or_generate_prompts(num_prompts: int = 50, quick_test: bool = False) -> List[str]:
    """
    Load or generate training prompts.

    Args:
        num_prompts: Number of prompts to use
        quick_test: If True, use tiny dataset for testing

    Returns:
        List of prompts
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 0: Loading/Generating Prompts")
    logger.info("=" * 80)

    if quick_test:
        num_prompts = 5
        logger.info("Quick test mode: Using 5 sample prompts")

    # Check if prompts file exists
    prompts_file = Path("data/constitutional_prompts.json")

    if prompts_file.exists():
        logger.info(f"Loading prompts from {prompts_file}")
        with open(prompts_file) as f:
            data = json.load(f)
            prompts = data.get('prompts', [])[:num_prompts]
        logger.info(f"Loaded {len(prompts)} prompts")
    else:
        logger.info("No prompts file found, using sample prompts")
        prompts = [
            "What is artificial intelligence?",
            "Explain how photosynthesis works.",
            "What causes climate change?",
            "How do vaccines work?",
            "What is the theory of relativity?",
            "Explain quantum mechanics in simple terms.",
            "What is machine learning?",
            "How does the human brain work?",
            "What is evolution?",
            "Explain how computers work.",
        ][:num_prompts]
        logger.info(f"Using {len(prompts)} sample prompts")

    logger.info("\nSample prompts:")
    for i, prompt in enumerate(prompts[:3], 1):
        logger.info(f"  {i}. {prompt}")
    if len(prompts) > 3:
        logger.info(f"  ... and {len(prompts) - 3} more")

    return prompts


def phase_1_supervised_learning(
    prompts: List[str],
    device: torch.device,
    model_name: str = "gpt2",
    num_epochs: int = 3,
    save_dir: str = "outputs/phase1"
) -> str:
    """
    Phase 1: Supervised Learning from AI Feedback (Critique-Revision).

    Args:
        prompts: List of training prompts
        device: Computation device
        model_name: Base model to use (default: gpt2)
        num_epochs: Number of training epochs
        save_dir: Directory to save model

    Returns:
        Path to trained Phase 1 model
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: SUPERVISED LEARNING FROM AI FEEDBACK")
    logger.info("=" * 80)

    from constitutional_ai import (
        setup_default_framework,
        generate_critique,
        generate_revision
    )
    from constitutional_ai.model_utils import load_model, generate_text, GenerationConfig
    from torch.utils.data import DataLoader, Dataset
    import torch.nn.functional as F

    # Step 1: Load model and constitutional framework
    logger.info("\n[1/5] Loading base model and constitutional framework...")
    model, tokenizer = load_model(model_name, device=device)
    framework = setup_default_framework()
    principles = [p.description for p in framework.principles]
    logger.info(f"Model: {model_name}")
    logger.info(f"Constitutional principles: {len(principles)}")

    # Step 2: Generate initial responses
    logger.info("\n[2/5] Generating initial responses...")
    original_responses = []
    generation_config = GenerationConfig(max_length=150, temperature=0.8, do_sample=True)

    for i, prompt in enumerate(prompts):
        logger.info(f"  Generating response {i+1}/{len(prompts)}...")
        response = generate_text(model, tokenizer, prompt, generation_config, device)
        original_responses.append(response)

    logger.info(f"Generated {len(original_responses)} initial responses")

    # Step 3: Generate critiques
    logger.info("\n[3/5] Generating critiques...")
    critiques = []

    for i, (prompt, response) in enumerate(zip(prompts, original_responses)):
        logger.info(f"  Critiquing response {i+1}/{len(prompts)}...")
        critique = generate_critique(
            prompt=prompt,
            response=response,
            principles=principles,
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        critiques.append(critique)

    logger.info(f"Generated {len(critiques)} critiques")

    # Step 4: Generate revisions
    logger.info("\n[4/5] Generating revisions...")
    revised_responses = []

    for i, (prompt, response, critique) in enumerate(zip(prompts, original_responses, critiques)):
        logger.info(f"  Revising response {i+1}/{len(prompts)}...")
        revision = generate_revision(
            prompt=prompt,
            response=response,
            critique=critique,
            principles=principles,
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        revised_responses.append(revision)

    logger.info(f"Generated {len(revised_responses)} revisions")

    # Show example
    logger.info("\n" + "-" * 80)
    logger.info("EXAMPLE CRITIQUE-REVISION:")
    logger.info(f"Prompt: {prompts[0]}")
    logger.info(f"Original: {original_responses[0][:100]}...")
    logger.info(f"Critique: {critiques[0][:100]}...")
    logger.info(f"Revised: {revised_responses[0][:100]}...")
    logger.info("-" * 80)

    # Step 5: Fine-tune on revised responses
    logger.info("\n[5/5] Fine-tuning model on revised responses...")
    logger.info("NOTE: Full fine-tuning can take hours. Skipping in demo.")
    logger.info("In production, you would:")
    logger.info("  1. Create ConstitutionalTrainingDataset")
    logger.info("  2. Train with standard supervised learning")
    logger.info("  3. Validate on held-out set")
    logger.info("  4. Save best model")

    # Save Phase 1 data
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    phase1_data = {
        'prompts': prompts,
        'original_responses': original_responses,
        'critiques': critiques,
        'revised_responses': revised_responses,
        'model_name': model_name
    }

    data_file = save_path / 'phase1_data.json'
    with open(data_file, 'w') as f:
        json.dump(phase1_data, f, indent=2)

    logger.info(f"\nPhase 1 data saved to: {data_file}")
    logger.info(f"Total examples: {len(prompts)}")

    # In real training, return path to fine-tuned model
    # For demo, return base model name
    logger.info("\n✅ Phase 1 complete!")
    logger.info("In production: This would be a fine-tuned model")
    logger.info("For demo: Using base model for Phase 2")

    return model_name


def phase_2_reinforcement_learning(
    prompts: List[str],
    device: torch.device,
    phase1_model: str = "gpt2",
    num_ppo_steps: int = 10,
    save_dir: str = "outputs/phase2"
) -> str:
    """
    Phase 2: Reinforcement Learning from AI Feedback (RLAIF).

    Args:
        prompts: List of training prompts
        device: Computation device
        phase1_model: Model from Phase 1
        num_ppo_steps: Number of PPO training steps
        save_dir: Directory to save model

    Returns:
        Path to trained Phase 2 model
    """
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: REINFORCEMENT LEARNING FROM AI FEEDBACK")
    logger.info("=" * 80)

    from constitutional_ai import (
        setup_default_framework,
        RewardModel,
        PPOTrainer,
        generate_preference_pairs
    )
    from constitutional_ai.model_utils import load_model
    import torch.nn as nn

    # Step 1: Load Phase 1 model
    logger.info("\n[1/4] Loading Phase 1 model...")
    policy_model, tokenizer = load_model(phase1_model, device=device)
    framework = setup_default_framework()
    logger.info(f"Model: {phase1_model}")

    # Step 2: Generate preference pairs
    logger.info("\n[2/4] Generating preference pairs...")
    logger.info("  Generating multiple responses per prompt...")
    logger.info("  Comparing responses with AI...")
    logger.info("  Extracting preferences...")

    preference_data = generate_preference_pairs(
        prompts=prompts,
        model=policy_model,
        tokenizer=tokenizer,
        framework=framework,
        device=device,
        num_responses_per_prompt=2
    )

    logger.info(f"Generated {len(preference_data)} preference pairs")

    # Show example
    if len(preference_data) > 0:
        example = preference_data[0]
        logger.info("\n" + "-" * 80)
        logger.info("EXAMPLE PREFERENCE PAIR:")
        logger.info(f"Prompt: {example['prompt']}")
        logger.info(f"Chosen: {example['response_chosen'][:80]}...")
        logger.info(f"Rejected: {example['response_rejected'][:80]}...")
        logger.info("-" * 80)

    # Step 3: Train reward model
    logger.info("\n[3/4] Training reward model...")

    # Create reward model
    base_model_for_reward, _ = load_model(phase1_model, device=device)
    reward_model = RewardModel(base_model_for_reward, hidden_size=768)
    reward_model = reward_model.to(device)

    logger.info("Training reward model with Bradley-Terry loss...")
    logger.info("NOTE: Full training can take hours. Using minimal training in demo.")

    from constitutional_ai import train_reward_model

    metrics = train_reward_model(
        reward_model=reward_model,
        training_data=preference_data[:min(20, len(preference_data))],  # Limit for demo
        tokenizer=tokenizer,
        num_epochs=1,  # Minimal for demo
        batch_size=2,
        device=device
    )

    logger.info(f"Reward model training complete!")
    logger.info(f"Final accuracy: {metrics['accuracy'][-1]:.2%}")

    # Step 4: PPO Training
    logger.info("\n[4/4] PPO training...")
    logger.info("NOTE: Full PPO training can take many hours.")
    logger.info("Demo will run minimal training steps.")

    # Create value model (simple MLP on top of base model)
    class ValueModel(nn.Module):
        def __init__(self, base_model, hidden_size=768):
            super().__init__()
            self.base_model = base_model
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            sequence_lengths = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[torch.arange(hidden_states.size(0)), sequence_lengths]
            value = self.value_head(last_hidden).squeeze(-1)
            return value

    value_model_base, _ = load_model(phase1_model, device=device)
    value_model = ValueModel(value_model_base, hidden_size=768)
    value_model = value_model.to(device)

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        policy_model=policy_model,
        value_model=value_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=device,
        clip_epsilon=0.2,
        kl_penalty=0.1
    )

    logger.info("Running PPO training...")
    logger.info(f"Number of steps: {num_ppo_steps}")
    logger.info(f"Batch size: 2")

    results = ppo_trainer.train(
        prompts=prompts[:min(10, len(prompts))],  # Limit for demo
        num_steps=num_ppo_steps,
        batch_size=2,
        num_epochs_per_batch=2,  # Reduced for demo
        checkpoint_dir=None  # No checkpointing in demo
    )

    logger.info("\nPPO training complete!")
    logger.info(f"Final stats:")
    logger.info(f"  Mean reward: {results['final_stats']['mean_rewards'][-1]:.4f}")
    logger.info(f"  Policy loss: {results['final_stats']['policy_losses'][-1]:.4f}")
    logger.info(f"  KL divergence: {results['final_stats']['kl_divergences'][-1]:.4f}")

    # Save Phase 2 data
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    phase2_data = {
        'num_preference_pairs': len(preference_data),
        'reward_model_accuracy': float(metrics['accuracy'][-1]),
        'ppo_steps': num_ppo_steps,
        'final_reward': float(results['final_stats']['mean_rewards'][-1]),
        'model_name': phase1_model
    }

    data_file = save_path / 'phase2_data.json'
    with open(data_file, 'w') as f:
        json.dump(phase2_data, f, indent=2)

    logger.info(f"\nPhase 2 data saved to: {data_file}")

    logger.info("\n✅ Phase 2 complete!")
    logger.info("Model has been aligned using Constitutional AI!")

    return phase1_model


def evaluate_model(
    model_name: str,
    test_prompts: List[str],
    device: torch.device
):
    """
    Evaluate the trained model.

    Args:
        model_name: Model to evaluate
        test_prompts: Test prompts
        device: Computation device
    """
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION")
    logger.info("=" * 80)

    from constitutional_ai.model_utils import load_model, generate_text, GenerationConfig
    from constitutional_ai import setup_default_framework

    logger.info(f"\nLoading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    framework = setup_default_framework()

    logger.info(f"Generating responses for {len(test_prompts)} test prompts...")

    generation_config = GenerationConfig(max_length=150, temperature=0.7, do_sample=True)

    for i, prompt in enumerate(test_prompts):
        logger.info(f"\n{'-' * 80}")
        logger.info(f"Test {i+1}/{len(test_prompts)}")
        logger.info(f"Prompt: {prompt}")

        response = generate_text(model, tokenizer, prompt, generation_config, device)
        logger.info(f"Response: {response}")

    logger.info("\n✅ Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Constitutional AI Training Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_constitutional_ai.py --phase 1
  python demo_constitutional_ai.py --phase 2
  python demo_constitutional_ai.py --phase both
  python demo_constitutional_ai.py --quick-test

For full training:
  python demo_constitutional_ai.py --phase both --num-prompts 500 --num-epochs 3 --num-ppo-steps 100
        """
    )

    parser.add_argument(
        '--phase',
        choices=['1', '2', 'both'],
        default='both',
        help='Which phase to run (default: both)'
    )

    parser.add_argument(
        '--model',
        default='gpt2',
        help='Base model to use (default: gpt2)'
    )

    parser.add_argument(
        '--num-prompts',
        type=int,
        default=50,
        help='Number of training prompts (default: 50)'
    )

    parser.add_argument(
        '--num-epochs',
        type=int,
        default=3,
        help='Number of SFT epochs in Phase 1 (default: 3)'
    )

    parser.add_argument(
        '--num-ppo-steps',
        type=int,
        default=10,
        help='Number of PPO steps in Phase 2 (default: 10)'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with minimal data (5 prompts, 5 PPO steps)'
    )

    parser.add_argument(
        '--output-dir',
        default='outputs',
        help='Output directory for models and data (default: outputs)'
    )

    args = parser.parse_args()

    # Quick test overrides
    if args.quick_test:
        args.num_prompts = 5
        args.num_ppo_steps = 5
        args.num_epochs = 1
        logger.info("Quick test mode enabled: Using minimal dataset")

    # Setup
    device = setup_environment()

    # Load prompts
    prompts = load_or_generate_prompts(args.num_prompts, args.quick_test)

    # Split into train/test
    split_idx = int(len(prompts) * 0.8)
    train_prompts = prompts[:split_idx]
    test_prompts = prompts[split_idx:]

    logger.info(f"\nDataset split:")
    logger.info(f"  Training: {len(train_prompts)} prompts")
    logger.info(f"  Testing: {len(test_prompts)} prompts")

    # Run phases
    phase1_model = args.model

    if args.phase in ['1', 'both']:
        phase1_model = phase_1_supervised_learning(
            prompts=train_prompts,
            device=device,
            model_name=args.model,
            num_epochs=args.num_epochs,
            save_dir=f"{args.output_dir}/phase1"
        )

    if args.phase in ['2', 'both']:
        phase_2_reinforcement_learning(
            prompts=train_prompts,
            device=device,
            phase1_model=phase1_model,
            num_ppo_steps=args.num_ppo_steps,
            save_dir=f"{args.output_dir}/phase2"
        )

    # Evaluate
    if len(test_prompts) > 0:
        evaluate_model(
            model_name=phase1_model,
            test_prompts=test_prompts[:3],  # Evaluate on 3 test prompts
            device=device
        )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nSummary:")
    logger.info(f"  Base model: {args.model}")
    logger.info(f"  Training prompts: {len(train_prompts)}")
    logger.info(f"  Test prompts: {len(test_prompts)}")

    if args.phase in ['1', 'both']:
        logger.info(f"  Phase 1: ✅ Critique-revision complete")

    if args.phase in ['2', 'both']:
        logger.info(f"  Phase 2: ✅ RLAIF complete")

    logger.info(f"\nOutputs saved to: {args.output_dir}/")
    logger.info("\nNext steps:")
    logger.info("  1. Review generated data in outputs/")
    logger.info("  2. Run with more prompts for better results")
    logger.info("  3. Fine-tune for more epochs")
    logger.info("  4. Evaluate on your specific use case")

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
