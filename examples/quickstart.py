#!/usr/bin/env python3
"""
Quick Start Demo for Constitutional AI

This is a simplified demo showing the core concepts of Constitutional AI
without requiring long training times.

Usage:
    python examples/quick_start_demo.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import Constitutional AI components
from constitutional_ai import (
    setup_default_framework,
    generate_critique,
    generate_revision,
    generate_comparison,
    RewardModel
)
from constitutional_ai.model_utils import generate_text, GenerationConfig


def demo_critique_revision():
    """Demonstrate Phase 1: Critique-Revision Cycle."""
    print("=" * 80)
    print("PHASE 1 DEMO: CRITIQUE-REVISION CYCLE")
    print("=" * 80)

    # Setup
    print("\n[1] Loading model and constitutional framework...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    framework = setup_default_framework()
    principles = [p.description for p in framework.principles]

    print(f"✓ Loaded model: GPT-2")
    print(f"✓ Loaded {len(principles)} constitutional principles")

    # Example prompt
    prompt = "What is artificial intelligence?"
    print(f"\n[2] Generating initial response...")
    print(f"Prompt: {prompt}")

    config = GenerationConfig(max_length=100, temperature=0.8, do_sample=True)
    original_response = generate_text(model, tokenizer, prompt, config, device)

    print(f"Original response: {original_response}")

    # Generate critique
    print(f"\n[3] Generating constitutional critique...")
    critique = generate_critique(
        prompt=prompt,
        response=original_response,
        principles=principles,
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    print(f"Critique: {critique}")

    # Generate revision
    print(f"\n[4] Generating improved revision...")
    revision = generate_revision(
        prompt=prompt,
        response=original_response,
        critique=critique,
        principles=principles,
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    print(f"Revised response: {revision}")

    print("\n✅ Phase 1 complete! The response has been improved via critique-revision.")
    print("\nKey takeaway: The model critiques its own response and generates")
    print("an improved version aligned with constitutional principles.")


def demo_preference_comparison():
    """Demonstrate Phase 2: Preference Comparison."""
    print("\n\n" + "=" * 80)
    print("PHASE 2 DEMO: PREFERENCE COMPARISON")
    print("=" * 80)

    # Setup
    print("\n[1] Loading model and constitutional framework...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    framework = setup_default_framework()
    principles = [p.description for p in framework.principles]

    print(f"✓ Model loaded")

    # Example prompt and responses
    prompt = "Explain climate change."
    response_a = "Climate change is when the weather gets different and sometimes bad things happen."
    response_b = "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily driven by human activities like burning fossil fuels, which increase greenhouse gas emissions."

    print(f"\n[2] Comparing two responses...")
    print(f"Prompt: {prompt}")
    print(f"\nResponse A: {response_a}")
    print(f"\nResponse B: {response_b}")

    # Generate comparison
    print(f"\n[3] AI evaluates which response better follows constitutional principles...")

    comparison = generate_comparison(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        principles=principles,
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    print(f"Preferred: Response {comparison['preferred']}")
    print(f"Reasoning: {comparison['comparison_text'][:200]}...")

    print("\n✅ Phase 2 demo complete! AI-generated preferences will train the reward model.")
    print("\nKey takeaway: The model evaluates response pairs to create training data")
    print("for the reward model, which will guide reinforcement learning.")


def demo_reward_model():
    """Demonstrate Reward Model."""
    print("\n\n" + "=" * 80)
    print("REWARD MODEL DEMO")
    print("=" * 80)

    # Setup
    print("\n[1] Creating reward model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    base_model = base_model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create reward model
    reward_model = RewardModel(base_model, hidden_size=768)
    reward_model = reward_model.to(device)
    reward_model.eval()

    print(f"✓ Reward model created")

    # Example responses to score
    prompt = "What is machine learning?"
    good_response = "Machine learning is a subset of AI where algorithms learn patterns from data to make predictions or decisions without explicit programming."
    bad_response = "Machine learning is computer stuff."

    print(f"\n[2] Scoring responses with reward model...")
    print(f"Prompt: {prompt}")
    print(f"\nGood response: {good_response}")
    print(f"Bad response: {bad_response}")

    with torch.no_grad():
        good_reward = reward_model.get_rewards(
            prompts=[prompt],
            responses=[good_response],
            tokenizer=tokenizer,
            device=device
        )

        bad_reward = reward_model.get_rewards(
            prompts=[prompt],
            responses=[bad_response],
            tokenizer=tokenizer,
            device=device
        )

    print(f"\n[3] Reward scores:")
    print(f"Good response reward: {good_reward.item():.4f}")
    print(f"Bad response reward: {bad_reward.item():.4f}")

    if good_reward.item() > bad_reward.item():
        print("✓ Reward model correctly prefers the better response!")
    else:
        print("Note: Untrained reward model may not discriminate well yet.")
        print("After training on preference data, it will learn to score responses.")

    print("\n✅ Reward model demo complete!")
    print("\nKey takeaway: The reward model scores responses. After training on")
    print("preference pairs, it learns to give higher scores to better responses.")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "    CONSTITUTIONAL AI - QUICK START DEMO".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    print("\nThis demo will show the key components of Constitutional AI:")
    print("  1. Phase 1: Critique-Revision Cycle")
    print("  2. Phase 2: Preference Comparison")
    print("  3. Reward Model")

    print("\nNote: This demo uses untrained models for illustration.")
    print("For production, train models on larger datasets.")

    input("\nPress Enter to start the demo...")

    # Run demos
    try:
        demo_critique_revision()
        input("\nPress Enter to continue to Phase 2 demo...")

        demo_preference_comparison()
        input("\nPress Enter to continue to Reward Model demo...")

        demo_reward_model()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        return

    # Summary
    print("\n\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)

    print("\nYou've seen the three key components of Constitutional AI:")
    print("\n1. CRITIQUE-REVISION (Phase 1):")
    print("   • Model generates response")
    print("   • Model critiques its own response")
    print("   • Model generates improved revision")
    print("   • Fine-tune on improved responses")

    print("\n2. PREFERENCE COMPARISON (Phase 2a):")
    print("   • Generate multiple responses per prompt")
    print("   • AI compares and ranks responses")
    print("   • Creates preference dataset")

    print("\n3. REWARD MODEL (Phase 2b):")
    print("   • Trains on preference pairs")
    print("   • Learns to score response quality")
    print("   • Used to guide PPO training")

    print("\n4. PPO TRAINING (Phase 2c) - Not shown in quick demo:")
    print("   • Uses reward model to guide training")
    print("   • Optimizes policy with clipped objective")
    print("   • Results in aligned model")

    print("\nNext steps:")
    print("  • Run full demo: python demo_constitutional_ai.py --quick-test")
    print("  • Read docs: docs/CONSTITUTIONAL_AI_ARCHITECTURE.md")
    print("  • Train on your data: python demo_constitutional_ai.py --phase both")

    print("\n" + "=" * 80)
    print("Thank you for exploring Constitutional AI!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
