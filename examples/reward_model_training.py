"""
Example: Training a Reward Model for Constitutional AI

This example demonstrates how to use Component 2 (Reward Model Training)
of the Constitutional AI implementation.

Requirements:
- torch
- transformers
- Constitutional AI framework (already installed)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import Constitutional AI components
from constitutional_ai.reward_model import (
    RewardModel,
    RewardModelTrainer,
    train_reward_model,
)


def main():
    """Run reward model training example."""

    print("=" * 80)
    print("Constitutional AI - Reward Model Training Example")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load model and tokenizer
    print("\n1. Loading base model...")
    model_name = "gpt2"
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = base_model.to(device)
    print(
        f"   Loaded {model_name} with {sum(p.numel() for p in base_model.parameters()):,} parameters"
    )

    # Create reward model
    print("\n2. Creating reward model...")
    reward_model = RewardModel(base_model, hidden_size=768)
    reward_model = reward_model.to(device)
    print("   Reward model created with classification head")
    print(f"   Total parameters: {sum(p.numel() for p in reward_model.parameters()):,}")

    # Create sample preference data
    # In practice, this would come from generate_preference_pairs()
    print("\n3. Creating sample preference data...")
    preference_data = [
        {
            "prompt": "What is photosynthesis?",
            "response_chosen": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen through chloroplasts.",
            "response_rejected": "Plants make food from sun.",
        },
        {
            "prompt": "Explain gravity.",
            "response_chosen": "Gravity is a fundamental force of nature that causes objects with mass to attract each other, described by Newton's law and Einstein's general relativity.",
            "response_rejected": "Things fall down because of gravity.",
        },
        {
            "prompt": "What is machine learning?",
            "response_chosen": "Machine learning is a field of artificial intelligence where algorithms learn patterns from data without being explicitly programmed, using techniques like neural networks.",
            "response_rejected": "Computers learning by themselves.",
        },
        {
            "prompt": "How does the internet work?",
            "response_chosen": "The internet works through a global network of interconnected computers using standardized protocols like TCP/IP to transmit data packets between nodes.",
            "response_rejected": "Computers connect with wires.",
        },
        {
            "prompt": "What is climate change?",
            "response_chosen": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily driven by human activities that increase greenhouse gas emissions.",
            "response_rejected": "Weather gets different over time.",
        },
    ]
    print(f"   Created {len(preference_data)} preference pairs")

    # Option 1: Use the simple train_reward_model function
    print("\n4. Training reward model (Method 1: Simple function)...")
    print("   This will take a few minutes...")

    metrics = train_reward_model(
        reward_model=reward_model,
        training_data=preference_data,
        tokenizer=tokenizer,
        num_epochs=2,
        batch_size=2,
        learning_rate=1e-5,
        device=device,
    )

    print("\n   Training Results:")
    for epoch, loss, acc in zip(
        metrics["epochs"], metrics["losses"], metrics["accuracy"], strict=True
    ):
        print(f"   Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

    # Test the trained reward model
    print("\n5. Testing trained reward model...")
    test_prompts = ["What is AI?", "Explain quantum physics"]
    test_responses_good = [
        "AI is artificial intelligence, the field of computer science focused on creating intelligent machines.",
        "Quantum physics studies matter and energy at atomic scales, where particles exhibit wave-particle duality.",
    ]
    test_responses_bad = ["AI is stuff", "Quantum is weird"]

    rewards_good = reward_model.get_rewards(test_prompts, test_responses_good, tokenizer, device)
    rewards_bad = reward_model.get_rewards(test_prompts, test_responses_bad, tokenizer, device)

    print("\n   Test Results:")
    for i, (prompt, _good, _bad, r_good, r_bad) in enumerate(
        zip(
            test_prompts,
            test_responses_good,
            test_responses_bad,
            rewards_good,
            rewards_bad,
            strict=True,
        )
    ):
        print(f"\n   Example {i+1}:")
        print(f"     Prompt: {prompt}")
        print(f"     Good response reward: {r_good.item():.4f}")
        print(f"     Bad response reward:  {r_bad.item():.4f}")
        print(f"     Preference correct: {r_good > r_bad}")

    # Option 2: Use the RewardModelTrainer class
    print("\n6. Alternative: Using RewardModelTrainer class...")
    print("   (This provides checkpointing and validation)")

    # Create new model for this example
    base_model2 = AutoModelForCausalLM.from_pretrained(model_name)
    reward_model2 = RewardModel(base_model2, hidden_size=768)

    trainer = RewardModelTrainer(
        reward_model=reward_model2,
        tokenizer=tokenizer,
        device=device,
        learning_rate=1e-5,
        batch_size=2,
    )

    # Train with automatic validation split
    metrics2 = trainer.train(
        training_data=preference_data,
        num_epochs=2,
        validation_split=0.2,
        save_dir=None,  # Set to path to save checkpoints
    )

    print("\n   Training complete!")
    if "val_accuracy" in metrics2:
        print(f"   Final validation accuracy: {metrics2['val_accuracy'][-1]:.4f}")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("- Generate preference pairs using generate_preference_pairs()")
    print("- Train on larger dataset (1000+ examples)")
    print("- Use trained reward model in PPO training (Component 4)")
    print("- Evaluate on held-out test set")


if __name__ == "__main__":
    main()
