"""
Example: Training with PPO for Constitutional AI

This example demonstrates how to use the PPO trainer for Constitutional AI.

Prerequisites:
- Trained reward model (from Component 2)
- Training prompts
- GPU recommended
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# Define Value Model
class ValueModel(nn.Module):
    """
    Value model for PPO training.
    Estimates state values for advantage estimation.
    """

    def __init__(self, base_model, hidden_size: int = 768):
        super().__init__()
        self.base_model = base_model

        # Value head: projects to scalar value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass to compute value estimate.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            value: State value estimates [batch_size]
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Use last hidden state of final token
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        last_token_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]

        # Compute value
        value = self.value_head(last_token_hidden).squeeze(-1)  # [batch_size]

        return value


def main():
    """Main training function."""
    print("PPO Training Example for Constitutional AI")
    print("=" * 60)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load models and tokenizer
    print("\n1. Loading models...")
    model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large'

    # Policy model (the model being trained)
    policy_model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"Policy model loaded: {model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")

    # 2. Create reward model
    print("\n2. Creating reward model...")
    from constitutional_ai.reward_model import RewardModel

    base_model_for_reward = AutoModelForCausalLM.from_pretrained(model_name)
    reward_model = RewardModel(base_model_for_reward, hidden_size=768)

    # NOTE: In practice, you would load trained reward model weights here:
    # reward_model.load_state_dict(torch.load('path/to/reward_model.pt'))
    print("Reward model created (Note: Should load trained weights in production)")

    # 3. Create value model
    print("\n3. Creating value model...")
    base_model_for_value = AutoModelForCausalLM.from_pretrained(model_name)
    value_model = ValueModel(base_model_for_value, hidden_size=768)
    print("Value model created")

    # 4. Initialize PPO trainer
    print("\n4. Initializing PPO trainer...")
    from constitutional_ai.ppo_trainer import PPOTrainer

    ppo_trainer = PPOTrainer(
        policy_model=policy_model,
        value_model=value_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=1e-5,      # Small learning rate for stability
        clip_epsilon=0.2,         # PPO clipping parameter
        kl_penalty=0.1,           # KL divergence penalty
        gamma=0.99,               # Discount factor
        gae_lambda=0.95,          # GAE lambda
        value_loss_coef=0.5,      # Value loss coefficient
        max_grad_norm=1.0         # Gradient clipping
    )
    print("PPO trainer initialized")

    # 5. Prepare training prompts
    print("\n5. Preparing training prompts...")
    training_prompts = [
        "What is artificial intelligence?",
        "How can I be helpful to others?",
        "Explain the concept of machine learning.",
        "What are the benefits of renewable energy?",
        "How does democracy work?",
        "What is the importance of education?",
        "How can we protect the environment?",
        "What is the scientific method?",
        "Explain the water cycle.",
        "What is the role of government?",
        # Add more prompts as needed
    ]
    print(f"Number of training prompts: {len(training_prompts)}")

    # 6. Training configuration
    print("\n6. Training configuration...")
    training_config = {
        'num_steps': 50,              # Number of training steps
        'batch_size': 4,              # Prompts per batch
        'num_epochs_per_batch': 4,    # Optimization epochs per batch
        'max_length': 150,            # Maximum response length
        'temperature': 1.0,           # Sampling temperature
        'checkpoint_dir': './checkpoints/ppo',  # Checkpoint directory
        'checkpoint_freq': 10         # Checkpoint every N steps
    }

    for key, value in training_config.items():
        print(f"  {key}: {value}")

    # 7. Run training
    print("\n7. Starting PPO training...")
    print("-" * 60)

    results = ppo_trainer.train(
        prompts=training_prompts,
        **training_config
    )

    # 8. Display results
    print("\n8. Training Results")
    print("=" * 60)

    final_stats = results['final_stats']
    print(f"\nFinal Statistics:")
    print(f"  Total steps: {final_stats['total_steps']}")
    print(f"  Average policy loss: {final_stats['avg_policy_loss']:.4f}")
    print(f"  Average value loss: {final_stats['avg_value_loss']:.4f}")
    print(f"  Average KL divergence: {final_stats['avg_kl_divergence']:.4f}")
    print(f"  Average reward: {final_stats['avg_reward']:.4f}")

    history = results['training_history']
    print(f"\nTraining History:")
    print(f"  Number of steps: {len(history['policy_losses'])}")
    print(f"  Initial policy loss: {history['policy_losses'][0]:.4f}")
    print(f"  Final policy loss: {history['policy_losses'][-1]:.4f}")
    print(f"  Initial reward: {history['mean_rewards'][0]:.4f}")
    print(f"  Final reward: {history['mean_rewards'][-1]:.4f}")

    # 9. Save final model
    print("\n9. Saving final model...")
    import os
    os.makedirs('./models/ppo_trained', exist_ok=True)

    torch.save(
        policy_model.state_dict(),
        './models/ppo_trained/policy_model.pt'
    )
    torch.save(
        value_model.state_dict(),
        './models/ppo_trained/value_model.pt'
    )
    print("Models saved to ./models/ppo_trained/")

    # 10. Test generation with trained model
    print("\n10. Testing generation with trained model...")
    policy_model.eval()

    test_prompts = [
        "How can AI be beneficial?",
        "What should I know about ethics?"
    ]

    from constitutional_ai.model_utils import generate_text, GenerationConfig

    gen_config = GenerationConfig(
        max_length=100,
        temperature=0.7,
        do_sample=True
    )

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = generate_text(
            policy_model,
            tokenizer,
            prompt,
            gen_config,
            device
        )
        print(f"Response: {response}")

    print("\n" + "=" * 60)
    print("PPO training complete!")


if __name__ == '__main__':
    # Optional: Set random seeds for reproducibility
    import random
    import numpy as np

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    main()
