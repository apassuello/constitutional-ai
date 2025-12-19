"""
Training handler for Constitutional AI demo.

Handles live training in the Gradio demo with progress tracking.
"""

import time
from typing import Any

import torch

from constitutional_ai import setup_default_framework
from constitutional_ai.critique_revision import critique_revision_pipeline, supervised_finetune


def train_model_live(
    model,
    tokenizer,
    prompts: list[str],
    num_epochs: int,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    progress_callback=None,
) -> tuple[bool, str, dict[str, Any]]:
    """
    Train model with Constitutional AI live in the demo.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        prompts: Training prompts
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        progress_callback: Gradio progress callback

    Returns:
        Tuple of (success, message, metrics)
    """
    try:
        start_time = time.time()
        device = model.device if hasattr(model, "device") else torch.device("cpu")

        # Setup framework
        framework = setup_default_framework()

        # Phase 1: Generate critique-revision data
        if progress_callback:
            progress_callback(0.0, desc="Generating critiques and revisions...")

        training_data = critique_revision_pipeline(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            framework=framework,
            num_epochs=1,  # One pass for data generation
            device=device,
        )

        num_examples = len(training_data)

        if progress_callback:
            progress_callback(
                0.3, desc=f"Generated {num_examples} training examples. Starting fine-tuning..."
            )

        # Phase 2: Fine-tune on revised responses
        train_prompts = [item["prompt"] for item in training_data]
        train_responses = [item["revised_response"] for item in training_data]

        def training_progress(epoch, batch, total_batches):
            progress = 0.3 + (epoch / num_epochs + batch / total_batches / num_epochs) * 0.7
            if progress_callback:
                progress_callback(
                    progress, desc=f"Epoch {epoch+1}/{num_epochs}, Batch {batch}/{total_batches}"
                )

        metrics = supervised_finetune(
            model=model,
            tokenizer=tokenizer,
            prompts=train_prompts,
            responses=train_responses,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            progress_callback=training_progress,
        )

        training_time = time.time() - start_time

        if progress_callback:
            progress_callback(1.0, desc="Training complete!")

        success_msg = f"""
        ✅ **Training Complete!**

        - **Examples**: {num_examples} critique-revision pairs
        - **Epochs**: {num_epochs}
        - **Final Loss**: {metrics.get('final_loss', 'N/A'):.4f}
        - **Time**: {training_time/60:.1f} minutes

        The model is now constitutionally trained and ready for comparison.
        Go to the **Comparison** tab to see before/after results!
        """

        return (
            True,
            success_msg,
            {**metrics, "training_time": training_time, "num_examples": num_examples},
        )

    except Exception as e:
        return False, f"❌ **Training failed:** {str(e)}", {}
