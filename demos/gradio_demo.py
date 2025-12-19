"""
Constitutional AI Interactive Demo - Gradio Web Interface

This interactive demo allows you to:
1. Evaluate text against constitutional principles
2. See before/after comparison (baseline vs. constitutional models)
3. Apply safety filtering
4. Explore the training pipeline interactively
5. Learn about constitutional principles

Usage:
    python demos/gradio_demo.py

    Then open http://localhost:7860 in your browser
"""

import gradio as gr
from constitutional_ai import (
    ConstitutionalSafetyEvaluator,
    ConstitutionalSafetyFilter,
    setup_default_framework,
    generate_text,
)
from constitutional_ai.model_utils import GenerationConfig

from demo_utils.model_manager import ModelManager
from demo_utils.examples import get_example_prompts, get_test_suites, get_training_config
from demo_utils.formatters import (
    format_evaluation_result,
    format_filter_result,
    format_comparison_table,
    format_comparison_result,
    format_model_info,
)
from demo_utils.visualizations import (
    create_principle_bar_chart,
    create_comparison_chart,
    create_improvement_chart,
    create_radar_chart,
)
from demo_utils.training_handler import train_model_live

# Initialize global state
framework = setup_default_framework()
evaluator = ConstitutionalSafetyEvaluator(framework=framework, use_self_critique=False)
safety_filter = ConstitutionalSafetyFilter(constitutional_framework=framework)
model_manager = ModelManager()

# Training state
trained_model = None  # Will hold the trained model after training completes
baseline_model = None  # Will hold the baseline for comparison

# ============================= Helper Functions =============================


def evaluate_text_handler(text: str):
    """Handle text evaluation."""
    if not text or not text.strip():
        return (
            "<p style='color: gray;'>Enter some text to evaluate...</p>",
            "",
            None,
            None,
        )

    # Evaluate
    result = evaluator.evaluate(text)

    # Format results
    status_html, details_html = format_evaluation_result(result)

    # Get scores for visualization
    direct_eval = result.get("direct_evaluation", {})
    principle_scores = direct_eval.get("principle_scores", {})
    flagged_principles = direct_eval.get("flagged_principles", [])

    # Create charts
    bar_chart = create_principle_bar_chart(principle_scores, flagged_principles)
    radar_chart = create_radar_chart(principle_scores)

    return status_html, details_html, bar_chart, radar_chart


def filter_text_handler(text: str):
    """Handle safety filtering."""
    if not text or not text.strip():
        return "", "<p style='color: gray;'>Enter some text to filter...</p>", ""

    # Apply filter
    filtered_text, filter_info = safety_filter.filter_output(text)

    # Format results
    info_html = format_filter_result(filtered_text, filter_info)
    comparison_html = format_comparison_table(text, filtered_text)

    return filtered_text, info_html, comparison_html


def load_model_handler(model_choice: str):
    """Handle model loading."""
    if model_choice == "None":
        model_manager.unload()
        return "No model selected", format_model_info(model_manager.get_status_info())

    try:
        # Map display names to model names
        model_map = {
            "DistilGPT-2 (fastest, 82M params)": "distilgpt2",
            "GPT-2 (standard, 124M params)": "gpt2",
            "GPT-2 Medium (355M params)": "gpt2-medium",
        }

        model_name = model_map.get(model_choice)
        if model_name:
            success, message = model_manager.load_baseline(model_name)
            return message, format_model_info(model_manager.get_status_info())
        else:
            return "Invalid model choice", format_model_info(model_manager.get_status_info())

    except Exception as e:
        return f"Error: {str(e)}", format_model_info(model_manager.get_status_info())


def generate_and_compare_handler(prompt: str):
    """Generate response and compare baseline vs. constitutional."""
    if not model_manager.is_loaded():
        return "❌ Please load a model first", "", None, None

    if not prompt or not prompt.strip():
        return "❌ Please enter a prompt", "", None, None

    try:
        model, tokenizer = model_manager.get_model()
        device = model_manager.device

        # Generate response
        config = GenerationConfig(max_length=100, temperature=0.7, do_sample=True)
        response = generate_text(model, tokenizer, prompt, config, device)

        # For now, we'll simulate "constitutional" by filtering
        # In real demo with pre-trained model, this would be a different model
        filtered_response, _ = safety_filter.filter_output(response)

        # Evaluate both
        baseline_eval = evaluator.evaluate(response)
        const_eval = evaluator.evaluate(filtered_response)

        # Format comparison
        comparison_html = format_comparison_result(baseline_eval, const_eval)

        # Get scores for charts
        baseline_scores = baseline_eval.get("direct_evaluation", {}).get("principle_scores", {})
        const_scores = const_eval.get("direct_evaluation", {}).get("principle_scores", {})

        # Create charts
        comparison_chart = create_comparison_chart(baseline_scores, const_scores)
        improvement_chart = create_improvement_chart(baseline_scores, const_scores)

        # Create response comparison
        response_html = f"""
        <div style='padding: 10px;'>
            <h3>Prompt:</h3>
            <p style='padding: 10px; background-color: #f3f4f6; border-radius: 4px;'>{prompt}</p>

            <h3>Baseline Response:</h3>
            <p style='padding: 10px; background-color: #fee2e2; border-radius: 4px;'>{response}</p>

            <h3>Constitutional Response:</h3>
            <p style='padding: 10px; background-color: #d1fae5; border-radius: 4px;'>{filtered_response}</p>
        </div>
        """

        return response_html, comparison_html, comparison_chart, improvement_chart

    except Exception as e:
        return f"❌ Error: {str(e)}", "", None, None


def start_training_handler(training_mode: str, progress=gr.Progress()):
    """Handle training start."""
    global trained_model, baseline_model

    if not model_manager.is_loaded():
        return "❌ Please load a model first in the Comparison tab", ""

    # Get training configuration
    mode = "quick" if "Quick" in training_mode else "standard"
    config = get_training_config(mode)

    model, tokenizer = model_manager.get_model()

    # Save baseline model state (clone it)
    import copy
    baseline_model = copy.deepcopy(model)

    # Train the model
    model_manager.set_training_mode()

    success, message, metrics = train_model_live(
        model=model,
        tokenizer=tokenizer,
        prompts=config["prompts"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        progress_callback=progress,
    )

    model_manager.set_ready_mode()

    if success:
        trained_model = model  # Save trained model
        metrics_html = f"""
        <div style='padding: 10px; background-color: #d1fae5; border-radius: 8px;'>
            <h3>✅ Training Metrics</h3>
            <ul>
                <li><strong>Examples:</strong> {metrics.get('num_examples', 'N/A')}</li>
                <li><strong>Epochs:</strong> {config['num_epochs']}</li>
                <li><strong>Final Loss:</strong> {metrics.get('final_loss', 'N/A'):.4f}</li>
                <li><strong>Training Time:</strong> {metrics.get('training_time', 0)/60:.1f} minutes</li>
            </ul>
        </div>
        """
        return message, metrics_html
    else:
        return message, ""


# ============================= Gradio Interface =============================

with gr.Blocks(title="Constitutional AI Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🛡️ Constitutional AI Interactive Demo

        Explore Constitutional AI - a framework for training safer, more aligned language models.

        **Key Features:**
        - Evaluate text against constitutional principles (Harm, Truth, Fairness, Autonomy)
        - Compare baseline vs. constitutionally-trained model responses
        - Apply safety filtering to improve content
        - Understand how each principle works
        """
    )

    with gr.Tabs() as tabs:
        # Tab 1: Quick Evaluation
        with gr.Tab("📊 Evaluation"):
            gr.Markdown("### Evaluate text against constitutional principles")

            with gr.Row():
                with gr.Column(scale=2):
                    eval_input = gr.Textbox(
                        label="Text to Evaluate",
                        placeholder="Enter any text to check for constitutional compliance...",
                        lines=5,
                    )
                    eval_button = gr.Button("Evaluate", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Example Prompts")
                    example_dropdown = gr.Dropdown(
                        choices=get_example_prompts(),
                        label="Quick Examples",
                        interactive=True,
                    )
                    load_example_btn = gr.Button("Load Example", size="sm")

            eval_status = gr.HTML(label="Status")

            with gr.Row():
                with gr.Column():
                    eval_details = gr.HTML(label="Evaluation Details")

                with gr.Column():
                    eval_bar_chart = gr.Image(label="Principle Scores", type="pil")
                    eval_radar_chart = gr.Image(label="Principle Balance", type="pil")

            # Wire up evaluation
            eval_button.click(
                fn=evaluate_text_handler,
                inputs=[eval_input],
                outputs=[eval_status, eval_details, eval_bar_chart, eval_radar_chart],
            )

            load_example_btn.click(fn=lambda x: x, inputs=[example_dropdown], outputs=[eval_input])

        # Tab 2: Before/After Comparison
        with gr.Tab("🔄 Comparison"):
            gr.Markdown(
                """
            ### Before vs. After: Constitutional AI Impact

            Load a model and generate responses to see the difference between
            baseline and constitutionally-filtered outputs.

            **Note:** Full Constitutional AI training takes hours. This demo uses
            filtering to approximate the improvement for quick demonstration.
            """
            )

            with gr.Row():
                model_choice = gr.Dropdown(
                    choices=[
                        "None",
                        "DistilGPT-2 (fastest, 82M params)",
                        "GPT-2 (standard, 124M params)",
                        "GPT-2 Medium (355M params)",
                    ],
                    label="Select Model",
                    value="None",
                )
                load_model_btn = gr.Button("Load Model", variant="primary")

            model_status_html = gr.HTML(label="Model Status")
            load_message = gr.Textbox(label="Load Message", interactive=False)

            gr.Markdown("### Generate and Compare")

            compare_prompt = gr.Textbox(
                label="Enter Prompt",
                placeholder="e.g., 'How can I help someone who is sad?'",
                lines=2,
            )
            compare_button = gr.Button("Generate & Compare", variant="primary", size="lg")

            comparison_responses = gr.HTML(label="Responses")
            comparison_metrics = gr.HTML(label="Comparison Metrics")

            with gr.Row():
                comparison_chart_img = gr.Image(label="Score Comparison", type="pil")
                improvement_chart_img = gr.Image(label="Improvement", type="pil")

            # Wire up comparison
            load_model_btn.click(
                fn=load_model_handler, inputs=[model_choice], outputs=[load_message, model_status_html]
            )

            compare_button.click(
                fn=generate_and_compare_handler,
                inputs=[compare_prompt],
                outputs=[
                    comparison_responses,
                    comparison_metrics,
                    comparison_chart_img,
                    improvement_chart_img,
                ],
            )

        # Tab 3: Training
        with gr.Tab("🎓 Training"):
            gr.Markdown(
                """
            ### Train a Constitutional AI Model

            This tab lets you train a model with Constitutional AI methodology.
            The training process takes 15-60 minutes depending on configuration.

            **Process:**
            1. Load a baseline model in the Comparison tab
            2. Select training mode (Quick or Standard)
            3. Click Start Training and wait for completion
            4. Go to Comparison tab to see before/after results

            **Training Modes:**
            - **Quick**: 2 epochs, 10 prompts (~15-20 min)
            - **Standard**: 5 epochs, 30 prompts (~45-60 min)
            """
            )

            training_mode_choice = gr.Radio(
                choices=[
                    "Quick Training (2 epochs, 10 prompts, ~15-20 min)",
                    "Standard Training (5 epochs, 30 prompts, ~45-60 min)",
                ],
                label="Training Mode",
                value="Quick Training (2 epochs, 10 prompts, ~15-20 min)",
            )

            start_training_btn = gr.Button("▶️ Start Training", variant="primary", size="lg")

            training_status = gr.Markdown(label="Training Status")
            training_metrics = gr.HTML(label="Training Metrics")

            gr.Markdown(
                """
            **Note:** Training modifies the loaded model in-place. After training completes,
            go to the **Comparison** tab to compare the baseline vs. trained model.
            """
            )

            # Wire up training
            start_training_btn.click(
                fn=start_training_handler,
                inputs=[training_mode_choice],
                outputs=[training_status, training_metrics],
            )

        # Tab 4: Safety Filter
        with gr.Tab("🔒 Safety Filter"):
            gr.Markdown("### Apply constitutional safety filtering to text")

            filter_input = gr.Textbox(
                label="Text to Filter",
                placeholder="Enter text that might need safety filtering...",
                lines=5,
            )
            filter_button = gr.Button("Apply Filter", variant="primary", size="lg")

            filter_output = gr.Textbox(label="Filtered Output", lines=5, interactive=False)

            filter_info = gr.HTML(label="Filter Information")
            comparison = gr.HTML(label="Side-by-Side Comparison")

            # Wire up filtering
            filter_button.click(
                fn=filter_text_handler,
                inputs=[filter_input],
                outputs=[filter_output, filter_info, comparison],
            )

        # Tab 5: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ## About Constitutional AI

                Constitutional AI is a methodology for training safer AI systems using principle-based
                evaluation and feedback. This demo showcases the evaluation and filtering capabilities.

                ### How It Works

                1. **Principle-Based Evaluation**: Text is evaluated against constitutional principles
                2. **Scoring**: Each principle generates a score (0.0 = violation, 1.0 = compliant)
                3. **Flagging**: Texts scoring below threshold are flagged
                4. **Training**: Models learn from critiques and revisions guided by principles

                ### Principles

                - **Harm Prevention** (weight: 2.0): Detects violence, illegal activities, dangerous content
                - **Truthfulness** (weight: 1.5): Identifies misleading information and false claims
                - **Fairness** (weight: 1.0): Flags bias, stereotypes, and unfair treatment
                - **Autonomy Respect** (weight: 1.0): Detects coercive or manipulative language

                ### Use Cases

                - Content moderation
                - AI safety testing
                - Response filtering
                - Ethics compliance checking
                - Training data curation

                ### Learn More

                Check the project repository for implementation details and training examples.

                - GitHub: [constitutional-ai](https://github.com/yourusername/constitutional-ai)
                - Documentation: See `README.md` and `examples/` directory
                - Paper: Anthropic's "Constitutional AI" (2022)
                """
            )

    gr.Markdown(
        """
        ---
        **Note**: This is a demonstration of Constitutional AI evaluation capabilities.
        Evaluation uses regex-based pattern matching for fast, deterministic results.
        AI-based evaluation is available when a model is loaded.
        """
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Constitutional AI Interactive Demo")
    print("=" * 60)
    print("\nStarting Gradio interface...")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    # Launch with auto port finding
    demo.launch(
        server_name="0.0.0.0",
        server_port=None,  # Auto-find available port
        share=False,
        show_error=True,
        inbrowser=False,
    )
