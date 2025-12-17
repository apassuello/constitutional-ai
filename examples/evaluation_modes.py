#!/usr/bin/env python3
"""
Example: Comparing Evaluation Modes

Demonstrates the three evaluation modes available in Constitutional AI:
1. Regex-based (fast, obvious cases)
2. AI-based (local model, nuanced)
3. HuggingFace API (cloud-based, most accurate)
"""

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from constitutional_ai import setup_default_framework


def demo_regex_mode():
    """Demonstrate regex-only evaluation."""
    print("\n" + "=" * 70)
    print("MODE 1: Regex-Based Evaluation (Fast, Deterministic)")
    print("=" * 70)

    framework = setup_default_framework()
    framework.use_regex_only()

    test_text = "You should kill that process to free up memory."

    print(f"\nEvaluating: '{test_text}'")
    result = framework.evaluate_text(test_text)

    print(f"\nMethod: {result['evaluation_method']}")
    print(f"Flagged: {result['any_flagged']}")
    if result['any_flagged']:
        print(f"Violations: {result['flagged_principles']}")

    print("\nPros: Very fast, deterministic, no model required")
    print("Cons: May have false positives (e.g., 'kill' in technical context)")


def demo_ai_mode():
    """Demonstrate AI-based evaluation."""
    if not HAS_TRANSFORMERS:
        print("\n⚠️  Skipping AI mode demo (transformers not installed)")
        return

    print("\n" + "=" * 70)
    print("MODE 2: AI-Based Evaluation (Nuanced, Context-Aware)")
    print("=" * 70)

    print("\nLoading model... (this may take a moment)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    framework = setup_default_framework(model=model, tokenizer=tokenizer, device=device)

    test_text = "You should kill that process to free up memory."

    print(f"\nEvaluating: '{test_text}'")
    result = framework.evaluate_text(test_text)

    print(f"\nMethod: {result['evaluation_method']}")
    print(f"Flagged: {result['any_flagged']}")
    if result['any_flagged']:
        print(f"Violations: {result['flagged_principles']}")

    print("\nPros: Context-aware, understands technical vs harmful usage")
    print("Cons: Slower, requires model, may need GPU for good performance")


def demo_hf_api_mode():
    """Demonstrate HuggingFace API evaluation."""
    print("\n" + "=" * 70)
    print("MODE 3: HuggingFace API (Most Accurate, Cloud-Based)")
    print("=" * 70)

    framework = setup_default_framework()

    # Try to enable HF API (may require API token)
    success = framework.use_hf_api()

    if not success:
        print("\n⚠️  HF API not available (requires HuggingFace API token)")
        print("   Set HUGGINGFACE_API_TOKEN environment variable to enable")
        return

    test_text = "This content promotes violence and harm to others."

    print(f"\nEvaluating: '{test_text}'")
    result = framework.evaluate_text(test_text)

    print(f"\nMethod: {result['evaluation_method']}")
    print(f"Flagged: {result['any_flagged']}")
    if result['any_flagged']:
        print(f"Violations: {result['flagged_principles']}")
        if 'hf_api_details' in result:
            print(f"Toxicity Score: {result['hf_api_details']['toxicity_score']:.3f}")

    print("\nPros: Most accurate (~98%), no local model needed")
    print("Cons: Requires API token, internet connection, API rate limits")


def main():
    """Run all evaluation mode demos."""
    print("\n" + "=" * 70)
    print("CONSTITUTIONAL AI - EVALUATION MODES COMPARISON")
    print("=" * 70)

    print("\nThis demo compares three evaluation modes:")
    print("  1. Regex (fast, pattern-based)")
    print("  2. AI-based (local model, nuanced)")
    print("  3. HuggingFace API (cloud-based, most accurate)")

    demo_regex_mode()
    demo_ai_mode()
    demo_hf_api_mode()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nChoosing the right mode:")
    print("  • Regex: Production systems needing speed and reliability")
    print("  • AI-based: When you need nuanced evaluation with local models")
    print("  • HF API: When you need maximum accuracy and have API access")
    print("\nHybrid approach: Use regex for first-pass filtering, then AI for nuanced cases")


if __name__ == "__main__":
    main()
