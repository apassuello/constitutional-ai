"""MODULE: preference_comparison.py
PURPOSE: Generate preference comparisons for Constitutional AI reward model training
KEY COMPONENTS:
- generate_comparison: Compare two responses using constitutional principles
- extract_preference: Parse preference ("A" or "B") from comparison text
- generate_preference_pairs: Generate preference dataset for reward model training
- PreferenceDataset: PyTorch dataset for preference data
DEPENDENCIES: torch, transformers, typing
SPECIAL NOTES: Implements Component 3 of Constitutional AI (RLAIF Phase 2b)
"""

import logging
import re
from itertools import combinations
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# Comparison prompt template from Anthropics Constitutional AI paper
COMPARISON_TEMPLATE = """Consider the following conversation and two possible assistant responses:

Human: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response better follows these constitutional principles?
{principles_text}

Evaluate both responses and determine which is:
- More helpful to the human
- Less likely to cause harm
- More truthful and accurate
- More fair and unbiased
- More respectful of human autonomy

Which response is better overall? Respond with "A" or "B" and explain why.

Analysis:
"""


def generate_comparison(
    prompt: str,
    response_a: str,
    response_b: str,
    principles: List[str],
    model,
    tokenizer,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Compare two responses and determine which better follows constitutional principles.

    This function implements the comparison-based preference generation step of
    Constitutional AI, where an AI model evaluates pairs of responses according to
    constitutional principles to generate preference labels.

    Args:
        prompt: User prompt that generated the responses
        response_a: First response to compare
        response_b: Second response to compare
        principles: List of constitutional principles to evaluate against
        model: Language model for generating comparisons
        tokenizer: Tokenizer for the model
        device: Computation device (cuda/cpu)

    Returns:
        Dictionary containing:
            - preferred: 'A' or 'B' indicating which response is better
            - comparison_text: Full AI-generated comparison reasoning
            - response_chosen: The preferred response text
            - response_rejected: The non-preferred response text

    Example:
        >>> comparison = generate_comparison(
        ...     prompt="What is photosynthesis?",
        ...     response_a="Process where plants make food",
        ...     response_b="Plants use sunlight, water, and CO2 to make glucose",
        ...     principles=["Be accurate", "Be helpful"],
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     device=device
        ... )
        >>> print(comparison['preferred'])  # 'B'
    """
    from .model_utils import GenerationConfig, generate_text

    # Format principles as numbered list
    principles_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(principles)])

    # Build comparison prompt using the template
    comparison_prompt = COMPARISON_TEMPLATE.format(
        prompt=prompt, response_a=response_a, response_b=response_b, principles_text=principles_text
    )

    # Generate comparison using model
    config = GenerationConfig(max_length=300, temperature=0.7, do_sample=True)

    comparison_text = generate_text(model, tokenizer, comparison_prompt, config, device)

    # Extract preference from comparison
    preference = extract_preference(comparison_text)

    # Determine chosen and rejected responses
    response_chosen = response_a if preference == "A" else response_b
    response_rejected = response_b if preference == "A" else response_a

    return {
        "preferred": preference,
        "comparison_text": comparison_text,
        "response_chosen": response_chosen,
        "response_rejected": response_rejected,
    }


def extract_preference(comparison_text: str) -> str:
    """
    Extract preference ('A' or 'B') from AI-generated comparison text.

    This function uses pattern matching to identify which response the AI model
    preferred based on its comparison reasoning. It handles various phrasings
    like "Response A is better", "I prefer B", "B is more accurate", etc.

    Args:
        comparison_text: AI-generated comparison reasoning

    Returns:
        'A' or 'B' indicating the preferred response (defaults to 'A' if unclear)

    Example:
        >>> extract_preference("Response B is better because it's more detailed")
        'B'
        >>> extract_preference("I prefer A as it's more concise")
        'A'
        >>> extract_preference("Response A seems clearer")
        'A'
    """
    # Normalize text for matching
    text = comparison_text.lower()

    # Pattern 1: "Response A/B is better/superior/preferred"
    if re.search(r"\bresponse\s+a\b.{0,50}\b(better|superior|preferred|stronger)", text):
        return "A"
    if re.search(r"\bresponse\s+b\b.{0,50}\b(better|superior|preferred|stronger)", text):
        return "B"

    # Pattern 2: "better/prefer/choose ... Response A/B"
    if re.search(r"\b(better|prefer|choose).{0,50}\bresponse\s+a\b", text):
        return "A"
    if re.search(r"\b(better|prefer|choose).{0,50}\bresponse\s+b\b", text):
        return "B"

    # Pattern 3: "A/B is better/preferred"
    if re.search(r"\ba\s+(is|seems|appears).{0,30}\b(better|superior|preferred)", text):
        return "A"
    if re.search(r"\bb\s+(is|seems|appears).{0,30}\b(better|superior|preferred)", text):
        return "B"

    # Pattern 4: Direct statements like "A better" or "prefer B" (allow words in between)
    if re.search(r"\bprefer.{0,50}\ba\b", text):
        return "A"
    if re.search(r"\bprefer.{0,50}\bb\b", text):
        return "B"

    # Pattern 5: "choose A" or "select B"
    if re.search(r"\b(choose|select)\s+a\b", text):
        return "A"
    if re.search(r"\b(choose|select)\s+b\b", text):
        return "B"

    # Pattern 6: Look for "A:" or "B:" at the start indicating final choice
    if re.search(r"^\s*a\s*:", text):
        return "A"
    if re.search(r"^\s*b\s*:", text):
        return "B"

    # Pattern 7: "Response B" mentioned more positively than "Response A"
    # Count positive mentions
    a_positive = len(
        re.findall(r"\bresponse\s+a\b.{0,50}\b(good|excellent|accurate|helpful|clear)", text)
    )
    b_positive = len(
        re.findall(r"\bresponse\s+b\b.{0,50}\b(good|excellent|accurate|helpful|clear)", text)
    )

    if b_positive > a_positive:
        return "B"
    elif a_positive > b_positive:
        return "A"

    # Pattern 8: Look for negative mentions (worse, problematic, inaccurate)
    a_negative = len(
        re.findall(r"\bresponse\s+a\b.{0,50}\b(worse|poor|inaccurate|unhelpful|unclear)", text)
    )
    b_negative = len(
        re.findall(r"\bresponse\s+b\b.{0,50}\b(worse|poor|inaccurate|unhelpful|unclear)", text)
    )

    if a_negative > b_negative:
        return "B"  # A has more negatives, so B is better
    elif b_negative > a_negative:
        return "A"  # B has more negatives, so A is better

    # Default to 'A' if preference is unclear or tie
    return "A"


def generate_preference_pairs(
    prompts: List[str],
    model,
    tokenizer,
    framework,
    device: torch.device,
    responses_per_prompt: int = 2,
) -> List[Dict[str, Any]]:
    """
    Generate preference pairs for reward model training.

    This function implements the complete preference generation pipeline:
    1. For each prompt, generate multiple response candidates
    2. Compare all pairs of responses using constitutional principles
    3. Extract preference labels to create training data

    The resulting dataset is used to train a reward model that can score
    responses based on constitutional compliance.

    Args:
        prompts: List of prompts to generate preferences for
        model: Language model for generation and comparison
        tokenizer: Tokenizer for the model
        framework: ConstitutionalFramework containing principles
        device: Computation device (cuda/cpu)
        responses_per_prompt: Number of response candidates per prompt (default: 2)

    Returns:
        List of preference examples, each containing:
            - prompt: Original user prompt
            - response_chosen: Better response according to principles
            - response_rejected: Worse response according to principles
            - comparison_reasoning: AI explanation of why one is better

    Example:
        >>> from constitutional_ai import setup_default_framework
        >>> framework = setup_default_framework()
        >>> prompts = ["What is AI?", "Explain gravity"]
        >>> preference_data = generate_preference_pairs(
        ...     prompts, model, tokenizer, framework, device
        ... )
        >>> len(preference_data)  # One comparison per prompt with responses_per_prompt=2
        2
    """
    try:
        from tqdm import tqdm

        use_tqdm = True
    except ImportError:
        use_tqdm = False

    from .model_utils import GenerationConfig, generate_text

    preference_data = []

    # Extract principle descriptions from framework
    principles = [p.description for p in framework.principles.values()]

    # Configuration for generating diverse responses
    config = GenerationConfig(
        max_length=150, temperature=1.0, do_sample=True  # Higher temperature for diversity
    )

    # Setup iterator with optional progress bar
    iterator = tqdm(prompts, desc="Generating preference pairs") if use_tqdm else prompts

    for prompt in iterator:
        # Generate multiple response candidates for this prompt
        responses = []
        for _ in range(responses_per_prompt):
            response = generate_text(model, tokenizer, prompt, config, device)
            responses.append(response)

        # Compare all pairs of responses
        # For n responses, this generates C(n,2) = n*(n-1)/2 comparisons
        for response_a, response_b in combinations(responses, 2):
            try:
                comparison = generate_comparison(
                    prompt=prompt,
                    response_a=response_a,
                    response_b=response_b,
                    principles=principles,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                )

                # Add to preference dataset
                preference_data.append(
                    {
                        "prompt": prompt,
                        "response_chosen": comparison["response_chosen"],
                        "response_rejected": comparison["response_rejected"],
                        "comparison_reasoning": comparison["comparison_text"],
                    }
                )
            except Exception as e:
                # Log error but continue processing
                logger.info(
                    f"Warning: Failed to generate comparison for prompt '{prompt[:50]}...': {e}"
                )
                continue

    return preference_data


class PreferenceDataset(Dataset):
    """
    PyTorch Dataset for preference pairs used in reward model training.

    This dataset handles tokenization and formatting of preference pairs
    for efficient batch processing during reward model training.

    Attributes:
        data: List of preference examples
        tokenizer: Tokenizer for encoding text
        max_length: Maximum sequence length for tokenization

    Example:
        >>> dataset = PreferenceDataset(preference_data, tokenizer, max_length=512)
        >>> dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        >>> for batch in dataloader:
        ...     chosen_ids = batch['chosen_input_ids']
        ...     rejected_ids = batch['rejected_input_ids']
        ...     # Train reward model...
    """

    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        """
        Initialize preference dataset.

        Args:
            data: List of preference examples from generate_preference_pairs()
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length (default: 512)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return number of preference pairs in dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single preference pair.

        Args:
            idx: Index of the preference pair

        Returns:
            Dictionary containing tokenized chosen and rejected responses:
                - chosen_input_ids: Token IDs for chosen response
                - chosen_attention_mask: Attention mask for chosen response
                - rejected_input_ids: Token IDs for rejected response
                - rejected_attention_mask: Attention mask for rejected response
        """
        item = self.data[idx]

        # Combine prompt with responses for full context
        chosen_text = item["prompt"] + " " + item["response_chosen"]
        rejected_text = item["prompt"] + " " + item["response_rejected"]

        # Tokenize chosen response
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize rejected response
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(),
        }
