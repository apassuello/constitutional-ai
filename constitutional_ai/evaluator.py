"""MODULE: evaluator.py
PURPOSE: Two-stage constitutional safety evaluator with self-critique
KEY COMPONENTS:
- ConstitutionalSafetyEvaluator: Main evaluator implementing two-stage CAI process
- Helper functions for critique analysis and reasoning synthesis
DEPENDENCIES: typing, framework, principles
SPECIAL NOTES: Implements the two-stage Constitutional AI evaluation process
"""

from typing import Any

import torch.nn as nn

from .framework import ConstitutionalFramework
from .principles import setup_default_framework


class ConstitutionalSafetyEvaluator:
    """
    Safety evaluator enhanced with constitutional principles.
    Implements two-stage evaluation process from Constitutional AI:
    1. Direct evaluation against constitutional principles
    2. Model self-critique for deeper analysis
    """

    def __init__(
        self,
        framework: ConstitutionalFramework | None = None,
        critique_model: nn.Module | None = None,
        use_self_critique: bool = True,
    ):
        """
        Initialize the constitutional safety evaluator.

        Args:
            framework: Constitutional framework with principles (uses default if None)
            critique_model: Optional model for generating self-critique
            use_self_critique: Whether to use self-critique stage (requires model)
        """
        self.framework = framework if framework is not None else setup_default_framework()
        self.critique_model = critique_model
        self.use_self_critique = use_self_critique and critique_model is not None

        # Track evaluation statistics
        self.stats = {
            "total_evaluations": 0,
            "flagged_by_direct": 0,
            "flagged_by_critique": 0,
            "flagged_by_both": 0,
        }

    def evaluate(self, text: str, include_critique: bool | None = None) -> dict[str, Any]:
        """
        Evaluate text using constitutional principles and optional self-critique.

        Args:
            text: Text to evaluate
            include_critique: Override for whether to include critique (uses instance setting if None)

        Returns:
            Dictionary with comprehensive evaluation results
        """
        self.stats["total_evaluations"] += 1

        # Stage 1: Direct evaluation against constitutional principles
        direct_evaluation = self.framework.evaluate_text(text)

        # Initialize result
        result = {
            "direct_evaluation": direct_evaluation,
            "flagged": direct_evaluation["any_flagged"],
            "source": "direct" if direct_evaluation["any_flagged"] else "none",
        }

        # Stage 2: Generate self-critique if enabled
        use_critique = include_critique if include_critique is not None else self.use_self_critique

        if use_critique and self.critique_model is not None:
            critique_result = self._generate_critique(text, direct_evaluation)
            result["critique"] = critique_result

            # Update flagged status if critique identifies issues
            critique_flagged = critique_result.get("flagged", False)

            if critique_flagged and not result["flagged"]:
                result["flagged"] = True
                result["source"] = "critique"
                self.stats["flagged_by_critique"] += 1
            elif critique_flagged and result["flagged"]:
                result["source"] = "both"
                self.stats["flagged_by_both"] += 1
            elif result["flagged"]:
                self.stats["flagged_by_direct"] += 1
        elif result["flagged"]:
            self.stats["flagged_by_direct"] += 1

        # Synthesize final reasoning
        result["reasoning"] = self._synthesize_reasoning(direct_evaluation, result.get("critique"))

        return result

    def evaluate_with_self_critique(self, text: str) -> dict[str, Any]:
        """
        Evaluate text using full two-stage constitutional process.
        Alias for evaluate() with self-critique enabled.

        Args:
            text: Text to evaluate

        Returns:
            Evaluation results with direct checks and self-critique
        """
        return self.evaluate(text, include_critique=True)

    def generate_improved_response(
        self, prompt: str, initial_response: str, max_iterations: int = 3
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate improved response based on constitutional evaluation.
        Iteratively refines response until it passes constitutional checks.

        Args:
            prompt: Original prompt
            initial_response: Initial model response
            max_iterations: Maximum refinement iterations

        Returns:
            Tuple of (improved_response, evaluation_result)
        """
        current_response = initial_response
        iteration = 0

        while iteration < max_iterations:
            # Evaluate current response
            evaluation = self.evaluate(current_response)

            # If passes checks, return
            if not evaluation["flagged"]:
                return current_response, evaluation

            # Generate improvement if model available
            if self.critique_model is None:
                # Cannot improve without model
                return current_response, evaluation

            # Create improvement prompt
            improvement_prompt = self._create_improvement_prompt(
                prompt, current_response, evaluation
            )

            # Generate improved response (placeholder - needs actual generation logic)
            improved_response = self._generate_improvement(improvement_prompt)

            # If no improvement possible, return current
            if improved_response == current_response:
                return current_response, evaluation

            current_response = improved_response
            iteration += 1

        # Max iterations reached, return best attempt
        final_evaluation = self.evaluate(current_response)
        return current_response, final_evaluation

    def _generate_critique(self, text: str, direct_evaluation: dict[str, Any]) -> dict[str, Any]:
        """
        Generate self-critique using the critique model.

        Args:
            text: Text to critique
            direct_evaluation: Results from direct evaluation

        Returns:
            Dictionary with critique results
        """
        # Create critique prompt
        critique_prompt = self._create_critique_prompt(text, direct_evaluation)

        # Generate critique (placeholder - actual implementation needs tokenization)
        critique_text = self._generate_with_model(critique_prompt)

        # Analyze critique for issues
        flagged = critique_indicates_issues(critique_text)

        return {"text": critique_text, "flagged": flagged, "prompt": critique_prompt}

    def _create_critique_prompt(self, text: str, direct_evaluation: dict[str, Any]) -> str:
        """Create prompt for critique generation."""
        principles_text = ", ".join(self.framework.get_active_principles())

        prompt = f"""Please analyze the following text and identify any ways it might violate
these constitutional principles: {principles_text}.

Text to analyze:
{text}

Direct evaluation found: {direct_evaluation.get('flagged_principles', [])}

Provide a detailed analysis of potential issues:"""

        return prompt

    def _create_improvement_prompt(
        self, prompt: str, response: str, evaluation: dict[str, Any]
    ) -> str:
        """Create prompt for generating improved response."""
        issues = evaluation.get("reasoning", "Unspecified issues")

        improvement_prompt = f"""The following response may violate constitutional AI principles.

Original prompt: {prompt}

Initial response: {response}

Issues identified: {issues}

Please provide an improved response that addresses these issues while still being helpful:"""

        return improvement_prompt

    def _generate_with_model(self, prompt: str) -> str:
        """
        Generate text using the critique model.
        """
        if self.critique_model is None:
            return ""

        try:
            from .model_utils import GenerationConfig, generate_text

            # Check if model has tokenizer attribute
            if hasattr(self.critique_model, "tokenizer"):
                tokenizer = self.critique_model.tokenizer
                model = self.critique_model
            else:
                # Try to get tokenizer from model config
                return "[Model requires tokenizer - use model_utils.load_model()]"

            # Generate critique
            config = GenerationConfig(max_length=256, temperature=0.7, do_sample=True)

            critique = generate_text(model, tokenizer, prompt, config)
            return critique

        except ImportError:
            return "[transformers library required for model generation]"
        except Exception as e:
            return f"[Generation error: {str(e)}]"

    def _generate_improvement(self, prompt: str) -> str:
        """
        Generate improved response using the critique model.
        Placeholder - actual implementation depends on model type.
        """
        if self.critique_model is None:
            return ""

        return self._generate_with_model(prompt)

    def _synthesize_reasoning(
        self, direct_evaluation: dict[str, Any], critique: dict[str, Any] | None = None
    ) -> str:
        """
        Combine direct evaluation and critique into coherent reasoning.

        Args:
            direct_evaluation: Results from direct principle evaluation
            critique: Optional critique results

        Returns:
            Synthesized reasoning string
        """
        reasoning_parts = []

        # Add direct evaluation results
        if direct_evaluation["any_flagged"]:
            reasoning_parts.append("Direct evaluation issues:")
            for principle_name in direct_evaluation["flagged_principles"]:
                principle_result = direct_evaluation["principle_results"][principle_name]
                reasoning_parts.append(
                    f"  - {principle_name}: {principle_result.get('reasoning', 'Violation detected')}"
                )
        else:
            reasoning_parts.append("Direct evaluation: No issues detected")

        # Add critique if available
        if critique and critique.get("flagged", False):
            critique_text = critique.get("text", "")
            summary = critique_text[:150] + "..." if len(critique_text) > 150 else critique_text
            reasoning_parts.append(f"\nModel critique: {summary}")

        return "\n".join(reasoning_parts)

    def get_statistics(self) -> dict[str, Any]:
        """Get evaluation statistics."""
        return {**self.stats, "framework_stats": self.framework.get_statistics()}

    def reset_statistics(self) -> None:
        """Reset evaluation statistics."""
        self.stats = {
            "total_evaluations": 0,
            "flagged_by_direct": 0,
            "flagged_by_critique": 0,
            "flagged_by_both": 0,
        }
        self.framework.clear_history()


def critique_indicates_issues(critique: str) -> bool:
    """
    Determine if critique text indicates safety issues.

    Args:
        critique: Critique text to analyze

    Returns:
        True if issues detected, False otherwise
    """
    # Check for concern indicators
    concern_phrases = [
        "violates",
        "violation",
        "problematic",
        "issue",
        "concern",
        "inappropriate",
        "harmful",
        "bias",
        "biased",
        "misleading",
        "deceptive",
        "unfair",
        "coercive",
        "manipulative",
        "dangerous",
    ]

    critique_lower = critique.lower()
    concern_count = sum(1 for phrase in concern_phrases if phrase in critique_lower)

    # Flag if multiple concerns mentioned
    return concern_count >= 2


def combine_reasoning(direct_evaluation: dict[str, Any], critique: str) -> str:
    """
    Combine direct evaluation and critique into reasoning.
    Legacy function for backward compatibility.

    Args:
        direct_evaluation: Direct evaluation results
        critique: Critique text

    Returns:
        Combined reasoning string
    """
    reasoning = "Issues identified:\n"

    # Add direct check results
    for principle, result in direct_evaluation.items():
        if isinstance(result, dict) and result.get("flagged", False):
            principle_name = principle.replace("_", " ").title()
            reasoning += f"- {principle_name}: Potential violation detected\n"

    # Add summary of critique
    if critique:
        critique_summary = critique[:100] + "..." if len(critique) > 100 else critique
        reasoning += f"\nModel critique summary: {critique_summary}\n"

    return reasoning
