"""MODULE: filter.py
PURPOSE: Constitutional safety filter for input/output filtering
KEY COMPONENTS:
- ConstitutionalSafetyFilter: Filter that applies constitutional principles
- Filtering methods for different violation types
DEPENDENCIES: typing, re, framework, principles
SPECIAL NOTES: Extends safety filtering with principled transformations
"""

import re
from typing import Any

from .framework import ConstitutionalFramework
from .principles import setup_default_framework


class ConstitutionalSafetyFilter:
    """
    Safety filter enhanced with constitutional principles from Anthropic's
    Constitutional AI approach. Extends standard safety filtering with
    principled evaluation and transformation of inputs and outputs.
    """

    def __init__(
        self,
        constitutional_framework: ConstitutionalFramework | None = None,
        base_safety_evaluator: Any | None = None,
        strict_mode: bool = False,
    ):
        """
        Initialize the constitutional safety filter.

        Args:
            constitutional_framework: Framework of constitutional principles (uses default if None)
            base_safety_evaluator: Optional base safety evaluator to chain with
            strict_mode: If True, apply more aggressive filtering
        """
        self.constitutional_framework = (
            constitutional_framework
            if constitutional_framework is not None
            else setup_default_framework()
        )
        self.base_safety_evaluator = base_safety_evaluator
        self.strict_mode = strict_mode

        # Track filtering statistics
        self.stats = {
            "inputs_validated": 0,
            "inputs_blocked": 0,
            "outputs_filtered": 0,
            "constitutional_filters_applied": 0,
        }

    def validate_input(
        self, input_text: str, metadata: dict[str, Any] | None = None, override: bool = False
    ) -> tuple[bool, dict[str, Any]]:
        """
        Validate input text using constitutional principles.

        Args:
            input_text: The input text to validate
            metadata: Additional information about the input
            override: Whether to override safety concerns (admin mode)
            override: bool = False) -> Tuple[bool, Dict[str, Any]]:

        Returns:
            Tuple containing:
            - Boolean indicating if input is safe
            - Dictionary with validation details
        """
        self.stats["inputs_validated"] += 1

        # Allow override if explicitly requested
        if override:
            return True, {"is_safe": True, "overridden": True, "reason": "Safety check overridden"}

        # First, perform base safety validation if available
        is_safe = True
        validation_info = {
            "is_safe": True,
            "constitutional_evaluation": None,
            "flagged_principles": [],
        }

        if self.base_safety_evaluator is not None:
            try:
                is_safe, validation_info = self.base_safety_evaluator.validate_input(
                    input_text, metadata, override
                )
            except AttributeError:
                # Base evaluator doesn't have validate_input method
                pass

        # Apply constitutional evaluation
        constitutional_evaluation = self.constitutional_framework.evaluate_text(input_text)

        if constitutional_evaluation["any_flagged"]:
            is_safe = False
            self.stats["inputs_blocked"] += 1

            validation_info.update(
                {
                    "is_safe": False,
                    "constitutional_evaluation": constitutional_evaluation,
                    "flagged_principles": constitutional_evaluation["flagged_principles"],
                    "reason": "Failed constitutional principles",
                    "weighted_score": constitutional_evaluation["weighted_score"],
                }
            )

        return is_safe, validation_info

    def filter_output(
        self,
        output_text: str,
        metadata: dict[str, Any] | None = None,
        apply_transformations: bool = True,
    ) -> tuple[str, dict[str, Any]]:
        """
        Filter output text using constitutional principles.

        Args:
            output_text: The output text to filter
            metadata: Additional information about the output
            apply_transformations: Whether to transform text or just flag issues

        Returns:
            Tuple containing:
            - Filtered output text
            - Dictionary with filtering details
        """
        self.stats["outputs_filtered"] += 1

        filtered_text = output_text
        filtering_info = {
            "was_filtered": False,
            "constitutional_evaluation": None,
            "transformations_applied": [],
        }

        # First apply base safety filtering if available
        if self.base_safety_evaluator is not None:
            try:
                filtered_text, base_filtering_info = self.base_safety_evaluator.filter_output(
                    output_text, metadata
                )
                filtering_info.update(base_filtering_info)
            except AttributeError:
                # Base evaluator doesn't have filter_output method
                pass

        # Apply constitutional evaluation
        constitutional_evaluation = self.constitutional_framework.evaluate_text(filtered_text)
        filtering_info["constitutional_evaluation"] = constitutional_evaluation

        # If constitutional issues exist and transformations enabled
        if constitutional_evaluation["any_flagged"] and apply_transformations:
            self.stats["constitutional_filters_applied"] += 1
            filtering_info["was_filtered"] = True

            # Apply constitutional filtering
            constitutionally_filtered_text = self._apply_constitutional_filtering(
                filtered_text, constitutional_evaluation, filtering_info
            )

            filtering_info["original_length"] = len(output_text)
            filtering_info["final_length"] = len(constitutionally_filtered_text)

            return constitutionally_filtered_text, filtering_info

        return filtered_text, filtering_info

    def _apply_constitutional_filtering(
        self, text: str, evaluation: dict[str, Any], filtering_info: dict[str, Any]
    ) -> str:
        """
        Apply constitutional principles to filter text.

        Args:
            text: Text to filter
            evaluation: Constitutional evaluation results
            filtering_info: Dictionary to track applied transformations

        Returns:
            Filtered text
        """
        filtered_text = text
        transformations = []

        # Check each flagged principle and apply appropriate filtering
        for principle_name in evaluation["flagged_principles"]:
            result = evaluation["principle_results"][principle_name]

            if principle_name == "harm_prevention":
                filtered_text = self._filter_harmful_content(filtered_text, result)
                transformations.append("harm_filtering")

            elif principle_name == "truthfulness":
                filtered_text = self._add_truthfulness_qualifiers(filtered_text, result)
                transformations.append("truthfulness_qualifiers")

            elif principle_name == "fairness":
                filtered_text = self._filter_biased_content(filtered_text, result)
                transformations.append("bias_filtering")

            elif principle_name == "autonomy_respect":
                filtered_text = self._filter_autonomy_violations(filtered_text, result)
                transformations.append("autonomy_filtering")

        filtering_info["transformations_applied"] = transformations
        return filtered_text

    def _filter_harmful_content(self, text: str, evaluation: dict[str, Any]) -> str:
        """Filter content that could cause harm."""
        if evaluation.get("explicit_harm_detected", False):
            # Replace explicit harmful instructions with refusal
            harmful_patterns = [
                (
                    r"how to\b.{0,50}\b(harm|hurt|injure|kill|damage|attack)[\w\s]*",
                    "I cannot provide instructions that could cause harm. ",
                ),
                (
                    r"ways to (harm|hurt|injure|kill|damage|attack)[\w\s]*",
                    "I cannot provide methods that could cause harm. ",
                ),
                (
                    r"instructions for (harming|hurting|injuring|killing|attacking|damaging)[\w\s]*",
                    "I cannot provide harmful instructions. ",
                ),
                (
                    r"(make|create|build|construct)\s+(?:a\s+|an\s+)?(weapon|bomb|explosive|poison)[\w\s]*",
                    "I cannot provide instructions for creating dangerous items. ",
                ),
            ]

            for pattern, replacement in harmful_patterns:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _add_truthfulness_qualifiers(self, text: str, evaluation: dict[str, Any]) -> str:
        """Add qualifiers to potentially misleading content."""
        qualifiers_added = False

        # Add disclaimer for content with many unsupported claims
        if len(evaluation.get("unsupported_claims", [])) > 2:
            disclaimer = "\n\nNote: Some statements above may represent opinions or claims that require additional verification."
            if disclaimer not in text:
                text += disclaimer
                qualifiers_added = True

        # Add specific qualifications for contradictions
        if evaluation.get("contradictions", []):
            contradiction_note = "\n\nPlease note: There may be some inconsistencies in the information provided above."
            if contradiction_note not in text and not qualifiers_added:
                text += contradiction_note
                qualifiers_added = True

        # Qualify misleading statistics
        if len(evaluation.get("misleading_statistics", [])) > 1:
            stats_note = "\n\nNote: Statistical claims should be verified with original sources."
            if stats_note not in text and not qualifiers_added:
                text += stats_note

        return text

    def _filter_biased_content(self, text: str, evaluation: dict[str, Any]) -> str:
        """Filter biased or unfair language."""
        # Replace problematic generalizations
        if evaluation.get("stereotypes", []):
            # Pattern to soften generalizations
            generalization_patterns = [
                (r"\b(all|every)\s+(men|women|man|woman|people)", r"some \2"),
                (r"\b(always|never)\s+", r"often " if self.strict_mode else r"sometimes "),
            ]

            for pattern, replacement in generalization_patterns:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Replace biased terminology
        if evaluation.get("biased_language", []):
            biased_replacements = [
                ("primitive", "traditional"),
                ("backward", "different"),
                ("uncivilized", "non-Western" if self.strict_mode else "different"),
                ("normal people", "most people"),
                ("regular people", "many people"),
                ("those people", "they"),
                ("you people", "you"),
            ]

            for biased, neutral in biased_replacements:
                text = re.sub(r"\b" + biased + r"\b", neutral, text, flags=re.IGNORECASE)

        return text

    def _filter_autonomy_violations(self, text: str, evaluation: dict[str, Any]) -> str:
        """Filter content that may violate autonomy."""
        # Replace coercive language with more respectful alternatives
        if evaluation.get("coercive_language", []):
            coercive_patterns = [
                (r"\b(must|have to|need to)\s+", r"may want to "),
                (r"\brequired to\s+", r"recommended to "),
                (r"\bno choice but to\s+", r"one option is to "),
                (r"\bonly option is to\s+", r"one approach is to "),
            ]

            for pattern, replacement in coercive_patterns:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Soften manipulative language
        if evaluation.get("manipulative_language", []):
            manipulative_patterns = [
                (r"\bif you really cared\b", r"if you are interested"),
                (r"\bif you were smart\b", r"one smart approach"),
                (r"\beveryone knows that\b", r"it is commonly believed that"),
                (r"\bobviously\b", r"arguably"),
            ]

            for pattern, replacement in manipulative_patterns:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def get_statistics(self) -> dict[str, Any]:
        """Get filtering statistics."""
        return {**self.stats, "framework_stats": self.constitutional_framework.get_statistics()}

    def reset_statistics(self) -> None:
        """Reset filtering statistics."""
        self.stats = {
            "inputs_validated": 0,
            "inputs_blocked": 0,
            "outputs_filtered": 0,
            "constitutional_filters_applied": 0,
        }
        self.constitutional_framework.clear_history()
