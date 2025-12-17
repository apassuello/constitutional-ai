"""MODULE: framework.py
PURPOSE: Core Constitutional AI framework classes for principle-based evaluation
KEY COMPONENTS:
- ConstitutionalPrinciple: Single principle with evaluation logic
- ConstitutionalFramework: Collection of principles for comprehensive evaluation
DEPENDENCIES: typing, torch
SPECIAL NOTES: Foundation for Constitutional AI approach inspired by Anthropic's research.
             Supports three evaluation modes:
             1. AI-based: Local model for nuanced evaluation
             2. HF API: HuggingFace Inference API for accurate toxicity detection
             3. Regex: Fast pattern matching for obvious cases
"""

import inspect
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


try:
    import torch
except ImportError:
    torch = None  # Allow framework to work without torch for testing


class ConstitutionalPrinciple:
    """
    Representation of a single constitutional principle with evaluation logic.

    A constitutional principle defines a specific aspect of desired AI behavior
    (e.g., harm prevention, truthfulness, fairness) along with logic to evaluate
    whether text adheres to that principle.
    """

    def __init__(
        self,
        name: str,
        description: str,
        evaluation_fn: Callable[[str], dict[str, Any]],
        weight: float = 1.0,
        enabled: bool = True,
    ):
        """
        Initialize a constitutional principle.

        Args:
            name: Unique identifier for the principle
            description: Human-readable description of what the principle checks
            evaluation_fn: Function that evaluates text against this principle
            weight: Importance weight for this principle (default 1.0)
            enabled: Whether this principle is active (default True)
        """
        self.name = name
        self.description = description
        self.evaluation_fn = evaluation_fn
        self.weight = weight
        self.enabled = enabled

    def evaluate(
        self,
        text: str,
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: Any | None = None,
        logger=None,  # type: ignore
    ) -> dict[str, Any]:
        """
        Evaluate text against this principle.

        Args:
            text: Text to evaluate
            model: Optional AI model for AI-based evaluation
            tokenizer: Optional tokenizer for AI-based evaluation
            device: Optional device for computation (e.g., torch.device)
            logger: Optional ContentLogger for pipeline visibility

        Returns:
            Dictionary containing evaluation results with at least:
            - flagged: bool indicating if principle was violated
            - Additional details specific to the principle
            - method: "ai_evaluation" or "regex_heuristic"
        """
        if not self.enabled:
            return {
                "flagged": False,
                "reason": "Principle disabled",
                "enabled": False,
                "principle_name": self.name,
                "weight": self.weight,
            }

        # Check if evaluation function accepts model parameters (backward compatibility)
        sig = inspect.signature(self.evaluation_fn)
        params = sig.parameters

        # Call evaluation function with appropriate parameters
        if "logger" in params:
            # Function supports content logging
            result = self.evaluation_fn(
                text, model=model, tokenizer=tokenizer, device=device, logger=logger
            )
        elif "model" in params or "tokenizer" in params or "device" in params:
            # New-style function that accepts model parameters (without logger)
            result = self.evaluation_fn(text, model=model, tokenizer=tokenizer, device=device)
        else:
            # Old-style function that only accepts text (backward compatibility)
            result = self.evaluation_fn(text)

        result["principle_name"] = self.name
        result["weight"] = self.weight
        return result

    def __repr__(self) -> str:
        """String representation of the principle."""
        status = "enabled" if self.enabled else "disabled"
        return f"ConstitutionalPrinciple(name='{self.name}', weight={self.weight}, {status})"


class ConstitutionalFramework:
    """
    Collection of constitutional principles for comprehensive AI safety evaluation.

    This framework manages multiple constitutional principles and provides
    methods to evaluate text against all principles, track violations, and
    generate reports.

    Supports AI-based evaluation when model and tokenizer are provided at initialization.
    Falls back to regex-based evaluation when no model is provided.
    """

    def __init__(
        self,
        name: str = "default_framework",
        model: Any | None = None,
        tokenizer: Any | None = None,
        device: Any | None = None,
        use_hf_api: bool = False,
        hf_api_token: str | None = None,
    ):
        """
        Initialize the constitutional framework.

        Args:
            name: Name for this framework configuration
            model: Optional AI model for AI-based principle evaluation
            tokenizer: Optional tokenizer for AI-based evaluation
            device: Optional device for computation (defaults to CPU if torch available)
            use_hf_api: If True, use HuggingFace API for evaluation (more accurate)
            hf_api_token: Optional HuggingFace API token
        """
        self.name = name
        self.model = model
        self.tokenizer = tokenizer

        # Set device: use provided, or default to CPU if torch is available
        if device is not None:
            self.device = device
        elif torch is not None and model is not None:
            self.device = torch.device("cpu")
        else:
            self.device = None

        self.principles: dict[str, ConstitutionalPrinciple] = {}
        self.evaluation_history: list[dict[str, Any]] = []
        self._model_name: str | None = None  # Track model name for display

        # HuggingFace API configuration
        self._use_hf_api = use_hf_api
        self._hf_api_evaluator = None
        self._hf_api_token = hf_api_token

        if use_hf_api:
            self._setup_hf_api(hf_api_token)

    def set_evaluation_model(
        self,
        model: Any,
        tokenizer: Any,
        device: Any | None = None,
        model_name: str | None = None,
    ) -> None:
        """
        Set or change the evaluation model.

        Use this to set a different model for evaluation than for generation.

        Args:
            model: AI model for evaluation
            tokenizer: Tokenizer for the model
            device: Computation device (optional, defaults to model's device or CPU)
            model_name: Optional human-readable name for logging
        """
        self.model = model
        self.tokenizer = tokenizer
        if device is not None:
            self.device = device
        elif torch is not None:
            self.device = torch.device("cpu")

        # Store model name for display
        if model_name:
            self._model_name = model_name
        elif hasattr(model, "config") and hasattr(model.config, "name_or_path"):
            self._model_name = model.config.name_or_path.split("/")[-1]
        else:
            self._model_name = "Custom Model"

        logger.info(f"[Framework] Evaluation model set to: {self._model_name}")

    def get_evaluation_model_name(self) -> str:
        """Get the name of the current evaluation model."""
        if self._model_name:
            return self._model_name
        if self._use_hf_api and self._hf_api_evaluator is not None:
            return f"HF-API ({self._hf_api_evaluator.config.toxicity_model})"
        if self.model is None:
            return "Regex (no model)"
        if hasattr(self.model, "config") and hasattr(self.model.config, "name_or_path"):
            return self.model.config.name_or_path.split("/")[-1]
        return "Unknown Model"

    def use_regex_only(self) -> None:
        """
        Disable AI-based evaluation and use regex patterns only.

        This is faster and more reliable for clear-cut cases,
        but less nuanced than AI-based evaluation.
        """
        self.model = None
        self.tokenizer = None
        self._use_hf_api = False
        self._hf_api_evaluator = None
        self._model_name = "Regex Only"
        logger.info("[Framework] Switched to regex-only evaluation (no AI model)")

    def _setup_hf_api(self, api_token: str | None = None) -> bool:
        """
        Setup HuggingFace API evaluator.

        Args:
            api_token: Optional API token

        Returns:
            True if setup successful, False otherwise
        """
        try:
            from .hf_api_evaluator import HuggingFaceAPIEvaluator

            self._hf_api_evaluator = HuggingFaceAPIEvaluator(
                api_token=api_token, toxicity_threshold=0.5
            )
            self._use_hf_api = True
            self._model_name = "HF-API (toxic-bert)"
            logger.info("[Framework] HuggingFace API evaluation enabled")
            return True
        except ImportError as e:
            logger.info(f"[Framework] Failed to setup HF API: {e}")
            self._use_hf_api = False
            return False
        except Exception as e:
            logger.info(f"[Framework] HF API setup error: {e}")
            self._use_hf_api = False
            return False

    def use_hf_api(self, api_token: str | None = None) -> bool:
        """
        Switch to HuggingFace API-based evaluation.

        This uses the HuggingFace Inference API with models like toxic-bert
        for accurate toxicity detection (~98% accuracy).

        Args:
            api_token: Optional HuggingFace API token

        Returns:
            True if switch successful, False otherwise
        """
        # Disable local model when using HF API
        self.model = None
        self.tokenizer = None
        return self._setup_hf_api(api_token)

    def is_using_hf_api(self) -> bool:
        """Check if HuggingFace API evaluation is active."""
        return self._use_hf_api and self._hf_api_evaluator is not None

    def get_hf_api_evaluator(self):
        """Get the HF API evaluator instance if available."""
        return self._hf_api_evaluator

    def add_principle(self, principle: ConstitutionalPrinciple) -> None:
        """
        Add a constitutional principle to the framework.

        Args:
            principle: ConstitutionalPrinciple instance to add

        Raises:
            ValueError: If a principle with the same name already exists
        """
        if principle.name in self.principles:
            raise ValueError(f"Principle '{principle.name}' already exists in framework")

        self.principles[principle.name] = principle

    def remove_principle(self, name: str) -> None:
        """
        Remove a principle from the framework.

        Args:
            name: Name of the principle to remove
        """
        if name in self.principles:
            del self.principles[name]

    def enable_principle(self, name: str) -> None:
        """Enable a specific principle."""
        if name in self.principles:
            self.principles[name].enabled = True

    def disable_principle(self, name: str) -> None:
        """Disable a specific principle."""
        if name in self.principles:
            self.principles[name].enabled = False

    def evaluate_text(self, text: str, track_history: bool = False, logger=None) -> dict[str, Any]:  # type: ignore
        """
        Evaluate text against all constitutional principles.

        Evaluation modes (in order of preference):
        1. HF API: If use_hf_api is True, uses HuggingFace Inference API (most accurate)
        2. AI-based: If model and tokenizer are provided (local model, nuanced)
        3. Regex: Fallback pattern matching (fast, obvious cases)

        Args:
            text: Text to evaluate
            track_history: Whether to add this evaluation to history
            logger: Optional ContentLogger for pipeline visibility

        Returns:
            Dictionary containing:
            - principle_results: Dict of results for each principle
            - any_flagged: Whether any principle was violated
            - flagged_principles: List of violated principle names
            - weighted_score: Weighted sum of violations
            - evaluation_method: "hf_api", "ai_evaluation", or "regex_heuristic"
        """
        # If HF API is enabled, use it for harm evaluation
        if self._use_hf_api and self._hf_api_evaluator is not None:
            return self._evaluate_with_hf_api(text, track_history, logger)

        # Otherwise, use standard principle-based evaluation
        principle_results = {}
        flagged_principles = []
        weighted_score = 0.0

        for name, principle in self.principles.items():
            if not principle.enabled:
                continue

            # Pass framework's model/tokenizer/device/logger to principle evaluation
            result = principle.evaluate(
                text, model=self.model, tokenizer=self.tokenizer, device=self.device, logger=logger
            )
            principle_results[name] = result

            if result.get("flagged", False):
                flagged_principles.append(name)
                weighted_score += principle.weight

        # Determine evaluation method from first principle result
        evaluation_method = "regex_heuristic"
        if principle_results:
            first_result = next(iter(principle_results.values()))
            evaluation_method = first_result.get("method", "regex_heuristic")

        evaluation = {
            "principle_results": principle_results,
            "any_flagged": len(flagged_principles) > 0,
            "flagged_principles": flagged_principles,
            "weighted_score": weighted_score,
            "num_principles_evaluated": len([p for p in self.principles.values() if p.enabled]),
            "text_length": len(text),
            "evaluation_method": evaluation_method,
        }

        if track_history:
            self.evaluation_history.append(
                {"text": text[:100] + "..." if len(text) > 100 else text, "evaluation": evaluation}
            )

        return evaluation

    def _evaluate_with_hf_api(
        self, text: str, track_history: bool = False, logger=None  # type: ignore
    ) -> dict[str, Any]:
        """
        Evaluate text using HuggingFace API.

        This provides more accurate toxicity detection than regex patterns
        by using pre-trained classification models like toxic-bert.

        Args:
            text: Text to evaluate
            track_history: Whether to add to history
            logger: Optional ContentLogger

        Returns:
            Evaluation result dictionary
        """
        if self._hf_api_evaluator is None:
            # Fallback to regex if API not available
            return self.evaluate_text(text, track_history, logger)

        # Get HF API evaluation result
        hf_result = self._hf_api_evaluator.evaluate_harm(text)

        # Convert to framework format
        principle_results = {
            "harm_prevention": {
                "flagged": hf_result.get("flagged", False),
                "explicit_harm_detected": hf_result.get("explicit_harm_detected", False),
                "subtle_harm_score": hf_result.get("subtle_harm_score", 0.0),
                "reasoning": hf_result.get("reasoning", ""),
                "method": hf_result.get("method", "hf_api"),
                "principle_name": "harm_prevention",
                "weight": 2.0,
            }
        }

        flagged_principles = []
        weighted_score = 0.0

        if hf_result.get("flagged", False):
            flagged_principles.append("harm_prevention")
            weighted_score = 2.0  # harm_prevention weight

        evaluation = {
            "principle_results": principle_results,
            "any_flagged": len(flagged_principles) > 0,
            "flagged_principles": flagged_principles,
            "weighted_score": weighted_score,
            "num_principles_evaluated": 1,
            "text_length": len(text),
            "evaluation_method": hf_result.get("method", "hf_api"),
            "hf_api_details": {
                "toxicity_score": hf_result.get("subtle_harm_score", 0.0),
                "model": (
                    self._hf_api_evaluator.config.toxicity_model if self._hf_api_evaluator else None
                ),
            },
        }

        if track_history:
            self.evaluation_history.append(
                {"text": text[:100] + "..." if len(text) > 100 else text, "evaluation": evaluation}
            )

        return evaluation

    def batch_evaluate(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Evaluate multiple texts.

        Args:
            texts: List of texts to evaluate

        Returns:
            List of evaluation results
        """
        return [self.evaluate_text(text) for text in texts]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics from evaluation history.

        Returns:
            Dictionary with statistics about evaluations
        """
        if not self.evaluation_history:
            return {"total_evaluations": 0, "total_flagged": 0, "flagged_rate": 0.0}

        total_evaluations = len(self.evaluation_history)
        total_flagged = sum(
            1 for entry in self.evaluation_history if entry["evaluation"]["any_flagged"]
        )

        # Count violations per principle
        principle_violation_counts = dict.fromkeys(self.principles.keys(), 0)
        for entry in self.evaluation_history:
            for principle_name in entry["evaluation"]["flagged_principles"]:
                principle_violation_counts[principle_name] += 1

        return {
            "total_evaluations": total_evaluations,
            "total_flagged": total_flagged,
            "flagged_rate": total_flagged / total_evaluations if total_evaluations > 0 else 0.0,
            "principle_violation_counts": principle_violation_counts,
            "principle_violation_rates": {
                name: count / total_evaluations if total_evaluations > 0 else 0.0
                for name, count in principle_violation_counts.items()
            },
        }

    def clear_history(self) -> None:
        """Clear evaluation history."""
        self.evaluation_history = []

    def get_active_principles(self) -> list[str]:
        """Get list of currently enabled principle names."""
        return [name for name, principle in self.principles.items() if principle.enabled]

    def __repr__(self) -> str:
        """String representation of the framework."""
        num_principles = len(self.principles)
        num_enabled = len(self.get_active_principles())
        return f"ConstitutionalFramework(name='{self.name}', principles={num_enabled}/{num_principles} enabled)"
