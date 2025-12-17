"""MODULE: hf_api_evaluator.py
PURPOSE: HuggingFace Inference API integration for Constitutional AI evaluation
KEY COMPONENTS:
- HuggingFaceAPIEvaluator: Class for API-based text classification
- evaluate_toxicity_api: Evaluate text toxicity using HF API
- get_hf_api_client: Get or create API client
DEPENDENCIES: huggingface_hub, requests
SPECIAL NOTES: Uses HF Inference API for accurate toxicity/harm detection without local models.
              Free tier available with rate limits (~30k requests/month).
              Much more accurate than regex for subtle harmful content.
"""

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class HFAPIConfig:
    """Configuration for HuggingFace API evaluator."""

    # Model to use for toxicity classification
    # Options: "unitary/toxic-bert", "martin-ha/toxic-comment-model", etc.
    toxicity_model: str = "unitary/toxic-bert"

    # API token (optional for public models, but recommended)
    api_token: str | None = None

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Thresholds
    toxicity_threshold: float = 0.5  # Score above this = flagged

    # Timeout in seconds
    timeout: float = 30.0

    # Whether to use API (False = regex fallback)
    enabled: bool = True


# Global config instance
_api_config: HFAPIConfig | None = None
_api_client = None


def get_api_config() -> HFAPIConfig:
    """Get current API configuration."""
    global _api_config
    if _api_config is None:
        _api_config = HFAPIConfig(
            api_token=os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        )
    return _api_config


def set_api_config(config: HFAPIConfig) -> None:
    """Set API configuration."""
    global _api_config, _api_client
    _api_config = config
    _api_client = None  # Reset client when config changes


def get_hf_api_client():
    """
    Get or create HuggingFace Inference API client.

    Returns:
        InferenceClient or None if unavailable
    """
    global _api_client

    if _api_client is not None:
        return _api_client

    config = get_api_config()

    if not config.enabled:
        return None

    try:
        from huggingface_hub import InferenceClient

        # Create client with optional token
        _api_client = InferenceClient(token=config.api_token)
        return _api_client

    except ImportError:
        logger.info("[HF-API] huggingface_hub not installed. Run: pip install huggingface_hub")
        return None
    except Exception as e:
        logger.info(f"[HF-API] Failed to create client: {e}")
        return None


def evaluate_toxicity_api(
    text: str, config: HFAPIConfig | None = None, verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate text toxicity using HuggingFace Inference API.

    Uses models like toxic-bert which have ~98% accuracy on toxicity detection,
    much better than regex patterns.

    Args:
        text: Text to evaluate
        config: Optional API configuration
        verbose: Whether to print debug info

    Returns:
        Dictionary with:
        - flagged: bool - Whether text is flagged as toxic
        - toxicity_score: float - Toxicity score (0-1)
        - labels: List[Dict] - Raw classification labels
        - method: str - "hf_api" or "hf_api_error"
        - model: str - Model used for classification
    """
    if config is None:
        config = get_api_config()

    if not config.enabled:
        return {
            "flagged": False,
            "toxicity_score": 0.0,
            "labels": [],
            "method": "hf_api_disabled",
            "model": None,
        }

    client = get_hf_api_client()
    if client is None:
        return {
            "flagged": False,
            "toxicity_score": 0.0,
            "labels": [],
            "method": "hf_api_unavailable",
            "model": None,
            "error": "API client unavailable",
        }

    # Truncate text if too long (API has limits)
    max_length = 512
    if len(text) > max_length:
        text = text[:max_length]

    # Retry logic
    last_error = None
    for attempt in range(config.max_retries):
        try:
            if verbose:
                logger.info(f"[HF-API] Evaluating with {config.toxicity_model}...")

            # Call the text classification endpoint
            result = client.text_classification(text, model=config.toxicity_model)

            if verbose:
                logger.info(f"[HF-API] Response: {result}")

            # Parse results - format varies by model
            # toxic-bert returns: [{'label': 'toxic', 'score': 0.99}, {'label': 'non-toxic', 'score': 0.01}]
            # or: [{'label': 'LABEL_1', 'score': 0.99}]

            toxicity_score = 0.0
            labels = []

            if isinstance(result, list):
                labels = result
                for item in result:
                    label = item.get("label", "").lower()
                    score = item.get("score", 0.0)

                    # Different models use different label names
                    if label in ["toxic", "label_1", "positive", "hate", "offensive"]:
                        toxicity_score = max(toxicity_score, score)

            flagged = toxicity_score >= config.toxicity_threshold

            return {
                "flagged": flagged,
                "toxicity_score": toxicity_score,
                "labels": labels,
                "method": "hf_api",
                "model": config.toxicity_model,
            }

        except Exception as e:
            last_error = str(e)
            if verbose:
                logger.info(f"[HF-API] Attempt {attempt + 1} failed: {e}")

            # Check for rate limiting
            if "rate limit" in str(e).lower() or "429" in str(e):
                wait_time = config.retry_delay * (2**attempt)  # Exponential backoff
                if verbose:
                    logger.info(f"[HF-API] Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < config.max_retries - 1:
                time.sleep(config.retry_delay)

    # All retries failed
    return {
        "flagged": False,
        "toxicity_score": 0.0,
        "labels": [],
        "method": "hf_api_error",
        "model": config.toxicity_model,
        "error": last_error,
    }


def evaluate_harm_with_hf_api(
    text: str, config: HFAPIConfig | None = None, verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate harm potential using HuggingFace API.

    This is a wrapper that integrates with the Constitutional AI framework.

    Args:
        text: Text to evaluate
        config: Optional API configuration
        verbose: Whether to print debug info

    Returns:
        Dictionary compatible with Constitutional AI framework:
        - flagged: bool
        - explicit_harm_detected: bool
        - subtle_harm_score: float
        - reasoning: str
        - method: str
    """
    result = evaluate_toxicity_api(text, config, verbose)

    # Convert to Constitutional AI format
    toxicity_score = result.get("toxicity_score", 0.0)
    flagged = result.get("flagged", False)
    method = result.get("method", "hf_api")

    # Determine if this is "explicit" (high confidence) harm
    explicit_harm = toxicity_score >= 0.8

    # Generate reasoning
    if flagged:
        if explicit_harm:
            reasoning = f"HF API detected explicit harmful content (toxicity: {toxicity_score:.2%})"
        else:
            reasoning = (
                f"HF API detected potentially harmful content (toxicity: {toxicity_score:.2%})"
            )
    else:
        reasoning = f"HF API found no significant harmful content (toxicity: {toxicity_score:.2%})"

    # Add model info if available
    if result.get("model"):
        reasoning += f" [Model: {result['model']}]"

    # Add error info if there was one
    if result.get("error"):
        reasoning += f" [Error: {result['error']}]"

    return {
        "flagged": flagged,
        "explicit_harm_detected": explicit_harm,
        "subtle_harm_score": toxicity_score,
        "reasoning": reasoning,
        "method": method,
        "raw_labels": result.get("labels", []),
    }


class HuggingFaceAPIEvaluator:
    """
    HuggingFace API-based evaluator for Constitutional AI.

    This class provides a clean interface for using HF Inference API
    for text classification/evaluation in the CAI framework.

    Usage:
        evaluator = HuggingFaceAPIEvaluator()
        result = evaluator.evaluate_harm("How do I hack a computer?")
        logger.info(result['flagged'])  # True
        logger.info(result['toxicity_score'])  # 0.95
    """

    def __init__(
        self,
        toxicity_model: str = "unitary/toxic-bert",
        api_token: str | None = None,
        toxicity_threshold: float = 0.5,
        enabled: bool = True,
    ):
        """
        Initialize the HF API evaluator.

        Args:
            toxicity_model: Model ID for toxicity classification
            api_token: HuggingFace API token (optional for public models)
            toxicity_threshold: Score threshold for flagging (0-1)
            enabled: Whether to use API (False = all calls return unflagged)
        """
        self.config = HFAPIConfig(
            toxicity_model=toxicity_model,
            api_token=api_token or os.environ.get("HF_API_TOKEN"),
            toxicity_threshold=toxicity_threshold,
            enabled=enabled,
        )
        self._client = None

    @property
    def client(self):
        """Lazy-load the API client."""
        if self._client is None and self.config.enabled:
            try:
                from huggingface_hub import InferenceClient

                self._client = InferenceClient(token=self.config.api_token)
            except Exception as e:
                logger.info(f"[HF-API] Failed to create client: {e}")
        return self._client

    def evaluate_harm(self, text: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate text for harmful content.

        Args:
            text: Text to evaluate
            verbose: Print debug info

        Returns:
            Dictionary with evaluation results
        """
        return evaluate_harm_with_hf_api(text, self.config, verbose)

    def evaluate_toxicity(self, text: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate text toxicity.

        Args:
            text: Text to evaluate
            verbose: Print debug info

        Returns:
            Dictionary with toxicity evaluation
        """
        return evaluate_toxicity_api(text, self.config, verbose)

    def is_available(self) -> bool:
        """Check if the API is available and configured."""
        if not self.config.enabled:
            return False
        return self.client is not None

    def get_evaluation_fn(self) -> Callable:
        """
        Get an evaluation function compatible with Constitutional AI framework.

        Returns:
            Function that takes text and returns evaluation dict
        """

        def eval_fn(text: str, **kwargs) -> Dict[str, Any]:
            return self.evaluate_harm(text)

        return eval_fn


# Convenience function for quick evaluation
def quick_evaluate(text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Quick evaluation of text toxicity using HF API.

    Args:
        text: Text to evaluate
        threshold: Toxicity threshold (0-1)

    Returns:
        Dictionary with evaluation results

    Example:
        >>> result = quick_evaluate("How do I make a bomb?")
        >>> print(result['flagged'])  # True
    """
    config = HFAPIConfig(toxicity_threshold=threshold)
    return evaluate_harm_with_hf_api(text, config)
