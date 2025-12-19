"""
Unit tests for hf_api_evaluator.py
Tests the HuggingFace API integration for toxicity/harm detection.
"""

import os
from unittest.mock import Mock, patch

from constitutional_ai.hf_api_evaluator import (
    HFAPIConfig,
    HuggingFaceAPIEvaluator,
    evaluate_harm_with_hf_api,
    evaluate_toxicity_api,
    get_api_config,
    get_hf_api_client,
    quick_evaluate,
    set_api_config,
)
from tests.mocks.hf_api import MockInferenceClient


class TestHFAPIConfig:
    """Test HFAPIConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = HFAPIConfig()

        assert config.toxicity_model == "unitary/toxic-bert"
        assert config.api_token is None
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.toxicity_threshold == 0.5
        assert config.timeout == 30.0
        assert config.enabled is True

    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = HFAPIConfig(
            toxicity_model="martin-ha/toxic-comment-model",
            api_token="test_token",
            max_retries=5,
            retry_delay=2.0,
            toxicity_threshold=0.7,
            timeout=60.0,
            enabled=False,
        )

        assert config.toxicity_model == "martin-ha/toxic-comment-model"
        assert config.api_token == "test_token"
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.toxicity_threshold == 0.7
        assert config.timeout == 60.0
        assert config.enabled is False


class TestGetApiConfig:
    """Test get_api_config() function."""

    def setup_method(self):
        """Clear global config before each test."""
        import constitutional_ai.hf_api_evaluator as module

        module._api_config = None
        module._api_client = None

    def test_get_api_config_creates_default(self):
        """Test get_api_config() creates default config."""
        config = get_api_config()

        assert isinstance(config, HFAPIConfig)
        assert config.toxicity_model == "unitary/toxic-bert"

    def test_get_api_config_returns_same_instance(self):
        """Test get_api_config() returns same instance."""
        config1 = get_api_config()
        config2 = get_api_config()

        assert config1 is config2

    def test_get_api_config_loads_from_hf_api_token_env(self):
        """Test loading API token from HF_API_TOKEN environment variable."""
        with patch.dict(os.environ, {"HF_API_TOKEN": "token_from_env"}):
            import constitutional_ai.hf_api_evaluator as module

            module._api_config = None

            config = get_api_config()

            assert config.api_token == "token_from_env"

    def test_get_api_config_loads_from_huggingface_token_env(self):
        """Test loading API token from HUGGINGFACE_TOKEN environment variable."""
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf_token"}):
            import constitutional_ai.hf_api_evaluator as module

            module._api_config = None

            config = get_api_config()

            assert config.api_token == "hf_token"

    def test_get_api_config_prefers_hf_api_token(self):
        """Test HF_API_TOKEN takes precedence over HUGGINGFACE_TOKEN."""
        with patch.dict(
            os.environ,
            {"HF_API_TOKEN": "preferred_token", "HUGGINGFACE_TOKEN": "fallback_token"},
        ):
            import constitutional_ai.hf_api_evaluator as module

            module._api_config = None

            config = get_api_config()

            assert config.api_token == "preferred_token"


class TestSetApiConfig:
    """Test set_api_config() function."""

    def setup_method(self):
        """Clear global config before each test."""
        import constitutional_ai.hf_api_evaluator as module

        module._api_config = None
        module._api_client = None

    def test_set_api_config_updates_global(self):
        """Test set_api_config() updates global config."""
        custom_config = HFAPIConfig(toxicity_model="custom-model")

        set_api_config(custom_config)
        config = get_api_config()

        assert config.toxicity_model == "custom-model"

    def test_set_api_config_resets_client(self):
        """Test set_api_config() resets the API client."""
        import constitutional_ai.hf_api_evaluator as module

        # Set a mock client
        module._api_client = Mock()

        custom_config = HFAPIConfig()
        set_api_config(custom_config)

        # Client should be reset to None
        assert module._api_client is None


class TestGetHfApiClient:
    """Test get_hf_api_client() function."""

    def setup_method(self):
        """Clear global state before each test."""
        import constitutional_ai.hf_api_evaluator as module

        module._api_config = None
        module._api_client = None

    def test_get_hf_api_client_returns_cached_client(self):
        """Test client caching."""
        import constitutional_ai.hf_api_evaluator as module

        mock_client = Mock()
        module._api_client = mock_client

        client = get_hf_api_client()

        assert client is mock_client

    def test_get_hf_api_client_when_disabled(self):
        """Test returns None when API is disabled."""
        config = HFAPIConfig(enabled=False)
        set_api_config(config)

        client = get_hf_api_client()

        assert client is None

    def test_get_hf_api_client_creates_client_successfully(self):
        """Test successful client creation."""
        # Mock the import at the module level
        with patch.dict("sys.modules", {"huggingface_hub": Mock()}):
            from unittest.mock import MagicMock

            mock_inference_class = MagicMock()
            mock_client = Mock()
            mock_inference_class.return_value = mock_client

            with patch(
                "builtins.__import__",
                side_effect=lambda name, *args, **kwargs: (
                    MagicMock(InferenceClient=mock_inference_class)
                    if name == "huggingface_hub"
                    else __import__(name, *args, **kwargs)
                ),
            ):
                config = HFAPIConfig(api_token="test_token")
                set_api_config(config)

                client = get_hf_api_client()

                assert client is mock_client
                mock_inference_class.assert_called_once_with(token="test_token")

    def test_get_hf_api_client_handles_import_error(self):
        """Test handling of ImportError when huggingface_hub not installed."""
        # Remove huggingface_hub from sys.modules if it exists
        import sys

        hf_module = sys.modules.pop("huggingface_hub", None)

        try:
            with patch.dict("sys.modules", {"huggingface_hub": None}):
                import constitutional_ai.hf_api_evaluator as module

                module._api_client = None

                client = get_hf_api_client()

                assert client is None
        finally:
            # Restore module if it was there
            if hf_module is not None:
                sys.modules["huggingface_hub"] = hf_module

    def test_get_hf_api_client_handles_general_exception(self):
        """Test handling of general exceptions during client creation."""

        # Create a mock module that raises exception when creating client
        mock_hf_module = Mock()
        mock_hf_module.InferenceClient = Mock(side_effect=Exception("Connection error"))

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf_module}):
            import constitutional_ai.hf_api_evaluator as module

            module._api_client = None

            client = get_hf_api_client()

            assert client is None


class TestEvaluateToxicityApi:
    """Test evaluate_toxicity_api() function."""

    def setup_method(self):
        """Reset global state before each test."""
        import constitutional_ai.hf_api_evaluator as module

        module._api_config = None
        module._api_client = None

    def test_evaluate_toxicity_api_with_disabled_config(self):
        """Test evaluation when API is disabled."""
        config = HFAPIConfig(enabled=False)

        result = evaluate_toxicity_api("Test text", config)

        assert result["flagged"] is False
        assert result["toxicity_score"] == 0.0
        assert result["labels"] == []
        assert result["method"] == "hf_api_disabled"
        assert result["model"] is None

    def test_evaluate_toxicity_api_with_unavailable_client(self):
        """Test evaluation when client is unavailable."""
        config = HFAPIConfig(enabled=True)

        with patch("constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=None):
            result = evaluate_toxicity_api("Test text", config)

            assert result["flagged"] is False
            assert result["toxicity_score"] == 0.0
            assert result["method"] == "hf_api_unavailable"
            assert "error" in result
            assert result["error"] == "API client unavailable"

    def test_evaluate_toxicity_api_successful_toxic_response(self):
        """Test successful evaluation with toxic response."""
        config = HFAPIConfig(toxicity_threshold=0.5)
        mock_client = MockInferenceClient(
            response=[{"label": "toxic", "score": 0.95}, {"label": "non-toxic", "score": 0.05}]
        )

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("This is toxic text", config)

            assert result["flagged"] is True
            assert result["toxicity_score"] == 0.95
            assert result["method"] == "hf_api"
            assert result["model"] == "unitary/toxic-bert"
            assert len(result["labels"]) == 2

    def test_evaluate_toxicity_api_successful_non_toxic_response(self):
        """Test successful evaluation with non-toxic response."""
        config = HFAPIConfig(toxicity_threshold=0.5)
        mock_client = MockInferenceClient(
            response=[
                {"label": "non-toxic", "score": 0.95},
                {"label": "toxic", "score": 0.05},
            ]
        )

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("This is safe text", config)

            assert result["flagged"] is False
            assert result["toxicity_score"] == 0.05
            assert result["method"] == "hf_api"

    def test_evaluate_toxicity_api_label_format_label_1(self):
        """Test parsing LABEL_1/LABEL_0 format."""
        config = HFAPIConfig(toxicity_threshold=0.5)
        mock_client = MockInferenceClient(
            response=[{"label": "LABEL_1", "score": 0.85}, {"label": "LABEL_0", "score": 0.15}]
        )

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("Test text", config)

            assert result["flagged"] is True
            assert result["toxicity_score"] == 0.85

    def test_evaluate_toxicity_api_label_format_hate(self):
        """Test parsing hate/offensive labels."""
        config = HFAPIConfig(toxicity_threshold=0.5)
        mock_client = MockInferenceClient(response=[{"label": "hate", "score": 0.75}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("Test text", config)

            assert result["flagged"] is True
            assert result["toxicity_score"] == 0.75

    def test_evaluate_toxicity_api_label_format_offensive(self):
        """Test parsing offensive label."""
        config = HFAPIConfig(toxicity_threshold=0.5)
        mock_client = MockInferenceClient(response=[{"label": "offensive", "score": 0.65}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("Test text", config)

            assert result["flagged"] is True
            assert result["toxicity_score"] == 0.65

    def test_evaluate_toxicity_api_label_format_positive(self):
        """Test parsing positive label (toxic variant)."""
        config = HFAPIConfig(toxicity_threshold=0.5)
        mock_client = MockInferenceClient(response=[{"label": "positive", "score": 0.88}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("Test text", config)

            assert result["flagged"] is True
            assert result["toxicity_score"] == 0.88

    def test_evaluate_toxicity_api_threshold_logic(self):
        """Test threshold logic for flagging."""
        config = HFAPIConfig(toxicity_threshold=0.7)
        mock_client = MockInferenceClient(response=[{"label": "toxic", "score": 0.65}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("Test text", config)

            # Score 0.65 < threshold 0.7, so not flagged
            assert result["flagged"] is False
            assert result["toxicity_score"] == 0.65

    def test_evaluate_toxicity_api_threshold_boundary(self):
        """Test threshold boundary condition (score == threshold)."""
        config = HFAPIConfig(toxicity_threshold=0.5)
        mock_client = MockInferenceClient(response=[{"label": "toxic", "score": 0.5}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("Test text", config)

            # Score 0.5 >= threshold 0.5, so flagged
            assert result["flagged"] is True
            assert result["toxicity_score"] == 0.5

    def test_evaluate_toxicity_api_text_truncation(self):
        """Test text truncation for long text (>512 chars)."""
        config = HFAPIConfig()
        long_text = "A" * 600
        mock_client = MockInferenceClient(response=[{"label": "non-toxic", "score": 0.05}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api(long_text, config)

            # Should truncate to 512 chars
            assert mock_client.call_count == 1
            # Result should still be valid
            assert result["method"] == "hf_api"

    def test_evaluate_toxicity_api_retry_logic_succeeds_on_second_attempt(self):
        """Test retry logic succeeds after one failure."""
        config = HFAPIConfig(max_retries=3, retry_delay=0.01)
        mock_client = MockInferenceClient(
            responses=[
                Exception("Temporary error"),
                [{"label": "toxic", "score": 0.8}],
            ]
        )

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("Test text", config)

            assert result["flagged"] is True
            assert result["toxicity_score"] == 0.8
            assert result["method"] == "hf_api"
            assert mock_client.call_count == 2

    def test_evaluate_toxicity_api_retry_logic_succeeds_on_third_attempt(self):
        """Test retry logic succeeds after two failures."""
        config = HFAPIConfig(max_retries=3, retry_delay=0.01)
        mock_client = MockInferenceClient(
            responses=[
                Exception("Error 1"),
                Exception("Error 2"),
                [{"label": "toxic", "score": 0.75}],
            ]
        )

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("Test text", config)

            assert result["flagged"] is True
            assert result["toxicity_score"] == 0.75
            assert result["method"] == "hf_api"
            assert mock_client.call_count == 3

    def test_evaluate_toxicity_api_rate_limit_detection(self):
        """Test rate limit detection triggers exponential backoff."""
        config = HFAPIConfig(max_retries=3, retry_delay=0.01)
        mock_client = MockInferenceClient(
            responses=[
                Exception("429 Rate limit exceeded"),
                [{"label": "toxic", "score": 0.6}],
            ]
        )

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            with patch("constitutional_ai.hf_api_evaluator.time.sleep") as mock_sleep:
                result = evaluate_toxicity_api("Test text", config)

                # Should use exponential backoff: retry_delay * 2^0 = 0.01
                mock_sleep.assert_called_once_with(0.01)
                assert result["method"] == "hf_api"

    def test_evaluate_toxicity_api_rate_limit_string_detection(self):
        """Test rate limit string detection (case insensitive)."""
        config = HFAPIConfig(max_retries=2, retry_delay=0.01)
        mock_client = MockInferenceClient(
            responses=[
                Exception("Rate Limit exceeded"),
                [{"label": "non-toxic", "score": 0.1}],
            ]
        )

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            with patch("constitutional_ai.hf_api_evaluator.time.sleep") as mock_sleep:
                result = evaluate_toxicity_api("Test text", config)

                mock_sleep.assert_called_once()
                assert result["method"] == "hf_api"

    def test_evaluate_toxicity_api_all_retries_fail(self):
        """Test all retries fail returns error dict."""
        config = HFAPIConfig(max_retries=3, retry_delay=0.01)
        mock_client = MockInferenceClient(
            responses=[
                Exception("Error 1"),
                Exception("Error 2"),
                Exception("Error 3"),
            ]
        )

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("Test text", config)

            assert result["flagged"] is False
            assert result["toxicity_score"] == 0.0
            assert result["method"] == "hf_api_error"
            assert result["model"] == "unitary/toxic-bert"
            assert "error" in result
            assert "Error 3" in result["error"]

    def test_evaluate_toxicity_api_verbose_mode(self):
        """Test verbose mode logging."""
        config = HFAPIConfig()
        mock_client = MockInferenceClient(response=[{"label": "toxic", "score": 0.9}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            with patch("constitutional_ai.hf_api_evaluator.logger.info") as mock_log:
                result = evaluate_toxicity_api("Test text", config, verbose=True)

                # Should log evaluation and response
                assert mock_log.call_count >= 2
                assert result["method"] == "hf_api"

    def test_evaluate_toxicity_api_with_default_config(self):
        """Test evaluation with default config (None parameter)."""
        mock_client = MockInferenceClient(response=[{"label": "non-toxic", "score": 0.1}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("Test text", config=None)

            assert result["method"] == "hf_api"
            assert result["model"] == "unitary/toxic-bert"


class TestEvaluateHarmWithHfApi:
    """Test evaluate_harm_with_hf_api() function."""

    def setup_method(self):
        """Reset global state before each test."""
        import constitutional_ai.hf_api_evaluator as module

        module._api_config = None
        module._api_client = None

    def test_evaluate_harm_with_hf_api_format_conversion(self):
        """Test format conversion to Constitutional AI dict format."""
        config = HFAPIConfig()
        mock_client = MockInferenceClient(response=[{"label": "toxic", "score": 0.7}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_harm_with_hf_api("Test text", config)

            # Check Constitutional AI format fields
            assert "flagged" in result
            assert "explicit_harm_detected" in result
            assert "subtle_harm_score" in result
            assert "reasoning" in result
            assert "method" in result
            assert "raw_labels" in result

    def test_evaluate_harm_with_hf_api_explicit_harm_detection(self):
        """Test explicit harm detection (score >= 0.8)."""
        config = HFAPIConfig()
        mock_client = MockInferenceClient(response=[{"label": "toxic", "score": 0.85}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_harm_with_hf_api("Very harmful text", config)

            assert result["explicit_harm_detected"] is True
            assert result["subtle_harm_score"] == 0.85
            assert "explicit harmful content" in result["reasoning"]

    def test_evaluate_harm_with_hf_api_subtle_harm_detection(self):
        """Test subtle harm detection (0.5 <= score < 0.8)."""
        config = HFAPIConfig()
        mock_client = MockInferenceClient(response=[{"label": "toxic", "score": 0.65}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_harm_with_hf_api("Potentially harmful text", config)

            assert result["explicit_harm_detected"] is False
            assert result["subtle_harm_score"] == 0.65
            assert "potentially harmful content" in result["reasoning"]

    def test_evaluate_harm_with_hf_api_no_harm_detected(self):
        """Test no harm detected (score < threshold)."""
        config = HFAPIConfig()
        mock_client = MockInferenceClient(response=[{"label": "non-toxic", "score": 0.95}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_harm_with_hf_api("Safe text", config)

            assert result["flagged"] is False
            assert result["explicit_harm_detected"] is False
            assert "no significant harmful content" in result["reasoning"]

    def test_evaluate_harm_with_hf_api_reasoning_includes_model(self):
        """Test reasoning includes model information."""
        config = HFAPIConfig()
        mock_client = MockInferenceClient(response=[{"label": "toxic", "score": 0.9}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_harm_with_hf_api("Test text", config)

            assert "[Model: unitary/toxic-bert]" in result["reasoning"]

    def test_evaluate_harm_with_hf_api_reasoning_includes_error(self):
        """Test reasoning includes error information when error occurs."""
        config = HFAPIConfig(max_retries=1, retry_delay=0.01)
        mock_client = MockInferenceClient(error=Exception("Connection timeout"))

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_harm_with_hf_api("Test text", config)

            assert "[Error: Connection timeout]" in result["reasoning"]
            assert result["method"] == "hf_api_error"

    def test_evaluate_harm_with_hf_api_raw_labels_preserved(self):
        """Test raw labels are preserved in result."""
        config = HFAPIConfig()
        expected_labels = [{"label": "toxic", "score": 0.8}, {"label": "non-toxic", "score": 0.2}]
        mock_client = MockInferenceClient(response=expected_labels)

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_harm_with_hf_api("Test text", config)

            assert result["raw_labels"] == expected_labels

    def test_evaluate_harm_with_hf_api_percentage_formatting(self):
        """Test toxicity score is formatted as percentage in reasoning."""
        config = HFAPIConfig()
        mock_client = MockInferenceClient(response=[{"label": "toxic", "score": 0.7543}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_harm_with_hf_api("Test text", config)

            # Should format as 75.43% or similar
            assert "75.43%" in result["reasoning"] or "75%" in result["reasoning"]


class TestHuggingFaceAPIEvaluatorInit:
    """Test HuggingFaceAPIEvaluator initialization."""

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        evaluator = HuggingFaceAPIEvaluator()

        assert evaluator.config.toxicity_model == "unitary/toxic-bert"
        assert evaluator.config.toxicity_threshold == 0.5
        assert evaluator.config.enabled is True
        assert evaluator._client is None

    def test_init_with_custom_toxicity_model(self):
        """Test initialization with custom toxicity model."""
        evaluator = HuggingFaceAPIEvaluator(toxicity_model="martin-ha/toxic-comment-model")

        assert evaluator.config.toxicity_model == "martin-ha/toxic-comment-model"

    def test_init_with_custom_api_token(self):
        """Test initialization with custom API token."""
        evaluator = HuggingFaceAPIEvaluator(api_token="custom_token")

        assert evaluator.config.api_token == "custom_token"

    def test_init_with_custom_threshold(self):
        """Test initialization with custom toxicity threshold."""
        evaluator = HuggingFaceAPIEvaluator(toxicity_threshold=0.7)

        assert evaluator.config.toxicity_threshold == 0.7

    def test_init_with_disabled_flag(self):
        """Test initialization with enabled=False."""
        evaluator = HuggingFaceAPIEvaluator(enabled=False)

        assert evaluator.config.enabled is False

    def test_init_api_token_from_env_variable(self):
        """Test API token loaded from environment variable."""
        with patch.dict(os.environ, {"HF_API_TOKEN": "env_token"}):
            evaluator = HuggingFaceAPIEvaluator()

            assert evaluator.config.api_token == "env_token"

    def test_init_explicit_token_overrides_env(self):
        """Test explicit token parameter overrides environment variable."""
        with patch.dict(os.environ, {"HF_API_TOKEN": "env_token"}):
            evaluator = HuggingFaceAPIEvaluator(api_token="explicit_token")

            assert evaluator.config.api_token == "explicit_token"


class TestHuggingFaceAPIEvaluatorClientProperty:
    """Test HuggingFaceAPIEvaluator.client property."""

    def test_client_property_lazy_loading(self):
        """Test client property lazy loads."""
        evaluator = HuggingFaceAPIEvaluator()

        assert evaluator._client is None

        with patch.dict("sys.modules", {"huggingface_hub": Mock()}):
            from unittest.mock import MagicMock

            mock_inference_class = MagicMock()
            mock_client = Mock()
            mock_inference_class.return_value = mock_client

            with patch(
                "builtins.__import__",
                side_effect=lambda name, *args, **kwargs: (
                    MagicMock(InferenceClient=mock_inference_class)
                    if name == "huggingface_hub"
                    else __import__(name, *args, **kwargs)
                ),
            ):
                client = evaluator.client

                assert client is mock_client
                assert evaluator._client is mock_client

    def test_client_property_returns_cached_client(self):
        """Test client property returns cached client."""
        evaluator = HuggingFaceAPIEvaluator()
        mock_client = Mock()
        evaluator._client = mock_client

        client = evaluator.client

        assert client is mock_client

    def test_client_property_handles_import_error(self):
        """Test client property handles ImportError gracefully."""
        evaluator = HuggingFaceAPIEvaluator()

        # Simulate ImportError by removing module
        import sys

        hf_module = sys.modules.pop("huggingface_hub", None)

        try:
            with patch.dict("sys.modules", {"huggingface_hub": None}):
                client = evaluator.client

                assert client is None
        finally:
            if hf_module is not None:
                sys.modules["huggingface_hub"] = hf_module

    def test_client_property_handles_general_exception(self):
        """Test client property handles general exceptions."""
        evaluator = HuggingFaceAPIEvaluator()

        with patch.dict("sys.modules", {"huggingface_hub": Mock()}):
            from unittest.mock import MagicMock

            mock_inference_class = MagicMock(side_effect=Exception("Error"))

            with patch(
                "builtins.__import__",
                side_effect=lambda name, *args, **kwargs: (
                    MagicMock(InferenceClient=mock_inference_class)
                    if name == "huggingface_hub"
                    else __import__(name, *args, **kwargs)
                ),
            ):
                evaluator._client = None  # Reset
                client = evaluator.client

                assert client is None

    def test_client_property_returns_none_when_disabled(self):
        """Test client property returns None when API disabled."""
        evaluator = HuggingFaceAPIEvaluator(enabled=False)

        client = evaluator.client

        assert client is None


class TestHuggingFaceAPIEvaluatorEvaluateHarm:
    """Test HuggingFaceAPIEvaluator.evaluate_harm() method."""

    def test_evaluate_harm_calls_evaluate_harm_with_hf_api(self):
        """Test evaluate_harm() calls underlying function."""
        evaluator = HuggingFaceAPIEvaluator()

        with patch("constitutional_ai.hf_api_evaluator.evaluate_harm_with_hf_api") as mock_evaluate:
            mock_evaluate.return_value = {"flagged": True, "reasoning": "Test"}

            result = evaluator.evaluate_harm("Test text")

            mock_evaluate.assert_called_once_with("Test text", evaluator.config, False)
            assert result["flagged"] is True

    def test_evaluate_harm_with_verbose_flag(self):
        """Test evaluate_harm() passes verbose flag."""
        evaluator = HuggingFaceAPIEvaluator()

        with patch("constitutional_ai.hf_api_evaluator.evaluate_harm_with_hf_api") as mock_evaluate:
            mock_evaluate.return_value = {"flagged": False}

            evaluator.evaluate_harm("Test text", verbose=True)

            mock_evaluate.assert_called_once_with("Test text", evaluator.config, True)


class TestHuggingFaceAPIEvaluatorEvaluateToxicity:
    """Test HuggingFaceAPIEvaluator.evaluate_toxicity() method."""

    def test_evaluate_toxicity_calls_evaluate_toxicity_api(self):
        """Test evaluate_toxicity() calls underlying function."""
        evaluator = HuggingFaceAPIEvaluator()

        with patch("constitutional_ai.hf_api_evaluator.evaluate_toxicity_api") as mock_evaluate:
            mock_evaluate.return_value = {"flagged": True, "toxicity_score": 0.8}

            result = evaluator.evaluate_toxicity("Test text")

            mock_evaluate.assert_called_once_with("Test text", evaluator.config, False)
            assert result["flagged"] is True

    def test_evaluate_toxicity_with_verbose_flag(self):
        """Test evaluate_toxicity() passes verbose flag."""
        evaluator = HuggingFaceAPIEvaluator()

        with patch("constitutional_ai.hf_api_evaluator.evaluate_toxicity_api") as mock_evaluate:
            mock_evaluate.return_value = {"flagged": False}

            evaluator.evaluate_toxicity("Test text", verbose=True)

            mock_evaluate.assert_called_once_with("Test text", evaluator.config, True)


class TestHuggingFaceAPIEvaluatorIsAvailable:
    """Test HuggingFaceAPIEvaluator.is_available() method."""

    def test_is_available_returns_false_when_disabled(self):
        """Test is_available() returns False when API disabled."""
        evaluator = HuggingFaceAPIEvaluator(enabled=False)

        assert evaluator.is_available() is False

    def test_is_available_returns_false_when_client_none(self):
        """Test is_available() returns False when client is None."""
        evaluator = HuggingFaceAPIEvaluator()
        evaluator._client = None

        # Simulate no client available
        import sys

        hf_module = sys.modules.pop("huggingface_hub", None)

        try:
            with patch.dict("sys.modules", {"huggingface_hub": None}):
                result = evaluator.is_available()
                assert result is False
        finally:
            if hf_module is not None:
                sys.modules["huggingface_hub"] = hf_module

    def test_is_available_returns_true_when_enabled_and_client_exists(self):
        """Test is_available() returns True when enabled and client exists."""
        evaluator = HuggingFaceAPIEvaluator()
        mock_client = Mock()
        evaluator._client = mock_client

        result = evaluator.is_available()
        assert result is True


class TestHuggingFaceAPIEvaluatorGetEvaluationFn:
    """Test HuggingFaceAPIEvaluator.get_evaluation_fn() method."""

    def test_get_evaluation_fn_returns_callable(self):
        """Test get_evaluation_fn() returns a callable function."""
        evaluator = HuggingFaceAPIEvaluator()

        eval_fn = evaluator.get_evaluation_fn()

        assert callable(eval_fn)

    def test_get_evaluation_fn_callable_evaluates_text(self):
        """Test returned function evaluates text correctly."""
        evaluator = HuggingFaceAPIEvaluator()

        with patch.object(evaluator, "evaluate_harm") as mock_evaluate:
            mock_evaluate.return_value = {"flagged": True, "reasoning": "Test"}

            eval_fn = evaluator.get_evaluation_fn()
            result = eval_fn("Test text")

            mock_evaluate.assert_called_once_with("Test text")
            assert result["flagged"] is True

    def test_get_evaluation_fn_callable_ignores_kwargs(self):
        """Test returned function ignores additional kwargs."""
        evaluator = HuggingFaceAPIEvaluator()

        with patch.object(evaluator, "evaluate_harm") as mock_evaluate:
            mock_evaluate.return_value = {"flagged": False}

            eval_fn = evaluator.get_evaluation_fn()
            result = eval_fn("Test text", extra_param="ignored")

            # Should still call with just text
            mock_evaluate.assert_called_once_with("Test text")
            assert result["flagged"] is False


class TestQuickEvaluate:
    """Test quick_evaluate() convenience function."""

    def setup_method(self):
        """Reset global state before each test."""
        import constitutional_ai.hf_api_evaluator as module

        module._api_config = None
        module._api_client = None

    def test_quick_evaluate_with_default_threshold(self):
        """Test quick_evaluate() with default threshold."""
        with patch("constitutional_ai.hf_api_evaluator.evaluate_harm_with_hf_api") as mock_evaluate:
            mock_evaluate.return_value = {"flagged": True}

            result = quick_evaluate("Test text")

            # Should create config with threshold 0.5
            call_args = mock_evaluate.call_args
            config = call_args[0][1]
            assert config.toxicity_threshold == 0.5
            assert result["flagged"] is True

    def test_quick_evaluate_with_custom_threshold(self):
        """Test quick_evaluate() with custom threshold."""
        with patch("constitutional_ai.hf_api_evaluator.evaluate_harm_with_hf_api") as mock_evaluate:
            mock_evaluate.return_value = {"flagged": False}

            result = quick_evaluate("Test text", threshold=0.8)

            # Should create config with threshold 0.8
            call_args = mock_evaluate.call_args
            config = call_args[0][1]
            assert config.toxicity_threshold == 0.8
            assert result["flagged"] is False


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    def setup_method(self):
        """Reset global state before each test."""
        import constitutional_ai.hf_api_evaluator as module

        module._api_config = None
        module._api_client = None

    def test_full_evaluation_flow_toxic_text(self):
        """Test full evaluation flow with toxic text."""
        mock_client = MockInferenceClient(response=[{"label": "toxic", "score": 0.9}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            evaluator = HuggingFaceAPIEvaluator()
            result = evaluator.evaluate_harm("Very harmful text here")

            assert result["flagged"] is True
            assert result["explicit_harm_detected"] is True
            assert result["subtle_harm_score"] == 0.9
            assert "explicit harmful content" in result["reasoning"]
            assert "unitary/toxic-bert" in result["reasoning"]

    def test_full_evaluation_flow_safe_text(self):
        """Test full evaluation flow with safe text."""
        mock_client = MockInferenceClient(response=[{"label": "non-toxic", "score": 0.95}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            evaluator = HuggingFaceAPIEvaluator()
            result = evaluator.evaluate_harm("How to bake a cake")

            assert result["flagged"] is False
            assert result["explicit_harm_detected"] is False
            assert "no significant harmful content" in result["reasoning"]

    def test_evaluator_with_custom_config(self):
        """Test evaluator with custom configuration."""
        mock_client = MockInferenceClient(response=[{"label": "toxic", "score": 0.6}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            evaluator = HuggingFaceAPIEvaluator(
                toxicity_model="custom-model", toxicity_threshold=0.7
            )

            result = evaluator.evaluate_toxicity("Test text")

            # Score 0.6 < threshold 0.7, so not flagged
            assert result["flagged"] is False
            assert result["toxicity_score"] == 0.6
            assert result["model"] == "custom-model"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Reset global state before each test."""
        import constitutional_ai.hf_api_evaluator as module

        module._api_config = None
        module._api_client = None

    def test_empty_text_evaluation(self):
        """Test evaluation with empty text."""
        mock_client = MockInferenceClient(response=[{"label": "non-toxic", "score": 0.01}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            evaluator = HuggingFaceAPIEvaluator()
            result = evaluator.evaluate_harm("")

            assert "flagged" in result
            assert "reasoning" in result

    def test_very_long_text_evaluation(self):
        """Test evaluation with very long text (>512 chars)."""
        long_text = "This is a test sentence. " * 100  # ~2500 chars
        mock_client = MockInferenceClient(response=[{"label": "non-toxic", "score": 0.05}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            evaluator = HuggingFaceAPIEvaluator()
            result = evaluator.evaluate_harm(long_text)

            # Should truncate and still work
            assert result["method"] == "hf_api"

    def test_unicode_text_evaluation(self):
        """Test evaluation with unicode characters."""
        mock_client = MockInferenceClient(response=[{"label": "non-toxic", "score": 0.02}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            evaluator = HuggingFaceAPIEvaluator()
            result = evaluator.evaluate_harm("Hello 世界 🌍")

            assert "flagged" in result

    def test_special_characters_in_text(self):
        """Test evaluation with special characters."""
        mock_client = MockInferenceClient(response=[{"label": "non-toxic", "score": 0.03}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            evaluator = HuggingFaceAPIEvaluator()
            result = evaluator.evaluate_harm("!@#$%^&*() <> {} []")

            assert "flagged" in result

    def test_multiple_labels_max_score_selection(self):
        """Test that maximum toxicity score is selected from multiple labels."""
        mock_client = MockInferenceClient(
            response=[
                {"label": "toxic", "score": 0.3},
                {"label": "hate", "score": 0.7},
                {"label": "offensive", "score": 0.5},
            ]
        )

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("Test text")

            # Should select max score 0.7 from all toxic labels
            assert result["toxicity_score"] == 0.7

    def test_empty_labels_list(self):
        """Test handling of empty labels list from API."""
        mock_client = MockInferenceClient(response=[])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("Test text")

            assert result["toxicity_score"] == 0.0
            assert result["flagged"] is False
            assert result["labels"] == []

    def test_none_config_parameter(self):
        """Test handling of None config parameter."""
        mock_client = MockInferenceClient(response=[{"label": "non-toxic", "score": 0.1}])

        with patch(
            "constitutional_ai.hf_api_evaluator.get_hf_api_client", return_value=mock_client
        ):
            result = evaluate_toxicity_api("Test text", config=None)

            # Should use default config
            assert result["method"] == "hf_api"
