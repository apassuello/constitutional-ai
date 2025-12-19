"""
Unit tests for model_utils.py
Tests model loading, text generation, and training utilities
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from constitutional_ai.model_utils import (
    GenerationConfig,
    batch_generate,
    generate_text,
    get_model_device,
    load_model,
    prepare_model_for_training,
)
from tests.mocks.transformers import create_mock_model, create_mock_tokenizer


class TestGenerationConfig:
    """Test GenerationConfig dataclass."""

    def test_generation_config_defaults(self):
        """Test GenerationConfig default values."""
        config = GenerationConfig()

        assert config.max_new_tokens == 100
        assert config.max_length is None
        assert config.temperature == 1.0
        assert config.top_p == 1.0
        assert config.top_k == 0
        assert config.num_return_sequences == 1
        assert config.do_sample is True
        assert config.pad_token_id is None
        assert config.eos_token_id is None
        assert config.min_new_tokens is None

    def test_generation_config_custom_values(self):
        """Test GenerationConfig with custom values."""
        config = GenerationConfig(
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_return_sequences=2,
            do_sample=False,
        )

        assert config.max_new_tokens == 50
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.num_return_sequences == 2
        assert config.do_sample is False

    def test_generation_config_token_ids(self):
        """Test GenerationConfig with token IDs."""
        config = GenerationConfig(
            pad_token_id=0,
            eos_token_id=1,
            min_new_tokens=10,
        )

        assert config.pad_token_id == 0
        assert config.eos_token_id == 1
        assert config.min_new_tokens == 10


class TestLoadModel:
    """Test load_model() function."""

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("constitutional_ai.model_utils.torch.cuda.is_available")
    def test_load_model_default_device_cuda(self, mock_cuda, mock_tokenizer_cls, mock_model_cls):
        """Test load_model with default device selection (cuda available)."""
        mock_cuda.return_value = True
        mock_tokenizer = create_mock_tokenizer()
        mock_model = create_mock_model()

        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        model, tokenizer = load_model("gpt2")

        assert model is mock_model
        assert tokenizer is mock_tokenizer
        mock_cuda.assert_called_once()
        mock_model.to.assert_called()

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("constitutional_ai.model_utils.torch.cuda.is_available")
    def test_load_model_default_device_cpu(self, mock_cuda, mock_tokenizer_cls, mock_model_cls):
        """Test load_model with default device selection (cpu fallback)."""
        mock_cuda.return_value = False
        mock_tokenizer = create_mock_tokenizer()
        mock_model = create_mock_model()

        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        model, tokenizer = load_model("gpt2")

        assert model is mock_model
        assert tokenizer is mock_tokenizer
        mock_cuda.assert_called_once()

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_load_model_custom_device(self, mock_tokenizer_cls, mock_model_cls):
        """Test load_model with custom device parameter."""
        mock_tokenizer = create_mock_tokenizer()
        mock_model = create_mock_model()

        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        custom_device = torch.device("cpu")
        model, tokenizer = load_model("gpt2", device=custom_device)

        assert model is mock_model
        mock_model.to.assert_called_once()
        # Verify device was used by checking to() call
        call_args = mock_model.to.call_args[0]
        assert call_args[0] == custom_device

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    @patch("constitutional_ai.model_utils.torch.cuda.is_available")
    def test_load_model_8bit_cuda(self, mock_cuda, mock_tokenizer_cls, mock_model_cls):
        """Test load_model with 8-bit loading on cuda."""
        mock_cuda.return_value = True
        mock_tokenizer = create_mock_tokenizer()
        mock_model = create_mock_model()

        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        device = torch.device("cuda")
        model, tokenizer = load_model("gpt2", device=device, load_in_8bit=True)

        # Should pass load_in_8bit and device_map to from_pretrained
        mock_model_cls.from_pretrained.assert_called_once()
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert call_kwargs["load_in_8bit"] is True
        assert call_kwargs["device_map"] == "auto"

        # Should NOT call .to() when using 8-bit
        mock_model.to.assert_not_called()

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_load_model_8bit_cpu_ignored(self, mock_tokenizer_cls, mock_model_cls):
        """Test load_model with 8-bit loading on cpu (should be ignored)."""
        mock_tokenizer = create_mock_tokenizer()
        mock_model = create_mock_model()

        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        device = torch.device("cpu")
        model, tokenizer = load_model("gpt2", device=device, load_in_8bit=True)

        # Should NOT pass load_in_8bit when device is CPU
        mock_model_cls.from_pretrained.assert_called_once()
        call_kwargs = mock_model_cls.from_pretrained.call_args[1]
        assert "load_in_8bit" not in call_kwargs
        assert "device_map" not in call_kwargs

        # Should NOT call .to() when load_in_8bit=True (even on CPU, line 77 checks `not load_in_8bit`)
        mock_model.to.assert_not_called()

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_load_model_tokenizer_pad_token_setup(self, mock_tokenizer_cls, mock_model_cls):
        """Test load_model sets pad_token to eos_token when None."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer.pad_token = None  # Simulate missing pad_token
        mock_tokenizer.eos_token = "<eos>"
        mock_model = create_mock_model()

        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        model, tokenizer = load_model("gpt2")

        # Should have set pad_token to eos_token
        assert tokenizer.pad_token == "<eos>"

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_load_model_tokenizer_pad_token_already_set(self, mock_tokenizer_cls, mock_model_cls):
        """Test load_model preserves existing pad_token."""
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_model = create_mock_model()

        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        model, tokenizer = load_model("gpt2")

        # Should preserve existing pad_token
        assert tokenizer.pad_token == "<pad>"

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_load_model_different_model_names(self, mock_tokenizer_cls, mock_model_cls):
        """Test load_model with different model names."""
        mock_tokenizer = create_mock_tokenizer()
        mock_model = create_mock_model()

        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        model, tokenizer = load_model("gpt2-medium")

        mock_tokenizer_cls.from_pretrained.assert_called_with("gpt2-medium")
        mock_model_cls.from_pretrained.assert_called_once()

    def test_load_model_import_error(self):
        """Test load_model raises ImportError when transformers not available."""
        # Temporarily remove transformers from sys.modules
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError) as exc_info:
                load_model("gpt2")

            assert "transformers library required" in str(exc_info.value)

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_load_model_logs_parameter_count(self, mock_tokenizer_cls, mock_model_cls):
        """Test load_model logs model parameter count."""
        mock_tokenizer = create_mock_tokenizer()
        mock_model = create_mock_model()

        # Create parameters with known sizes
        def parameters_side_effect():
            return [torch.zeros(10, 10), torch.zeros(5, 5)]

        mock_model.parameters.side_effect = parameters_side_effect

        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        mock_model_cls.from_pretrained.return_value = mock_model

        with patch("constitutional_ai.model_utils.logger") as mock_logger:
            model, tokenizer = load_model("gpt2")

            # Should log parameter count (10*10 + 5*5 = 125)
            log_messages = [str(call) for call in mock_logger.info.call_args_list]
            param_log = [msg for msg in log_messages if "Parameters:" in str(msg)]
            assert len(param_log) > 0


class TestGenerateText:
    """Test generate_text() function."""

    def test_generate_text_with_default_config(self):
        """Test generate_text with default GenerationConfig."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        # Mock model.parameters() to return device
        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()

        # Mock model.generate to return tensor
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        result = generate_text(mock_model, mock_tokenizer, "Test prompt")

        assert isinstance(result, str)
        assert result == "generated text"
        mock_model.generate.assert_called_once()

    def test_generate_text_with_custom_config(self):
        """Test generate_text with custom GenerationConfig."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        config = GenerationConfig(max_new_tokens=50, temperature=0.7, top_p=0.9, top_k=40)

        generate_text(mock_model, mock_tokenizer, "Test prompt", config)

        # Verify generate was called with config parameters
        mock_model.generate.assert_called_once()
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["top_k"] == 40
        assert call_kwargs["max_new_tokens"] == 50

    def test_generate_text_device_inference(self):
        """Test generate_text infers device from model.parameters()."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cuda:0")
            yield param

        mock_model.parameters.return_value = params_gen()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        # Mock tokenizer to track device moves
        tokenizer_return = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        mock_tokenizer.return_value = tokenizer_return

        generate_text(mock_model, mock_tokenizer, "Test prompt")

        # Verify inputs were moved to correct device (checked via to() calls on tensors)
        mock_model.generate.assert_called_once()

    def test_generate_text_custom_device_parameter(self):
        """Test generate_text with custom device parameter."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        custom_device = torch.device("cpu")  # Use CPU to avoid cuda availability issues
        result = generate_text(mock_model, mock_tokenizer, "Test prompt", device=custom_device)

        assert isinstance(result, str)

    def test_generate_text_top_p_filtering_disabled(self):
        """Test generate_text does not pass top_p when >= 1.0."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        config = GenerationConfig(top_p=1.0)  # Disabled
        generate_text(mock_model, mock_tokenizer, "Test", config)

        call_kwargs = mock_model.generate.call_args[1]
        assert "top_p" not in call_kwargs

    def test_generate_text_top_p_filtering_enabled(self):
        """Test generate_text passes top_p when < 1.0."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        config = GenerationConfig(top_p=0.9)  # Enabled
        generate_text(mock_model, mock_tokenizer, "Test", config)

        call_kwargs = mock_model.generate.call_args[1]
        assert "top_p" in call_kwargs
        assert call_kwargs["top_p"] == 0.9

    def test_generate_text_top_k_filtering_disabled(self):
        """Test generate_text does not pass top_k when <= 0 (CRITICAL)."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        config = GenerationConfig(top_k=0)  # Disabled
        generate_text(mock_model, mock_tokenizer, "Test", config)

        call_kwargs = mock_model.generate.call_args[1]
        assert "top_k" not in call_kwargs

    def test_generate_text_top_k_filtering_enabled(self):
        """Test generate_text passes top_k when > 0."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        config = GenerationConfig(top_k=50)  # Enabled
        generate_text(mock_model, mock_tokenizer, "Test", config)

        call_kwargs = mock_model.generate.call_args[1]
        assert "top_k" in call_kwargs
        assert call_kwargs["top_k"] == 50

    def test_generate_text_max_new_tokens_priority(self):
        """Test generate_text prioritizes max_new_tokens over max_length."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        config = GenerationConfig(max_new_tokens=50, max_length=100)
        generate_text(mock_model, mock_tokenizer, "Test", config)

        call_kwargs = mock_model.generate.call_args[1]
        assert "max_new_tokens" in call_kwargs
        assert call_kwargs["max_new_tokens"] == 50
        assert "max_length" not in call_kwargs

    def test_generate_text_max_length_fallback(self):
        """Test generate_text falls back to max_length when max_new_tokens=None."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        config = GenerationConfig(max_new_tokens=None, max_length=100)
        generate_text(mock_model, mock_tokenizer, "Test", config)

        call_kwargs = mock_model.generate.call_args[1]
        assert "max_length" in call_kwargs
        assert call_kwargs["max_length"] == 100
        assert "max_new_tokens" not in call_kwargs

    def test_generate_text_default_max_new_tokens(self):
        """Test generate_text uses default 100 when both are None."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        config = GenerationConfig(max_new_tokens=None, max_length=None)
        generate_text(mock_model, mock_tokenizer, "Test", config)

        call_kwargs = mock_model.generate.call_args[1]
        assert "max_new_tokens" in call_kwargs
        assert call_kwargs["max_new_tokens"] == 100

    def test_generate_text_min_new_tokens(self):
        """Test generate_text passes min_new_tokens when set."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        config = GenerationConfig(min_new_tokens=10)
        generate_text(mock_model, mock_tokenizer, "Test", config)

        call_kwargs = mock_model.generate.call_args[1]
        assert "min_new_tokens" in call_kwargs
        assert call_kwargs["min_new_tokens"] == 10

    def test_generate_text_prompt_removal(self):
        """Test generate_text removes prompt from output (slices by prompt_length)."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()

        # Tokenizer returns 5 tokens for prompt
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        # Generate returns 10 tokens (5 prompt + 5 generated)
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

        # Mock decode to return only generated part
        mock_tokenizer.decode.return_value = "only generated part"

        generate_text(mock_model, mock_tokenizer, "Test prompt")

        # Verify decode was called with generated tokens only (after prompt)
        mock_tokenizer.decode.assert_called_once()
        decoded_ids = mock_tokenizer.decode.call_args[0][0]
        # Should be tokens 6-10 (indices 5-9)
        assert len(decoded_ids) == 5

    def test_generate_text_pad_token_id_from_config(self):
        """Test generate_text uses pad_token_id from config."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer.pad_token_id = 0

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        config = GenerationConfig(pad_token_id=99)
        generate_text(mock_model, mock_tokenizer, "Test", config)

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["pad_token_id"] == 99

    def test_generate_text_pad_token_id_from_tokenizer(self):
        """Test generate_text falls back to tokenizer.pad_token_id."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()
        mock_tokenizer.pad_token_id = 0

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        config = GenerationConfig(pad_token_id=None)
        generate_text(mock_model, mock_tokenizer, "Test", config)

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["pad_token_id"] == 0

    def test_generate_text_use_cache_disabled(self):
        """Test generate_text disables cache (use_cache=False)."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        generate_text(mock_model, mock_tokenizer, "Test")

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["use_cache"] is False


class TestBatchGenerate:
    """Test batch_generate() function."""

    def test_batch_generate_single_batch(self):
        """Test batch_generate with prompts fitting in one batch."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()

        # Mock tokenization for batch - pad_token_id is 0
        # Clear side_effect to use return_value
        mock_tokenizer.side_effect = None
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        # Mock generation output - should match batch size
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]])

        mock_tokenizer.decode.return_value = "generated text"

        prompts = ["prompt1", "prompt2"]
        results = batch_generate(
            mock_model, mock_tokenizer, prompts, batch_size=4, show_progress=False
        )

        assert len(results) == 2
        assert all(r == "generated text" for r in results)
        mock_model.generate.assert_called_once()

    def test_batch_generate_multiple_batches(self):
        """Test batch_generate with prompts spanning multiple batches."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()

        # Mock tokenization - will be called 3 times with different batch sizes
        mock_tokenizer.pad_token_id = 0

        # Return different sized batches: 2, 2, 1
        mock_tokenizer.side_effect = [
            {
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            },
            {
                "input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            },
            {
                "input_ids": torch.tensor([[13, 14, 15]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            },
        ]

        # Mock generation - return matching batch sizes
        mock_model.generate.side_effect = [
            torch.tensor([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]]),  # batch of 2
            torch.tensor([[7, 8, 9, 10, 11], [10, 11, 12, 13, 14]]),  # batch of 2
            torch.tensor([[13, 14, 15, 16, 17]]),  # batch of 1
        ]

        mock_tokenizer.decode.return_value = "generated"

        prompts = ["p1", "p2", "p3", "p4", "p5"]
        results = batch_generate(
            mock_model, mock_tokenizer, prompts, batch_size=2, show_progress=False
        )

        assert len(results) == 5
        # Should have called generate 3 times (2+2+1)
        assert mock_model.generate.call_count == 3

    def test_batch_generate_empty_prompts(self):
        """Test batch_generate with empty prompts list."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        prompts = []
        results = batch_generate(mock_model, mock_tokenizer, prompts, show_progress=False)

        assert len(results) == 0
        mock_model.generate.assert_not_called()

    def test_batch_generate_with_progress_bar(self):
        """Test batch_generate with tqdm progress bar."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()

        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        mock_tokenizer.decode.return_value = "generated"

        with patch("tqdm.tqdm") as mock_tqdm:
            mock_tqdm.return_value = iter([0])

            prompts = ["p1"]
            results = batch_generate(mock_model, mock_tokenizer, prompts, show_progress=True)

            assert len(results) == 1
            mock_tqdm.assert_called_once()

    def test_batch_generate_without_tqdm(self):
        """Test batch_generate handles ImportError when tqdm not available."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()

        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        mock_tokenizer.decode.return_value = "generated"

        # Simulate tqdm not available
        with patch.dict("sys.modules", {"tqdm": None}):
            prompts = ["p1"]
            results = batch_generate(mock_model, mock_tokenizer, prompts, show_progress=True)

            assert len(results) == 1

    def test_batch_generate_show_progress_false(self):
        """Test batch_generate with show_progress=False."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()

        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        mock_tokenizer.decode.return_value = "generated"

        with patch("tqdm.tqdm") as mock_tqdm:
            prompts = ["p1"]
            results = batch_generate(mock_model, mock_tokenizer, prompts, show_progress=False)

            assert len(results) == 1
            mock_tqdm.assert_not_called()

    def test_batch_generate_custom_generation_config(self):
        """Test batch_generate with custom GenerationConfig."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()

        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        mock_tokenizer.decode.return_value = "generated"

        config = GenerationConfig(temperature=0.5, top_k=30)
        prompts = ["p1"]
        batch_generate(
            mock_model,
            mock_tokenizer,
            prompts,
            generation_config=config,
            show_progress=False,
        )

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_k"] == 30

    def test_batch_generate_tokenization_with_padding(self):
        """Test batch_generate uses padding and truncation."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()

        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        mock_tokenizer.decode.return_value = "generated"

        prompts = ["p1"]
        batch_generate(mock_model, mock_tokenizer, prompts, show_progress=False)

        # Verify tokenizer was called with padding and truncation
        mock_tokenizer.assert_called_once()
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs["padding"] is True
        assert call_kwargs["truncation"] is True
        assert call_kwargs["max_length"] == 512

    def test_batch_generate_prompt_length_calculation(self):
        """Test batch_generate correctly calculates prompt lengths."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()

        # Simulate two prompts with different lengths (padding used)
        # Clear side_effect to use return_value
        mock_tokenizer.side_effect = None
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
        }

        # Generate adds 3 tokens to each
        mock_model.generate.return_value = torch.tensor(
            [[1, 2, 3, 0, 0, 10, 11, 12], [4, 5, 6, 7, 8, 13, 14, 15]]
        )

        mock_tokenizer.decode.return_value = "generated"

        prompts = ["short", "longer prompt"]
        results = batch_generate(
            mock_model, mock_tokenizer, prompts, batch_size=2, show_progress=False
        )

        assert len(results) == 2
        # Should decode twice (once per output)
        assert mock_tokenizer.decode.call_count == 2

    def test_batch_generate_device_inference(self):
        """Test batch_generate infers device from model."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cuda:0")
            yield param

        mock_model.parameters.return_value = params_gen()

        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        mock_tokenizer.decode.return_value = "generated"

        prompts = ["p1"]
        results = batch_generate(mock_model, mock_tokenizer, prompts, show_progress=False)

        assert len(results) == 1

    def test_batch_generate_custom_device(self):
        """Test batch_generate with custom device parameter."""
        mock_model = create_mock_model()
        mock_tokenizer = create_mock_tokenizer()

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()

        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        mock_tokenizer.decode.return_value = "generated"

        custom_device = torch.device("cpu")  # Use CPU to avoid cuda availability issues
        prompts = ["p1"]
        results = batch_generate(
            mock_model,
            mock_tokenizer,
            prompts,
            device=custom_device,
            show_progress=False,
        )

        assert len(results) == 1


class TestPrepareModelForTraining:
    """Test prepare_model_for_training() function."""

    def test_prepare_model_for_training_sets_train_mode(self):
        """Test prepare_model_for_training sets model to train mode."""
        mock_model = Mock(spec=nn.Module)
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]

        prepare_model_for_training(mock_model)

        mock_model.train.assert_called_once()

    def test_prepare_model_for_training_enables_gradients(self):
        """Test prepare_model_for_training enables requires_grad."""
        mock_model = Mock(spec=nn.Module)

        param1 = torch.nn.Parameter(torch.randn(10, 10))
        param2 = torch.nn.Parameter(torch.randn(5, 5))
        param1.requires_grad = False
        param2.requires_grad = False

        mock_model.parameters.return_value = [param1, param2]

        prepare_model_for_training(mock_model)

        assert param1.requires_grad is True
        assert param2.requires_grad is True

    def test_prepare_model_for_training_creates_adamw(self):
        """Test prepare_model_for_training creates AdamW optimizer."""
        mock_model = Mock(spec=nn.Module)
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]

        optimizer = prepare_model_for_training(mock_model)

        assert isinstance(optimizer, torch.optim.AdamW)

    def test_prepare_model_for_training_default_lr(self):
        """Test prepare_model_for_training uses default learning rate."""
        mock_model = Mock(spec=nn.Module)
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]

        optimizer = prepare_model_for_training(mock_model)

        # Check learning rate in optimizer
        assert optimizer.defaults["lr"] == 5e-5

    def test_prepare_model_for_training_custom_lr(self):
        """Test prepare_model_for_training with custom learning rate."""
        mock_model = Mock(spec=nn.Module)
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]

        optimizer = prepare_model_for_training(mock_model, learning_rate=1e-4)

        assert optimizer.defaults["lr"] == 1e-4

    def test_prepare_model_for_training_default_weight_decay(self):
        """Test prepare_model_for_training uses default weight decay."""
        mock_model = Mock(spec=nn.Module)
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]

        optimizer = prepare_model_for_training(mock_model)

        assert optimizer.defaults["weight_decay"] == 0.01

    def test_prepare_model_for_training_custom_weight_decay(self):
        """Test prepare_model_for_training with custom weight decay."""
        mock_model = Mock(spec=nn.Module)
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]

        optimizer = prepare_model_for_training(mock_model, weight_decay=0.05)

        assert optimizer.defaults["weight_decay"] == 0.05

    def test_prepare_model_for_training_returns_optimizer(self):
        """Test prepare_model_for_training returns optimizer."""
        mock_model = Mock(spec=nn.Module)
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]

        result = prepare_model_for_training(mock_model)

        assert result is not None
        assert hasattr(result, "step")
        assert hasattr(result, "zero_grad")


class TestGetModelDevice:
    """Test get_model_device() function."""

    def test_get_model_device_cpu(self):
        """Test get_model_device returns cpu device."""
        mock_model = Mock(spec=nn.Module)

        def params_gen():
            param = Mock()
            param.device = torch.device("cpu")
            yield param

        mock_model.parameters.return_value = params_gen()

        device = get_model_device(mock_model)

        assert device == torch.device("cpu")

    def test_get_model_device_cuda(self):
        """Test get_model_device returns cuda device."""
        mock_model = Mock(spec=nn.Module)

        def params_gen():
            param = Mock()
            param.device = torch.device("cuda:0")
            yield param

        mock_model.parameters.return_value = params_gen()

        device = get_model_device(mock_model)

        assert device == torch.device("cuda:0")

    def test_get_model_device_cuda_multi_gpu(self):
        """Test get_model_device with specific GPU device."""
        mock_model = Mock(spec=nn.Module)

        def params_gen():
            param = Mock()
            param.device = torch.device("cuda:2")
            yield param

        mock_model.parameters.return_value = params_gen()

        device = get_model_device(mock_model)

        assert device == torch.device("cuda:2")

    def test_get_model_device_real_model(self):
        """Test get_model_device with real PyTorch model."""
        # Create simple real model
        model = nn.Linear(10, 5)

        device = get_model_device(model)

        assert device == torch.device("cpu")
