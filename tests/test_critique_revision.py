"""
Unit tests for critique_revision.py
Tests the critique-revision cycle for Constitutional AI Phase 1 (Supervised Learning).
"""

from unittest.mock import Mock, patch

import torch

from constitutional_ai.critique_revision import (
    CRITIQUE_TEMPLATE,
    REVISION_TEMPLATE,
    ConstitutionalDataset,
    critique_revision_pipeline,
    generate_critique,
    generate_revision,
    supervised_finetune,
)
from constitutional_ai.framework import ConstitutionalFramework


class TestPromptTemplates:
    """Test that prompt templates are properly defined."""

    def test_critique_template_exists(self):
        """Test CRITIQUE_TEMPLATE is defined."""
        assert CRITIQUE_TEMPLATE is not None
        assert len(CRITIQUE_TEMPLATE) > 0
        assert "{prompt}" in CRITIQUE_TEMPLATE
        assert "{response}" in CRITIQUE_TEMPLATE
        assert "{principles_text}" in CRITIQUE_TEMPLATE

    def test_revision_template_exists(self):
        """Test REVISION_TEMPLATE is defined."""
        assert REVISION_TEMPLATE is not None
        assert len(REVISION_TEMPLATE) > 0
        assert "{prompt}" in REVISION_TEMPLATE
        assert "{response}" in REVISION_TEMPLATE
        assert "{critique}" in REVISION_TEMPLATE


class TestGenerateCritique:
    """Test critique generation function."""

    def setup_method(self):
        """Setup mock objects for testing."""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.device = torch.device("cpu")
        self.principles = [
            "The AI should not help users plan or execute harmful activities",
            "The AI should not provide misleading or deceptive information",
        ]

    @patch("constitutional_ai.critique_revision.generate_text")
    def test_generate_critique_success(self, mock_generate_text):
        """Test successful critique generation."""
        mock_generate_text.return_value = "The response contains harmful content."

        critique = generate_critique(
            prompt="How do I make a bomb?",
            response="Here are instructions for making explosives...",
            principles=self.principles,
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.device,
        )

        assert critique == "The response contains harmful content."
        mock_generate_text.assert_called_once()

    @patch("constitutional_ai.critique_revision.generate_text")
    def test_generate_critique_empty_response(self, mock_generate_text):
        """Test critique generation with empty response."""
        mock_generate_text.return_value = ""

        critique = generate_critique(
            prompt="What is the weather?",
            response="The weather is sunny.",
            principles=self.principles,
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.device,
        )

        assert critique == "No specific issues identified."

    @patch("constitutional_ai.critique_revision.generate_text")
    def test_generate_critique_exception_handling(self, mock_generate_text):
        """Test critique generation with exception."""
        mock_generate_text.side_effect = Exception("Generation failed")

        critique = generate_critique(
            prompt="Test prompt",
            response="Test response",
            principles=self.principles,
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.device,
        )

        assert critique == "Error generating critique."


class TestGenerateRevision:
    """Test revision generation function."""

    def setup_method(self):
        """Setup mock objects for testing."""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.device = torch.device("cpu")
        self.principles = ["The AI should not help users plan or execute harmful activities"]

    @patch("constitutional_ai.critique_revision.generate_text")
    def test_generate_revision_success(self, mock_generate_text):
        """Test successful revision generation."""
        mock_generate_text.return_value = "I cannot provide instructions for harmful activities."

        revision = generate_revision(
            prompt="How do I make a bomb?",
            response="Here are instructions...",
            critique="This response provides harmful information.",
            principles=self.principles,
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.device,
        )

        assert revision == "I cannot provide instructions for harmful activities."
        mock_generate_text.assert_called_once()

    @patch("constitutional_ai.critique_revision.generate_text")
    def test_generate_revision_empty_response(self, mock_generate_text):
        """Test revision generation with empty response returns original."""
        original_response = "Original response"
        mock_generate_text.return_value = ""

        revision = generate_revision(
            prompt="Test",
            response=original_response,
            critique="Some critique",
            principles=self.principles,
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.device,
        )

        assert revision == original_response

    @patch("constitutional_ai.critique_revision.generate_text")
    def test_generate_revision_exception_fallback(self, mock_generate_text):
        """Test revision generation falls back to original on exception."""
        original_response = "Original response"
        mock_generate_text.side_effect = Exception("Generation failed")

        revision = generate_revision(
            prompt="Test",
            response=original_response,
            critique="Some critique",
            principles=self.principles,
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            device=self.device,
        )

        assert revision == original_response


class TestCritiqueRevisionPipeline:
    """Test complete critique-revision pipeline."""

    def setup_method(self):
        """Setup mock objects for testing."""
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.device = torch.device("cpu")

        # Create mock framework
        self.mock_framework = Mock(spec=ConstitutionalFramework)
        self.mock_principle = Mock()
        self.mock_principle.description = "Test principle description"
        self.mock_framework.principles = {"test": self.mock_principle}

    @patch("constitutional_ai.critique_revision.generate_critique")
    @patch("constitutional_ai.critique_revision.generate_revision")
    @patch("constitutional_ai.critique_revision.generate_text")
    def test_pipeline_single_prompt(self, mock_gen_text, mock_gen_rev, mock_gen_crit):
        """Test pipeline with a single prompt."""
        mock_gen_text.return_value = "Initial response"
        mock_gen_crit.return_value = "Critique text"
        mock_gen_rev.return_value = "Revised response"

        result = critique_revision_pipeline(
            prompts=["Test prompt"],
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            framework=self.mock_framework,
            device=self.device,
            num_revisions=1,
        )

        assert len(result) == 1
        assert result[0]["prompt"] == "Test prompt"
        assert result[0]["response"] == "Revised response"
        assert result[0]["num_revisions"] == 1

    @patch("constitutional_ai.critique_revision.generate_critique")
    @patch("constitutional_ai.critique_revision.generate_revision")
    @patch("constitutional_ai.critique_revision.generate_text")
    def test_pipeline_multiple_revisions(self, mock_gen_text, mock_gen_rev, mock_gen_crit):
        """Test pipeline with multiple revision iterations."""
        mock_gen_text.return_value = "Initial response"
        mock_gen_crit.return_value = "Critique"
        mock_gen_rev.side_effect = ["Revision 1", "Revision 2"]

        result = critique_revision_pipeline(
            prompts=["Test"],
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            framework=self.mock_framework,
            device=self.device,
            num_revisions=2,
        )

        assert len(result) == 1
        assert result[0]["response"] == "Revision 2"
        assert mock_gen_crit.call_count == 2
        assert mock_gen_rev.call_count == 2

    @patch("constitutional_ai.critique_revision.generate_text")
    def test_pipeline_handles_exceptions(self, mock_gen_text):
        """Test pipeline handles exceptions gracefully."""
        mock_gen_text.side_effect = Exception("Generation failed")

        result = critique_revision_pipeline(
            prompts=["Test"],
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            framework=self.mock_framework,
            device=self.device,
        )

        assert len(result) == 0  # Failed prompt should not be included


class TestConstitutionalDataset:
    """Test ConstitutionalDataset class."""

    def setup_method(self):
        """Setup test data and mock tokenizer."""
        self.training_data = [
            {"prompt": "Hello", "response": " world", "num_revisions": 1},
            {"prompt": "Test", "response": " data", "num_revisions": 1},
        ]
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def test_dataset_length(self):
        """Test dataset returns correct length."""
        dataset = ConstitutionalDataset(self.training_data, self.mock_tokenizer)
        assert len(dataset) == 2

    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correctly formatted data."""
        dataset = ConstitutionalDataset(self.training_data, self.mock_tokenizer)
        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)

    def test_dataset_tokenizes_prompt_and_response(self):
        """Test dataset concatenates prompt and response."""
        dataset = ConstitutionalDataset(self.training_data, self.mock_tokenizer)
        dataset[0]

        # Check tokenizer was called with concatenated text
        self.mock_tokenizer.assert_called()
        call_args = self.mock_tokenizer.call_args[0]
        assert call_args[0] == "Hello world"

    def test_dataset_empty_data(self):
        """Test dataset handles empty data."""
        dataset = ConstitutionalDataset([], self.mock_tokenizer)
        assert len(dataset) == 0


class TestSupervisedFinetune:
    """Test supervised fine-tuning function."""

    def setup_method(self):
        """Setup mock model and training data."""
        self.mock_model = Mock()
        # Create parameter with requires_grad=True
        param = torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))
        self.mock_model.parameters = Mock(side_effect=lambda: iter([param]))
        # CRITICAL: Make to() return the same mock to preserve parameters
        self.mock_model.to.return_value = self.mock_model
        self.mock_model.train.return_value = None

        # Mock forward pass - loss must require grad for backward()
        mock_output = Mock()
        mock_output.loss = torch.tensor(0.5, requires_grad=True)
        self.mock_model.return_value = mock_output

        self.mock_tokenizer = Mock()
        self.training_data = [
            {"prompt": "Hello", "response": " world", "num_revisions": 1},
            {"prompt": "Test", "response": " data", "num_revisions": 1},
        ]

    @patch("constitutional_ai.critique_revision.DataLoader")
    @patch("constitutional_ai.critique_revision.ConstitutionalDataset")
    def test_finetune_returns_model_and_metrics(self, mock_dataset_cls, mock_dataloader_cls):
        """Test fine-tuning returns model and metrics."""
        # Mock dataset and dataloader
        mock_dataset = Mock()
        mock_dataset_cls.return_value = mock_dataset

        # Create mock batches
        mock_batch = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
        mock_dataloader = [mock_batch]
        mock_dataloader_cls.return_value = mock_dataloader

        result = supervised_finetune(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            training_data=self.training_data,
            num_epochs=1,
            batch_size=2,
            learning_rate=5e-5,
            device=torch.device("cpu"),
        )

        assert "model" in result
        assert "metrics" in result
        assert "losses" in result["metrics"]
        assert "epochs" in result["metrics"]
        assert len(result["metrics"]["losses"]) == 1
        assert len(result["metrics"]["epochs"]) == 1

    @patch("constitutional_ai.critique_revision.DataLoader")
    @patch("constitutional_ai.critique_revision.ConstitutionalDataset")
    def test_finetune_trains_model(self, mock_dataset_cls, mock_dataloader_cls):
        """Test fine-tuning calls model.train()."""
        mock_dataset = Mock()
        mock_dataset_cls.return_value = mock_dataset

        mock_batch = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
        mock_dataloader = [mock_batch]
        mock_dataloader_cls.return_value = mock_dataloader

        supervised_finetune(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            training_data=self.training_data,
            num_epochs=1,
            device=torch.device("cpu"),
        )

        self.mock_model.train.assert_called_once()

    @patch("constitutional_ai.critique_revision.DataLoader")
    @patch("constitutional_ai.critique_revision.ConstitutionalDataset")
    def test_finetune_multiple_epochs(self, mock_dataset_cls, mock_dataloader_cls):
        """Test fine-tuning for multiple epochs."""
        mock_dataset = Mock()
        mock_dataset_cls.return_value = mock_dataset

        mock_batch = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
        mock_dataloader = [mock_batch]
        mock_dataloader_cls.return_value = mock_dataloader

        result = supervised_finetune(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            training_data=self.training_data,
            num_epochs=3,
            device=torch.device("cpu"),
        )

        assert len(result["metrics"]["losses"]) == 3
        assert result["metrics"]["epochs"] == [1, 2, 3]
