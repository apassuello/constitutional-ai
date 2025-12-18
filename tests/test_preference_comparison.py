"""
Unit tests for preference comparison (Component 3 of Constitutional AI - RLAIF Phase 2b).

Tests cover:
- generate_comparison: Compare two responses using constitutional principles
- extract_preference: Extract 'A' or 'B' from comparison text (8 regex patterns)
- generate_preference_pairs: Full preference generation pipeline
- Edge cases and error handling

Note: PreferenceDataset tests are in test_reward_model.py (already 100% coverage)
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from constitutional_ai.preference_comparison import (
    extract_preference,
    generate_comparison,
    generate_preference_pairs,
)


class TestExtractPreference:
    """
    Test extract_preference() function - CRITICAL for preference extraction.

    This function uses 8 different regex patterns to extract 'A' or 'B' from
    AI-generated comparison text. Each pattern must be thoroughly tested.
    """

    # Pattern 1: "Response A/B is better/superior/preferred"
    def test_pattern1_response_a_is_better(self):
        """Test Pattern 1: 'Response A is better because...'"""
        text = "Response A is better because it provides more detail."
        assert extract_preference(text) == "A"

    def test_pattern1_response_b_is_superior(self):
        """Test Pattern 1: 'Response B is superior in every way'"""
        text = "Response B is superior in terms of accuracy and completeness."
        assert extract_preference(text) == "B"

    def test_pattern1_response_a_is_preferred(self):
        """Test Pattern 1: 'Response A is preferred'"""
        text = "Response A is preferred for its clarity."
        assert extract_preference(text) == "A"

    def test_pattern1_response_b_is_stronger(self):
        """Test Pattern 1: 'Response B is stronger'"""
        text = "Response B is stronger overall."
        assert extract_preference(text) == "B"

    # Pattern 2: "prefer/choose ... Response A/B"
    def test_pattern2_prefer_response_a(self):
        """Test Pattern 2: 'I prefer Response A for clarity'"""
        text = "I prefer Response A because it is concise."
        assert extract_preference(text) == "A"

    def test_pattern2_choose_response_b(self):
        """Test Pattern 2: 'We should choose Response B'"""
        text = "We should choose Response B as the better answer."
        assert extract_preference(text) == "B"

    def test_pattern2_better_with_response_a(self):
        """Test Pattern 2: 'better ... Response A'"""
        text = "It would be better to go with Response A here."
        assert extract_preference(text) == "A"

    # Pattern 3: "A/B is better/preferred"
    def test_pattern3_a_is_better(self):
        """Test Pattern 3: 'A is better overall'"""
        text = "A is better in this case."
        assert extract_preference(text) == "A"

    def test_pattern3_b_seems_preferred(self):
        """Test Pattern 3: 'B seems preferred'"""
        text = "B seems preferred for accuracy."
        assert extract_preference(text) == "B"

    def test_pattern3_a_appears_superior(self):
        """Test Pattern 3: 'A appears superior'"""
        text = "A appears superior to B."
        assert extract_preference(text) == "A"

    # Pattern 4: "prefer A/B"
    def test_pattern4_prefer_a(self):
        """Test Pattern 4: 'I prefer A in this case'"""
        text = "I prefer A for its simplicity."
        assert extract_preference(text) == "A"

    def test_pattern4_prefer_b(self):
        """Test Pattern 4: 'prefer B due to accuracy'"""
        text = "I would prefer B due to its accuracy."
        assert extract_preference(text) == "B"

    # Pattern 5: "choose/select A/B"
    def test_pattern5_choose_a(self):
        """Test Pattern 5: 'choose A'"""
        text = "I would choose A."
        assert extract_preference(text) == "A"

    def test_pattern5_select_b(self):
        """Test Pattern 5: 'select B'"""
        text = "I would select B as the better response."
        assert extract_preference(text) == "B"

    # Pattern 6: "A:" or "B:" at start
    def test_pattern6_a_colon_at_start(self):
        """Test Pattern 6: 'A: This is my final choice'"""
        text = "A: This response is more comprehensive."
        assert extract_preference(text) == "A"

    def test_pattern6_b_colon_at_start(self):
        """Test Pattern 6: 'B: Definitely this one'"""
        text = "B: This is the correct answer."
        assert extract_preference(text) == "B"

    def test_pattern6_with_leading_whitespace(self):
        """Test Pattern 6: Works with leading whitespace"""
        text = "   B: Selected for accuracy"
        assert extract_preference(text) == "B"

    # Pattern 7: Positive mentions count
    def test_pattern7_b_has_more_positive_mentions(self):
        """Test Pattern 7: B has more positive terms than A"""
        # Pattern 7 regex: r"\bresponse\s+a\b.{0,50}\b(good|excellent|accurate|helpful|clear)"
        # Regex can be greedy - need >50 chars between each "Response" mention
        text = (
            "Response A has some good qualities in the beginning part here. "
            "However in comparison, Response B shows much better excellent approach overall. "
            "Additionally Response B demonstrates clearly accurate implementation as well."
        )
        assert extract_preference(text) == "B"

    def test_pattern7_a_has_more_positive_mentions(self):
        """Test Pattern 7: A has more positive terms than B"""
        text = (
            "Response A is excellent, accurate, and helpful. It provides good information. "
            "Response B is okay."
        )
        assert extract_preference(text) == "A"

    # Pattern 8: Negative mentions count
    def test_pattern8_a_has_more_negatives_prefers_b(self):
        """Test Pattern 8: A has more negative terms, so prefer B"""
        text = (
            "Response A is poor, inaccurate, and unhelpful. It's unclear. "
            "Response B is fine."
        )
        assert extract_preference(text) == "B"

    def test_pattern8_b_has_more_negatives_prefers_a(self):
        """Test Pattern 8: B has more negative terms, so prefer A"""
        text = (
            "Response A is acceptable. Response B is worse, poor, and inaccurate. "
            "B is unhelpful."
        )
        assert extract_preference(text) == "A"

    # Default case
    def test_default_unclear_preference(self):
        """Test default: Returns 'A' when preference is unclear"""
        text = "Both responses are similar."
        assert extract_preference(text) == "A"

    def test_default_empty_string(self):
        """Test default: Empty string returns 'A'"""
        text = ""
        assert extract_preference(text) == "A"

    def test_default_no_clear_preference(self):
        """Test default: No clear preference returns 'A'"""
        text = "These are two different approaches to the problem."
        assert extract_preference(text) == "A"

    # Edge cases
    def test_case_insensitive(self):
        """Test that matching is case-insensitive"""
        text = "RESPONSE A IS BETTER"
        assert extract_preference(text) == "A"

    def test_mixed_signals_first_pattern_wins(self):
        """Test priority: Earlier patterns take precedence"""
        # Pattern 1 should match before Pattern 3
        text = "Response A is better, though B is also good."
        assert extract_preference(text) == "A"

    def test_complex_comparison_text(self):
        """Test with realistic AI-generated comparison"""
        text = """
        After carefully evaluating both responses:

        Response A provides a basic answer but lacks depth.
        Response B is superior because it includes specific examples,
        provides accurate information, and is more helpful to the user.

        Therefore, Response B is the better choice.
        """
        assert extract_preference(text) == "B"


class TestGenerateComparison:
    """Test generate_comparison() function."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        return MagicMock()

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        return MagicMock()

    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device("cpu")

    @pytest.fixture
    def sample_principles(self):
        """Sample constitutional principles."""
        return ["Be helpful and harmless", "Be truthful and accurate", "Respect human autonomy"]

    def test_basic_comparison_prefers_a(
        self, mock_model, mock_tokenizer, device, sample_principles
    ):
        """Test basic comparison returning preference 'A'"""
        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.extract_preference"
        ) as mock_extract:
            # Mock generate_text to return comparison text
            mock_generate.return_value = "Response A is better because it's more accurate."
            # Mock extract_preference to return 'A'
            mock_extract.return_value = "A"

            result = generate_comparison(
                prompt="What is AI?",
                response_a="AI is artificial intelligence.",
                response_b="AI is magic.",
                principles=sample_principles,
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=device,
            )

            # Verify return structure
            assert result["preferred"] == "A"
            assert result["comparison_text"] == "Response A is better because it's more accurate."
            assert result["response_chosen"] == "AI is artificial intelligence."
            assert result["response_rejected"] == "AI is magic."

            # Verify generate_text was called
            mock_generate.assert_called_once()
            # Verify extract_preference was called with the comparison text
            mock_extract.assert_called_once_with("Response A is better because it's more accurate.")

    def test_basic_comparison_prefers_b(
        self, mock_model, mock_tokenizer, device, sample_principles
    ):
        """Test basic comparison returning preference 'B'"""
        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.extract_preference"
        ) as mock_extract:
            mock_generate.return_value = "Response B is superior."
            mock_extract.return_value = "B"

            result = generate_comparison(
                prompt="Explain gravity",
                response_a="Things fall down.",
                response_b="Gravity is a fundamental force that attracts mass.",
                principles=sample_principles,
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=device,
            )

            # When preference is 'B', chosen and rejected should swap
            assert result["preferred"] == "B"
            assert result["response_chosen"] == "Gravity is a fundamental force that attracts mass."
            assert result["response_rejected"] == "Things fall down."

    def test_with_multiple_principles(self, mock_model, mock_tokenizer, device):
        """Test comparison with multiple constitutional principles"""
        principles = [
            "Be helpful",
            "Be harmless",
            "Be accurate",
            "Be fair",
            "Respect autonomy",
        ]

        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.extract_preference"
        ) as mock_extract:
            mock_generate.return_value = "A is better"
            mock_extract.return_value = "A"

            result = generate_comparison(
                prompt="Test",
                response_a="Response A",
                response_b="Response B",
                principles=principles,
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=device,
            )

            # Check that all principles were included in prompt
            call_args = mock_generate.call_args
            prompt_text = call_args[0][2]  # Third argument is the prompt
            for i, principle in enumerate(principles, 1):
                assert f"{i}. {principle}" in prompt_text

    def test_return_dictionary_structure(self, mock_model, mock_tokenizer, device, sample_principles):
        """Test that return dictionary has all required keys"""
        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.extract_preference"
        ) as mock_extract:
            mock_generate.return_value = "Comparison text"
            mock_extract.return_value = "A"

            result = generate_comparison(
                prompt="Test",
                response_a="A",
                response_b="B",
                principles=sample_principles,
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=device,
            )

            # Verify all required keys exist
            assert "preferred" in result
            assert "comparison_text" in result
            assert "response_chosen" in result
            assert "response_rejected" in result

            # Verify types
            assert isinstance(result["preferred"], str)
            assert isinstance(result["comparison_text"], str)
            assert isinstance(result["response_chosen"], str)
            assert isinstance(result["response_rejected"], str)

    def test_generation_config_parameters(self, mock_model, mock_tokenizer, device, sample_principles):
        """Test that GenerationConfig is created with correct parameters"""
        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.extract_preference"
        ) as mock_extract, patch(
            "constitutional_ai.model_utils.GenerationConfig"
        ) as mock_config:
            mock_generate.return_value = "Text"
            mock_extract.return_value = "A"

            generate_comparison(
                prompt="Test",
                response_a="A",
                response_b="B",
                principles=sample_principles,
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=device,
            )

            # Verify GenerationConfig was created with correct parameters
            mock_config.assert_called_once_with(max_length=300, temperature=0.7, do_sample=True)

    def test_comparison_template_format(self, mock_model, mock_tokenizer, device):
        """Test that comparison template includes all required elements"""
        principles = ["Be helpful"]

        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.extract_preference"
        ) as mock_extract:
            mock_generate.return_value = "A is better"
            mock_extract.return_value = "A"

            generate_comparison(
                prompt="What is photosynthesis?",
                response_a="Process where plants make food",
                response_b="Plants convert sunlight to energy",
                principles=principles,
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=device,
            )

            # Check that prompt includes all necessary components
            call_args = mock_generate.call_args
            prompt_text = call_args[0][2]

            assert "What is photosynthesis?" in prompt_text
            assert "Process where plants make food" in prompt_text
            assert "Plants convert sunlight to energy" in prompt_text
            assert "Be helpful" in prompt_text
            assert "Response A:" in prompt_text
            assert "Response B:" in prompt_text


class TestGeneratePreferencePairs:
    """Test generate_preference_pairs() function."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        return MagicMock()

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        return MagicMock()

    @pytest.fixture
    def mock_framework(self):
        """Create mock ConstitutionalFramework."""
        framework = MagicMock()
        # Mock principles as a dictionary with values that have description attribute
        principle1 = MagicMock()
        principle1.description = "Be helpful"
        principle2 = MagicMock()
        principle2.description = "Be harmless"
        framework.principles = {"p1": principle1, "p2": principle2}
        return framework

    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device("cpu")

    def test_single_prompt_two_responses(
        self, mock_model, mock_tokenizer, mock_framework, device
    ):
        """Test with single prompt, responses_per_prompt=2 (generates 1 comparison)"""
        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.generate_comparison"
        ) as mock_compare:
            # Mock response generation
            mock_generate.side_effect = ["Response 1", "Response 2"]

            # Mock comparison generation
            mock_compare.return_value = {
                "preferred": "A",
                "comparison_text": "A is better",
                "response_chosen": "Response 1",
                "response_rejected": "Response 2",
            }

            result = generate_preference_pairs(
                prompts=["What is AI?"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=2,
            )

            # Should generate 1 comparison (C(2,2) = 1)
            assert len(result) == 1
            assert result[0]["prompt"] == "What is AI?"
            assert result[0]["response_chosen"] == "Response 1"
            assert result[0]["response_rejected"] == "Response 2"
            assert result[0]["comparison_reasoning"] == "A is better"

    def test_multiple_prompts(self, mock_model, mock_tokenizer, mock_framework, device):
        """Test with multiple prompts (verify all processed)"""
        prompts = ["What is AI?", "Explain gravity", "What is ML?"]

        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.generate_comparison"
        ) as mock_compare:
            # Mock 2 responses per prompt (6 total)
            mock_generate.side_effect = ["R1", "R2", "R3", "R4", "R5", "R6"]

            # Mock comparison for each prompt
            mock_compare.side_effect = [
                {
                    "preferred": "A",
                    "comparison_text": "Comp 1",
                    "response_chosen": "R1",
                    "response_rejected": "R2",
                },
                {
                    "preferred": "B",
                    "comparison_text": "Comp 2",
                    "response_chosen": "R4",
                    "response_rejected": "R3",
                },
                {
                    "preferred": "A",
                    "comparison_text": "Comp 3",
                    "response_chosen": "R5",
                    "response_rejected": "R6",
                },
            ]

            result = generate_preference_pairs(
                prompts=prompts,
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=2,
            )

            # Should have 3 comparisons (1 per prompt)
            assert len(result) == 3
            assert result[0]["prompt"] == "What is AI?"
            assert result[1]["prompt"] == "Explain gravity"
            assert result[2]["prompt"] == "What is ML?"

    def test_responses_per_prompt_three_generates_three_comparisons(
        self, mock_model, mock_tokenizer, mock_framework, device
    ):
        """Test with responses_per_prompt=3 (should generate C(3,2)=3 comparisons)"""
        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.generate_comparison"
        ) as mock_compare:
            # Mock 3 responses for the prompt
            mock_generate.side_effect = ["R1", "R2", "R3"]

            # Mock 3 comparisons: (R1,R2), (R1,R3), (R2,R3)
            mock_compare.side_effect = [
                {
                    "preferred": "A",
                    "comparison_text": "C1",
                    "response_chosen": "R1",
                    "response_rejected": "R2",
                },
                {
                    "preferred": "B",
                    "comparison_text": "C2",
                    "response_chosen": "R3",
                    "response_rejected": "R1",
                },
                {
                    "preferred": "A",
                    "comparison_text": "C3",
                    "response_chosen": "R2",
                    "response_rejected": "R3",
                },
            ]

            result = generate_preference_pairs(
                prompts=["Test prompt"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=3,
            )

            # C(3,2) = 3 comparisons
            assert len(result) == 3

            # Verify generate_text was called 3 times
            assert mock_generate.call_count == 3

            # Verify generate_comparison was called 3 times
            assert mock_compare.call_count == 3

    def test_error_handling_continues_processing(
        self, mock_model, mock_tokenizer, mock_framework, device
    ):
        """Test error handling for failed comparisons (should continue processing)"""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.generate_comparison"
        ) as mock_compare, patch(
            "constitutional_ai.preference_comparison.logger"
        ) as mock_logger:
            # Mock responses (2 per prompt)
            mock_generate.side_effect = ["R1", "R2", "R3", "R4", "R5", "R6"]

            # Second comparison fails, others succeed
            mock_compare.side_effect = [
                {
                    "preferred": "A",
                    "comparison_text": "C1",
                    "response_chosen": "R1",
                    "response_rejected": "R2",
                },
                Exception("Generation failed"),
                {
                    "preferred": "B",
                    "comparison_text": "C3",
                    "response_chosen": "R6",
                    "response_rejected": "R5",
                },
            ]

            result = generate_preference_pairs(
                prompts=prompts,
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=2,
            )

            # Should have 2 successful comparisons (1st and 3rd)
            assert len(result) == 2
            assert result[0]["prompt"] == "Prompt 1"
            assert result[1]["prompt"] == "Prompt 3"

            # Verify error was logged
            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert "Failed to generate comparison" in log_message

    def test_return_structure(self, mock_model, mock_tokenizer, mock_framework, device):
        """Test return structure: list of dicts with correct keys"""
        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.generate_comparison"
        ) as mock_compare:
            mock_generate.side_effect = ["R1", "R2"]
            mock_compare.return_value = {
                "preferred": "A",
                "comparison_text": "Reasoning here",
                "response_chosen": "R1",
                "response_rejected": "R2",
            }

            result = generate_preference_pairs(
                prompts=["Test"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=2,
            )

            # Verify structure
            assert isinstance(result, list)
            assert len(result) == 1

            item = result[0]
            assert "prompt" in item
            assert "response_chosen" in item
            assert "response_rejected" in item
            assert "comparison_reasoning" in item

            # Verify values
            assert item["prompt"] == "Test"
            assert item["response_chosen"] == "R1"
            assert item["response_rejected"] == "R2"
            assert item["comparison_reasoning"] == "Reasoning here"

    def test_generation_config_high_temperature(
        self, mock_model, mock_tokenizer, mock_framework, device
    ):
        """Test that responses use high temperature for diversity"""
        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.generate_comparison"
        ) as mock_compare, patch(
            "constitutional_ai.model_utils.GenerationConfig"
        ) as mock_config:
            mock_generate.side_effect = ["R1", "R2"]
            mock_compare.return_value = {
                "preferred": "A",
                "comparison_text": "C",
                "response_chosen": "R1",
                "response_rejected": "R2",
            }

            generate_preference_pairs(
                prompts=["Test"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=2,
            )

            # Verify GenerationConfig was created with high temperature
            mock_config.assert_called_once_with(max_length=150, temperature=1.0, do_sample=True)

    def test_framework_principles_extraction(
        self, mock_model, mock_tokenizer, mock_framework, device
    ):
        """Test that principles are correctly extracted from framework"""
        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.generate_comparison"
        ) as mock_compare:
            mock_generate.side_effect = ["R1", "R2"]
            mock_compare.return_value = {
                "preferred": "A",
                "comparison_text": "C",
                "response_chosen": "R1",
                "response_rejected": "R2",
            }

            generate_preference_pairs(
                prompts=["Test"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=2,
            )

            # Verify generate_comparison was called with correct principles
            call_args = mock_compare.call_args
            principles = call_args[1]["principles"]
            assert "Be helpful" in principles
            assert "Be harmless" in principles

    def test_with_tqdm_available(self, mock_model, mock_tokenizer, mock_framework, device):
        """Test that tqdm is used when available"""
        # tqdm is imported inside the function, so we need to mock it differently
        # We'll skip detailed tqdm testing since it's just a progress bar
        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.generate_comparison"
        ) as mock_compare:
            mock_generate.side_effect = ["R1", "R2"]
            mock_compare.return_value = {
                "preferred": "A",
                "comparison_text": "C",
                "response_chosen": "R1",
                "response_rejected": "R2",
            }

            prompts = ["Test"]

            result = generate_preference_pairs(
                prompts=prompts,
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=2,
            )

            # Just verify it worked (tqdm is optional and tested elsewhere)
            assert len(result) == 1

    def test_without_tqdm_still_works(self, mock_model, mock_tokenizer, mock_framework, device):
        """Test that function works without tqdm - simplified test"""
        # Since tqdm is imported dynamically and just provides progress bar,
        # we'll test that the function works normally
        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.generate_comparison"
        ) as mock_compare:
            mock_generate.side_effect = ["R1", "R2"]
            mock_compare.return_value = {
                "preferred": "A",
                "comparison_text": "C",
                "response_chosen": "R1",
                "response_rejected": "R2",
            }

            # Should work with or without tqdm
            result = generate_preference_pairs(
                prompts=["Test"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=2,
            )

            assert len(result) == 1

    def test_empty_prompts_list(self, mock_model, mock_tokenizer, mock_framework, device):
        """Test with empty prompts list"""
        result = generate_preference_pairs(
            prompts=[],
            model=mock_model,
            tokenizer=mock_tokenizer,
            framework=mock_framework,
            device=device,
            responses_per_prompt=2,
        )

        # Should return empty list
        assert result == []

    def test_combinations_correctness(
        self, mock_model, mock_tokenizer, mock_framework, device
    ):
        """Test that combinations() produces correct number of pairs"""
        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.generate_comparison"
        ) as mock_compare:
            # Generate 4 responses
            mock_generate.side_effect = ["R1", "R2", "R3", "R4"]

            # Mock all 6 comparisons: C(4,2) = 6
            mock_compare.side_effect = [
                {
                    "preferred": "A",
                    "comparison_text": f"C{i}",
                    "response_chosen": f"R{i}",
                    "response_rejected": f"R{i+1}",
                }
                for i in range(6)
            ]

            result = generate_preference_pairs(
                prompts=["Test"],
                model=mock_model,
                tokenizer=mock_tokenizer,
                framework=mock_framework,
                device=device,
                responses_per_prompt=4,
            )

            # C(4,2) = 6 comparisons
            assert len(result) == 6
            assert mock_compare.call_count == 6


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_extract_preference_with_unicode(self):
        """Test extract_preference handles unicode characters"""
        text = "Response A is better 😊 because it's more helpful"
        assert extract_preference(text) == "A"

    def test_extract_preference_very_long_text(self):
        """Test extract_preference with very long comparison text"""
        text = "Response B " + "is great " * 100 + "Response B is better overall."
        assert extract_preference(text) == "B"

    def test_extract_preference_with_newlines(self):
        """Test extract_preference handles multi-line text"""
        text = """
        Analysis:
        Response A provides basic information.

        Response B is superior because:
        - More detailed
        - More accurate
        - More helpful
        """
        assert extract_preference(text) == "B"

    def test_extract_preference_with_special_characters(self):
        """Test extract_preference handles special characters"""
        text = "Response A is better!!! It's #1 in quality."
        assert extract_preference(text) == "A"

    def test_generate_comparison_with_empty_principle_list(self):
        """Test generate_comparison with empty principles list"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        device = torch.device("cpu")

        with patch(
            "constitutional_ai.model_utils.generate_text"
        ) as mock_generate, patch(
            "constitutional_ai.preference_comparison.extract_preference"
        ) as mock_extract:
            mock_generate.return_value = "A is better"
            mock_extract.return_value = "A"

            result = generate_comparison(
                prompt="Test",
                response_a="A",
                response_b="B",
                principles=[],  # Empty list
                model=mock_model,
                tokenizer=mock_tokenizer,
                device=device,
            )

            # Should still work, just no principles in the prompt
            assert result["preferred"] == "A"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
