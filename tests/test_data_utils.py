"""
Unit tests for data_utils.py

Tests helper functions for creating test prompts and managing datasets.
"""

from constitutional_ai.data_utils import create_default_prompts


class TestCreateDefaultPrompts:
    """Test create_default_prompts function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        result = create_default_prompts()
        assert isinstance(result, list)

    def test_correct_length(self):
        """Test that list contains 15 prompts."""
        result = create_default_prompts()
        assert len(result) == 15

    def test_all_strings(self):
        """Test that all elements are strings."""
        result = create_default_prompts()
        assert all(isinstance(prompt, str) for prompt in result)

    def test_all_non_empty(self):
        """Test that all prompts are non-empty."""
        result = create_default_prompts()
        assert all(len(prompt) > 0 for prompt in result)

    def test_most_end_with_question_mark(self):
        """Test that most prompts end with question mark."""
        result = create_default_prompts()
        # Most prompts should be questions
        question_count = sum(1 for prompt in result if prompt.endswith("?"))
        assert question_count >= 10  # At least 10 out of 15 should be questions

    def test_covers_different_principles(self):
        """Test that prompts cover different constitutional principles."""
        result = create_default_prompts()

        # Check that we have diverse prompts
        # Should include health, conflict, truthfulness, fairness, autonomy topics
        combined_text = " ".join(result).lower()

        # Harm prevention topics
        assert "health" in combined_text or "fitness" in combined_text
        assert "conflict" in combined_text or "peaceful" in combined_text

        # Truthfulness topics
        assert "capital" in combined_text or "france" in combined_text
        assert "vaccine" in combined_text or "climate" in combined_text

        # Fairness topics
        assert "leader" in combined_text or "diversity" in combined_text

        # Autonomy topics
        assert "career" in combined_text or "decision" in combined_text

    def test_prompts_are_sensible_questions(self):
        """Test that prompts are reasonable questions."""
        result = create_default_prompts()

        # Should contain common question words
        combined_text = " ".join(result).lower()
        question_words = ["how", "what", "why", "when", "where", "who"]

        # At least some prompts should use question words
        assert any(word in combined_text for word in question_words)

    def test_deterministic_output(self):
        """Test that function returns same prompts on multiple calls."""
        result1 = create_default_prompts()
        result2 = create_default_prompts()

        assert result1 == result2
        assert len(result1) == len(result2) == 15
