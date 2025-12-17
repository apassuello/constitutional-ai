"""
Unit tests for framework.py
Tests the core Constitutional AI framework classes (ConstitutionalPrinciple and ConstitutionalFramework).
"""

from typing import Any

import pytest

from constitutional_ai.framework import ConstitutionalFramework, ConstitutionalPrinciple


class TestConstitutionalPrinciple:
    """Test ConstitutionalPrinciple class."""

    def setup_method(self):
        """Setup test fixtures."""

        # Simple evaluation function that always flags
        def eval_always_flag(text: str) -> dict[str, Any]:
            return {"flagged": True, "reason": "Test flagging", "score": 0.8}

        # Evaluation function that flags if "bad" in text
        def eval_contains_bad(text: str) -> dict[str, Any]:
            flagged = "bad" in text.lower()
            return {
                "flagged": flagged,
                "reason": "Contains 'bad'" if flagged else "Clean",
                "score": 1.0 if flagged else 0.0,
            }

        # Evaluation function that never flags
        def eval_never_flag(text: str) -> dict[str, Any]:
            return {"flagged": False, "reason": "Always passes", "score": 0.0}

        self.eval_always_flag = eval_always_flag
        self.eval_contains_bad = eval_contains_bad
        self.eval_never_flag = eval_never_flag

    def test_principle_initialization(self):
        """Test basic principle initialization."""
        principle = ConstitutionalPrinciple(
            name="test_principle",
            description="Test description",
            evaluation_fn=self.eval_always_flag,
            weight=0.5,
            enabled=True,
        )

        assert principle.name == "test_principle"
        assert principle.description == "Test description"
        assert principle.weight == 0.5
        assert principle.enabled is True
        assert callable(principle.evaluation_fn)

    def test_principle_default_values(self):
        """Test principle initialization with default values."""
        principle = ConstitutionalPrinciple(
            name="test", description="desc", evaluation_fn=self.eval_always_flag
        )

        assert principle.weight == 1.0
        assert principle.enabled is True

    def test_principle_evaluate_enabled(self):
        """Test evaluation when principle is enabled."""
        principle = ConstitutionalPrinciple(
            name="test", description="desc", evaluation_fn=self.eval_contains_bad, weight=0.7
        )

        result = principle.evaluate("This is bad text")

        assert result["flagged"] is True
        assert result["principle_name"] == "test"
        assert result["weight"] == 0.7
        assert result["reason"] == "Contains 'bad'"
        assert result["score"] == 1.0

    def test_principle_evaluate_disabled(self):
        """Test evaluation when principle is disabled."""
        principle = ConstitutionalPrinciple(
            name="test",
            description="desc",
            evaluation_fn=self.eval_always_flag,
            weight=0.7,
            enabled=False,
        )

        result = principle.evaluate("Any text")

        assert result["flagged"] is False
        assert result["reason"] == "Principle disabled"
        assert result["enabled"] is False
        assert "principle_name" in result
        assert "weight" in result

    def test_principle_evaluate_clean_text(self):
        """Test evaluation with clean text."""
        principle = ConstitutionalPrinciple(
            name="test", description="desc", evaluation_fn=self.eval_contains_bad
        )

        result = principle.evaluate("This is good text")

        assert result["flagged"] is False
        assert result["reason"] == "Clean"
        assert result["score"] == 0.0

    def test_principle_repr(self):
        """Test string representation of principle."""
        principle = ConstitutionalPrinciple(
            name="test_prin",
            description="desc",
            evaluation_fn=self.eval_always_flag,
            weight=0.5,
            enabled=True,
        )

        repr_str = repr(principle)
        assert "test_prin" in repr_str
        assert "0.5" in repr_str
        assert "enabled" in repr_str

    def test_principle_repr_disabled(self):
        """Test string representation when disabled."""
        principle = ConstitutionalPrinciple(
            name="test", description="desc", evaluation_fn=self.eval_always_flag, enabled=False
        )

        repr_str = repr(principle)
        assert "disabled" in repr_str

    def test_principle_with_complex_evaluation(self):
        """Test principle with complex evaluation function."""

        def complex_eval(text: str) -> dict[str, Any]:
            word_count = len(text.split())
            flagged = word_count > 10
            return {
                "flagged": flagged,
                "reason": f"Word count: {word_count}",
                "word_count": word_count,
                "details": {"threshold": 10},
            }

        principle = ConstitutionalPrinciple(
            name="length_check", description="Check text length", evaluation_fn=complex_eval
        )

        result = principle.evaluate("Short text")
        assert result["flagged"] is False
        assert result["word_count"] == 2

        result = principle.evaluate("This is a very long text with more than ten words for sure")
        assert result["flagged"] is True
        assert result["word_count"] > 10

    def test_principle_weight_variations(self):
        """Test principle with different weight values."""
        for weight in [0.0, 0.5, 1.0, 2.0, 10.0]:
            principle = ConstitutionalPrinciple(
                name=f"test_{weight}",
                description="desc",
                evaluation_fn=self.eval_always_flag,
                weight=weight,
            )

            result = principle.evaluate("test")
            assert result["weight"] == weight

    def test_principle_empty_text(self):
        """Test principle evaluation with empty text."""
        principle = ConstitutionalPrinciple(
            name="test", description="desc", evaluation_fn=self.eval_contains_bad
        )

        result = principle.evaluate("")
        assert "flagged" in result
        assert result["flagged"] is False  # Empty string doesn't contain "bad"


class TestConstitutionalFramework:
    """Test ConstitutionalFramework class."""

    def setup_method(self):
        """Setup test fixtures."""

        # Create sample evaluation functions
        def eval_harm(text: str) -> dict[str, Any]:
            harmful_words = ["kill", "hurt", "destroy"]
            flagged = any(word in text.lower() for word in harmful_words)
            return {
                "flagged": flagged,
                "reason": "Contains harmful content" if flagged else "No harm detected",
            }

        def eval_truthfulness(text: str) -> dict[str, Any]:
            lie_indicators = ["definitely", "guaranteed", "always", "never"]
            flagged = any(word in text.lower() for word in lie_indicators)
            return {
                "flagged": flagged,
                "reason": "Contains absolute claims" if flagged else "Truthful",
            }

        def eval_fairness(text: str) -> dict[str, Any]:
            bias_words = ["all", "every", "none"]
            flagged = any(word in text.lower() for word in bias_words)
            return {"flagged": flagged, "reason": "Contains generalization" if flagged else "Fair"}

        self.eval_harm = eval_harm
        self.eval_truthfulness = eval_truthfulness
        self.eval_fairness = eval_fairness

        # Create sample principles
        self.harm_principle = ConstitutionalPrinciple(
            name="harm_prevention",
            description="Prevents harmful content",
            evaluation_fn=eval_harm,
            weight=1.0,
        )

        self.truth_principle = ConstitutionalPrinciple(
            name="truthfulness",
            description="Ensures truthful responses",
            evaluation_fn=eval_truthfulness,
            weight=0.8,
        )

        self.fairness_principle = ConstitutionalPrinciple(
            name="fairness",
            description="Ensures fair treatment",
            evaluation_fn=eval_fairness,
            weight=0.6,
        )

    def test_framework_initialization(self):
        """Test basic framework initialization."""
        framework = ConstitutionalFramework(name="test_framework")

        assert framework.name == "test_framework"
        assert len(framework.principles) == 0
        assert len(framework.evaluation_history) == 0

    def test_framework_default_name(self):
        """Test framework with default name."""
        framework = ConstitutionalFramework()
        assert framework.name == "default_framework"

    def test_add_principle(self):
        """Test adding a principle to framework."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)

        assert len(framework.principles) == 1
        assert "harm_prevention" in framework.principles
        assert framework.principles["harm_prevention"] == self.harm_principle

    def test_add_multiple_principles(self):
        """Test adding multiple principles."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)
        framework.add_principle(self.truth_principle)
        framework.add_principle(self.fairness_principle)

        assert len(framework.principles) == 3
        assert "harm_prevention" in framework.principles
        assert "truthfulness" in framework.principles
        assert "fairness" in framework.principles

    def test_add_duplicate_principle_raises_error(self):
        """Test that adding duplicate principle raises ValueError."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)

        with pytest.raises(ValueError, match="already exists"):
            framework.add_principle(self.harm_principle)

    def test_remove_principle(self):
        """Test removing a principle."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)
        framework.add_principle(self.truth_principle)

        assert len(framework.principles) == 2

        framework.remove_principle("harm_prevention")

        assert len(framework.principles) == 1
        assert "harm_prevention" not in framework.principles
        assert "truthfulness" in framework.principles

    def test_remove_nonexistent_principle(self):
        """Test removing a principle that doesn't exist (should not raise error)."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)

        # Should not raise an error
        framework.remove_principle("nonexistent")
        assert len(framework.principles) == 1

    def test_enable_principle(self):
        """Test enabling a principle."""
        framework = ConstitutionalFramework()
        self.harm_principle.enabled = False
        framework.add_principle(self.harm_principle)

        assert framework.principles["harm_prevention"].enabled is False

        framework.enable_principle("harm_prevention")

        assert framework.principles["harm_prevention"].enabled is True

    def test_disable_principle(self):
        """Test disabling a principle."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)

        assert framework.principles["harm_prevention"].enabled is True

        framework.disable_principle("harm_prevention")

        assert framework.principles["harm_prevention"].enabled is False

    def test_enable_nonexistent_principle(self):
        """Test enabling nonexistent principle (should not raise error)."""
        framework = ConstitutionalFramework()
        framework.enable_principle("nonexistent")  # Should not crash

    def test_disable_nonexistent_principle(self):
        """Test disabling nonexistent principle (should not raise error)."""
        framework = ConstitutionalFramework()
        framework.disable_principle("nonexistent")  # Should not crash

    def test_evaluate_text_no_violations(self):
        """Test evaluating text with no violations."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)
        framework.add_principle(self.truth_principle)

        result = framework.evaluate_text("This is safe and reasonable text.")

        assert result["any_flagged"] is False
        assert len(result["flagged_principles"]) == 0
        assert result["weighted_score"] == 0.0
        assert result["num_principles_evaluated"] == 2
        assert result["text_length"] == len("This is safe and reasonable text.")
        assert "principle_results" in result

    def test_evaluate_text_with_violations(self):
        """Test evaluating text with violations."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)
        framework.add_principle(self.truth_principle)

        result = framework.evaluate_text("This will definitely hurt someone.")

        assert result["any_flagged"] is True
        assert len(result["flagged_principles"]) == 2  # Both harm and truthfulness
        assert "harm_prevention" in result["flagged_principles"]
        assert "truthfulness" in result["flagged_principles"]
        assert result["weighted_score"] == 1.8  # 1.0 + 0.8

    def test_evaluate_text_partial_violations(self):
        """Test evaluating text with only some principles violated."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)
        framework.add_principle(self.truth_principle)
        framework.add_principle(self.fairness_principle)

        result = framework.evaluate_text("I will hurt you.")

        assert result["any_flagged"] is True
        assert "harm_prevention" in result["flagged_principles"]
        assert "truthfulness" not in result["flagged_principles"]
        assert "fairness" not in result["flagged_principles"]
        assert result["weighted_score"] == 1.0  # Only harm principle

    def test_evaluate_text_with_disabled_principle(self):
        """Test that disabled principles are not evaluated."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)
        framework.add_principle(self.truth_principle)
        framework.disable_principle("harm_prevention")

        result = framework.evaluate_text("This will hurt and definitely cause harm.")

        # Only truthfulness should be evaluated
        assert result["any_flagged"] is True
        assert "truthfulness" in result["flagged_principles"]
        assert "harm_prevention" not in result["principle_results"]
        assert result["num_principles_evaluated"] == 1
        assert result["weighted_score"] == 0.8  # Only truthfulness weight

    def test_evaluate_text_track_history(self):
        """Test that evaluation history is tracked when requested."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)

        assert len(framework.evaluation_history) == 0

        framework.evaluate_text("Test text 1", track_history=True)
        assert len(framework.evaluation_history) == 1

        framework.evaluate_text("Test text 2", track_history=True)
        assert len(framework.evaluation_history) == 2

        # Default is not to track
        framework.evaluate_text("Test text 3")
        assert len(framework.evaluation_history) == 2

    def test_evaluate_text_history_truncation(self):
        """Test that long texts are truncated in history."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)

        long_text = "A" * 200
        framework.evaluate_text(long_text, track_history=True)

        stored_text = framework.evaluation_history[0]["text"]
        assert len(stored_text) <= 103  # 100 chars + "..."
        assert stored_text.endswith("...")

    def test_evaluate_text_history_no_truncation_short_text(self):
        """Test that short texts are not truncated in history."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)

        short_text = "Short"
        framework.evaluate_text(short_text, track_history=True)

        stored_text = framework.evaluation_history[0]["text"]
        assert stored_text == "Short"
        assert not stored_text.endswith("...")

    def test_batch_evaluate(self):
        """Test batch evaluation of multiple texts."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)
        framework.add_principle(self.truth_principle)

        texts = ["Safe text", "This will hurt", "Definitely safe", "All people are the same"]

        results = framework.batch_evaluate(texts)

        assert len(results) == 4
        assert results[0]["any_flagged"] is False
        assert results[1]["any_flagged"] is True  # Contains "hurt"
        assert results[2]["any_flagged"] is True  # Contains "definitely"
        assert results[3]["any_flagged"] is False

    def test_batch_evaluate_empty_list(self):
        """Test batch evaluation with empty list."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)

        results = framework.batch_evaluate([])
        assert len(results) == 0

    def test_get_statistics_empty_history(self):
        """Test statistics with no evaluation history."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)

        stats = framework.get_statistics()

        assert stats["total_evaluations"] == 0
        assert stats["total_flagged"] == 0
        assert stats["flagged_rate"] == 0.0

    def test_get_statistics_with_history(self):
        """Test statistics with evaluation history."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)
        framework.add_principle(self.truth_principle)

        # Evaluate multiple texts
        framework.evaluate_text("Safe text", track_history=True)
        framework.evaluate_text("Will hurt", track_history=True)
        framework.evaluate_text("Definitely safe", track_history=True)
        framework.evaluate_text("Another safe text", track_history=True)

        stats = framework.get_statistics()

        assert stats["total_evaluations"] == 4
        assert stats["total_flagged"] == 2  # "Will hurt" and "Definitely safe"
        assert stats["flagged_rate"] == 0.5
        assert "principle_violation_counts" in stats
        assert "principle_violation_rates" in stats

    def test_get_statistics_principle_counts(self):
        """Test that principle violation counts are tracked correctly."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)
        framework.add_principle(self.truth_principle)
        framework.add_principle(self.fairness_principle)

        # Text violating multiple principles
        framework.evaluate_text("Will definitely hurt all people", track_history=True)
        # Text violating only harm
        framework.evaluate_text("Will destroy", track_history=True)
        # Safe text
        framework.evaluate_text("Safe text", track_history=True)

        stats = framework.get_statistics()

        assert stats["principle_violation_counts"]["harm_prevention"] == 2
        assert stats["principle_violation_counts"]["truthfulness"] == 1
        assert stats["principle_violation_counts"]["fairness"] == 1
        assert stats["principle_violation_rates"]["harm_prevention"] == 2 / 3
        assert stats["principle_violation_rates"]["truthfulness"] == 1 / 3

    def test_clear_history(self):
        """Test clearing evaluation history."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)

        framework.evaluate_text("Test 1", track_history=True)
        framework.evaluate_text("Test 2", track_history=True)
        assert len(framework.evaluation_history) == 2

        framework.clear_history()
        assert len(framework.evaluation_history) == 0

    def test_get_active_principles(self):
        """Test getting list of active principles."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)
        framework.add_principle(self.truth_principle)
        framework.add_principle(self.fairness_principle)

        active = framework.get_active_principles()
        assert len(active) == 3
        assert "harm_prevention" in active
        assert "truthfulness" in active
        assert "fairness" in active

    def test_get_active_principles_with_disabled(self):
        """Test getting active principles when some are disabled."""
        framework = ConstitutionalFramework()
        framework.add_principle(self.harm_principle)
        framework.add_principle(self.truth_principle)
        framework.add_principle(self.fairness_principle)
        framework.disable_principle("truthfulness")

        active = framework.get_active_principles()
        assert len(active) == 2
        assert "harm_prevention" in active
        assert "fairness" in active
        assert "truthfulness" not in active

    def test_framework_repr(self):
        """Test string representation of framework."""
        framework = ConstitutionalFramework(name="my_framework")
        framework.add_principle(self.harm_principle)
        framework.add_principle(self.truth_principle)

        repr_str = repr(framework)
        assert "my_framework" in repr_str
        assert "2/2" in repr_str  # 2 enabled out of 2 total
        assert "enabled" in repr_str

    def test_framework_repr_with_disabled(self):
        """Test string representation with disabled principles."""
        framework = ConstitutionalFramework(name="test")
        framework.add_principle(self.harm_principle)
        framework.add_principle(self.truth_principle)
        framework.disable_principle("harm_prevention")

        repr_str = repr(framework)
        assert "1/2" in repr_str  # 1 enabled out of 2 total


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_framework_evaluation(self):
        """Test evaluating with no principles added."""
        framework = ConstitutionalFramework()

        result = framework.evaluate_text("Any text")

        assert result["any_flagged"] is False
        assert len(result["flagged_principles"]) == 0
        assert result["weighted_score"] == 0.0
        assert result["num_principles_evaluated"] == 0
        assert len(result["principle_results"]) == 0

    def test_evaluation_with_all_disabled_principles(self):
        """Test evaluation when all principles are disabled."""

        def eval_fn(text: str) -> dict[str, Any]:
            return {"flagged": True, "reason": "Test"}

        framework = ConstitutionalFramework()
        principle = ConstitutionalPrinciple("test", "desc", eval_fn, enabled=False)
        framework.add_principle(principle)

        result = framework.evaluate_text("Any text")

        assert result["any_flagged"] is False
        assert result["num_principles_evaluated"] == 0

    def test_principle_with_exception_in_evaluation(self):
        """Test handling of exceptions in evaluation function."""

        def eval_raises_exception(text: str) -> dict[str, Any]:
            raise ValueError("Evaluation error")

        principle = ConstitutionalPrinciple("error_principle", "desc", eval_raises_exception)

        framework = ConstitutionalFramework()
        framework.add_principle(principle)

        # This should raise the exception (not caught by framework)
        with pytest.raises(ValueError, match="Evaluation error"):
            framework.evaluate_text("Test")

    def test_very_long_text_evaluation(self):
        """Test evaluation with very long text."""

        def eval_fn(text: str) -> dict[str, Any]:
            return {"flagged": len(text) > 1000, "reason": "Length check"}

        framework = ConstitutionalFramework()
        principle = ConstitutionalPrinciple("length", "desc", eval_fn)
        framework.add_principle(principle)

        long_text = "A" * 10000
        result = framework.evaluate_text(long_text)

        assert result["any_flagged"] is True
        assert result["text_length"] == 10000

    def test_empty_text_evaluation(self):
        """Test evaluation with empty text."""

        def eval_fn(text: str) -> dict[str, Any]:
            return {"flagged": len(text) == 0, "reason": "Empty"}

        framework = ConstitutionalFramework()
        principle = ConstitutionalPrinciple("empty_check", "desc", eval_fn)
        framework.add_principle(principle)

        result = framework.evaluate_text("")

        assert result["text_length"] == 0
        assert result["any_flagged"] is True

    def test_unicode_text_evaluation(self):
        """Test evaluation with unicode text."""

        def eval_fn(text: str) -> dict[str, Any]:
            return {"flagged": "🚫" in text, "reason": "Emoji check"}

        framework = ConstitutionalFramework()
        principle = ConstitutionalPrinciple("unicode", "desc", eval_fn)
        framework.add_principle(principle)

        result = framework.evaluate_text("Hello 🚫 World")

        assert result["any_flagged"] is True

    def test_principle_weight_zero(self):
        """Test principle with zero weight."""

        def eval_fn(text: str) -> dict[str, Any]:
            return {"flagged": True, "reason": "Always flags"}

        framework = ConstitutionalFramework()
        principle = ConstitutionalPrinciple("zero_weight", "desc", eval_fn, weight=0.0)
        framework.add_principle(principle)

        result = framework.evaluate_text("Test")

        assert result["any_flagged"] is True
        assert result["weighted_score"] == 0.0  # Weight is zero

    def test_principle_negative_weight(self):
        """Test principle with negative weight."""

        def eval_fn(text: str) -> dict[str, Any]:
            return {"flagged": True, "reason": "Always flags"}

        framework = ConstitutionalFramework()
        principle = ConstitutionalPrinciple("negative", "desc", eval_fn, weight=-1.0)
        framework.add_principle(principle)

        result = framework.evaluate_text("Test")

        assert result["any_flagged"] is True
        assert result["weighted_score"] == -1.0  # Negative weight allowed

    def test_very_high_weight(self):
        """Test principle with very high weight."""

        def eval_fn(text: str) -> dict[str, Any]:
            return {"flagged": True, "reason": "Always flags"}

        framework = ConstitutionalFramework()
        principle = ConstitutionalPrinciple("high_weight", "desc", eval_fn, weight=1000.0)
        framework.add_principle(principle)

        result = framework.evaluate_text("Test")

        assert result["weighted_score"] == 1000.0

    def test_multiple_evaluations_same_text(self):
        """Test evaluating the same text multiple times."""

        def eval_fn(text: str) -> dict[str, Any]:
            return {"flagged": "bad" in text, "reason": "Check"}

        framework = ConstitutionalFramework()
        principle = ConstitutionalPrinciple("test", "desc", eval_fn)
        framework.add_principle(principle)

        text = "This is bad"
        result1 = framework.evaluate_text(text)
        result2 = framework.evaluate_text(text)

        # Results should be identical
        assert result1["any_flagged"] == result2["any_flagged"]
        assert result1["weighted_score"] == result2["weighted_score"]

    def test_batch_evaluate_with_mixed_results(self):
        """Test batch evaluation with mix of clean and flagged texts."""

        def eval_fn(text: str) -> dict[str, Any]:
            return {"flagged": "bad" in text.lower(), "reason": "Check"}

        framework = ConstitutionalFramework()
        principle = ConstitutionalPrinciple("test", "desc", eval_fn)
        framework.add_principle(principle)

        texts = ["Good", "bad", "Good", "BAD", "good"]
        results = framework.batch_evaluate(texts)

        flags = [r["any_flagged"] for r in results]
        assert flags == [False, True, False, True, False]

    def test_evaluation_result_contains_all_expected_keys(self):
        """Test that evaluation result contains all expected keys."""

        def eval_fn(text: str) -> dict[str, Any]:
            return {"flagged": False, "reason": "Test"}

        framework = ConstitutionalFramework()
        principle = ConstitutionalPrinciple("test", "desc", eval_fn)
        framework.add_principle(principle)

        result = framework.evaluate_text("Test")

        required_keys = [
            "principle_results",
            "any_flagged",
            "flagged_principles",
            "weighted_score",
            "num_principles_evaluated",
            "text_length",
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"
