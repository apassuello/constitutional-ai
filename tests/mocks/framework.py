"""Mocks for Constitutional AI framework components."""

from unittest.mock import Mock


def create_mock_framework(flagged=False, score=0.0):
    """
    Create a mock ConstitutionalFramework.

    Args:
        flagged: Whether text should be flagged
        score: Weighted violation score

    Returns:
        Mock framework with evaluate_text method

    Usage:
        # Safe text
        framework = create_mock_framework(flagged=False, score=0.0)

        # Harmful text
        framework = create_mock_framework(flagged=True, score=3.5)

        result = framework.evaluate_text("some text")
    """
    framework = Mock()
    framework.name = "mock_constitutional_framework"
    framework.principles = {}

    # Mock evaluate_text method
    framework.evaluate_text.return_value = {
        "flagged": flagged,
        "flagged_principles": ["harm"] if flagged else [],
        "weighted_score": score,
        "evaluation_method": "regex_heuristic",
        "any_flagged": flagged,
    }

    # Mock get_statistics
    framework.get_statistics.return_value = {
        "total_evaluations": 0,
        "flagged_count": 0,
    }

    return framework


def create_mock_evaluator(flagged=False, score=1.0):
    """
    Create a mock ConstitutionalSafetyEvaluator.

    Args:
        flagged: Whether response should be flagged
        score: Combined evaluation score

    Returns:
        Mock evaluator with evaluate method

    Usage:
        # Safe response
        evaluator = create_mock_evaluator(flagged=False, score=0.0)

        # Harmful response
        evaluator = create_mock_evaluator(flagged=True, score=2.5)

        result = evaluator.evaluate("some response")
    """
    evaluator = Mock()

    # Mock evaluate method
    evaluator.evaluate.return_value = {
        "flagged": flagged,
        "direct_evaluation": {
            "weighted_score": score,
            "flagged_principles": ["harm"] if flagged else [],
            "evaluation_method": "regex_heuristic",
        },
        "critique": "Test critique response",
        "combined_flagged": flagged,
    }

    # Mock get_statistics
    evaluator.get_statistics.return_value = {
        "total_evaluations": 0,
        "flagged_by_direct": 0,
        "flagged_by_critique": 0,
    }

    return evaluator
