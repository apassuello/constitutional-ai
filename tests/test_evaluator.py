"""
Unit tests for evaluator.py
Tests the ConstitutionalSafetyEvaluator class with two-stage evaluation:
1. Direct constitutional principle evaluation
2. AI-generated self-critique
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from constitutional_ai.evaluator import (
    ConstitutionalSafetyEvaluator,
    combine_reasoning,
    critique_indicates_issues,
)
from constitutional_ai.framework import ConstitutionalFramework
from constitutional_ai.principles import setup_default_framework


class TestConstitutionalSafetyEvaluatorInit:
    """Test ConstitutionalSafetyEvaluator initialization."""

    def test_init_with_default_framework(self):
        """Test initialization with default framework (None parameter)."""
        evaluator = ConstitutionalSafetyEvaluator(framework=None)

        assert evaluator.framework is not None
        assert isinstance(evaluator.framework, ConstitutionalFramework)
        assert evaluator.framework.name == "default_constitutional_framework"

    def test_init_with_custom_framework(self):
        """Test initialization with custom framework."""
        custom_framework = Mock(spec=ConstitutionalFramework)
        custom_framework.name = "custom"

        evaluator = ConstitutionalSafetyEvaluator(framework=custom_framework)

        assert evaluator.framework is custom_framework
        assert evaluator.framework.name == "custom"

    def test_init_without_critique_model(self):
        """Test initialization without critique model."""
        evaluator = ConstitutionalSafetyEvaluator(critique_model=None)

        assert evaluator.critique_model is None
        assert evaluator.use_self_critique is False

    def test_init_with_critique_model(self):
        """Test initialization with critique model."""
        mock_model = Mock(spec=nn.Module)

        evaluator = ConstitutionalSafetyEvaluator(
            critique_model=mock_model, use_self_critique=True
        )

        assert evaluator.critique_model is mock_model
        assert evaluator.use_self_critique is True

    def test_init_use_self_critique_logic_enabled(self):
        """Test use_self_critique enabled only if model provided."""
        mock_model = Mock(spec=nn.Module)

        evaluator = ConstitutionalSafetyEvaluator(
            critique_model=mock_model, use_self_critique=True
        )

        assert evaluator.use_self_critique is True

    def test_init_use_self_critique_logic_disabled_no_model(self):
        """Test use_self_critique disabled when no model provided."""
        evaluator = ConstitutionalSafetyEvaluator(
            critique_model=None, use_self_critique=True
        )

        # use_self_critique should be False even though requested
        assert evaluator.use_self_critique is False

    def test_init_statistics_initialization(self):
        """Test statistics dictionary is properly initialized."""
        evaluator = ConstitutionalSafetyEvaluator()

        assert "total_evaluations" in evaluator.stats
        assert "flagged_by_direct" in evaluator.stats
        assert "flagged_by_critique" in evaluator.stats
        assert "flagged_by_both" in evaluator.stats

        assert evaluator.stats["total_evaluations"] == 0
        assert evaluator.stats["flagged_by_direct"] == 0
        assert evaluator.stats["flagged_by_critique"] == 0
        assert evaluator.stats["flagged_by_both"] == 0

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters provided."""
        mock_framework = Mock(spec=ConstitutionalFramework)
        mock_model = Mock(spec=nn.Module)

        evaluator = ConstitutionalSafetyEvaluator(
            framework=mock_framework,
            critique_model=mock_model,
            use_self_critique=True,
        )

        assert evaluator.framework is mock_framework
        assert evaluator.critique_model is mock_model
        assert evaluator.use_self_critique is True


class TestEvaluateMethod:
    """Test main evaluate() method."""

    def setup_method(self):
        """Setup mock framework and evaluator."""
        self.mock_framework = Mock(spec=ConstitutionalFramework)
        self.mock_framework.get_active_principles.return_value = ["harm_prevention", "truthfulness"]

    def test_evaluate_direct_path_no_critique(self):
        """Test direct evaluation path without critique."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": False,
            "flagged_principles": [],
            "principle_results": {},
        }

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework, critique_model=None
        )

        result = evaluator.evaluate("Clean text")

        assert result["flagged"] is False
        assert result["source"] == "none"
        assert "direct_evaluation" in result
        assert "reasoning" in result
        assert "critique" not in result

    def test_evaluate_direct_flagged_no_critique(self):
        """Test direct evaluation flags without critique."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": True,
            "flagged_principles": ["harm_prevention"],
            "principle_results": {
                "harm_prevention": {"flagged": True, "reasoning": "Harmful content"}
            },
        }

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework, critique_model=None
        )

        result = evaluator.evaluate("Harmful text")

        assert result["flagged"] is True
        assert result["source"] == "direct"
        assert "critique" not in result

    def test_evaluate_with_critique_enabled(self):
        """Test evaluation with critique enabled."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": False,
            "flagged_principles": [],
            "principle_results": {},
        }

        mock_model = Mock(spec=nn.Module)
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework,
            critique_model=mock_model,
            use_self_critique=True,
        )

        with patch.object(evaluator, "_generate_critique") as mock_gen_critique:
            mock_gen_critique.return_value = {
                "text": "No issues found",
                "flagged": False,
                "prompt": "test prompt",
            }

            result = evaluator.evaluate("Test text")

            assert "critique" in result
            mock_gen_critique.assert_called_once()

    def test_evaluate_statistics_tracking_flagged_by_direct(self):
        """Test statistics tracking for direct-only flagging."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": True,
            "flagged_principles": ["harm_prevention"],
            "principle_results": {
                "harm_prevention": {"flagged": True, "reasoning": "Harmful content detected"}
            },
        }

        mock_model = Mock(spec=nn.Module)
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework,
            critique_model=mock_model,
            use_self_critique=True,
        )

        with patch.object(evaluator, "_generate_critique") as mock_gen_critique:
            mock_gen_critique.return_value = {
                "text": "No issues",
                "flagged": False,
                "prompt": "test",
            }

            evaluator.evaluate("Test text")

            assert evaluator.stats["flagged_by_direct"] == 1
            assert evaluator.stats["flagged_by_critique"] == 0
            assert evaluator.stats["flagged_by_both"] == 0

    def test_evaluate_statistics_tracking_flagged_by_critique(self):
        """Test statistics tracking for critique-only flagging."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": False,
            "flagged_principles": [],
            "principle_results": {},
        }

        mock_model = Mock(spec=nn.Module)
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework,
            critique_model=mock_model,
            use_self_critique=True,
        )

        with patch.object(evaluator, "_generate_critique") as mock_gen_critique:
            mock_gen_critique.return_value = {
                "text": "Issues found: harmful, problematic",
                "flagged": True,
                "prompt": "test",
            }

            evaluator.evaluate("Test text")

            assert evaluator.stats["flagged_by_direct"] == 0
            assert evaluator.stats["flagged_by_critique"] == 1
            assert evaluator.stats["flagged_by_both"] == 0

    def test_evaluate_statistics_tracking_flagged_by_both(self):
        """Test statistics tracking when both direct and critique flag."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": True,
            "flagged_principles": ["harm_prevention"],
            "principle_results": {
                "harm_prevention": {"flagged": True, "reasoning": "Harmful content detected"}
            },
        }

        mock_model = Mock(spec=nn.Module)
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework,
            critique_model=mock_model,
            use_self_critique=True,
        )

        with patch.object(evaluator, "_generate_critique") as mock_gen_critique:
            mock_gen_critique.return_value = {
                "text": "Issues: harmful, problematic",
                "flagged": True,
                "prompt": "test",
            }

            evaluator.evaluate("Test text")

            assert evaluator.stats["flagged_by_direct"] == 0
            assert evaluator.stats["flagged_by_critique"] == 0
            assert evaluator.stats["flagged_by_both"] == 1

    def test_evaluate_source_attribution_direct(self):
        """Test source attribution for direct flagging."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": True,
            "flagged_principles": ["harm_prevention"],
            "principle_results": {
                "harm_prevention": {"flagged": True, "reasoning": "Harmful content"}
            },
        }

        evaluator = ConstitutionalSafetyEvaluator(framework=self.mock_framework)

        result = evaluator.evaluate("Test text")

        assert result["source"] == "direct"

    def test_evaluate_source_attribution_critique(self):
        """Test source attribution for critique-only flagging."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": False,
            "flagged_principles": [],
            "principle_results": {},
        }

        mock_model = Mock(spec=nn.Module)
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework,
            critique_model=mock_model,
            use_self_critique=True,
        )

        with patch.object(evaluator, "_generate_critique") as mock_gen_critique:
            mock_gen_critique.return_value = {
                "text": "Issues: problematic, harmful",
                "flagged": True,
                "prompt": "test",
            }

            result = evaluator.evaluate("Test text")

            assert result["source"] == "critique"

    def test_evaluate_source_attribution_both(self):
        """Test source attribution when both flag."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": True,
            "flagged_principles": ["harm_prevention"],
            "principle_results": {
                "harm_prevention": {"flagged": True, "reasoning": "Harmful content"}
            },
        }

        mock_model = Mock(spec=nn.Module)
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework,
            critique_model=mock_model,
            use_self_critique=True,
        )

        with patch.object(evaluator, "_generate_critique") as mock_gen_critique:
            mock_gen_critique.return_value = {
                "text": "Issues: harmful, problematic",
                "flagged": True,
                "prompt": "test",
            }

            result = evaluator.evaluate("Test text")

            assert result["source"] == "both"

    def test_evaluate_source_attribution_none(self):
        """Test source attribution when nothing flags."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": False,
            "flagged_principles": [],
            "principle_results": {},
        }

        evaluator = ConstitutionalSafetyEvaluator(framework=self.mock_framework)

        result = evaluator.evaluate("Clean text")

        assert result["source"] == "none"

    def test_evaluate_include_critique_parameter_override_true(self):
        """Test include_critique parameter override to True."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": False,
            "flagged_principles": [],
            "principle_results": {},
        }

        mock_model = Mock(spec=nn.Module)
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework,
            critique_model=mock_model,
            use_self_critique=False,  # Disabled by default
        )

        with patch.object(evaluator, "_generate_critique") as mock_gen_critique:
            mock_gen_critique.return_value = {
                "text": "No issues",
                "flagged": False,
                "prompt": "test",
            }

            result = evaluator.evaluate("Test text", include_critique=True)

            # Should use critique despite use_self_critique=False
            assert "critique" in result
            mock_gen_critique.assert_called_once()

    def test_evaluate_include_critique_parameter_override_false(self):
        """Test include_critique parameter override to False."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": False,
            "flagged_principles": [],
            "principle_results": {},
        }

        mock_model = Mock(spec=nn.Module)
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework,
            critique_model=mock_model,
            use_self_critique=True,  # Enabled by default
        )

        with patch.object(evaluator, "_generate_critique") as mock_gen_critique:
            result = evaluator.evaluate("Test text", include_critique=False)

            # Should NOT use critique despite use_self_critique=True
            assert "critique" not in result
            mock_gen_critique.assert_not_called()

    def test_evaluate_increments_total_evaluations(self):
        """Test that total_evaluations counter increments."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": False,
            "flagged_principles": [],
            "principle_results": {},
        }

        evaluator = ConstitutionalSafetyEvaluator(framework=self.mock_framework)

        assert evaluator.stats["total_evaluations"] == 0

        evaluator.evaluate("Test 1")
        assert evaluator.stats["total_evaluations"] == 1

        evaluator.evaluate("Test 2")
        assert evaluator.stats["total_evaluations"] == 2

    def test_evaluate_calls_synthesize_reasoning(self):
        """Test that evaluate calls _synthesize_reasoning."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": False,
            "flagged_principles": [],
            "principle_results": {},
        }

        evaluator = ConstitutionalSafetyEvaluator(framework=self.mock_framework)

        with patch.object(evaluator, "_synthesize_reasoning") as mock_synthesize:
            mock_synthesize.return_value = "Test reasoning"

            result = evaluator.evaluate("Test text")

            mock_synthesize.assert_called_once()
            assert result["reasoning"] == "Test reasoning"


class TestEvaluateWithSelfCritique:
    """Test evaluate_with_self_critique() alias method."""

    def test_evaluate_with_self_critique_calls_evaluate(self):
        """Test that method calls evaluate() with include_critique=True."""
        mock_framework = Mock(spec=ConstitutionalFramework)
        mock_framework.evaluate_text.return_value = {
            "any_flagged": False,
            "flagged_principles": [],
            "principle_results": {},
        }

        evaluator = ConstitutionalSafetyEvaluator(framework=mock_framework)

        with patch.object(evaluator, "evaluate") as mock_evaluate:
            mock_evaluate.return_value = {"flagged": False}

            evaluator.evaluate_with_self_critique("Test text")

            mock_evaluate.assert_called_once_with("Test text", include_critique=True)


class TestGenerateImprovedResponse:
    """Test generate_improved_response() method."""

    def setup_method(self):
        """Setup mock framework and evaluator."""
        self.mock_framework = Mock(spec=ConstitutionalFramework)

    def test_generate_improved_response_early_exit_when_checks_pass(self):
        """Test early exit when checks pass on first iteration."""
        self.mock_framework.evaluate_text.return_value = {
            "any_flagged": False,
            "flagged_principles": [],
            "principle_results": {},
        }

        evaluator = ConstitutionalSafetyEvaluator(framework=self.mock_framework)

        with patch.object(evaluator, "evaluate") as mock_eval:
            mock_eval.return_value = {"flagged": False}

            response, evaluation = evaluator.generate_improved_response(
                prompt="Test prompt",
                initial_response="Good response",
                max_iterations=3,
            )

            assert response == "Good response"
            assert evaluation["flagged"] is False
            mock_eval.assert_called_once()

    def test_generate_improved_response_max_iterations_limit(self):
        """Test max iterations limit (iteration >= max_iterations)."""
        mock_model = Mock(spec=nn.Module)
        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework, critique_model=mock_model
        )

        with patch.object(evaluator, "evaluate") as mock_eval:
            # Always flag as problematic
            mock_eval.return_value = {
                "flagged": True,
                "reasoning": "Issues found",
            }

            with patch.object(evaluator, "_create_improvement_prompt") as mock_prompt:
                mock_prompt.return_value = "Improve this"

                with patch.object(evaluator, "_generate_improvement") as mock_improve:
                    # Return different responses each time to avoid early exit
                    mock_improve.side_effect = [
                        "Improved 1",
                        "Improved 2",
                        "Improved 3",
                    ]

                    response, evaluation = evaluator.generate_improved_response(
                        prompt="Test",
                        initial_response="Bad response",
                        max_iterations=3,
                    )

                    # Should stop after 3 iterations: initial eval + 3 improvement evals + final eval
                    assert mock_eval.call_count == 4
                    assert evaluation["flagged"] is True

    def test_generate_improved_response_no_critique_model_cannot_improve(self):
        """Test behavior without critique model (cannot improve)."""
        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework, critique_model=None
        )

        with patch.object(evaluator, "evaluate") as mock_eval:
            mock_eval.return_value = {"flagged": True, "reasoning": "Issues"}

            response, evaluation = evaluator.generate_improved_response(
                prompt="Test", initial_response="Bad response"
            )

            # Should return original response since no model to improve
            assert response == "Bad response"
            assert evaluation["flagged"] is True

    def test_generate_improved_response_improvement_prompt_generation(self):
        """Test improvement prompt generation."""
        mock_model = Mock(spec=nn.Module)

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework, critique_model=mock_model
        )

        with patch.object(evaluator, "evaluate") as mock_eval:
            mock_eval.side_effect = [
                {"flagged": True, "reasoning": "Issues"},
                {"flagged": False},
            ]

            with patch.object(
                evaluator, "_create_improvement_prompt"
            ) as mock_create_prompt:
                mock_create_prompt.return_value = "Improve this"

                with patch.object(evaluator, "_generate_improvement") as mock_improve:
                    mock_improve.return_value = "Improved response"

                    evaluator.generate_improved_response(
                        prompt="Test", initial_response="Bad response"
                    )

                    mock_create_prompt.assert_called_once()

    def test_generate_improved_response_no_improvement_scenario(self):
        """Test no improvement scenario (same response returned)."""
        mock_model = Mock(spec=nn.Module)

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework, critique_model=mock_model
        )

        with patch.object(evaluator, "evaluate") as mock_eval:
            mock_eval.return_value = {"flagged": True, "reasoning": "Issues"}

            with patch.object(evaluator, "_generate_improvement") as mock_improve:
                # Return same response (no improvement)
                mock_improve.return_value = "Bad response"

                response, evaluation = evaluator.generate_improved_response(
                    prompt="Test", initial_response="Bad response"
                )

                # Should exit early when no improvement
                assert response == "Bad response"

    def test_generate_improved_response_successful_improvement(self):
        """Test successful iterative improvement."""
        mock_model = Mock(spec=nn.Module)

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework, critique_model=mock_model
        )

        with patch.object(evaluator, "evaluate") as mock_eval:
            mock_eval.side_effect = [
                {"flagged": True, "reasoning": "Issues"},
                {"flagged": False, "reasoning": "Clean"},
            ]

            with patch.object(evaluator, "_generate_improvement") as mock_improve:
                mock_improve.return_value = "Improved response"

                response, evaluation = evaluator.generate_improved_response(
                    prompt="Test", initial_response="Bad response"
                )

                assert response == "Improved response"
                assert evaluation["flagged"] is False


class TestCritiqueGeneration:
    """Test critique generation methods."""

    def setup_method(self):
        """Setup mock framework and evaluator."""
        self.mock_framework = Mock(spec=ConstitutionalFramework)
        self.mock_framework.get_active_principles.return_value = ["harm_prevention"]

    def test_generate_critique_with_mock_model(self):
        """Test _generate_critique() with mock model."""
        mock_model = Mock(spec=nn.Module)
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(
            framework=self.mock_framework, critique_model=mock_model
        )

        with patch.object(evaluator, "_generate_with_model") as mock_gen:
            # Return critique with 2+ concern phrases to flag
            mock_gen.return_value = "This text violates principles and is problematic"

            direct_eval = {"any_flagged": False, "flagged_principles": []}

            result = evaluator._generate_critique("Test text", direct_eval)

            assert "text" in result
            assert "flagged" in result
            assert "prompt" in result
            assert result["flagged"] is True  # Should detect issues (2 concern phrases)
            mock_gen.assert_called_once()

    def test_create_critique_prompt_formatting(self):
        """Test _create_critique_prompt() formatting."""
        self.mock_framework.get_active_principles.return_value = [
            "harm_prevention",
            "truthfulness",
        ]

        evaluator = ConstitutionalSafetyEvaluator(framework=self.mock_framework)

        direct_eval = {
            "any_flagged": True,
            "flagged_principles": ["harm_prevention"],
        }

        prompt = evaluator._create_critique_prompt("Test text", direct_eval)

        assert "Test text" in prompt
        assert "harm_prevention" in prompt
        assert "truthfulness" in prompt
        assert "Direct evaluation found:" in prompt

    def test_critique_indicates_issues_zero_concerns(self):
        """Test critique_indicates_issues() with 0 concern phrases."""
        critique = "This text appears to be completely safe and appropriate."

        flagged = critique_indicates_issues(critique)

        assert flagged is False

    def test_critique_indicates_issues_one_concern(self):
        """Test critique_indicates_issues() with 1 concern phrase."""
        critique = "This might be problematic in some contexts."

        flagged = critique_indicates_issues(critique)

        assert flagged is False  # Needs >= 2 concerns

    def test_critique_indicates_issues_two_concerns(self):
        """Test critique_indicates_issues() with 2 concern phrases."""
        critique = "This is problematic and violates the principle."

        flagged = critique_indicates_issues(critique)

        assert flagged is True

    def test_critique_indicates_issues_five_plus_concerns(self):
        """Test critique_indicates_issues() with 5+ concern phrases."""
        critique = "This violates principles, is harmful, biased, misleading, and problematic."

        flagged = critique_indicates_issues(critique)

        assert flagged is True

    def test_critique_indicates_issues_case_insensitive(self):
        """Test critique_indicates_issues() is case-insensitive."""
        critique = "This VIOLATES principles and is PROBLEMATIC."

        flagged = critique_indicates_issues(critique)

        assert flagged is True


class TestModelGeneration:
    """Test model text generation methods."""

    def test_generate_with_model_valid_model_tokenizer(self):
        """Test _generate_with_model() with valid model/tokenizer."""
        mock_model = Mock(spec=nn.Module)
        mock_tokenizer = Mock()
        mock_model.tokenizer = mock_tokenizer

        evaluator = ConstitutionalSafetyEvaluator(critique_model=mock_model)

        with patch("constitutional_ai.model_utils.generate_text") as mock_gen:
            mock_gen.return_value = "Generated text"

            result = evaluator._generate_with_model("Test prompt")

            assert result == "Generated text"
            mock_gen.assert_called_once()

    def test_generate_with_model_missing_tokenizer(self):
        """Test error handling: missing tokenizer."""
        mock_model = Mock(spec=nn.Module)
        # No tokenizer attribute

        evaluator = ConstitutionalSafetyEvaluator(critique_model=mock_model)

        result = evaluator._generate_with_model("Test prompt")

        assert "Model requires tokenizer" in result

    def test_generate_with_model_import_error(self):
        """Test error handling: ImportError."""
        mock_model = Mock(spec=nn.Module)
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(critique_model=mock_model)

        with patch(
            "constitutional_ai.model_utils.generate_text",
            side_effect=ImportError("No module"),
        ):
            result = evaluator._generate_with_model("Test prompt")

            assert "transformers library required" in result

    def test_generate_with_model_general_exception(self):
        """Test error handling: general Exception."""
        mock_model = Mock(spec=nn.Module)
        mock_model.tokenizer = Mock()

        evaluator = ConstitutionalSafetyEvaluator(critique_model=mock_model)

        with patch(
            "constitutional_ai.model_utils.generate_text",
            side_effect=Exception("Generation failed"),
        ):
            result = evaluator._generate_with_model("Test prompt")

            assert "Generation error" in result
            assert "Generation failed" in result

    def test_generate_with_model_none_model_returns_empty(self):
        """Test with None model (returns empty string)."""
        evaluator = ConstitutionalSafetyEvaluator(critique_model=None)

        result = evaluator._generate_with_model("Test prompt")

        assert result == ""

    def test_generate_improvement_calls_generate_with_model(self):
        """Test _generate_improvement() calls _generate_with_model()."""
        mock_model = Mock(spec=nn.Module)

        evaluator = ConstitutionalSafetyEvaluator(critique_model=mock_model)

        with patch.object(evaluator, "_generate_with_model") as mock_gen:
            mock_gen.return_value = "Improved text"

            result = evaluator._generate_improvement("Test prompt")

            assert result == "Improved text"
            mock_gen.assert_called_once_with("Test prompt")

    def test_generate_improvement_no_model_returns_empty(self):
        """Test _generate_improvement() with no model returns empty."""
        evaluator = ConstitutionalSafetyEvaluator(critique_model=None)

        result = evaluator._generate_improvement("Test prompt")

        assert result == ""


class TestReasoningSynthesis:
    """Test reasoning synthesis method."""

    def setup_method(self):
        """Setup evaluator."""
        self.evaluator = ConstitutionalSafetyEvaluator()

    def test_synthesize_reasoning_flagged_direct_evaluation(self):
        """Test _synthesize_reasoning() with flagged direct evaluation."""
        direct_eval = {
            "any_flagged": True,
            "flagged_principles": ["harm_prevention"],
            "principle_results": {
                "harm_prevention": {
                    "flagged": True,
                    "reasoning": "Contains harmful instructions",
                }
            },
        }

        reasoning = self.evaluator._synthesize_reasoning(direct_eval)

        assert "Direct evaluation issues:" in reasoning
        assert "harm_prevention" in reasoning
        assert "Contains harmful instructions" in reasoning

    def test_synthesize_reasoning_unflagged_direct_evaluation(self):
        """Test _synthesize_reasoning() with unflagged direct evaluation."""
        direct_eval = {"any_flagged": False, "flagged_principles": [], "principle_results": {}}

        reasoning = self.evaluator._synthesize_reasoning(direct_eval)

        assert "Direct evaluation: No issues detected" in reasoning

    def test_synthesize_reasoning_with_critique_flagged(self):
        """Test _synthesize_reasoning() with flagged critique."""
        direct_eval = {"any_flagged": False, "flagged_principles": [], "principle_results": {}}

        critique = {
            "flagged": True,
            "text": "This text contains subtle manipulation tactics that could be problematic.",
        }

        reasoning = self.evaluator._synthesize_reasoning(direct_eval, critique)

        assert "Model critique:" in reasoning
        assert "subtle manipulation" in reasoning

    def test_synthesize_reasoning_without_critique(self):
        """Test _synthesize_reasoning() without critique."""
        direct_eval = {"any_flagged": False, "flagged_principles": [], "principle_results": {}}

        reasoning = self.evaluator._synthesize_reasoning(direct_eval, None)

        assert "Model critique:" not in reasoning

    def test_synthesize_reasoning_text_truncation(self):
        """Test text truncation (>150 chars)."""
        direct_eval = {"any_flagged": False, "flagged_principles": [], "principle_results": {}}

        long_text = "A" * 200
        critique = {"flagged": True, "text": long_text}

        reasoning = self.evaluator._synthesize_reasoning(direct_eval, critique)

        # Should truncate to 150 chars + "..."
        assert "..." in reasoning
        assert len(reasoning.split("Model critique:")[1].strip()) <= 154  # 150 + "..."

    def test_synthesize_reasoning_multiple_flagged_principles(self):
        """Test reasoning synthesis with multiple flagged principles."""
        direct_eval = {
            "any_flagged": True,
            "flagged_principles": ["harm_prevention", "truthfulness"],
            "principle_results": {
                "harm_prevention": {"flagged": True, "reasoning": "Harmful content"},
                "truthfulness": {"flagged": True, "reasoning": "Misleading claims"},
            },
        }

        reasoning = self.evaluator._synthesize_reasoning(direct_eval)

        assert "harm_prevention" in reasoning
        assert "truthfulness" in reasoning
        assert "Harmful content" in reasoning
        assert "Misleading claims" in reasoning


class TestStatistics:
    """Test statistics methods."""

    def setup_method(self):
        """Setup mock framework."""
        self.mock_framework = Mock(spec=ConstitutionalFramework)
        self.mock_framework.get_statistics.return_value = {
            "total_evaluations": 10,
            "total_flagged": 3,
        }

    def test_get_statistics_returns_correct_structure(self):
        """Test get_statistics() returns correct structure."""
        evaluator = ConstitutionalSafetyEvaluator(framework=self.mock_framework)

        # Manually set some stats
        evaluator.stats["total_evaluations"] = 5
        evaluator.stats["flagged_by_direct"] = 2
        evaluator.stats["flagged_by_critique"] = 1
        evaluator.stats["flagged_by_both"] = 1

        stats = evaluator.get_statistics()

        assert stats["total_evaluations"] == 5
        assert stats["flagged_by_direct"] == 2
        assert stats["flagged_by_critique"] == 1
        assert stats["flagged_by_both"] == 1
        assert "framework_stats" in stats
        assert stats["framework_stats"]["total_evaluations"] == 10

    def test_reset_statistics_clears_counters(self):
        """Test reset_statistics() clears counters."""
        evaluator = ConstitutionalSafetyEvaluator(framework=self.mock_framework)

        # Set some stats
        evaluator.stats["total_evaluations"] = 10
        evaluator.stats["flagged_by_direct"] = 5
        evaluator.stats["flagged_by_critique"] = 3
        evaluator.stats["flagged_by_both"] = 2

        evaluator.reset_statistics()

        assert evaluator.stats["total_evaluations"] == 0
        assert evaluator.stats["flagged_by_direct"] == 0
        assert evaluator.stats["flagged_by_critique"] == 0
        assert evaluator.stats["flagged_by_both"] == 0
        self.mock_framework.clear_history.assert_called_once()


class TestCritiqueIndicatesIssuesFunction:
    """Test standalone critique_indicates_issues() function."""

    def test_no_concern_phrases(self):
        """Test with no concern phrases."""
        critique = "This text is completely safe and appropriate."

        assert critique_indicates_issues(critique) is False

    def test_one_concern_phrase(self):
        """Test with one concern phrase."""
        critique = "This seems problematic."

        assert critique_indicates_issues(critique) is False

    def test_two_concern_phrases(self):
        """Test with two concern phrases (threshold)."""
        critique = "This violates the principle and is problematic."

        assert critique_indicates_issues(critique) is True

    def test_multiple_concern_phrases(self):
        """Test with multiple concern phrases."""
        critique = (
            "This is harmful, biased, misleading, and violates multiple principles."
        )

        assert critique_indicates_issues(critique) is True

    def test_case_insensitivity(self):
        """Test case-insensitive detection."""
        critique = "This VIOLATES principles and is PROBLEMATIC."

        assert critique_indicates_issues(critique) is True

    def test_all_concern_phrases(self):
        """Test that all concern phrases are detected."""
        concern_phrases = [
            "violates",
            "violation",
            "problematic",
            "issue",
            "concern",
            "inappropriate",
            "harmful",
            "bias",
            "biased",
            "misleading",
            "deceptive",
            "unfair",
            "coercive",
            "manipulative",
            "dangerous",
        ]

        # Test with each pair
        for i in range(0, len(concern_phrases), 2):
            if i + 1 < len(concern_phrases):
                critique = f"This is {concern_phrases[i]} and {concern_phrases[i+1]}."
                assert (
                    critique_indicates_issues(critique) is True
                ), f"Failed for: {concern_phrases[i]}, {concern_phrases[i+1]}"


class TestCombineReasoningFunction:
    """Test legacy combine_reasoning() function."""

    def test_combine_reasoning_with_flagged_principles(self):
        """Test combining reasoning with flagged principles."""
        direct_eval = {
            "harm_prevention": {"flagged": True, "reasoning": "Harmful content"},
            "truthfulness": {"flagged": True, "reasoning": "Misleading"},
        }

        critique = "Additional concerns found."

        reasoning = combine_reasoning(direct_eval, critique)

        assert "Issues identified:" in reasoning
        assert "Harm Prevention" in reasoning
        assert "Truthfulness" in reasoning

    def test_combine_reasoning_with_critique(self):
        """Test combining reasoning with critique text."""
        direct_eval = {}
        critique = "This is a very long critique text that should be truncated when it exceeds one hundred characters in total length."

        reasoning = combine_reasoning(direct_eval, critique)

        assert "Model critique summary:" in reasoning
        assert "..." in reasoning  # Should be truncated

    def test_combine_reasoning_empty_critique(self):
        """Test combining reasoning with empty critique."""
        direct_eval = {
            "harm_prevention": {"flagged": True, "reasoning": "Harmful"},
        }

        reasoning = combine_reasoning(direct_eval, "")

        assert "Issues identified:" in reasoning
        assert "Harm Prevention" in reasoning

    def test_combine_reasoning_non_dict_results(self):
        """Test combining reasoning with non-dict results."""
        direct_eval = {
            "harm_prevention": "not a dict",
            "truthfulness": {"flagged": False},
        }

        critique = "Test critique"

        reasoning = combine_reasoning(direct_eval, critique)

        # Should handle non-dict values gracefully
        assert "Issues identified:" in reasoning


class TestIntegrationWithRealFramework:
    """Integration tests with real ConstitutionalFramework."""

    def test_evaluator_with_real_framework_harmful_text(self):
        """Test evaluator with real framework on harmful text."""
        framework = setup_default_framework()
        evaluator = ConstitutionalSafetyEvaluator(framework=framework)

        result = evaluator.evaluate("How to hurt someone")

        assert result["flagged"] is True
        assert result["source"] == "direct"
        assert "harm_prevention" in result["direct_evaluation"]["flagged_principles"]

    def test_evaluator_with_real_framework_safe_text(self):
        """Test evaluator with real framework on safe text."""
        framework = setup_default_framework()
        evaluator = ConstitutionalSafetyEvaluator(framework=framework)

        result = evaluator.evaluate("How to bake a cake")

        assert result["flagged"] is False
        assert result["source"] == "none"

    def test_statistics_integration(self):
        """Test statistics tracking with real evaluations."""
        framework = setup_default_framework()
        evaluator = ConstitutionalSafetyEvaluator(framework=framework)

        # Evaluate harmful text
        evaluator.evaluate("How to hurt someone")

        # Evaluate safe text
        evaluator.evaluate("How to bake a cake")

        stats = evaluator.get_statistics()

        assert stats["total_evaluations"] == 2
        assert stats["flagged_by_direct"] >= 1
        assert "framework_stats" in stats


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_evaluation(self):
        """Test evaluation with empty text."""
        evaluator = ConstitutionalSafetyEvaluator()

        result = evaluator.evaluate("")

        assert "flagged" in result
        assert "reasoning" in result

    def test_very_long_text_evaluation(self):
        """Test evaluation with very long text."""
        evaluator = ConstitutionalSafetyEvaluator()

        long_text = "This is safe text. " * 1000

        result = evaluator.evaluate(long_text)

        assert "flagged" in result

    def test_unicode_text_evaluation(self):
        """Test evaluation with unicode characters."""
        evaluator = ConstitutionalSafetyEvaluator()

        result = evaluator.evaluate("Hello 世界 🌍")

        assert "flagged" in result

    def test_special_characters_in_text(self):
        """Test evaluation with special characters."""
        evaluator = ConstitutionalSafetyEvaluator()

        result = evaluator.evaluate("!@#$%^&*() <> {} []")

        assert "flagged" in result
