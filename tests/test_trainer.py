"""
Unit tests for trainer.py
Tests the RLAIFTrainer class focusing on algorithmic logic:
- Score computation algorithms
- Response selection logic
- Training data generation
- Statistics tracking
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from constitutional_ai.trainer import RLAIFTrainer
from tests.mocks.framework import create_mock_framework


class TestComputeCombinedScore:
    """Test _compute_combined_score() algorithmic logic."""

    def test_compute_combined_score_basic_formula(self):
        """Test basic formula: constitutional_score + (critique_score * 0.5)."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluation = {"direct_evaluation": {"weighted_score": 2.0}}
        critique = "unsafe harmful"  # Should score 2.0

        score = trainer._compute_combined_score(evaluation, critique)

        # Expected: 2.0 + (2.0 * 0.5) = 3.0
        assert score == 3.0

    def test_compute_combined_score_zero_scores(self):
        """Test with zero scores."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluation = {"direct_evaluation": {"weighted_score": 0.0}}
        critique = "clean safe appropriate"  # Should score 0.0

        score = trainer._compute_combined_score(evaluation, critique)

        # Expected: 0.0 + (0.0 * 0.5) = 0.0
        assert score == 0.0

    def test_compute_combined_score_large_critique_score(self):
        """Test with large critique score."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluation = {"direct_evaluation": {"weighted_score": 3.0}}
        # 15 negative terms (capped at 10.0)
        critique = "unsafe harmful biased incorrect misleading deceptive inappropriate problematic concerning violation issue unfair coercive manipulative dangerous"

        score = trainer._compute_combined_score(evaluation, critique)

        # Expected: 3.0 + (10.0 * 0.5) = 8.0
        assert score == 8.0

    def test_compute_combined_score_only_constitutional(self):
        """Test with only constitutional score (no critique issues)."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluation = {"direct_evaluation": {"weighted_score": 5.0}}
        critique = "Everything looks good"  # Should score 0.0 (no negative terms)

        score = trainer._compute_combined_score(evaluation, critique)

        # Expected: 5.0 + (0.0 * 0.5) = 5.0
        assert score == 5.0

    def test_compute_combined_score_only_critique(self):
        """Test with only critique score (no constitutional issues)."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluation = {"direct_evaluation": {"weighted_score": 0.0}}
        critique = "unsafe harmful biased incorrect"  # Should score 4.0

        score = trainer._compute_combined_score(evaluation, critique)

        # Expected: 0.0 + (4.0 * 0.5) = 2.0
        assert score == 2.0

    def test_compute_combined_score_missing_direct_evaluation(self):
        """Test with missing direct_evaluation key."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluation = {}  # Missing direct_evaluation
        critique = "unsafe harmful"  # Should score 2.0

        score = trainer._compute_combined_score(evaluation, critique)

        # Expected: 0.0 + (2.0 * 0.5) = 1.0
        assert score == 1.0

    def test_compute_combined_score_missing_weighted_score(self):
        """Test with missing weighted_score key."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluation = {"direct_evaluation": {}}  # Missing weighted_score
        critique = "unsafe harmful"  # Should score 2.0

        score = trainer._compute_combined_score(evaluation, critique)

        # Expected: 0.0 + (2.0 * 0.5) = 1.0
        assert score == 1.0

    def test_compute_combined_score_fractional_values(self):
        """Test with fractional values."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluation = {"direct_evaluation": {"weighted_score": 1.5}}
        critique = "unsafe"  # Should score 1.0

        score = trainer._compute_combined_score(evaluation, critique)

        # Expected: 1.5 + (1.0 * 0.5) = 2.0
        assert score == 2.0

    def test_compute_combined_score_maximum_possible(self):
        """Test maximum possible combined score."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluation = {"direct_evaluation": {"weighted_score": 10.0}}
        # All 14 unique negative terms
        critique = "unsafe harmful biased incorrect misleading deceptive inappropriate problematic concerning violation issue unfair coercive manipulative"

        score = trainer._compute_combined_score(evaluation, critique)

        # Expected: 10.0 + (min(14, 10.0) * 0.5) = 10.0 + (10.0 * 0.5) = 15.0
        assert score == 15.0


class TestExtractScoreFromCritique:
    """Test _extract_score_from_critique() algorithmic logic."""

    def test_extract_score_zero_negative_terms(self):
        """Test with 0 negative terms."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        critique = "This response is clean, appropriate, and helpful."

        score = trainer._extract_score_from_critique(critique)

        assert score == 0.0

    def test_extract_score_one_negative_term(self):
        """Test with 1 negative term."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        critique = "This response is unsafe."

        score = trainer._extract_score_from_critique(critique)

        assert score == 1.0

    def test_extract_score_five_negative_terms(self):
        """Test with 5 negative terms."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        critique = "This is unsafe, harmful, biased, incorrect, and misleading."

        score = trainer._extract_score_from_critique(critique)

        assert score == 5.0

    def test_extract_score_ten_negative_terms(self):
        """Test with 10 negative terms (at cap)."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        critique = "unsafe harmful biased incorrect misleading deceptive inappropriate problematic concerning violation"

        score = trainer._extract_score_from_critique(critique)

        assert score == 10.0

    def test_extract_score_fifteen_negative_terms_capped(self):
        """Test with 15 negative terms (should cap at 10)."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        critique = "unsafe harmful biased incorrect misleading deceptive inappropriate problematic concerning violation issue unfair coercive manipulative extra"

        score = trainer._extract_score_from_critique(critique)

        # Should cap at 10.0
        assert score == 10.0

    def test_extract_score_case_insensitive(self):
        """Test case-insensitive matching."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        critique = "This is UNSAFE, Harmful, BIASED."

        score = trainer._extract_score_from_critique(critique)

        assert score == 3.0

    def test_extract_score_multiple_occurrences_same_term(self):
        """Test that only unique terms count (not multiple occurrences)."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        critique = "unsafe unsafe unsafe"

        score = trainer._extract_score_from_critique(critique)

        # Only counts unique terms (1 unique term = 1.0)
        assert score == 1.0

    def test_extract_score_all_14_negative_terms(self):
        """Test with all 14 negative terms listed in code."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        # All 14 terms from the code
        critique = "unsafe harmful biased incorrect misleading deceptive inappropriate problematic concerning violation issue unfair coercive manipulative"

        score = trainer._extract_score_from_critique(critique)

        # Should cap at 10.0
        assert score == 10.0

    def test_extract_score_partial_word_matches(self):
        """Test that partial word matches count (e.g., 'unsafe' in 'unsafely')."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        critique = "This is unsafely done and harmfully executed."

        score = trainer._extract_score_from_critique(critique)

        # 'unsafe' in 'unsafely' and 'harmful' in 'harmfully' should both count
        assert score == 2.0

    def test_extract_score_empty_string(self):
        """Test with empty string."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        critique = ""

        score = trainer._extract_score_from_critique(critique)

        assert score == 0.0

    def test_extract_score_special_characters(self):
        """Test with special characters around negative terms."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        critique = "(unsafe) [harmful] {biased}"

        score = trainer._extract_score_from_critique(critique)

        assert score == 3.0


class TestSelectBestResponse:
    """Test _select_best_response() algorithmic logic."""

    def test_select_best_response_simple_case(self):
        """Test simple case with clear best response."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluations = [
            {"combined_score": 3.0},
            {"combined_score": 1.0},
            {"combined_score": 2.0},
        ]

        best_idx = trainer._select_best_response(evaluations)

        # Should return index 1 (lowest score 1.0)
        assert best_idx == 1

    def test_select_best_response_first_on_tie(self):
        """Test that first index is returned on tie."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluations = [
            {"combined_score": 1.0},
            {"combined_score": 1.0},
            {"combined_score": 1.0},
        ]

        best_idx = trainer._select_best_response(evaluations)

        # Should return index 0 (first on tie)
        assert best_idx == 0

    def test_select_best_response_single_element(self):
        """Test with single element."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluations = [{"combined_score": 5.0}]

        best_idx = trainer._select_best_response(evaluations)

        # Should return index 0
        assert best_idx == 0

    def test_select_best_response_zero_score(self):
        """Test with zero score (perfect response)."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluations = [
            {"combined_score": 3.0},
            {"combined_score": 0.0},
            {"combined_score": 2.0},
        ]

        best_idx = trainer._select_best_response(evaluations)

        # Should return index 1 (score 0.0)
        assert best_idx == 1

    def test_select_best_response_large_scores(self):
        """Test with large score values."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluations = [
            {"combined_score": 100.0},
            {"combined_score": 50.0},
            {"combined_score": 75.0},
        ]

        best_idx = trainer._select_best_response(evaluations)

        # Should return index 1 (score 50.0)
        assert best_idx == 1

    def test_select_best_response_fractional_scores(self):
        """Test with fractional scores."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluations = [
            {"combined_score": 2.5},
            {"combined_score": 1.3},
            {"combined_score": 2.7},
        ]

        best_idx = trainer._select_best_response(evaluations)

        # Should return index 1 (score 1.3)
        assert best_idx == 1

    def test_select_best_response_negative_scores(self):
        """Test with negative scores (edge case)."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluations = [
            {"combined_score": 1.0},
            {"combined_score": -1.0},
            {"combined_score": 0.0},
        ]

        best_idx = trainer._select_best_response(evaluations)

        # Should return index 1 (lowest score -1.0)
        assert best_idx == 1

    def test_select_best_response_returns_int(self):
        """Test that return value is int (not numpy.int64)."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy, None, None)

        evaluations = [
            {"combined_score": 3.0},
            {"combined_score": 1.0},
        ]

        best_idx = trainer._select_best_response(evaluations)

        assert isinstance(best_idx, int)


class TestRLAIFTrainerInit:
    """Test RLAIFTrainer initialization."""

    def test_init_with_default_framework(self):
        """Test initialization with default framework (None)."""
        mock_policy = Mock(spec=nn.Module)

        trainer = RLAIFTrainer(mock_policy, constitutional_framework=None)

        assert trainer.constitutional_framework is not None
        assert trainer.constitutional_framework.name == "default_constitutional_framework"

    def test_init_with_custom_framework(self):
        """Test initialization with custom framework."""
        mock_policy = Mock(spec=nn.Module)
        mock_framework = create_mock_framework()
        mock_framework.name = "custom_framework"

        trainer = RLAIFTrainer(mock_policy, constitutional_framework=mock_framework)

        assert trainer.constitutional_framework is mock_framework
        assert trainer.constitutional_framework.name == "custom_framework"

    def test_init_critique_model_defaults_to_policy(self):
        """Test that critique_model defaults to policy_model if None."""
        mock_policy = Mock(spec=nn.Module)

        trainer = RLAIFTrainer(mock_policy, critique_model=None)

        assert trainer.critique_model is mock_policy

    def test_init_critique_model_custom(self):
        """Test with custom critique model."""
        mock_policy = Mock(spec=nn.Module)
        mock_critique = Mock(spec=nn.Module)

        trainer = RLAIFTrainer(mock_policy, critique_model=mock_critique)

        assert trainer.critique_model is mock_critique

    def test_init_device_default_cuda_available(self):
        """Test device selection when cuda is available."""
        mock_policy = Mock(spec=nn.Module)

        with patch("torch.cuda.is_available", return_value=True):
            trainer = RLAIFTrainer(mock_policy, device=None)

            assert trainer.device.type == "cuda"

    def test_init_device_default_cuda_unavailable(self):
        """Test device selection when cuda is unavailable."""
        mock_policy = Mock(spec=nn.Module)

        with patch("torch.cuda.is_available", return_value=False):
            trainer = RLAIFTrainer(mock_policy, device=None)

            assert trainer.device.type == "cpu"

    def test_init_device_explicit(self):
        """Test with explicit device."""
        mock_policy = Mock(spec=nn.Module)
        device = torch.device("cpu")

        trainer = RLAIFTrainer(mock_policy, device=device)

        assert trainer.device is device

    def test_init_statistics_initialization(self):
        """Test statistics dictionary is properly initialized."""
        mock_policy = Mock(spec=nn.Module)

        trainer = RLAIFTrainer(mock_policy)

        assert "training_iterations" in trainer.stats
        assert "total_prompts_processed" in trainer.stats
        assert "total_responses_generated" in trainer.stats
        assert "avg_constitutional_score" in trainer.stats
        assert "improvement_rate" in trainer.stats

        assert trainer.stats["training_iterations"] == 0
        assert trainer.stats["total_prompts_processed"] == 0
        assert trainer.stats["total_responses_generated"] == 0
        assert trainer.stats["avg_constitutional_score"] == 0.0
        assert trainer.stats["improvement_rate"] == 0.0

    def test_init_ppo_parameters(self):
        """Test PPO parameters are stored correctly."""
        mock_policy = Mock(spec=nn.Module)

        trainer = RLAIFTrainer(
            mock_policy,
            learning_rate=2e-6,
            ppo_epsilon=0.3,
            ppo_value_coef=0.6,
            ppo_entropy_coef=0.02,
            kl_penalty_coef=0.03,
        )

        assert trainer.learning_rate == 2e-6
        assert trainer.ppo_epsilon == 0.3
        assert trainer.ppo_value_coef == 0.6
        assert trainer.ppo_entropy_coef == 0.02
        assert trainer.kl_penalty_coef == 0.03

    def test_init_creates_evaluator(self):
        """Test that evaluator is created during initialization."""
        mock_policy = Mock(spec=nn.Module)
        mock_framework = create_mock_framework()

        trainer = RLAIFTrainer(mock_policy, constitutional_framework=mock_framework)

        assert trainer.evaluator is not None

    def test_init_ppo_trainer_is_none(self):
        """Test that ppo_trainer is None initially (lazy init)."""
        mock_policy = Mock(spec=nn.Module)

        trainer = RLAIFTrainer(mock_policy)

        assert trainer.ppo_trainer is None

    def test_init_reward_model_optional(self):
        """Test reward model is optional."""
        mock_policy = Mock(spec=nn.Module)

        trainer = RLAIFTrainer(mock_policy, reward_model=None)

        assert trainer.reward_model is None

    def test_init_value_model_optional(self):
        """Test value model is optional."""
        mock_policy = Mock(spec=nn.Module)

        trainer = RLAIFTrainer(mock_policy, value_model=None)

        assert trainer.value_model is None

    def test_init_temperature_parameter(self):
        """Test temperature parameter storage."""
        mock_policy = Mock(spec=nn.Module)

        trainer = RLAIFTrainer(mock_policy, temperature=0.8)

        assert trainer.temperature == 0.8


class TestGenerateTrainingData:
    """Test generate_training_data() method."""

    def test_generate_training_data_processes_all_prompts(self):
        """Test that all prompts are processed."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_generate_response") as mock_gen:
            mock_gen.return_value = "Response"

            with patch.object(trainer.evaluator, "evaluate") as mock_eval:
                mock_eval.return_value = {
                    "flagged": False,
                    "direct_evaluation": {"weighted_score": 0.0},
                }

                with patch.object(trainer, "_generate_critique") as mock_critique:
                    mock_critique.return_value = "Clean"

                    prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
                    data = trainer.generate_training_data(prompts, num_responses_per_prompt=2)

                    assert len(data) == 3

    def test_generate_training_data_num_responses_per_prompt(self):
        """Test that num_responses_per_prompt is respected."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_generate_response") as mock_gen:
            mock_gen.return_value = "Response"

            with patch.object(trainer.evaluator, "evaluate") as mock_eval:
                mock_eval.return_value = {
                    "flagged": False,
                    "direct_evaluation": {"weighted_score": 0.0},
                }

                with patch.object(trainer, "_generate_critique") as mock_critique:
                    mock_critique.return_value = "Clean"

                    prompts = ["Prompt 1"]
                    data = trainer.generate_training_data(prompts, num_responses_per_prompt=5)

                    assert len(data[0]["responses"]) == 5
                    assert len(data[0]["evaluations"]) == 5

    def test_generate_training_data_statistics_tracking(self):
        """Test statistics tracking during data generation."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_generate_response") as mock_gen:
            mock_gen.return_value = "Response"

            with patch.object(trainer.evaluator, "evaluate") as mock_eval:
                mock_eval.return_value = {
                    "flagged": False,
                    "direct_evaluation": {"weighted_score": 0.0},
                }

                with patch.object(trainer, "_generate_critique") as mock_critique:
                    mock_critique.return_value = "Clean"

                    prompts = ["Prompt 1", "Prompt 2"]
                    trainer.generate_training_data(prompts, num_responses_per_prompt=3)

                    # Should track: 2 prompts * 3 responses = 6 total responses
                    assert trainer.stats["total_responses_generated"] == 6
                    assert trainer.stats["total_prompts_processed"] == 2

    def test_generate_training_data_data_structure(self):
        """Test correct data structure with all required keys."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_generate_response") as mock_gen:
            mock_gen.return_value = "Response"

            with patch.object(trainer.evaluator, "evaluate") as mock_eval:
                mock_eval.return_value = {
                    "flagged": False,
                    "direct_evaluation": {"weighted_score": 0.0},
                }

                with patch.object(trainer, "_generate_critique") as mock_critique:
                    mock_critique.return_value = "Clean"

                    prompts = ["Prompt 1"]
                    data = trainer.generate_training_data(prompts, num_responses_per_prompt=2)

                    assert "prompt" in data[0]
                    assert "responses" in data[0]
                    assert "evaluations" in data[0]
                    assert "best_response_idx" in data[0]

    def test_generate_training_data_evaluations_structure(self):
        """Test evaluation structure contains all required keys."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_generate_response") as mock_gen:
            mock_gen.return_value = "Response"

            with patch.object(trainer.evaluator, "evaluate") as mock_eval:
                mock_eval.return_value = {
                    "flagged": True,
                    "direct_evaluation": {"weighted_score": 2.0},
                }

                with patch.object(trainer, "_generate_critique") as mock_critique:
                    mock_critique.return_value = "unsafe harmful"

                    prompts = ["Prompt 1"]
                    data = trainer.generate_training_data(prompts, num_responses_per_prompt=1)

                    eval_dict = data[0]["evaluations"][0]
                    assert "constitutional_eval" in eval_dict
                    assert "critique" in eval_dict
                    assert "combined_score" in eval_dict
                    assert "flagged" in eval_dict

    def test_generate_training_data_best_response_selection(self):
        """Test that best_response_idx is computed correctly."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_generate_response") as mock_gen:
            mock_gen.return_value = "Response"

            with patch.object(trainer.evaluator, "evaluate") as mock_eval:
                mock_eval.return_value = {
                    "flagged": False,
                    "direct_evaluation": {"weighted_score": 0.0},
                }

                with patch.object(trainer, "_generate_critique") as mock_critique:
                    mock_critique.return_value = "Clean"

                    with patch.object(trainer, "_select_best_response") as mock_select:
                        mock_select.return_value = 2

                        prompts = ["Prompt 1"]
                        data = trainer.generate_training_data(prompts, num_responses_per_prompt=3)

                        assert data[0]["best_response_idx"] == 2
                        mock_select.assert_called_once()

    def test_generate_training_data_calls_compute_combined_score(self):
        """Test that _compute_combined_score is called for each response."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_generate_response") as mock_gen:
            mock_gen.return_value = "Response"

            with patch.object(trainer.evaluator, "evaluate") as mock_eval:
                mock_eval.return_value = {
                    "flagged": False,
                    "direct_evaluation": {"weighted_score": 0.0},
                }

                with patch.object(trainer, "_generate_critique") as mock_critique:
                    mock_critique.return_value = "Clean"

                    with patch.object(trainer, "_compute_combined_score") as mock_combined:
                        mock_combined.return_value = 1.0

                        prompts = ["Prompt 1"]
                        trainer.generate_training_data(prompts, num_responses_per_prompt=3)

                        # Should be called 3 times (once per response)
                        assert mock_combined.call_count == 3


class TestTrainMethod:
    """Test train() method."""

    def test_train_initializes_ppo_trainer(self):
        """Test that _initialize_ppo_trainer is called."""
        mock_policy = Mock(spec=nn.Module)
        mock_tokenizer = Mock()
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_initialize_ppo_trainer") as mock_init:
            with patch.object(trainer, "ppo_trainer") as mock_ppo:
                mock_ppo.train.return_value = {
                    "final_avg_reward": 1.0,
                    "final_kl_divergence": 0.01,
                    "training_history": {"step_avg_rewards": []},
                }

                trainer.train(["Prompt 1"], tokenizer=mock_tokenizer)

                mock_init.assert_called_once()

    def test_train_updates_statistics(self):
        """Test that statistics are updated after training."""
        mock_policy = Mock(spec=nn.Module)
        mock_tokenizer = Mock()
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_initialize_ppo_trainer"):
            with patch.object(trainer, "ppo_trainer") as mock_ppo:
                mock_ppo.train.return_value = {
                    "final_avg_reward": 1.5,
                    "final_kl_divergence": 0.02,
                    "training_history": {"step_avg_rewards": [1.0, 1.5]},
                }

                prompts = ["Prompt 1", "Prompt 2"]
                trainer.train(prompts, num_steps=10, tokenizer=mock_tokenizer)

                assert trainer.stats["training_iterations"] == 10
                assert trainer.stats["total_prompts_processed"] == 2

    def test_train_tracks_rewards(self):
        """Test that rewards are tracked in statistics."""
        mock_policy = Mock(spec=nn.Module)
        mock_tokenizer = Mock()
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_initialize_ppo_trainer"):
            with patch.object(trainer, "ppo_trainer") as mock_ppo:
                mock_ppo.train.return_value = {
                    "final_avg_reward": 2.0,
                    "final_kl_divergence": 0.01,
                    "training_history": {"step_avg_rewards": [1.0, 1.5, 2.0]},
                }

                trainer.train(["Prompt 1"], tokenizer=mock_tokenizer)

                assert trainer.stats["avg_constitutional_score"] > 0

    def test_train_computes_improvement_rate(self):
        """Test that improvement_rate is computed correctly."""
        mock_policy = Mock(spec=nn.Module)
        mock_tokenizer = Mock()
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_initialize_ppo_trainer"):
            with patch.object(trainer, "ppo_trainer") as mock_ppo:
                mock_ppo.train.return_value = {
                    "final_avg_reward": 3.0,
                    "final_kl_divergence": 0.01,
                    "training_history": {"step_avg_rewards": [1.0, 2.0, 3.0]},
                }

                trainer.train(["Prompt 1"], tokenizer=mock_tokenizer)

                # Improvement: 3.0 - 1.0 = 2.0
                assert trainer.stats["improvement_rate"] == 2.0

    def test_train_validates_when_validation_prompts_provided(self):
        """Test that validate() is called when validation_prompts provided."""
        mock_policy = Mock(spec=nn.Module)
        mock_tokenizer = Mock()
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_initialize_ppo_trainer"):
            with patch.object(trainer, "ppo_trainer") as mock_ppo:
                mock_ppo.train.return_value = {
                    "final_avg_reward": 1.0,
                    "final_kl_divergence": 0.01,
                    "training_history": {"step_avg_rewards": []},
                }

                with patch.object(trainer, "validate") as mock_validate:
                    mock_validate.return_value = 0.5

                    validation_prompts = ["Val 1", "Val 2"]
                    results = trainer.train(
                        ["Prompt 1"],
                        tokenizer=mock_tokenizer,
                        validation_prompts=validation_prompts,
                    )

                    mock_validate.assert_called_once_with(validation_prompts, mock_tokenizer)
                    assert "validation_results" in results
                    assert results["validation_results"]["constitutional_score"] == 0.5

    def test_train_no_validation_when_prompts_not_provided(self):
        """Test that validate() is not called when validation_prompts is None."""
        mock_policy = Mock(spec=nn.Module)
        mock_tokenizer = Mock()
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_initialize_ppo_trainer"):
            with patch.object(trainer, "ppo_trainer") as mock_ppo:
                mock_ppo.train.return_value = {
                    "final_avg_reward": 1.0,
                    "final_kl_divergence": 0.01,
                    "training_history": {"step_avg_rewards": []},
                }

                with patch.object(trainer, "validate") as mock_validate:
                    results = trainer.train(
                        ["Prompt 1"], tokenizer=mock_tokenizer, validation_prompts=None
                    )

                    mock_validate.assert_not_called()
                    assert results["validation_results"] == {}

    def test_train_raises_error_when_no_tokenizer(self):
        """Test that train raises error when no tokenizer available."""
        mock_policy = Mock(spec=nn.Module)
        # No tokenizer attribute
        trainer = RLAIFTrainer(mock_policy)

        with pytest.raises(ValueError, match="Tokenizer required"):
            trainer.train(["Prompt 1"], tokenizer=None)

    def test_train_returns_correct_structure(self):
        """Test that train returns correct result structure."""
        mock_policy = Mock(spec=nn.Module)
        mock_tokenizer = Mock()
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_initialize_ppo_trainer"):
            with patch.object(trainer, "ppo_trainer") as mock_ppo:
                mock_ppo.train.return_value = {
                    "final_avg_reward": 1.0,
                    "final_kl_divergence": 0.01,
                    "training_history": {"step_avg_rewards": []},
                }

                results = trainer.train(["Prompt 1"], tokenizer=mock_tokenizer)

                assert "ppo_results" in results
                assert "validation_results" in results
                assert "final_stats" in results


class TestValidateMethod:
    """Test validate() method."""

    def test_validate_sets_model_to_eval_mode(self):
        """Test that model is set to eval mode."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_generate_response") as mock_gen:
            mock_gen.return_value = "Response"

            with patch.object(trainer.evaluator, "evaluate") as mock_eval:
                mock_eval.return_value = {"direct_evaluation": {"weighted_score": 0.0}}

                with patch.object(trainer, "_generate_critique") as mock_critique:
                    mock_critique.return_value = "Clean"

                    trainer.validate(["Prompt 1"])

                    mock_policy.eval.assert_called_once()

    def test_validate_computes_average_score(self):
        """Test that average constitutional score is computed correctly."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_generate_response") as mock_gen:
            mock_gen.return_value = "Response"

            with patch.object(trainer.evaluator, "evaluate") as mock_eval:
                mock_eval.return_value = {"direct_evaluation": {"weighted_score": 2.0}}

                with patch.object(trainer, "_generate_critique") as mock_critique:
                    mock_critique.return_value = "unsafe harmful"  # Score: 2.0

                    # Expected combined scores: 2.0 + (2.0 * 0.5) = 3.0 each
                    score = trainer.validate(["Prompt 1", "Prompt 2", "Prompt 3"])

                    assert score == 3.0

    def test_validate_uses_torch_no_grad(self):
        """Test that validation uses torch.no_grad()."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_generate_response") as mock_gen:
            mock_gen.return_value = "Response"

            with patch.object(trainer.evaluator, "evaluate") as mock_eval:
                mock_eval.return_value = {"direct_evaluation": {"weighted_score": 0.0}}

                with patch.object(trainer, "_generate_critique") as mock_critique:
                    mock_critique.return_value = "Clean"

                    with patch("torch.no_grad") as mock_no_grad:
                        mock_no_grad.return_value.__enter__ = Mock()
                        mock_no_grad.return_value.__exit__ = Mock()

                        trainer.validate(["Prompt 1"])

                        mock_no_grad.assert_called_once()


class TestGetStatistics:
    """Test get_statistics() method."""

    def test_get_statistics_returns_all_stats(self):
        """Test that get_statistics returns complete structure."""
        mock_policy = Mock(spec=nn.Module)
        mock_framework = create_mock_framework()
        mock_framework.get_statistics.return_value = {"framework_stat": 1}

        trainer = RLAIFTrainer(mock_policy, constitutional_framework=mock_framework)
        trainer.evaluator.get_statistics = Mock(return_value={"evaluator_stat": 2})

        trainer.stats["training_iterations"] = 10

        stats = trainer.get_statistics()

        assert "training_iterations" in stats
        assert "evaluator_stats" in stats
        assert "framework_stats" in stats
        assert stats["training_iterations"] == 10
        assert stats["evaluator_stats"]["evaluator_stat"] == 2
        assert stats["framework_stats"]["framework_stat"] == 1


class TestInitializePPOTrainer:
    """Test _initialize_ppo_trainer() method."""

    def test_initialize_ppo_trainer_creates_trainer(self):
        """Test that PPOTrainer is created."""
        mock_policy = Mock(spec=nn.Module)
        mock_tokenizer = Mock()
        trainer = RLAIFTrainer(mock_policy)

        with patch("constitutional_ai.trainer.PPOTrainer") as mock_ppo_class:
            mock_ppo_instance = Mock()
            mock_ppo_class.return_value = mock_ppo_instance

            trainer._initialize_ppo_trainer(mock_tokenizer)

            assert trainer.ppo_trainer is mock_ppo_instance
            mock_ppo_class.assert_called_once()

    def test_initialize_ppo_trainer_only_once(self):
        """Test that PPOTrainer is only created once (lazy init)."""
        mock_policy = Mock(spec=nn.Module)
        mock_tokenizer = Mock()
        trainer = RLAIFTrainer(mock_policy)

        with patch("constitutional_ai.trainer.PPOTrainer") as mock_ppo_class:
            mock_ppo_instance = Mock()
            mock_ppo_class.return_value = mock_ppo_instance

            trainer._initialize_ppo_trainer(mock_tokenizer)
            first_trainer = trainer.ppo_trainer

            # Call again
            trainer._initialize_ppo_trainer(mock_tokenizer)
            second_trainer = trainer.ppo_trainer

            # Should be same instance (only created once)
            assert first_trainer is second_trainer
            assert mock_ppo_class.call_count == 1

    def test_initialize_ppo_trainer_with_correct_parameters(self):
        """Test that PPOTrainer is initialized with correct parameters."""
        mock_policy = Mock(spec=nn.Module)
        mock_value = Mock(spec=nn.Module)
        mock_reward = Mock()
        mock_tokenizer = Mock()

        trainer = RLAIFTrainer(
            mock_policy,
            value_model=mock_value,
            reward_model=mock_reward,
            learning_rate=2e-6,
            ppo_epsilon=0.3,
            ppo_value_coef=0.6,
            kl_penalty_coef=0.03,
            device=torch.device("cpu"),
        )

        with patch("constitutional_ai.trainer.PPOTrainer") as mock_ppo_class:
            trainer._initialize_ppo_trainer(mock_tokenizer)

            call_kwargs = mock_ppo_class.call_args.kwargs
            assert call_kwargs["policy_model"] is mock_policy
            assert call_kwargs["value_model"] is mock_value
            assert call_kwargs["reward_model"] is mock_reward
            assert call_kwargs["tokenizer"] is mock_tokenizer
            assert call_kwargs["learning_rate"] == 2e-6
            assert call_kwargs["clip_epsilon"] == 0.3
            assert call_kwargs["value_loss_coef"] == 0.6
            assert call_kwargs["kl_penalty"] == 0.03


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_prompts_list(self):
        """Test generate_training_data with empty prompts list."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_generate_response") as mock_gen:
            mock_gen.return_value = "Response"

            data = trainer.generate_training_data([])

            assert len(data) == 0
            assert trainer.stats["total_prompts_processed"] == 0

    def test_single_prompt(self):
        """Test generate_training_data with single prompt."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        with patch.object(trainer, "_generate_response") as mock_gen:
            mock_gen.return_value = "Response"

            with patch.object(trainer.evaluator, "evaluate") as mock_eval:
                mock_eval.return_value = {
                    "flagged": False,
                    "direct_evaluation": {"weighted_score": 0.0},
                }

                with patch.object(trainer, "_generate_critique") as mock_critique:
                    mock_critique.return_value = "Clean"

                    data = trainer.generate_training_data(["Single prompt"])

                    assert len(data) == 1
                    assert trainer.stats["total_prompts_processed"] == 1

    def test_critique_with_all_negative_terms_exceeds_cap(self):
        """Test that critique score is properly capped at 10.0."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        # All 14 unique negative terms (exceeds cap of 10)
        critique = "unsafe harmful biased incorrect misleading deceptive inappropriate problematic concerning violation issue unfair coercive manipulative"

        score = trainer._extract_score_from_critique(critique)

        # Should cap at 10.0 even though 14 terms present
        assert score == 10.0

    def test_combined_score_with_both_zero(self):
        """Test combined score when both components are zero."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        evaluation = {"direct_evaluation": {"weighted_score": 0.0}}
        critique = "This is perfectly fine and appropriate."

        score = trainer._compute_combined_score(evaluation, critique)

        assert score == 0.0

    def test_select_best_response_all_same_scores(self):
        """Test select_best_response when all scores are identical."""
        mock_policy = Mock(spec=nn.Module)
        trainer = RLAIFTrainer(mock_policy)

        evaluations = [
            {"combined_score": 5.0},
            {"combined_score": 5.0},
            {"combined_score": 5.0},
            {"combined_score": 5.0},
        ]

        best_idx = trainer._select_best_response(evaluations)

        # Should return first index
        assert best_idx == 0
