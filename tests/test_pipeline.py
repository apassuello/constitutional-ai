"""
Unit tests for pipeline.py
Tests the ConstitutionalPipeline orchestration logic with comprehensive mocking.
Focus: Testing orchestration, NOT full integration.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from constitutional_ai.pipeline import ConstitutionalPipeline
from tests.mocks.framework import create_mock_framework
from tests.mocks.transformers import create_mock_model, create_mock_tokenizer


class TestPipelineInit:
    """Test ConstitutionalPipeline initialization."""

    def test_init_device_default_cuda_available(self):
        """Test device defaults to cuda when available."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()

        with patch("torch.cuda.is_available", return_value=True):
            pipeline = ConstitutionalPipeline(base_model=model, tokenizer=tokenizer, device=None)

            assert pipeline.device.type == "cuda"

    def test_init_device_default_cpu_no_cuda(self):
        """Test device defaults to cpu when cuda unavailable."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()

        with patch("torch.cuda.is_available", return_value=False):
            pipeline = ConstitutionalPipeline(base_model=model, tokenizer=tokenizer, device=None)

            assert pipeline.device.type == "cpu"

    def test_init_device_custom_parameter(self):
        """Test custom device parameter is used."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        custom_device = torch.device("cpu")

        pipeline = ConstitutionalPipeline(
            base_model=model, tokenizer=tokenizer, device=custom_device
        )

        assert pipeline.device == custom_device

    def test_init_framework_default_none(self):
        """Test framework defaults to setup_default_framework() when None."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()

        with patch("constitutional_ai.pipeline.setup_default_framework") as mock_setup:
            mock_framework = create_mock_framework()
            mock_setup.return_value = mock_framework

            pipeline = ConstitutionalPipeline(
                base_model=model, tokenizer=tokenizer, constitutional_framework=None
            )

            mock_setup.assert_called_once()
            assert pipeline.constitutional_framework == mock_framework

    def test_init_framework_custom_parameter(self):
        """Test custom framework parameter is used."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        custom_framework = create_mock_framework()

        pipeline = ConstitutionalPipeline(
            base_model=model,
            tokenizer=tokenizer,
            constitutional_framework=custom_framework,
        )

        assert pipeline.constitutional_framework == custom_framework

    def test_init_value_model_none(self):
        """Test value_model is None when not provided."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()

        pipeline = ConstitutionalPipeline(base_model=model, tokenizer=tokenizer, value_model=None)

        assert pipeline.value_model is None

    def test_init_value_model_provided(self):
        """Test value_model is set when provided."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        value_model = create_mock_model()

        pipeline = ConstitutionalPipeline(
            base_model=model, tokenizer=tokenizer, value_model=value_model
        )

        assert pipeline.value_model == value_model
        value_model.to.assert_called_once()

    def test_init_model_moved_to_device(self):
        """Test base_model is moved to device."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()

        ConstitutionalPipeline(base_model=model, tokenizer=tokenizer)

        model.to.assert_called_once()

    def test_init_training_state_initialized(self):
        """Test training state flags are initialized to False."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()

        pipeline = ConstitutionalPipeline(base_model=model, tokenizer=tokenizer)

        assert pipeline.phase1_complete is False
        assert pipeline.phase2_complete is False

    def test_init_statistics_initialized(self):
        """Test statistics dictionary is properly initialized."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()

        pipeline = ConstitutionalPipeline(base_model=model, tokenizer=tokenizer)

        assert pipeline.stats["phase1_samples_processed"] == 0
        assert pipeline.stats["phase1_revisions_generated"] == 0
        assert pipeline.stats["phase2_preference_pairs"] == 0
        assert pipeline.stats["phase2_ppo_steps"] == 0
        assert pipeline.stats["total_training_time"] == 0.0

    def test_init_reward_model_none(self):
        """Test reward_model is None initially (initialized in Phase 2)."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()

        pipeline = ConstitutionalPipeline(base_model=model, tokenizer=tokenizer)

        assert pipeline.reward_model is None

    def test_init_training_history_initialized(self):
        """Test training_history dictionary is initialized."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()

        pipeline = ConstitutionalPipeline(base_model=model, tokenizer=tokenizer)

        assert "phase1" in pipeline.training_history
        assert "phase2" in pipeline.training_history
        assert pipeline.training_history["phase1"] == {}
        assert pipeline.training_history["phase2"] == {}

    def test_init_all_parameters(self):
        """Test initialization with all parameters provided."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        framework = create_mock_framework()
        value_model = create_mock_model()
        device = torch.device("cpu")

        pipeline = ConstitutionalPipeline(
            base_model=model,
            tokenizer=tokenizer,
            device=device,
            constitutional_framework=framework,
            value_model=value_model,
            phase1_learning_rate=1e-4,
            phase2_learning_rate=5e-7,
            reward_model_learning_rate=2e-5,
            temperature=0.8,
            ppo_epsilon=0.1,
            ppo_value_coef=0.3,
            ppo_entropy_coef=0.02,
            kl_penalty_coef=0.03,
        )

        assert pipeline.base_model == model
        assert pipeline.tokenizer == tokenizer
        assert pipeline.device == device
        assert pipeline.constitutional_framework == framework
        assert pipeline.value_model == value_model
        assert pipeline.phase1_learning_rate == 1e-4
        assert pipeline.phase2_learning_rate == 5e-7
        assert pipeline.reward_model_learning_rate == 2e-5
        assert pipeline.temperature == 0.8
        assert pipeline.ppo_epsilon == 0.1
        assert pipeline.ppo_value_coef == 0.3
        assert pipeline.ppo_entropy_coef == 0.02
        assert pipeline.kl_penalty_coef == 0.03


class TestModelIntrospection:
    """Test model introspection logic in _run_phase2() lines 398-414."""

    def setup_method(self):
        """Setup common mocks for phase2 tests."""
        self.tokenizer = create_mock_tokenizer()

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_model_introspection_gpt2_style_n_embd(
        self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs
    ):
        """Test Path 1: model.config.n_embd exists and not None (GPT-2 style)."""
        # Setup model with GPT-2 style config
        model = create_mock_model(hidden_size=1024, config_style="gpt2")
        pipeline = ConstitutionalPipeline(base_model=model, tokenizer=self.tokenizer)
        pipeline.phase1_complete = True

        # Mock preference generation
        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        # Mock reward model
        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        # Mock reward trainer
        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        # Mock PPO trainer
        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        # Run phase 2
        pipeline._run_phase2(
            prompts=["test"],
            num_epochs=1,
            responses_per_prompt=2,
            reward_model_epochs=1,
            ppo_steps=10,
            ppo_batch_size=4,
            ppo_epochs_per_batch=1,
        )

        # Verify RewardModel was called with correct hidden_size from n_embd
        mock_rm.assert_called_once()
        call_kwargs = mock_rm.call_args[1]
        assert call_kwargs["hidden_size"] == 1024

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_model_introspection_modern_hidden_size(
        self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs
    ):
        """Test Path 2: model.config.hidden_size exists and not None (modern style)."""
        # Setup model with modern config
        model = create_mock_model(hidden_size=768, config_style="modern")
        pipeline = ConstitutionalPipeline(base_model=model, tokenizer=self.tokenizer)
        pipeline.phase1_complete = True

        # Mock preference generation
        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        # Mock reward model
        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        # Mock reward trainer
        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        # Mock PPO trainer
        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        # Run phase 2
        pipeline._run_phase2(
            prompts=["test"],
            num_epochs=1,
            responses_per_prompt=2,
            reward_model_epochs=1,
            ppo_steps=10,
            ppo_batch_size=4,
            ppo_epochs_per_batch=1,
        )

        # Verify RewardModel was called with correct hidden_size
        mock_rm.assert_called_once()
        call_kwargs = mock_rm.call_args[1]
        assert call_kwargs["hidden_size"] == 768

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_model_introspection_config_exists_no_size_attrs(
        self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs
    ):
        """Test Path 3: config exists but neither n_embd nor hidden_size → fallback 768."""
        # Setup model with config but no size attributes
        model = create_mock_model(config_style="modern")
        # Remove both size attributes
        model.config.n_embd = None
        model.config.hidden_size = None

        pipeline = ConstitutionalPipeline(base_model=model, tokenizer=self.tokenizer)
        pipeline.phase1_complete = True

        # Mock preference generation
        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        # Mock reward model
        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        # Mock reward trainer
        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        # Mock PPO trainer
        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        # Run phase 2
        pipeline._run_phase2(
            prompts=["test"],
            num_epochs=1,
            responses_per_prompt=2,
            reward_model_epochs=1,
            ppo_steps=10,
            ppo_batch_size=4,
            ppo_epochs_per_batch=1,
        )

        # Verify RewardModel was called with fallback hidden_size=768
        mock_rm.assert_called_once()
        call_kwargs = mock_rm.call_args[1]
        assert call_kwargs["hidden_size"] == 768

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_model_introspection_no_config_attribute(
        self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs
    ):
        """Test Path 4: model has no config attribute → fallback 768."""
        # Setup model with no config
        model = create_mock_model(config_style="no_config")
        pipeline = ConstitutionalPipeline(base_model=model, tokenizer=self.tokenizer)
        pipeline.phase1_complete = True

        # Mock preference generation
        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        # Mock reward model
        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        # Mock reward trainer
        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        # Mock PPO trainer
        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        # Run phase 2
        pipeline._run_phase2(
            prompts=["test"],
            num_epochs=1,
            responses_per_prompt=2,
            reward_model_epochs=1,
            ppo_steps=10,
            ppo_batch_size=4,
            ppo_epochs_per_batch=1,
        )

        # Verify RewardModel was called with fallback hidden_size=768
        mock_rm.assert_called_once()
        call_kwargs = mock_rm.call_args[1]
        assert call_kwargs["hidden_size"] == 768


class TestStatisticsComputation:
    """Test statistics computation in _run_phase1() and _run_phase2()."""

    def setup_method(self):
        """Setup common mocks."""
        self.model = create_mock_model()
        self.tokenizer = create_mock_tokenizer()

    @patch("constitutional_ai.pipeline.critique_revision_pipeline")
    @patch("constitutional_ai.pipeline.supervised_finetune")
    def test_phase1_statistics_samples_processed(self, mock_sft, mock_critique_revision):
        """Test phase1_samples_processed = len(training_data)."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        # Mock critique_revision_pipeline to return 5 training samples
        mock_critique_revision.return_value = {
            "training_data": [{"prompt": f"p{i}", "response": f"r{i}"} for i in range(5)],
            "stats": {},
        }

        mock_sft.return_value = {"model": Mock(), "metrics": {}}

        pipeline._run_phase1(prompts=["test"], num_epochs=1, num_revisions=2, batch_size=4)

        assert pipeline.stats["phase1_samples_processed"] == 5

    @patch("constitutional_ai.pipeline.critique_revision_pipeline")
    @patch("constitutional_ai.pipeline.supervised_finetune")
    def test_phase1_statistics_revisions_generated(self, mock_sft, mock_critique_revision):
        """Test phase1_revisions_generated = len(training_data) * num_revisions."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        # Mock critique_revision_pipeline to return 10 training samples
        mock_critique_revision.return_value = {
            "training_data": [{"prompt": f"p{i}", "response": f"r{i}"} for i in range(10)],
            "stats": {},
        }

        mock_sft.return_value = {"model": Mock(), "metrics": {}}

        pipeline._run_phase1(prompts=["test"], num_epochs=1, num_revisions=3, batch_size=4)

        # 10 samples * 3 revisions = 30
        assert pipeline.stats["phase1_revisions_generated"] == 30

    @patch("constitutional_ai.pipeline.critique_revision_pipeline")
    @patch("constitutional_ai.pipeline.supervised_finetune")
    def test_phase1_statistics_different_data_sizes(self, mock_sft, mock_critique_revision):
        """Test correct counting with different data sizes."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        # Mock with 20 samples
        mock_critique_revision.return_value = {
            "training_data": [{"prompt": f"p{i}", "response": f"r{i}"} for i in range(20)],
            "stats": {},
        }

        mock_sft.return_value = {"model": Mock(), "metrics": {}}

        pipeline._run_phase1(prompts=["test"], num_epochs=1, num_revisions=5, batch_size=8)

        assert pipeline.stats["phase1_samples_processed"] == 20
        assert pipeline.stats["phase1_revisions_generated"] == 100  # 20 * 5

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_phase2_statistics_preference_pairs(
        self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs
    ):
        """Test phase2_preference_pairs = len(preference_data)."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)
        pipeline.phase1_complete = True

        # Mock preference generation with 8 pairs
        mock_gen_prefs.return_value = [
            {"prompt": f"p{i}", "response_chosen": f"c{i}", "response_rejected": f"r{i}"}
            for i in range(8)
        ]

        # Mock reward model
        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        # Mock reward trainer
        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        # Mock PPO trainer
        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        pipeline._run_phase2(
            prompts=["test"],
            num_epochs=1,
            responses_per_prompt=2,
            reward_model_epochs=1,
            ppo_steps=10,
            ppo_batch_size=4,
            ppo_epochs_per_batch=1,
        )

        assert pipeline.stats["phase2_preference_pairs"] == 8

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_phase2_statistics_ppo_steps(self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs):
        """Test phase2_ppo_steps is set correctly."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)
        pipeline.phase1_complete = True

        # Mock preference generation
        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        # Mock reward model
        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        # Mock reward trainer
        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        # Mock PPO trainer
        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        pipeline._run_phase2(
            prompts=["test"],
            num_epochs=1,
            responses_per_prompt=2,
            reward_model_epochs=1,
            ppo_steps=50,
            ppo_batch_size=4,
            ppo_epochs_per_batch=1,
        )

        assert pipeline.stats["phase2_ppo_steps"] == 50


class TestCheckpointSaveLoad:
    """Test checkpoint save/load logic (lines 543-588)."""

    def setup_method(self):
        """Setup common mocks."""
        self.model = create_mock_model()
        self.tokenizer = create_mock_tokenizer()
        self.temp_dir = "/tmp/test_checkpoints"

    def test_save_checkpoint_reward_model_exists(self):
        """Test save_checkpoint() with reward_model exists → save state_dict."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        # Create a mock reward model
        reward_model = Mock()
        reward_model.state_dict.return_value = {"reward_weights": torch.randn(10, 10)}
        pipeline.reward_model = reward_model

        with patch("torch.save") as mock_save:
            pipeline._save_phase2_checkpoint("/tmp/checkpoint.pt")

            mock_save.assert_called_once()
            checkpoint = mock_save.call_args[0][0]
            assert checkpoint["reward_model_state_dict"] is not None
            assert "reward_weights" in checkpoint["reward_model_state_dict"]

    def test_save_checkpoint_reward_model_none(self):
        """Test save_checkpoint() with reward_model None → save None."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)
        pipeline.reward_model = None

        with patch("torch.save") as mock_save:
            pipeline._save_phase2_checkpoint("/tmp/checkpoint.pt")

            mock_save.assert_called_once()
            checkpoint = mock_save.call_args[0][0]
            assert checkpoint["reward_model_state_dict"] is None

    def test_save_checkpoint_value_model_exists(self):
        """Test save_checkpoint() with value_model exists → save state_dict."""
        value_model = create_mock_model()
        pipeline = ConstitutionalPipeline(
            base_model=self.model, tokenizer=self.tokenizer, value_model=value_model
        )

        with patch("torch.save") as mock_save:
            pipeline._save_phase2_checkpoint("/tmp/checkpoint.pt")

            mock_save.assert_called_once()
            checkpoint = mock_save.call_args[0][0]
            assert checkpoint["value_model_state_dict"] is not None

    def test_save_checkpoint_value_model_none(self):
        """Test save_checkpoint() with value_model None → save None."""
        pipeline = ConstitutionalPipeline(
            base_model=self.model, tokenizer=self.tokenizer, value_model=None
        )

        with patch("torch.save") as mock_save:
            pipeline._save_phase2_checkpoint("/tmp/checkpoint.pt")

            mock_save.assert_called_once()
            checkpoint = mock_save.call_args[0][0]
            assert checkpoint["value_model_state_dict"] is None

    def test_load_checkpoint_reward_model_exists_both(self):
        """Test load_checkpoint() with reward_model_state_dict AND self.reward_model exists → load."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        # Create reward model
        reward_model = Mock()
        pipeline.reward_model = reward_model

        checkpoint = {
            "model_state_dict": {"base": torch.randn(5, 5)},
            "reward_model_state_dict": {"reward": torch.randn(3, 3)},
            "value_model_state_dict": None,
            "phase1_complete": True,
            "phase2_complete": False,
            "training_history": {},
            "stats": {},
        }

        with patch("torch.load", return_value=checkpoint):
            pipeline._load_phase2_checkpoint("/tmp/checkpoint.pt")

            reward_model.load_state_dict.assert_called_once_with(
                checkpoint["reward_model_state_dict"]
            )

    def test_load_checkpoint_reward_model_state_dict_none(self):
        """Test load_checkpoint() with reward_model_state_dict None → skip."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        # Create reward model
        reward_model = Mock()
        pipeline.reward_model = reward_model

        checkpoint = {
            "model_state_dict": {"base": torch.randn(5, 5)},
            "reward_model_state_dict": None,
            "value_model_state_dict": None,
            "phase1_complete": True,
            "phase2_complete": False,
            "training_history": {},
            "stats": {},
        }

        with patch("torch.load", return_value=checkpoint):
            pipeline._load_phase2_checkpoint("/tmp/checkpoint.pt")

            reward_model.load_state_dict.assert_not_called()

    def test_load_checkpoint_reward_model_self_none(self):
        """Test load_checkpoint() with self.reward_model None → skip."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)
        pipeline.reward_model = None

        checkpoint = {
            "model_state_dict": {"base": torch.randn(5, 5)},
            "reward_model_state_dict": {"reward": torch.randn(3, 3)},
            "value_model_state_dict": None,
            "phase1_complete": True,
            "phase2_complete": False,
            "training_history": {},
            "stats": {},
        }

        with patch("torch.load", return_value=checkpoint):
            # Should not raise error
            pipeline._load_phase2_checkpoint("/tmp/checkpoint.pt")

    def test_load_checkpoint_value_model_exists_both(self):
        """Test load_checkpoint() with value_model_state_dict AND self.value_model exists → load."""
        value_model = create_mock_model()
        pipeline = ConstitutionalPipeline(
            base_model=self.model, tokenizer=self.tokenizer, value_model=value_model
        )

        checkpoint = {
            "model_state_dict": {"base": torch.randn(5, 5)},
            "reward_model_state_dict": None,
            "value_model_state_dict": {"value": torch.randn(4, 4)},
            "phase1_complete": True,
            "phase2_complete": False,
            "training_history": {},
            "stats": {},
        }

        with patch("torch.load", return_value=checkpoint):
            pipeline._load_phase2_checkpoint("/tmp/checkpoint.pt")

            value_model.load_state_dict.assert_called_once_with(
                checkpoint["value_model_state_dict"]
            )

    def test_load_checkpoint_value_model_state_dict_none(self):
        """Test load_checkpoint() with value_model_state_dict None → skip."""
        value_model = create_mock_model()
        pipeline = ConstitutionalPipeline(
            base_model=self.model, tokenizer=self.tokenizer, value_model=value_model
        )

        checkpoint = {
            "model_state_dict": {"base": torch.randn(5, 5)},
            "reward_model_state_dict": None,
            "value_model_state_dict": None,
            "phase1_complete": True,
            "phase2_complete": False,
            "training_history": {},
            "stats": {},
        }

        with patch("torch.load", return_value=checkpoint):
            pipeline._load_phase2_checkpoint("/tmp/checkpoint.pt")

            value_model.load_state_dict.assert_not_called()

    def test_load_checkpoint_value_model_self_none(self):
        """Test load_checkpoint() with self.value_model None → skip."""
        pipeline = ConstitutionalPipeline(
            base_model=self.model, tokenizer=self.tokenizer, value_model=None
        )

        checkpoint = {
            "model_state_dict": {"base": torch.randn(5, 5)},
            "reward_model_state_dict": None,
            "value_model_state_dict": {"value": torch.randn(4, 4)},
            "phase1_complete": True,
            "phase2_complete": False,
            "training_history": {},
            "stats": {},
        }

        with patch("torch.load", return_value=checkpoint):
            # Should not raise error
            pipeline._load_phase2_checkpoint("/tmp/checkpoint.pt")


class TestStateManagement:
    """Test state management flags (phase1_complete, phase2_complete)."""

    def setup_method(self):
        """Setup common mocks."""
        self.model = create_mock_model()
        self.tokenizer = create_mock_tokenizer()

    @patch("constitutional_ai.pipeline.critique_revision_pipeline")
    @patch("constitutional_ai.pipeline.supervised_finetune")
    def test_phase1_complete_flag_set_after_run_phase1(self, mock_sft, mock_critique_revision):
        """Test phase1_complete flag is set to True after _run_phase1()."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        assert pipeline.phase1_complete is False

        # Mock phase1 components
        mock_critique_revision.return_value = {
            "training_data": [{"prompt": "p1", "response": "r1"}],
            "stats": {},
        }
        mock_sft.return_value = {"model": Mock(), "metrics": {}}

        pipeline._run_phase1(prompts=["test"], num_epochs=1, num_revisions=2, batch_size=4)

        assert pipeline.phase1_complete is True

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_phase2_complete_flag_set_in_train_method(
        self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs
    ):
        """Test phase2_complete flag is set to True in train() method."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        assert pipeline.phase2_complete is False

        # Mock all phase1 components
        with patch("constitutional_ai.pipeline.critique_revision_pipeline") as mock_cr:
            mock_cr.return_value = {
                "training_data": [{"prompt": "p1", "response": "r1"}],
                "stats": {},
            }

            with patch("constitutional_ai.pipeline.supervised_finetune") as mock_sft:
                mock_sft.return_value = {"model": Mock(), "metrics": {}}

                # Mock phase2 components
                mock_gen_prefs.return_value = [
                    {
                        "prompt": "p1",
                        "response_chosen": "c1",
                        "response_rejected": "r1",
                    }
                ]

                mock_rm_instance = Mock()
                mock_rm.return_value = mock_rm_instance
                mock_rm_instance.to.return_value = mock_rm_instance

                mock_trainer_instance = Mock()
                mock_rm_trainer.return_value = mock_trainer_instance
                mock_trainer_instance.train.return_value = {
                    "final_loss": 0.5,
                    "final_accuracy": 0.8,
                }

                mock_ppo_instance = Mock()
                mock_ppo.return_value = mock_ppo_instance
                mock_ppo_instance.train.return_value = {
                    "final_avg_reward": 2.5,
                    "final_kl_divergence": 0.01,
                }

                # Run full pipeline
                pipeline.train(
                    training_prompts=["test"],
                    phase1_epochs=1,
                    phase1_num_revisions=2,
                    phase1_batch_size=4,
                    phase2_epochs=1,
                    phase2_responses_per_prompt=2,
                    phase2_reward_model_epochs=1,
                    phase2_ppo_steps=10,
                    phase2_ppo_batch_size=4,
                    phase2_ppo_epochs_per_batch=1,
                )

                assert pipeline.phase2_complete is True

    @patch("constitutional_ai.pipeline.critique_revision_pipeline")
    @patch("constitutional_ai.pipeline.supervised_finetune")
    def test_phase1_complete_flag_persists_in_training_history(
        self, mock_sft, mock_critique_revision
    ):
        """Test phase1_complete persists and _run_phase1 returns data for history."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        # Mock phase1 components
        mock_critique_revision.return_value = {
            "training_data": [{"prompt": "p1", "response": "r1"}],
            "stats": {},
        }
        mock_sft.return_value = {"model": Mock(), "metrics": {}}

        result = pipeline._run_phase1(prompts=["test"], num_epochs=1, num_revisions=2, batch_size=4)

        assert pipeline.phase1_complete is True
        # _run_phase1 returns data that train() will store in training_history
        assert "training_data_size" in result
        assert "sft_results" in result


class TestValidationConditionals:
    """Test validation conditionals (lines 334-340, 468-474)."""

    def setup_method(self):
        """Setup common mocks."""
        self.model = create_mock_model()
        self.tokenizer = create_mock_tokenizer()

    @patch("constitutional_ai.pipeline.critique_revision_pipeline")
    @patch("constitutional_ai.pipeline.supervised_finetune")
    def test_phase1_validation_prompts_provided_calls_evaluate(
        self, mock_sft, mock_critique_revision
    ):
        """Test validation_prompts provided → call evaluate_constitutional_compliance()."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        # Mock phase1 components
        mock_critique_revision.return_value = {
            "training_data": [{"prompt": "p1", "response": "r1"}],
            "stats": {},
        }
        mock_sft.return_value = {"model": Mock(), "metrics": {}}

        with patch.object(pipeline, "evaluate_constitutional_compliance") as mock_eval:
            mock_eval.return_value = {"avg_score": 0.9, "violation_rate": 0.1}

            result = pipeline._run_phase1(
                prompts=["test"],
                num_epochs=1,
                num_revisions=2,
                batch_size=4,
                validation_prompts=["val1", "val2"],
            )

            mock_eval.assert_called_once()
            assert result["validation_results"]["avg_score"] == 0.9

    @patch("constitutional_ai.pipeline.critique_revision_pipeline")
    @patch("constitutional_ai.pipeline.supervised_finetune")
    def test_phase1_validation_prompts_none_skips_evaluate(self, mock_sft, mock_critique_revision):
        """Test validation_prompts None → skip validation."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        # Mock phase1 components
        mock_critique_revision.return_value = {
            "training_data": [{"prompt": "p1", "response": "r1"}],
            "stats": {},
        }
        mock_sft.return_value = {"model": Mock(), "metrics": {}}

        with patch.object(pipeline, "evaluate_constitutional_compliance") as mock_eval:
            result = pipeline._run_phase1(
                prompts=["test"],
                num_epochs=1,
                num_revisions=2,
                batch_size=4,
                validation_prompts=None,
            )

            mock_eval.assert_not_called()
            assert result["validation_results"] == {}

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_phase2_validation_prompts_provided_calls_evaluate(
        self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs
    ):
        """Test Phase 2 validation_prompts provided → call evaluate_constitutional_compliance()."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)
        pipeline.phase1_complete = True

        # Mock phase2 components
        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        with patch.object(pipeline, "evaluate_constitutional_compliance") as mock_eval:
            mock_eval.return_value = {"avg_score": 0.85, "violation_rate": 0.15}

            result = pipeline._run_phase2(
                prompts=["test"],
                num_epochs=1,
                responses_per_prompt=2,
                reward_model_epochs=1,
                ppo_steps=10,
                ppo_batch_size=4,
                ppo_epochs_per_batch=1,
                validation_prompts=["val1", "val2"],
            )

            mock_eval.assert_called_once()
            assert result["validation_results"]["avg_score"] == 0.85

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_phase2_validation_prompts_none_skips_evaluate(
        self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs
    ):
        """Test Phase 2 validation_prompts None → skip validation."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)
        pipeline.phase1_complete = True

        # Mock phase2 components
        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        with patch.object(pipeline, "evaluate_constitutional_compliance") as mock_eval:
            result = pipeline._run_phase2(
                prompts=["test"],
                num_epochs=1,
                responses_per_prompt=2,
                reward_model_epochs=1,
                ppo_steps=10,
                ppo_batch_size=4,
                ppo_epochs_per_batch=1,
                validation_prompts=None,
            )

            mock_eval.assert_not_called()
            assert result["validation_results"] == {}


class TestPhase1Orchestration:
    """Test Phase 1 orchestration logic."""

    def setup_method(self):
        """Setup common mocks."""
        self.model = create_mock_model()
        self.tokenizer = create_mock_tokenizer()

    @patch("constitutional_ai.pipeline.critique_revision_pipeline")
    @patch("constitutional_ai.pipeline.supervised_finetune")
    def test_phase1_calls_critique_revision_pipeline(self, mock_sft, mock_critique_revision):
        """Test _run_phase1() calls critique_revision_pipeline with correct params."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        mock_critique_revision.return_value = {
            "training_data": [{"prompt": "p1", "response": "r1"}],
            "stats": {},
        }
        mock_sft.return_value = {"model": Mock(), "metrics": {}}

        pipeline._run_phase1(
            prompts=["test1", "test2"], num_epochs=3, num_revisions=5, batch_size=8
        )

        mock_critique_revision.assert_called_once()
        call_kwargs = mock_critique_revision.call_args[1]
        assert call_kwargs["prompts"] == ["test1", "test2"]
        assert call_kwargs["num_revisions"] == 5
        assert call_kwargs["model"] == self.model
        assert call_kwargs["tokenizer"] == self.tokenizer

    @patch("constitutional_ai.pipeline.critique_revision_pipeline")
    @patch("constitutional_ai.pipeline.supervised_finetune")
    def test_phase1_calls_supervised_finetune(self, mock_sft, mock_critique_revision):
        """Test _run_phase1() calls supervised_finetune with correct params."""
        pipeline = ConstitutionalPipeline(
            base_model=self.model,
            tokenizer=self.tokenizer,
            phase1_learning_rate=1e-4,
        )

        training_data = [{"prompt": f"p{i}", "response": f"r{i}"} for i in range(5)]
        mock_critique_revision.return_value = {
            "training_data": training_data,
            "stats": {},
        }
        mock_sft.return_value = {"model": Mock(), "metrics": {}}

        pipeline._run_phase1(prompts=["test"], num_epochs=3, num_revisions=2, batch_size=4)

        mock_sft.assert_called_once()
        call_kwargs = mock_sft.call_args[1]
        assert call_kwargs["model"] == self.model
        assert call_kwargs["tokenizer"] == self.tokenizer
        assert call_kwargs["training_data"] == training_data
        assert call_kwargs["num_epochs"] == 3
        assert call_kwargs["batch_size"] == 4
        assert call_kwargs["learning_rate"] == 1e-4

    @patch("constitutional_ai.pipeline.critique_revision_pipeline")
    @patch("constitutional_ai.pipeline.supervised_finetune")
    def test_phase1_returns_correct_structure(self, mock_sft, mock_critique_revision):
        """Test _run_phase1() returns correct result structure."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        mock_critique_revision.return_value = {
            "training_data": [{"prompt": "p1", "response": "r1"}],
            "stats": {},
        }
        mock_sft.return_value = {"model": Mock(), "metrics": {"loss": 0.5}}

        result = pipeline._run_phase1(prompts=["test"], num_epochs=1, num_revisions=2, batch_size=4)

        assert "training_data_size" in result
        assert "sft_results" in result
        assert "validation_results" in result
        assert result["training_data_size"] == 1
        assert result["sft_results"]["metrics"]["loss"] == 0.5


class TestPhase2Orchestration:
    """Test Phase 2 orchestration logic."""

    def setup_method(self):
        """Setup common mocks."""
        self.model = create_mock_model()
        self.tokenizer = create_mock_tokenizer()

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_phase2_calls_generate_preference_pairs(
        self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs
    ):
        """Test _run_phase2() calls generate_preference_pairs with correct params."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)
        pipeline.phase1_complete = True

        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        pipeline._run_phase2(
            prompts=["test1", "test2"],
            num_epochs=1,
            responses_per_prompt=4,
            reward_model_epochs=1,
            ppo_steps=10,
            ppo_batch_size=4,
            ppo_epochs_per_batch=1,
        )

        mock_gen_prefs.assert_called_once()
        call_kwargs = mock_gen_prefs.call_args[1]
        assert call_kwargs["prompts"] == ["test1", "test2"]
        assert call_kwargs["responses_per_prompt"] == 4
        assert call_kwargs["model"] == self.model
        assert call_kwargs["tokenizer"] == self.tokenizer

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_phase2_initializes_reward_model(
        self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs
    ):
        """Test _run_phase2() initializes RewardModel."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)
        pipeline.phase1_complete = True

        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        assert pipeline.reward_model is None

        pipeline._run_phase2(
            prompts=["test"],
            num_epochs=1,
            responses_per_prompt=2,
            reward_model_epochs=1,
            ppo_steps=10,
            ppo_batch_size=4,
            ppo_epochs_per_batch=1,
        )

        mock_rm.assert_called_once()
        assert pipeline.reward_model == mock_rm_instance

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_phase2_trains_reward_model(self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs):
        """Test _run_phase2() trains reward model."""
        pipeline = ConstitutionalPipeline(
            base_model=self.model,
            tokenizer=self.tokenizer,
            reward_model_learning_rate=2e-5,
        )
        pipeline.phase1_complete = True

        preference_data = [
            {"prompt": f"p{i}", "response_chosen": f"c{i}", "response_rejected": f"r{i}"}
            for i in range(3)
        ]
        mock_gen_prefs.return_value = preference_data

        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        pipeline._run_phase2(
            prompts=["test"],
            num_epochs=1,
            responses_per_prompt=2,
            reward_model_epochs=5,
            ppo_steps=10,
            ppo_batch_size=8,
            ppo_epochs_per_batch=1,
        )

        mock_rm_trainer.assert_called_once()
        trainer_kwargs = mock_rm_trainer.call_args[1]
        assert trainer_kwargs["reward_model"] == mock_rm_instance
        assert trainer_kwargs["tokenizer"] == self.tokenizer
        assert trainer_kwargs["learning_rate"] == 2e-5
        assert trainer_kwargs["batch_size"] == 8

        mock_trainer_instance.train.assert_called_once()
        train_kwargs = mock_trainer_instance.train.call_args[1]
        assert train_kwargs["training_data"] == preference_data
        assert train_kwargs["num_epochs"] == 5

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_phase2_initializes_ppo_trainer(
        self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs
    ):
        """Test _run_phase2() initializes PPOTrainer."""
        value_model = create_mock_model()
        pipeline = ConstitutionalPipeline(
            base_model=self.model,
            tokenizer=self.tokenizer,
            value_model=value_model,
            phase2_learning_rate=1e-6,
            ppo_epsilon=0.15,
            ppo_value_coef=0.4,
            kl_penalty_coef=0.025,
        )
        pipeline.phase1_complete = True

        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        pipeline._run_phase2(
            prompts=["test"],
            num_epochs=1,
            responses_per_prompt=2,
            reward_model_epochs=1,
            ppo_steps=10,
            ppo_batch_size=4,
            ppo_epochs_per_batch=1,
        )

        mock_ppo.assert_called_once()
        ppo_kwargs = mock_ppo.call_args[1]
        assert ppo_kwargs["policy_model"] == self.model
        assert ppo_kwargs["value_model"] == value_model
        assert ppo_kwargs["reward_model"] == mock_rm_instance
        assert ppo_kwargs["tokenizer"] == self.tokenizer
        assert ppo_kwargs["learning_rate"] == 1e-6
        assert ppo_kwargs["clip_epsilon"] == 0.15
        assert ppo_kwargs["value_loss_coef"] == 0.4
        assert ppo_kwargs["kl_penalty"] == 0.025

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_phase2_runs_ppo_training(self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs):
        """Test _run_phase2() runs PPO training."""
        pipeline = ConstitutionalPipeline(
            base_model=self.model, tokenizer=self.tokenizer, temperature=0.9
        )
        pipeline.phase1_complete = True

        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 3.5,
            "final_kl_divergence": 0.015,
        }

        pipeline._run_phase2(
            prompts=["test1", "test2"],
            num_epochs=1,
            responses_per_prompt=2,
            reward_model_epochs=1,
            ppo_steps=50,
            ppo_batch_size=16,
            ppo_epochs_per_batch=4,
        )

        mock_ppo_instance.train.assert_called_once()
        train_kwargs = mock_ppo_instance.train.call_args[1]
        assert train_kwargs["prompts"] == ["test1", "test2"]
        assert train_kwargs["num_steps"] == 50
        assert train_kwargs["batch_size"] == 16
        assert train_kwargs["num_epochs_per_batch"] == 4
        assert train_kwargs["max_length"] == 150
        assert train_kwargs["temperature"] == 0.9

    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_phase2_returns_correct_structure(
        self, mock_ppo, mock_rm_trainer, mock_rm, mock_gen_prefs
    ):
        """Test _run_phase2() returns correct result structure."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)
        pipeline.phase1_complete = True

        mock_gen_prefs.return_value = [
            {"prompt": f"p{i}", "response_chosen": f"c{i}", "response_rejected": f"r{i}"}
            for i in range(5)
        ]

        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.3,
            "final_accuracy": 0.85,
        }

        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 4.0,
            "final_kl_divergence": 0.02,
        }

        result = pipeline._run_phase2(
            prompts=["test"],
            num_epochs=1,
            responses_per_prompt=2,
            reward_model_epochs=1,
            ppo_steps=10,
            ppo_batch_size=4,
            ppo_epochs_per_batch=1,
        )

        assert "preference_pairs" in result
        assert "reward_model_results" in result
        assert "ppo_results" in result
        assert "validation_results" in result
        assert result["preference_pairs"] == 5
        assert result["reward_model_results"]["final_loss"] == 0.3
        assert result["ppo_results"]["final_avg_reward"] == 4.0


class TestGetStatistics:
    """Test get_statistics() method."""

    def test_get_statistics_returns_correct_structure(self):
        """Test get_statistics() returns correct structure."""
        model = create_mock_model()
        tokenizer = create_mock_tokenizer()
        pipeline = ConstitutionalPipeline(base_model=model, tokenizer=tokenizer)

        # Manually set some stats
        pipeline.stats["phase1_samples_processed"] = 100
        pipeline.phase1_complete = True
        pipeline.training_history["phase1"] = {"some": "data"}

        stats = pipeline.get_statistics()

        assert "pipeline_stats" in stats
        assert "phase1_complete" in stats
        assert "phase2_complete" in stats
        assert "training_history" in stats
        assert stats["phase1_complete"] is True
        assert stats["phase2_complete"] is False
        assert stats["pipeline_stats"]["phase1_samples_processed"] == 100


class TestTrainMethodOrchestration:
    """Test train() method orchestration."""

    def setup_method(self):
        """Setup common mocks."""
        self.model = create_mock_model()
        self.tokenizer = create_mock_tokenizer()

    @patch("constitutional_ai.pipeline.critique_revision_pipeline")
    @patch("constitutional_ai.pipeline.supervised_finetune")
    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_train_method_runs_both_phases(
        self,
        mock_ppo,
        mock_rm_trainer,
        mock_rm,
        mock_gen_prefs,
        mock_sft,
        mock_critique_revision,
    ):
        """Test train() runs both Phase 1 and Phase 2."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        # Mock Phase 1
        mock_critique_revision.return_value = {
            "training_data": [{"prompt": "p1", "response": "r1"}],
            "stats": {},
        }
        mock_sft.return_value = {"model": Mock(), "metrics": {}}

        # Mock Phase 2
        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        result = pipeline.train(training_prompts=["test1", "test2"])

        assert result["phase1_complete"] is True
        assert result["phase2_complete"] is True
        assert "training_history" in result
        assert "statistics" in result

    @patch("constitutional_ai.pipeline.critique_revision_pipeline")
    @patch("constitutional_ai.pipeline.supervised_finetune")
    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_train_method_saves_phase1_checkpoint(
        self,
        mock_ppo,
        mock_rm_trainer,
        mock_rm,
        mock_gen_prefs,
        mock_sft,
        mock_critique_revision,
    ):
        """Test train() saves Phase 1 checkpoint when save_dir provided."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        # Mock Phase 1
        mock_critique_revision.return_value = {
            "training_data": [{"prompt": "p1", "response": "r1"}],
            "stats": {},
        }
        mock_sft.return_value = {"model": Mock(), "metrics": {}}

        # Mock Phase 2
        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        with patch.object(pipeline, "_save_phase1_checkpoint") as mock_save_p1:
            with patch.object(pipeline, "_save_phase2_checkpoint") as mock_save_p2:
                pipeline.train(training_prompts=["test"], save_dir="/tmp/checkpoints")

                mock_save_p1.assert_called_once_with("/tmp/checkpoints/phase1_checkpoint.pt")
                mock_save_p2.assert_called_once_with("/tmp/checkpoints/phase2_checkpoint.pt")

    @patch("constitutional_ai.pipeline.critique_revision_pipeline")
    @patch("constitutional_ai.pipeline.supervised_finetune")
    @patch("constitutional_ai.pipeline.generate_preference_pairs")
    @patch("constitutional_ai.pipeline.RewardModel")
    @patch("constitutional_ai.pipeline.RewardModelTrainer")
    @patch("constitutional_ai.pipeline.PPOTrainer")
    def test_train_method_resume_from_phase1(
        self,
        mock_ppo,
        mock_rm_trainer,
        mock_rm,
        mock_gen_prefs,
        mock_sft,
        mock_critique_revision,
    ):
        """Test train() with resume_from_phase1=True skips Phase 1."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        # Mock Phase 2
        mock_gen_prefs.return_value = [
            {"prompt": "p1", "response_chosen": "c1", "response_rejected": "r1"}
        ]

        mock_rm_instance = Mock()
        mock_rm.return_value = mock_rm_instance
        mock_rm_instance.to.return_value = mock_rm_instance

        mock_trainer_instance = Mock()
        mock_rm_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.train.return_value = {
            "final_loss": 0.5,
            "final_accuracy": 0.8,
        }

        mock_ppo_instance = Mock()
        mock_ppo.return_value = mock_ppo_instance
        mock_ppo_instance.train.return_value = {
            "final_avg_reward": 2.5,
            "final_kl_divergence": 0.01,
        }

        with patch.object(pipeline, "_load_phase1_checkpoint") as mock_load:
            with patch.object(pipeline, "_save_phase2_checkpoint"):
                pipeline.train(
                    training_prompts=["test"],
                    save_dir="/tmp/checkpoints",
                    resume_from_phase1=True,
                )

                mock_load.assert_called_once_with("/tmp/checkpoints/phase1_checkpoint.pt")
                mock_critique_revision.assert_not_called()
                mock_sft.assert_not_called()

    def test_train_method_resume_from_phase1_requires_save_dir(self):
        """Test train() with resume_from_phase1=True raises error if no save_dir."""
        pipeline = ConstitutionalPipeline(base_model=self.model, tokenizer=self.tokenizer)

        with pytest.raises(ValueError, match="save_dir required"):
            pipeline.train(training_prompts=["test"], resume_from_phase1=True, save_dir=None)
