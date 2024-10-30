from __future__ import annotations

import pytest
import torch

from robust_llm.config.attack_configs import RandomTokenAttackConfig
from robust_llm.config.configs import (
    AdversarialTrainingConfig,
    DatasetConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    SaveTo,
    TrainingConfig,
)
from robust_llm.config.model_configs import ModelConfig
from robust_llm.dist_utils import is_main_process
from robust_llm.pipelines.training_pipeline import run_training_pipeline
from robust_llm.training.state_classes import (
    AdversarialPipelineState,
    TrainingPipelineState,
)
from robust_llm.utils import interpolate_config


@pytest.mark.multigpu
def test_training_pipeline_final_state():
    config = ExperimentConfig(
        experiment_type="training",
        environment=EnvironmentConfig(
            test_mode=True,
            allow_checkpointing=False,  # Otherwise checkpoints might already exist.
        ),
        evaluation=EvaluationConfig(),
        model=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            # We have to set this explicitly because we are not loading with Hydra,
            # so interpolation doesn't happen.
            inference_type="classification",
            max_minibatch_size=4,
            eval_minibatch_multiplier=1,
            env_minibatch_multiplier=0.5,
            effective_batch_size=4,
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/IMDB",
            revision="2.1.0",
            n_train=5,
            n_val=2,
        ),
        training=TrainingConfig(
            save_prefix="test_training_pipeline",
            save_to=SaveTo.NONE,
            save_name="TEST_SAVE_NAME",
            # TODO(GH#990): Make lr scheduler configurable.
            lr_scheduler_type="constant",
        ),
    )
    assert config.training is not None
    final_state = run_training_pipeline(config)
    assert isinstance(final_state, TrainingPipelineState)
    assert final_state.epoch == config.training.num_train_epochs
    assert final_state.config == config
    assert len(final_state.dataset_state.clean_dataset) == config.dataset.n_train
    assert len(final_state.dataset_state.val_dataset) == config.dataset.n_val
    assert final_state.dataset_state.adv_dataset is None
    assert final_state.model_state.wrapped_model.n_params == 7629056
    assert final_state.training_state.lr_scheduler.get_last_lr()[0] == 5e-5
    assert isinstance(final_state.rng_state.torch_rng_state, torch.Tensor)
    assert final_state.rng_state.torch_rng_state.shape[0] == 5056
    assert final_state.rng_state.torch_rng_state.min() == 0
    assert final_state.rng_state.torch_rng_state.max() == 255
    if is_main_process():
        assert final_state.flops == pytest.approx(1.785e11, rel=0.05)


@pytest.mark.multigpu
def test_adv_training_pipeline_final_state():
    config = ExperimentConfig(
        experiment_type="training",
        environment=EnvironmentConfig(
            test_mode=True,
            allow_checkpointing=False,  # Otherwise checkpoints might already exist.
        ),
        evaluation=EvaluationConfig(
            num_iterations=2,
            evaluation_attack=RandomTokenAttackConfig(),
        ),
        model=ModelConfig(
            # We use a finetuned model so that the classification head isn't
            # randomly initialized.
            name_or_path="AlignmentResearch/robust_llm_pythia-14m_clf_imdb_v-ian-067_s-0",  # noqa: E501
            family="pythia",
            # We have to set this explicitly because we are not loading with Hydra,
            # so interpolation doesn't happen.
            inference_type="classification",
            max_minibatch_size=4,
            eval_minibatch_multiplier=1,
            env_minibatch_multiplier=0.5,
            effective_batch_size=4,
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/IMDB",
            revision="2.1.0",
            n_train=5,
            n_val=2,
        ),
        training=TrainingConfig(
            save_prefix="test_adv_training_pipeline",
            save_to=SaveTo.NONE,
            save_name="TEST_SAVE_NAME",
            adversarial=AdversarialTrainingConfig(
                num_examples_to_generate_each_round=2,
                num_adversarial_training_rounds=4,
                training_attack=RandomTokenAttackConfig(),
            ),
            # TODO(GH#990): Make lr scheduler configurable.
            lr_scheduler_type="constant",
        ),
    )
    interpolated = interpolate_config(config)
    final_state = run_training_pipeline(interpolated)
    assert isinstance(final_state, AdversarialPipelineState)
    assert (
        interpolated.training is not None
        and interpolated.training.adversarial is not None
    )
    assert (
        final_state.epoch
        == interpolated.training.num_train_epochs
        * interpolated.training.adversarial.num_adversarial_training_rounds
    )
    assert final_state.config == interpolated
    assert len(final_state.dataset_state.clean_dataset) == interpolated.dataset.n_train
    assert len(final_state.dataset_state.val_dataset) == interpolated.dataset.n_val
    assert final_state.dataset_state.adv_dataset is not None
    assert (
        len(final_state.dataset_state.adv_dataset)
        == interpolated.training.adversarial.num_examples_to_generate_each_round
        * interpolated.training.adversarial.num_adversarial_training_rounds
    )
    assert (
        0
        < max(final_state.dataset_state.adv_losses.keys())
        <= len(final_state.dataset_state.adv_dataset)
    )
    assert (
        0
        < max(final_state.dataset_state.clean_index_map.values())
        <= len(final_state.dataset_state.clean_dataset)
    )
    assert (
        0
        < max(final_state.dataset_state.adv_index_map.values())
        <= len(final_state.dataset_state.adv_dataset)
    )
    assert final_state.model_state.wrapped_model.n_params == 7629056
    assert final_state.training_state.lr_scheduler.get_last_lr()[0] == 5e-5
    assert isinstance(final_state.rng_state.torch_rng_state, torch.Tensor)
    assert final_state.rng_state.torch_rng_state.shape[0] == 5056
    assert final_state.rng_state.torch_rng_state.min() == 0
    assert final_state.rng_state.torch_rng_state.max() == 255
    assert torch.where(final_state.rng_state.torch_rng_state == 255)[0].tolist() == [
        136,
        744,
        1192,
        1202,
        2034,
        2345,
        3040,
        3178,
        3425,
        3457,
        3746,
        4474,
        4499,
    ]
    if is_main_process():
        assert pytest.approx(1.06e12, rel=0.06) == final_state.flops
