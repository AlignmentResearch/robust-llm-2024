from __future__ import annotations

import pytest
from omegaconf import OmegaConf

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
from robust_llm.pipelines.training_pipeline import run_training_pipeline


@pytest.mark.multigpu
def test_training_pipeline_doesnt_crash():
    config = ExperimentConfig(
        experiment_type="training",
        environment=EnvironmentConfig(
            test_mode=True,
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
            save_strategy="no",
            save_name="TEST_SAVE_NAME",
            # TODO(GH#990): Make lr scheduler configurable.
            lr_scheduler_type="constant",
        ),
    )
    run_training_pipeline(config)


@pytest.mark.multigpu
def test_adv_training_pipeline_doesnt_crash():
    config = ExperimentConfig(
        experiment_type="training",
        environment=EnvironmentConfig(
            test_mode=True,
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
            save_strategy="no",
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
    interpolated = OmegaConf.to_object(OmegaConf.structured(config))
    assert isinstance(interpolated, ExperimentConfig)
    run_training_pipeline(interpolated)
