from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from datasets import Dataset
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
from robust_llm.dist_utils import DistributedRNG
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.pipelines.training_pipeline import run_training_pipeline
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.trainer import AdversarialTrainer, AdversarialTrainingState
from robust_llm.training import AdversarialTraining


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
            adversarial=AdversarialTrainingConfig(
                num_examples_to_generate_each_round=2,
                num_adversarial_training_rounds=4,
                training_attack=RandomTokenAttackConfig(),
            ),
        ),
    )
    interpolated = OmegaConf.to_object(OmegaConf.structured(config))
    assert isinstance(interpolated, ExperimentConfig)
    run_training_pipeline(interpolated)


def test_adv_training_state():
    state = AdversarialTrainingState(
        current_round=7,
        rng=DistributedRNG(42, None),
        adversarial_dataset=Dataset.from_dict(
            {
                "text": ["a", "b", "c"],
                "clf_label": [0, 1, 0],
            }
        ),
        clean_indices=[0, 1, 2],
        adversarial_indices=[1, 2],
        adversarial_losses={"1": 0.1, "2": 0.2},
        training_attack_rng=DistributedRNG(4, None),
        validation_attack_rng=DistributedRNG(2, None),
    )
    state.save("/tmp/")
    loaded = AdversarialTrainingState.load("/tmp/", None)
    assert loaded.current_round == state.current_round
    assert loaded.adversarial_dataset["text"] == state.adversarial_dataset["text"]
    assert (
        loaded.adversarial_dataset["clf_label"]
        == state.adversarial_dataset["clf_label"]
    )
    assert state.rng.random() == loaded.rng.random()
    assert state.training_attack_rng is not None
    assert state.validation_attack_rng is not None
    assert loaded.training_attack_rng is not None
    assert loaded.validation_attack_rng is not None
    assert state.training_attack_rng.random() == loaded.training_attack_rng.random()
    assert state.validation_attack_rng.random() == loaded.validation_attack_rng.random()


@pytest.fixture
def adv_trainer() -> AdversarialTrainer:
    config = ExperimentConfig(
        experiment_type="training",
        environment=EnvironmentConfig(
            test_mode=True,
            allow_checkpointing=False,
        ),
        evaluation=EvaluationConfig(
            num_iterations=2,
            evaluation_attack=RandomTokenAttackConfig(),
        ),
        model=ModelConfig(
            name_or_path="AlignmentResearch/robust_llm_pythia-14m_clf_imdb_v-ian-067_s-0",  # noqa: E501
            family="pythia",
            inference_type="classification",
            max_minibatch_size=1,
            eval_minibatch_multiplier=1,
            env_minibatch_multiplier=1,
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/IMDB",
            revision="2.1.0",
            n_train=2,
            n_val=2,
        ),
        training=TrainingConfig(
            save_prefix="adv_trainer_test",
            adversarial=AdversarialTrainingConfig(
                num_examples_to_generate_each_round=2,
                num_adversarial_training_rounds=2,
                training_attack=RandomTokenAttackConfig(),
                max_augmented_data_size=4,
            ),
        ),
    )
    args = OmegaConf.to_object(OmegaConf.structured(config))
    assert isinstance(args, ExperimentConfig)
    assert args.evaluation is not None
    assert args.training is not None
    untokenized_train_set = load_rllm_dataset(args.dataset, split="train")
    untokenized_val_set = load_rllm_dataset(args.dataset, split="validation")
    num_classes = untokenized_train_set.num_classes
    victim = WrappedModel.from_config(
        args.model, accelerator=None, num_classes=num_classes
    )
    train_set = untokenized_train_set.tokenize(victim.right_tokenizer)
    val_set = untokenized_val_set.tokenize(victim.right_tokenizer)
    training = AdversarialTraining(
        config=args.training,
        train_rllm_dataset=train_set,
        eval_rllm_dataset={"validation": val_set},
        victim=victim,
        model_name="dummy",
        environment_config=args.environment,
        evaluation_config=args.evaluation,
        run_name=args.run_name,
        validation_attack_config=args.evaluation.evaluation_attack,
        hash="dummy_adv_training",
        validation_iterations=args.evaluation.num_iterations,
        local_files_path=Path("/tmp/dummy_local_files_path"),
    )
    trainer = training.setup_trainer()
    return trainer


def test_empty_adversarial_dataset(adv_trainer: AdversarialTrainer):
    adv_trainer.update_augmented_training_set(False, 0)

    assert adv_trainer.train_dataset.data == adv_trainer.regular_dataset.data
    assert adv_trainer.adversarial_indices == []


def test_weight_adv_examples_by_loss(adv_trainer: AdversarialTrainer):
    assert adv_trainer.eval_dataset is not None
    val_set = adv_trainer.eval_dataset["validation"]
    assert isinstance(val_set, Dataset)

    adv_trainer.max_augmented_data_size = 3
    adv_trainer.loss_rank_weight = 1.0
    adv_trainer.sampling_decay = 1.0
    adv_trainer.add_new_adversarial_examples(val_set)
    adv_trainer.adversarial_losses = {"0": 0.0, "1": float("inf")}
    adv_trainer.update_augmented_training_set(False, 0)
    assert adv_trainer.adversarial_indices == [1]
    assert len(adv_trainer.train_dataset) == 3


def test_weight_adv_examples_by_recency(adv_trainer: AdversarialTrainer):
    assert adv_trainer.eval_dataset is not None
    val_set = adv_trainer.eval_dataset["validation"]
    assert isinstance(val_set, Dataset)

    adv_trainer.max_augmented_data_size = 3
    adv_trainer.loss_rank_weight = 0.0
    adv_trainer.sampling_decay = 22
    adv_trainer.add_new_adversarial_examples(val_set)
    adv_trainer.update_augmented_training_set(False, 0)
    assert adv_trainer.adversarial_indices == [1]
    assert len(adv_trainer.train_dataset) == 3


def test_equal_weight_adv_examples(adv_trainer: AdversarialTrainer):
    assert adv_trainer.eval_dataset is not None
    val_set = adv_trainer.eval_dataset["validation"]
    assert isinstance(val_set, Dataset)

    adv_trainer.max_augmented_data_size = 3
    adv_trainer.loss_rank_weight = 0.0
    adv_trainer.sampling_decay = 0.0
    adv_trainer.add_new_adversarial_examples(val_set)
    adv_trainer.update_augmented_training_set(False, 0)
    assert adv_trainer.adversarial_indices == [0]
    assert len(adv_trainer.train_dataset) == 3


def test_compute_loss(adv_trainer: AdversarialTrainer):
    # Fixes Accelerator state being shared across tests, which results in
    # gather_for_metrics issues inside the compute_loss method.
    adv_trainer.accelerator.gradient_state.active_dataloader = None

    adv_trainer.adversarial_indices = [1, 2, 0]
    adv_trainer.train_dataset = Dataset.from_dict(
        {
            "text": ["a", "b", "c", "d"],
            "clf_label": [0, 1, 0, 1],
        }
    )
    mock_model = MagicMock()
    mock_model.return_value = {
        "loss": torch.tensor([0.7]),
        "logits": torch.tensor(
            [[-np.inf, 0.0], [0.0, -np.inf], [-np.inf, 0.0], [-np.inf, 0.0]]
        ),
    }
    inputs = {"labels": torch.tensor([0, 1, 0, 1])}
    loss = adv_trainer.compute_loss(mock_model, inputs)
    assert isinstance(loss, torch.Tensor)
    assert torch.isclose(loss, torch.tensor([0.7]))
    assert adv_trainer.adversarial_losses == {
        "1": np.inf,
        "2": np.inf,
        "0": 0,
    }
