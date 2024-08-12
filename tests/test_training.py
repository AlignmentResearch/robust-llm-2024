import random
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from datasets import Dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer, GPTNeoXPreTrainedModel

from robust_llm.config.attack_configs import RandomTokenAttackConfig
from robust_llm.config.callback_configs import CallbackConfig
from robust_llm.config.configs import (
    AdversarialTrainingConfig,
    DatasetConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from robust_llm.config.model_configs import ModelConfig
from robust_llm.models import GPTNeoXModel
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.pipelines.training_pipeline import run_training_pipeline
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.scoring_callbacks import build_binary_scoring_callback
from robust_llm.trainer import AdversarialTrainer, AdversarialTrainingState
from robust_llm.training import (
    AdversarialTraining,
    _evaluate_dataset,
    _get_updated_attack_iterations,
)
from robust_llm.utils import FakeClassifierWithPositiveList


def test_evaluate_dataset():
    cfg = DatasetConfig(
        dataset_type="AlignmentResearch/PasswordMatch",
        revision="<2.1.0",
        n_train=10,
        n_val=10,
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    train = load_rllm_dataset(cfg, split="train").tokenize(tokenizer)

    # assume all are marked positive
    positives = tokenizer.__call__(
        train.ds["text"],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    accelerator = Accelerator()
    model = FakeClassifierWithPositiveList(tokenizer=tokenizer, positives=positives)
    # We fake the type with 'cast' because we are using a FakeClassifierWithPositiveList
    victim = GPTNeoXModel(
        cast(GPTNeoXPreTrainedModel, model),
        tokenizer,
        accelerator=accelerator,
        inference_type=InferenceType("classification"),
        train_minibatch_size=2,
        eval_minibatch_size=2,
        generation_config=None,
        family="gpt_neox",
    )
    expected = [d["clf_label"] == 1 for d in train.ds]  # type: ignore
    scoring_callback_config = CallbackConfig(
        callback_name="successes_from_text", callback_return_type="binary"
    )
    callback = build_binary_scoring_callback(scoring_callback_config)
    actual = _evaluate_dataset(
        dataset=train,
        victim=victim,
        victim_success_binary_callback=callback,
    )
    assert actual == expected


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
            train_minibatch_size=2,
            eval_minibatch_size=3,
            minibatch_multiplier=1,
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/IMDB",
            revision="<2.1.0",
            n_train=2,
            n_val=2,
        ),
        training=TrainingConfig(model_save_path_prefix_or_hf=None),
    )
    run_training_pipeline(config)


def test_adv_training_pipeline_doesnt_crash():
    config = ExperimentConfig(
        experiment_type="training",
        environment=EnvironmentConfig(
            test_mode=True,
        ),
        evaluation=EvaluationConfig(
            evaluation_attack=RandomTokenAttackConfig(initial_n_its=2),
        ),
        model=ModelConfig(
            # We use a finetuned model so that the classification head isn't
            # randomly initialized. It's fine to use a model that is already
            # partially trained; we are just testing that the attack doesn't
            # crash and need non-zero pre-attack accuracy for that purpose.
            name_or_path="AlignmentResearch/robust_llm_pythia-imdb-14m-mz-ada-v3",
            family="pythia",
            # We have to set this explicitly because we are not loading with Hydra,
            # so interpolation doesn't happen.
            inference_type="classification",
            train_minibatch_size=2,
            eval_minibatch_size=3,
            minibatch_multiplier=1,
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/IMDB",
            revision="<2.1.0",
            n_train=2,
            n_val=2,
        ),
        training=TrainingConfig(
            model_save_path_prefix_or_hf=None,
            adversarial=AdversarialTrainingConfig(
                num_examples_to_generate_each_round=2,
                num_adversarial_training_rounds=2,
                training_attack=RandomTokenAttackConfig(initial_n_its=2),
            ),
        ),
    )
    interpolated = OmegaConf.to_object(OmegaConf.structured(config))
    assert isinstance(interpolated, ExperimentConfig)
    run_training_pipeline(interpolated)


@pytest.mark.parametrize(
    "min_its, max_its, target, expected",
    [
        (1, 100, 0.5, 8),
        (1, 100, 0.75, 12),
        (1, 100, 0.125, 2),
        (1, 10, 0.75, 10),
        (1, 100, 0, 1),
        (4, 100, 0.125, 4),
    ],
)
def test_attack_modulation(
    min_its: int,
    max_its: int,
    target: float,
    expected: int,
):
    attack_config = RandomTokenAttackConfig(initial_n_its=4)
    victim_successes = [True, True, True, False]  # ASR 25%
    new_its = _get_updated_attack_iterations(
        old_its=attack_config.initial_n_its,
        victim_successes=victim_successes,
        min_attack_iterations=min_its,
        max_attack_iterations=max_its,
        target_adversarial_success_rate=target,
    )
    assert new_its == expected


def test_adv_training_state():
    state = AdversarialTrainingState(
        current_round=7,
        rng=np.random.default_rng(42),
        adversarial_dataset=Dataset.from_dict(
            {
                "text": ["a", "b", "c"],
                "clf_label": [0, 1, 0],
            }
        ),
        training_attack_rng=random.Random(4),
        validation_attack_rng=random.Random(2),
    )
    state.save("/tmp/")
    loaded = AdversarialTrainingState.load("/tmp/")
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
        ),
        evaluation=EvaluationConfig(
            evaluation_attack=RandomTokenAttackConfig(initial_n_its=2),
        ),
        model=ModelConfig(
            name_or_path="AlignmentResearch/robust_llm_pythia-imdb-14m-mz-ada-v3",
            family="pythia",
            inference_type="classification",
            train_minibatch_size=1,
            eval_minibatch_size=1,
            minibatch_multiplier=1,
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/IMDB",
            revision="<2.1.0",
            n_train=2,
            n_val=2,
        ),
        training=TrainingConfig(
            model_save_path_prefix_or_hf=None,
            adversarial=AdversarialTrainingConfig(
                num_examples_to_generate_each_round=2,
                num_adversarial_training_rounds=2,
                training_attack=RandomTokenAttackConfig(initial_n_its=2),
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
    model_name_to_save = "dummy"
    training = AdversarialTraining(
        config=args.training,
        train_rllm_dataset=train_set,
        eval_rllm_dataset={"validation": val_set},
        victim=victim,
        model_name_to_save=model_name_to_save,
        environment_config=args.environment,
        evaluation_config=args.evaluation,
        run_name=args.run_name,
        validation_attack_config=args.evaluation.evaluation_attack,
    )
    trainer = training.setup_trainer()
    return trainer


def test_get_train_dataloader(adv_trainer: AdversarialTrainer):
    assert adv_trainer.eval_dataset is not None
    val_set = adv_trainer.eval_dataset["validation"]
    assert isinstance(val_set, Dataset)

    assert adv_trainer.train_dataset.num_rows == 2
    adv_trainer.add_new_adversarial_examples(val_set)
    dataloader = adv_trainer.get_train_dataloader()
    assert adv_trainer.train_dataset.num_rows == 4
    assert len(dataloader) == 4


def test_empty_adversarial_dataset(adv_trainer: AdversarialTrainer):
    result_dataset, result_indices = adv_trainer.get_augmented_training_set()

    assert result_dataset == adv_trainer.regular_dataset
    assert result_indices == []


def test_weight_adv_examples_by_loss(adv_trainer: AdversarialTrainer):
    assert adv_trainer.eval_dataset is not None
    val_set = adv_trainer.eval_dataset["validation"]
    assert isinstance(val_set, Dataset)

    adv_trainer.max_augmented_data_size = 3
    adv_trainer.loss_rank_weight = 1.0
    adv_trainer.sampling_decay = 1.0
    adv_trainer.add_new_adversarial_examples(val_set)
    adv_trainer.adversarial_losses = {0: 0.0, 1: float("inf")}
    result_dataset, result_indices = adv_trainer.get_augmented_training_set()
    assert result_indices == [1]
    assert len(result_dataset) == 3


def test_weight_adv_examples_by_recency(adv_trainer: AdversarialTrainer):
    assert adv_trainer.eval_dataset is not None
    val_set = adv_trainer.eval_dataset["validation"]
    assert isinstance(val_set, Dataset)

    adv_trainer.max_augmented_data_size = 3
    adv_trainer.loss_rank_weight = 0.0
    adv_trainer.sampling_decay = 22
    adv_trainer.add_new_adversarial_examples(val_set)
    result_dataset, result_indices = adv_trainer.get_augmented_training_set()
    assert result_indices == [1]
    assert len(result_dataset) == 3


def test_equal_weight_adv_examples(adv_trainer: AdversarialTrainer):
    assert adv_trainer.eval_dataset is not None
    val_set = adv_trainer.eval_dataset["validation"]
    assert isinstance(val_set, Dataset)

    adv_trainer.max_augmented_data_size = 3
    adv_trainer.loss_rank_weight = 0.0
    adv_trainer.sampling_decay = 0.0
    adv_trainer.add_new_adversarial_examples(val_set)
    result_dataset, result_indices = adv_trainer.get_augmented_training_set()
    assert result_indices == [0]
    assert len(result_dataset) == 3


def test_compute_loss(adv_trainer: AdversarialTrainer):
    adv_trainer.adversarial_indices = [1, 2, 0]
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
        1: np.inf,
        2: np.inf,
        0: 0,
    }
