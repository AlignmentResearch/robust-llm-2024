from unittest.mock import MagicMock, patch

import torch
from accelerate import Accelerator

from robust_llm.config.configs import (
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
)
from robust_llm.models import WrappedModel
from robust_llm.pipelines.utils import safe_run_pipeline
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset


def test_victim_num_classes():
    TEST_CASES = [
        # dataset_type, expected_num_classes
        ("AlignmentResearch/PasswordMatch", 2),
        ("AlignmentResearch/IMDB", 2),
    ]

    for dataset_type, expected_num_classes in TEST_CASES:
        config = ExperimentConfig(
            experiment_type="evaluation",
            dataset=DatasetConfig(
                dataset_type=dataset_type, n_train=10, revision="2.1.0"
            ),
            model=ModelConfig(
                name_or_path="EleutherAI/pythia-14m",
                family="pythia",
                # We have to set this explicitly because we are not loading with Hydra,
                # so interpolation doesn't happen.
                inference_type="classification",
                max_minibatch_size=4,
                eval_minibatch_multiplier=1,
                env_minibatch_multiplier=0.5,
            ),
            evaluation=EvaluationConfig(),
        )

        train_dataset = load_rllm_dataset(config.dataset, split="train")
        num_classes = train_dataset.num_classes
        victim = WrappedModel.from_config(
            config.model, accelerator=None, num_classes=num_classes
        )

        assert victim.config.num_labels == expected_num_classes


def test_safe_run_pipeline():
    def pipeline_fn(config: ExperimentConfig, accelerator: Accelerator):
        try:
            print(config.model.max_minibatch_size)
            if config.model.max_minibatch_size >= 16:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return WrappedModel.from_config(config.model, accelerator=None)
        except Exception as e:
            print(f"Got exception {e}")
            raise e

    args = ExperimentConfig(
        experiment_type="evaluation",
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/IMDB", n_train=10, revision="2.1.0"
        ),
        model=ModelConfig(
            name_or_path="EleutherAI/pythia-14m",
            family="pythia",
            # We have to set this explicitly because we are not loading with Hydra,
            # so interpolation doesn't happen.
            inference_type="classification",
            max_minibatch_size=16,
            train_minibatch_multiplier=0.5,
            eval_minibatch_multiplier=1,
            env_minibatch_multiplier=1,
            effective_batch_size=16,
        ),
        evaluation=EvaluationConfig(),
    )
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache"),
        patch("accelerate.utils.memory.should_reduce_batch_size", return_value=True),
    ):
        accelerator = MagicMock()
        model = safe_run_pipeline(pipeline_fn, args, accelerator=accelerator)
    assert args.model.max_minibatch_size == 8
    assert args.model.env_minibatch_multiplier == 1
    assert isinstance(model, WrappedModel)
    assert model.train_minibatch_size == 4
    assert model.eval_minibatch_size == 8
    assert model.effective_batch_size == 16
    assert model.gradient_accumulation_steps == 4
