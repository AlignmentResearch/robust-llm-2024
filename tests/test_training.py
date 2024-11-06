from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset

from robust_llm.__main__ import run
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
from robust_llm.dist_utils import dist_rmtree, is_main_process
from robust_llm.training.state_classes import (
    AdversarialPipelineState,
    TrainingPipelineState,
    get_checkpoint_path,
)
from robust_llm.training.training_utils import find_completed_checkpoints
from robust_llm.utils import interpolate_config


def assert_datasets_equal(
    dataset1: Dataset,
    dataset2: Dataset,
    check_features: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    ignore_columns: list[str] | None = None,
) -> None:
    """
    Assert that two HuggingFace datasets are equal.

    Args:
        dataset1: First dataset to compare
        dataset2: Second dataset to compare
        check_features: Whether to check if features (column types) match
        rtol: Relative tolerance for floating point comparisons
        atol: Absolute tolerance for floating point comparisons
        ignore_columns: List of column names to ignore in comparison

    Raises:
        AssertionError: If datasets are not equal, with details about the difference
    """
    # Check lengths match
    assert len(dataset1) == len(
        dataset2
    ), f"Dataset lengths don't match: {len(dataset1)} != {len(dataset2)}"

    # Get column names, excluding ignored columns
    ignore_columns = ignore_columns or []
    columns1 = set(dataset1.column_names) - set(ignore_columns)
    columns2 = set(dataset2.column_names) - set(ignore_columns)

    # Check column names match
    assert columns1 == columns2, f"Column names don't match: {columns1} != {columns2}"

    # Check features match if requested
    if check_features:
        features1 = {
            k: v for k, v in dataset1.features.items() if k not in ignore_columns
        }
        features2 = {
            k: v for k, v in dataset2.features.items() if k not in ignore_columns
        }
        assert (
            features1 == features2
        ), f"Features don't match: {features1} != {features2}"

    # Compare each column
    for column in columns1:
        values1 = dataset1[column]
        values2 = dataset2[column]

        # Handle different types of data
        if isinstance(values1[0], (int, str, bool)) or (
            isinstance(values1[0], float) and np.isnan(values1[0])
        ):
            assert values1 == values2, f"Values don't match in column {column}"
        elif isinstance(values1[0], float):
            assert np.allclose(
                values1, values2, rtol=rtol, atol=atol
            ), f"Floating point values don't match in column {column}"
        elif isinstance(values1[0], (list, dict)):
            assert all(
                v1 == v2 for v1, v2 in zip(values1, values2)
            ), f"List/dict values don't match in column {column}"
        elif isinstance(values1[0], torch.Tensor):
            assert all(
                torch.allclose(v1, v2, rtol=rtol, atol=atol)
                for v1, v2 in zip(values1, values2)
            ), f"Tensor values don't match in column {column}"
        else:
            # For other types, use direct comparison
            assert values1 == values2, f"Values don't match in column {column}"


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
            max_minibatch_size=2,
            eval_minibatch_multiplier=1,
            # We must set env_minibatch_multiplier to 1.0 to avoid it being changed by
            # safe_run_pipeline, which would change the hash.
            env_minibatch_multiplier=1.0,
            effective_batch_size=2,
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
    final_state = run(config)
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
    assert final_state.flops == pytest.approx(1.785e11, rel=0.05)


@pytest.mark.multigpu
def test_adv_training_pipeline_state_and_resumption(capsys: pytest.CaptureFixture):
    config = ExperimentConfig(
        experiment_type="training",
        environment=EnvironmentConfig(
            test_mode=True,
            allow_checkpointing=True,  # Enable checkpointing for resumption test
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
            inference_type="classification",
            max_minibatch_size=2,
            eval_minibatch_multiplier=1,
            # We must set env_minibatch_multiplier to 1.0 to avoid it being changed by
            # safe_run_pipeline, which would change the hash.
            env_minibatch_multiplier=1.0,
            effective_batch_size=2,
        ),
        dataset=DatasetConfig(
            dataset_type="AlignmentResearch/IMDB",
            revision="2.1.0",
            n_train=5,
            n_val=2,
        ),
        training=TrainingConfig(
            save_prefix="test_adv_training_pipeline",
            save_to=SaveTo.DISK,
            save_total_limit=10,
            save_name="TEST_SAVE_NAME",
            adversarial=AdversarialTrainingConfig(
                num_examples_to_generate_each_round=2,
                num_adversarial_training_rounds=4,
                training_attack=RandomTokenAttackConfig(),
            ),
            lr_scheduler_type="constant",
        ),
    )

    interpolated = interpolate_config(config)
    checkpoint_dir = get_checkpoint_path(
        Path(interpolated.environment.save_root) / "checkpoints", interpolated
    )
    dist_rmtree(checkpoint_dir)

    # Initial run and state validation
    initial_run_state = run(interpolated)
    assert isinstance(initial_run_state, AdversarialPipelineState)

    # Validate final state
    assert (
        interpolated.training is not None
        and interpolated.training.adversarial is not None
    )
    assert (
        initial_run_state.epoch
        == interpolated.training.num_train_epochs
        * interpolated.training.adversarial.num_adversarial_training_rounds
    )
    assert initial_run_state.config == interpolated
    assert (
        len(initial_run_state.dataset_state.clean_dataset)
        == interpolated.dataset.n_train
    )
    assert (
        len(initial_run_state.dataset_state.val_dataset) == interpolated.dataset.n_val
    )
    assert initial_run_state.dataset_state.adv_dataset is not None
    assert (
        len(initial_run_state.dataset_state.adv_dataset)
        == interpolated.training.adversarial.num_examples_to_generate_each_round
        * interpolated.training.adversarial.num_adversarial_training_rounds
    )
    assert (
        0
        < max(initial_run_state.dataset_state.adv_losses.keys())
        <= len(initial_run_state.dataset_state.adv_dataset)
    )
    assert (
        0
        < max(initial_run_state.dataset_state.clean_index_map.values())
        <= len(initial_run_state.dataset_state.clean_dataset)
    )
    assert (
        0
        < max(initial_run_state.dataset_state.adv_index_map.values())
        <= len(initial_run_state.dataset_state.adv_dataset)
    )
    assert initial_run_state.model_state.wrapped_model.n_params == 7629056
    assert initial_run_state.training_state.lr_scheduler.get_last_lr()[0] == 5e-5
    assert isinstance(initial_run_state.rng_state.torch_rng_state, torch.Tensor)
    assert initial_run_state.rng_state.torch_rng_state.shape[0] == 5056
    assert initial_run_state.rng_state.torch_rng_state.min() == 0
    assert initial_run_state.rng_state.torch_rng_state.max() == 255
    assert torch.where(initial_run_state.rng_state.torch_rng_state == 255)[
        0
    ].tolist() == [
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
    assert initial_run_state.flops == pytest.approx(1.06e12, rel=0.06)

    # Clear captured output from initial run
    initial_stdout = capsys.readouterr()[-1]
    if is_main_process():
        assert "No saved state found" in initial_stdout

    # Remove all but the first checkpoint
    assert checkpoint_dir.exists()
    completed_checkpoints = find_completed_checkpoints(checkpoint_dir)
    for subdir in completed_checkpoints[:-1]:
        dist_rmtree(subdir)

    # Rerun from the previous checkpoint
    rerun_state = run(interpolated)
    rerun_stdout = capsys.readouterr()[-1]
    assert isinstance(rerun_stdout, str)

    # Assert the expected checkpoint loading message appears in stdout
    expected_checkpoint_path = checkpoint_dir / "epoch_0003"
    if is_main_process():
        assert f"Loading state from {expected_checkpoint_path}" in rerun_stdout

    assert isinstance(rerun_state, AdversarialPipelineState)

    # Check various properties are unchanged between initial run and rerun
    assert rerun_state.epoch == initial_run_state.epoch
    assert rerun_state.config == initial_run_state.config
    assert_datasets_equal(
        rerun_state.dataset_state.clean_dataset.ds,
        initial_run_state.dataset_state.clean_dataset.ds,
    )
    assert_datasets_equal(
        rerun_state.dataset_state.val_dataset.ds,
        initial_run_state.dataset_state.val_dataset.ds,
    )
    assert rerun_state.dataset_state.adv_dataset is not None
    assert_datasets_equal(
        rerun_state.dataset_state.adv_dataset,
        initial_run_state.dataset_state.adv_dataset,
    )
    assert (
        rerun_state.dataset_state.adv_losses
        == initial_run_state.dataset_state.adv_losses
    )
    assert (
        rerun_state.dataset_state.clean_index_map
        == initial_run_state.dataset_state.clean_index_map
    )
    assert (
        rerun_state.dataset_state.adv_index_map
        == initial_run_state.dataset_state.adv_index_map
    )
    assert (
        rerun_state.model_state.wrapped_model.n_params
        == initial_run_state.model_state.wrapped_model.n_params
    )
    assert (
        rerun_state.training_state.lr_scheduler.get_last_lr()[0]
        == initial_run_state.training_state.lr_scheduler.get_last_lr()[0]
    )
    assert torch.allclose(
        rerun_state.rng_state.torch_rng_state,
        initial_run_state.rng_state.torch_rng_state,
    )
    assert rerun_state.flops == initial_run_state.flops

    # Validate model outputs are consistent between runs
    example_prompt = "Hello, World!"
    initial_model = initial_run_state.model_state.wrapped_model
    rerun_model = rerun_state.model_state.wrapped_model
    encoded = initial_model.tokenize(example_prompt, return_tensors="pt")
    initial_run_logits = initial_model.forward(**encoded)["logits"]
    rerun_logits = rerun_model.forward(**encoded)["logits"]
    assert torch.allclose(initial_run_logits, rerun_logits)
