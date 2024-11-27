import dataclasses
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hypothesis import assume, given
from hypothesis import strategies as st
from transformers import AutoTokenizer

from robust_llm.config.configs import (
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    SaveTo,
    TrainingConfig,
)
from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.config.model_configs import ModelConfig
from robust_llm.dist_utils import dist_rmtree
from robust_llm.experiment_utils import get_all_n_rounds_to_evaluate_pythia
from robust_llm.utils import (
    BalancedSampler,
    deterministic_hash,
    deterministic_hash_config,
    flatten_dict,
    is_correctly_padded,
    nested_list_to_tuple,
)


@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory for testing."""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    (test_dir / "test_file.txt").write_text("Test content")
    yield test_dir


def test_remove_directory_success(temp_directory):
    """Test successful removal of a directory."""
    dist_rmtree(str(temp_directory))
    assert not temp_directory.exists()


@patch("shutil.rmtree")
@patch("time.sleep")
def test_remove_directory_retry(mock_sleep, mock_rmtree, temp_directory):
    """Test retrying removal on OSError."""
    mock_rmtree.side_effect = [OSError("Test error"), None]

    dist_rmtree(str(temp_directory), retries=2, cooldown_seconds=1)

    assert mock_rmtree.call_count == 2
    assert mock_sleep.call_count == 1
    mock_sleep.assert_called_with(1)


@patch("shutil.rmtree")
@patch("time.sleep")
def test_remove_directory_max_retries(mock_sleep, mock_rmtree, temp_directory):
    """Test maximum retries reached."""
    mock_rmtree.side_effect = OSError("Test error")

    with pytest.raises(OSError, match="Test error"):
        dist_rmtree(str(temp_directory), retries=4)

    assert mock_rmtree.call_count == 5
    assert mock_sleep.call_count == 4


@patch("shutil.rmtree")
@patch("time.sleep")
def test_remove_directory_exponential_backoff(mock_sleep, mock_rmtree, temp_directory):
    """Test exponential backoff in sleep times."""
    mock_rmtree.side_effect = [OSError("Test error")] * 4 + [None]

    dist_rmtree(str(temp_directory))

    assert mock_rmtree.call_count == 5
    assert mock_sleep.call_count == 4


def test_remove_directory_permission_error(temp_directory):
    """Test handling of PermissionError."""
    with patch("shutil.rmtree", side_effect=PermissionError("Permission denied")):
        with pytest.raises(PermissionError, match="Permission denied"):
            dist_rmtree(str(temp_directory), retries=0)


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.mark.parametrize("regular_data_size", [1, 2, 3])
@pytest.mark.parametrize("adversarial_data_size", [1, 2, 3])
def test_balanced_sampler(regular_data_size: int, adversarial_data_size: int):
    regular_data = [0] * regular_data_size
    adversarial_data = [1] * adversarial_data_size
    balanced_sampler = BalancedSampler(regular_data, adversarial_data)

    assert len(balanced_sampler) == 2 * regular_data_size

    indices_for_regular_data = []

    for i, idx in enumerate(balanced_sampler):
        if i % 2 == 0:
            assert idx < regular_data_size
            indices_for_regular_data.append(idx)
        else:
            assert regular_data_size <= idx < regular_data_size + adversarial_data_size

    # Ensure all regular data points were sampled exactly once
    assert list(sorted(indices_for_regular_data)) == list(range(regular_data_size))


@given(text1=st.text(), text2=st.text())
def test_is_correctly_padded_true(tokenizer, text1: str, text2: str):
    """Test the `is_correctly_padded` function.

    We do this by tokenizing some input texts and
    checking that the returned masks pass the test.
    """
    # If both texts are empty then the mask is empty.
    assume(text1 != "" and text2 != "")
    texts = [text1, text2]
    padding_side = "right"
    tokenizer.padding_side = padding_side
    tokenized = tokenizer(texts, padding=True, return_tensors="pt")
    masks = tokenized["attention_mask"]
    for mask in masks:
        assert is_correctly_padded(mask, padding_side)

    padding_side = "left"
    tokenizer.padding_side = padding_side
    tokenized = tokenizer(texts, padding=True, return_tensors="pt")
    masks = tokenized["attention_mask"]
    for mask in masks:
        assert is_correctly_padded(mask, padding_side)


def test_is_correctly_padded_false():
    left_mask = torch.tensor([0, 1, 1, 1, 1, 1])
    right_mask = torch.tensor([1, 1, 1, 1, 1, 0])
    bad_mask = torch.tensor([1, 1, 0, 0, 0, 1, 1])

    assert not is_correctly_padded(left_mask, "right")
    assert not is_correctly_padded(right_mask, "left")
    assert not is_correctly_padded(bad_mask, "right")
    assert not is_correctly_padded(bad_mask, "left")


def test_nested_list_to_tuple():
    nested = [[1, 2], [3, 4]]
    assert nested_list_to_tuple(nested) == ((1, 2), (3, 4))


@pytest.mark.parametrize(
    "attack, start_rounds, middle_rounds, end_rounds",
    [
        ("rt", 1, 9, 0),
        ("gcg", 1, 9, 0),
        ("rt", 0, 10, 0),
        ("gcg", 0, 10, 0),
        ("rt", 10, 8, 5),
        ("gcg", 10, 8, 5),
    ],
)
def test_get_all_n_rounds_to_evaluate(attack, start_rounds, middle_rounds, end_rounds):
    n_rounds = get_all_n_rounds_to_evaluate_pythia(
        attack, start_rounds, middle_rounds, end_rounds
    )
    assert len(n_rounds) == 10
    max_adv_tr_rounds = 250 if attack == "rt" else 60
    n_adv_tr_rounds = [
        np.clip(x, 5, max_adv_tr_rounds) + 1
        for x in [953, 413, 163, 59, 21, 8, 6, 3, 1, 1]
    ]
    for i, rounds in enumerate(n_rounds):
        assert len(rounds) == len(set(rounds))
        assert all(0 < round <= n_adv_tr_rounds[i] for round in rounds)
        assert len(rounds) == min(
            start_rounds + middle_rounds + end_rounds, n_adv_tr_rounds[i]
        )


def test_hash():
    cfg1 = ExperimentConfig(
        experiment_type="training",
        environment=EnvironmentConfig(
            test_mode=True,
            allow_checkpointing=False,  # Otherwise checkpoints might already exist.
            wandb_info_filename="THIS IS FILE 1",
            save_root="/tmp",
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
            save_to=SaveTo.NONE,
            save_name="TEST_SAVE_NAME",
            # TODO(GH#990): Make lr scheduler configurable.
            lr_scheduler_type="constant",
        ),
    )

    # Replace just the wandb_info_filename.
    cfg2 = dataclasses.replace(
        cfg1,
        environment=dataclasses.replace(
            cfg1.environment,
            wandb_info_filename="THIS IS FILE 2",
        ),
    )

    assert deterministic_hash_config(cfg1) == deterministic_hash_config(cfg1)
    assert deterministic_hash_config(cfg2) == deterministic_hash_config(cfg2)
    assert deterministic_hash_config(cfg1) == deterministic_hash_config(cfg2)

    assert deterministic_hash(cfg1) == deterministic_hash(cfg1)
    assert deterministic_hash(cfg2) == deterministic_hash(cfg2)
    assert deterministic_hash(cfg1) != deterministic_hash(cfg2)


def test_flatten_dict():
    d = {
        "a": 1,
        "b": {"c": 2, "d": {"e": 3}},
        "f": {"g": 4},
    }
    assert flatten_dict(d) == {"a": 1, "b.c": 2, "b.d.e": 3, "f.g": 4}
