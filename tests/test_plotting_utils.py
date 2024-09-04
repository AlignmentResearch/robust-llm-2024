from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from robust_llm.plotting_utils.tools import (
    _get_num_params_from_name,
    _get_pretraining_fraction,
    extract_size_from_model_name,
    get_metrics_adv_training,
    postprocess_data,
    prepare_adv_training_data,
)
from robust_llm.wandb_utils.constants import FINAL_PYTHIA_CHECKPOINT


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "name_or_path": ["pythia-14m", "pythia-6.9b"],
            "revision": ["adv-training-round-0", "adv-training-round-1"],
            "metric_1": [0.1, 0.2],
            "metric_2": [0.3, 0.4],
            "_step": [1, 2],
        }
    )


@patch("robust_llm.wandb_utils.wandb_api_tools.WANDB_API.runs")
def test_get_metrics_adv_training(mock_runs, sample_data):
    mock_run = MagicMock()
    mock_run.history.return_value = sample_data
    mock_run.summary = {"summary_key": "value"}
    mock_runs.return_value = [mock_run]

    # Call the function under test
    result = get_metrics_adv_training("test_group", ["metric_1"], ["summary_key"])

    # Assertions
    assert not result.empty
    assert "metric_1" in result.columns


def test_postprocess_data(sample_data):
    postprocess_data(sample_data)
    assert "num_params" in sample_data.columns
    assert "pretraining_fraction" in sample_data.columns


def test_get_num_params_from_name():
    assert _get_num_params_from_name("pythia-70m") == 44_672_000
    with pytest.raises(AssertionError):
        _get_num_params_from_name("invalid_model_name")


def test_get_pretraining_fraction():
    assert _get_pretraining_fraction("model-ch-1000") == 1000 / FINAL_PYTHIA_CHECKPOINT
    assert _get_pretraining_fraction("model-no-checkpoint") == 1.0


@patch("robust_llm.wandb_utils.wandb_api_tools.WANDB_API.runs")
def test_prepare_adv_training_data(mock_runs, sample_data):
    mock_run = MagicMock()
    mock_run.history.return_value = sample_data
    mock_runs.return_value = [mock_run]

    result = prepare_adv_training_data("test_name", ["summary_key"], ["metric_1"])
    assert not result.empty
    assert "metric_1" in result.columns


def test_extract_size_from_model_name():
    assert extract_size_from_model_name("Qwen/Qwen1.5-0.5B-Chat") == 500_000_000
    assert extract_size_from_model_name("Qwen/Qwen1.5-1.8B-Chat") == 1_800_000_000
    assert extract_size_from_model_name("meta-llama/Llama-2-7b-hf") == 7_000_000_000
    assert extract_size_from_model_name("EleutherAI/pythia-14m") == 14_000_000
