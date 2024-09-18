from unittest.mock import MagicMock

import pandas as pd
import pytest

from robust_llm.attacks.attack import AttackData, AttackOutput
from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.metrics.average_initial_breach import (
    compute_aib_from_attack_output,
    compute_all_aibs,
    get_average_of_proportion,
)
from robust_llm.metrics.iterations_for_success import (
    compute_all_ifs_metrics,
    compute_ifs_metric_from_wandb,
    compute_iterations_for_success,
)
from robust_llm.metrics.metric_utils import _compute_clf_asr_from_logits
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.scoring_callbacks.scoring_callback_utils import BinaryCallbackOutput
from robust_llm.wandb_utils.wandb_api_tools import get_wandb_run


@pytest.mark.parametrize(
    "input_list, proportion, expected",
    [
        ([1, 2, 3, 4, 5], 0.6, 2.0),
        ([1, 2, 3, 4, 5], 0.0, 1.0),
        ([1, 2, 3, 4, 5], 1.0, 3.0),
        ([5, 4, 3, 2, 1], 0.4, 1.5),
        ([1.5, 2.5, 3.5, 4.5, 5.5], 0.8, 3),
        ([10, 20, 30, 40, 50], 1.0, 30.0),
        ([-5, -4, -3, -2, -1], 0.6, -4.0),
    ],
)
def test_get_average_of_proportion(input_list, proportion, expected):
    result = get_average_of_proportion(input_list, proportion)
    assert pytest.approx(result, 0.001) == expected


def test_empty_list():
    with pytest.raises(ZeroDivisionError):
        get_average_of_proportion([], 0.5)


def test_invalid_proportion():
    with pytest.raises(AssertionError):
        get_average_of_proportion([1, 2, 3], 1.5)
    with pytest.raises(AssertionError):
        get_average_of_proportion([1, 2, 3], -0.1)


@pytest.fixture(scope="session")
def rllm_dataset():
    dataset_config = DatasetConfig(
        dataset_type="AlignmentResearch/WordLength",
        config_name="pos",  # We want all the labels to be 1.
        revision="2.2.0",
        n_train=2,
        n_val=2,
    )
    dataset = load_rllm_dataset(dataset_config, split="validation")
    return dataset


def test_ifs_metric_logits(rllm_dataset):
    attack_data = AttackData(
        iteration_texts=[
            ["attack_text00", "attack_text01", "attack_text02"],
            ["attack_text10", "attack_text11", "attack_text12"],
        ],
        logits=[
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
        ],
    )
    attack_out = AttackOutput(rllm_dataset, attack_data)

    dummy_callback = MagicMock(side_effect=lambda x, y: MagicMock())
    model = MagicMock()
    model.forward = MagicMock()

    result = compute_iterations_for_success(attack_out, dummy_callback, model)
    # We count the iterations one-indexed, because the first iteration_text
    # comes *after* the first round of the attack.
    assert result.asr_per_iteration == [0.0, 0.0, 0.5, 1.0]
    expected_deciles = [0, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    assert result.ifs_per_decile == expected_deciles
    assert model.forward.call_count == 0
    assert dummy_callback.call_count == 0


def test_ifs_metric_no_logits(rllm_dataset):
    attack_data = AttackData(
        iteration_texts=[
            ["attack_text00", "attack_text01", "attack_text02"],
            ["attack_text10", "attack_text11", "attack_text12"],
        ],
        logits=None,
    )
    attack_out = AttackOutput(rllm_dataset, attack_data)

    dummy_callback = MagicMock()
    dummy_callback.side_effect = [
        BinaryCallbackOutput(successes=[True, True]),
        BinaryCallbackOutput(successes=[True, False]),
        BinaryCallbackOutput(successes=[False, False]),
    ]
    model = MagicMock()
    model.forward = MagicMock()

    result = compute_iterations_for_success(attack_out, dummy_callback, model)
    # We count the iterations one-indexed, because the first iteration_text
    # comes *after* the first round of the attack.
    assert result.asr_per_iteration == [0.0, 0.0, 0.5, 1.0]
    expected_deciles = [0, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    assert result.ifs_per_decile == expected_deciles
    assert model.forward.call_count == 0
    assert dummy_callback.call_count == 3


def test_ifs_metric_logits_no_success(rllm_dataset):
    attack_data = AttackData(
        iteration_texts=[
            ["attack_text00", "attack_text01", "attack_text02"],
            ["attack_text10", "attack_text11", "attack_text12"],
        ],
        logits=[
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ],
    )
    attack_out = AttackOutput(rllm_dataset, attack_data)

    dummy_callback = MagicMock(side_effect=lambda x, y: MagicMock())
    model = MagicMock()
    model.forward = MagicMock()

    result = compute_iterations_for_success(attack_out, dummy_callback, model)
    # We count the iterations one-indexed, because the first iteration_text
    # comes *after* the first round of the attack.
    assert result.asr_per_iteration == [0.0, 0.0, 0.5, 0.5]
    expected_deciles = [0, 2, 2, 2, 2, 2, None, None, None, None, None]
    assert result.ifs_per_decile == expected_deciles
    assert model.forward.call_count == 0
    assert dummy_callback.call_count == 0


def test_compute_clf_asr_from_logits():
    logits = [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    labels = [1, 1, 1, 1]
    asr = _compute_clf_asr_from_logits(logits, labels)
    assert asr == 0.5


@pytest.fixture
def mock_attack_output():
    dataset_config = DatasetConfig(
        dataset_type="AlignmentResearch/WordLength",
        config_name="pos",  # We want all the labels to be 1.
        revision="2.2.0",
        n_train=4,
        n_val=4,
    )
    dataset = load_rllm_dataset(dataset_config, split="validation")
    return AttackOutput(
        dataset=dataset,
        attack_data=AttackData(
            iteration_texts=[
                ["attack_text00", "attack_text01", "attack_text02"],
                ["attack_text10", "attack_text11", "attack_text12"],
            ],
            logits=[
                [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            ],
        ),
    )


def test_compute_aib_from_attack_output(mock_attack_output):
    results = compute_aib_from_attack_output(mock_attack_output)
    assert pytest.approx(results.aib_per_decile[6], 0.01) == 2.5


def test_compute_aib_with_different_proportion(mock_attack_output):
    results = compute_aib_from_attack_output(mock_attack_output)
    assert pytest.approx(results.aib_per_decile[8], 0.01) == 8.0 / 3


def test_recompute_ifs_metric():
    # TODO (Oskar): Implement the same for the AIB metric
    group_name = "ian_103a_gcg_pythia_helpful"
    run_index = "0000"
    run = get_wandb_run(group_name, run_index)
    results = compute_ifs_metric_from_wandb(group_name, run_index)
    for decile in range(11):
        key = f"robustness_metric@{decile/10:.1f}"
        wandb_result = run.history(keys=[key])[key].squeeze()
        computed = results.ifs_per_decile[decile]
        assert pytest.approx(computed, 0.01) == wandb_result


def test_aib_data_shape():
    group_name = "ian_103a_gcg_pythia_helpful"
    aib_df = compute_all_aibs(group_name, 1, 1, 1, 1)
    assert isinstance(aib_df, pd.DataFrame)
    assert len(aib_df) == 11
    assert aib_df.columns.tolist() == ["model_idx", "seed_idx", "aib", "decile"]
    assert aib_df.model_idx.eq(0).all()
    assert aib_df.seed_idx.eq(0).all()
    assert aib_df.decile.between(0, 10).all()


def test_ifs_data_shape():
    group_name = "ian_103a_gcg_pythia_helpful"
    asr_df, ifs_df = compute_all_ifs_metrics(group_name, 1, 1, 1)

    assert isinstance(ifs_df, pd.DataFrame)
    assert len(ifs_df) == 11
    assert ifs_df.columns.tolist() == ["model_idx", "seed_idx", "ifs", "decile"]
    assert ifs_df.model_idx.eq(0).all()
    assert ifs_df.seed_idx.eq(0).all()
    assert ifs_df.decile.between(0, 10).all()

    assert isinstance(asr_df, pd.DataFrame)
    assert len(asr_df) == 11
    assert asr_df.columns.tolist() == ["model_idx", "seed_idx", "asr", "iteration"]
    assert asr_df.model_idx.eq(0).all()
    assert asr_df.seed_idx.eq(0).all()
    assert asr_df.iteration.between(0, 10).all()
