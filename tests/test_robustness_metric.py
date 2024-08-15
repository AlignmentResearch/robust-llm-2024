from unittest.mock import MagicMock

import pytest

from robust_llm.attacks.attack import AttackData, AttackOutput
from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.evaluation import (
    _compute_clf_asr_from_logits,
    compute_robustness_metric_iterations,
)
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.scoring_callbacks.scoring_callback_utils import BinaryCallbackOutput


@pytest.fixture(scope="session")
def rllm_dataset():
    dataset_config = DatasetConfig(
        dataset_type="AlignmentResearch/WordLength",
        config_name="pos",  # We want all the labels to be 0.
        revision="2.2.0",
        n_train=2,
        n_val=2,
    )
    dataset = load_rllm_dataset(dataset_config, split="validation")
    return dataset


def test_robustness_metric_logits(rllm_dataset):
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

    score = compute_robustness_metric_iterations(attack_out, dummy_callback, model)
    # We count the iterations one-indexed, because the first iteration_text
    # comes *after* the first round of the attack.
    assert score == 3
    assert model.forward.call_count == 0
    assert dummy_callback.call_count == 0


def test_robustness_metric_no_logits(rllm_dataset):
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

    score = compute_robustness_metric_iterations(attack_out, dummy_callback, model)
    # We count the iterations one-indexed, because the first iteration_text
    # comes *after* the first round of the attack.
    assert score == 3
    assert model.forward.call_count == 0
    assert dummy_callback.call_count == 3


def test_robustness_metric_logits_no_success(rllm_dataset):
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

    score = compute_robustness_metric_iterations(attack_out, dummy_callback, model)
    # We count the iterations one-indexed, because the first iteration_text
    # comes *after* the first round of the attack.
    assert score is None
    assert model.forward.call_count == 0
    assert dummy_callback.call_count == 0


def test_compute_clf_asr_from_logits():
    logits = [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    labels = [1, 1, 1, 1]
    asr = _compute_clf_asr_from_logits(logits, labels)
    assert asr == 0.5
