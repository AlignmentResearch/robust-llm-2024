from unittest.mock import MagicMock

import numpy as np
import pytest
from datasets import Dataset, concatenate_datasets

from robust_llm.config.attack_configs import RandomTokenAttackConfig
from robust_llm.config.configs import (
    AdversarialTrainingConfig,
    AttackScheduleConfig,
    ExperimentConfig,
    TrainingConfig,
)
from robust_llm.dist_utils import DistributedRNG
from robust_llm.training.state_classes import AdversarialTrainingState
from robust_llm.training.training_utils import construct_combined_dataset


@pytest.fixture
def config():
    return ExperimentConfig(
        experiment_type="training",
        training=TrainingConfig(
            adversarial=AdversarialTrainingConfig(
                num_adversarial_training_rounds=3,
                num_examples_to_generate_each_round=100,
                attack_schedule=AttackScheduleConfig(
                    start=1,
                    end=10,
                ),
                max_augmented_data_size=10,
                max_adv_data_proportion=0.4,
                loss_rank_weight=0.5,
                adv_sampling_decay=-1.0,
                training_attack=RandomTokenAttackConfig(),
            ),
            num_train_epochs=5,
        ),
    )


@pytest.fixture
def accelerator():
    mock = MagicMock()
    mock.prepare = lambda data: data
    mock.is_main_process = True
    return mock


@pytest.fixture
def state(config, accelerator):
    state = AdversarialTrainingState(
        epoch=0,
        accelerator=accelerator,
        config=config,
        dataset_state=MagicMock(),
        model_state=MagicMock(),
        training_state=MagicMock(),
        rng_state=MagicMock(),
    )
    state.dataset_state.append_to_adv_dataset = MagicMock()  # type: ignore
    state.dataset_state.append_to_adv_dataset.return_value = state.dataset_state
    state.dataset_state.adv_losses = {}
    return state


def test_training_round(state):
    assert state.training_round == 0
    state.epoch = 5
    assert state.training_round == 1


def test_is_new_round(state):
    assert state.is_new_round()
    state.epoch = 1
    assert not state.is_new_round()
    state.epoch = 5
    assert state.is_new_round()


def test_training_is_finished(state):
    assert not state.training_is_finished()
    state.epoch = 15
    assert state.training_is_finished()


def test_should_save_trained_model(state):
    assert not state.should_save_trained_model()
    state.epoch = 5
    assert state.should_save_trained_model()
    state.epoch = 14
    assert not state.should_save_trained_model()
    state.epoch = 15
    assert state.should_save_trained_model()


def test_get_revision(state):
    assert state.get_revision() == "adv-round-0"
    state.epoch = 5
    assert state.get_revision() == "adv-round-1"


def test_should_augment_dataset(state):
    assert state.should_augment_dataset()
    state.epoch = 1
    assert not state.should_augment_dataset()
    state.epoch = 5
    assert state.should_augment_dataset()


def test_construct_combined_dataset():
    ds1 = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 2]})
    ds2 = Dataset.from_dict({"text": ["d", "e", "f"], "label": [3, 4, 5]})
    index_map1 = {0: 2, 2: 0, 4: 1}
    index_map2 = {1: 1, 3: 0, 5: 2}

    combined_ds = construct_combined_dataset(ds1, ds2, index_map1, index_map2)

    assert len(combined_ds) == 6
    assert combined_ds[0]["text"] == "c"
    assert combined_ds[1]["text"] == "e"
    assert combined_ds[2]["text"] == "a"
    assert combined_ds[3]["text"] == "d"
    assert combined_ds[4]["text"] == "b"
    assert combined_ds[5]["text"] == "f"

    assert combined_ds[0]["label"] == 2
    assert combined_ds[1]["label"] == 4
    assert combined_ds[2]["label"] == 0
    assert combined_ds[3]["label"] == 3
    assert combined_ds[4]["label"] == 1
    assert combined_ds[5]["label"] == 5


def test_get_adv_indices(state):
    state.dataset_state.adv_dataset = Dataset.from_dict(
        {"text": ["a", "b", "c", "d", "e"], "label": [0, 1, 2, 3, 4]}
    )
    state.dataset_state.adv_losses = {0: 0.5, 1: 0.3, 2: 0.7, 3: 0.2, 4: 0.6}
    state.rng_state.distributed_rng = DistributedRNG(
        seed=42, accelerator=state.accelerator
    )

    adv_indices = state._get_adv_indices(state.dataset_state.adv_dataset, n_adv=3)

    assert len(adv_indices) == 3
    assert all(0 <= idx < 5 for idx in adv_indices)
    assert len(set(adv_indices)) == 3  # All indices should be unique


def test_augment_dataset(state):
    clean_ds = Dataset.from_dict(
        {
            "text": ["a", "b", "c", "d", "e", "f", "g", "h"],
            "label": [0, 1, 2, 3, 4, 5, 6, 7],
        }
    )
    adv_ds = Dataset.from_dict({"text": ["x", "y", "z", "w"], "label": [8, 9, 10, 11]})

    state.dataset_state.clean_dataset.for_training.return_value = clean_ds
    state.dataset_state.adv_losses = {}
    new_adv_example = Dataset.from_dict({"text": ["new"], "label": [12]})
    state.dataset_state.adv_dataset = concatenate_datasets([adv_ds, new_adv_example])
    state.dataset_state.append_to_adv_dataset.return_value = state.dataset_state

    state.rng_state.distributed_rng = DistributedRNG(
        seed=42, accelerator=state.accelerator
    )

    state.augment_dataset()

    assert (
        len(state.dataset_state.clean_index_map) == 6
    )  # max_augmented_data_size * (1 - max_adv_data_proportion)
    assert (
        len(state.dataset_state.adv_index_map) == 4
    )  # max_augmented_data_size * max_adv_data_proportion
    assert all(
        0 <= idx <= 7 for idx in state.dataset_state.clean_index_map.values()
    )  # Values are in clean
    assert all(
        0 <= idx <= 4 for idx in state.dataset_state.adv_index_map.values()
    )  # Values are in adv
    assert any(
        5 <= idx <= 9 for idx in state.dataset_state.clean_index_map.keys()
    )  # Keys are in full
    assert any(
        3 <= idx <= 9 for idx in state.dataset_state.adv_index_map.keys()
    )  # Keys are in full


def test_dataset_shuffling_and_combining(state):
    # Create clean and adversarial datasets
    clean_ds = Dataset.from_dict(
        {"text": [f"clean_{i}" for i in range(8)], "label": list(range(8))}
    )
    adv_ds = Dataset.from_dict(
        {"text": [f"adv_{i}" for i in range(5)], "label": list(range(8, 13))}
    )

    state.dataset_state.clean_dataset.for_training.return_value = clean_ds
    state.dataset_state.adv_dataset = adv_ds

    # Mock the random number generator
    state.rng_state.distributed_rng.choice.side_effect = [
        [0, 1, 2, 3, 4, 5],  # Indices for clean dataset
        [0, 1, 2, 3],  # Indices for adv dataset
        [5, 2, 9, 0, 7, 3, 1, 8, 6, 4],  # Shuffling indices
    ]
    # So concatenated is
    # clean_0, clean_1, clean_2, clean_3, clean_4, clean_5, adv_0, adv_1, adv_2, adv_3
    # and then shuffled is
    # clean_5, clean_2, adv_3, clean_0, adv_1, clean_3, clean_1, adv_2, adv_0, clean_4

    # Call the augment_dataset method
    state.augment_dataset()

    # Check the resulting index maps
    expected_clean_index_map = {0: 5, 1: 2, 3: 0, 5: 3, 6: 1, 9: 4}
    expected_adv_index_map = {2: 3, 4: 1, 7: 2, 8: 0}

    assert state.dataset_state.clean_index_map == expected_clean_index_map
    assert state.dataset_state.adv_index_map == expected_adv_index_map

    # Now let's construct the combined dataset and verify its contents
    combined_ds = construct_combined_dataset(
        clean_ds,
        adv_ds,
        state.dataset_state.clean_index_map,
        state.dataset_state.adv_index_map,
    )

    expected_texts = [
        "clean_5",
        "clean_2",
        "adv_3",
        "clean_0",
        "adv_1",
        "clean_3",
        "clean_1",
        "adv_2",
        "adv_0",
        "clean_4",
    ]

    assert combined_ds["text"] == expected_texts


def test_dataset_shuffling_edge_cases(state):
    state.adv_config.max_augmented_data_size = 5
    state.adv_config.max_adv_data_proportion = 0.4

    # Edge case 1: No adversarial examples
    clean_ds = Dataset.from_dict(
        {"text": [f"clean_{i}" for i in range(5)], "label": list(range(5))}
    )
    adv_ds = Dataset.from_dict({"text": [], "label": []})

    state.dataset_state.clean_dataset.for_training.return_value = clean_ds
    state.dataset_state.adv_dataset = adv_ds

    state.rng_state.distributed_rng.choice.side_effect = [
        [0, 1, 2, 3, 4],  # Indices for clean dataset
        [4, 2, 0, 3, 1],  # Shuffling indices
    ]

    state.augment_dataset()

    assert len(state.dataset_state.clean_index_map) == 5
    assert len(state.dataset_state.adv_index_map) == 0

    # Edge case 2: More adversarial examples than can be used
    clean_ds = Dataset.from_dict(
        {"text": [f"clean_{i}" for i in range(5)], "label": list(range(5))}
    )
    adv_ds = Dataset.from_dict(
        {"text": [f"adv_{i}" for i in range(5)], "label": list(range(5, 10))}
    )

    state.dataset_state.clean_dataset.for_training.return_value = clean_ds
    state.dataset_state.adv_dataset = adv_ds

    state.rng_state.distributed_rng.choice.side_effect = [
        [0, 1, 2],  # Indices for clean dataset
        [0, 1],  # Indices for adv dataset
        [4, 2, 0, 3, 1],  # Shuffling indices
    ]

    state.augment_dataset()

    assert len(state.dataset_state.clean_index_map) == 3
    assert len(state.dataset_state.adv_index_map) == 2

    combined_ds = construct_combined_dataset(
        clean_ds,
        adv_ds,
        state.dataset_state.clean_index_map,
        state.dataset_state.adv_index_map,
    )

    assert len(combined_ds) == 5
    assert sum(1 for text in combined_ds["text"] if text.startswith("clean_")) == 3
    assert sum(1 for text in combined_ds["text"] if text.startswith("adv_")) == 2
    assert all(
        combined_ds["text"][i].startswith("clean_")
        for i in state.dataset_state.clean_index_map.keys()
    )
    assert all(
        combined_ds["text"][i].startswith("adv_")
        for i in state.dataset_state.adv_index_map.keys()
    )


def test_consistent_shuffling(state):
    state.adv_config.max_augmented_data_size = 10
    state.adv_config.max_adv_data_proportion = 0.4

    clean_ds = Dataset.from_dict(
        {"text": [f"clean_{i}" for i in range(8)], "label": list(range(8))}
    )
    adv_ds = Dataset.from_dict(
        {"text": [f"adv_{i}" for i in range(5)], "label": list(range(8, 13))}
    )

    state.dataset_state.clean_dataset.for_training.return_value = clean_ds
    state.dataset_state.adv_dataset = adv_ds

    # Use a fixed seed for reproducibility
    fixed_rng = np.random.default_rng(42)

    state.rng_state.distributed_rng.choice.side_effect = [
        list(fixed_rng.choice(8, size=6, replace=False)),
        list(fixed_rng.choice(5, size=4, replace=False)),
        list(fixed_rng.choice(10, size=10, replace=False)),
    ]

    state.augment_dataset()

    first_clean_index_map = state.dataset_state.clean_index_map.copy()
    first_adv_index_map = state.dataset_state.adv_index_map.copy()

    # Reset the random number generator
    fixed_rng = np.random.default_rng(42)

    state.rng_state.distributed_rng.choice.side_effect = [
        list(fixed_rng.choice(8, size=6, replace=False)),
        list(fixed_rng.choice(5, size=4, replace=False)),
        list(fixed_rng.choice(10, size=10, replace=False)),
    ]

    state.augment_dataset()

    assert state.dataset_state.clean_index_map == first_clean_index_map
    assert state.dataset_state.adv_index_map == first_adv_index_map


if __name__ == "__main__":
    pytest.main()
