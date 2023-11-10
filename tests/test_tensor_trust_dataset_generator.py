import numpy as np
import pytest

from robust_llm.dataset_management.tensor_trust.tensor_trust_dataset_generator import (
    _generate_dataset,
)


def test_generates_consistently_for_set_seed():
    seed = np.random.randint(low=0, high=100)
    words = ["hello", "goodbye", "good", "bad", "yellow", "red"]
    first_generation = _generate_dataset(words=words, dataset_size=5, seed=seed)
    second_generation = _generate_dataset(words=words, dataset_size=5, seed=seed)

    for i in range(len(first_generation)):
        assert first_generation[i] == second_generation[i]
