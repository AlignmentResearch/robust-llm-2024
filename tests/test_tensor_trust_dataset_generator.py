import numpy as np
import pytest

from robust_llm.dataset_management.tensor_trust.tensor_trust_dataset_generator import (
    _generate_dataset,
    _shuffle_tensor_trust_dataset,
)


def test_generates_consistently_for_set_seed():
    seed = np.random.randint(low=0, high=100)
    words = ["hello", "goodbye", "banana", "apple", "yellow", "red", "green", "blue"]
    first_generation = _generate_dataset(words=words, dataset_size=6, seed=seed)
    second_generation = _generate_dataset(words=words, dataset_size=6, seed=seed)

    for i, _el in enumerate(first_generation):
        assert first_generation[i] == second_generation[i]  # list equality


def test_balanced_shuffle():
    seed = np.random.randint(low=0, high=100)
    words = [f"word-{i}" for i in range(100)]
    contexts, questions, answers = _generate_dataset(
        words=words, dataset_size=100, seed=seed
    )
    (
        _shuffled_contexts,
        _shuffled_questions,
        shuffled_answers,
    ) = _shuffle_tensor_trust_dataset(
        contexts, questions, answers, seed=seed, balanced=True
    )
    left_half_answers = shuffled_answers[:50]
    right_half_answers = shuffled_answers[50:]
    left_half_positives = np.where(np.array(left_half_answers) == "Access Granted")[0]
    left_half_negatives = np.where(np.array(left_half_answers) == "Access Denied")[0]
    right_half_positives = np.where(np.array(right_half_answers) == "Access Granted")[0]
    right_half_negatives = np.where(np.array(right_half_answers) == "Access Denied")[0]
    assert len(left_half_positives) == len(left_half_negatives)
    assert len(right_half_positives) == len(right_half_negatives)
    assert len(left_half_positives) == len(right_half_positives)
    assert len(left_half_negatives) == len(right_half_negatives)
