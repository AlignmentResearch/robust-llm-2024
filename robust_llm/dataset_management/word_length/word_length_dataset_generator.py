import string
from typing import Sequence, Tuple

import numpy as np
from datasets import Dataset
from numpy.random import Generator

from robust_llm.dataset_management.word_length.constants import (
    CONTEXT_STRING,
    FIRST_WORD,
    SECOND_WORD,
)
from robust_llm.file_utils import compute_dataset_management_path, compute_dataset_path

DATASET_PATH = f"{compute_dataset_path()}/word_length"
RESOURCES_PATH = f"{compute_dataset_management_path()}/resources"


def get_word_length_datasets(
    train_set_size: int | None,
    validation_set_size: int | None,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    """Generates a dataset of word order examples.

    This includes a string of gibberish after the second word.
    """
    if train_set_size is None and validation_set_size is None:
        raise ValueError(
            "Train and validation size are None, so "
            "there is nothing to generate. Exiting..."
        )

    total_size = 0
    if train_set_size is not None:
        assert train_set_size > 0
        total_size += train_set_size
    if validation_set_size is not None:
        assert validation_set_size > 0
        total_size += validation_set_size

    # words.txt is from
    # https://svnweb.freebsd.org/csrg/share/dict/words?revision=61569
    word_filename = f"{RESOURCES_PATH}/words.txt"
    with open(word_filename, "r") as f:
        words = f.read().splitlines()

    instructions, random_strings, labels = _generate_dataset(
        words=words, dataset_size=total_size, seed=seed
    )

    dataset = _prepare_supervised_dataset(instructions, random_strings, labels)

    print("The first few dataset examples are:")
    print(dataset[:5])

    if train_set_size is not None:
        train_dataset = dataset.select(range(train_set_size))
    else:
        train_dataset = Dataset.from_dict({"text": [], "label": []})

    if validation_set_size is not None:
        validation_dataset = dataset.select(
            range(train_set_size if train_set_size else 0, total_size)
        )
    else:
        validation_dataset = Dataset.from_dict({"text": [], "label": []})

    return train_dataset, validation_dataset


def _prepare_supervised_dataset(
    instructions: Sequence[str],
    random_strings: Sequence[str],
    labels: Sequence[int],
) -> Dataset:

    text_chunked = [
        [instruction, random_string]
        for instruction, random_string in zip(instructions, random_strings)
    ]
    text = ["".join(pair) for pair in text_chunked]

    dataset = Dataset.from_dict(
        {
            "text": text,
            "text_chunked": text_chunked,
            "label": labels,
        }
    )

    return dataset


def _get_random_strings(dataset_size: int, rng: Generator) -> Sequence[str]:
    # Make the random strings be between size 1 and size 40
    string_lengths = rng.integers(1, 40, size=dataset_size)

    # Generate all the random characters needed
    total_length = string_lengths.sum()
    random_chars = rng.choice(list(string.printable), size=total_length, replace=True)

    # Get the strings from the random characters
    random_strings = []
    start = 0
    for length in string_lengths:
        random_strings.append("".join(random_chars[start : start + length]))
        start += length

    assert len(random_strings) == dataset_size

    return random_strings


def _generate_dataset(
    words: Sequence[str], dataset_size: int, seed: int
) -> Tuple[Sequence[str], Sequence[str], Sequence[int]]:

    words_array = np.array(words)

    rng = np.random.default_rng(seed=seed)
    word_1_subset_indices = rng.choice(len(words), size=dataset_size, replace=True)
    word_2_subset_indices = rng.choice(len(words), size=dataset_size, replace=True)

    instructions = []
    labels = []
    for i, (word1, word2) in enumerate(
        zip(
            words_array[word_1_subset_indices],
            words_array[word_2_subset_indices],
        )
    ):
        if i % 1000 == 0:
            print(f"Generated {i} word_length examples so far, out of {dataset_size}")

        partial_instruction = CONTEXT_STRING.replace(FIRST_WORD, word1)
        instruction = partial_instruction.replace(SECOND_WORD, word2)

        label = 0 if len(word1) < len(word2) else 1

        instructions.append(instruction)
        labels.append(label)

    random_strings = _get_random_strings(dataset_size, rng)

    print(f"Generated {dataset_size} word_length examples total.")

    return instructions, random_strings, labels
