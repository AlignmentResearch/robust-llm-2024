import abc
import dataclasses

import numpy as np
from datasets import Dataset

from robust_llm.file_utils import compute_dataset_path
from robust_llm.utils import write_lines_to_file


def all_binary_strings_of_length(length: int) -> list[str]:
    """
    Returns a list of all binary strings of `length`, with spaces separating bits.

    For example, calling with `length` of 2 returns:
    ["0 0", "0 1", "1 0", "1 1"]
    """
    if length < 1:
        raise ValueError("n should be greater than or equal to 1.")

    strings = []
    for i in range(2**length):
        binary_string = bin(i)[2:].zfill(length)
        spaced_binary_string = " ".join(binary_string)
        strings.append(spaced_binary_string)

    return strings


@dataclasses.dataclass
class Tomita:
    seed: int = 42
    name: str = "tomitabase"
    max_length: int = 5  # short for debugging purposes

    # Model specific configuration. Defaults to BERT.
    context_length: int = 512
    context_buffer: int = 10
    final_special_token_length: int = 3

    @property
    def largest_string_the_model_can_handle(self) -> int:
        return (
            self.context_length - self.context_buffer - self.final_special_token_length
        )

    def __post_init__(self):
        self.rng = np.random.default_rng(seed=self.seed)
        assert self.max_length <= self.largest_string_the_model_can_handle

    @abc.abstractmethod
    def is_in_language(self, digits: list[int]) -> bool:
        pass

    @abc.abstractmethod
    def generate_true(self, count: int = 1) -> list[str]:
        pass

    @abc.abstractmethod
    def generate_false(self, count: int = 1) -> list[str]:
        pass

    def generate_dataset(
        self, train_size: int = 10_000, validation_size: int = 3000, test_size: int = 0
    ) -> tuple[Dataset, Dataset, Dataset]:
        assert train_size > 0
        assert validation_size >= 0
        assert test_size >= 0
        assert train_size % 2 == 0
        assert validation_size % 2 == 0
        assert test_size % 2 == 0

        total_size = train_size + validation_size + test_size
        true_size = total_size // 2
        false_size = total_size // 2

        trues = self.generate_true(count=true_size)
        falses = self.generate_false(count=false_size)

        half_train_size = train_size // 2
        half_validation_size = validation_size // 2
        half_test_size = test_size // 2

        train_set: dict[str, list[str]]
        train_set = label_and_shuffle(
            trues[:half_train_size], falses[:half_train_size], rng=self.rng
        )

        # The test and validation sets areallowed to be empty,
        # which would otherwise cause bugs for this form of indexing.
        val_set: dict[str, list[str]] = {"text": [], "label": []}
        if validation_size > 0:
            val_set = label_and_shuffle(
                trues[half_train_size : half_train_size + half_validation_size],
                falses[half_train_size : half_train_size + half_validation_size],
                rng=self.rng,
            )
        test_set: dict[str, list[str]] = {"text": [], "label": []}
        if test_size > 0:
            test_set = label_and_shuffle(
                trues[-half_test_size:],
                falses[-half_test_size:],
                rng=self.rng,
            )

        return (
            Dataset.from_dict(train_set),
            Dataset.from_dict(val_set),
            Dataset.from_dict(test_set),
        )

    def _classify_saved_binary_strings(
        self, binary_strings: list[str]
    ) -> tuple[list[str], list[str]]:
        trues = []
        falses = []
        for string in binary_strings:
            int_version = [int(el) for el in string.split(" ")]
            if self.is_in_language(int_version):
                trues.append(string)
            else:
                falses.append(string)

        return trues, falses

    def make_complete_dataset(self, length: int = 10) -> None:
        trues, falses = self._classify_saved_binary_strings(
            all_binary_strings_of_length(length)
        )

        # Save the trues and falses as trues_i and falses_i in the 'self.name' folder
        write_lines_to_file(
            trues, f"{compute_dataset_path()}/tomita/{self.name}/trues_{length}.txt"
        )
        write_lines_to_file(
            falses, f"{compute_dataset_path()}/tomita/{self.name}/falses_{length}.txt"
        )

    def string_to_digit_list(self, string: str) -> list[int]:
        """Converts a space separated digit string into a list of ints."""
        return [int(c) for c in string.split(" ")]


def label_and_shuffle(
    trues: list[str],
    falses: list[str],
    rng: np.random.Generator,
) -> dict[str, list[str]]:
    """Given a list of true and false strings, returns a labelled, shufffled dataset.

    Args:
        trues: list of strings belonging to the regular langauge.
        falses: list of strings not belonging to the regular language.
        rng: NumPy random number generator.

    Returns:
        A dictionary with keys "text" and "label". The "text" field
        contains a shuffled list of strings, and the "label" field
        contains their corresponding integer labels
        (1 if in the regular language, 0 if not in the regular language).
    """
    labelled_trues = [(el, 1) for el in trues]
    labelled_falses = [(el, 0) for el in falses]
    labelled_dataset = labelled_trues + labelled_falses
    rng.shuffle(labelled_dataset)  # type: ignore
    data, labels = zip(*labelled_dataset)
    return {"text": list(data), "label": list(labels)}
