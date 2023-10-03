import dataclasses
import numpy as np

from typing_extensions import override

from robust_llm.language_generators.tomita_base import TomitaBase


@dataclasses.dataclass
class Tomita1(TomitaBase):  # 1*
    name: str = "tomita1"

    @override
    def is_in_language(self, digits: list[int]) -> bool:
        assert len(digits) > 0  # for simplicity don't allow empty
        return all(digit == 1 for digit in digits)

    @override
    def generate_true(self, count: int = 1):
        """Generate a string of ones with random length
        between 1 and `self.max_length`, inclusive."""
        assert count > 0
        assert isinstance(count, int)

        string_lengths = self.rng.integers(
            low=1,  # for simplicity, don't allow empty string
            high=self.max_length + 1,
            size=(count,),
            dtype=np.int32,
        )
        return [
            " ".join("1" * el) for el in string_lengths
        ]  # put spaces between the digits for more natural tokenization

    @override
    def generate_false(self, count: int = 1):
        """Generate a random string of 0s and 1s with random length
        between one and `self.max_length` (inclusive),
        and rerunning until a string which is not all 1s is found.
        """
        assert count > 0
        assert isinstance(count, int)

        def generate_one():
            num_digits = self.rng.integers(
                low=1, high=self.max_length + 1
            )  # don't allow empty
            digits = [1]
            while self.is_in_language(digits):  # this catches the empty list too
                digits = self.rng.integers(
                    low=0, high=2, size=(num_digits,), dtype=np.int8
                ).tolist()
            return " ".join(
                [str(el) for el in digits]
            )  # put spaces between the digits for more natural tokenization

        all_strings = []
        for _ in range(count):  # I think this is hard to parallelize efficiently
            all_strings.append(generate_one())
        return all_strings
