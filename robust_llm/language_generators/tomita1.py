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
    def generate_true(self, num: int = 1):
        # Generate a string of ones of random length, from one up to length num
        assert num > 0
        assert isinstance(num, int)

        string_lengths = self.rng.integers(
            low=1,  # for simplicity, don't allow empty string
            high=self.max_length + 1,
            size=(num,),
            dtype=np.int32,
        )
        return [
            " ".join("1" * el) for el in string_lengths
        ]  # put spaces between the digits for more natural tokenization

    @override
    def generate_false(self, num: int = 1):
        # Generate a random string of 0s and 1s of random length,
        # from zero up to length num, checking to make sure it's not all ones
        assert num > 0
        assert isinstance(num, int)

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
        for _ in range(num):  # I think this is hard to parallelize efficiently
            all_strings.append(generate_one())
        return all_strings
