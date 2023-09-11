import dataclasses
import numpy as np

from typing_extensions import override

from robust_llm.language_generators.tomita_base import TomitaBase


@dataclasses.dataclass
class Tomita2(TomitaBase):  # (10)*
    name: str = "tomita2"

    @override
    def is_in_language(self, digits: list[int]) -> bool:
        assert len(digits) > 0  # for simplicity don't allow empty

        # First make sure the list is even
        if not len(digits) % 2 == 0:
            return False

        # Make sure that the list alternates 0 and 1, starting with 1, and ending with 0
        current_digit = 1
        for el in digits:
            if not el == current_digit:
                return False
            current_digit = 1 - current_digit

        return True

    # Overrides
    def generate_true(self, how_many_to_generate: int = 1):
        # Generate a string of ones of random length, from zero up to length n / 2
        assert how_many_to_generate > 0
        assert isinstance(how_many_to_generate, int)

        max_halflength = self.max_length // 2  # int

        half_lengths = self.rng.integers(
            low=1,  # for simplicity, don't allow empty string
            high=max_halflength + 1,
            size=(how_many_to_generate,),
            dtype=np.int32,
        )
        return [
            " ".join("10" * el) for el in half_lengths
        ]  # put spaces between the digits for more natural tokenization

    def generate_false(self, how_many_to_generate: int = 1):
        assert how_many_to_generate > 0
        assert isinstance(how_many_to_generate, int)

        def generate_one():
            num_digits = self.rng.integers(
                low=1, high=self.max_length + 1
            )  # don't allow empty
            digits = [1, 0]
            while self.is_in_language(digits):  # this catches the empty list too
                digits = self.rng.integers(
                    low=0, high=2, size=(num_digits,), dtype=np.int8
                )
            return " ".join(
                [str(el) for el in digits]
            )  # put spaces between the digits for more natural tokenization

        all_strings = []
        for _ in range(
            how_many_to_generate
        ):  # I think this is hard to parallelize efficiently
            all_strings.append(generate_one())
        return all_strings


if __name__ == "__main__":
    tomita2 = Tomita2(
        max_length=10
    )  # 10 is fine since this is just to test if it looks reasonable
    train, val, test = tomita2.generate_dataset(
        10, 4, 4
    )  # these are fine since just need to ensure it looks ok in terms of diversity and correctness
    print(train)
    print(val)
    print(test)
