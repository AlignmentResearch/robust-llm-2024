import dataclasses
import numpy as np

from robust_llm.language_generators.tomita_base import TomitaBase


def is_all_tens(the_list: list[int]):
    assert len(the_list) > 0  # for simplicity don't allow empty

    # First make sure the list is even
    if not len(the_list) % 2 == 0:
        return False

    # Make sure that the list alternates 0 and 1, starting with 1, and ending with 0
    current_digit = 1
    for el in the_list:
        if not el == current_digit:
            return False
        current_digit = 1 - current_digit

    return True


@dataclasses.dataclass
class Tomita2(TomitaBase):  # 10*
    # Overrides
    def generate_true(self, num: int = 1):
        # Generate a string of ones of random length, from zero up to length n / 2
        assert num > 0
        assert isinstance(num, int)

        max_halflength = self.max_length // 2  # int

        num = self.rng.integers(
            low=1,  # for simplicity, don't allow empty string
            high=max_halflength + 1,
            size=(num,),
            dtype=np.int32,
        )
        return [
            " ".join("10" * el) for el in num
        ]  # put spaces between the digits for more natural tokenization

    def generate_false(self, num: int = 1):
        # Generate a random string of 0s and 1s of random length, from zero up to length n,
        # checking to make sure it's not all ones
        assert num > 0
        assert isinstance(num, int)

        def generate_one():
            num_digits = self.rng.integers(
                low=1, high=self.max_length + 1
            )  # don't allow empty
            digits = [1, 0]
            while is_all_tens(digits):  # this catches the empty list too
                digits = self.rng.integers(
                    low=0, high=2, size=(num_digits,), dtype=np.int8
                )
            return " ".join(
                [str(el) for el in digits]
            )  # put spaces between the digits for more natural tokenization

        all_strings = []
        for _ in range(num):  # I think this is hard to parallelize efficiently
            all_strings.append(generate_one())
        return all_strings


if __name__ == "__main__":
    tomita2 = Tomita2(max_length=10)
    train, val, test = tomita2.generate_dataset(10, 4, 4)
    print(train)
    print(val)
    print(test)
