import dataclasses

import numpy as np
from typing_extensions import override

from robust_llm.dataset_management.tomita.tomita import Tomita


@dataclasses.dataclass
class Tomita4(Tomita):  # doesn't contain "000" as a substring
    name: str = "tomita4"

    def __post_init__(self):
        if self.max_length < 3:
            raise ValueError("max_length must be at least 3 for Tomita4")
        super().__post_init__()

    @override
    def is_in_language(self, digits: list[int]) -> bool:
        assert len(digits) > 0  # for simplicity don't allow empty

        # Check if the list contains three consecutive zeros
        for i in range(len(digits) - 2):
            if digits[i] == 0 and digits[i + 1] == 0 and digits[i + 2] == 0:
                return False

        return True

    @override
    def generate_true(self, count: int = 1):
        """Generate a random string that doesn't contain three consecutive zeros."""
        assert count > 0
        assert isinstance(count, int)

        def generate_one():
            num_digits = self.rng.integers(
                low=1, high=self.max_length + 1
            )  # don't allow empty

            # Generate digits one-by-one, making sure we don't have three
            # consecutive zeros
            digits = []
            for i in range(num_digits):
                if i < 2:
                    digits.append(self.rng.integers(low=0, high=2))
                else:
                    # If the last two digits are both 0, then we need to choose 1
                    # Otherwise, we can choose either 0 or 1
                    if digits[-2] == 0 and digits[-1] == 0:
                        digits.append(1)
                    else:
                        digits.append(self.rng.integers(low=0, high=2))

            assert self.is_in_language(digits)

            return " ".join(
                [str(el) for el in digits]
            )  # put spaces between the digits for more natural tokenization

        all_strings = []
        for _ in range(count):  # I think this is hard to parallelize efficiently
            all_strings.append(generate_one())
        return all_strings

    @override
    def generate_false(self, count: int = 1):
        assert count > 0
        assert isinstance(count, int)

        def generate_one():
            # Choose how many digits in our list
            num_extra_digits = self.rng.integers(
                low=0,
                high=self.max_length
                + 1
                - 3,  # how many we add on top of the "000" part
            )

            # Choose where to put the "000" in the digits
            zeros_location = self.rng.integers(low=0, high=num_extra_digits + 1)

            # Generate the digits
            digits = self.rng.integers(
                low=0, high=2, size=(num_extra_digits,), dtype=np.int8
            )

            # Insert the "000" into the digits
            digits = np.insert(digits, zeros_location, [0, 0, 0])

            assert not self.is_in_language(list(digits))

            return " ".join(
                [str(el) for el in digits]
            )  # put spaces between the digits for more natural tokenization

        all_strings = []
        for _ in range(count):  # I think this is hard to parallelize efficiently
            all_strings.append(generate_one())
        return all_strings


if __name__ == "__main__":
    tomita4 = Tomita4(max_length=10, seed=0)
    train, val, test = tomita4.generate_dataset(10, 4, 4)
    print(train)
    print(val)
    print(test)
