import dataclasses
import numpy as np
import re

from robust_llm.language_generators.tomita_base import TomitaBase


PATTERN = re.compile("0*1*0*1*")


@dataclasses.dataclass
class Tomita7(TomitaBase):  # 0*1*0*1*
    # Overrides
    def is_in_language(self, the_list: list[int]) -> bool:
        assert len(the_list) > 0  # for simplicity don't allow empty

        string_list = [str(el) for el in the_list]

        # Check if the list satisfies the regular expression 1*0*1*0*
        return bool(PATTERN.fullmatch("".join(string_list)))

    # Overrides
    def generate_true(self, num: int = 1):
        # Generate a random string that satisfies 0*1*0*1*
        # We do this by choosing the four cutoff locations, and then filling in the digits in between
        # The cutoff locations are sampled from Beta(2, 5) so that we ideally get more than a couple
        assert num > 0
        assert isinstance(num, int)

        def generate_one():
            # Let each "*" be of length 0 with 25%
            star_values = self.rng.choice([0, 1], p=[0.25, 0.75], size=(4,))

            cutoffs_0_to_1 = self.rng.beta(a=2, b=5, size=(4,))
            cutoffs = cutoffs_0_to_1 * self.max_length
            cutoffs = [int(np.round(el)) for el in cutoffs]

            digit_list = []
            for i, star_value in enumerate(star_values):
                current_digit = i % 2
                if star_value == 0:
                    continue
                elif star_value == 1:
                    list_to_merge = [current_digit] * cutoffs[i]
                    digit_list += list_to_merge
                else:
                    raise ValueError("star_value must be 0 or 1")

            # Cut the list to the correct length
            digit_list = digit_list[: self.max_length]

            # We don't accept the empty string
            if digit_list == []:
                digit_list.append(self.rng.choice([0, 1]))

            assert self.is_in_language(digit_list)

            return " ".join(
                [str(el) for el in digit_list]
            )  # put spaces between the digits for more natural tokenization

        all_strings = []
        for _ in range(num):  # I think this is hard to parallelize efficiently
            all_strings.append(generate_one())
        return all_strings

    # Overrides
    def generate_false(self, num: int = 1):
        # Generate a random string of 0s and 1s of random length, from zero up to length n,
        # checking to make sure it's not all ones
        assert num > 0
        assert isinstance(num, int)

        def generate_one():
            # print("low was", np.log(4))
            # print("high was", np.log(self.max_length + 1))
            # Sample in a log-uniform way
            num_digits = self.rng.uniform(
                low=np.log(4), high=np.log(self.max_length + 1)
            )  # don't allow empty
            num_digits = int(np.round(np.exp(num_digits)))

            if num_digits == 4:
                return "1 0 1 0"  # only option when n = 4

            digits = [0]
            while self.is_in_language(digits):  # this catches the empty list too
                # print("digits were", digits)
                digits = self.rng.integers(
                    low=0, high=2, size=(num_digits,), dtype=np.int8
                )

            assert not self.is_in_language(digits)

            return " ".join(
                [str(el) for el in digits]
            )  # put spaces between the digits for more natural tokenization

        all_strings = []
        for _ in range(num):  # I think this is hard to parallelize efficiently
            all_strings.append(generate_one())
        return all_strings


if __name__ == "__main__":
    tomita7 = Tomita7(max_length=10, seed=0)
    train, val, test = tomita7.generate_dataset(10, 4, 4)
    print(train)
    print(val)
    print(test)
