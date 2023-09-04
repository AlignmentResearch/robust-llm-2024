import dataclasses
import numpy as np

from robust_llm.language_generators.tomita_base import TomitaBase


@dataclasses.dataclass
class Tomita1(TomitaBase):  # 1*
    # Overrides
    def is_in_language(self, the_list: list[int]) -> bool:
        assert len(the_list) > 0  # for simplicity don't allow empty
        return all(digit == 1 for digit in the_list)


    # Overrides
    def generate_true(self, num: int = 1):
        # Generate a string of ones of random length, from zero up to length n
        assert num > 0
        assert isinstance(num, int)

        num = self.rng.integers(
            low=1,  # for simplicity, don't allow empty string
            high=self.max_length + 1,
            size=(num,),
            dtype=np.int32,
        )
        return [
            " ".join("1" * el) for el in num
        ]  # put spaces between the digits for more natural tokenization

    # Overrides
    def generate_false(self, num: int = 1):
        # Generate a random string of 0s and 1s of random length, from zero up to length n,
        # checking to make sure it's not all ones
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
                )
            return " ".join(
                [str(el) for el in digits]
            )  # put spaces between the digits for more natural tokenization

        all_strings = []
        for _ in range(num):  # I think this is hard to parallelize efficiently
            all_strings.append(generate_one())
        return all_strings
    
    def sort_saved_binary_strings(self, binary_strings: list):
        trues = []
        falses = []
        for string in binary_strings:
            if "0" in string:
                falses.append(string)
            else:
                trues.append(string)

        return trues, falses
    

if __name__ == "__main__":
    t = Tomita1(50)  
    for i in range(3, 10):
        with open(f"robust_llm/datasets/{i}.txt", 'r') as afile:
            big_list_of_strings = []
            for line in afile:
                big_list_of_strings.append(line.rstrip())

        trues, falses = t.sort_saved_binary_strings(big_list_of_strings)
        print("trues", trues)
        print("falses", falses)
        break