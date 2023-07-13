import dataclasses
import numpy as np


def is_all_ones(the_list: list[int]):
    assert len(the_list) > 0  # for simplicity don't allow empty
    return all(digit == 1 for digit in the_list)


@dataclasses.dataclass
class Tomita1:
    # 1*
    max_length: int
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(seed=self.seed)

    def label_and_shuffle(self, trues: list[str], falses: list[str]):
        labelled_trues = [(el, 1) for el in trues]
        labelled_falses = [(el, 0) for el in falses]
        labelled_dataset = labelled_trues + labelled_falses
        self.rng.shuffle(labelled_dataset)
        data, labels = zip(*labelled_dataset)
        return {"text": list(data), "label": list(labels)}

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
            while is_all_ones(digits):  # this catches the empty list too
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

    def generate_dataset(self, train_size=10_000, val_size=3000, test_size=3000):
        assert train_size > 0
        assert val_size > 0
        assert test_size > 0
        assert train_size % 2 == 0
        assert val_size % 2 == 0
        assert test_size % 2 == 0

        total_size = train_size + val_size + test_size
        true_size = int(total_size / 2)
        false_size = int(total_size / 2)

        trues = self.generate_true(num=true_size)
        falses = self.generate_false(num=false_size)

        half_train_size = int(train_size / 2)
        half_val_size = int(val_size / 2)
        half_test_size = int(test_size / 2)

        train_set = self.label_and_shuffle(
            trues[:half_train_size], falses[:half_train_size]
        )
        val_set = self.label_and_shuffle(
            trues[half_train_size : half_train_size + half_val_size],
            falses[half_train_size : half_train_size + half_val_size],
        )
        test_set = self.label_and_shuffle(
            trues[-half_test_size:], falses[-half_test_size:]
        )

        return train_set, val_set, test_set
