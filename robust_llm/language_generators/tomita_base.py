import abc
import dataclasses
import numpy as np


@dataclasses.dataclass
class TomitaBase:
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

    @abc.abstractmethod
    def generate_true(self, num: int = 1):
        pass

    @abc.abstractmethod
    def generate_false(self, num: int = 1):
        pass

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
