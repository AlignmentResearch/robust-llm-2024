import abc
import dataclasses
import numpy as np

from robust_llm.utils import write_lines_to_file


BERT_CONTEXT_LENGTH = 512
BUFFER = 10
FINAL_SPECIAL_TOKEN_LENGTH = 3
MAX_BERT_CONTEXT_LENGTH = BERT_CONTEXT_LENGTH - BUFFER - FINAL_SPECIAL_TOKEN_LENGTH


@dataclasses.dataclass
class TomitaBase:
    max_length: int = MAX_BERT_CONTEXT_LENGTH
    seed: int = 42
    name: str = "tomitabase"

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
    def is_in_language(self, string: str) -> bool:
        pass

    @abc.abstractmethod
    def generate_true(self, num: int = 1) -> list[str]:
        pass

    @abc.abstractmethod
    def generate_false(self, num: int = 1) -> list[str]:
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

    def sort_saved_binary_strings(self, binary_strings: list):
        trues = []
        falses = []
        for string in binary_strings:
            int_version = [int(el) for el in string.split(" ")]
            if self.is_in_language(int_version):
                trues.append(string)
            else:
                falses.append(string)

        return trues, falses

    def make_complete_dataset(self, length: int = 10):
        with open(f"robust_llm/datasets/{length}.txt", "r") as afile:
            big_list_of_strings = []
            for line in afile:
                big_list_of_strings.append(line.rstrip())

        trues, falses = self.sort_saved_binary_strings(big_list_of_strings)

        # Save the trues and falses as trues_i and falses_i in the tomita1 folder
        write_lines_to_file(
            trues,
            f"/home/dev/robust-llm/robust_llm/datasets/{self.name}/trues_{length}.txt",
        )
        write_lines_to_file(
            falses,
            f"/home/dev/robust-llm/robust_llm/datasets/{self.name}/falses_{length}.txt",
        )
