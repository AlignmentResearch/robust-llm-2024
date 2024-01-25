from typing import Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
import pytest
from datasets.utils.typing import ListLike
from transformers.trainer_utils import PredictionOutput

from robust_llm.utils import (
    Dataset,
    get_incorrect_predictions,
    search_for_adversarial_examples,
)


class MockDataset(Dataset):
    def __init__(
        self,
        text: List[str],
        label: List[str],
    ):
        self.text = text
        self.label = label

    @property
    def num_rows(self) -> int:
        return len(self.text)

    @property
    def column_names(self) -> List[str]:
        return ["text", "label"]

    def _getitem(
        self, key: Union[int, slice, str, ListLike[int]], **kwargs
    ) -> Union[Dict, List]:
        assert not isinstance(key, list), "List indexing not supported"
        assert not isinstance(key, slice), "Slice indexing not supported"
        assert not isinstance(key, tuple), "Tuple indexing not supported"
        if isinstance(key, str):
            return getattr(self, key)
        return dict(text=self.text[key], label=self.label[key])

    def select(
        self,
        indices: Iterable,
        keep_in_memory: bool = False,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> Dataset:
        return MockDataset(
            text=[self.text[i] for i in indices], label=[self.label[i] for i in indices]
        )

    def shuffle(
        self,
        seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: Optional[bool] = None,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> Dataset:
        return self


class MockTrainer:
    def __init__(
        self,
        predictions: Mapping[str, List[float]],
        label_ids: Mapping[str, int],
    ):
        self.predictions = predictions
        self.label_ids = label_ids

    def predict(self, test_dataset: MockDataset) -> PredictionOutput:
        return PredictionOutput(
            predictions=np.array(
                [self.predictions[text] for text in test_dataset.text]
            ),
            label_ids=np.array([self.label_ids[text] for text in test_dataset.text]),
            metrics=None,
        )


@pytest.fixture
def correct_trainer():
    return MockTrainer(
        predictions={"text1": [0.9, 0.1], "text2": [0.1, 0.9], "text3": [0.2, 0.8]},
        label_ids={"text1": 0, "text2": 1, "text3": 1},
    )


@pytest.fixture
def incorrect_trainer():
    return MockTrainer(
        predictions={"text1": [0.9, 0.1], "text2": [0.1, 0.9], "text3": [0.8, 0.2]},
        label_ids={"text1": 0, "text2": 1, "text3": 1},
    )


@pytest.fixture
def dataset():
    return MockDataset(["text1", "text2", "text3"], ["non_adv", "adv", "non_adv"])


def test_get_incorrect_predictions(incorrect_trainer, dataset):
    # Run the function
    result = get_incorrect_predictions(incorrect_trainer, dataset)  # type: ignore

    # Check the results
    assert result["text"] == ["text3"]
    assert result["label"] == ["non_adv"]


def test_no_incorrect_predictions(correct_trainer, dataset):
    # Run the function
    result = get_incorrect_predictions(correct_trainer, dataset)  # type: ignore

    # Check the results
    assert result["text"] == []
    assert result["label"] == []


def test_search_for_adversarial_examples(incorrect_trainer, dataset):
    # Run the function
    result, num_searched = search_for_adversarial_examples(
        incorrect_trainer,  # type: ignore
        dataset,
        min_num_new_examples_to_add=1,
        max_num_search_for_adversarial_examples=10,
        adversarial_example_search_minibatch_size=2,
    )

    # Check the results
    assert result["text"] == ["text3"]
    assert result["label"] == ["non_adv"]
    assert num_searched == 4


def test_no_adversarial_examples(correct_trainer, dataset):
    # Run the function
    result, num_searched = search_for_adversarial_examples(
        correct_trainer,  # type: ignore
        dataset,
        min_num_new_examples_to_add=1,
        max_num_search_for_adversarial_examples=10,
        adversarial_example_search_minibatch_size=2,
    )

    # Check the results
    assert result["text"] == []
    assert result["label"] == []
    assert num_searched == 4
