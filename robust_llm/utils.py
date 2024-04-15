from __future__ import annotations

import abc
import os
import random
import uuid
from argparse import Namespace
from typing import Any, Dict, Iterator, Optional, Protocol, Sequence, Sized

import torch
import torch.utils.data
from accelerate import Accelerator
from datasets import Dataset
from torch.nn.parameter import Parameter
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import SequenceClassifierOutput


class LanguageModel(Protocol):
    """Protocol for a language model.

    This is used to unify:
    1) the defence pipeline's `DefendedModel` class
    2) HuggingFace's `transformers.PreTrainedModel` class.
    """

    @abc.abstractmethod
    def __call__(self, **inputs) -> Any:
        """Run the inputs through self.model, with required safety considerations."""
        pass

    @property
    @abc.abstractmethod
    def config(self) -> PretrainedConfig:
        pass

    @abc.abstractmethod
    def to(self, *args, **kwargs) -> LanguageModel:
        pass

    @abc.abstractmethod
    def forward(self, **inputs):
        pass

    @abc.abstractmethod
    def train(self) -> LanguageModel:
        pass

    @abc.abstractmethod
    def eval(self) -> LanguageModel:
        pass

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abc.abstractmethod
    def training(self) -> bool:
        pass

    @abc.abstractmethod
    def parameters(self) -> Iterator[Parameter]:
        pass


def tokenize_dataset(
    dataset: Dataset | dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    padding: str = "do_not_pad",
    truncation: bool = True,
    return_tensors: Optional[str] = None,
) -> dict[str, Any]:
    if isinstance(dataset, Dataset):
        dataset = {key: dataset[key] for key in dataset.features.keys()}
    # Padding seems necessary in order to avoid an error
    tokenized_data = tokenizer(
        dataset["text"],
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )
    return {**dataset, **tokenized_data}


def get_unique_overlap(smaller_dataset: Dataset, larger_dataset: Dataset) -> list[str]:
    """Calculate the overlap between the "text" columns of two datasets.

    Assumes that there are no duplicate entries in either of the datasets.
    If there are duplicate entries, they will be removed before the overlap
    is calculated, and you will get a smaller calculated overlap than
    you would expect.

    Args:
        smaller_dataset: A dataset with a "text" column.
        larger_dataset: A dataset with a "text" column.

    Returns:
        A list of unique texts that appear in both datasets.
    """
    return list(set(smaller_dataset["text"]).intersection(set(larger_dataset["text"])))


def write_lines_to_file(lines: list[str], file_path: str) -> None:
    # If the folder doesn't exist yet, make one
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the file
    with open(file_path, "w") as afile:
        afile.writelines([line + "\n" for line in lines])


def ask_for_confirmation(prompt: str) -> bool:
    while True:
        answer = input(prompt + " (y/n) ")
        if answer.lower() == "y":
            return True
        elif answer.lower() == "n":
            return False
        else:
            print("Please answer with 'y' or 'n'.")


def div_maybe_nan(a: int, b: int) -> float:
    if b == 0:
        return float("nan")
    return a / b


def make_unique_name_to_save(base_name_or_path: str) -> str:
    """Get a unique name used for saving the model."""
    base_processed = base_name_or_path.replace("/", "_")
    # Create a long random id.
    id = uuid.uuid1().hex
    return f"{id}_from_{base_processed}"


def get_randint_with_exclusions(
    high: int, exclusions: Sequence[int], rng: Optional[random.Random] = None
) -> int:
    """Get a random integer from [0, `high`), excluding the integers in `exclusions`."""
    assert len(exclusions) < high, "Too many excluded values!"
    MAX_NUM_ITERS = 1000

    value: Optional[int] = None
    randint_fn = rng.randint if rng else random.randint

    # Replaced a previous implementation where we explicitly create a set of allowed
    # values. It was super slow when `high` was large and `exclusions` was small.
    iter = 0
    while value is None or value in exclusions:
        value = randint_fn(0, high - 1)
        iter += 1
        if iter > MAX_NUM_ITERS:
            raise ValueError("Too many iterations!")

    return value


def prepare_model_with_accelerate(
    accelerator: Accelerator, model: PreTrainedModel
) -> PreTrainedModel:
    model = accelerator.prepare(model)
    # When using FSDP, there is some lazy initialization that happens. Enforce it here
    # to avoid issues from lack of proper initialization (e.g. when accessing embedding
    # layer in GCG).
    _ = model(torch.tensor([[0]], device=accelerator.device))

    return model


class FakeModelForSequenceClassification:
    """Fake model class used in tests."""

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def num_labels(self) -> int:
        return 2

    @property
    def config(self):
        return Namespace(
            pad_token_id=1,
            eos_token_id=2,
            task_specific_params={},
            id2label={0: "LABEL_0", 1: "LABEL_1"},
        )

    def can_generate(self) -> bool:
        return False

    def eval(self) -> FakeModelForSequenceClassification:
        return self

    def to(self, *args, **kwargs) -> FakeModelForSequenceClassification:
        return self

    def forward(
        self, input_ids: torch.Tensor, *args, **kwargs
    ) -> SequenceClassifierOutput:
        """Since we are mimicking a sequence classification model, we return logits in
        the shape (batch_size, num_labels)."""
        return SequenceClassifierOutput(
            logits=torch.rand(input_ids.shape[0], self.num_labels),  # type: ignore
        )

    def __call__(
        self, input_ids: torch.Tensor, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        return self.forward(input_ids, *args, **kwargs)


def equal_ignore_padding(x: torch.Tensor, y: torch.Tensor, pad_token_id: int) -> bool:
    """Checks if two 1D tensors are equal, ignoring padding at the end."""
    while x[-1] == pad_token_id:
        x = x[:-1]
    while y[-1] == pad_token_id:
        y = y[:-1]
    return x.equal(y)


class FakeClassifierWithPositiveList(FakeModelForSequenceClassification):
    """Fake classification model with a pre-defined list of positive examples."""

    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, positives: Sequence[torch.Tensor]
    ):
        self.tokenizer = tokenizer
        self.positives = positives

    def forward(
        self, input_ids: torch.Tensor, *args, **kwargs
    ) -> SequenceClassifierOutput:
        logits = []
        pad_token_id = self.tokenizer.pad_token_id
        assert pad_token_id is not None
        for x in input_ids:
            logits.append(
                [0.0, 1.0]
                if any(
                    [equal_ignore_padding(x, y, pad_token_id) for y in self.positives]
                )
                else [1.0, 0.0]
            )
        return SequenceClassifierOutput(logits=torch.tensor(logits))  # type: ignore


class BalancedSampler(torch.utils.data.Sampler[int]):
    """A sampler that alternates between regular and adversarial data.

    Note that regardless of the relative sizes, regular and adversarial data will be
    sampled in equal proportions. Adversarial data points might be sampled many times
    during one loop (if there are more regular data than adversarial data).
    """

    def __init__(self, regular_data: Sized, adversarial_data: Sized) -> None:
        self.regular_data = regular_data
        self.adversarial_data = adversarial_data

        self.regular_data_sampler = torch.utils.data.RandomSampler(regular_data)
        self.adversarial_data_sampler = torch.utils.data.RandomSampler(
            adversarial_data, num_samples=len(regular_data)
        )

    def __iter__(self) -> Iterator[int]:
        n = len(self.regular_data)
        iter_regular = iter(self.regular_data_sampler)
        iter_adversarial = iter(self.adversarial_data_sampler)
        for _ in range(n):
            yield next(iter_regular)
            yield n + next(iter_adversarial)

    def __len__(self) -> int:
        return 2 * len(self.regular_data)
