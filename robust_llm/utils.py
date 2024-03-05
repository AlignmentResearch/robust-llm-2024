from __future__ import annotations

import abc
import os
import random
from argparse import Namespace
from typing import TYPE_CHECKING, Any, Generator, Iterator, Optional, Protocol, Sequence

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.nn.parameter import Parameter

from robust_llm.configs import ExperimentConfig

if TYPE_CHECKING:
    from robust_llm.trainer import AdversarialTrainer

import wandb
from datasets import Dataset
from transformers import PretrainedConfig, PreTrainedTokenizerBase, Trainer


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
    padding: str = "max_length",
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


def get_overlap(smaller_dataset: Dataset, larger_dataset: Dataset) -> list[str]:
    return list(set(smaller_dataset["text"]).intersection(set(larger_dataset["text"])))


def write_lines_to_file(lines: list[str], file_path: str) -> None:
    # If the folder doesn't exist yet, make one
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the file
    with open(file_path, "w") as afile:
        afile.writelines([line + "\n" for line in lines])


def get_incorrect_predictions(trainer: Trainer, dataset: Dataset) -> dict[str, list]:
    incorrect_predictions = {"text": [], "label": []}  # type: ignore

    assert dataset is not None
    assert dataset.num_rows > 0

    outputs = trainer.predict(test_dataset=dataset)  # type: ignore
    logits = outputs.predictions
    labels = outputs.label_ids

    # Extract the incorrect predictions
    predictions = np.argmax(logits, axis=-1)
    incorrect_indices = np.where(predictions != labels)[0].astype(int)

    # Return the incorrectly predicted examples, along with their true labels
    for incorrect_index in incorrect_indices:
        incorrect_predictions["text"].append(dataset["text"][incorrect_index])
        incorrect_predictions["label"].append(dataset["label"][incorrect_index])

    return incorrect_predictions


def search_for_adversarial_examples(
    adversarial_trainer: AdversarialTrainer,
    attack_dataset: Dataset,
    min_num_new_examples_to_add: int,
    max_num_search_for_adversarial_examples: int,
    adversarial_example_search_minibatch_size: int,
) -> tuple[dict[str, list], int]:
    """
    Iterates through a shuffled `attack_dataset` for examples that the model
    misclassifies, and returns them (and their correct labels) in a dict.

    Args:
        adversarial_trainer (AdversarialTrainer): trainer from which we use the
            model to make predictions.
        attack_dataset (Dataset): dataset in which to search for adversarial
            examples.
        min_num_new_examples_to_add (int): the minimum number of examples to
            return. The function may return fewer than this examples if
            `max_num_search_for_adversarial_examples` has been exceeded, or if the
            entire dataset contains fewer than the desired count of adversarial
            examples.
        max_num_search_for_adversarial_examples (int): the maximum number of
            examples to search over from `attack_dataset`. The function may return
            up to one minibatch's worth of examples more than this number.

    Returns:
        dict[str, list]: A dict containing the adversarial examples and their
            true labels.
        int: the number of examples searched through
    """

    number_searched = 0
    adversarial_examples = {"text": [], "label": []}  # type: ignore

    for minibatch in yield_minibatch(
        attack_dataset, adversarial_example_search_minibatch_size
    ):
        number_searched += adversarial_example_search_minibatch_size

        # Search for adversarial examples
        incorrect_predictions = get_incorrect_predictions(
            trainer=adversarial_trainer, dataset=minibatch
        )

        # Add these (if any) to the already-found adversarial examples
        adversarial_examples["text"] += incorrect_predictions["text"]
        adversarial_examples["label"] += incorrect_predictions["label"]

        # If we found enough adversarial examples, stop searching
        if len(adversarial_examples["text"]) >= min_num_new_examples_to_add:
            break

        # If we passed the threshold of how many examples to search over, stop searching
        if number_searched >= max_num_search_for_adversarial_examples:
            print(
                f"Stopping search after {number_searched} examples searched (limit was {max_num_search_for_adversarial_examples})"  # noqa: E501
            )
            break

    return adversarial_examples, number_searched


def yield_minibatch(
    dataset: Dataset, minibatch_size: int
) -> Generator[Dataset, None, None]:
    shuffled_dataset = dataset.shuffle()

    # Yield minibatches of the given size
    for i in range(0, shuffled_dataset.num_rows, minibatch_size):
        upper_limit = min(i + minibatch_size, shuffled_dataset.num_rows)
        yield shuffled_dataset.select(range(i, upper_limit))


def log_dataset_to_wandb(dataset: Dataset, dataset_name: str) -> None:
    dataset_table = wandb.Table(columns=["text", "label"])

    for text, label in zip(
        dataset["text"],
        dataset["label"],
    ):
        dataset_table.add_data(text, label)

    wandb.log({dataset_name: dataset_table}, commit=False)


def log_config_to_wandb(config: ExperimentConfig) -> None:
    """Logs the job config to wandb."""
    if not wandb.run:
        raise ValueError("wandb should have been initialized by now, exiting...")
    config_yaml = yaml.load(OmegaConf.to_yaml(config), Loader=yaml.FullLoader)
    wandb.run.summary["experiment_yaml"] = config_yaml


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
    # Use wandb run ID as a unique identifier.
    assert wandb.run is not None
    return f"{wandb.run.id}_from_{base_processed}"


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


class FakeModel:
    """Fake model class used in tests."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def num_labels(self) -> int:
        return 2

    @property
    def config(self):
        return Namespace(pad_token_id=1, eos_token_id=2)

    def __call__(self, input: torch.Tensor) -> Namespace:
        return Namespace(
            logits=torch.rand((input.shape[0], input.shape[1], self.vocab_size))
        )
