from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robust_llm.adversarial_trainer import AdversarialTrainer


import numpy as np
import os

from torch.utils.data import DataLoader
from typing import Generator

from datasets import Dataset
from transformers import Trainer


def check_input_length(input_text, tokenizer):
    max_length = tokenizer.max_length

    inputs = tokenizer.encode(input_text, truncation=True, padding=False)
    input_length = len(inputs)

    if input_length > max_length:
        print(
            f"Warning: Input length ({input_length}) exceeds the maximum model length ({max_length})."
        )
        raise ValueError


def tokenize_dataset(dataset, tokenizer):
    # Padding seems necessary in order to avoid an error
    tokenized_data = tokenizer(dataset["text"], padding="max_length", truncation=True)
    return {"text": dataset["text"], "label": dataset["label"], **tokenized_data}


def get_overlap(
    smaller_dataset: dict[str, list[str]], larger_dataset: dict[str, list[str]]
) -> list[str]:
    return list(set(smaller_dataset["text"]).intersection(set(larger_dataset["text"])))


def print_overlaps(train_set, val_set, test_set):
    # How much of val set is in train set?
    train_val_overlap = get_overlap(smaller_dataset=val_set, larger_dataset=train_set)
    print("train val overlap size", len(train_val_overlap))
    print("train val overlap proportion", len(train_val_overlap) / len(val_set["text"]))
    print()

    # How much of test set is in train set?
    train_test_overlap = get_overlap(smaller_dataset=test_set, larger_dataset=train_set)
    print("train test overlap size", len(train_test_overlap))
    print(
        "train test overlap proportion", len(train_test_overlap) / len(test_set["text"])
    )
    print()

    # How much of test set is in val set?
    val_test_overlap = get_overlap(smaller_dataset=test_set, larger_dataset=val_set)
    print("val test overlap size", len(val_test_overlap))
    print("val test overlap proportion", len(val_test_overlap) / len(test_set["text"]))
    print()


def write_lines_to_file(lines, file_path):
    # If the folder doesn't exist yet, make one
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the file
    with open(file_path, "w") as afile:
        afile.writelines([line + "\n" for line in lines])


def get_incorrect_predictions(trainer: Trainer, dataset: Dataset) -> dict[str, list]:
    incorrect_predictions = {"text": [], "label": []}

    if not dataset:
        return incorrect_predictions

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
    min_num_adversarial_examples_to_add: int,
    max_num_search_for_adversarial_examples: int,
    adversarial_example_search_minibatch_size: int,
) -> dict[str, list]:
    number_searched = 0
    adversarial_examples = {"text": [], "label": []}

    for minibatch in yield_minibatch(
        attack_dataset, adversarial_example_search_minibatch_size
    ):
        print("current minibatch size:", len(minibatch["text"]))

        # Search for adversarial examples
        incorrect_predictions = get_incorrect_predictions(
            trainer=adversarial_trainer, dataset=minibatch
        )

        # Add these (if any) to the already-found adversarial examples
        adversarial_examples["text"] += incorrect_predictions["text"]
        adversarial_examples["label"] += incorrect_predictions["label"]

        # If we found enough adversarial examples, stop searching
        if len(adversarial_examples["text"]) >= min_num_adversarial_examples_to_add:
            break

        # If we passed the threshold of how many examples to search for, stop searching
        number_searched += adversarial_example_search_minibatch_size
        if number_searched >= max_num_search_for_adversarial_examples:
            print(
                f"Stopping search after {number_searched} examples searched (limit was {max_num_search_for_adversarial_examples})"
            )
            break

    return adversarial_examples


def yield_minibatch(
    dataset: Dataset, minibatch_size: int
) -> Generator[Dataset, None, None]:
    shuffled_dataset = dataset.shuffle()

    # Yield minibatches of the given size
    for i in range(0, shuffled_dataset.num_rows, minibatch_size):
        upper_limit = min(i + minibatch_size, shuffled_dataset.num_rows)
        yield shuffled_dataset.select(range(i, upper_limit))
