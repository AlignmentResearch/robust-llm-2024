import numpy as np
import os

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
    train_val_overlap = get_overlap(smaller_set=val_set, larger_set=train_set)
    print("train val overlap size", len(train_val_overlap))
    print("train val overlap proportion", len(train_val_overlap) / len(val_set["text"]))
    print()

    # How much of test set is in train set?
    train_test_overlap = get_overlap(smaller_set=test_set, larger_set=train_set)
    print("train test overlap size", len(train_test_overlap))
    print(
        "train test overlap proportion", len(train_test_overlap) / len(test_set["text"])
    )
    print()

    # How much of test set is in val set?
    val_test_overlap = get_overlap(smaller_set=test_set, larger_set=val_set)
    print("val test overlap size", len(val_test_overlap))
    print("val test overlap proportion", len(val_test_overlap) / len(test_set["text"]))
    print()


def write_lines_to_file(lines, file_path):
    # If the folder doesn't exist yet, make one
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the file
    with open(file_path, "w") as afile:
        afile.writelines(lines)


def get_incorrect_predictions(trainer: Trainer, dataset: Dataset) -> dict[str, list]:
    outputs = trainer.predict(test_dataset=dataset)
    logits = outputs.predictions
    labels = outputs.label_ids

    # Extract the incorrect predictions
    predictions = np.argmax(logits, axis=-1)
    incorrect_indices = np.where(predictions != labels)[0].astype(int)

    # Return the incorrectly predicted examples, along with their true labels
    incorrect_predictions = {"text": [], "label": []}
    for incorrect_index in incorrect_indices:
        incorrect_predictions["text"].append(dataset["text"][incorrect_index])
        incorrect_predictions["label"].append(dataset["label"][incorrect_index])

    return incorrect_predictions
