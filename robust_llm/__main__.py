import dataclasses

from datasets import Dataset
from simple_parsing import ArgumentParser
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from robust_llm.language_generators.tomita_base import TomitaBase
from robust_llm.language_generators.tomita1 import Tomita1
from robust_llm.language_generators.tomita2 import Tomita2
from robust_llm.language_generators.tomita4 import Tomita4

from robust_llm.training import Training

BERT_CONTEXT_LENGTH = 512
BUFFER = 5
CONTEXT_LENGTH = BERT_CONTEXT_LENGTH - 3 - BUFFER  # 3 for special tokens


def make_language_generator_from_args(args):
    if args.language_generator == "Tomita1":
        language_generator = Tomita1(args.max_length)
    elif args.language_generator == "Tomita2":
        language_generator = Tomita2(args.max_length)
    elif args.language_generator == "Tomita4":
        language_generator = Tomita4(args.max_length)
    else:
        raise ValueError(f"Unknown language generator: {args.language_generator}")

    return language_generator


def tokenize_dataset(dataset, tokenizer):
    # Padding seems necessary in order to avoid an error
    tokenized_data = tokenizer(dataset["text"], padding="max_length", truncation=True)
    return {"text": dataset["text"], "label": dataset["label"], **tokenized_data}


def get_overlap(smaller_set, larger_set):
    overlap = []
    for s in smaller_set["text"]:
        if s in larger_set["text"]:
            overlap.append(s)
    return overlap


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--language_generator",
        choices=["Tomita1", "Tomita2", "Tomita4"],
        default="Tomita1",
        help="Choose the regular language to use (Tomita1, Tomita2, Tomita4). "
             "Defaults to Tomita1.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=500,
        help="The maximum length of the strings to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed to use for the random number generator used to make the dataset.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="The number of epochs to train for.",
    )

    # Parse the command-line arguments.
    args = parser.parse_args()

    language_generator = make_language_generator_from_args(args)

    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_size = 1000
    val_size = 1000
    test_size = 200

    print()
    print("train_size", train_size)
    print("val_size", val_size)
    print("test_size", test_size)
    print()

    train_set, val_set, test_set = language_generator.generate_dataset(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    # How much of val set is in train set?
    train_val_overlap = get_overlap(smaller_set=val_set, larger_set=train_set)
    print("train val overlap size", len(train_val_overlap))
    print("train val overlap proportion", len(train_val_overlap) / len(val_set["text"]))
    print()

    # How much of test set is in train set?
    train_test_overlap = get_overlap(smaller_set=test_set, larger_set=train_set)
    print("train test overlap size", len(train_test_overlap))
    print("train test overlap proportion", len(train_test_overlap) / len(test_set["text"]))
    print()

    # How much of test set is in val set?
    val_test_overlap = get_overlap(smaller_set=test_set, larger_set=val_set)
    print("val test overlap size", len(val_test_overlap))
    print("val test overlap proportion", len(val_test_overlap) / len(test_set["text"]))
    print()

    print("Tokenizing datasets...")
    tokenized_train_dataset = Dataset.from_dict(tokenize_dataset(train_set, tokenizer))
    tokenized_val_dataset = Dataset.from_dict(tokenize_dataset(val_set, tokenizer))
    tokenized_test_dataset = Dataset.from_dict(tokenize_dataset(test_set, tokenizer))

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased", num_labels=2
    )

    training = Training(
        hparams={},
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        model=model,
        train_epochs=args.num_train_epochs,
    )
    training.run_trainer()


if __name__ == "__main__":
    # my_language_generator = Tomita1(
    #     CONTEXT_LENGTH, seed=41
    # )
    main()
