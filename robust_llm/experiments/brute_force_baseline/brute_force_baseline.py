from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb
from robust_llm.language_generators.dataset_generator import load_adversarial_dataset
from robust_llm.parsing import setup_argument_parser
from robust_llm.training import Training
from robust_llm.utils import get_overlap, tokenize_dataset


def main():
    # Setup the argument parser for the language classification task
    # and parse in the command-line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Choose a model and a tokenizer
    model_name = "bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    print()
    print("baseline up to length:", args.brute_force_length)
    print("proportion of brute force set:", args.proportion)
    print()

    brute_force_dataset = load_adversarial_dataset(
        args.language_generator, args.brute_force_length
    )
    tokenized_brute_force_dataset = Dataset.from_dict(
        tokenize_dataset(brute_force_dataset, tokenizer)
    )
    shuffled_brute_force_dataset = tokenized_brute_force_dataset.shuffle()
    train_set = shuffled_brute_force_dataset.select(
        range(int(args.proportion * len(tokenized_brute_force_dataset)))
    )
    val_set = brute_force_dataset

    print("Tokenizing datasets...")
    tokenized_train_dataset = Dataset.from_dict(tokenize_dataset(train_set, tokenizer))
    tokenized_val_dataset = Dataset.from_dict(tokenize_dataset(val_set, tokenizer))

    base_training_args = {
        "hparams": {},
        "train_dataset": tokenized_train_dataset,
        "eval_dataset": {"eval": tokenized_val_dataset},
        "model": model,
        "train_epochs": args.num_train_epochs,
    }

    # Set up the training environment
    if args.adversarial_training:
        raise ValueError(
            "This is a brute force baseline -- adversarial training is not supported."
        )
    else:
        training = Training(
            **base_training_args,
        )

    # Log the training arguments to wandb
    if not wandb.run:
        raise ValueError("wandb should have been initialized by now, exiting...")
    for k, v in sorted(list(vars(args).items())):
        wandb.run.summary[f"args/{k}"] = v

    # Log the train-val overlap to wandb
    if args.train_set_size > 0 and args.val_set_size > 0:
        if not wandb.run:
            raise ValueError("wandb should have been initialized by now, exiting...")
        train_val_overlap = get_overlap(smaller_dataset=val_set, larger_dataset=train_set)  # type: ignore
        wandb.run.summary["train_val_overlap_size"] = len(train_val_overlap)
        wandb.run.summary["train_val_overlap_over_train_set_size"] = len(
            train_val_overlap
        ) / len(train_set["text"])
        wandb.run.summary["train_val_overlap_over_val_set_size"] = len(
            train_val_overlap
        ) / len(val_set["text"])

    # Perform the training
    training.run_trainer()


if __name__ == "__main__":
    main()
