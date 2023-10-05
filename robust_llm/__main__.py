from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb
from robust_llm.language_generators import make_language_generator
from robust_llm.parsing import setup_argument_parser
from robust_llm.training import AdversarialTraining, Training
from robust_llm.utils import get_overlap, tokenize_dataset


def main():
    # Setup the argument parser for the language classification task
    # and parse in the command-line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Make the language generator to generate the strings in
    # and not in the chosen regular language
    language_generator = make_language_generator(
        args.language_generator, args.max_length
    )

    # Choose a model and a tokenizer
    model_name = "bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    print()
    print("train_size:", args.train_set_size)
    print("val_size:", args.val_set_size)
    print()

    train_set, val_set, _ = language_generator.generate_dataset(
        train_size=args.train_set_size,
        val_size=args.val_set_size,
        test_size=0,
    )

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
        training = AdversarialTraining(
            **base_training_args,
            num_adversarial_training_rounds=args.num_adversarial_training_rounds,
            tokenizer=tokenizer,
            language_generator_name=args.language_generator,
            brute_force_attack=args.brute_force_attack,
            brute_force_length=args.brute_force_length,
            random_sample_attack=args.random_sample_attack,
            min_num_adversarial_examples_to_add=args.min_num_adversarial_examples_to_add,
            max_num_search_for_adversarial_examples=args.max_num_search_for_adversarial_examples,
            adversarial_example_search_minibatch_size=args.adversarial_example_search_minibatch_size,
        )
    else:
        training = Training(
            **base_training_args,
        )

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
