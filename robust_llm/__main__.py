from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from robust_llm.language_generators import make_language_generator_from_args
from robust_llm.parsing import setup_argument_parser
from robust_llm.training import Training, AdversarialTraining
from robust_llm.utils import print_overlaps, tokenize_dataset


def main():
    # Setup the argument parser for the language classification task
    # and parse in the command-line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Make the language generator to generate the strings in
    # and not in the chosen regular language
    language_generator = make_language_generator_from_args(args)

    # Choose a model and a tokenizer
    model_name = "bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    print()
    print("train_size:", args.train_set_size)
    print("val_size:", args.val_set_size)
    print("test_size:", args.test_set_size)
    print()

    train_set, val_set, test_set = language_generator.generate_dataset(
        train_size=args.train_set_size,
        val_size=args.val_set_size,
        test_size=args.test_set_size,
    )

    print_overlaps(train_set, val_set, test_set)

    print("Tokenizing datasets...")
    tokenized_train_dataset = Dataset.from_dict(tokenize_dataset(train_set, tokenizer))
    tokenized_val_dataset = Dataset.from_dict(tokenize_dataset(val_set, tokenizer))
    tokenized_test_dataset = Dataset.from_dict(tokenize_dataset(test_set, tokenizer))

    base_training_args = {
        "hparams": {},
        "train_dataset": tokenized_train_dataset,
        "eval_dataset": tokenized_val_dataset,
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

    # Perform the training
    training.run_trainer()

    # Print overlaps again so it's stored in the output logged in wandb
    # TODO: log this directly to wandb instead of printing it
    print_overlaps(train_set, val_set, test_set)


if __name__ == "__main__":
    main()
