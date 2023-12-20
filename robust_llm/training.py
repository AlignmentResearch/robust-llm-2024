import dataclasses
from datetime import date
from typing import Optional

import evaluate
import numpy as np
import wandb.util
from datasets import Dataset
from transformers import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from typing_extensions import override

import wandb
from robust_llm.adversarial_trainer import (
    AdversarialTrainer,
    AdversarialTrainerDatasetManagementCallback,
    AdversarialTrainerLoggingCallback,
)
from robust_llm.callbacks import CrossTrainRunStepRecordingWandbCallback
from robust_llm.dataset_management.tomita.tomita_base import TomitaBase
from robust_llm.dataset_management.tomita.tomita_dataset_generator import (
    load_adversarial_dataset,
)
from robust_llm.utils import (
    search_for_adversarial_examples,
    tokenize_dataset,
    yield_minibatch,
)


@dataclasses.dataclass
class Training:
    hparams: dict
    experiment_name: str
    run_name: str
    job_type: str
    train_dataset: Dataset
    eval_dataset: dict[str, Dataset]
    model: PreTrainedModel
    train_epochs: int = 3
    eval_steps: int = 10
    logging_steps: int = 10
    trainer: Optional[Trainer] = None

    def __post_init__(self):
        accuracy = evaluate.load("accuracy")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")

        self.metrics = evaluate.combine([accuracy, precision, recall, f1])

        # TODO: it would be nice to not have to do this manually
        # I'm concerned that by initializing wandb here, we're losing
        # information that is stored when we let the trainer
        # automatically initialize wandb
        # https://docs.wandb.ai/guides/integrations/huggingface#customize-wandbinit
        wandb.init(
            project="robust-llm",
            group=self.experiment_name,
            job_type=self.job_type,
            name=self.run_name,
        )

    def setup_trainer(self) -> Trainer:
        hf_training_args = TrainingArguments(
            output_dir="test_trainer",
            num_train_epochs=self.train_epochs,
            eval_steps=self.eval_steps,
            evaluation_strategy="steps",
            logging_steps=self.logging_steps,
        )

        trainer = Trainer(
            model=self.model,
            args=hf_training_args,
            train_dataset=self.train_dataset,  # type: ignore
            eval_dataset=self.eval_dataset,  # type: ignore
            compute_metrics=self.compute_metrics,
        )
        trainer.add_callback(CrossTrainRunStepRecordingWandbCallback)

        self.trainer = trainer

        return trainer

    def run_trainer(self) -> None:
        trainer = self.setup_trainer()

        self.log_datasets()

        trainer.evaluate(eval_dataset=self.eval_dataset["validation"])  # type: ignore

        if self.train_epochs <= 0:
            print(f"Not training, since train_epochs={self.train_epochs}.")
            return

        trainer.train()

    def compute_metrics(self, eval_preds: EvalPrediction) -> dict:
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        computed_metrics = self.metrics.compute(
            predictions=predictions, references=labels
        )

        if computed_metrics is None:
            raise ValueError("computed_metrics is None, exiting...")

        return computed_metrics

    def log_datasets(self) -> None:
        """Save the training and eval datasets to wandb."""
        to_log = {}

        if self.trainer is None:
            raise ValueError(
                "self.trainer should have been assigned by now, exiting..."
            )
        if self.trainer.train_dataset is None:
            raise ValueError(
                "self.trainer.train_dataset should have been assigned by now, exiting..."
            )
        if self.trainer.eval_dataset is None:
            raise ValueError(
                "self.trainer.eval_dataset should have been assigned by now, exiting..."
            )

        in_train = "label" in self.trainer.train_dataset.column_names  # type: ignore
        in_eval = "label" in self.trainer.eval_dataset["validation"].column_names  # type: ignore
        assert in_train == in_eval

        if "label" in self.trainer.train_dataset.column_names:  # type: ignore
            train_table = wandb.Table(columns=["text", "label"])
            for text, label in zip(
                self.trainer.train_dataset["text"], self.trainer.train_dataset["label"]
            ):
                train_table.add_data(text, label)
            to_log["train_dataset"] = train_table
        else:
            train_table = wandb.Table(columns=["text"])
            for text in self.trainer.train_dataset["text"]:
                train_table.add_data(text)
            to_log["train_dataset"] = train_table

        label_in_validation = "label" in self.trainer.eval_dataset["validation"].column_names  # type: ignore
        if label_in_validation:
            eval_table = wandb.Table(columns=["text", "label"])
            for text, label in zip(
                self.trainer.eval_dataset["validation"]["text"],
                self.trainer.eval_dataset["validation"]["label"],
            ):
                eval_table.add_data(text, label)
            to_log["validation_dataset"] = eval_table
        else:
            eval_table = wandb.Table(columns=["text"])
            for text in self.trainer.eval_dataset["validation"]["text"]:
                eval_table.add_data(text)
            to_log["validation_dataset"] = eval_table

        wandb.log(to_log, commit=False)


@dataclasses.dataclass(kw_only=True)
# TODO: make sure kw_only is not breaking anything.
# I put it there because of this:
# https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
class AdversarialTraining(Training):
    """
    Creates an AdversarialTrainer wrapped for logging, etc.

    Parameters:
        tokenizer:
            Huggingface tokenizer used for tokenizing the dataset. Should match the model we're using.
        num_iterative_training_rounds:
            One round of adversarial training involves first finding some number of adversarial examples,
            adding them to an "augmented train set", and training on that for some number of epochs.
        dataset_type:
            The type of dataset to use. Either "tomita" or "tensor_trust".
        language_generator:
            The language generator which should be created to generate datapoints for training and evaluation.
            Only relevant in the Tomita setting.
        brute_force_attack:
            Whether to use a "brute force attack" to generate adversarial examples. This means testing on all possible examples up to a given length.
        brute_force_length:
            The maximum string length to use for the brute force attack. Note that the brute force dataset size grows as 2^length.
        min_num_new_examples_to_add:
            When searching for adversarial examples in the brute force attack, we usually don't stop looking until we surpass this number.
            If we search the entire brute force dataset and don't find enough examples, we do stop looking.
            If we surpass max_num_search_for_adversarial_examples during the search, we also do stop looking.
        max_num_search_for_adversarial_examples:
            When searching for adversarial examples in the brute force dataset, we stop looking if we search more or equal to this number of examples.
            In practice we'll often search a few more than this number since we do the search in minibatches.
        adversarial_example_search_minibatch_size:
            The number of datapoints to consider at once when searching for adversarial examples.
        skip_first_training_round:
            Whether to skip the first round of training. Useful for doing "exclusively" adversarial training.
        use_probabilistic_robustness_check:
            Whether to determine model robustness by randomly selecting some examples from the brute force dataset and testing only on those,
            rather than the default of checking against the entire brute force dataset.
        non_adversarial_baseline:
            If true, don't train on adversarial examples, just train on random examples, whether or not the models gets them right.
    """

    tokenizer: PreTrainedTokenizerBase
    num_iterative_training_rounds: int
    dataset_type: str
    language_generator: Optional[TomitaBase]
    brute_force_attack: bool
    brute_force_length: int
    min_num_new_examples_to_add: int
    max_num_search_for_adversarial_examples: int
    adversarial_example_search_minibatch_size: int
    skip_first_training_round: bool = False
    use_probabilistic_robustness_check: bool = False
    non_adversarial_baseline: bool = False

    def __post_init__(self):
        super().__post_init__()

        assert type(self.eval_dataset) is dict
        assert "validation" in self.eval_dataset

        # Standardize the language generator name
        if self.language_generator is None:
            self.language_generator_name = (
                "(no language generator, tensor trust setting)"
            )
        else:
            self.language_generator_name: str = self.language_generator.name

        self.current_iterative_training_round: int = 0

    @override
    def setup_trainer(self) -> AdversarialTrainer:
        hf_training_args = TrainingArguments(
            output_dir="adversarial_trainer",
            num_train_epochs=self.train_epochs,
            eval_steps=self.eval_steps,
            evaluation_strategy="steps",
            logging_steps=self.logging_steps,
        )
        trainer = AdversarialTrainer(
            model=self.model,
            args=hf_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )
        trainer.add_callback(CrossTrainRunStepRecordingWandbCallback)
        trainer.add_callback(AdversarialTrainerLoggingCallback(self))
        trainer.add_callback(AdversarialTrainerDatasetManagementCallback(self))

        # Save the trainer as an attribute
        self.trainer = trainer

        return trainer

    @override
    def run_trainer(self) -> None:
        # Set up the trainer
        adversarial_trainer = self.setup_trainer()

        # Prepare the attack dataset
        attack_dataset = None
        if self.brute_force_attack:
            if self.dataset_type not in ["tomita"]:
                raise ValueError(
                    f"Brute force attack not yet supported in dataset type {self.dataset_type}, exiting..."
                )

            brute_force_dataset = load_adversarial_dataset(
                self.language_generator_name, self.brute_force_length
            )
            tokenized_brute_force_dataset = Dataset.from_dict(
                tokenize_dataset(brute_force_dataset, self.tokenizer)
            )
            attack_dataset = tokenized_brute_force_dataset

            if not self.use_probabilistic_robustness_check:
                # Save the attack dataset as one of the datasets to do eval on
                self.eval_dataset["brute_force_attack_dataset"] = attack_dataset

        else:
            # Just find mistakes in the validation set
            assert "validation" in self.eval_dataset
            attack_dataset = self.eval_dataset["validation"]

        # Log the datasets
        self.log_datasets()

        # Run the adversarial training loop
        for i in range(self.num_iterative_training_rounds):
            print("Starting training round", i)
            self.current_iterative_training_round = i

            # Train for "one round" (i.e., num_train_epochs) on the (eventually, adversarial example-augmented) train set
            # Note that the first round is just normal training on the train set
            # NOTE: this is where wandb.init() is called by default
            if i == 0 and self.skip_first_training_round:
                print("Skipping first round of training...")
            else:
                if self.train_epochs == 0:
                    raise ValueError(
                        "Adversarial training should be done with >0 train epochs, exiting..."
                    )
                adversarial_trainer.train()

            incorrect_predictions: dict[str, list[str]]
            number_examples_searched: int

            print("Searching for mistakes...")
            (
                incorrect_predictions,
                number_examples_searched,
            ) = search_for_adversarial_examples(
                adversarial_trainer,
                attack_dataset,
                min_num_new_examples_to_add=self.min_num_new_examples_to_add,
                max_num_search_for_adversarial_examples=self.max_num_search_for_adversarial_examples,
                adversarial_example_search_minibatch_size=self.adversarial_example_search_minibatch_size,
            )

            print(f"Model made {len(incorrect_predictions['text'])} mistakes.")

            examples_to_actually_add_to_train_set = incorrect_predictions

            if self.non_adversarial_baseline:
                num_examples_to_add = (
                    self.min_num_new_examples_to_add
                    + self.adversarial_example_search_minibatch_size // 2
                )
                print(
                    f"Non-adversarial baseline: NOT adding those mistakes, instead adding the first {num_examples_to_add} random examples..."
                )
                examples_to_actually_add_to_train_set = next(
                    yield_minibatch(attack_dataset, num_examples_to_add)
                )

            wandb.log(
                {
                    "train/iterative_training_round": self.current_iterative_training_round,
                    "misc/number_examples_searched": number_examples_searched,
                    "misc/number_successful_attacks": len(
                        incorrect_predictions["text"]
                    ),
                },
                commit=False,
            )

            # Check if we have perfect accuracy now. If so, we're done.
            if len(incorrect_predictions["text"]) == 0:
                print(
                    f"~~~In round {i} of adversarial training, model got perfect accuracy on the {number_examples_searched} examples tried, so stopping adversarial training.~~~"
                )
                break

            # Log the successful attacks and the examples to add to the training set
            to_log = {}
            successful_attacks_table = wandb.Table(columns=["text", "correct label"])
            for text_string, correct_label in zip(
                incorrect_predictions["text"], incorrect_predictions["label"]
            ):
                successful_attacks_table.add_data(text_string, correct_label)
            to_log[f"successful_attacks_after_round_{i}"] = successful_attacks_table
            actual_examples_added_table = wandb.Table(columns=["text", "correct label"])
            for text_string, correct_label in zip(
                examples_to_actually_add_to_train_set["text"],
                examples_to_actually_add_to_train_set["label"],
            ):
                actual_examples_added_table.add_data(text_string, correct_label)
            to_log[
                f"examples_added_to_training_set_after_round_{i}"
            ] = actual_examples_added_table
            wandb.log(to_log, commit=False)

            # Save the new examples to the adversarial trainer
            for text, true_label in zip(
                examples_to_actually_add_to_train_set["text"],
                examples_to_actually_add_to_train_set["label"],  # true label
            ):
                adversarial_trainer.new_examples["text"].append(text)
                adversarial_trainer.new_examples["label"].append(true_label)

            # Save the adversarial dataset as an eval set
            tokenized_new_examples = Dataset.from_dict(
                tokenize_dataset(adversarial_trainer.new_examples, self.tokenizer)
            )
            self.eval_dataset[
                "all_examples_added_during_iterative_training"
            ] = tokenized_new_examples

    @override
    def log_datasets(self) -> None:
        # First log the train and evaluation sets
        super().log_datasets()

        if self.use_probabilistic_robustness_check:
            return

        to_log = {}

        # Save the adversarial training dataset to a wandb table
        if self.eval_dataset.get("brute_force_attack_dataset", None) is None:
            raise ValueError(
                "self.trainer.attack_dataset should have been assigned by now, exiting..."
            )
        adversarial_table = wandb.Table(columns=["text", "label"])
        for text, label in zip(
            self.eval_dataset["brute_force_attack_dataset"]["text"],
            self.eval_dataset["brute_force_attack_dataset"]["label"],
        ):
            adversarial_table.add_data(text, label)
        to_log["brute_force_attack_dataset"] = adversarial_table

        wandb.log(to_log, commit=False)
