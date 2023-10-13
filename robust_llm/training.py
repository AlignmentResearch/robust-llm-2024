import dataclasses
from typing import Optional

import evaluate
import numpy as np
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
from robust_llm.language_generators.dataset_generator import load_adversarial_dataset
from robust_llm.utils import (
    search_for_adversarial_examples,
    tokenize_dataset,
)


@dataclasses.dataclass
class Training:
    hparams: dict
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
        )

    def setup_trainer(self):
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

    def run_trainer(self):
        trainer = self.setup_trainer()

        self.log_datasets()

        trainer.evaluate(eval_dataset=self.eval_dataset)  # type: ignore
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

        # Save the training dataset to a wandb table
        if self.trainer is None:
            raise ValueError(
                "self.trainer should have been assigned by now, exiting..."
            )
        if self.trainer.train_dataset is None:
            raise ValueError(
                "self.trainer.train_dataset should have been assigned by now, exiting..."
            )
        train_table = wandb.Table(columns=["text", "label"])
        for text, label in zip(
            self.trainer.train_dataset["text"],
            self.trainer.train_dataset["label"],
        ):
            train_table.add_data(text, label)
        to_log["train_dataset"] = train_table

        # Save the eval dataset to a wandb table
        if self.trainer.eval_dataset is None:
            raise ValueError(
                "self.trainer.eval_dataset should have been assigned by now, exiting..."
            )
        eval_table = wandb.Table(columns=["text", "label"])
        for text, label in zip(
            self.trainer.eval_dataset["eval"]["text"],
            self.trainer.eval_dataset["eval"]["label"],
        ):
            eval_table.add_data(text, label)
        to_log["eval_dataset"] = eval_table

        wandb.log(to_log, commit=False)


@dataclasses.dataclass(kw_only=True)
# TODO: make sure kw_only is not breaking anything.
# I put it there because of this:
# https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
class AdversarialTraining(Training):
    tokenizer: PreTrainedTokenizerBase
    num_adversarial_training_rounds: int
    language_generator_name: str
    brute_force_attack: bool
    brute_force_length: int
    min_num_adversarial_examples_to_add: int
    max_num_search_for_adversarial_examples: int
    adversarial_example_search_minibatch_size: int
    attack_dataset: Optional[Dataset] = None
    current_adversarial_training_round: int = 0
    skip_first_training_round: bool = False
    use_probabilistic_robustness_check: bool = False

    def __post_init__(self):
        super().__post_init__()

        assert type(self.eval_dataset) is dict
        assert "eval" in self.eval_dataset

        # Standardize the language generator name
        self.language_generator_name = self.language_generator_name.lower()

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
    def run_trainer(self):
        # Set up the trainer
        adversarial_trainer = self.setup_trainer()

        # Prepare the attack dataset
        attack_dataset = None
        if self.brute_force_attack:
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
            # Just find mistakes in the eval set
            assert "eval" in self.eval_dataset
            attack_dataset = self.eval_dataset["eval"]

        # Log the datasets
        self.log_datasets()

        # Run the adversarial training loop
        for i in range(self.num_adversarial_training_rounds):
            print("Starting training round", i)
            self.current_adversarial_training_round = i

            to_log = {}

            # Train for "one round" (i.e., num_train_epochs) on the (eventually, adversarial example-augmented) train set
            # Note that the first round is just normal training on the train set
            # NOTE: this is where wandb.init() is called by default
            if i == 0 and self.skip_first_training_round:
                print("Skipping first round of training...")
            else:
                adversarial_trainer.train()

            (
                incorrect_predictions,
                number_examples_searched,
            ) = search_for_adversarial_examples(
                adversarial_trainer,
                attack_dataset,
                min_num_adversarial_examples_to_add=self.min_num_adversarial_examples_to_add,
                max_num_search_for_adversarial_examples=self.max_num_search_for_adversarial_examples,
                adversarial_example_search_minibatch_size=self.adversarial_example_search_minibatch_size,
            )

            to_log["misc/number_examples_searched"] = number_examples_searched

            # Check if we have perfect accuracy now. If so, we're done.
            if len(incorrect_predictions["text"]) == 0:
                print(
                    f"~~~Model got perfect accuracy on the {number_examples_searched} examples tried, so stopping adversarial training.~~~"
                )
                break

            print(f"Model made {len(incorrect_predictions['text'])} mistakes.")

            # Append the incorrect predictions to the table (text, correct label)
            successful_attacks_table = wandb.Table(columns=["text", "correct label"])
            for text_string, correct_label in zip(
                incorrect_predictions["text"],
                incorrect_predictions["label"],
            ):
                successful_attacks_table.add_data(text_string, correct_label)
            to_log[f"successful_attacks_after_round_{i}"] = successful_attacks_table
            to_log[f"misc/number_successful_attacks"] = len(
                incorrect_predictions["text"]
            )

            # Add the incorrect predictions to the adversarial dataset
            for text, true_label in zip(
                incorrect_predictions["text"],
                incorrect_predictions["label"],  # true label
            ):
                adversarial_trainer.adversarial_examples["text"].append(text)
                adversarial_trainer.adversarial_examples["label"].append(true_label)

            # Save the adversarial dataset to the eval sets
            tokenized_adversarial_examples = Dataset.from_dict(
                tokenize_dataset(
                    adversarial_trainer.adversarial_examples, self.tokenizer
                )
            )
            self.eval_dataset["adversarial_examples"] = tokenized_adversarial_examples

            wandb.log(to_log, commit=False)

    @override
    def log_datasets(self):
        # First log the train and eval sets
        super().log_datasets()

        if self.use_probabilistic_robustness_check:
            return

        to_log = {}

        # Save the adversarial training dataset to a wandb table
        if self.eval_dataset["brute_force_attack_dataset"] is None:
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
