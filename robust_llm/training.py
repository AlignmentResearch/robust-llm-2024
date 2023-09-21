import dataclasses
import evaluate
import numpy as np
import wandb

from datasets import Dataset

from transformers import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    Trainer,
)
from typing import Optional
from typing_extensions import override

from robust_llm.adversarial_trainer import (
    AdversarialTrainer,
    AdversarialTrainerLoggingCallback,
)
from robust_llm.language_generators.dataset_generator import load_adversarial_dataset
from robust_llm.utils import (
    get_incorrect_predictions,
    search_for_adversarial_examples,
    tokenize_dataset,
)


@dataclasses.dataclass
class Training:
    hparams: dict
    train_dataset: Dataset
    eval_dataset: Dataset
    model: PreTrainedModel
    train_epochs: int = 3
    eval_steps: int = 10
    logging_steps: int = 10
    trainer: Optional[Trainer] = None

    def __post_init__(self):
        self.metric = evaluate.load("accuracy")

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
            report_to=["wandb"],
        )

        trainer = Trainer(
            model=self.model,
            args=hf_training_args,
            train_dataset=self.train_dataset,  # type: ignore
            eval_dataset=self.eval_dataset,  # type: ignore
            compute_metrics=self.compute_metrics,
        )

        # Save the trainer as an attribute
        self.trainer = trainer

        return trainer

    def run_trainer(self):
        trainer = self.setup_trainer()

        self.log_datasets()

        trainer.evaluate(eval_dataset=self.eval_dataset)  # type: ignore
        trainer.train()

    def compute_metrics(self, eval_pred: EvalPrediction) -> dict:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        computed_metrics = self.metric.compute(
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
            self.trainer.eval_dataset["text"],  # type: ignore
            self.trainer.eval_dataset["label"],  # type: ignore
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
    random_sample_attack: bool
    min_num_adversarial_examples_to_add: int
    max_num_search_for_adversarial_examples: int
    adversarial_example_search_minibatch_size: int
    attack_dataset: Optional[Dataset] = None

    def __post_init__(self):
        super().__post_init__()

        self.language_generator_name = self.language_generator_name.lower()

        # Make sure that only one of brute force and random sample is set to true
        assert not (self.brute_force_attack and self.random_sample_attack)

    @override
    def setup_trainer(self) -> AdversarialTrainer:
        hf_training_args = TrainingArguments(
            output_dir="adversarial_trainer",
            num_train_epochs=self.train_epochs,
            eval_steps=self.eval_steps,
            evaluation_strategy="steps",
            logging_steps=self.logging_steps,
            report_to=["wandb"],
        )
        trainer = AdversarialTrainer(
            model=self.model,
            args=hf_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
        )
        trainer.add_callback(AdversarialTrainerLoggingCallback(self))

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

        elif self.random_sample_attack:
            raise NotImplementedError("Random sample attack not implemented yet.")

        else:
            # Just find mistakes in the eval set
            attack_dataset = self.eval_dataset

        self.attack_dataset = attack_dataset

        self.log_datasets()

        # Run the adversarial training loop
        for i in range(self.num_adversarial_training_rounds):
            print("Starting training round", i)

            # Train for "one round" (i.e., num_train_epochs) on the (eventually, adversarial example-augmented) train set
            # Note that the first round is just normal training on the train set
            # NOTE: this is where wandb.init() is called by default
            adversarial_trainer.train()
            adversarial_trainer.evaluate()

            incorrect_predictions = search_for_adversarial_examples(
                adversarial_trainer,
                attack_dataset,
                min_num_adversarial_examples_to_add=self.min_num_adversarial_examples_to_add,
                max_num_search_for_adversarial_examples=self.max_num_search_for_adversarial_examples,
                adversarial_example_search_minibatch_size=self.adversarial_example_search_minibatch_size,
            )

            # Check if we have perfect accuracy now. If so, we're done.
            if len(incorrect_predictions["text"]) == 0:
                print(
                    "Model got perfect accuracy on the adversarial dataset, so stopping adversarial training."
                )
                break

            print(f"Model made {len(incorrect_predictions['text'])} mistakes.")

            # Append the incorrect predictions to the table (adv training round, text, incorrect label, correct label)
            table = wandb.Table(
                columns=["example text", "predicted label", "correct label"]
            )
            for text_string, correct_label in zip(
                incorrect_predictions["text"],
                incorrect_predictions["label"],
            ):
                table.add_data(text_string, 1 - correct_label, correct_label)
            wandb.log({f"successful_attacks_after_round_{i}": table}, commit=False)

            # Add the incorrect predictions to the adversarial dataset
            # If we already added that incorrect prediction, don't add it again
            for text, true_label in zip(
                incorrect_predictions["text"],
                incorrect_predictions["label"],  # true label
            ):
                adversarial_trainer.adversarial_examples["text"].append(text)
                adversarial_trainer.adversarial_examples["label"].append(true_label)

    @override
    def log_datasets(self):
        super().log_datasets()

        to_log = {}

        if self.attack_dataset is None:
            raise ValueError(
                "self.trainer.attack_dataset should have been assigned by now, exiting..."
            )
        adversarial_table = wandb.Table(columns=["text", "label"])
        for text, label in zip(
            self.attack_dataset["text"],
            self.attack_dataset["label"],
        ):
            adversarial_table.add_data(text, label)
        to_log["brute_force_attack_dataset"] = adversarial_table

        wandb.log(to_log, commit=False)
