import dataclasses
import evaluate
import numpy as np

from datasets import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from typing_extensions import override

from robust_llm.adversarial_trainer import AdversarialTrainer
from robust_llm.language_generators.dataset_generator import load_adversarial_dataset
from robust_llm.utils import get_incorrect_predictions, tokenize_dataset


@dataclasses.dataclass
class Training:
    """
    Manage training and evaluation of a model.
    """

    hparams: dict
    train_dataset: Dataset
    eval_dataset: Dataset
    model: AutoModelForSequenceClassification
    train_epochs: int = 3
    eval_steps: int = 150
    logging_steps: int = 150

    def __post_init__(self):
        self.metric = evaluate.load("accuracy")

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
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        return trainer

    def run_trainer(self):
        # Set up the trainer
        trainer = self.setup_trainer()

        # Perform an initial evaluation, then train
        trainer.evaluate(eval_dataset=self.eval_dataset)
        trainer.train()

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)


@dataclasses.dataclass(kw_only=True)
# TODO: make sure kw_only is not breaking anything.
# I put it there because of this:
# https://medium.com/@aniscampos/python-dataclass-inheritance-finally-686eaf60fbb5
class AdversarialTraining(Training):
    tokenizer: AutoTokenizer
    num_adversarial_training_rounds: int
    language_generator_name: str
    brute_force_attack: bool
    brute_force_length: int
    random_sample_attack: bool

    def __post_init__(self):
        super().__post_init__()

        self.language_generator_name = self.language_generator_name.lower()

        # Make sure that only one of brute force and random sample is set to true
        assert not (self.brute_force_attack and self.random_sample_attack)

    @override
    def setup_trainer(self):
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

        return trainer

    @override
    def run_trainer(self):
        # Set up the trainer
        adversarial_trainer = self.setup_trainer()

        # Prepare the attack dataset
        attack_dataset = None
        if self.brute_force_attack:
            # Load in the brute force test set
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

        # Run the adversarial training loop
        for _ in range(self.num_adversarial_training_rounds):
            # Train for "one round" (i.e., num_train_epochs) on the train set
            adversarial_trainer.train()

            # TODO: sample if we're doing random sample attack

            # Get the incorrect predictions
            incorrect_predictions = get_incorrect_predictions(
                adversarial_trainer, attack_dataset
            )

            # Check if we have perfect accuracy now. If so, we're done.
            if len(incorrect_predictions["text"]) == 0:
                print("Model got perfect accuracy, so stopping adversarial training.")
                break

            # Add the incorrect predictions to the adversarial dataset
            # If we already added that incorrect prediction, don't add it again
            for text, true_label in zip(
                incorrect_predictions["text"],
                incorrect_predictions["label"],  # true label
            ):
                adversarial_trainer.adversarial_examples["text"].append(text)
                adversarial_trainer.adversarial_examples["label"].append(true_label)
