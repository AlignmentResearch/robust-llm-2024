from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robust_llm.training import AdversarialTraining, Training

from datasets import concatenate_datasets, Dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from typing_extensions import override

import wandb

from robust_llm.utils import get_incorrect_predictions


class TrainerLoggingCallback(TrainerCallback):
    def __init__(self, training: Training) -> None:
        super().__init__()
        self.training = training
        

class AdversarialTrainerLoggingCallback(TrainerCallback):
    def __init__(self, training: AdversarialTraining) -> None:
        super().__init__()
        self.training = training
        self.adversarial_training_round: int = 0

    @override
    def on_evaluate(self, args, state, control, **kwargs) -> None:
        
        if self.training.trainer is None:
            raise ValueError("self.training.trainer should have been initialized by now, exiting...")
        
        to_log = {}

        # Accuracy on the entire attack set
        if self.training.attack_dataset is not None:
            # Accuracy on all of the attack set (of which adversarial examples will be a subset)
            attack_set = self.training.attack_dataset

            incorrect_predictions_attack_set = get_incorrect_predictions(
                trainer=self.training.trainer, dataset=attack_set
            )

            to_log["attack_set_accuracy"] = 1 - len(
                incorrect_predictions_attack_set
            ) / len(attack_set)
        
        # Accuracy on adversarial examples only
        adversarial_examples = self.training.trainer.get_tokenized_adversarial_dataset()  # type: ignore

        if adversarial_examples is not None:
            incorrect_predictions_adversarial = get_incorrect_predictions(
                trainer=self.training.trainer, dataset=adversarial_examples
            )

            to_log["adversarial_examples_accuracy"] = 1 - len(
                incorrect_predictions_adversarial
            ) / len(adversarial_examples)

        # Accuracy on augmented train set (original train set + adversarial examples, so never None)
        augmented_train_set = self.training.trainer.get_augmented_training_set()  # type: ignore

        incorrect_predictions_augmented = get_incorrect_predictions(
            trainer=self.training.trainer, dataset=augmented_train_set
        )

        to_log["augmented_train_set_accuracy"] = 1 - len(
            incorrect_predictions_augmented
        ) / len(augmented_train_set)

        wandb.log(to_log, commit=False)

    @override
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # Log the round, the augmented train set, and the augmented train set size
        to_log = {}

        to_log["adversarial_training_round"] = self.adversarial_training_round

        train_dataset_plus_adv_examples = (
            self.training.trainer.get_augmented_training_set()  # type: ignore
        )

        table = wandb.Table(columns=["text", "label"])
        for text_string, correct_label in zip(
            train_dataset_plus_adv_examples["text"],
            train_dataset_plus_adv_examples["label"],
        ):
            table.add_data(text_string, correct_label)

        to_log[
            f"augmented_train_set_start_round_{self.adversarial_training_round}"
        ] = table

        to_log[f"augmented_train_set_size"] = len(train_dataset_plus_adv_examples)

        wandb.log(to_log, commit=False)

    @override
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.adversarial_training_round += 1
