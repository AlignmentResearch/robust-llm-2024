from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robust_llm.training import AdversarialTraining

from datasets import Dataset, concatenate_datasets
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from typing_extensions import override

import wandb
from robust_llm.utils import get_incorrect_predictions, tokenize_dataset


class AdversarialTrainer(Trainer):
    def __init__(self, **trainer_kwargs):
        super().__init__(**trainer_kwargs)

        self.adversarial_examples: dict = {"text": [], "label": []}

    @override
    def get_train_dataloader(self):
        # This method is called at the start of each training loop, when my_trainer.train() is called
        # In turn, the train_dataloader it returns is called at the start of each training epoch
        # https://github.com/huggingface/transformers/blob/5a4f340df74b42b594aedf60199eea95cdb9bed0/src/transformers/trainer.py#L812

        old_train_set = self.train_dataset  # does this deep copy?

        self.train_dataset = self.get_augmented_training_set()

        train_dataloader_to_return = super().get_train_dataloader()

        self.train_dataset = old_train_set

        return train_dataloader_to_return

        # TODO: test this to make sure the dataloader pulls from the augmented dataset

    def get_augmented_training_set(self) -> Dataset:
        # Augment the train set with the new adversarial examples
        if len(self.adversarial_examples["text"]) > 0:
            # Tokenize the new examples
            tokenized_adversarial_examples = self.get_tokenized_adversarial_dataset()

            train_dataset_plus_adv_examples = concatenate_datasets(
                [
                    self.train_dataset,  # type: ignore
                    tokenized_adversarial_examples,
                ]
            )

        else:
            train_dataset_plus_adv_examples = self.train_dataset

        return train_dataset_plus_adv_examples  # type: ignore

    def get_tokenized_adversarial_dataset(self) -> Dataset:
        assert len(self.adversarial_examples["text"]) > 0

        # Tokenize the new examples
        tokenized_adversarial_examples = Dataset.from_dict(
            tokenize_dataset(self.adversarial_examples, self.tokenizer)
        )

        assert (
            self.train_dataset.features.type  # type: ignore
            == tokenized_adversarial_examples.features.type
        )

        return tokenized_adversarial_examples


class AdversarialTrainerLoggingCallback(TrainerCallback):
    """
    Logs the accuracy on the attack set, adversarial examples, and augmented train set whenever evaluation happens.
    Also logs the augmented train set at the start of each training round.
    """

    def __init__(self, training: AdversarialTraining) -> None:
        super().__init__()
        self.training = training
        self.adversarial_training_round: int = 0

    @override
    def on_evaluate(self, args, state, control, **kwargs) -> None:
        if self.training.trainer is None:
            raise ValueError(
                "self.training.trainer should have been initialized by now, exiting..."
            )

        to_log = {}

        # Accuracy on the entire attack set
        if self.training.attack_dataset is not None:
            assert len(self.training.attack_dataset) > 0

            # Accuracy on all of the attack set (of which adversarial examples will be a subset)
            attack_set = self.training.attack_dataset

            incorrect_predictions_attack_set = get_incorrect_predictions(
                trainer=self.training.trainer, dataset=attack_set
            )

            to_log["attack_set_accuracy"] = 1 - len(
                incorrect_predictions_attack_set
            ) / len(attack_set)

        assert type(self.training.trainer) is AdversarialTrainer

        if len(self.training.trainer.adversarial_examples["text"]) > 0:
            # Accuracy on adversarial examples only
            adversarial_examples = self.training.trainer.get_tokenized_adversarial_dataset()  # type: ignore

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

        # TODO: make sure everything I wanted to log is actually appearing on wandb.

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
