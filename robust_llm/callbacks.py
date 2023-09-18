from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robust_llm.training import AdversarialTraining, Training

from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from typing_extensions import override

import wandb


class AdversarialTrainerDatasetManagementCallback(TrainerCallback):
    def __init__(self, training: AdversarialTraining) -> None:
        super().__init__()
        self.training = training
        self.adversarial_training_round: int = 0

    @override
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:

        # This is a bit wonky, since it'll keep updating the augmented train set
        # and be evaluating on something new after the start of each adversarial training round
        self.training.eval_dataset["augmented_train_set"] = self.training.trainer.get_augmented_training_set()  # type: ignore


class AdversarialTrainerLoggingCallback(TrainerCallback):
    def __init__(self, training: AdversarialTraining) -> None:
        super().__init__()
        self.training = training

    @override
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        to_log = {}

        to_log[
            "train/adversarial_training_round"
        ] = self.training.current_adversarial_training_round

        assert self.training.trainer is not None

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
            f"augmented_train_set_start_round_{self.training.current_adversarial_training_round}"
        ] = table

        to_log[f"train/augmented_train_set_size"] = len(train_dataset_plus_adv_examples)

        wandb.log(to_log, commit=False)
