from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from robust_llm.utils import get_overlap, tokenize_dataset


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


class AdversarialTrainerDatasetManagementCallback(TrainerCallback):
    def __init__(self, training: AdversarialTraining) -> None:
        super().__init__()
        self.training = training
        self.adversarial_training_round: int = 0

    @override
    def on_train_begin(  # type: ignore[misc]
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
    def on_train_begin(  # type: ignore[misc]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        to_log: dict[str, Any] = {}

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

        to_log[
            f"misc/augmented_train_set_size"
        ] = train_dataset_plus_adv_examples.num_rows

        wandb.log(to_log, commit=False)

    @override
    def on_train_end(  # type: ignore[misc]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        
        if self.training.use_probabilistic_robustness_check:
            return
        
        to_log: dict[str, Any] = {}

        augmented_train_set = self.training.trainer.get_augmented_training_set()  # type: ignore

        # Record how much of the brute force attack set is in the train set
        overlap = get_overlap(
            self.training.eval_dataset["brute_force_attack_dataset"].to_dict(),  # type: ignore
            augmented_train_set.to_dict(),  # type: ignore
        )
        proportion_of_attack_in_train = (
            len(overlap)
            / self.training.eval_dataset["brute_force_attack_dataset"].num_rows
        )
        to_log["misc/proportion_of_attack_in_train"] = proportion_of_attack_in_train

        wandb.log(to_log, commit=False)
