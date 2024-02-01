from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robust_llm.training import AdversarialTraining

import wandb
from datasets import Dataset, concatenate_datasets
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from typing_extensions import override

from robust_llm.utils import get_overlap, tokenize_dataset


class AdversarialTrainer(Trainer):
    train_dataset: Dataset | None

    def __init__(self, **trainer_kwargs):
        super().__init__(**trainer_kwargs)

        # text_chunked is not needed for training.
        # Remove it so that it's possible to merge datasets later on.
        assert self.train_dataset is not None
        if "text_chunked" in self.train_dataset.features:
            self.train_dataset = self.train_dataset.remove_columns("text_chunked")

        self.new_examples: dict = {"text": [], "label": []}

    @override
    def get_train_dataloader(self):
        # This method is called at the start of each training loop, when
        # my_trainer.train() is called. In turn, the train_dataloader it returns
        # is called at the start of each training epoch
        # https://github.com/huggingface/transformers/blob/5a4f340df74b42b594aedf60199eea95cdb9bed0/src/transformers/trainer.py#L812

        old_train_set = self.train_dataset  # does this deep copy?
        self.train_dataset = self.get_augmented_training_set()
        train_dataloader_to_return = super().get_train_dataloader()
        self.train_dataset = old_train_set
        return train_dataloader_to_return

        # TODO: test this to make sure the dataloader pulls from the augmented dataset

    def get_augmented_training_set(self) -> Dataset:
        # Augment the train set with the new adversarial examples
        if len(self.new_examples["text"]) > 0:
            # Tokenize the new examples
            tokenized_new_examples = self.get_tokenized_adversarial_dataset()

            train_dataset_plus_adv_examples = concatenate_datasets(
                [
                    self.train_dataset,  # type: ignore
                    tokenized_new_examples,
                ]
            )

        else:
            train_dataset_plus_adv_examples = self.train_dataset

        return train_dataset_plus_adv_examples  # type: ignore

    def get_tokenized_adversarial_dataset(self) -> Dataset:
        assert len(self.new_examples["text"]) > 0

        # Tokenize the new examples
        assert self.tokenizer is not None
        tokenized_new_examples = Dataset.from_dict(
            tokenize_dataset(self.new_examples, self.tokenizer)
        )

        assert (
            self.train_dataset.features.type  # type: ignore
            == tokenized_new_examples.features.type
        )

        return tokenized_new_examples


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
        # and be evaluating on something new after the start of each adversarial
        # training round
        augmented_train_set = self.training.trainer.get_augmented_training_set()  # type: ignore  # noqa: E501
        self.training.eval_dataset["augmented_train_set"] = augmented_train_set


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

        _current_round = self.training.current_iterative_training_round
        to_log[f"augmented_train_set_start_round_{_current_round}"] = table

        _num_rows = train_dataset_plus_adv_examples.num_rows
        to_log["misc/augmented_train_set_size"] = _num_rows

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

        assert isinstance(self.training.trainer, AdversarialTrainer)
        augmented_train_set = self.training.trainer.get_augmented_training_set()

        # Record how much of the validation set is in the train set
        overlap = get_overlap(
            self.training.eval_dataset["validation"],
            augmented_train_set,
        )
        proportion_of_validation_in_train = (
            len(overlap) / self.training.eval_dataset["validation"].num_rows
        )
        to_log["misc/proportion_of_validation_in_train"] = (
            proportion_of_validation_in_train
        )

        wandb.log(to_log, commit=False)
