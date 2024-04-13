from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from robust_llm.logging_utils import log_dataset_to_wandb

if TYPE_CHECKING:
    from robust_llm.training import AdversarialTraining

import torch.utils.data
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

from robust_llm.utils import BalancedSampler, get_overlap, tokenize_dataset


class TrainerWithBatchSizeStoring(Trainer):
    """A Trainer that also stores the batch size of the current batch.

    This is necessary because when we do logging, we want to know not
    only how many batches we've seen, but also how many datapoints that
    corresponds to. In particular, the final batch will usually be
    smaller than the others, so in order to not over-count, we need
    to manually record how many datapoints were in a given batch.
    """

    def __init__(self, **trainer_kwargs):
        super().__init__(**trainer_kwargs)
        self._current_batch_size: int = -1

    @override
    def training_step(  # type: ignore[misc]
        self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        loss = super().training_step(model=model, inputs=inputs)
        self._current_batch_size = inputs["input_ids"].shape[0]
        return loss

    @property
    def current_batch_size(self) -> int:
        return self._current_batch_size


class AdversarialTrainer(TrainerWithBatchSizeStoring):
    train_dataset: Dataset | None

    def __init__(self, use_balanced_sampling: bool, **trainer_kwargs):
        super().__init__(**trainer_kwargs)

        self.use_balanced_sampling = use_balanced_sampling

        # text_chunked is not needed for training.
        # Remove it so that it's possible to merge datasets later on.
        assert self.train_dataset is not None
        if "text_chunked" in self.train_dataset.features:
            self.train_dataset = self.train_dataset.remove_columns("text_chunked")

        self.regular_dataset = self.train_dataset
        self.new_examples: dict = {"text": [], "label": []}
        self.adversarial_dataset = Dataset.from_dict({})

    @override
    def get_train_dataloader(self):
        # This method is called at the start of each training loop, when
        # my_trainer.train() is called. In turn, the train_dataloader it returns
        # is called at the start of each training epoch
        # https://github.com/huggingface/transformers/blob/5a4f340df74b42b594aedf60199eea95cdb9bed0/src/transformers/trainer.py#L812

        self.train_dataset = self.get_augmented_training_set()
        train_dataloader = super().get_train_dataloader()
        return train_dataloader

        # TODO: test this to make sure the dataloader pulls from the augmented dataset

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        assert self.train_dataset is not None

        use_balanced_sampling = self.use_balanced_sampling
        if use_balanced_sampling and len(self.adversarial_dataset) == 0:
            warnings.warn(
                "Balanced sampling requested but no adversarial examples found;"
                " falling back to regular sampling."
            )
            use_balanced_sampling = False

        if use_balanced_sampling:
            assert len(self.train_dataset) == len(self.regular_dataset) + len(
                self.adversarial_dataset
            )
            return BalancedSampler(
                regular_data=self.regular_dataset,
                adversarial_data=self.adversarial_dataset,
            )
        else:
            return super()._get_train_sampler()

    def get_augmented_training_set(self) -> Dataset:
        # Augment the train set with the new adversarial examples
        if len(self.new_examples["text"]) > 0:
            # Tokenize the new examples
            self.adversarial_dataset = self.get_tokenized_adversarial_dataset()

            train_dataset_plus_adv_examples = concatenate_datasets(
                [
                    self.regular_dataset,  # type: ignore
                    self.adversarial_dataset,
                ]
            )

        else:
            train_dataset_plus_adv_examples = self.regular_dataset

        return train_dataset_plus_adv_examples  # type: ignore

    def get_tokenized_adversarial_dataset(self) -> Dataset:
        assert len(self.new_examples["text"]) > 0

        # Tokenize the new examples
        assert self.tokenizer is not None
        tokenized_new_examples = Dataset.from_dict(
            tokenize_dataset(self.new_examples, self.tokenizer)
        )

        assert (
            self.regular_dataset.features.type  # type: ignore
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
        if self.training.log_full_datasets_to_wandb:
            assert self.training.trainer is not None

            train_dataset_plus_adv_examples = (
                self.training.trainer.get_augmented_training_set()  # type: ignore
            )

            current_round = self.training.current_iterative_training_round
            dataset_name = f"augmented_train_set_start_round_{current_round}"
            log_dataset_to_wandb(train_dataset_plus_adv_examples, dataset_name)
            wandb.log(
                {
                    "misc/augmented_train_set_size": train_dataset_plus_adv_examples.num_rows  # noqa: E501
                },
                commit=False,
            )

    @override
    def on_train_end(  # type: ignore[misc]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        # TODO(michal): consider removing this as it was mostly relevant for Tomita?
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
