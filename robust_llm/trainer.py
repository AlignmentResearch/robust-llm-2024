from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

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

from robust_llm.utils import BalancedSampler


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
        self, model: torch.nn.Module, inputs: dict[str, Union[torch.Tensor, Any]]
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

        # Will be set in add_new_adversarial_examples.
        # TODO (ian): avoid statefulness if possible
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
        if len(self.adversarial_dataset) > 0:
            train_dataset_plus_adv_examples = concatenate_datasets(
                [
                    self.regular_dataset,
                    self.adversarial_dataset,
                ]
            )
        else:
            train_dataset_plus_adv_examples = self.regular_dataset

        return train_dataset_plus_adv_examples  # type: ignore

    def add_new_adversarial_examples(self, new_examples: Dataset) -> None:
        if len(self.adversarial_dataset) == 0:
            self.adversarial_dataset = new_examples
        else:
            self.adversarial_dataset = concatenate_datasets(
                [self.adversarial_dataset, new_examples]
            )


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
        self.training.eval_rllm_dataset["augmented_train_set"] = augmented_train_set


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

            current_round = self.training.current_adversarial_training_round
            dataset_name = f"augmented_train_set_start_round_{current_round}"
            log_dataset_to_wandb(train_dataset_plus_adv_examples, dataset_name)
            wandb.log(
                {
                    "misc/augmented_train_set_size": train_dataset_plus_adv_examples.num_rows  # noqa: E501
                },
                commit=False,
            )
