from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

from robust_llm.callbacks import CustomLoggingWandbCallback
from robust_llm.logging_utils import log_dataset_to_wandb
from robust_llm.rllm_datasets.dataset_utils import cast_and_concatenate

if TYPE_CHECKING:
    from robust_llm.training import AdversarialTraining

import torch.utils.data
import wandb
from datasets import Dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINING_ARGS_NAME
from typing_extensions import override

from robust_llm.debug_utils import assert_dicts_equal
from robust_llm.utils import BalancedSampler


def assert_training_args_equal(
    args1: TrainingArguments,
    args2: TrainingArguments,
    exclusions: tuple[str, ...] = ("logging_dir",),
) -> bool:
    return assert_dicts_equal(
        {k: v for k, v in args1.to_dict().items() if k not in exclusions},
        {k: v for k, v in args2.to_dict().items() if k not in exclusions},
    )


class RLLMTrainer(Trainer):
    """A Trainer with some extra features for the robust-llm project.

    If we are resuming from a checkpoint, checks that the training args are unchanged.

    Also stores the batch size of the current batch.
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

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
        trial: Optional[dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
        **kwargs,
    ):
        """Wrapper around HuggingFace Trainer.train that checks training args.

        Args:
            resume_from_checkpoint: If a string, the path to a checkpoint to load.
                If True, resume from the latest checkpoint in the output directory.
                If False, start training from scratch.
            trial: The trial run or the hyperparameter dictionary for hyperparameter
                search. This should be None if not using hyperparameter search.
            ignore_keys_for_eval: A list of keys in the output of your model (if
                it is a dictionary) that should be ignored when gathering predictions
                for evaluation during the training.
        """
        assert not isinstance(resume_from_checkpoint, bool), (
            "Unlike HuggingFace, we don't support passing a boolean to "
            "`resume_from_checkpoint` because we want to handle the choice of path "
            "one level up in the training script."
        )
        if isinstance(resume_from_checkpoint, str):
            # Check that the training args are unchanged
            assert_training_args_equal(
                self.args,
                torch.load(os.path.join(resume_from_checkpoint, TRAINING_ARGS_NAME)),
            )

        super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs,
        )


class AdversarialTrainer(RLLMTrainer):
    train_dataset: Dataset

    def __init__(self, use_balanced_sampling: bool, **trainer_kwargs):
        super().__init__(**trainer_kwargs)

        self.use_balanced_sampling = use_balanced_sampling

        # text_chunked is not needed for training.
        # Remove it so that it's possible to merge datasets later on.
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
        """Return the training set with adversarial examples added.

        If no adversarial examples have been added, this will return the
        regular training set.

        When adding the adversarial examples, we have to use a custom
        concatenate function here to make sure the features line up (since
        otherwise we'd have a mismatch between ClassLabel and Value(int)).
        """
        if len(self.adversarial_dataset) > 0:
            train_dataset_plus_adv_examples = cast_and_concatenate(
                self.regular_dataset,
                self.adversarial_dataset,
            )
        else:
            train_dataset_plus_adv_examples = self.regular_dataset

        return train_dataset_plus_adv_examples  # type: ignore

    def add_new_adversarial_examples(self, new_examples: Dataset) -> None:
        """Add new adversarial examples to the adversarial dataset.

        When adding the adversarial examples, we have to use a custom
        concatenate function here to make sure the features line up (since
        otherwise we'd have a mismatch between ClassLabel and Value(int)).
        """
        if len(self.adversarial_dataset) == 0:
            self.adversarial_dataset = new_examples
        else:
            self.adversarial_dataset = cast_and_concatenate(
                self.adversarial_dataset,
                new_examples,
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
        if self.training.config.log_full_datasets_to_wandb:
            assert self.training.trainer is not None
            assert self.training.state is not None

            train_dataset_plus_adv_examples = (
                self.training.trainer.get_augmented_training_set()  # type: ignore
            )

            current_round = self.training.state.current_round
            dataset_name = f"augmented_train_set_start_round_{current_round}"
            log_dataset_to_wandb(train_dataset_plus_adv_examples, dataset_name)
            wandb.log(
                {
                    "misc/augmented_train_set_size": train_dataset_plus_adv_examples.num_rows  # noqa: E501
                },
                commit=False,
            )


class AdversarialTrainingStateCallback(CustomLoggingWandbCallback):
    """Callback to save and restore the state of adversarial training.

    This is necessary because state like the current adversarial training round is
    not saved by default, and we need to be able to resume training
    at the current round. We also need to ensure that the global step
    keeps increasing across rounds for consistency with checkpoint numbers.
    """

    def __init__(self, training: AdversarialTraining) -> None:
        super().__init__(training)
        self.training: AdversarialTraining = training
        self.state = None

    @override
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Save the state of the training after each round."""
        self.state = state
        super().on_train_end(args, state, control, **kwargs)

    @override
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Restore the training state on subsequent rounds."""
        if self.state is not None:
            assert self.training.trainer is not None
            state = self.state
            self.training.trainer.state = state
        super().on_train_begin(args, state, control, **kwargs)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        assert self.training.trainer is not None
        assert self.training.state is not None
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        # We set trial to None as this is a HuggingFace argument for
        # hyperparameter search, which we are not using.
        run_dir = self.training.trainer._get_output_dir(trial=None)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.training.state.save(output_dir)
