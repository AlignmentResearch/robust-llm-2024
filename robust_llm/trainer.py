from __future__ import annotations

import json
import math
import os
import time
import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from accelerate import Accelerator

from robust_llm import logger
from robust_llm.callbacks import CustomLoggingWandbCallback
from robust_llm.logging_utils import log_dataset_to_wandb, wandb_log
from robust_llm.rllm_datasets.dataset_utils import cast_and_concatenate

if TYPE_CHECKING:
    from robust_llm.training import AdversarialTraining

import torch.nn.functional as F
import torch.utils.data
from datasets import Dataset
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.optimization import get_scheduler
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINING_ARGS_NAME, TrainOutput
from typing_extensions import override

from robust_llm.debug_utils import assert_dicts_equal
from robust_llm.dist_utils import DistributedRNG
from robust_llm.utils import BalancedSampler

ADV_STATE_NAME = "adversarial_training_state.json"
ADV_DATA_NAME = "adversarial_data.hf"
ADV_FILES = (ADV_STATE_NAME, ADV_DATA_NAME)


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
        self.rng = DistributedRNG(seed=self.args.seed, accelerator=self.accelerator)
        if self.args.n_gpu > 1:  # Number of GPUs in this process
            # We conflate the number of GPUs with the number of processes
            # because that's always true in our launched jobs.
            # If the number of devices and processes is different, the code will
            # probably run fine, but some hyperparameters may be off.
            warnings.warn(
                f"Process uses {self.args.n_gpu} GPUs. Training with >1 GPU per"
                " process may not work as expected. Set CUDA_VISIBLE_DEVICES."
            )
            raise ValueError("Training with >1 GPU per process.")

        if (
            self.accelerator is not None
            and self.accelerator.num_processes > 1
            and self.args.gradient_accumulation_steps > 1
        ):
            effective_batch_size = (
                self.accelerator.num_processes
                * self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps
            )
            assert isinstance(self.train_dataset, Dataset)
            if effective_batch_size > len(self.train_dataset):
                warnings.warn(
                    f"Effective batch size {effective_batch_size} is larger than"
                    f" the number training examples {len(self.train_dataset)}."
                    " FSDP training with gradient_accumulation_steps > 1 may break:"
                    " https://github.com/huggingface/transformers/issues/33413"
                )
                raise ValueError("Effective batch size is larger than training set")

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
    ) -> TrainOutput:
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
            kwargs: Additional keyword arguments to pass to the HuggingFace Trainer.
        """
        assert resume_from_checkpoint is not True, (
            "Unlike HuggingFace, we don't support passing `True` to "
            "`resume_from_checkpoint` because we want to handle the choice of path "
            "one level up in the training script."
        )
        if isinstance(resume_from_checkpoint, str):
            # Check that the training args are unchanged
            assert_training_args_equal(
                self.args,
                torch.load(os.path.join(resume_from_checkpoint, TRAINING_ARGS_NAME)),
            )
            self._load_rng_state(resume_from_checkpoint)

        return super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs,
        )


class AdversarialTrainer(RLLMTrainer):
    # train_dataset is the attribute used by the HuggingFace Trainer
    train_dataset: Dataset

    def __init__(
        self,
        use_balanced_sampling: bool,
        max_adv_data_proportion: float,
        max_augmented_data_size: int,
        loss_rank_weight: float,
        sampling_decay: float,
        num_training_steps: int,
        **trainer_kwargs,
    ):
        self._lr_scheduler = None
        super().__init__(**trainer_kwargs)

        self.use_balanced_sampling = use_balanced_sampling
        self.max_adv_data_proportion = max_adv_data_proportion
        self.max_augmented_data_size = max_augmented_data_size
        self.loss_rank_weight = loss_rank_weight
        self.sampling_decay = sampling_decay
        self.num_training_steps = num_training_steps
        self.rng = DistributedRNG(seed=self.args.seed, accelerator=self.accelerator)

        # text_chunked is not needed for training.
        # Remove it so that it's possible to merge datasets later on.
        if "text_chunked" in self.train_dataset.features:
            self.train_dataset = self.train_dataset.remove_columns("text_chunked")
        # We store all the clean data for sampling purposes
        self.regular_dataset = self.train_dataset

        # We incrementally build up a bank of adversarial examples
        # cf. add_new_adversarial_examples.
        self.adversarial_dataset = Dataset.from_dict(
            {f: [] for f in self.train_dataset.features},
            features=self.train_dataset.features,
        )
        # Track the index in self.adversarial_dataset of each adversarial example
        # in self.train_dataset
        self.adversarial_indices: list[int] = []
        self.clean_indices: list[int] = list(range(len(self.regular_dataset)))
        # Track the last loss on each adversarial example in self.adversarial_dataset
        # We use a 'str' since the keys are str when loading from checkpoint.
        self.adversarial_losses: dict[str, float] = {}
        # Store computed losses across batches
        self.computed_losses: list[float] = []
        self.in_eval_loop = False
        self.num_epochs_done = 0

        # This is for the initial step, where we just want to get the model
        # prepared.
        self.do_dummy_train_step = False

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, lr_scheduler):
        """Set the learning rate scheduler and store it for continuity."""
        if lr_scheduler is not None and self._lr_scheduler is None:
            self._lr_scheduler = lr_scheduler

    @property
    def _created_lr_scheduler(self):
        return self._lr_scheduler is not None

    @_created_lr_scheduler.setter
    def _created_lr_scheduler(self, value):
        pass

    @override  # type: ignore[misc]
    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        """Overrides HF Trainer.create_scheduler to ensure learning rate continuity.

        N.B. do not use `num_training_steps` from the arguments!
        """
        if optimizer is None:
            assert isinstance(self.optimizer, torch.optim.Optimizer)
            optimizer = self.optimizer
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(self.args.lr_scheduler_kwargs, (dict, type(None)))
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.get_warmup_steps(self.num_training_steps),
            num_training_steps=self.num_training_steps,
            scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
        )
        return self.lr_scheduler

    @property
    def is_main_process(self) -> bool:
        return self.accelerator is None or self.accelerator.is_main_process

    @override
    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
        trial: Optional[dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[list[str]] = None,
        **kwargs,
    ) -> TrainOutput:
        if (self.args.gradient_accumulation_steps > 1) and not self.do_dummy_train_step:
            # When gradient_accumulation_steps > 1, Trainer may stop short
            # of the desired number of epochs due to a rounding issue:
            # https://github.com/huggingface/transformers/issues/33455
            # For ordinary training this isn't a big deal. For adversarial
            # training, AdversarialTrainer.maybe_update_adversarial_losses only
            # runs every full epoch, so we set max_steps (which overrides
            # num_train_epochs) to hit our desired epoch count.
            global_minibatch_size = (
                self.accelerator.num_processes * self.args.per_device_train_batch_size
            )
            num_minibatches_per_epoch = math.ceil(
                len(self.train_dataset) / global_minibatch_size
            )
            num_minibatches = num_minibatches_per_epoch * self.args.num_train_epochs
            self.args.max_steps = math.ceil(
                num_minibatches / self.args.gradient_accumulation_steps
            )
        output = super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs,
        )
        if self.do_dummy_train_step or resume_from_checkpoint is not None:
            # Skip length mismatch check if we're doing a dummy train step or
            # resuming from a checkpoint.
            self.computed_losses = []
            self.num_epochs_done = 0
            return output
        self.check_num_epochs_done()
        return output

    def check_num_epochs_done(self) -> None:
        """Check that the number of epochs done is as expected.

        In order to track adversarial losses efficiently, we append to the
        computed_losses list in the `compute_loss` method for each batch.
        Once all losses have been computed, we update the adversarial losses
        and reset the computed_losses list. This is done in the
        `maybe_update_adversarial_losses` method.
        After a full training loop is completed, we want to check that we were
        not left with dangling values in computed_losses. This is easy in the
        case of gradient_accumulation_steps=1, but for larger values, we need
        to account for the fact that there may be a partial epoch at the end.
        """
        if not self.is_main_process:
            return
        if self.args.gradient_accumulation_steps == 1:
            assert self.computed_losses == []
        else:
            # self.computed_losses is not empty if we have a partial epoch.
            # We set carefully max_steps in
            # AdversarialTraining.__post_init__ to hit the right number of
            # epochs but we can still overshoot by a few (less than
            # gradient_accumulation_steps) mini-batches. E.g., if we have a
            # training dataset [0,1,2,3,4], mini-batch size of 2, gradient
            # accumulation steps of 2, and 3 epochs:
            # - Minibatch 1:  0, 1
            # - Minibatch 2:  2, 3. Step 1.
            # - Minibatch 3:  4.
            # - Minibatch 4:  0, 1. Step 2.
            # ...
            # - Minibatch 8:  2, 3. Step 4.
            # - Minibatch 9:  4.    Finished epoch 3.
            # - Minibatch 10: 0, 1. Step 5.
            # The correct choice of max_steps is 5, but this leaves the
            # losses for [0,1] in self.computed_losses.
            if len(self.computed_losses) > 0:
                self.computed_losses = []
        assert self.num_epochs_done == self.args.num_train_epochs, (
            f"Expected {self.args.num_train_epochs} epochs,"
            f" got {self.num_epochs_done}."
        )
        self.num_epochs_done = 0

    def maybe_update_adversarial_losses(self):
        """Update the adversarial losses if all losses have been computed."""
        if len(self.computed_losses) < len(self.train_dataset):
            # Wait for remaining batches to finish computing losses
            return
        assert len(self.computed_losses) == len(self.train_dataset)
        adv_losses = self.computed_losses[
            len(self.computed_losses) - len(self.adversarial_indices) :
        ]
        for index, success in zip(self.adversarial_indices, adv_losses, strict=True):
            # We use a 'str' since the keys are str when loading from checkpoint.
            self.adversarial_losses[str(index)] = success
        self.computed_losses = []
        self.num_epochs_done += 1

    @override
    def compute_loss(self, model, inputs, return_outputs=False):
        """Overrides HF Trainer.compute_loss to track attack successes."""
        # HACK: For the initial step, we return a dummy loss of 0
        if self.do_dummy_train_step:
            return torch.tensor(0.0, device=self.model.device, requires_grad=True)

        if self.in_eval_loop:
            return super().compute_loss(model, inputs, return_outputs)
        assert self.label_smoother is None
        outputs = model(**inputs)
        assert isinstance(outputs, dict)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"]
        logits = outputs["logits"]
        labels = inputs["labels"]

        if self.accelerator is not None:
            logits, labels = self.accelerator.gather_for_metrics((logits, labels))

        if self.is_main_process:
            losses = F.cross_entropy(logits, labels, reduction="none")
            self.computed_losses += losses.tolist()
        self.maybe_update_adversarial_losses()

        return (loss, outputs) if return_outputs else loss

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

    def update_augmented_training_set(
        self, log_full_datasets_to_wandb: bool, round: int | None = None
    ) -> None:
        """Resample the training set from clean/attacked.

        If no adversarial examples have been added, the augmented dataset
        will just be the regular training set.

        When adding the adversarial examples, we have to use a custom
        concatenate function here to make sure the features line up (since
        otherwise we'd have a mismatch between ClassLabel and Value(int)).
        """
        n_train = min(
            len(self.regular_dataset) + len(self.adversarial_dataset),
            self.max_augmented_data_size,
        )
        n_adv = min(
            int(n_train * self.max_adv_data_proportion),
            len(self.adversarial_dataset),
        )
        n_clean = n_train - n_adv
        self.clean_indices = self.rng.choice(
            len(self.regular_dataset),
            size=n_clean,
            replace=False,
        )
        self.adversarial_indices = self._get_adv_indices(n_adv)
        self.set_training_dataset()
        assert len(self.train_dataset) == n_train

        logger.debug(
            "Updating augmented training set to {} examples".format(
                len(self.train_dataset)
            )
        )
        wandb_log(
            {"train/n_train": n_train, "train/n_adv": n_adv, "train/n_clean": n_clean},
            commit=False,
        )
        if log_full_datasets_to_wandb:
            assert round is not None
            dataset_name = f"augmented_train_set_start_round_{round}"
            log_dataset_to_wandb(self.train_dataset, dataset_name)

    def set_training_dataset(self):
        clean_data = self.regular_dataset.select(self.clean_indices)
        adv_data = self.adversarial_dataset.select(self.adversarial_indices)
        train_dataset_plus_adv_examples = cast_and_concatenate(
            clean_data,
            adv_data,
        )
        self.train_dataset = train_dataset_plus_adv_examples

    def _get_adv_indices(self, n_adv: int) -> list[int]:
        """Get indices of adversarial examples to use for training.

        We have two distinct ways of ranking adversarial examples: by time
        and by loss. The time rank is simply the order in which the adversarial
        examples were generated. The loss rank is the ordering of the adversarial
        examples by their loss on the last time they were evaluated. We then
        compute a weighted average of these two rankings, where the weights are
        determined by the loss_rank_weight parameter. The weights are exponentiated
        and normalized to form a probability distribution, which is used to sample
        the adversarial examples.

        Args:
            n_adv: Number of adversarial examples to sample.

        Returns:
            adv_indices: Indices of adversarial examples to use for training.
        """
        assert self.rng is not None
        n = len(self.adversarial_dataset)
        if n == 0:
            return []
        time_ranks = np.arange(n)
        # We use a 'str' since the keys are str when loading from checkpoint.
        losses = [self.adversarial_losses.get(str(i), float("inf")) for i in range(n)]
        loss_ranks = np.argsort(losses)
        ranks = (
            1 - self.loss_rank_weight
        ) * time_ranks + self.loss_rank_weight * loss_ranks
        weights = np.exp(self.sampling_decay * (ranks - ranks.max()))
        sampling_probs = weights / weights.sum()
        adv_indices = self.rng.choice(
            n,
            size=n_adv,
            replace=False,
            p=sampling_probs,
        )
        return adv_indices

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
        assert isinstance(self.training.trainer, AdversarialTrainer)
        self.training.eval_rllm_dataset["augmented_train_set"] = (  # type: ignore  # noqa: E501
            self.training.trainer.train_dataset
        )


class AdversarialTrainerLoggingCallback(TrainerCallback):
    def __init__(self, training: AdversarialTraining) -> None:
        super().__init__()
        self.training = training
        self.save_start_time = time.time()

    @override
    def on_step_end(  # type: ignore[misc]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if control.should_save:
            self.save_start_time = time.time()

    @override
    def on_epoch_end(  # type: ignore[misc]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if control.should_save:
            self.save_start_time = time.time()

    @override
    def on_save(  # type: ignore[misc]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        time_to_save = time.time() - self.save_start_time
        logger.info(f"Saved checkpoint in {time_to_save:.1f} seconds.")

    @override
    def on_train_begin(  # type: ignore[misc]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if self.training.config.log_full_datasets_to_wandb:
            assert isinstance(self.training.trainer, AdversarialTrainer)

            current_round = self.training.round
            train_ds = self.training.trainer.train_dataset
            dataset_name = f"augmented_train_set_start_round_{current_round}"
            log_dataset_to_wandb(train_ds, dataset_name)
            wandb_log(
                {"misc/augmented_train_set_size": train_ds.num_rows},  # noqa: E501
                commit=False,
            )


class AdversarialTrainingState:
    """State for adversarial training, including the current round and RNG states.

    This is useful for saving and loading the state of the adversarial training in
    a way that can be resumed later.

    Attributes:
        current_round: The current round of adversarial training.
        rng: The random number generator state.
        adversarial_dataset: The adversarial dataset.
        clean_indices: The indices of clean examples in the training set. This is
            necessary to reconstruct the augmented training set from the clean and
            adversarial datasets.
        adversarial_indices: The indices of adversarial examples in the training set.
            This is necessary to reconstruct the augmented training set from the clean
            and adversarial datasets.
        adversarial_losses: A dict of losses for each adversarial example. The keys
            are strings to keep compatibility with loading from JSON. These are used
            to rank adversarial examples for sampling during training.
        training_attack_rng: The random number generator state for the training attack.
        validation_attack_rng: The random number generator state for the
            validation attack.
        total_flops: The total FLOPs for the model.
        process: The process index of the accelerator.
    """

    def __init__(
        self,
        current_round: int,
        rng: DistributedRNG,
        adversarial_dataset: Dataset,
        clean_indices: list[int],
        adversarial_indices: list[int],
        adversarial_losses: dict[str, float],
        training_attack_rng: Optional[DistributedRNG],
        validation_attack_rng: Optional[DistributedRNG],
        total_flops: float = 0.0,
    ) -> None:
        self.rng = rng
        self.current_round = current_round
        self.adversarial_dataset = adversarial_dataset
        self.clean_indices = clean_indices
        self.adversarial_indices = adversarial_indices
        self.adversarial_losses = adversarial_losses
        self.training_attack_rng = training_attack_rng
        self.validation_attack_rng = validation_attack_rng
        self.total_flops = total_flops

    def to_dict(self) -> dict:
        return {
            "rng": self.rng.getstate(),
            "current_round": self.current_round,
            "total_flops": self.total_flops,
            "training_attack_rng": (
                self.training_attack_rng.getstate()
                if self.training_attack_rng is not None
                else None
            ),
            "validation_attack_rng": (
                self.validation_attack_rng.getstate()
                if self.validation_attack_rng is not None
                else None
            ),
            "num_adv_examples": len(self.adversarial_dataset),
            "clean_indices": self.clean_indices,
            "adversarial_indices": self.adversarial_indices,
            "adversarial_losses": self.adversarial_losses,
        }

    @property
    def process(self) -> int:
        return 0 if self.rng.accelerator is None else self.rng.accelerator.process_index

    def save(self, checkpoint_dir: str) -> None:
        logger.debug(f"Saving adversarial training state: {self.to_dict()}, ")
        start_time = time.time()
        json.dump(
            obj=self.to_dict(),
            fp=open(os.path.join(checkpoint_dir, ADV_STATE_NAME), "w"),
        )
        self.adversarial_dataset.save_to_disk(
            os.path.join(checkpoint_dir, ADV_DATA_NAME)
        )
        end_time = time.time()
        logger.info(
            f"Saved adversarial training state to {checkpoint_dir} in "
            f"{end_time - start_time:.1f} seconds."
        )

    @classmethod
    def load(
        cls, checkpoint_dir: str, accelerator: Accelerator | None
    ) -> AdversarialTrainingState:
        state_path = os.path.join(checkpoint_dir, ADV_STATE_NAME)
        with open(state_path, "r") as f:
            state = json.load(f)
            rng = DistributedRNG(None, accelerator=accelerator)
            rng.setstate(state["rng"])
            current_round = state["current_round"]
            total_flops = state["total_flops"]
            if state["training_attack_rng"] is not None:
                training_attack_rng = DistributedRNG(None, accelerator=accelerator)
                training_attack_rng.setstate(state["training_attack_rng"])
            else:
                training_attack_rng = None
            if state["validation_attack_rng"] is not None:
                validation_attack_rng = DistributedRNG(None, accelerator=accelerator)
                validation_attack_rng.setstate(state["validation_attack_rng"])
            else:
                validation_attack_rng = None
            clean_indices = state["clean_indices"]
            adversarial_indices = state["adversarial_indices"]
            adversarial_losses = state["adversarial_losses"]
        adversarial_dataset = Dataset.load_from_disk(
            os.path.join(checkpoint_dir, ADV_DATA_NAME)
        )
        process = 0 if rng.accelerator is None else rng.accelerator.process_index
        out = cls(
            current_round=current_round,
            total_flops=total_flops,
            rng=rng,
            adversarial_dataset=adversarial_dataset,
            training_attack_rng=training_attack_rng,
            validation_attack_rng=validation_attack_rng,
            clean_indices=clean_indices,
            adversarial_indices=adversarial_indices,
            adversarial_losses=adversarial_losses,
        )
        logger.info(
            f"(Process {process}) Loaded adversarial training state: {out.to_dict()}"
        )
        return out

    @classmethod
    def from_training(cls, training) -> AdversarialTrainingState:
        return cls(
            current_round=training.round,
            total_flops=training.total_flops,
            rng=training.trainer.rng,
            adversarial_dataset=training.trainer.adversarial_dataset,
            training_attack_rng=getattr(training.training_attack, "rng"),
            validation_attack_rng=getattr(training.validation_attack, "rng"),
            clean_indices=training.trainer.clean_indices,
            adversarial_indices=training.trainer.adversarial_indices,
            adversarial_losses=training.trainer.adversarial_losses,
        )

    def apply_to_training(self, training: AdversarialTraining) -> int:
        logger.info(
            f"Resuming adversarial training at round {self.current_round} "
            f"with {len(self.adversarial_dataset)} adversarial examples."
        )
        assert isinstance(training.trainer, AdversarialTrainer)
        training.trainer.rng = self.rng
        training.trainer.adversarial_dataset = self.adversarial_dataset
        loaded_training_attack_rng = None
        loaded_validation_attack_rng = None
        if self.training_attack_rng is not None:
            setattr(training.training_attack, "rng", self.training_attack_rng)
            assert training.training_attack is not None
            loaded_training_attack_rng = training.training_attack.rng.getstate()
        if self.validation_attack_rng is not None:
            setattr(training.validation_attack, "rng", self.validation_attack_rng)
            assert training.validation_attack is not None
            loaded_validation_attack_rng = training.validation_attack.rng.getstate()
        training.total_flops = self.total_flops
        training.trainer.clean_indices = self.clean_indices
        training.trainer.adversarial_indices = self.adversarial_indices
        training.trainer.set_training_dataset()
        training.trainer.adversarial_losses = self.adversarial_losses
        logger.debug(
            "Applied state to training: "
            f"examples={len(training.trainer.adversarial_dataset)}, "
            f"flops={training.total_flops:.2E}, "
            f"rng={training.trainer.rng.getstate()}, "
            f"training_attack_rng={loaded_training_attack_rng}, "
            f"validation_attack_rng={loaded_validation_attack_rng}."
            f"clean_indices={len(training.trainer.clean_indices)}, "
            f"adversarial_indices={len(training.trainer.adversarial_indices)}, "
            f"train_dataset_size={len(training.trainer.train_dataset)}, "
            f"adversarial_losses={len(training.trainer.adversarial_losses)}, "
            f"process={self.process}, "
        )
        return self.current_round


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

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Save the full AdversarialTrainingState alongside HF trainer state."""
        assert isinstance(self.training.trainer, AdversarialTrainer)
        if self.training.trainer.do_dummy_train_step:
            # We avoid saving state in a dummy train step because we don't
            # want to try to resume in this round.
            return
        assert self.training.trainer.rng is not None
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        # We set trial to None as this is a HuggingFace argument for
        # hyperparameter search, which we are not using.
        run_dir = self.training.trainer._get_output_dir(trial=None)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        adv_state = AdversarialTrainingState.from_training(self.training)
        adv_state.save(output_dir)
        self.training.clean_checkpoints_and_return_valid()


class EvaluationLoopCallback(TrainerCallback):
    """Callback to record whether we are in an evaluation loop"""

    def __init__(self, training: AdversarialTraining) -> None:
        super().__init__()
        assert isinstance(training.trainer, AdversarialTrainer)
        self.trainer = training.trainer

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.trainer.in_eval_loop = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if control.should_evaluate:
            self.trainer.in_eval_loop = True
