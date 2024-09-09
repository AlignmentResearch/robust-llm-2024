from __future__ import annotations

import json
import os
import random
import time
import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from robust_llm import logger
from robust_llm.callbacks import CustomLoggingWandbCallback
from robust_llm.logging_utils import log_dataset_to_wandb, wandb_log
from robust_llm.rllm_datasets.dataset_utils import cast_and_concatenate
from robust_llm.utils import nested_list_to_tuple

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
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINING_ARGS_NAME, TrainOutput
from typing_extensions import override

from robust_llm.debug_utils import assert_dicts_equal
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
        self.rng = np.random.default_rng(seed=self.args.seed)

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
        **trainer_kwargs,
    ):
        super().__init__(**trainer_kwargs)

        self.use_balanced_sampling = use_balanced_sampling
        self.max_adv_data_proportion = max_adv_data_proportion
        self.max_augmented_data_size = max_augmented_data_size
        self.loss_rank_weight = loss_rank_weight
        self.sampling_decay = sampling_decay
        self.rng = np.random.default_rng(seed=self.args.seed)

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
        # Track the last loss on each adversarial example in self.adversarial_dataset
        self.adversarial_losses: dict[int, float] = {}
        # Store computed losses across batches
        self.computed_losses: list[float] = []
        self.in_eval_loop = False

        # This is for the initial step, where we just want to get the model
        # prepared.
        self.do_dummy_train_step = False

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
            self.adversarial_losses[index] = success
        self.computed_losses = []

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

        losses = F.cross_entropy(logits, labels, reduction="none")
        if self.accelerator is None or self.accelerator.is_main_process:
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
        clean_indices = self.rng.choice(
            len(self.regular_dataset),
            size=n_clean,
            replace=False,
        )
        adv_indices = self._get_adv_indices(n_adv)
        clean_data = self.regular_dataset.select(clean_indices)
        adv_data = self.adversarial_dataset.select(adv_indices)
        train_dataset_plus_adv_examples = cast_and_concatenate(
            clean_data,
            adv_data,
        )
        assert len(train_dataset_plus_adv_examples) == n_train
        self.train_dataset = train_dataset_plus_adv_examples
        self.adversarial_indices = adv_indices.tolist()

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

    def _get_adv_indices(self, n_adv: int) -> np.ndarray:
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
        n = len(self.adversarial_dataset)
        if n == 0:
            return np.array([], dtype=int)
        time_ranks = np.arange(n)
        losses = [self.adversarial_losses.get(i, float("inf")) for i in range(n)]
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
    """

    def __init__(
        self,
        current_round: int,
        rng: np.random.Generator,
        adversarial_dataset: Dataset,
        training_attack_rng: Optional[random.Random],
        validation_attack_rng: Optional[random.Random],
        total_flops: float = 0.0,
    ) -> None:
        self.rng = rng
        self.current_round = current_round
        self.adversarial_dataset = adversarial_dataset
        self.training_attack_rng = training_attack_rng
        self.validation_attack_rng = validation_attack_rng
        self.total_flops = total_flops

    def to_dict(self) -> dict:
        return {
            "rng": self.rng.bit_generator.state,
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
        }

    def save(self, checkpoint_dir: str) -> None:
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
    def load(cls, checkpoint_dir: str) -> "AdversarialTrainingState":
        state_path = os.path.join(checkpoint_dir, ADV_STATE_NAME)
        with open(state_path, "r") as f:
            state = json.load(f)
            rng = np.random.default_rng()
            rng.bit_generator.state = state["rng"]
            current_round = state["current_round"]
            total_flops = state["total_flops"]
            if state["training_attack_rng"] is not None:
                training_attack_rng = random.Random()
                training_attack_rng.setstate(
                    nested_list_to_tuple(state["training_attack_rng"])
                )
            else:
                training_attack_rng = None
            if state["validation_attack_rng"] is not None:
                validation_attack_rng = random.Random()
                validation_attack_rng.setstate(
                    nested_list_to_tuple(state["validation_attack_rng"])
                )
            else:
                validation_attack_rng = None
        adversarial_dataset = Dataset.load_from_disk(
            os.path.join(checkpoint_dir, ADV_DATA_NAME)
        )
        return cls(
            current_round=current_round,
            total_flops=total_flops,
            rng=rng,
            adversarial_dataset=adversarial_dataset,
            training_attack_rng=training_attack_rng,
            validation_attack_rng=validation_attack_rng,
        )

    def apply_to_training(self, training: AdversarialTraining) -> int:
        logger.info(
            f"Resuming adversarial training at round {self.current_round} "
            f"with {len(self.adversarial_dataset)} adversarial examples."
        )
        assert isinstance(training.trainer, AdversarialTrainer)
        training.trainer.rng = self.rng
        training.trainer.adversarial_dataset = self.adversarial_dataset
        if self.training_attack_rng is not None:
            setattr(training.training_attack, "rng", self.training_attack_rng)
        if self.validation_attack_rng is not None:
            setattr(training.validation_attack, "rng", self.validation_attack_rng)
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
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        # We set trial to None as this is a HuggingFace argument for
        # hyperparameter search, which we are not using.
        run_dir = self.training.trainer._get_output_dir(trial=None)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        adv_state = AdversarialTrainingState(
            current_round=self.training.round,
            total_flops=self.training.total_flops,
            rng=self.training.trainer.rng,
            adversarial_dataset=self.training.trainer.adversarial_dataset,
            training_attack_rng=getattr(self.training.training_attack, "rng"),
            validation_attack_rng=getattr(self.training.validation_attack, "rng"),
        )
        adv_state.save(output_dir)


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
