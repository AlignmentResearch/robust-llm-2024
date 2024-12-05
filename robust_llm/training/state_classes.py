from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from functools import partial
from math import ceil
from pathlib import Path
from typing import TypeVar
from venv import logger

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from transformers import PreTrainedTokenizerBase
from transformers.optimization import Adafactor
from typing_extensions import override

from robust_llm.attacks.attack import AttackOutput
from robust_llm.attacks.attack_utils import create_attack
from robust_llm.batch_job_utils import zero_pad
from robust_llm.config.callback_configs import CallbackConfig
from robust_llm.config.configs import ExperimentConfig, get_checkpoint_path
from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.config.model_configs import ModelConfig
from robust_llm.dist_utils import assert_same_data_between_processes, rmtree_if_exists
from robust_llm.evaluation import do_adversarial_evaluation
from robust_llm.logging_utils import log, wandb_log
from robust_llm.metrics.metrics import maybe_compute_robustness_metrics
from robust_llm.models.model_utils import compute_batch_sizes_from_config
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.dataset_utils import cast_and_concatenate
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.scoring_callbacks.build_scoring_callback import (
    build_binary_scoring_callback,
)
from robust_llm.scoring_callbacks.scoring_callback_utils import CallbackInput
from robust_llm.state_classes.rng_state import RNGState
from robust_llm.training.training_utils import (
    AttackSchedule,
    construct_combined_dataset,
    find_most_recent_checkpoint,
)
from robust_llm.utils import print_time

OPTIMIZER_MAP = {
    "adamw_torch": AdamW,
    # Hardcoded kwargs copied from huggingface Trainer.
    "adafactor": partial(Adafactor, scale_parameter=False, relative_step=False),
}


def build_lr_scheduler(
    optimizer: Optimizer, config: ExperimentConfig, accelerator: Accelerator
):
    """Build a learning rate scheduler from the training config."""
    train_mb_size, _ = compute_batch_sizes_from_config(config.model)
    training_config = config.training
    assert training_config is not None
    # Adjust the train_mb_size to not be larger than the dataset that's used
    # in each process.
    dataset_size = config.dataset.n_train
    train_mb_size = max(
        1,
        min(
            dataset_size // accelerator.num_processes,
            train_mb_size,
        ),
    )
    num_batches = ceil(dataset_size / train_mb_size)
    # Adjust the number of batches to account for the number of processes.
    num_batches = num_batches // accelerator.num_processes
    num_epochs = training_config.num_train_epochs

    if training_config.lr_scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    elif training_config.lr_scheduler_type == "linear":
        if training_config.adversarial is not None:
            raise NotImplementedError(
                "TODO(GH#990): Implement linear LR schedule for adv training."
            )
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1,
            end_factor=0.0,
            total_iters=num_epochs * num_batches,
        )
    else:
        raise ValueError(
            f"Unsupported lr_scheduler_type: {training_config.lr_scheduler_type}"
        )


StateT = TypeVar("StateT", bound="TrainingPipelineState")


@dataclass
class DatasetState:
    """The state of the dataset.

    TODO(ian): Maybe separate into a subclass that handles adv dataset.

    Attributes:
        clean_dataset:
            The clean dataset used for training.
        val_dataset:
            The validation dataset used for training.
        adv_dataset:
            The adversarial dataset used for training.
        adv_losses:
            A dictionary mapping adversarial examples in adv_dataset to the most
            recent loss of the model on that example.
        clean_index_map:
            A dictionary mapping an index in the full dataset to its index in the
            clean dataset.
        adv_index_map:
            A dictionary mapping the index in the full dataset to the index in the
            adversarial dataset.
    """

    clean_dataset: RLLMDataset
    val_dataset: RLLMDataset
    adv_dataset: Dataset | None
    adv_losses: dict[int, float] = field(default_factory=dict)
    clean_index_map: dict[int, int] = field(default_factory=dict)
    adv_index_map: dict[int, int] = field(default_factory=dict)

    def save(self, path: Path, process_index: int):
        dataset_path = path / "dataset"
        dataset_path.mkdir(exist_ok=True)

        if self.adv_dataset is not None:
            self.adv_dataset.save_to_disk(dataset_path / f"adv_dataset_{process_index}")

        adv_losses_path = dataset_path / f"adv_losses_{process_index}.json"
        # TODO(ian): Decide if json is good or this should be a tensor?
        with adv_losses_path.open("w") as f:
            json.dump(self.adv_losses, f)

        clean_index_map_path = dataset_path / f"clean_index_map_{process_index}.json"
        with clean_index_map_path.open("w") as f:
            json.dump(self.clean_index_map, f)

        adv_index_map_path = dataset_path / f"adv_index_map_{process_index}.json"
        with adv_index_map_path.open("w") as f:
            json.dump(self.adv_index_map, f)

    @staticmethod
    def load(
        path: Path,
        process_index: int,
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizerBase,
    ):
        clean_dataset = load_rllm_dataset(dataset_config, split="train")
        clean_dataset = clean_dataset.tokenize(tokenizer)
        val_dataset = load_rllm_dataset(dataset_config, split="validation")
        val_dataset = val_dataset.tokenize(tokenizer)

        dataset_path = path / "dataset"
        adv_dataset_path = dataset_path / f"adv_dataset_{process_index}"
        if adv_dataset_path.exists():
            adv_dataset = Dataset.load_from_disk(
                str(adv_dataset_path),
                # We have to set keep_in_memory=True so that we can delete
                # the dataset from disk even if we loaded from it.
                keep_in_memory=True,
            )
        else:
            adv_dataset = None

        adv_losses_path = dataset_path / f"adv_losses_{process_index}.json"
        with adv_losses_path.open("r") as f:
            adv_losses = json.load(f)
            # Convert the keys to integers since they are stored as strings.
            adv_losses = {int(k): v for k, v in adv_losses.items()}

        clean_index_map_path = dataset_path / f"clean_index_map_{process_index}.json"
        with clean_index_map_path.open("r") as f:
            clean_index_map = json.load(f)
            # Convert the keys to integers since they are stored as strings.
            clean_index_map = {int(k): v for k, v in clean_index_map.items()}

        adv_index_map_path = dataset_path / f"adv_index_map_{process_index}.json"
        with adv_index_map_path.open("r") as f:
            adv_index_map = json.load(f)
            # Convert the keys to integers since they are stored as strings.
            adv_index_map = {int(k): v for k, v in adv_index_map.items()}

        return DatasetState(
            clean_dataset=clean_dataset,
            val_dataset=val_dataset,
            adv_dataset=adv_dataset,
            adv_losses=adv_losses,
            clean_index_map=clean_index_map,
            adv_index_map=adv_index_map,
        )

    def append_to_adv_dataset(self, adv_dataset: Dataset):
        assert self.adv_dataset is not None

        if len(self.adv_dataset) == 0:
            new_adv_dataset = adv_dataset
        else:
            new_adv_dataset = cast_and_concatenate(self.adv_dataset, adv_dataset)

        return DatasetState(
            clean_dataset=self.clean_dataset,
            val_dataset=self.val_dataset,
            adv_dataset=new_adv_dataset,
            adv_losses=self.adv_losses,
            clean_index_map=self.clean_index_map,
            adv_index_map=self.adv_index_map,
        )


@dataclass
class ModelState:
    wrapped_model: WrappedModel

    def save(
        self, path: Path, process_index: int, retries: int, cooldown_seconds: float
    ):
        self.wrapped_model.save_local(
            path / "model", retries=retries, cooldown_seconds=cooldown_seconds
        )

    @staticmethod
    def load(
        path: Path,
        process_index: int,
        model_config: ModelConfig,
        accelerator: Accelerator,
    ):
        model_path = path / "model"
        # Update the path in the model config temporarily to load
        # the checkpoint.
        temp_model_config = dataclasses.replace(
            model_config, name_or_path=str(model_path)
        )
        wrapped_model = WrappedModel.from_config(
            temp_model_config, accelerator=accelerator
        )
        return ModelState(wrapped_model=wrapped_model)


@dataclass
class TrainingState:
    # TODO(GH#990): Add learning rate schedule.
    optimizer: Optimizer
    lr_scheduler: LRScheduler

    def save(self, path: Path, process_index: int):
        torch.save(self.optimizer.state_dict(), path / f"optimizer_{process_index}.pt")
        torch.save(
            self.lr_scheduler.state_dict(), path / f"lr_scheduler_{process_index}.pt"
        )

    @staticmethod
    def load(
        path: Path,
        process_index: int,
        model_state: ModelState,
        config: ExperimentConfig,
        accelerator: Accelerator,
    ):
        training_config = config.training
        assert training_config is not None
        optimizer_path = path / f"optimizer_{process_index}.pt"
        scheduler_path = path / f"lr_scheduler_{process_index}.pt"
        # Ignore the type since it doesn't understand partial
        optimizer = OPTIMIZER_MAP[training_config.optimizer](  # type: ignore
            model_state.wrapped_model.model.parameters(),
            lr=training_config.learning_rate,
        )
        optimizer.load_state_dict(torch.load(optimizer_path))
        lr_scheduler = build_lr_scheduler(optimizer, config, accelerator)
        lr_scheduler.load_state_dict(torch.load(scheduler_path))
        return TrainingState(optimizer=optimizer, lr_scheduler=lr_scheduler)


@dataclass
class TrainingPipelineState:
    """Base class for state of the training loop."""

    epoch: int
    accelerator: Accelerator
    config: ExperimentConfig
    dataset_state: DatasetState
    model_state: ModelState
    training_state: TrainingState
    rng_state: RNGState
    flops: float

    def __post_init__(self):
        # To avoid issues in type checking, assert that the training config is
        # not None.
        training_config = self.config.training
        assert training_config is not None
        self.training_config = training_config
        self.check_same_state_across_processes()

    def check_same_state_across_processes(self):
        assert_same_data_between_processes(
            self.accelerator,
            [self.epoch, self.flops, self.training_state.lr_scheduler.get_last_lr()[0]],
        )

    def update_after_epoch(self, losses: list[torch.Tensor]) -> TrainingPipelineState:
        self.epoch += 1
        self.rng_state = self.rng_state.update_states()
        return self

    def training_is_finished(self) -> bool:
        return self.epoch == self.training_config.num_train_epochs

    def should_save_trained_model(self) -> bool:
        """Whether to save a model for further analysis at this epoch.

        If in adversarial training, we save the model at the end of each round.
        If not in adversarial training, we save the model at the end of training.
        """
        return self.training_is_finished()

    def should_augment_dataset(self) -> bool:
        """Return True if we should change the dataset."""
        return False

    def get_full_dataset(self) -> Dataset:
        """Get the full dataset for training."""
        # TODO(ian): Work out if we should shuffle here, elsewhere, or not at all.
        # Get the full dataset using the index map from the dataset state.
        return self.dataset_state.clean_dataset.for_training()

    @print_time()
    def save(self):
        self.accelerator.wait_for_everyone()
        process_index = self.accelerator.process_index

        epoch = self.epoch
        checkpoint_path = get_checkpoint_path(self.config)
        epoch_path = checkpoint_path / f"epoch_{zero_pad(epoch)}"
        log(f"Saving checkpoint to {epoch_path}", main_process_only=False)
        log(f"Saving checkpoint to {epoch_path}", level="print")

        # Make the directory on a single process.
        if self.accelerator.is_main_process:
            if epoch_path.exists():
                if not (epoch_path / "save_complete").exists():
                    log(
                        f"Deleting incomplete checkpoint: {epoch_path}",
                        main_process_only=False,
                    )
                    assert checkpoint_path in epoch_path.parents
                    rmtree_if_exists(epoch_path)
                else:
                    raise FileExistsError(
                        f"Path {epoch_path} already exists. Aborting save."
                    )
            try:
                epoch_path.mkdir(parents=True)
            except FileExistsError as e:
                raise FileExistsError(
                    f"Path {epoch_path} already exists even though it shouldn't."
                    " Maybe a race condition?"
                ) from e

        # Wait for main process to finish making the directory.
        self.accelerator.wait_for_everyone()

        self.model_state.save(
            epoch_path,
            process_index,
            retries=self.training_config.upload_retries,
            cooldown_seconds=self.training_config.upload_cooldown,
        )
        self.training_state.save(epoch_path, process_index)
        self.dataset_state.save(epoch_path, process_index)
        self.rng_state.save(epoch_path, process_index, self.accelerator)

        if self.accelerator.is_main_process:
            state_dict = {
                "epoch": epoch,
                "flops": self.flops,
                # Save the config as a dictionary. We don't load this config
                # since subclasses are lost, it's just here for debugging.
                # We use default=str to serialize enums and other objects, since
                # we aren't intending to deserialize this.
                "config": json.dumps(dataclasses.asdict(self.config), default=str),
            }
            with open(epoch_path / "state.json", "w") as f:
                json.dump(state_dict, f)

        # Touch a file to indicate that the save is complete and that this
        # checkpoint can be used for loading.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            with open(epoch_path / "save_complete", "w") as f:
                f.write("")
        self.accelerator.wait_for_everyone()

    @print_time()
    def augment_dataset(self):
        """This is a placeholder for subclasses to override."""
        raise NotImplementedError

    def process_outputs(self, outputs):
        """Do any processing of the model outputs that is necessary."""

    def should_evaluate(self) -> bool:
        """Whether to evaluate the model at this epoch.

        For plain finetuning, we evaluate every epoch.
        """
        return True

    def evaluate(self, local_files_path: Path):
        """Evaluate the model at this epoch.

        For plain finetuning, we compute accuracy on the eval
        set.
        TODO(ian): Add support for more metrics, generative tasks, etc.
        """
        dataset = self.dataset_state.val_dataset
        victim = self.model_state.wrapped_model
        # TODO(ian): Reusing the success callback for now, but this is not very general.
        success_callback = build_binary_scoring_callback(
            CallbackConfig(
                callback_name="successes_from_text",
                callback_return_type="binary",
            ),
        )
        callback_input = CallbackInput(
            # TODO(ian): Work out where to apply chat template.
            input_data=victim.maybe_apply_user_template(dataset.ds["text"]),
            clf_label_data=dataset.ds["clf_label"],
            gen_target_data=dataset.ds["gen_target"],
        )
        pre_attack_out = success_callback(
            victim,
            callback_input,
        )
        pre_attack_successes = pre_attack_out.successes
        accuracy = sum(pre_attack_successes) / len(pre_attack_successes)
        log(f"Accuracy on validation set at epoch {self.epoch}: {accuracy}")
        wandb_log({"eval/accuracy": accuracy}, commit=True)

    @classmethod
    def load(
        cls: type[StateT],
        config: ExperimentConfig,
        accelerator: Accelerator,
    ) -> StateT:
        """Load a state from disk.

        We do most things on both processes since both need copies of the state.

        Note: We don't reload the config because subclasses are lost when
        deserializing. This is fine because the configs are necessarily
        identical, given that we look it up by hash value.
        """
        checkpoint_path = get_checkpoint_path(config)
        process_index = accelerator.process_index

        subdir = find_most_recent_checkpoint(checkpoint_path)

        with open(subdir / "state.json", "r") as f:
            state_dict = json.load(f)

        epoch = state_dict["epoch"]
        flops = state_dict["flops"]
        assert zero_pad(epoch) == subdir.name.split("_")[-1]

        model_state = ModelState.load(subdir, process_index, config.model, accelerator)

        rng_state = RNGState.load(subdir, process_index, accelerator)
        rng_state.set_random_states()

        # Load the dataset state after the model state since it depends on the
        # tokenizer.
        tokenizer = model_state.wrapped_model.right_tokenizer
        dataset_state = DatasetState.load(
            subdir, process_index, config.dataset, tokenizer
        )

        # Load the optimizer state afterwards since it depends on the model.
        training_state = TrainingState.load(
            subdir,
            process_index,
            model_state=model_state,
            config=config,
            accelerator=accelerator,
        )

        return cls(
            epoch=epoch,
            accelerator=accelerator,
            config=config,
            dataset_state=dataset_state,
            model_state=model_state,
            training_state=training_state,
            rng_state=rng_state,
            flops=flops,
        )

    def get_revision(self) -> str:
        return "main"

    @print_time()
    def save_trained_model(self) -> None:
        # Make sure everything is in sync before saving.
        self.accelerator.wait_for_everyone()

        epoch = self.epoch
        checkpoint_path = get_checkpoint_path(self.config)
        epoch_path = checkpoint_path / f"epoch_{zero_pad(epoch)}"
        model_path = epoch_path / "model"
        if not model_path.exists():
            log(
                f"Expected model at path {model_path} doesn't exist!",
                level="warning",
            )
            return
        done_saving = model_path / "done-saving"
        if not done_saving.exists():
            log(
                f"Model at path {model_path} isn't done saving!",
                level="warning",
            )
            return
        model_name = self.training_config.save_name
        revision = self.get_revision()
        with open(model_path / ".gitignore", "a") as f:
            f.write("\nmodel_name.txt\nrevision.txt")
        with open(model_path / "model_name.txt", "w") as f:
            f.write(model_name)
        with open(model_path / "revision.txt", "w") as f:
            f.write(revision)

    def mark_as_finished(self) -> None:
        epoch = self.epoch
        checkpoint_path = get_checkpoint_path(self.config)
        epoch_path = checkpoint_path / f"epoch_{zero_pad(epoch)}"
        model_path = epoch_path / "model"
        with open(model_path / ".gitignore", "a") as f:
            f.write("\ndone-training")
        with open(model_path / "done-training", "w") as f:
            pass

    def log_epoch(self):
        wandb_log({"epoch": self.epoch}, commit=False)


@dataclass
class AdversarialPipelineState(TrainingPipelineState):
    """State for adversarial training."""

    def __post_init__(self):
        # To avoid issues in type checking, assert that the training config is
        # not None.
        training_config = self.config.training
        assert training_config is not None
        adv_config = training_config.adversarial
        assert adv_config is not None
        self.training_config = training_config
        self.adv_config = adv_config

    @property
    def training_round(self) -> int:
        """Which adversarial training round are we on?"""
        epochs_per_round = self.training_config.num_train_epochs
        return self.epoch // epochs_per_round

    def is_new_round(self) -> bool:
        """Is this the first epoch of a new adversarial training round?

        This is used to determine when to generate new adversarial examples and whether
        to save an intermediate model.
        """
        epochs_per_round = self.training_config.num_train_epochs
        return self.epoch % epochs_per_round == 0

    def is_last_epoch_of_round(self) -> bool:
        """Is this the last epoch of an adversarial training round?

        This is used to determine when to evaluate.
        """
        epochs_per_round = self.training_config.num_train_epochs
        return self.epoch % epochs_per_round == epochs_per_round - 1

    @override
    def training_is_finished(self) -> bool:
        num_rounds = self.adv_config.num_adversarial_training_rounds
        return self.training_round == num_rounds

    @override
    def should_save_trained_model(self) -> bool:
        """Whether to save a model for further analysis at this epoch.

        We save the model at the end of each round.
        """
        return self.epoch != 0 and (self.is_new_round() or self.training_is_finished())

    def generate_adv_examples(self) -> Dataset:
        self.model_state.wrapped_model.model.eval()
        num_rounds = self.adv_config.num_adversarial_training_rounds
        num_examples = self.adv_config.num_examples_to_generate_each_round

        input_rllm_dataset = self.dataset_state.clean_dataset.get_random_subset(
            n=num_examples,
            accelerator=self.accelerator,
            generator=self.rng_state.distributed_rng,
        )
        # TODO(ian): Maybe pass an RNG object here? So we don't use the same seed
        # and sequence for each round.
        attack = create_attack(
            exp_config=self.config,
            victim=self.model_state.wrapped_model,
            is_training=True,
        )

        attack_schedule = AttackSchedule(self.adv_config.attack_schedule, num_rounds)
        n_its = attack_schedule[self.training_round]
        log(f"Generating {num_examples} adversarial examples with {n_its} iterations.")

        attack_out = attack.get_attacked_dataset(
            input_rllm_dataset,
            n_its=n_its,
            epoch=self.epoch,
        )
        log(f"Attack flops: {attack_out.flops:.2E} flops.", level="debug")
        self.flops += attack_out.flops
        self._compute_metrics_on_adv_examples(attack_out)

        adv_examples = attack_out.dataset.as_adversarial_examples().for_training()
        log(f"Generated {len(adv_examples)} adversarial examples.")
        return adv_examples

    def _compute_metrics_on_adv_examples(self, attack_out: AttackOutput):
        # Logging success from generating adv examples
        assert self.config.evaluation is not None
        cb_config = self.config.evaluation.final_success_binary_callback
        callback = build_binary_scoring_callback(cb_config)
        attack_metrics = maybe_compute_robustness_metrics(
            compute_robustness_metric=self.config.evaluation.compute_robustness_metric,
            attack_out=attack_out,
            success_callback=callback,
            model=self.model_state.wrapped_model,
        )
        if attack_metrics is not None:
            wandb_log(attack_metrics.unwrap_metrics(prefix="train"), commit=False)
            attack_metrics.export_wandb_table()

    @override
    def get_revision(self) -> str:
        return f"adv-training-round-{self.training_round}"

    @override
    def get_full_dataset(self) -> Dataset:
        """Resample the training set from clean/attacked.

        If no adversarial examples have been added, the augmented dataset
        will just be the regular training set.

        When adding the adversarial examples, we have to use a custom
        concatenate function here to make sure the features line up (since
        otherwise we'd have a mismatch between ClassLabel and Value(int)).
        """
        clean_ds = super().get_full_dataset()
        adv_ds = self.dataset_state.adv_dataset
        assert adv_ds is not None
        return construct_combined_dataset(
            clean_ds,
            adv_ds,
            self.dataset_state.clean_index_map,
            self.dataset_state.adv_index_map,
        )

    def _get_adv_indices(self, adv_ds: Dataset, n_adv: int) -> list[int]:
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
            adv_ds: Adversarial dataset.
            n_adv: Number of adversarial examples to sample.

        Returns:
            adv_indices: Indices of adversarial examples to use for training.
        """
        d_rng = self.rng_state.distributed_rng
        assert d_rng is not None
        n = len(adv_ds)
        if n == 0:
            return []

        # Adv dataset is ordered by when examples were added.
        # TODO(ian/oskar): Think about whether this makes sense when multiple
        # examples were added at once.
        time_ranks = np.arange(n)

        # We use a 'str' since the keys are str when loading from checkpoint.
        losses = [self.dataset_state.adv_losses.get(i, float("inf")) for i in range(n)]
        loss_ranks = np.argsort(losses)

        alpha = self.adv_config.loss_rank_weight
        ranks = (1 - alpha) * time_ranks + alpha * loss_ranks

        weights = np.exp(self.adv_config.adv_sampling_decay * (ranks - ranks.max()))
        sampling_probs = weights / weights.sum()

        adv_indices = d_rng.choice(
            n,
            size=n_adv,
            replace=False,
            p=sampling_probs,
        )
        return adv_indices

    @override
    def should_augment_dataset(self) -> bool:
        """Return True if we should generate adversarial examples.

        For now this is just based on whether the adversarial config is set, but
        in the future it could depend on the epoch or other factors.
        """
        return self.is_new_round()

    @override
    @print_time()
    def augment_dataset(self):
        """Augment the dataset with adversarial examples.

        TODO(ian): Clean up how we keep track of the full dataset.
        """
        adv_examples = self.generate_adv_examples()
        self.dataset_state = self.dataset_state.append_to_adv_dataset(adv_examples)
        # Now we also have to update the index maps.
        clean_ds = self.dataset_state.clean_dataset.for_training()
        adv_ds = self.dataset_state.adv_dataset
        assert adv_ds is not None

        n_train = min(
            len(clean_ds) + len(adv_ds),
            self.adv_config.max_augmented_data_size,
        )
        n_adv = min(
            int(n_train * self.adv_config.max_adv_data_proportion),
            len(adv_ds),
        )
        n_clean = n_train - n_adv

        clean_indices = self.rng_state.distributed_rng.choice(
            len(clean_ds),
            size=n_clean,
            replace=False,
        )
        clean_index_map = {i: clean_indices[i] for i in range(len(clean_indices))}
        adv_indices = self._get_adv_indices(adv_ds, n_adv)
        # adv_indices_map maps the index in full_ds to the index in adv_ds.
        # Note that we have to add n_clean to the indices since the clean data
        # comes first.
        adv_index_map = {n_clean + i: adv_indices[i] for i in range(len(adv_indices))}

        # Now we shuffle the dataset, and keep track of where that leaves
        # the indices of the clean and adversarial examples.
        permutation = self.rng_state.distributed_rng.choice(
            n_train,
            size=n_train,
            replace=False,
        )
        perm_dict = {v: i for i, v in enumerate(permutation)}

        clean_index_map = {perm_dict[k]: v for k, v in clean_index_map.items()}
        adv_index_map = {perm_dict[k]: v for k, v in adv_index_map.items()}
        self.dataset_state.clean_index_map = clean_index_map
        self.dataset_state.adv_index_map = adv_index_map

        logger.debug("Updating augmented training set to %s examples", n_train)
        wandb_log(
            {"train/n_train": n_train, "train/n_adv": n_adv, "train/n_clean": n_clean},
            commit=False,
        )

    @override
    def update_after_epoch(self, losses: list[torch.Tensor]) -> TrainingPipelineState:
        """Update the state after an epoch of training.

        In adversarial training we also have to update the losses of the
        adversarial examples.
        """
        state = super().update_after_epoch(losses)
        for i, loss in enumerate(losses):
            adv_index = self.dataset_state.adv_index_map.get(i)
            if adv_index is not None:
                state.dataset_state.adv_losses[adv_index] = loss.item()
        return state

    def should_evaluate(self) -> bool:
        """Whether to evaluate the model at this epoch.

        For adv training, we evaluate after every round
        """
        evaluate_during_training = self.adv_config.evaluate_during_training
        return evaluate_during_training and self.is_last_epoch_of_round()

    @override
    def evaluate(self, local_files_path: Path):
        victim = self.model_state.wrapped_model
        victim.eval()

        if self.adv_config.evaluate_during_training:
            eval_config = self.config.evaluation
            assert eval_config is not None

            attack = create_attack(
                exp_config=self.config,
                victim=victim,
                is_training=False,
            )
            cb_config = eval_config.final_success_binary_callback
            callback = build_binary_scoring_callback(cb_config)

            compute_robustness_metric = eval_config.compute_robustness_metric
            do_adversarial_evaluation(
                victim=victim,
                dataset=self.dataset_state.val_dataset,
                attack=attack,
                n_its=eval_config.num_iterations,
                final_success_binary_callback=callback,
                num_examples_to_log_detailed_info=eval_config.num_examples_to_log_detailed_info,  # noqa: E501
                adv_training_round=self.training_round,
                local_files_path=local_files_path,
                # We don't use checkpointing of attacks during adversarial training
                resume_from_checkpoint=False,
                compute_robustness_metric=compute_robustness_metric,
                upload_artifacts=False,
            )

    def log_epoch(self):
        wandb_log(
            {
                "epoch": self.epoch,
                "adversarial_training_round": self.training_round,
            },
            commit=False,
        )
