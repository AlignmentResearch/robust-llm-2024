from __future__ import annotations

import dataclasses
import json
import shutil
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TypeVar
from venv import logger

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from datasets import Dataset
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from transformers import PreTrainedTokenizerBase
from transformers.optimization import Adafactor
from typing_extensions import override

from robust_llm.attacks.attack_utils import create_attack
from robust_llm.batch_job_utils import zero_pad
from robust_llm.config.configs import ExperimentConfig, SaveTo, TrainingConfig
from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.config.model_configs import ModelConfig
from robust_llm.dist_utils import DistributedRNG
from robust_llm.logging_utils import log, wandb_log
from robust_llm.models.model_disk_utils import generate_model_save_path
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.dataset_utils import cast_and_concatenate
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.training.training_utils import (
    AttackSchedule,
    construct_combined_dataset,
    find_most_recent_checkpoint,
    get_sorted_checkpoints,
)
from robust_llm.utils import deterministic_hash

OPTIMIZER_MAP = {
    "adamw_torch": AdamW,
    # Hardcoded kwargs copied from huggingface Trainer.
    "adafactor": partial(Adafactor, scale_parameter=False, relative_step=False),
}

StateT = TypeVar("StateT", bound="TrainingPipelineState")


@dataclass
class DatasetState:
    """The state of the dataset.

    TODO(ian): Maybe separate into a subclass that handles adv dataset.

    Attributes:
        clean_dataset:
            The clean dataset used for training.
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

        dataset_path = path / "dataset"
        adv_dataset_path = dataset_path / f"adv_dataset_{process_index}"
        if adv_dataset_path.exists():
            adv_dataset = Dataset.load_from_disk(str(adv_dataset_path))
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
            adv_dataset=new_adv_dataset,
            adv_losses=self.adv_losses,
            clean_index_map=self.clean_index_map,
            adv_index_map=self.adv_index_map,
        )


@dataclass
class ModelState:
    wrapped_model: WrappedModel

    def save(self, path: Path, process_index: int):
        self.wrapped_model.save_local(path / "model")

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

    def save(self, path: Path, process_index: int):
        torch.save(self.optimizer.state_dict(), path / f"optimizer_{process_index}.pt")

    @staticmethod
    def load(
        path: Path,
        process_index: int,
        model_state: ModelState,
        training_config: TrainingConfig,
    ):
        optimizer_path = path / f"optimizer_{process_index}.pt"
        # Ignore the type since it doesn't understand partial
        optimizer = OPTIMIZER_MAP[training_config.optimizer](  # type: ignore
            model_state.wrapped_model.model.parameters(),
            lr=training_config.learning_rate,
        )
        optimizer.load_state_dict(torch.load(optimizer_path))
        return TrainingState(optimizer=optimizer)


@dataclass
class RNGState:
    """The state of the random number generators.

    Contains two types:
    - '_rng_state's, which tracks some global state.
    - '_rng's, which can be used to generate random numbers.
    """

    torch_rng_state: torch.Tensor
    distributed_rng: DistributedRNG

    def update_states(self) -> RNGState:
        return RNGState(
            torch_rng_state=torch.random.get_rng_state(),
            distributed_rng=self.distributed_rng,
        )

    def set_random_states(self):
        torch.random.set_rng_state(self.torch_rng_state)

    def save(self, path: Path, process_index: int, accelerator: Accelerator):
        rng_path = path / "rng"
        rng_path.mkdir(exist_ok=True)
        torch.save(
            self.torch_rng_state, rng_path / f"torch_rng_state_{process_index}.pt"
        )
        dist_rng_state = self.distributed_rng.getstate()
        torch.save(
            dist_rng_state, rng_path / f"distributed_rng_state_{process_index}.pt"
        )

    @staticmethod
    def load(path: Path, process_index: int, accelerator: Accelerator) -> RNGState:
        rng_path = path / "rng"
        if accelerator.is_main_process:
            dist_rng_state = torch.load(
                rng_path / f"distributed_rng_state_{process_index}.pt"
            )
        else:
            dist_rng_state = None
        dist_rng = DistributedRNG(seed=0, accelerator=accelerator)
        dist_rng.setstate(dist_rng_state)

        return RNGState(
            torch_rng_state=torch.load(
                rng_path / f"torch_rng_state_{process_index}.pt"
            ),
            distributed_rng=dist_rng,
        )


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

    def save(self, path: Path):
        self.accelerator.wait_for_everyone()
        process_index = self.accelerator.process_index

        hex_hash = deterministic_hash(self.config)
        epoch = self.epoch
        path = path / hex_hash / f"epoch_{zero_pad(epoch)}"
        log(f"Saving checkpoint to {path}")

        # Make the directory on a single process.
        if self.accelerator.is_main_process:
            try:
                path.mkdir(parents=True)
            except FileExistsError as e:
                raise FileExistsError(
                    f"Path {path} already exists. Aborting save."
                ) from e

        # Wait for main process to finish making the directory.
        self.accelerator.wait_for_everyone()

        self.model_state.save(path, process_index)
        self.training_state.save(path, process_index)
        self.dataset_state.save(path, process_index)
        self.rng_state.save(path, process_index, self.accelerator)

        if self.accelerator.is_main_process:
            state_dict = {
                "epoch": epoch,
                "flops": self.flops,
            }
            with open(path / "state.json", "w") as f:
                json.dump(state_dict, f)

        # Touch a file to indicate that the save is complete and that this
        # checkpoint can be used for loading.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            with open(path / "save_complete", "w") as f:
                f.write("")
        self.accelerator.wait_for_everyone()

    def cleanup_checkpoints(self, path: Path):
        """Delete old checkpoints to save disk space.

        The process is: iterate in reverse order through the checkpoints, and
        once we have found save_total_limit checkpoints that are safely saved,
        we delete all older checkpoints.
        """
        # Only do file operations on the main process.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:

            hex_hash = deterministic_hash(self.config)
            path = path / hex_hash
            if not path.exists():
                return

            save_total_limit = self.training_config.save_total_limit
            safe_checkpoint_epochs: list[str] = []
            log(f"Cleaning up checkpoints in {path}")
            for subdir in get_sorted_checkpoints(path):
                if len(safe_checkpoint_epochs) >= save_total_limit:
                    assert hex_hash in str(subdir)
                    assert subdir.name not in safe_checkpoint_epochs
                    shutil.rmtree(subdir)
                elif (subdir / "save_complete").exists():
                    safe_checkpoint_epochs.append(subdir.name)
                else:
                    log(f"Deleting incomplete checkpoint: {subdir}")
                    assert hex_hash in str(subdir)
                    assert subdir.name not in safe_checkpoint_epochs
                    shutil.rmtree(subdir)
            log(f"Keeping checkpoints: {safe_checkpoint_epochs}")

        self.accelerator.wait_for_everyone()

    def augment_dataset(self):
        """This is a placeholder for subclasses to override."""
        raise NotImplementedError

    def process_outputs(self, outputs):
        """Do any processing of the model outputs that is necessary."""

    @classmethod
    def load(
        cls: type[StateT],
        config: ExperimentConfig,
        path: Path,
        accelerator: Accelerator,
    ) -> StateT:
        """Load a state from disk.

        We do most things on both processes since both need copies of the state.

        Note: We don't reload the config because subclasses are lost when
        deserializing. This is fine because the configs are necessarily
        identical, given that we look it up by hash value.
        """
        hex_hash = deterministic_hash(config)
        path = path / hex_hash

        process_index = accelerator.process_index

        subdir = find_most_recent_checkpoint(path)

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
        training_config = config.training
        assert training_config is not None
        training_state = TrainingState.load(
            subdir, process_index, model_state, training_config
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

    def save_trained_model(self, models_path: Path) -> None:
        # Make sure everything is in sync before saving.
        self.accelerator.wait_for_everyone()

        save_to = self.training_config.save_to
        model_name = self.training_config.save_name
        revision = self.get_revision()
        if save_to == SaveTo.NONE:
            logger.info("Not saving the model/tokenizer since save_to=none")
            return
        assert model_name is not None
        if save_to in (SaveTo.DISK, SaveTo.BOTH):
            self._save_model_to_disk(models_path, model_name, revision)
        if save_to in (SaveTo.HF, SaveTo.BOTH):
            self._save_model_to_hf(model_name, revision)
        if save_to == SaveTo.HF_ELSE_DISK:
            try:
                self._save_model_to_hf(model_name, revision)
            except Exception as error:
                logger.error(
                    "Failed to save to HuggingFace, saving to disk instead: %s", error
                )
                self._save_model_to_disk(models_path, model_name, revision)

    def _save_model_to_disk(self, models_path: Path, model_name: str, revision: str):
        output_dir = generate_model_save_path(
            storage_path=models_path,
            model_name=model_name,
            revision=revision,
        )
        if wandb.run is not None:
            wandb.run.summary["saved_dir"] = str(output_dir)

        logger.info("Saving the model/tokenizer to %s", output_dir)
        self.model_state.wrapped_model.save_local(output_dir=output_dir)

    def _save_model_to_hf(self, model_name: str, revision: str) -> None:
        hf_name = f"AlignmentResearch/robust_llm_{model_name}"
        self.model_state.wrapped_model.push_to_hub(
            repo_id=hf_name,
            revision=revision,
            retries=self.training_config.upload_retries,
            cooldown_seconds=self.training_config.upload_cooldown,
        )

        # Record the saving on wandb.
        logger.info(
            "Saving the model/tokenizer to HuggingFace as %s, revision %s",
            hf_name,
            revision,
        )
        if wandb.run is not None:
            wandb.run.summary["saved_hf_name"] = hf_name


@dataclass
class AdversarialTrainingState(TrainingPipelineState):
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
            attack_config=self.adv_config.training_attack,
            run_name="default-run",  # TODO: work out if we should keep run_name
            logging_name="TODO: Remove this",  # This is unused
            victim=self.model_state.wrapped_model,
        )

        attack_schedule = AttackSchedule(self.adv_config.attack_schedule, num_rounds)
        n_its = attack_schedule[self.training_round]
        log(f"Generating {num_examples} adversarial examples with {n_its} iterations.")

        with self.model_state.wrapped_model.flop_count_context() as attack_flops:
            attack_out = attack.get_attacked_dataset(input_rllm_dataset, n_its=n_its)
        log(f"Attack flops: {attack_flops.flops:.2E} flops.")
        self.flops += attack_flops.flops

        adv_examples = attack_out.dataset.as_adversarial_examples().for_training()
        log(f"Generated {len(adv_examples)} adversarial examples.")
        return adv_examples

    @override
    def get_revision(self) -> str:
        return f"adv-round-{self.training_round}"

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
