from __future__ import annotations

import abc
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

import pandas as pd
import torch
from accelerate import Accelerator
from datasets import Dataset
from typing_extensions import override

from robust_llm.batch_job_utils import zero_pad
from robust_llm.config.configs import (
    AttackConfig,
    ExperimentConfig,
    get_checkpoint_path,
)
from robust_llm.dist_utils import (
    DistributedRNG,
    rmtree_if_exists,
    try_except_main_process_loop,
)
from robust_llm.logging_utils import format_attack_status, log
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.state_classes.rng_state import RNGState
from robust_llm.training.training_utils import (
    find_most_recent_checkpoint,
    get_sorted_checkpoints,
)

# Name of the subdirectory where the attack states are saved
ATTACK_STATE_SUBDIR = "attack_states"

AttackStateT = TypeVar("AttackStateT", bound="AttackState")


@dataclass(frozen=True)
class AttackedExample:
    """Tracks all state needed to record the result of an attack on an example.

    Attributes:
        example_index: The index of the example in the dataset.
        attacked_text: The final attacked text.
        iteration_texts: The text at each iteration of the attack.
        logits: The logits at each iteration of the attack (if available).
        flops: The number of FLOPs used in the attack.
    """

    example_index: int
    attacked_text: str
    iteration_texts: list[str]
    logits: list[list[float]] | list[None]
    flops: int

    def __post_init__(self):
        """Some sanity checks on the data."""
        if self.logits is not None:
            assert len(self.iteration_texts) == len(self.logits)
        assert self.attacked_text == self.iteration_texts[-1]


@dataclass(frozen=True)
class AttackState:
    """Tracks all state needed to run an attack.

    This class should be applicable for any attack that runs example-by example
    and iteration-by-iteration.
    """

    previously_attacked_examples: tuple[AttackedExample, ...]
    rng_state: RNGState

    def __post_init__(self):
        """Some sanity checks on the data, plus set the next example index."""
        # Check that the indices are contiguous and start from 0
        self._check_example_ordering()

    @property
    def example_index(self) -> int:
        return len(self.previously_attacked_examples)

    def attack_is_finished(self, dataset: Dataset) -> bool:
        """Check if the attack is finished."""
        if self.example_index > len(dataset):
            raise ValueError("Attack state has more examples than the dataset.")
        return self.example_index == len(dataset)

    def save(self, path: Path, process_index: int, accelerator: Accelerator):
        """Save the state to disk."""
        # Put the current next example index in the path for sorting.
        path = path / f"example_{zero_pad(self.example_index, 5)}"
        if accelerator.is_main_process:
            path.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()
        log(f"Saving checkpoint to {path}", main_process_only=False)

        # Save the examples and the rng state separately
        self.rng_state.save(path, process_index, accelerator)

        # Save the examples
        examples_path = path / f"examples_{process_index}.pkl"
        with examples_path.open("wb") as f:
            pickle.dump(self.previously_attacked_examples, f)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            with open(path / "save_complete", "w") as f:
                f.write("")

    @classmethod
    def load(
        cls, path: Path, process_index: int, accelerator: Accelerator
    ) -> AttackState:
        """Load the state from disk."""
        # Load the examples and the rng state separately
        rng_state = RNGState.load(path, process_index, accelerator)
        rng_state.set_random_states()

        # Load the examples
        examples_path = path / f"examples_{process_index}.pkl"
        with examples_path.open("rb") as f:
            previously_attacked_examples = pickle.load(f)

        return cls(previously_attacked_examples, rng_state)

    def get_attacked_input_texts(self):
        """Get the attacked texts from the examples."""
        # TODO(ian): Remove this check, it's redundant with the one in __post_init__
        self._check_example_ordering()
        return [e.attacked_text for e in self.previously_attacked_examples]

    def get_logits_cache(self) -> list[list[list[float]]] | None:
        """Get the logits cache from the examples."""
        # TODO(ian): Remove this check, it's redundant with the one in __post_init__
        self._check_example_ordering()
        logits_cache = [e.logits for e in self.previously_attacked_examples]
        return logits_cache if any(logits_cache) else None  # type: ignore

    def get_all_iteration_texts(self):
        """Get all the iteration texts from the examples."""
        # TODO(ian): Remove this check, it's redundant with the one in __post_init__
        self._check_example_ordering()
        return [e.iteration_texts for e in self.previously_attacked_examples]

    def get_attack_flops(self):
        """Get the total number of FLOPs used in the attack."""
        return sum(e.flops for e in self.previously_attacked_examples)

    def _check_example_ordering(self) -> None:
        """Sanity check that the examples are ordered correctly"""
        example_indices = [e.example_index for e in self.previously_attacked_examples]
        assert example_indices == list(range(len(self.previously_attacked_examples)))


def extract_attack_config(args: ExperimentConfig, training: bool) -> AttackConfig:
    if training:
        assert args.training is not None
        assert args.training.adversarial is not None
        return args.training.adversarial.training_attack
    else:
        assert args.evaluation is not None
        return args.evaluation.evaluation_attack


@dataclass
class AttackedRawInputOutput:
    """Contains a record of how the attack did at each iteration.

    Includes the attack strings. This is used after the fact to compute
    robustness metrics.

    Attributes:
        iteration_texts: (n_examples, n_iterations) nested lists of attacked texts
        logits: (n_examples, n_iterations, n_classes) nested lists of logits. Only
            saved in the classification setting.
    """

    iteration_texts: list[list[str]] = field(default_factory=list)
    logits: list[list[list[float]]] | None = None

    @classmethod
    def from_dfs(cls, dfs_dict: dict[int, pd.DataFrame]) -> "AttackedRawInputOutput":
        """Constructs an AttackedRawInputOutput object from a dictionary of DataFrames.

        Args:
            dfs_dict: A dictionary of DataFrames, where the keys are indices
                from the original dataset and the DataFrames have columns
                "iteration_texts" and "logits".
        """
        iteration_texts = []
        logits = []
        for example_ix, example_df in dfs_dict.items():
            iteration_texts.append(example_df["iteration_texts"].tolist())
            if "logits" in example_df.columns:
                logits.append(example_df["logits"].tolist())
        if logits == []:
            out_logits = None
        else:
            out_logits = logits

        assert out_logits is None or len(out_logits) == len(iteration_texts)
        return cls(iteration_texts=iteration_texts, logits=out_logits)

    def to_dfs(self) -> list[pd.DataFrame]:
        dfs = []
        if self.logits is None:
            for iteration_texts in self.iteration_texts:
                df = pd.DataFrame({"iteration_texts": iteration_texts})
                dfs.append(df)
            return dfs
        else:
            for iteration_texts, logits in zip(self.iteration_texts, self.logits):
                df = pd.DataFrame(
                    {"iteration_texts": iteration_texts, "logits": logits}
                )
                dfs.append(df)
            return dfs


@dataclass
class AttackOutput:
    """Wraps all output from get_attacked_dataset.

    Attributes:
        dataset:
            The dataset of adversarial examples.
        attack_data:
            A record of how the attack did at each iteration, to be used to
            compute robustness metrics.
        flops:
            The number of FLOPs used by the attack.
        global_info:
            A dictionary of additional information about the attack as a whole,
            to be logged.
        per_example_info:
            A dictionary of additional information about each example, to be
            logged alongside the individual examples.
    """

    dataset: RLLMDataset
    attack_data: AttackedRawInputOutput | None
    flops: int
    global_info: dict[str, Any] = field(default_factory=dict)
    per_example_info: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self):
        dataset_len = len(self.dataset.ds)
        for key, value in self.per_example_info.items():
            if len(value) != dataset_len:
                raise ValueError(
                    f"Length of per_example_info[{key}] ({len(value)})"
                    f" does not match length of dataset ({dataset_len})"
                )


class Attack(Generic[AttackStateT], abc.ABC):
    """Base class for all attacks.

    Attributes:
        CAN_CHECKPOINT: Whether the attack type supports checkpointing.
        attack_config: Configuration for the attack.
        attack_state_class: The AttackState subclass to use for this attack.
    """

    CAN_CHECKPOINT: bool

    def __init__(
        self,
        exp_config: ExperimentConfig,
        victim: WrappedModel,
        is_training: bool,
    ) -> None:
        """Constructor for the Attack class.

        Args:
            exp_config: ExperimentConfig object containing the configuration for the
                attack.
            victim: The model to be attacked.
            is_training: Whether the attack is being used for training or evaluation.
        """
        assert victim.accelerator is not None, "Accelerator must be provided"
        self.exp_config = exp_config
        self.attack_config = extract_attack_config(exp_config, is_training)
        self.victim = victim
        save_limit_gt_0 = self.attack_config.save_total_limit > 0
        self.can_checkpoint = save_limit_gt_0 and self.CAN_CHECKPOINT

    @abc.abstractmethod
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
        n_its: int,
        resume_from_checkpoint: bool = True,
        epoch: int | None = None,
    ) -> AttackOutput:
        """Produces a dataset of adversarial examples.

        Args:
            dataset: RLLMDataset of original examples to start from.
            n_its: Number of iterations to run the attack.
            resume_from_checkpoint: Whether to resume from the last checkpoint.
            epoch: The current epoch number if in adv training, otherwise None. Used
                to determine the save path for checkpoints.

        Subclasses should return:
            An AttackOutput object.
        """

    @property
    def attack_state_class(self) -> type[AttackState]:
        """The AttackState class to use for this attack.

        TODO(ian): Find a cleaner way to get this information.
        """
        return AttackState

    def get_first_attack_state(
        self, resume_from_checkpoint: bool, epoch: int | None
    ) -> AttackState:
        accelerator = self.victim.accelerator
        assert accelerator is not None

        if resume_from_checkpoint and not self.attack_config.disable_checkpointing:
            try:
                checkpoint_path = self.get_attack_checkpoint_path(epoch)
                # This was written for train states but works for attack states too
                subdir = find_most_recent_checkpoint(checkpoint_path)

                attack_state = self.attack_state_class.load(
                    subdir,
                    accelerator.process_index,
                    accelerator,
                )
                return attack_state
            except FileNotFoundError:
                log(
                    "No attack state found, starting from scratch.",
                    main_process_only=False,
                )

        return self.initialize_attack_state()

    def initialize_attack_state(self) -> AttackState:
        """Get a fresh attack state."""
        dist_rng = DistributedRNG(self.attack_config.seed, self.victim.accelerator)
        rng_state = RNGState(
            torch_rng_state=torch.random.get_rng_state(),
            distributed_rng=dist_rng,
        )
        cls = self.attack_state_class
        return cls(previously_attacked_examples=tuple(), rng_state=rng_state)

    def run_attack_loop(
        self,
        dataset: RLLMDataset,
        attack_state: AttackStateT,
        n_its: int,
        checkpoint_path: Path | None,
    ) -> AttackStateT:
        """Run the attack loop on the dataset.

        NOTE: Assumes that the attack is iterative. This might not always be true, and
        would require a refactor to fix.

        Args:
            dataset: The dataset to attack.
            attack_state: The current state of the attack.
            n_its: The number of iterations to run the attack.
            checkpoint_path: The path to save the attack state to (None to
                disable checkpoint saving).
        """
        save_steps = self.attack_config.save_steps
        accelerator = self.victim.accelerator
        assert accelerator is not None

        start_time = time.perf_counter()
        ATTACK_LOG_INTERVAL_SECS = 30  # Seconds between progress updates
        last_log_time = start_time - ATTACK_LOG_INTERVAL_SECS - 1  # Force a log
        start_index = attack_state.example_index
        total_examples = len(dataset.ds)

        while not attack_state.attack_is_finished(dataset.ds):
            attack_state = self.run_attack_on_example(
                # TODO(ian): Work out how to make sure the dataset doesn't
                # change when reloading from disk. (Maybe track dataset fingerprint?)
                dataset,
                n_its,
                attack_state,
            )
            if (
                checkpoint_path is not None
                and not self.attack_config.disable_checkpointing
                and attack_state.example_index % save_steps == 0
            ):
                attack_state.save(
                    checkpoint_path,
                    accelerator.process_index,
                    accelerator,
                )
                self.cleanup_checkpoints(checkpoint_path)

            # Log if it's been at least ATTACK_LOG_INTERVAL_SECS seconds.
            current_time = time.perf_counter()
            if (
                current_time - last_log_time > ATTACK_LOG_INTERVAL_SECS
                or attack_state.attack_is_finished(dataset.ds)  # Always log at the end
            ):
                status = format_attack_status(
                    total_examples=total_examples,
                    current_index=attack_state.example_index,
                    start_index=start_index,
                    current_time=current_time,
                    start_time=start_time,
                )
                log(status)
                last_log_time = current_time

        return attack_state

    @abc.abstractmethod
    def run_attack_on_example(
        self, dataset: RLLMDataset, n_its: int, attack_state: AttackStateT
    ) -> AttackStateT:
        """Attacks a single example.

        NOTE: Assumes that the attack is iterative. This might not always be true, and
        would require a refactor to fix.
        """

    def get_attack_checkpoint_path(self, epoch: int | None) -> Path:
        """Get the path to save the attack state."""
        checkpoint_path = get_checkpoint_path(self.exp_config)
        if epoch is not None:
            checkpoint_path = checkpoint_path / f"epoch_{zero_pad(epoch)}"
        checkpoint_path = checkpoint_path / ATTACK_STATE_SUBDIR
        return checkpoint_path

    def _cleanup_checkpoints(self, checkpoint_path: Path):
        """Delete old checkpoints to save disk space.

        The process is: iterate in reverse order through the checkpoints, and
        once we have found save_total_limit checkpoints that are safely saved,
        we delete all older checkpoints.
        """
        if not checkpoint_path.exists():
            return

        save_total_limit = self.attack_config.save_total_limit
        safe_checkpoint_epochs: list[str] = []
        log(f"Cleaning up checkpoints in {checkpoint_path}", main_process_only=False)
        for subdir in get_sorted_checkpoints(checkpoint_path):
            if len(safe_checkpoint_epochs) >= save_total_limit:
                assert checkpoint_path in subdir.parents
                assert subdir.name not in safe_checkpoint_epochs
                rmtree_if_exists(subdir)
            elif (subdir / "save_complete").exists():
                safe_checkpoint_epochs.append(subdir.name)
            else:
                log(
                    f"Deleting incomplete checkpoint: {subdir}",
                    main_process_only=False,
                )
                assert checkpoint_path in subdir.parents
                assert subdir.name not in safe_checkpoint_epochs
                rmtree_if_exists(subdir)
        log(
            f"Keeping checkpoints: {safe_checkpoint_epochs}",
            main_process_only=False,
        )

    def cleanup_checkpoints(self, checkpoint_path: Path):
        return try_except_main_process_loop(
            retries=3,
            cooldown_seconds=5,
            accelerator=self.victim.accelerator,
            func=self._cleanup_checkpoints,
            checkpoint_path=checkpoint_path,
        )


class IdentityAttack(Attack):
    """Returns the original dataset.

    A trivial 'attack' that could be used for debugging.
    """

    CAN_CHECKPOINT = False

    @override
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
        n_its: int = 1,
        resume_from_checkpoint: bool = True,
        epoch: int | None = None,
    ) -> AttackOutput:
        dataset = dataset.with_attacked_text(
            dataset.ds["text"],
        )
        attack_out = AttackOutput(
            dataset=dataset,
            attack_data=None,
            flops=0,
        )
        return attack_out

    def run_attack_on_example(
        self, dataset: RLLMDataset, n_its: int, attack_state: AttackStateT
    ) -> AttackStateT:
        """Unused in IdentityAttack."""
        raise NotImplementedError("IdentityAttack does not need run_attack_on_example.")


class PromptAttackMode(Enum):
    """Enum class for prompt attack modes.

    Currently this just covers single vs multi-prompt.
    """

    SINGLEPROMPT = "single-prompt"
    MULTIPROMPT = "multi-prompt"
