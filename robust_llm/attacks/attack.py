from __future__ import annotations

import abc
import importlib
import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Type, TypeVar

import pandas as pd
from typing_extensions import override

from robust_llm.config.configs import (
    AttackConfig,
    ExperimentConfig,
    get_checkpoint_path,
)
from robust_llm.dist_utils import DistributedRNG, dist_rmtree, is_main_process
from robust_llm.logging_utils import log
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset

T = TypeVar("T", bound="AttackState")

ATTACK_STATE_NAME = "attack_state.json"
CONFIG_NAME = "config.json"
STATES_NAME = "attack_states"
CLASS_KEY = "class_name"


def extract_attack_config(args: ExperimentConfig, training: bool) -> AttackConfig:
    if training:
        assert args.training is not None
        assert args.training.adversarial is not None
        return args.training.adversarial.training_attack
    else:
        assert args.evaluation is not None
        return args.evaluation.evaluation_attack


@dataclass
class AttackState:
    """State used for checkpointing attacks.

    Attributes:
        rng_state: State of the random number generator.
        example_index: Index of the current example being attacked.
        attacked_texts: List of final attacked texts for each example.
        all_iteration_texts: List of attacked texts for each iteration on each
            example, for use computing robustness metric.
        attacks_info: Dictionary of additional information about the attack.
    """

    rng_state: dict[str, Any]
    example_index: int = 0
    attacked_texts: list[str] = field(default_factory=list)
    all_iteration_texts: list[list[str]] = field(default_factory=list)
    attacks_info: dict[str, list[Any]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save the AttackState to a JSON file."""
        state_dict = asdict(self)
        # Add class information for proper loading
        state_dict[CLASS_KEY] = (
            f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls: Type[T], path: Path) -> T:
        """Load an AttackState from a JSON file."""
        with open(path, encoding="utf-8") as f:
            state_dict = json.load(f)

        # Get the correct class to instantiate
        class_path = state_dict.pop(CLASS_KEY)
        if class_path != f"{cls.__module__}.{cls.__qualname__}":
            # Import and use the correct class,
            # e.g. split `search_based.SearchBasedAttackState` into
            # module `search_based` and class `SearchBasedAttackState`
            module_name, class_name = class_path.rsplit(".", 1)

            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)

        return cls(**state_dict)


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
        global_info:
            A dictionary of additional information about the attack as a whole,
            to be logged.
        per_example_info:
            A dictionary of additional information about each example, to be
            logged alongside the individual examples.
    """

    dataset: RLLMDataset
    attack_data: AttackedRawInputOutput | None
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


class Attack(abc.ABC):
    """Base class for all attacks.

    Attributes:
        CAN_CHECKPOINT: Whether the attack type supports checkpointing.
        attack_config: Configuration for the attack.
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
        self.rng = DistributedRNG(self.attack_config.seed, victim.accelerator)
        self.attack_state = AttackState(rng_state=self.rng.getstate())
        save_limit_gt_0 = self.attack_config.save_total_limit > 0
        self.can_checkpoint = (
            save_limit_gt_0 and self.CAN_CHECKPOINT and not is_training
        )

    @abc.abstractmethod
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
        n_its: int,
        resume_from_checkpoint: bool = True,
    ) -> AttackOutput:
        """Produces a dataset of adversarial examples.

        Args:
            dataset: RLLMDataset of original examples to start from.
            n_its: Number of iterations to run the attack.
            resume_from_checkpoint: Whether to resume from the last checkpoint.

        Subclasses should return:
            A tuple `(dataset, info_dict)` where:
                `dataset` is an RLLMDataset of adversarial examples containing at
                    least `text`, `attacked_text`, and `clf_label` columns, with `text`
                    containing the original text, `attacked_text` containing the
                    attacked text, and `attacked_clf_label` containing the
                    (potentially adjusted) attacked label
                `info_dict` is a dictionary of additional information (e.g.
                    metrics) about the attack.
        """

    def states_directory(self) -> Path:
        return get_checkpoint_path(
            Path(self.attack_config.save_prefix) / STATES_NAME, self.exp_config
        )

    def list_checkpoints(self) -> list[str]:
        """Lists valid checkpoints, newest first."""
        checkpoint_dir = self.states_directory()
        if not os.path.exists(checkpoint_dir):
            return []
        checkpoints_with_attack_state = [
            f
            for f in os.listdir(checkpoint_dir)
            # Only iterate over directories that contain the attack state.
            if os.path.isdir(os.path.join(checkpoint_dir, f))
            and ATTACK_STATE_NAME in os.listdir(os.path.join(checkpoint_dir, f))
        ]
        return sorted(
            checkpoints_with_attack_state,
            # The key is used to sort the list by the number in the checkpoint name
            # in descending order.
            key=lambda x: int(x.split("-")[-1]),
        )[::-1]

    def _save_state(self):
        """Saves the attack state to disk."""
        output_dir = (
            self.states_directory() / f"checkpoint-{self.attack_state.example_index}"
        )
        log(f"Saving state to {output_dir}")
        os.makedirs(output_dir)
        self.attack_state.save(output_dir / ATTACK_STATE_NAME)
        with open(os.path.join(output_dir, CONFIG_NAME), "w") as f:
            json.dump(asdict(self.attack_config), f)
        self.clean_states()

    def maybe_save_state(self):
        assert self.victim.accelerator is not None
        if not is_main_process():
            return
        if not self.can_checkpoint:
            return
        if self.attack_state.example_index % self.attack_config.save_steps == 0:
            self._save_state()

    def clean_states(self) -> None:
        """Delete old checkpoints if necessary to stay within save_total_limit."""
        for i, path in enumerate(self.list_checkpoints()):
            if i >= self.attack_config.save_total_limit and is_main_process():
                dist_rmtree(os.path.join(self.states_directory(), path))

    def maybe_load_state(self) -> bool:
        """Loads the most recent state from disk if one exists.

        Returns:
            True if a state was loaded, False otherwise.
        """
        if not self.can_checkpoint:
            return False
        for checkpoint in self.list_checkpoints():
            state_path = self.states_directory() / checkpoint / ATTACK_STATE_NAME
            self.attack_state = AttackState.load(state_path)
            with open(
                os.path.join(self.states_directory(), checkpoint, CONFIG_NAME)
            ) as f:
                attack_config = json.load(f)
            current_config = asdict(self.attack_config)
            for key, value in attack_config.items():
                assert current_config[key] == value, (
                    f"Config mismatch: {key} is {getattr(self.attack_config, key)} "
                    f"but should be {value}"
                )
            log(f"Loaded state from {state_path}", main_process_only=False)
            return True
        return False


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
        resume_from_checkpoint: bool = False,
    ) -> AttackOutput:
        dataset = dataset.with_attacked_text(
            dataset.ds["text"],
        )
        attack_out = AttackOutput(
            dataset=dataset,
            attack_data=None,
        )
        return attack_out


class PromptAttackMode(Enum):
    """Enum class for prompt attack modes.

    Currently this just covers single vs multi-prompt.
    """

    SINGLEPROMPT = "single-prompt"
    MULTIPROMPT = "multi-prompt"
