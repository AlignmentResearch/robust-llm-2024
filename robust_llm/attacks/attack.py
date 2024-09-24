import abc
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd
import wandb
from typing_extensions import override

from robust_llm import logger
from robust_llm.config.configs import AttackConfig
from robust_llm.dist_utils import DistributedRNG, is_main_process
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset

ATTACK_STATE_NAME = "attack_state.json"
CONFIG_NAME = "config.json"


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


@dataclass
class AttackData:
    """Contains a record of how the attack did at each iteration.

    Includes the attack strings. This is used after the fact to compute
    robustness metrics.

    NOTE: iteration_texts and logits are stored per-example. In other words,
    the shape of iteration_texts should be (n_examples, n_iterations), and
    the shape of logits should be (n_examples, n_iterations, n_classes).

    TODO(ian): Add attributes and come up with a more informative name.
    """

    iteration_texts: list[list[str]] = field(default_factory=list)
    logits: list[list[list[float]]] | None = None

    def to_wandb_tables(self) -> list[wandb.Table]:
        tables = []
        n_examples = len(self.iteration_texts)
        for example_ix in range(n_examples):
            n_its = len(self.iteration_texts[example_ix])

            if self.logits is None:
                table = wandb.Table(columns=["iteration_texts"])
                for iteration in range(n_its):
                    table.add_data(self.iteration_texts[example_ix][iteration])

            else:
                table = wandb.Table(columns=["iteration_texts", "logits"])
                for iteration in range(n_its):
                    table.add_data(
                        self.iteration_texts[example_ix][iteration],
                        self.logits[example_ix][iteration],
                    )
            tables.append(table)
        return tables

    @classmethod
    def from_dfs(cls, dfs_dict: dict[int, pd.DataFrame]) -> "AttackData":
        """Constructs an AttackData object from a dictionary of DataFrames.

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
    attack_data: AttackData | None
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
        logging_name: Name of the attack, for the purposes of logging. Possible
            examples include "training_attack" or "validation_attack".
    """

    CAN_CHECKPOINT: bool

    def __init__(
        self,
        attack_config: AttackConfig,
        victim: WrappedModel,
        run_name: str,
        logging_name: Optional[str] = None,
    ) -> None:
        """Constructor for the Attack class.

        Args:
        attack_config: Configuration for the attack.
        victim: The model to be attacked.
        run_name: Name of the run, for the purposes of saving checkpoints.
            This is unrelated to the wandb `name` used elsewhere.
        logging_name: Name of the attack, for the purposes of logging. Possible
            examples include "training_attack" or "validation_attack".
        """
        assert victim.accelerator is not None, "Accelerator must be provided"
        self.attack_config = attack_config
        self.victim = victim
        self.run_name = run_name
        self.rng = DistributedRNG(attack_config.seed, victim.accelerator)
        self.logging_name = logging_name
        self.attack_state = AttackState(rng_state=self.rng.getstate())
        save_limit_gt_0 = self.attack_config.save_total_limit > 0
        run_name_not_default = self.run_name != "default-run"
        self.can_checkpoint = (
            save_limit_gt_0 and run_name_not_default and self.CAN_CHECKPOINT
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

    def states_directory(self) -> str:
        return os.path.join(self.attack_config.save_prefix, self.run_name)

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
        output_dir = os.path.join(
            self.states_directory(), f"checkpoint-{self.attack_state.example_index}"
        )
        logger.debug(f"Saving state to {output_dir}")
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, ATTACK_STATE_NAME), "w") as f:
            json.dump(asdict(self.attack_state), f)
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
                shutil.rmtree(os.path.join(self.states_directory(), path))

    def maybe_load_state(self) -> bool:
        """Loads the most recent state from disk if one exists.

        Returns:
            True if a state was loaded, False otherwise.
        """
        if not self.can_checkpoint:
            return False
        for checkpoint in self.list_checkpoints():
            state_path = os.path.join(
                self.states_directory(), checkpoint, ATTACK_STATE_NAME
            )
            with open(state_path) as f:
                attack_state = json.load(f)
            for key, value in attack_state.items():
                setattr(self.attack_state, key, value)
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
            logger.info(f"Loaded state from {state_path}")
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
