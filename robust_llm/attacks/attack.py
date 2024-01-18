import abc
from enum import Enum, auto
from typing import Optional

from datasets import Dataset

from robust_llm.configs import AttackConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec


class SingleAttackResult(Enum):
    """Result of a single trial for a single data point."""

    # Example went from correctly classified into incorrectly classified
    SUCCESS = auto()
    # Example remained correctly classified
    FAILURE = auto()
    # Example was skipped because it was already incorrectly classified
    SKIPPED = auto()


class Attack(abc.ABC):
    """Base class for all attacks."""

    def __init__(
        self,
        attack_config: AttackConfig,
        modifiable_chunks_spec: ModifiableChunksSpec,
    ) -> None:
        """Constructor for the Attack class.

        Args:
            attack_config: configuration for the attack
            modifiable_chunks_spec: Tuple of bools specifying which chunks of the
                original text can be modified. For example, when (True,), the whole
                text can be modified. When (False, True), only the second part of
                the text can be modified for each example.
        """
        self.attack_config = attack_config
        self.modifiable_chunks_spec = modifiable_chunks_spec

    @abc.abstractmethod
    def get_attacked_dataset(
        self,
        dataset: Optional[Dataset],
        max_n_outputs: Optional[int] = None,
    ) -> Dataset:
        """Produces a dataset of adversarial examples.

        Args:
            dataset: dataset of original examples to start from. `text` and `label`
                columns are required. If there are non-trivial restrictions on
                modifying the text, i.e. `len(self.modifiable_chunks_spec) > 1`, then
                `text_chunked` column is also required, which for every example
                contains a list of strings, being the chunks of the original text
                corresponding to the `self.modifiable_chunks_spec` specification.
            max_n_outputs: If specified, the attack will try to return up to
                `max_n_outputs` examples. Defaults to None.

        Returns:
            A dataset of adversarial examples with `text` and `label` columns.
            Optionally, an `attack_result` column can be included, which contains
            `SingleAttackResult` values for each example.
        """
        pass


class IdentityAttack(Attack):
    """Returns the original dataset.

    A trivial 'attack' that could be used for debugging.
    """

    def get_attacked_dataset(
        self,
        dataset: Optional[Dataset],
        max_n_outputs: Optional[int] = None,
    ) -> Dataset:
        assert dataset is not None

        if max_n_outputs is not None:
            dataset = dataset.select(range(max_n_outputs))

        return dataset
