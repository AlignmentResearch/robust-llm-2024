import abc
from typing import Optional

from datasets import Dataset
from typing_extensions import override

from robust_llm.configs import AttackConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec


class Attack(abc.ABC):
    """Base class for all attacks."""

    # Whether the attack always requires an input to `get_attacked_dataset` method.
    REQUIRES_INPUT_DATASET: bool

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
            A dataset of adversarial examples containing at least `text` and `label`
            columns, with `text` containing the attacked text and `label` containing the
            ORIGINAL label (which might no longer be correct). If the `dataset` argument
            was specified, the returned dataset must also contain the `original_text`
            column, containing unmodified original text.
        """
        pass


class IdentityAttack(Attack):
    """Returns the original dataset.

    A trivial 'attack' that could be used for debugging.
    """

    REQUIRES_INPUT_DATASET = True

    @override
    def get_attacked_dataset(
        self,
        dataset: Optional[Dataset],
        max_n_outputs: Optional[int] = None,
    ) -> Dataset:
        assert dataset is not None

        dataset = dataset.add_column(
            "original_text",
            dataset["text"],
            new_fingerprint=None,  # type: ignore
        )

        if max_n_outputs is not None:
            dataset = dataset.select(range(max_n_outputs))

        return dataset
