import abc
from typing import Any, Dict, Optional, Tuple

from datasets import Dataset
from typing_extensions import override

from robust_llm.configs import AttackConfig, EnvironmentConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec
from robust_llm.logging_utils import LoggingCounter


class Attack(abc.ABC):
    """Base class for all attacks."""

    # Whether the attack always requires an input to `get_attacked_dataset` method.
    REQUIRES_INPUT_DATASET: bool
    # Whether the attack needs to be trained before it can be effectively used.
    REQUIRES_TRAINING: bool

    def __init__(
        self,
        attack_config: AttackConfig,
        environment_config: EnvironmentConfig,
        modifiable_chunks_spec: ModifiableChunksSpec,
        logging_name: Optional[str] = None,
    ) -> None:
        """Constructor for the Attack class.

        Args:
            attack_config: Configuration for the attack.
            environment_config: Configuration for the environment.
            modifiable_chunks_spec: Tuple of bools specifying which chunks of the
                original text can be modified. For example, when (True,), the whole
                text can be modified. When (False, True), only the second part of
                the text can be modified for each example.
            logging_name: Name of the attack, for the purposes of logging. Possible
                examples include "training_attack" or "validation_attack".
        """
        self.attack_config = attack_config
        self.environment_config = environment_config
        self.modifiable_chunks_spec = modifiable_chunks_spec
        self.logging_name = logging_name

        if self.REQUIRES_TRAINING and self.attack_config.log_frequency is not None:
            assert logging_name is not None
            self.logging_counter = LoggingCounter(_name=logging_name)

    @abc.abstractmethod
    def get_attacked_dataset(
        self,
        dataset: Optional[Dataset],
        max_n_outputs: Optional[int] = None,
    ) -> Tuple[Dataset, Dict[str, Any]]:
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
            A tuple `(dataset, info_dict)` where:
            `dataset` is a dataset of adversarial examples containing at least `text`
            and `label` columns, with `text` containing the attacked text and `label`
            containing the ORIGINAL label (which might no longer be correct). If the
            `dataset` argument was specified, the returned dataset must also contain
            the `original_text` column, containing unmodified original text.
            `info_dict` is a dictionary of additional information (e.g. metrics) about
            the attack.
        """
        pass

    def train(self, dataset: Dataset) -> None:
        """Trains the attack on the given dataset.

        This method need only be overridden (and called)
        when `REQUIRES_TRAINING` is True.

        Args:
            dataset: Dataset of examples to train the attack on.
                Requires `text`, `text_chunked`, and `label` columns.
        """
        raise NotImplementedError


class IdentityAttack(Attack):
    """Returns the original dataset.

    A trivial 'attack' that could be used for debugging.
    """

    REQUIRES_INPUT_DATASET = True
    REQUIRES_TRAINING = False

    @override
    def get_attacked_dataset(
        self,
        dataset: Optional[Dataset],
        max_n_outputs: Optional[int] = None,
    ) -> Tuple[Dataset, Dict[str, Any]]:
        assert dataset is not None

        dataset = dataset.add_column(
            "original_text",
            dataset["text"],
            new_fingerprint=None,  # type: ignore
        )

        if max_n_outputs is not None:
            dataset = dataset.select(range(max_n_outputs))

        return dataset, {}
