import abc
from enum import Enum
from typing import Any, Optional

from typing_extensions import override

from robust_llm.config.configs import AttackConfig
from robust_llm.logging_utils import LoggingCounter
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


class Attack(abc.ABC):
    """Base class for all attacks."""

    # Whether the attack needs to be trained before it can be effectively used.
    REQUIRES_TRAINING: bool

    def __init__(
        self,
        attack_config: AttackConfig,
        logging_name: Optional[str] = None,
    ) -> None:
        """Constructor for the Attack class.

        Args:
            attack_config: Configuration for the attack.
            logging_name: Name of the attack, for the purposes of logging. Possible
                examples include "training_attack" or "validation_attack".
        """
        self.attack_config = attack_config
        self.logging_name = logging_name

        if self.REQUIRES_TRAINING and self.attack_config.log_frequency is not None:
            assert logging_name is not None
            self.logging_counter = LoggingCounter(_name=logging_name)

    @abc.abstractmethod
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
    ) -> tuple[RLLMDataset, dict[str, Any]]:
        """Produces a dataset of adversarial examples.

        Args:
            dataset: RLLMDataset of original examples to start from.
        Returns:
            A tuple `(dataset, info_dict)` where:
                `dataset` is an RLLMDataset of adversarial examples containing at
                    least `text`, `attacked_text`, and `clf_label` columns, with `text`
                    containing the original text, `attacked_text` containing the
                    attacked text, and `attacked_clf_label` containing the
                    (potentially adjusted) attacked label
                `info_dict` is a dictionary of additional information (e.g.
                    metrics) about the attack.
        """

    def train(self, dataset: RLLMDataset) -> None:
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

    REQUIRES_TRAINING = False

    @override
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
    ) -> tuple[RLLMDataset, dict[str, Any]]:
        dataset = dataset.with_attacked_text(
            dataset.ds["text"],
        )

        return dataset, {}


class PromptAttackMode(Enum):
    """Enum class for prompt attack modes.

    Currently this just covers single vs multi-prompt.
    """

    SINGLEPROMPT = "single-prompt"
    MULTIPROMPT = "multi-prompt"
