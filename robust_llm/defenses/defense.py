"""Base class for model defense and defense factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from robust_llm.config.defense_configs import DefenseConfig
from robust_llm.config.model_configs import ModelConfig
from robust_llm.models import WrappedModel


class Defenses(Enum):
    IDENTITY = "identity"
    PERPLEXITY = "perplexity"
    RETOKENIZATION = "retokenization"
    PARAPHRASE = "paraphrase"


class DefendedModel(WrappedModel, ABC):
    """Wrapper class for WrappedModel modified to be robust to attacks.

    This class itself is also a subclass of WrappedModel.

    The wrapper is designed to be subclassed by specific defenses, which can then
    override the `forward` and `__init__` methods to implement the defense.

    Args:
        victim: The model to be defended.
    """

    def __init__(self, victim: WrappedModel):
        self._underlying_model = victim

    @property
    def family(self):
        return self._underlying_model.family

    @property
    @abstractmethod
    def defense_config(self) -> DefenseConfig:
        """Return the DefenseConfig of the defense.

        The reason we make this a property rather than having it in init is that
        we want the DefendedModel subclasses to be able to use their respective
        configs without having to typecheck that self.defense_config is the
        correct subclass of DefenseConfig.
        """

    @property
    def model(self) -> PreTrainedModel:
        return self._underlying_model.model

    @property
    def accelerator(self):
        return self._underlying_model.accelerator

    @property
    def right_tokenizer(self):
        return self._underlying_model.right_tokenizer

    @cached_property
    def left_tokenizer(self):
        return self._underlying_model.left_tokenizer

    @property
    def inference_type(self):
        return self._underlying_model.inference_type

    @property
    def train_minibatch_size(self):
        return self._underlying_model.train_minibatch_size

    @property
    def eval_minibatch_size(self):
        return self._underlying_model.eval_minibatch_size

    @property
    def n_params(self):
        return self._underlying_model.n_params

    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> PreTrainedTokenizerBase:
        raise NotImplementedError(
            "DefendedModel can't load a tokenizer;"
            " this should be done by the underlying WrappedModel."
        )


class FilteringDefendedModel(DefendedModel):
    """A DefendedModel that works by filtering out adversarial examples.

    FilteringDefendedModels don't modify the forward method but instead have a
    separate filter method that takes a list of text inputs and returns a list
    of booleans indicating whether the defense flagged each input as
    adversarial.
    """

    @abstractmethod
    def filter(self, text_inputs: list[str]) -> list[bool]:
        """Indicates whether each input should be filtered out.

        Args:
            text_inputs: The list of text inputs to filter.

        Subclasses should return:
            A list of booleans indicating whether the defense flagged each input
            as adversarial (True) or not (False).
        """


class MutatingDefendedModel(DefendedModel):
    """A DefendedModel that works by mutating the input.

    MutatingDefendedModels modify the forward method directly, changing the input
    before it is passed to the underlying model.
    """
