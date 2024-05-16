"""Base class for model defense and defense factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from robust_llm.config.defense_configs import DefenseConfig
from robust_llm.config.model_configs import ModelConfig
from robust_llm.models import WrappedModel
from robust_llm.models.model_utils import SuppressPadTokenWarning


class Defenses(Enum):
    IDENTITY = "identity"
    PERPLEXITY = "perplexity"
    RETOKENIZATION = "retokenization"
    PARAPHRASE = "paraphrase"


class DefendedModel(WrappedModel, ABC):
    """Wrapper class for WrappedModel modified to be robust to attacks that is
    also a subclass of WrappedModel.

    The wrapper is designed to be subclassed by specific defenses, which can then
    override the `forward` and `__init__` methods to implement the defense.

    Args:
        victim: The model to be defended.
    """

    def __init__(self, victim: WrappedModel):
        self._underlying_model = victim

    @property
    @abstractmethod
    def defense_config(self) -> DefenseConfig:
        """Return the DefenseConfig of the defense.

        The reason we make this a property rather than having it in init is that
        we want the DefendedModel subclasses to be able to use their respective
        configs without having to typecheck that self.defense_config is the
        correct subclass of DefenseConfig.
        """
        pass

    @property
    def model(self) -> PreTrainedModel:
        return self._underlying_model.model

    @property
    def accelerator(self):
        return self._underlying_model.accelerator

    @property
    def tokenizer(self):
        return self._underlying_model.tokenizer

    @property
    def inference_type(self):
        return self._underlying_model.inference_type

    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> PreTrainedTokenizerBase:
        raise NotImplementedError(
            "DefendedModel can't load a tokenizer;"
            " this should be done by the underlying WrappedModel."
        )

    def call_model(
        self,
        inp: torch.Tensor | None = None,
        add_cls: bool = True,
        add_sep: bool = True,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Call the underlying model with the given inputs.

        Currently assumes that we don't have to do anything with special tokens.

        TODO (ian): Deprecate this method to reduce code duplication.
        """
        assert add_cls is False
        assert add_sep is False
        # return _call_model(self, inp, inputs_embeds)

        assert (inp is not None) != (
            inputs_embeds is not None
        ), "exactly one of inp, inputs_embeds must be provided"

        if inp is not None:
            return self(input_ids=inp).logits

        if inputs_embeds is not None:
            with SuppressPadTokenWarning(self):
                return self(inputs_embeds=inputs_embeds).logits

        raise ValueError("exactly one of inp, inputs_embeds must be provided")
