"""Base class for model defense and defense factory."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

import torch
from datasets import Dataset
from torch.nn.parameter import Parameter
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

from robust_llm.configs import DefenseConfig
from robust_llm.utils import LanguageModel


class Defenses(Enum):
    IDENTITY = "identity"
    PERPLEXITY = "perplexity"
    RETOKENIZATION = "retokenization"
    PARAPHRASE = "paraphrase"


@dataclass
class DefendedModel(LanguageModel):
    """Wrapper class for model modified to be robust to attacks.

    The model can be any model that implements the `LanguageModel` protocol, but most
    likely it will be a `transformers.PreTrainedModel`.

    The wrapper is designed to be subclassed by specific defenses, which can then
    override the `__call__` and `__post_init__` methods to implement the defense.
    By default, it is an identity wrapper.

    Args:
        defense_config: config of the defense
        init_model: the model to be defended
        tokenizer: tokenizer used by the model
        dataset: dataset used to train or tune the defense, e.g. setting max perplexity
        decoder: additional model used in some defenses, such as to compute perplexity
    """

    defense_config: DefenseConfig
    init_model: LanguageModel
    tokenizer: PreTrainedTokenizerBase
    dataset: Optional[Dataset] = None
    decoder: Optional[PreTrainedModel] = None

    def __post_init__(self) -> None:
        """Perform any necessary post-initialization steps,
        in particular updating the model.
        """
        self._model = self.init_model

        # It was necessary to add this since being a pipeline
        # means you have a "can_generate" method
        self.can_generate = lambda: False

    def __call__(self, **inputs):
        return self.forward(**inputs)

    @property
    def config(self) -> PretrainedConfig:
        return self.model.config

    def to(self, *args, **kwargs):
        # For torch modules (but not tensors), "to" modifies in-place.
        self._model.to(*args, **kwargs)
        return self

    def forward(self, **inputs):
        """Run the inputs through self.model, with required safety considerations."""
        return self.model(**inputs)

    @property
    def model(self) -> LanguageModel:
        return self._model

    @property
    def training(self) -> bool:
        return self.model.training

    def eval(self) -> LanguageModel:
        self.model.eval()
        return self

    def train(self) -> LanguageModel:
        self.model.train()
        return self

    @property
    def device(self) -> torch.device:
        return self.model.device

    def parameters(self) -> Iterator[Parameter]:
        return self.model.parameters()
