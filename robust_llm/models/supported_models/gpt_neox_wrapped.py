from __future__ import annotations

import torch
from accelerate import Accelerator
from transformers import (
    GPTNeoXPreTrainedModel,
    GPTNeoXTokenizerFast,
    PreTrainedTokenizerBase,
)
from typing_extensions import override

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.model_utils import InferenceType, _call_model
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("gpt_neox")
class GPTNeoXModel(WrappedModel):
    # NOTE: Pythia models are based on GPTNeoX
    CONTEXT_LENGTH = 2048

    def __init__(
        self,
        model: GPTNeoXPreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator | None,
        inference_type: InferenceType,
    ) -> None:
        # TODO (ian): Decide whether this assert is worthwhile (it makes testing
        # harder).
        # assert isinstance(model, GPTNeoXPreTrainedModel)
        super().__init__(
            model,
            tokenizer,
            accelerator,
            inference_type,
        )
        # Special setup needed for pythia.
        self.model.config.pad_token_id = model.config.eos_token_id

    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> GPTNeoXTokenizerFast:
        """Load the tokenizer."""
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(
            model_config.name_or_path,
            revision=model_config.revision,
            padding_side=model_config.padding_side,
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=False,
        )
        assert isinstance(tokenizer, GPTNeoXTokenizerFast)  # for type-checking

        # Special setup needed for pythia.
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @override
    def call_model(
        self,
        inp: torch.Tensor | None = None,
        add_cls: bool = False,
        add_sep: bool = False,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # add_cls and add_sep are ignored but included for consistency with
        # other models.
        assert add_cls is False
        assert add_sep is False
        return _call_model(self.model, inp, inputs_embeds)
