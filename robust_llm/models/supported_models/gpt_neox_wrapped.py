from __future__ import annotations

from accelerate import Accelerator
from transformers import (
    GPTNeoXPreTrainedModel,
    GPTNeoXTokenizerFast,
    PreTrainedTokenizerBase,
)

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.model_utils import InferenceType
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
        train_minibatch_size: int,
        eval_minibatch_size: int,
    ) -> None:
        # TODO (ian): Decide whether this assert is worthwhile (it makes testing
        # harder).
        # assert isinstance(model, GPTNeoXPreTrainedModel)
        super().__init__(
            model,
            tokenizer,
            accelerator,
            inference_type,
            train_minibatch_size,
            eval_minibatch_size,
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
