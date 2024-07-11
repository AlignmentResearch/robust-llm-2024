from __future__ import annotations

from typing import Literal

from accelerate import Accelerator
from transformers import (
    GemmaPreTrainedModel,
    GemmaTokenizerFast,
    PreTrainedTokenizerBase,
)

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("gemma")
class GemmaModel(WrappedModel):
    # Context length from:
    # https://huggingface.co/google/gemma-1.1-2b/blob/main/config.json
    # https://huggingface.co/google/gemma-2-9b/blob/main/config.json
    CONTEXT_LENGTH = 8192

    def __init__(
        self,
        model: GemmaPreTrainedModel,
        right_tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator | None,
        inference_type: InferenceType,
        train_minibatch_size: int,
        eval_minibatch_size: int,
        generation_config: GenerationConfig | None,
        family: Literal["gemma"],
    ) -> None:
        super().__init__(
            model,
            right_tokenizer,
            accelerator,
            inference_type,
            train_minibatch_size,
            eval_minibatch_size,
            generation_config=generation_config,
            family=family,
        )

    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> GemmaTokenizerFast:
        """Load the tokenizer."""
        tokenizer = GemmaTokenizerFast.from_pretrained(
            model_config.name_or_path,
            revision=model_config.revision,
            padding_side="right",  # Left padding is handled separately
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=False,
        )
        assert isinstance(tokenizer, GemmaTokenizerFast)  # for type-checking

        return tokenizer
