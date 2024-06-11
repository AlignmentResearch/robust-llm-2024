from __future__ import annotations

from accelerate import Accelerator
from transformers import (
    PreTrainedTokenizerBase,
    Qwen2PreTrainedModel,
    Qwen2TokenizerFast,
)
from typing_extensions import override

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.chat_templates import get_qwen_template
from robust_llm.models.model_utils import InferenceType, PromptTemplate
from robust_llm.models.wrapped_chat_model import WrappedChatModel
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("qwen-chat")
class QwenChatModel(WrappedChatModel):
    CONTEXT_LENGTH = 32768

    def __init__(
        self,
        model: Qwen2PreTrainedModel,
        right_tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator | None,
        inference_type: InferenceType,
        train_minibatch_size: int,
        eval_minibatch_size: int,
        generation_config: GenerationConfig | None,
    ) -> None:
        super().__init__(
            model,
            right_tokenizer,
            accelerator,
            inference_type,
            train_minibatch_size,
            eval_minibatch_size,
            generation_config=generation_config,
        )

    @override
    def forward(self, **inputs):
        # Qwen is idiosyncratic in that it requires use_cache=True to use provided
        # past_key_values. This is not the default behavior for other models, where
        # use_cache indicates whether to return past_key_values.
        if "past_key_values" in inputs:
            inputs["use_cache"] = True
        return super().forward(**inputs)

    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> Qwen2TokenizerFast:
        """Load the tokenizer."""
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            model_config.name_or_path,
            revision=model_config.revision,
            padding_side="right",  # Left padding is handled separately
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=False,
        )
        assert isinstance(tokenizer, Qwen2TokenizerFast)  # for type-checking

        return tokenizer

    @override
    def get_prompt_template(
        self,
        unmodifiable_prefix: str,
        modifiable_infix: str,
        unmodifiable_suffix: str,
    ) -> PromptTemplate:
        """Returns a PromptTemplate for the given text chunks."""
        return get_qwen_template(
            unmodifiable_prefix,
            modifiable_infix,
            unmodifiable_suffix,
        )
