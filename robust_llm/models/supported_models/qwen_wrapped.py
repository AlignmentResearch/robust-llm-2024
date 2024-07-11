from __future__ import annotations

from typing import Literal

from accelerate import Accelerator
from transformers import (
    PreTrainedTokenizerBase,
    Qwen2PreTrainedModel,
    Qwen2TokenizerFast,
)
from typing_extensions import override

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("qwen1.5")
@WrappedModel.register_subclass("qwen2")
class QwenModel(WrappedModel):
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
        family: Literal["qwen1.5", "qwen2"],
        system_prompt: str | None = None,
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
            system_prompt=system_prompt,
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
