from __future__ import annotations

from typing import Literal

from accelerate import Accelerator
from transformers import (
    GemmaPreTrainedModel,
    GemmaTokenizerFast,
    PreTrainedTokenizerBase,
)
from typing_extensions import override

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.prompt_templates import Conversation
from robust_llm.models.wrapped_chat_model import WrappedChatModel
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("gemma-chat")
class GemmaChatModel(WrappedChatModel):
    # Context length from:
    # https://huggingface.co/google/gemma-1.1-2b-it/blob/main/config.json
    # https://huggingface.co/google/gemma-2-9b-it/blob/main/config.json
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
        family: Literal["gemma-chat"],
        system_prompt: str | None = None,
        seed: int = 0,
    ) -> None:
        if system_prompt is not None:
            raise ValueError("GemmaChatModel does not support system_prompt.")
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
            seed=seed,
        )

    @override
    def forward(self, **inputs):
        # Gemma is idiosyncratic in that it requires use_cache=True to use
        # provided past_key_values. This is not the default behavior for other
        # models (except Qwen), where use_cache indicates whether to return
        # past_key_values.
        if "past_key_values" in inputs:
            inputs["use_cache"] = True
        return super().forward(**inputs)

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

    @override
    def init_conversation(self) -> Conversation:
        return Conversation(
            prompt_prefix="<bos>",
            system_prefix="",
            system_suffix="",
            user_prefix="<start_of_turn>user\n",
            user_suffix="<end_of_turn>\n",
            assistant_prefix="<start_of_turn>model\n",
            assistant_suffix="<end_of_turn>\n",
            system_prompt=self.system_prompt,
        )
