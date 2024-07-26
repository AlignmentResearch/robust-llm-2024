from __future__ import annotations

from typing import Literal

from accelerate import Accelerator
from transformers import (
    LlamaPreTrainedModel,
    LlamaTokenizerFast,
    PreTrainedTokenizerBase,
)
from typing_extensions import override

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.prompt_templates import Conversation
from robust_llm.models.wrapped_chat_model import WrappedChatModel
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("tinyllama")
class TinyLlamaChatModel(WrappedChatModel):
    CONTEXT_LENGTH = 8192

    def __init__(
        self,
        model: LlamaPreTrainedModel,
        right_tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator | None,
        inference_type: InferenceType,
        train_minibatch_size: int,
        eval_minibatch_size: int,
        generation_config: GenerationConfig | None,
        family: Literal["tinyllama"],
        system_prompt: str | None = None,
        seed: int = 0,
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
            seed=seed,
        )

    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> LlamaTokenizerFast:
        """Load the tokenizer."""
        tokenizer = LlamaTokenizerFast.from_pretrained(
            model_config.name_or_path,
            revision=model_config.revision,
            padding_side="right",  # Left padding is handled separately
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=False,
        )
        assert isinstance(tokenizer, LlamaTokenizerFast)  # for type-checking

        return tokenizer

    @override
    def init_conversation(self) -> Conversation:
        return Conversation(
            prompt_prefix="",
            system_prefix="<|system|>\n",
            system_suffix="</s>\n",
            user_prefix="<|user|>\n",
            user_suffix="</s>\n",
            assistant_prefix="<|assistant|>\n",
            assistant_suffix="</s>\n",
            system_prompt=self.system_prompt,
        )
