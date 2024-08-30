from __future__ import annotations

from transformers import Qwen2TokenizerFast
from typing_extensions import override

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.prompt_templates import Conversation
from robust_llm.models.wrapped_chat_model import WrappedChatModel
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("qwen1.5-chat")
@WrappedModel.register_subclass("qwen2-chat")
class QwenChatModel(WrappedChatModel):
    CONTEXT_LENGTH = 32768

    def post_init(self):
        super().post_init()
        assert self.family in ["qwen1.5-chat", "qwen2-chat"]

    @override
    def forward(self, **inputs):
        # Qwen is idiosyncratic in that it requires use_cache=True to use
        # provided past_key_values. This is not the default behavior for other
        # models (except Gemma), where use_cache indicates whether to return
        # past_key_values.
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
    def init_conversation(self) -> Conversation:
        """Qwen-specific conversation template."""
        return Conversation(
            prompt_prefix="",
            system_prefix="<|im_start|>system\n",
            system_suffix="<|im_end|>\n",
            user_prefix="<|im_start|>user\n",
            user_suffix="<|im_end|>\n",
            assistant_prefix="<|im_start|>assistant\n",
            assistant_suffix="<|im_end|>\n",
            system_prompt=self.system_prompt or "You are a helpful assistant.",
        )
