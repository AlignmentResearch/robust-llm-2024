from __future__ import annotations

from typing import Literal

from accelerate import Accelerator
from transformers import (
    GPTNeoXPreTrainedModel,
    GPTNeoXTokenizerFast,
    PreTrainedTokenizerBase,
)
from typing_extensions import override

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.prompt_templates import Conversation
from robust_llm.models.wrapped_chat_model import WrappedChatModel
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("pythia-chat")
class GPTNeoXChatModel(WrappedChatModel):
    # NOTE: Pythia models are based on GPTNeoX
    CONTEXT_LENGTH = 2048

    def __init__(
        self,
        model: GPTNeoXPreTrainedModel,
        right_tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator | None,
        inference_type: InferenceType,
        train_minibatch_size: int,
        eval_minibatch_size: int,
        generation_config: GenerationConfig | None,
        family: Literal["pythia-chat"],
        system_prompt: str | None = None,
    ) -> None:
        # TODO (ian): Decide whether this assert is worthwhile (it makes testing
        # harder).
        # assert isinstance(model, GPTNeoXPreTrainedModel)
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
            padding_side="right",  # Left padding is handled separately
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=False,
        )
        assert isinstance(tokenizer, GPTNeoXTokenizerFast)  # for type-checking

        # Special setup needed for pythia.
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @override
    def init_conversation(self) -> Conversation:
        return Conversation(
            prompt_prefix="",
            system_prefix="<|im_start|>system\n",
            system_suffix="<|im_end|>\n",
            user_prefix="<|im_start|>user\n",
            user_suffix="<|im_end|>\n",
            assistant_prefix="<|im_start|>assistant\n",
            assistant_suffix="<|im_end|>\n",
            system_prompt=self.system_prompt,
        )
