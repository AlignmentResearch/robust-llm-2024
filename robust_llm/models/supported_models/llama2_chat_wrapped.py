from typing import Literal

from accelerate import Accelerator
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing_extensions import override

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.prompt_templates import Conversation
from robust_llm.models.wrapped_chat_model import WrappedChatModel
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("llama2-chat")
@WrappedModel.register_subclass("vicuna")
class Llama2ChatModel(WrappedChatModel):
    CONTEXT_LENGTH = 4096

    def __init__(
        self,
        model: LlamaForCausalLM,
        right_tokenizer: LlamaTokenizer,
        accelerator: Accelerator | None,
        inference_type: InferenceType,
        train_minibatch_size: int,
        eval_minibatch_size: int,
        generation_config: GenerationConfig | None,
        family: Literal["llama2-chat"],
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

        # Special setup needed for llama.
        self.model.config.pad_token_id = model.config.eos_token_id

    @override
    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> LlamaTokenizer:
        """Load the tokenizer."""
        tokenizer = LlamaTokenizer.from_pretrained(
            model_config.name_or_path,
            revision=model_config.revision,
            padding_side="right",  # Left padding is handled separately
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=False,
        )

        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @override
    def init_conversation(self) -> Conversation:
        return Conversation(
            prompt_prefix="<s>[INST] ",
            system_prefix="<<SYS>>\n",
            system_suffix="\n<</SYS>>\n\n",
            user_prefix="",
            user_suffix=" [/INST] ",
            assistant_prefix="",
            assistant_suffix=" </s>",
            system_prompt=self.system_prompt,
            repeat_prompt_prefix=True,
        )
