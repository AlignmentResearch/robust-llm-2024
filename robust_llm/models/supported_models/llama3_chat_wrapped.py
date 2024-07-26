from typing import Literal

from accelerate import Accelerator
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from typing_extensions import override

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.prompt_templates import Conversation
from robust_llm.models.wrapped_chat_model import WrappedChatModel
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("llama3-chat")
class Llama3ChatModel(WrappedChatModel):
    CONTEXT_LENGTH = 4096

    def __init__(
        self,
        model: LlamaForCausalLM,
        right_tokenizer: PreTrainedTokenizerFast,
        accelerator: Accelerator | None,
        inference_type: InferenceType,
        train_minibatch_size: int,
        eval_minibatch_size: int,
        generation_config: GenerationConfig | None,
        family: Literal["llama3-chat"],
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

        # Special setup needed for llama.
        self.model.config.pad_token_id = model.config.eos_token_id

    @override
    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> PreTrainedTokenizerFast:
        """Load the tokenizer."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
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
            prompt_prefix="<|begin_of_text|>",
            system_prefix="<|start_header_id|>system<|end_header_id|>\n\n",
            system_suffix="<|eot_id|>",
            user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
            user_suffix="<|eot_id|>",
            assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
            assistant_suffix="<|eot_id|>",
            system_prompt=self.system_prompt,
        )
