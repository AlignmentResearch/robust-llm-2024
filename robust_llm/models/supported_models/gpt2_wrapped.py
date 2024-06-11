from accelerate import Accelerator, DistributedType
from transformers import GPT2PreTrainedModel, GPT2TokenizerFast, PreTrainedTokenizerBase
from typing_extensions import override

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("gpt2")
class GPT2Model(WrappedModel):
    CONTEXT_LENGTH = 1024

    def __init__(
        self,
        model: GPT2PreTrainedModel,
        right_tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator | None,
        inference_type: InferenceType,
        train_minibatch_size: int,
        eval_minibatch_size: int,
        generation_config: GenerationConfig | None,
    ) -> None:
        # TODO (ian): Decide whether this assert is worthwhile (it makes testing
        # harder).
        # assert isinstance(model, GPT2PreTrainedModel)
        super().__init__(
            model,
            right_tokenizer,
            accelerator,
            inference_type,
            train_minibatch_size,
            eval_minibatch_size,
            generation_config=generation_config,
        )

        # Special setup needed for gpt2.
        self.model.config.pad_token_id = model.config.eos_token_id

    @override
    @classmethod
    def load_tokenizer(
        cls,
        model_config: ModelConfig,
    ) -> GPT2TokenizerFast:
        """Load the tokenizer."""
        tokenizer = GPT2TokenizerFast.from_pretrained(
            model_config.name_or_path,
            revision=model_config.revision,
            padding_side="right",  # Left padding is handled separately
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=False,
        )
        assert isinstance(tokenizer, GPT2TokenizerFast)  # for type-checking

        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @override
    def add_accelerator(self, accelerator: Accelerator) -> None:
        # GPT-2 was crashing when using FSDP. In the (unlikely) case we want to run FSDP
        # with GPT-2 in the future, investigate this.
        assert accelerator.distributed_type != DistributedType.FSDP, "not supported!"
        super().add_accelerator(accelerator)
