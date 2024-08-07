from __future__ import annotations

from transformers import GPTNeoXTokenizerFast

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.wrapped_model import WrappedModel


@WrappedModel.register_subclass("gpt_neox")
@WrappedModel.register_subclass("pythia")
class GPTNeoXModel(WrappedModel):
    # NOTE: Pythia models are based on GPTNeoX
    CONTEXT_LENGTH = 2048

    def post_init(self):
        super().post_init()
        assert self.family in ["gpt_neox", "pythia"]
        # Special setup needed for pythia.
        self.model.config.pad_token_id = self.model.config.eos_token_id

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
