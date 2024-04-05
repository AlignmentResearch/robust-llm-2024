from transformers import (
    GPTNeoXForCausalLM,
    GPTNeoXForSequenceClassification,
    GPTNeoXTokenizerFast,
    PreTrainedModel,
)

from robust_llm.model_loaders.model_loader import ModelLoader


class PythiaModelLoader(ModelLoader):
    """Loads a Pythia model."""

    CONTEXT_LENGTH = 2048

    @classmethod
    def load_model(
        cls,
        model_name_or_path: str,
        revision: str = "main",
        num_labels: int = 2,
    ) -> GPTNeoXForSequenceClassification:
        """Load the model."""
        model = GPTNeoXForSequenceClassification.from_pretrained(
            model_name_or_path,
            revision=revision,
            use_cache=False,  # otherwise returns last key/values attentions
            num_labels=num_labels,
        )
        assert isinstance(model, PreTrainedModel)  # for type-checking
        model.config.pad_token_id = model.config.eos_token_id
        return model

    @classmethod
    def load_tokenizer(
        cls,
        model_name_or_path: str,
        revision: str = "main",
        padding_side: str = "right",
        clean_up_tokenization_spaces: bool = False,
    ) -> GPTNeoXTokenizerFast:
        """Load the tokenizer."""
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(
            model_name_or_path,
            revision=revision,
            padding_side=padding_side,
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        assert isinstance(tokenizer, GPTNeoXTokenizerFast)  # for type-checking

        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @classmethod
    def load_decoder(
        cls,
        model_name_or_path: str,
        revision: str = "main",
    ) -> GPTNeoXForCausalLM:
        decoder = GPTNeoXForCausalLM.from_pretrained(
            model_name_or_path,
            revision=revision,
            use_cache=False,  # otherwise returns last key/values attentions
        )
        assert isinstance(decoder, PreTrainedModel)
        decoder.config.pad_token_id = decoder.config.eos_token_id
        return decoder
