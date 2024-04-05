from transformers import (
    GPT2ForSequenceClassification,
    GPT2TokenizerFast,
    PreTrainedModel,
)

from robust_llm.model_loaders.model_loader import ModelLoader


class GPT2ModelLoader(ModelLoader):
    """Loads a GPT2 model.

    Works for both OpenAI and Stanford Mistral base models.
    """

    CONTEXT_LENGTH = 1024

    @classmethod
    def load_model(
        cls,
        model_name_or_path: str,
        revision: str = "main",
        num_labels: int = 2,
    ) -> GPT2ForSequenceClassification:
        """Load the model."""
        model = GPT2ForSequenceClassification.from_pretrained(
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
    ) -> GPT2TokenizerFast:
        """Load the tokenizer."""
        tokenizer = GPT2TokenizerFast.from_pretrained(
            model_name_or_path,
            revision=revision,
            padding_side=padding_side,
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        assert isinstance(tokenizer, GPT2TokenizerFast)  # for type-checking

        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @classmethod
    def load_decoder(
        cls,
        model_name_or_path: str,
        revision: str = "main",
    ) -> PreTrainedModel:
        raise NotImplementedError("gpt2 decoder not implemented")
