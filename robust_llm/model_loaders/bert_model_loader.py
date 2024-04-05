from transformers import (
    BertForSequenceClassification,
    BertLMHeadModel,
    BertTokenizerFast,
    PreTrainedModel,
)

from robust_llm.model_loaders.model_loader import ModelLoader


class BERTModelLoader(ModelLoader):
    """Loads a BERT model."""

    CONTEXT_LENGTH = 512

    @classmethod
    def load_model(
        cls,
        model_name_or_path: str,
        revision: str = "main",
        num_labels: int = 2,
    ) -> BertForSequenceClassification:
        """Load the model."""
        model = BertForSequenceClassification.from_pretrained(
            model_name_or_path,
            revision=revision,
            use_cache=False,  # otherwise returns last key/values attentions
            num_labels=num_labels,
        )
        assert isinstance(model, PreTrainedModel)  # for type-checking
        return model

    @classmethod
    def load_tokenizer(
        cls,
        model_name_or_path: str,
        revision: str = "main",
        padding_side: str = "right",
        clean_up_tokenization_spaces: bool = False,
    ) -> BertTokenizerFast:
        """Load the tokenizer."""
        print("WARNING: BERT ignores padding_side")
        tokenizer = BertTokenizerFast.from_pretrained(
            model_name_or_path,
            revision=revision,
            model_max_length=cls.CONTEXT_LENGTH,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        assert isinstance(tokenizer, BertTokenizerFast)  # for type-checking
        return tokenizer

    @classmethod
    def load_decoder(
        cls,
        model_name_or_path: str,
        revision: str = "main",
    ) -> BertLMHeadModel:
        decoder = BertLMHeadModel.from_pretrained(
            model_name_or_path, revision=revision, is_decoder=True
        )
        assert isinstance(decoder, PreTrainedModel)
        return decoder
