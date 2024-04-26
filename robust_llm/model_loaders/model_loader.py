from abc import ABC, abstractmethod

from transformers import PreTrainedModel


class ModelLoader(ABC):
    """Abstract class for loading a model."""

    @classmethod
    @abstractmethod
    def load_model(
        cls,
        model_name_or_path: str,
        revision: str = "main",
        num_labels: int = 2,
    ) -> PreTrainedModel:
        """Load the model."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_tokenizer(
        cls,
        model_name_or_path: str,
        revision: str = "main",
        padding_side: str = "right",
        clean_up_tokenization_spaces: bool = False,
    ):
        """Load the tokenizer."""
        raise NotImplementedError

    @classmethod
    def load_decoder(cls, model_name_or_path: str, revision: str = "main"):
        """Load the decoder."""
        raise NotImplementedError
