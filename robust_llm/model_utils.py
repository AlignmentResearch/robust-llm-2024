from typing import Type, cast

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from robust_llm.configs import ModelFamily
from robust_llm.model_loaders import (
    BERTModelLoader,
    GPT2ModelLoader,
    ModelLoader,
    PythiaModelLoader,
)

MODEL_FAMILY_CLASSES: dict[ModelFamily, Type[ModelLoader]] = {
    "gpt2": GPT2ModelLoader,
    "pythia": PythiaModelLoader,
    "bert": BERTModelLoader,
}


def _prepare_model(
    model_name_or_path: str,
    model_family: str,
    num_labels: int,
    revision: str = "main",
) -> PreTrainedModel:
    assert model_family in MODEL_FAMILY_CLASSES
    model_family = cast(ModelFamily, model_family)
    model_loader = MODEL_FAMILY_CLASSES[model_family]
    model = model_loader.load_model(
        model_name_or_path,
        revision=revision,
        num_labels=num_labels,
    )
    return model


def _prepare_tokenizer(
    model_name_or_path: str,
    model_family: str,
    revision: str = "main",
    padding_side: str = "right",
) -> PreTrainedTokenizerBase:
    # When this setting is on, one undesired consequence can be that
    # decode(encode(text)) != text.
    clean_up_tokenization_spaces = False

    assert model_family in MODEL_FAMILY_CLASSES
    model_family = cast(ModelFamily, model_family)
    model_loader = MODEL_FAMILY_CLASSES[model_family]
    tokenizer = model_loader.load_tokenizer(
        model_name_or_path,
        revision=revision,
        padding_side=padding_side,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )
    return tokenizer


def _prepare_decoder(
    decoder_name: str,
    model_family: str,
    revision: str = "main",
) -> PreTrainedModel:
    assert model_family in MODEL_FAMILY_CLASSES
    model_family = cast(ModelFamily, model_family)
    model_loader = MODEL_FAMILY_CLASSES[model_family]
    decoder = model_loader.load_decoder(
        model_name_or_path=decoder_name,
        revision=revision,
    )
    return decoder
