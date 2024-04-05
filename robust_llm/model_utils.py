from typing import Optional

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertLMHeadModel,
    GPTNeoXForCausalLM,
    GPTNeoXForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def _prepare_model(
    model_name_or_path: str,
    is_pythia: bool,
    checkpoint: Optional[int],
    num_classes: int,
) -> PreTrainedModel:
    # TODO(GH#103): make it compatible with tasks where num_labels > 2.

    if is_pythia:
        revision = "main"
        # One of the original Pythia checkpoints.
        if model_name_or_path.startswith("EleutherAI/pythia"):
            assert checkpoint is not None
            revision = f"step{checkpoint}"

        model = GPTNeoXForSequenceClassification.from_pretrained(
            model_name_or_path,
            revision=revision,
            use_cache=False,  # otherwise returns last key/values attentions
            num_labels=num_classes,
        )
        assert isinstance(model, PreTrainedModel)
        model.config.pad_token_id = model.config.eos_token_id
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=num_classes
        )

    return model


def _prepare_tokenizer(
    model_name_or_path: str,
    is_pythia: bool,
    checkpoint: Optional[int],
    padding_side: str = "right",
) -> PreTrainedTokenizerBase:
    # When this setting is on, one undesired consequence can be that
    # decode(encode(text)) != text.
    clean_up_tokenization_spaces = False

    if is_pythia:
        revision = "main"
        # One of the original Pythia checkpoints.
        if model_name_or_path.startswith("EleutherAI/pythia"):
            assert checkpoint is not None
            revision = f"step{checkpoint}"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=revision,
            padding_side=padding_side,
            model_max_length=2048,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    return tokenizer


def _prepare_decoder(
    decoder_name: str,
    is_pythia: bool,
    checkpoint: Optional[int],
) -> PreTrainedModel:
    # TODO(GH#103): make it compatible with tasks where num_labels > 2.
    if "bert" in decoder_name:
        decoder = BertLMHeadModel.from_pretrained(decoder_name, is_decoder=True)
        assert isinstance(decoder, PreTrainedModel)
    elif is_pythia:
        revision = "main"
        # One of the original Pythia checkpoints.
        if decoder_name.startswith("EleutherAI/pythia"):
            assert checkpoint is not None
            revision = f"step{checkpoint}"
        decoder = GPTNeoXForCausalLM.from_pretrained(
            decoder_name,
            revision=revision,
            use_cache=False,  # otherwise returns last key/values attentions
        )
        assert isinstance(decoder, PreTrainedModel)
        decoder.config.pad_token_id = decoder.config.eos_token_id
    else:
        raise ValueError(f"Unknown model name {decoder_name}")

    return decoder
