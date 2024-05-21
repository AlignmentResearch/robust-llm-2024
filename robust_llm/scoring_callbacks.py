from enum import Enum
from typing import Callable, Sequence, cast

import torch

from robust_llm.models.model_utils import (
    InferenceType,
    classification_losses_from_logits,
    classification_successes_from_logits,
)
from robust_llm.models.wrapped_model import WrappedModel

# TODO (ian): Make this more restrictive
InputData = Sequence[str] | torch.Tensor
LabelData = Sequence[int] | Sequence[str]
OutputData = Sequence[float] | torch.Tensor | Sequence[bool]

ScoringCallback = Callable[[WrappedModel, InputData, LabelData], OutputData]
BinaryCallback = Callable[[WrappedModel, InputData, LabelData], Sequence[bool]]
TensorCallback = Callable[[WrappedModel, InputData, LabelData], torch.Tensor]


class CallbackReturnType(Enum):
    TENSOR = "tensor"
    BINARY = "binary"


class CallbackRegistry:
    """Registry for ScoringCallbacks.

    We use this so we can create arbitrary ScoringCallbacks and reference them
    from the config.
    """

    _binary_registry: dict[str, BinaryCallback] = {}
    _tensor_registry: dict[str, TensorCallback] = {}

    @classmethod
    def register_callback(cls, name: str, return_type: str):
        """Registers a ScoringCallback.

        Args:
            name: The name to associate with the ScoringCallback.
            type: The return type of the ScoringCallback. Should be a str
                matching one of the values in CallbackReturnType.
        """

        callback_type = CallbackReturnType(return_type)

        def decorator(function) -> ScoringCallback:
            match callback_type:
                case CallbackReturnType.BINARY:
                    cls._binary_registry[name] = function
                    return function
                case CallbackReturnType.TENSOR:
                    cls._tensor_registry[name] = function
                    return function

        return decorator

    @classmethod
    def get_binary_callback(cls, name: str) -> BinaryCallback:
        try:
            return cls._binary_registry[name]
        except KeyError:
            raise ValueError(
                f"BinaryCallback {name} not found in registry."
                " Is it definitely a BinaryCallback?"
            )

    @classmethod
    def get_tensor_callback(cls, name: str) -> TensorCallback:
        try:
            return cls._tensor_registry[name]
        except KeyError:
            raise ValueError(
                f"TensorCallback {name} not found in registry."
                " Is it definitely a TensorCallback?"
            )


def _validate_classification_labels(label_data: LabelData) -> list[int]:
    if not isinstance(label_data, list):
        raise ValueError("Expected label_data to be a list.")
    if not all(isinstance(x, int) for x in label_data):
        raise ValueError("Expected all elements of label_data to be ints.")
    label_data = cast(list[int], label_data)
    return label_data


@CallbackRegistry.register_callback(name="successes_from_text", return_type="binary")
def successes_from_text_callback(
    victim: WrappedModel,
    input_data: list[str],
    label_data: LabelData,
) -> list[bool]:
    """Compute the successes from the text input.

    Args:
        victim: The model to evaluate.
        input_data: The input data. List of strings, one string for each
            sequence in the batch.
        label_data: The labels.

    Returns:
        List of booleans, one for each sequence in the batch, indicating
        whether the model got the correct answer on that sequence.
    """
    tokenized = victim.tokenizer(input_data, return_tensors="pt", padding=True)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    if victim.inference_type == InferenceType.CLASSIFICATION:
        label_data = _validate_classification_labels(label_data)

        out = victim.classification_output_from_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = out["logits"]
        successes = classification_successes_from_logits(logits, label_data)
    elif victim.inference_type == InferenceType.GENERATION:
        raise NotImplementedError("Generation not supported yet.")
    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")
    return successes


@CallbackRegistry.register_callback(name="successes_from_tokens", return_type="binary")
def successes_from_tokens_callback(
    victim: WrappedModel, input_data: torch.Tensor, label_data: LabelData
) -> Sequence[bool]:
    """Compute the successes from the token inputs.

    Args:
        victim: The model to evaluate.
        input_data: Shape (batch, seq_len). Batch of input tokens.
        label_data: The labels.

    Returns:
        List of booleans, one for each sequence in the batch, indicating
        whether the model got the correct answer on that sequence.
    """
    if victim.inference_type == InferenceType.CLASSIFICATION:
        label_data = _validate_classification_labels(label_data)

        out = victim.classification_output_from_tokens(
            input_ids=input_data,
            attention_mask=torch.ones_like(input_data),
            use_cache=False,
        )
        logits = out["logits"]
        successes = classification_successes_from_logits(logits, label_data)
    elif victim.inference_type == InferenceType.GENERATION:
        raise NotImplementedError("Generation not supported yet.")
    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")
    return successes


@CallbackRegistry.register_callback(name="losses_from_text", return_type="tensor")
def losses_from_text_callback(
    victim: WrappedModel, input_data: list[str], label_data: LabelData
) -> torch.Tensor:
    raise NotImplementedError("Loss from text not supported yet.")
    # return victim.loss_from_text(input_data, label_data)


@CallbackRegistry.register_callback(name="losses_from_embeds", return_type="tensor")
def losses_from_embeds_callback(
    victim: WrappedModel,
    input_data: torch.Tensor,
    label_data: LabelData,
) -> torch.Tensor:
    """Compute the losses from the embeddings input.

    Args:
        victim: The model to evaluate.
        input_data: The embeddings. Tensor of shape (batch_size, seq_len, hidden_size).
        label_data: The labels, either ints (for classification) or strings (for
            generation).
    """
    if victim.inference_type == InferenceType.CLASSIFICATION:
        label_data = _validate_classification_labels(label_data)

        out = victim.classification_output_from_embeddings(
            embeddings=input_data,
            use_cache=False,
        )
        logits = out["logits"]
        losses = classification_losses_from_logits(logits, label_data)
    elif victim.inference_type == InferenceType.GENERATION:
        raise NotImplementedError("Generation not supported yet.")
    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")
    return losses
