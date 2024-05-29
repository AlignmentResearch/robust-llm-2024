from dataclasses import dataclass
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
InputData = Sequence[str] | torch.Tensor | dict[str, torch.Tensor]


@dataclass
class CallbackInput:
    input_data: InputData
    clf_label_data: Sequence[int] | None = None
    gen_target_data: Sequence[str] | None = None


LabelData = Sequence[int] | Sequence[str]
OutputData = Sequence[float] | torch.Tensor | Sequence[bool]

ScoringCallback = Callable[[WrappedModel, CallbackInput], OutputData]
BinaryCallback = Callable[[WrappedModel, CallbackInput], Sequence[bool]]
TensorCallback = Callable[[WrappedModel, CallbackInput], torch.Tensor]


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
    if not isinstance(label_data, Sequence):
        raise ValueError("Expected label_data to be a Sequence.")
    if not all(isinstance(x, int) for x in label_data):
        raise ValueError("Expected all elements of label_data to be ints.")
    label_data = cast(list[int], label_data)
    return label_data


def _validate_text_input(input_data: InputData) -> list[str]:
    if not isinstance(input_data, Sequence):
        raise ValueError("Expected input_data to be a Sequence.")
    if not all(isinstance(x, str) for x in input_data):
        raise ValueError("Expected all elements of input_data to be str.")
    input_data = cast(list[str], input_data)
    return input_data


def _validate_tokens_input(input_data: InputData) -> torch.Tensor:
    if not isinstance(input_data, torch.Tensor):
        raise ValueError("Expected input_data to be a tensor.")
    if torch.is_floating_point(input_data):
        raise ValueError("Expected elements of input_data to be int.")
    if input_data.ndim != 2:
        raise ValueError("Expected input_data to be 2D: (batch, seq_len).")
    input_data = cast(torch.Tensor, input_data)
    return input_data


def _validate_embeddings_input(input_data: InputData) -> dict[str, torch.Tensor]:
    if not isinstance(input_data, dict):
        raise ValueError("Expected input_data to be a dict.")
    if input_data.keys() != {"input_ids", "embeddings"}:
        raise ValueError(
            "Expected input_data to have keys 'input_ids' and 'embeddings'."
        )
    input_ids = input_data["input_ids"]
    embeddings = input_data["embeddings"]
    # Reuse the tokens input validation for the input_ids.
    input_ids = _validate_tokens_input(input_ids)

    if not torch.is_floating_point(embeddings):
        raise ValueError("Expected elements of input_data to be float.")
    if embeddings.ndim != 3:
        raise ValueError("Expected input_data to be 3D: (batch, seq_len, hidden_size).")
    if not embeddings.shape[-1] > 100:
        raise ValueError(
            "Expected embedding dimension to be greater than 100."
            " (If your embeddings really are smaller than this, you may need to"
            " change this check.)"
        )
    embeddings = cast(torch.Tensor, embeddings)
    return dict(input_ids=input_ids, embeddings=embeddings)


@CallbackRegistry.register_callback(name="successes_from_text", return_type="binary")
def successes_from_text_callback(
    victim: WrappedModel,
    callback_input: CallbackInput,
) -> list[bool]:
    """Compute the successes from the text input.

    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.

    Returns:
        List of booleans, one for each sequence in the batch, indicating
        whether the model got the correct answer on that sequence.
    """

    input_data = _validate_text_input(callback_input.input_data)
    tokenized = victim.tokenizer(input_data, return_tensors="pt", padding=True)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    if victim.inference_type == InferenceType.CLASSIFICATION:
        if callback_input.clf_label_data is None:
            raise ValueError("Label data required for classification.")
        label_data = _validate_classification_labels(callback_input.clf_label_data)

        out = victim.classification_output_from_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
    victim: WrappedModel, callback_input: CallbackInput
) -> Sequence[bool]:
    """Compute the successes from the token inputs.

    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.

    Returns:
        List of booleans, one for each sequence in the batch, indicating
        whether the model got the correct answer on that sequence.
    """
    input_data = _validate_tokens_input(callback_input.input_data)
    if victim.inference_type == InferenceType.CLASSIFICATION:
        assert callback_input.clf_label_data is not None
        label_data = _validate_classification_labels(callback_input.clf_label_data)

        out = victim.classification_output_from_tokens(
            input_ids=input_data,
            attention_mask=torch.ones_like(input_data),
        )
        logits = out["logits"]
        successes = classification_successes_from_logits(logits, label_data)
    elif victim.inference_type == InferenceType.GENERATION:
        raise NotImplementedError("Generation not supported yet.")
    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")
    return successes


@CallbackRegistry.register_callback(name="losses_from_embeds", return_type="tensor")
def losses_from_embeds_callback(
    victim: WrappedModel,
    callback_input: CallbackInput,
    use_no_grad: bool = False,
) -> torch.Tensor:
    """Compute the losses from the embeddings input.

    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.
        use_no_grad: Whether to use torch.no_grad() when computing the losses.
    """
    input_data = _validate_embeddings_input(callback_input.input_data)
    if victim.inference_type == InferenceType.CLASSIFICATION:
        assert callback_input.clf_label_data is not None
        label_data = _validate_classification_labels(callback_input.clf_label_data)

        out = victim.classification_output_from_embeddings(
            input_ids=input_data["input_ids"],
            embeddings=input_data["embeddings"],
            use_no_grad=use_no_grad,
        )
        logits = out["logits"]
        losses = classification_losses_from_logits(logits, label_data)
    elif victim.inference_type == InferenceType.GENERATION:
        raise NotImplementedError("Generation not supported yet.")
    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")
    return losses


@CallbackRegistry.register_callback(name="losses_from_text", return_type="tensor")
def losses_from_text_callback(
    victim: WrappedModel,
    callback_input: CallbackInput,
) -> torch.Tensor:
    """Compute losses from the text input.

    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.
    """
    input_data = _validate_text_input(callback_input.input_data)
    tokenized = victim.tokenizer(input_data, return_tensors="pt", padding=True)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    if victim.inference_type == InferenceType.CLASSIFICATION:
        if callback_input.clf_label_data is None:
            raise ValueError("Label data required for classification.")
        label_data = _validate_classification_labels(callback_input.clf_label_data)

        out = victim.classification_output_from_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = out["logits"].to(device="cpu")
        losses = classification_losses_from_logits(logits, label_data)
    elif victim.inference_type == InferenceType.GENERATION:
        raise NotImplementedError("Generation not supported yet.")
    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")

    # Postconditions.
    assert losses.shape[0] == len(input_data)
    return losses
