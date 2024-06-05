from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import cast

import torch

from robust_llm.models.model_utils import (
    InferenceType,
    classification_losses_from_logits,
    classification_successes_from_logits,
    generation_losses_from_logits,
    generation_successes_from_logits,
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
            return_type: The return type of the ScoringCallback. Should be a str
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

    # TODO(GH#440): Support batch sizes greater than 1.
    if embeddings.shape[0] != 1:
        raise NotImplementedError("Batch sizes greater than 1 not supported yet.")
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
        list of booleans, one for each sequence in the batch, indicating
        whether the model got the correct answer on that sequence.
    """

    input_data = _validate_text_input(callback_input.input_data)

    label_data: LabelData
    if victim.inference_type == InferenceType.CLASSIFICATION:
        if callback_input.clf_label_data is None:
            raise ValueError("Label data required for classification.")
        label_data = _validate_classification_labels(callback_input.clf_label_data)
        successes = _classification_success_from_text(victim, input_data, label_data)

    elif victim.inference_type == InferenceType.GENERATION:
        if callback_input.gen_target_data is None:
            raise ValueError("Target data required for generation.")
        label_data = _validate_text_input(callback_input.gen_target_data)
        successes = _generation_successes_from_text(victim, input_data, label_data)

    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")

    return successes


def _classification_success_from_text(
    victim: WrappedModel,
    input_data: list[str],
    clf_label_data: list[int],
) -> list[bool]:
    tokenized = victim.tokenizer(input_data, return_tensors="pt", padding=True)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    return _classification_success_from_tokens(
        victim, input_ids, attention_mask, clf_label_data
    )


def _classification_success_from_tokens(
    victim: WrappedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    clf_label_data: list[int],
) -> list[bool]:

    all_successes: list[bool] = []
    output_generator = victim.classification_output_from_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    batch_start = 0
    for out in output_generator:
        logits = out["logits"]
        batch_length = logits.shape[0]
        batch_label_data = clf_label_data[batch_start : batch_start + batch_length]
        successes = classification_successes_from_logits(
            logits,
            batch_label_data,
        )
        all_successes.extend(successes)
        batch_start += batch_length
    return all_successes


def _generation_successes_from_text(
    victim: WrappedModel,
    input_data: list[str],
    gen_target_data: list[str],
) -> list[bool]:
    """Compute the successes from the text input.

    TODO(ian): Maybe make this into a generator.
    """
    # We don't pad or return torch tensors here yet because we need to append
    # the target tokens and keep track of their indices before padding.
    prompt_input_ids = victim.tokenizer(input_data)["input_ids"]
    prompt_input_ids = cast(list[list[int]], prompt_input_ids)
    return _generation_successes_from_tokens(
        victim,
        prompt_input_ids,
        None,
        gen_target_data,
    )


def _generation_successes_from_tokens(
    victim: WrappedModel,
    prompt_input_ids: list[list[int]],
    prompt_attention_mask: torch.Tensor | None,
    gen_target_data: list[str],
) -> list[bool]:

    target_input_ids = victim.tokenizer(gen_target_data)["input_ids"]
    assert isinstance(prompt_input_ids, list)
    assert isinstance(target_input_ids, list)

    full_input_ids = get_full_encoded_prompts(prompt_input_ids, target_input_ids)

    # To pad the inputs we have to format as a dict. It's fine to be missing the
    # attention mask since it's all ones anyway and we'll get a new one from
    # pad.
    tokenized = victim.tokenizer.pad(
        dict(input_ids=full_input_ids), return_tensors="pt"
    )

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    all_successes: list[bool] = []
    output_generator = victim.generation_output_from_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    batch_start = 0
    for out in output_generator:
        logits = out["logits"]
        batch_length = logits.shape[0]
        batch_target_ids = target_input_ids[batch_start : batch_start + batch_length]
        batch_attention_mask = attention_mask[batch_start : batch_start + batch_length]
        successes = generation_successes_from_logits(
            logits,
            batch_attention_mask,
            batch_target_ids,
        )
        all_successes.extend(successes)
        batch_start += batch_length
    return all_successes


def _generation_losses_from_text(
    victim: WrappedModel,
    input_data: list[str],
    gen_target_data: list[str],
) -> torch.Tensor:
    """Compute the losses from the text input.

    TODO(ian): Maybe make this into a generator.
    """
    # We don't pad or return torch tensors here yet because we need to append
    # the target tokens and keep track of their indices before padding.
    prompt_input_ids = victim.tokenizer(input_data)["input_ids"]
    prompt_input_ids = cast(list[list[int]], prompt_input_ids)
    return _generation_losses_from_tokens(
        victim=victim,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=None,
        gen_target_data=gen_target_data,
    )


def _generation_losses_from_tokens(
    victim: WrappedModel,
    prompt_input_ids: list[list[int]],
    prompt_attention_mask: torch.Tensor | None,
    gen_target_data: list[str],
) -> torch.Tensor:

    target_input_ids = victim.tokenizer(gen_target_data)["input_ids"]
    assert isinstance(prompt_input_ids, list)
    assert isinstance(target_input_ids, list)

    full_input_ids = get_full_encoded_prompts(prompt_input_ids, target_input_ids)

    # To pad the inputs we have to format as a dict. It's fine to be missing the
    # attention mask since it's all ones anyway and we'll get a new one from
    # pad.
    tokenized = victim.tokenizer.pad(
        dict(input_ids=full_input_ids), return_tensors="pt"
    )

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    all_losses: list[torch.Tensor] = []
    output_generator = victim.generation_output_from_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    batch_start = 0
    for out in output_generator:
        logits = out["logits"]
        assert isinstance(logits, torch.Tensor)
        batch_length = logits.shape[0]
        batch_target_ids = target_input_ids[batch_start : batch_start + batch_length]
        batch_attention_mask = attention_mask[batch_start : batch_start + batch_length]
        losses = generation_losses_from_logits(
            logits,
            batch_attention_mask,
            batch_target_ids,
        )
        all_losses.append(losses)
        batch_start += batch_length
    return torch.cat(all_losses)


def _generation_losses_from_embeds(
    victim: WrappedModel,
    prompt_input_ids: torch.Tensor,
    prompt_input_embeds: torch.Tensor,
    gen_target_data: Sequence[str],
    use_no_grad: bool,
) -> torch.Tensor:
    """Compute the losses from the embeddings input.

    Args:
        victim: The model to evaluate.
        prompt_input_ids: The tokenized prompt input.
        prompt_input_embeds: The embedded prompt input.
        gen_target_data: The target data.
        use_no_grad: Whether to use torch.no_grad() when computing the losses.
    """
    assert len(gen_target_data) == 1
    target_input_ids = victim.tokenizer(
        list(gen_target_data),
        return_tensors="pt",
        padding=False,
    )["input_ids"]
    assert isinstance(target_input_ids, torch.Tensor)
    target_embeds = victim.get_embeddings(target_input_ids)

    full_embeds = get_full_embeds(prompt_input_embeds, target_embeds)

    all_losses: list[torch.Tensor] = []
    output_generator = victim.generation_output_from_embeddings(
        # It's fine that input_ids doesn't have the target since it's just for
        # checking for cache hits with early parts of the prompt anyway.
        input_ids=prompt_input_ids,
        embeddings=full_embeds,
        use_no_grad=use_no_grad,
    )

    batch_start = 0
    for out in output_generator:
        logits = out["logits"]
        assert isinstance(logits, torch.Tensor)
        batch_length = logits.shape[0]
        batch_target_ids = target_input_ids[batch_start : batch_start + batch_length]
        losses = generation_losses_from_logits(
            logits=logits,
            attention_mask=None,
            goal=batch_target_ids.tolist(),
        )
        all_losses.append(losses)
        batch_start += batch_length
    return torch.cat(all_losses)


def get_full_encoded_prompts(
    prompt_ids: list[list[int]],
    target_ids: list[list[int]],
) -> list[list[int]]:
    """Get the full tokenized prompts by concatenating the prompt and target tokens.

    We can neglect the attention mask because the inputs should not have been
    padded yet. Padding will be done afterwards.

    Args:
        prompt_ids: The tokenized prompt input.
        target_ids: The tokenized target input.

    Returns:
        A list of the full input tokens, combining the prompt and target tokens.
    """

    return [prompt + target for prompt, target in zip(prompt_ids, target_ids)]


def get_full_embeds(
    prompt_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
) -> torch.Tensor:
    """Get the full embedded prompts by concatenating prompt and target.

    We can neglect the attention mask because the inputs should not be padded
    when using embeds.

    Args:
        prompt_embeds: A tensor containing the embedded prompt inputs.
        target_embeds: A tensor containing the embedded target inputs.

    Returns:
        A tensor containing the full input embeddings, combining the prompt
        and target embeddings.
    """
    assert prompt_embeds.ndim == target_embeds.ndim == 3
    assert prompt_embeds.shape[0] == prompt_embeds.shape[0]

    if prompt_embeds.shape[0] != 1:
        raise NotImplementedError("Batch sizes greater than 1 not supported yet.")

    return torch.cat((prompt_embeds, target_embeds), dim=1)


@CallbackRegistry.register_callback(name="successes_from_tokens", return_type="binary")
def successes_from_tokens_callback(
    victim: WrappedModel, callback_input: CallbackInput
) -> Sequence[bool]:
    """Compute the successes from the token inputs.

    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.

    Returns:
        list of booleans, one for each sequence in the batch, indicating
        whether the model got the correct answer on that sequence.
    """
    input_data = _validate_tokens_input(callback_input.input_data)
    label_data: LabelData
    if victim.inference_type == InferenceType.CLASSIFICATION:
        assert callback_input.clf_label_data is not None
        label_data = _validate_classification_labels(callback_input.clf_label_data)
        # We don't have attention masks for tokens, so we just use ones.
        # TODO(ian): Work out if we should pass an attention mask, or can assume
        # they're all unpadded.
        assert not torch.any(input_data == victim.tokenizer.pad_token_id)
        attention_mask = torch.ones_like(input_data)
        successes = _classification_success_from_tokens(
            victim, input_data, attention_mask, label_data
        )
    elif victim.inference_type == InferenceType.GENERATION:
        assert callback_input.gen_target_data is not None
        label_data = _validate_text_input(callback_input.gen_target_data)
        # Currently the _generation_successes_from_tokens function expects the
        # tokens to be a list of lists of ints, so we convert it here.
        # TODO(GH#441): Standardize this.
        list_input_data = input_data.tolist()
        successes = _generation_successes_from_tokens(
            victim, list_input_data, None, label_data
        )
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

    TODO(GH#440): Make this work for more batch sizes greater than 1.
    Args:
        victim: The model to evaluate.
        callback_input: The input data and labels.
        use_no_grad: Whether to use torch.no_grad() when computing the losses.
    """
    input_data = _validate_embeddings_input(callback_input.input_data)
    label_data: LabelData
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
        assert callback_input.gen_target_data is not None
        losses = _generation_losses_from_embeds(
            victim,
            input_data["input_ids"],
            input_data["embeddings"],
            callback_input.gen_target_data,
            use_no_grad=use_no_grad,
        )

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
    label_data: LabelData
    if victim.inference_type == InferenceType.CLASSIFICATION:
        if callback_input.clf_label_data is None:
            raise ValueError("Label data required for classification.")
        label_data = _validate_classification_labels(callback_input.clf_label_data)

        tokenized = victim.tokenizer(input_data, return_tensors="pt", padding=True)
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        all_losses = []
        output_generator = victim.classification_output_from_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        batch_start = 0
        for out in output_generator:
            logits = out["logits"]
            batch_length = logits.shape[0]
            batch_label_data = label_data[batch_start : batch_start + batch_length]
            minibatch_losses = classification_losses_from_logits(
                logits, batch_label_data
            )
            all_losses.append(minibatch_losses)
            batch_start += batch_length
        losses = torch.cat(all_losses)

    elif victim.inference_type == InferenceType.GENERATION:
        if callback_input.gen_target_data is None:
            raise ValueError("Target data required for generation.")
        label_data = _validate_text_input(callback_input.gen_target_data)
        return _generation_losses_from_text(
            victim=victim,
            input_data=input_data,
            gen_target_data=label_data,
        )
    else:
        raise ValueError(f"Unknown/unsupported inference type: {victim.inference_type}")

    # Postconditions.
    assert losses.shape[0] == len(input_data)
    return losses
