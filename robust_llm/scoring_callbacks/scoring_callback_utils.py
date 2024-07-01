# TODO (ian): Make this more restrictive
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Sequence, cast

import torch
import wandb
from typing_extensions import override

from robust_llm.logging_utils import should_log
from robust_llm.models.model_utils import (
    AutoregressiveOutput,
    classification_successes_from_logits,
    generation_losses_from_logits,
    generation_successes_from_logits,
)
from robust_llm.models.wrapped_model import WrappedModel

InputData = Sequence[str] | torch.Tensor | dict[str, torch.Tensor]


@dataclass(frozen=True)
class CallbackInput:
    """The input to a ScoringCallback.

    Attributes:
        input_data:
            The input data to evaluate.
        original_input_data:
            The original input data, if available. Provide this when
            'input_data' represents an attacked dataset, since some
            ScoringCallbacks need the clean inputs.
        clf_label_data:
            The classification label data. Must be provided if we are doing
            classification.
        gen_target_data:
            The generation target data. Must be provided if we are doing
            generation.
    """

    input_data: InputData
    original_input_data: InputData | None = None
    clf_label_data: Sequence[int] | None = None
    gen_target_data: Sequence[str] | None = None


@dataclass(kw_only=True)
class CallbackOutput(ABC):
    info: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def maybe_log_info(self, table_name: str) -> None:
        """Log info to wandb if self.info is not empty."""


@dataclass(kw_only=True)
class BinaryCallbackOutput(CallbackOutput):
    successes: list[bool]

    @override
    def maybe_log_info(self, table_name: str) -> None:
        if should_log():
            if len(self.info) == 0:
                return

            expected_len = len(self.successes)
            for value in self.info.values():
                assert len(value) == expected_len

            table = wandb.Table(columns=["success"] + list(self.info.keys()))
            for i in range(expected_len):
                row = [self.successes[i]]
                for key, value in self.info.items():
                    row.append(value[i])
                table.add_data(*row)
            wandb.log({table_name: table}, commit=False)


@dataclass(kw_only=True)
class TensorCallbackOutput(CallbackOutput):
    losses: torch.Tensor

    @override
    def maybe_log_info(self, table_name: str) -> None:
        raise NotImplementedError("TensorCallbackOutput does not support logging info.")


LabelData = Sequence[int] | Sequence[str]

ScoringCallback = Callable[[WrappedModel, CallbackInput], CallbackOutput]
BinaryCallback = Callable[[WrappedModel, CallbackInput], BinaryCallbackOutput]
TensorCallback = Callable[[WrappedModel, CallbackInput], TensorCallbackOutput]


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


def _classification_success_from_text(
    victim: WrappedModel,
    input_data: list[str],
    clf_label_data: list[int],
) -> list[bool]:
    # We use right-padding for non-autoregressive outputs.
    tokenized = victim.tokenize(input_data, return_tensors="pt", padding_side="right")
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
        assert victim.accelerator is not None
        assert isinstance(logits, torch.Tensor)
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
    prompt_input_ids = victim.tokenize(input_data)["input_ids"]
    prompt_input_ids = cast(list[list[int]], prompt_input_ids)
    return _generation_successes_from_tokens(
        victim,
        prompt_input_ids,
        gen_target_data,
    )


def _generation_successes_from_tokens(
    victim: WrappedModel,
    prompt_input_ids: list[list[int]],
    gen_target_data: list[str],
) -> list[bool]:

    target_input_ids = victim.tokenize(gen_target_data)["input_ids"]
    assert isinstance(prompt_input_ids, list)
    assert isinstance(target_input_ids, list)

    full_input_ids = get_full_encoded_prompts(prompt_input_ids, target_input_ids)

    # To pad the inputs we have to format as a dict. It's fine to be missing the
    # attention mask since it's all ones anyway and we'll get a new one from
    # pad.
    # We pad on the right because we're computing logits, not doing autoregressive
    # generation.
    tokenized = victim.pad(
        dict(input_ids=full_input_ids), padding_side="right", return_tensors="pt"
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
        assert victim.accelerator is not None
        assert isinstance(logits, torch.Tensor)
        batch_length = logits.shape[0]
        batch_input_ids = input_ids[batch_start : batch_start + batch_length]
        batch_target_ids = target_input_ids[batch_start : batch_start + batch_length]
        successes = generation_successes_from_logits(
            logits=logits,
            input_ids=batch_input_ids.tolist(),
            goal=batch_target_ids,
        )
        all_successes.extend(successes)
        batch_start += batch_length
    return all_successes


def _generation_losses_from_text(
    victim: WrappedModel,
    input_data: list[str],
    gen_target_data: list[str],
    use_no_grad: bool,
) -> torch.Tensor:
    """Compute the losses from the text input.

    TODO(ian): Maybe make this into a generator.
    """
    # We don't pad or return torch tensors here yet because we need to append
    # the target tokens and keep track of their indices before padding.
    prompt_inps = victim.tokenize(input_data)
    prompt_input_ids = cast(list[list[int]], prompt_inps["input_ids"])
    return _generation_losses_from_tokens(
        victim=victim,
        prompt_input_ids=prompt_input_ids,
        gen_target_data=gen_target_data,
        use_no_grad=use_no_grad,
    )


def _generation_losses_from_tokens(
    victim: WrappedModel,
    prompt_input_ids: list[list[int]],
    gen_target_data: list[str],
    use_no_grad: bool,
) -> torch.Tensor:

    target_input_ids = victim.tokenize(gen_target_data)["input_ids"]
    assert isinstance(prompt_input_ids, list)
    assert isinstance(target_input_ids, list)

    full_input_ids = get_full_encoded_prompts(prompt_input_ids, target_input_ids)

    # To pad the inputs we have to format as a dict. It's fine to be missing the
    # attention mask since it's all ones anyway and we'll get a new one from
    # pad.
    # We pad on the right because we're computing logits, not doing autoregressive
    # generation.
    tokenized = victim.pad(
        dict(input_ids=full_input_ids), padding_side="right", return_tensors="pt"
    )

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    all_losses: list[torch.Tensor] = []
    output_generator = victim.generation_output_from_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_no_grad=use_no_grad,
    )

    batch_start = 0
    for out in output_generator:
        logits = out["logits"]
        assert victim.accelerator is not None
        assert isinstance(logits, torch.Tensor)
        batch_length = logits.shape[0]
        batch_input_ids = input_ids[batch_start : batch_start + batch_length]
        batch_target_ids = target_input_ids[batch_start : batch_start + batch_length]
        losses = generation_losses_from_logits(
            logits=logits,
            input_ids=batch_input_ids.tolist(),
            goal=batch_target_ids,
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
    target_input_ids = victim.tokenize(
        list(gen_target_data),
        return_tensors="pt",
        padding_side=None,
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
    full_input_ids = get_full_encoded_prompts(
        prompt_input_ids.tolist(), target_input_ids.tolist()
    )

    batch_start = 0
    for out in output_generator:
        logits = out["logits"]
        assert victim.accelerator is not None
        assert isinstance(logits, torch.Tensor)
        batch_length = logits.shape[0]
        batch_input_ids = full_input_ids[batch_start : batch_start + batch_length]
        batch_target_ids = target_input_ids[batch_start : batch_start + batch_length]
        losses = generation_losses_from_logits(
            logits=logits,
            input_ids=batch_input_ids,
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


def _output_generator_from_text(
    victim: WrappedModel,
    callback_input: CallbackInput,
) -> Iterator[list[AutoregressiveOutput]]:
    # We use left-padding for autoregressive outputs.
    input_data = _validate_text_input(callback_input.input_data)
    tokenized = victim.tokenize(input_data, return_tensors="pt", padding_side="left")
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    output_generator = victim.autoregressive_generation_from_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return output_generator
