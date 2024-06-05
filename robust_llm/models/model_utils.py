from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Optional, TypeVar

import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedType
from torch.distributed.fsdp import (
    FullStateDictConfig,  # pyright: ignore[reportPrivateImportUsage]
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel,  # pyright: ignore[reportPrivateImportUsage]
)
from torch.distributed.fsdp import (
    StateDictType,  # pyright: ignore[reportPrivateImportUsage]
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
from trl import AutoModelForCausalLMWithValueHead

from robust_llm.utils import is_correctly_padded

if TYPE_CHECKING:
    from robust_llm.models import WrappedModel

# The tensors are (batch_size, n_heads, seq_len, head_dim).
PastKeyValues = tuple[tuple[torch.Tensor, ...], ...]

Tdict = TypeVar("Tdict", bound=dict)


class InferenceType(Enum):
    """The type of inference the model is used for.

    This is used to determine the type of model to load from HuggingFace.
    The reason TRL models are a separate category is that we need to load a
    model with a value head.
    """

    CLASSIFICATION = "classification"
    GENERATION = "generation"
    TRL = "trl"


def load_hf_model(
    name_or_path: str,
    revision: str,
    inference_type: InferenceType,
    strict_load: bool,
    num_classes: Optional[int] = None,
) -> PreTrainedModel:
    """Loads a model from HuggingFace.

    NOTE: We have to suppress a type error because the from_pretrained method
    returns a tuple when output_loading_info is set to True but this is not reflected
    in the type hints.

    Args:
        name_or_path: The name or path of the model.
        revision: The revision of the model.
        inference_type (InferenceType): The type of inference the model is used for.
        strict_load: Whether to enforce that no weights are ignored or randomly
            initialized while loading.
        num_classes: The number of classes for a classification model.
    """
    # Even though num_labels is optional, passing None to it will cause an error
    # because if a value is passed, it must be an int. Two is the default.
    if num_classes is None:
        num_classes = 2

    match inference_type:
        case InferenceType.CLASSIFICATION:
            model, loading_info = AutoModelForSequenceClassification.from_pretrained(
                name_or_path,
                revision=revision,
                output_loading_info=True,
                num_labels=num_classes,
                use_cache=False,  # By default we don't want to output cache values.
            )
        case InferenceType.GENERATION:
            model, loading_info = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                revision=revision,
                output_loading_info=True,
                use_cache=False,  # By default we don't want to output cache values.
            )
        case InferenceType.TRL:
            # We can't use output_loading_info=True because it's not supported
            # for TRL models. This means strict_load must be False.
            if strict_load:
                raise ValueError(
                    "strict_load must be False for TRL models because"
                    " `output_loading_info` is not supported."
                )
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                name_or_path,
                revision=revision,
            )

    # Optionally, check that there are no weights skipped or randomly initialized.
    if strict_load:
        assert loading_info_is_empty(loading_info)  # type: ignore
    return model


def classification_losses_from_logits(
    logits: torch.Tensor, goal: Sequence[int]
) -> torch.Tensor:
    """Compute the classification loss from the logits.

    Args:
        logits: Shape (batch, n_classes). The logits from the model.
        goal: Len batch. The goal classes.

    Returns:
        The classification loss, shape (batch,).
    """
    assert logits.shape[0] == len(goal)
    goal_tensor = torch.tensor(goal, device=logits.device)
    return F.cross_entropy(logits, goal_tensor, reduction="none")


def generation_losses_from_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor | None,
    goal: Sequence[Sequence[int]],
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the generation losses from the logits.

    NOTE: The sequence length of the attention mask can be longer than the
    sequence length of the logits. This happens when we are using caching, because
    logits are not returned for the cached tokens.

    Args:
        logits: Shape (batch, l_seq_len, vocab_size). The logits from the model.
        attention_mask: Shape (batch, a_seq_len). The attention mask. If None,
            we assume no padding.
        goal: List of length 'batch' of lists of token ids. The tokenized goals.
        reduction: The reduction to apply to the losses. Either "mean" or "sum",
            default "mean".

    Returns:
        The generation losses, shape (batch,)."""
    # Just make a dummy attention mask of ones if one isn't provided.
    if attention_mask is None:
        attention_mask = torch.ones(
            logits.shape[:2], device=logits.device, dtype=torch.int16
        )

    assert logits.ndim == 3, "Logits should be (batch, seq_len, vocab_size)."
    assert attention_mask.ndim == 2, "Attention mask should be (batch, seq_len)."
    assert len(goal) == attention_mask.shape[0] == logits.shape[0]

    # TODO(ian): Vectorize if possible.
    losses = []
    for example_logits, example_goal, example_mask in zip(logits, goal, attention_mask):
        # Drop padding tokens from the logits, assuming that they're all
        # at the end. We subtract an additional one to account for the shift, where
        # the ith token is predicted by the i-1th logit.
        n_padding_tokens = sum(example_mask == 0)
        non_padding_logits = example_logits[: len(example_logits) - n_padding_tokens]
        shifted_logits = non_padding_logits[:-1]

        goal_len = len(example_goal)
        # Get logprobs for all possible tokens at the goal positions
        goal_position_logprobs = torch.log_softmax(shifted_logits[-goal_len:], dim=-1)
        # Get logprobs just for the actual goal tokens
        goal_token_logprobs = goal_position_logprobs[
            torch.arange(goal_len), example_goal
        ]
        if reduction == "mean":
            losses.append(-goal_token_logprobs.mean())
        elif reduction == "sum":
            losses.append(-goal_token_logprobs.sum())
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
    return torch.stack(losses)


def classification_successes_from_logits(
    logits: torch.Tensor, goal: Sequence[int]
) -> list[bool]:
    """Compute the classification successes from the logits.

    Args:
        logits: Shape (batch, n_classes). The logits from the model.
        goal: list of length 'batch'. The goal classes.

    Returns:
        list of length 'batch'. Whether the goal class is the most likely class.
    """
    assert logits.shape[0] == len(goal)
    predicted_classes = torch.argmax(logits, dim=1)
    return (predicted_classes == torch.tensor(goal, device=logits.device)).tolist()


def generation_successes_from_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    goal: Sequence[Sequence[int]],
) -> list[bool]:
    """Compute the generation successes from the logits.

    NOTE: The sequence length of the attention mask can be longer than the
    sequence length of the logits. This happens when we are using caching, because
    logits are not returned for the cached tokens.

    Args:
        logits: Shape (batch, l_seq_len, vocab_size). The logits from the model.
        attention_mask: Shape (batch, a_seq_len). The attention mask.
        goal: list of length 'batch' of lists of token ids. The tokenized goals.

    Returns:
        list of length 'batch'. Whether the goal string is the most likely string.
    """
    assert logits.ndim == 3, "Logits should be (batch, seq_len, vocab_size)."
    assert attention_mask.ndim == 2, "Attention mask should be (batch, seq_len)."
    assert len(goal) == attention_mask.shape[0] == logits.shape[0]

    # TODO(ian): Vectorize if possible.
    successes = []
    for example_logits, example_goal, example_mask in zip(logits, goal, attention_mask):
        # Make sure that all padding tokens are at the end (i.e. we're doing
        # right padding).
        assert is_correctly_padded(example_mask, "right")
        # Drop padding tokens from the logits, now that we know they're all
        # at the end. We subtract an additional one to account for the shift, where
        # the ith token is predicted by the i-1th logit.
        n_padding_tokens = sum(example_mask == 0)
        non_padding_logits = example_logits[: len(example_logits) - n_padding_tokens]
        shifted_logits = non_padding_logits[:-1]

        # Take logits from the end of the sequence, since the goal is at the end.
        goal_len = len(example_goal)
        predicted_tokens = torch.argmax(shifted_logits[-goal_len:], dim=1)
        assert len(predicted_tokens) == goal_len
        successes.append(predicted_tokens.tolist() == example_goal)
    return successes


def _get_embedding_weights(
    accelerator: Accelerator, embedding: torch.nn.Module
) -> torch.Tensor:
    """Get the weights from an embedding layer.

    If we are using FSDP, we need to handle the embedding layer differently.
    """
    if accelerator.distributed_type == DistributedType.FSDP:
        # Implementation based on Accelerator.get_state_dict(); however, we want to load
        # parameters in all processes, not just in the rank 0 process.
        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=False, rank0_only=False
        )
        with FullyShardedDataParallel.state_dict_type(
            embedding, StateDictType.FULL_STATE_DICT, full_state_dict_config
        ):
            return embedding.state_dict()["weight"]

    return embedding.weight


def loading_info_is_empty(loading_info: dict[str, str]) -> bool:
    """Checks whether there is any loading info.

    This is useful for checking whether there are any weights of the loaded
    model that are unused, or new weights that are randomly initialized.

    Args:
        loading_info: a dictionary mapping potential events to lists of weight names.
            If the loaded model uses exactly all of the downloaded weights, then
            all these lists should be empty.

    Returns:
        True if all the lists in loading_info are empty, False otherwise.
    """
    return all(len(v) == 0 for v in loading_info.values())


def prepare_model_with_accelerate(
    accelerator: Accelerator, model: PreTrainedModel
) -> PreTrainedModel:
    model = accelerator.prepare(model)
    # When using FSDP, there is some lazy initialization that happens. Enforce it here
    # to avoid issues from lack of proper initialization (e.g. when accessing embedding
    # layer in GCG).
    _ = model(torch.tensor([[0]], device=accelerator.device))

    return model


class SuppressPadTokenWarning:
    """Context manager to suppress pad token warnings.

    These warnings occur when you call a model with inputs_embeds rather than
    tokens. We get the embeddings by running the input tokens through the
    embedding layer. When we run the model on embeddings rather than tokens,
    information about whether some of the input tokens were padding tokens is lost,
    so padding tokens (if present) can't be masked out and huggingface
    (reasonably) gives a warning: it's important to mask out padding tokens
    since otherwise they are interpreted as normal input tokens and
    they affect the output of the model.

    The problem is the warning is repeated for every single call to the model,
    which can be annoying and make the logs unreadable. Additionally, since the
    warning is not from the 'warnings' module, it is not easy to suppress.

    This context manager suppresses the warning by disabling the padding token
    for the duration of the model call. Since we shouldn't have any padding
    tokens in the input sequence due to the issues mentioned above, and since
    the padding token is not used when calling the model with inputs_embeds,
    this should be safe.
    """

    def __init__(self, model: PreTrainedModel | WrappedModel):
        self.model = model
        self.saved_pad_token = model.config.pad_token_id

    def __enter__(self):
        self.model.config.pad_token_id = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.config.pad_token_id = self.saved_pad_token


def combine_output_dicts(dicts: Sequence[Tdict]) -> Tdict:
    """Combine outputs into a single object."""
    target_keys = dicts[0].keys()
    target_type = type(dicts[0])
    target_types = {k: type(dicts[0][k]) for k in target_keys}

    assert all(d.keys() == target_keys for d in dicts)
    assert all(isinstance(d, target_type) for d in dicts)
    assert all(isinstance(d[k], target_types[k]) for d in dicts for k in target_keys)

    combined = target_type()
    for key in target_keys:
        nones = [d[key] is None for d in dicts]
        if any(nones):
            assert all(nones)
            combined[key] = None
        else:
            combined[key] = torch.cat([d[key] for d in dicts])
    return combined


def build_dataloader(minibatch_size: int, **kwargs):
    """Build a DataLoader from arbitrary keyword arguments.

    This saves us having to manually specify the inputs multiple times.

    Args:
        minibatch_size: The size of the minibatches.
        kwargs: The keyword arguments with the actual data we want in the
            DataLoader. The keys should be the names of the fields in the dataset
            (e.g. input_ids, attention_mask, ...).
    """
    dataset = datasets.Dataset.from_dict(
        {
            **kwargs,
        }
    ).with_format("torch")

    dataloader = DataLoader(
        dataset=dataset,  # type: ignore  # Typehint is wrong in DataLoader.
        batch_size=minibatch_size,
    )
    return dataloader


def dict_to_device(d: Tdict, device: str | torch.device) -> Tdict:
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device=device)
    return d


@contextmanager
def maybe_no_grad(use_no_grad: bool):
    if use_no_grad:
        with torch.no_grad():
            yield
    else:
        yield
