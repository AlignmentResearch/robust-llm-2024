from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import AutoModelForCausalLMWithValueHead

if TYPE_CHECKING:
    from robust_llm.models import WrappedModel

# The tensors are (batch_size, n_heads, seq_len, head_dim).
PastKeyValues = tuple[tuple[torch.Tensor, ...], ...]

Tdict = TypeVar("Tdict", bound=dict)
T = TypeVar("T")


class InferenceType(Enum):
    """The type of inference the model is used for.

    This is used to determine the type of model to load from HuggingFace.
    The reason TRL models are a separate category is that we need to load a
    model with a value head.
    """

    CLASSIFICATION = "classification"
    GENERATION = "generation"
    TRL = "trl"


@dataclass(frozen=True)
class AutoregressiveOutput:
    """The output of an autoregressive model.

    Attributes:
        input_text:
            The input text given to the model.
        output_text:
            The output text from the model.
        clean_input_text:
            The unattacked version of the input_text, if available.
    """

    input_text: str
    output_text: str
    clean_input_text: str | None = None

    def with_clean_input_text(self, clean_input_text: str) -> AutoregressiveOutput:
        """Returns a new AutoregressiveOutput with the clean_input_text set."""
        return AutoregressiveOutput(
            input_text=self.input_text,
            output_text=self.output_text,
            clean_input_text=clean_input_text,
        )

    def get_full_text(self, delimiter: str = "\n-----\n") -> str:
        """Get the full text (input + output) for logging to wandb."""
        return self.input_text + delimiter + self.output_text


def load_hf_model(
    name_or_path: str,
    revision: str,
    inference_type: InferenceType,
    strict_load: bool,
    torch_dtype: torch.dtype,
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
        torch_dtype: Data type of the model.
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
                torch_dtype=torch_dtype,
            )
        case InferenceType.GENERATION:
            model, loading_info = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                revision=revision,
                output_loading_info=True,
                use_cache=False,  # By default we don't want to output cache values.
                torch_dtype=torch_dtype,
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
                torch_dtype=torch_dtype,
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
    input_ids: Sequence[Sequence[int]],
    goal: Sequence[Sequence[int]],
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the generation losses from the logits.

    Args:
        logits: Shape (batch, l_seq_len, vocab_size). The logits from the model.
        input_ids: Shape (batch, a_seq_len). The input_ids.
        goal: List of length 'batch' of lists of token ids. The tokenized goals.
        reduction: The reduction to apply to the losses. Either "mean" or "sum",
            default "mean".

    Returns:
        The generation losses, shape (batch,)."""

    def score_fn(logits: torch.Tensor, goal: Sequence[int]):
        return loss_on_goal(logits, goal, reduction)

    outs = fn_of_generation_logits(
        logits=logits,
        input_ids=input_ids,
        goal=goal,
        score_fn=score_fn,
    )
    return torch.stack(outs)


def loss_on_goal(logits: torch.Tensor, goal: Sequence[int], reduction: str = "mean"):
    """Compute the generation losses from the logits.

    Args:
        logits:
            Shape (batch, target_len, vocab_size). The logits from the model on
            the target.
        goal:
            List of length 'target_len'. The tokenized goals.
        reduction:
            The reduction to apply to the losses. Either "mean" or "sum",
            default "mean".
    """
    all_logprobs = torch.log_softmax(logits, dim=-1)
    # Get logprobs just for the actual goal tokens
    goal_logprobs = all_logprobs[torch.arange(len(goal)), goal]
    if reduction == "mean":
        return -goal_logprobs.mean()
    elif reduction == "sum":
        return -goal_logprobs.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


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
    input_ids: Sequence[Sequence[int]],
    goal: Sequence[Sequence[int]],
) -> list[bool]:
    """Compute the generation successes from the logits.

    NOTE: The sequence length of the attention mask can be longer than the
    sequence length of the logits. This happens when we are using caching, because
    logits are not returned for the cached tokens.

    Args:
        logits: Shape (batch, l_seq_len, vocab_size). The logits from the model.
        input_ids: Shape (batch, a_seq_len). The input_ids.
        goal: List of length 'batch' of lists of token ids. The tokenized goals.

    Returns:
        list of length 'batch'. Whether the goal string is the most likely string.
    """

    return fn_of_generation_logits(
        logits=logits,
        input_ids=input_ids,
        goal=goal,
        score_fn=success_on_goal,
    )


def fn_of_generation_logits(
    logits: torch.Tensor,
    input_ids: Sequence[Sequence[int]],
    goal: Sequence[Sequence[int]],
    score_fn: Callable[[torch.Tensor, Sequence[int]], T],
) -> list[T]:
    """Compute a function of the generation logits.

    This is an abstraction of both generation_successes_from_logits
    and generation_losses_from_logits.
    Args:
        logits:
            Shape (batch, l_seq_len, vocab_size). The logits from the model.
        input_ids:
            Shape (batch, l_seq_len). The input ids, used to locate the
            target in the logits.
        goal:
            List of length 'batch' of lists of token ids. The tokenized goals.
        score_fn:
            The function to compute. Should take the logits of the target
            and the target itself and return a scalar.
    """
    assert logits.ndim == 3, "Logits should be (batch, seq_len, vocab_size)."
    assert len(goal) == logits.shape[0]

    # TODO(ian): Vectorize if possible.
    scores = []
    for example_logits, example_ids, example_goal in zip(logits, input_ids, goal):
        # Find the target in the logits using the input_ids.
        # We look for the LAST occurrence of the target in the input_ids, since
        # the target should be at the end of the sequence.
        target_slice = get_target_slice(example_ids, example_goal)

        goal_logits = example_logits[target_slice]
        scores.append(score_fn(goal_logits, example_goal))
    return scores


def get_target_slice(input_ids: Sequence[int], target: Sequence[int]) -> slice:
    """Get a slice corresponding to the last occurrence of 'target' using 'input_ids'.

    NOTE:
    - The slice will correspond to the target *in the logits*, which are offset
        by one. Thus when checking that the slice is correct, we have to add one.
    - The slice is *negative*, so it can be used in the logits even when caching
        means we get fewer logits than input_ids.

    """
    assert len(input_ids) > 0 and isinstance(input_ids[0], int)

    # Get the start of the last occurrence of the target in the input_ids.
    start_index = find_subsequence_start_indices(input_ids, target)[-1]
    start_index_in_logits = start_index - 1
    end_index_in_logits = start_index_in_logits + len(target)
    negative_start = start_index_in_logits - len(input_ids)
    negative_end = end_index_in_logits - len(input_ids)
    target_slice = slice(negative_start, negative_end)

    # We use 'or None' here to handle the case where target_slice.stop + 1 is 0,
    # which would mess up the slice.
    # Note that we don't have to worry about the case where target_slice.stop is
    # 0, since we subtracted 1 earlier.
    assert input_ids[target_slice.start + 1 : (target_slice.stop + 1) or None] == target
    return target_slice


def find_subsequence_start_indices(
    all_tokens: Sequence[T], subsequence: Sequence[T]
) -> list[int]:
    """Find all starting indices of a subsequence in a list of tokens.

    If the subsequence is not found, return an empty list.
    """
    indices = []
    for i in range(len(all_tokens) - len(subsequence) + 1):
        if all_tokens[i : i + len(subsequence)] == subsequence:
            indices.append(i)
    return indices


def success_on_goal(logits: torch.Tensor, goal: Sequence[int]):
    """Compute the generation successes from the logits.

    Args:
        logits:
            Shape (batch, target_len, vocab_size). The logits from the model on
            the target.
        goal:
            List of length 'target_len'. The tokenized goals.
    """
    # Take logits from the end of the sequence, since the goal is at the end.
    predicted_tokens = torch.argmax(logits, dim=1)
    assert len(predicted_tokens) == len(goal)
    return predicted_tokens.tolist() == goal


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


@dataclass(frozen=True)
class PromptTemplate:
    """This is a general class for prompt templates,
    that should encompass both chat models and non-chat
    models

    The basic idea is that there is some part before user input, and
    some part after user input but before model input, and that should
    be all off the content in the prompt.

    For example, for a simple chat format:
        before_attack="User: Hi, I'm an user! "
        after_attack="\nAssistant:"
    """

    before_attack: str = ""
    after_attack: str = ""

    def build_prompt(self, *, attack_text: str = "", target: str = "") -> str:
        prompt = self.before_attack + attack_text + self.after_attack + target
        return prompt


def remove_padding_tokens(
    tokenizer: PreTrainedTokenizerBase, texts: list[str]
) -> list[str]:
    """Remove padding tokens from a list of output texts.

    Args:
        tokenizer: The tokenizer used to tokenize the text.
        texts: The list of texts to remove padding tokens from.

    Returns:
        The list of texts with padding tokens removed.
    """
    return [text.replace(tokenizer.pad_token, "") for text in texts]


def get_num_parameters(model: PreTrainedModel) -> int:
    """Get the number of parameters in a model.

    Needed because TRL models don't have num_parameters attribute.
    """
    try:
        return model.num_parameters()
    except AttributeError:
        try:
            return sum(p.numel() for p in model.parameters())
        except Exception as e:
            raise ValueError(f"Could not get num parameters for {type(model)}") from e
