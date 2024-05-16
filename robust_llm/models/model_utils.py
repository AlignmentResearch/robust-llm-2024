from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
from trl import AutoModelForCausalLMWithValueHead

if TYPE_CHECKING:
    from robust_llm.models import WrappedModel


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
                use_cache=False,  # Otherwise returns last key/values attentions.
            )
        case InferenceType.GENERATION:
            model, loading_info = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                revision=revision,
                output_loading_info=True,
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


def _call_model(
    model: PreTrainedModel,
    inp: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calls a pretrained model and returns the logits."""

    assert (inp is not None) != (
        inputs_embeds is not None
    ), "exactly one of inp, inputs_embeds must be provided"

    if inp is not None:
        return model(input_ids=inp).logits

    if inputs_embeds is not None:
        with SuppressPadTokenWarning(model):
            return model(inputs_embeds=inputs_embeds).logits

    raise ValueError("exactly one of inp, inputs_embeds must be provided")


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

    def __init__(self, model: "PreTrainedModel | WrappedModel"):
        self.model = model
        self.saved_pad_token = model.config.pad_token_id

    def __enter__(self):
        self.model.config.pad_token_id = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.config.pad_token_id = self.saved_pad_token
