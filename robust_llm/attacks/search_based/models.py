from abc import ABC, abstractmethod
from typing import Optional

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
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing_extensions import override


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
        return model(inp).logits

    if inputs_embeds is not None:
        with SuppressPadTokenWarning(model):
            return model(inputs_embeds=inputs_embeds).logits

    raise ValueError("exactly one of inp, inputs_embeds must be provided")


def _get_embedding_weights(
    accelerator: Accelerator, embedding: torch.nn.Module
) -> torch.Tensor:
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


class SearchBasedAttackWrappedModel(ABC):
    """Combines a model and a tokenizer and includes model-specific settings for GCG."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator,
        cls_token_id: int | None = None,
        sep_token_id: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

    def get_tokens(
        self,
        inp: str | list[str],
        return_tensors: Optional[str] = "pt",
        add_special: bool = False,
    ) -> torch.Tensor:
        """Handles all the arguments we have to add to the tokenizer."""
        tokens = self.tokenizer(
            inp,
            return_tensors=return_tensors,
            add_special_tokens=add_special,
        ).input_ids
        if return_tensors == "pt":
            tokens = tokens.to(dtype=torch.int64)
        return tokens

    def decode_tokens(
        self,
        inp: torch.Tensor,
        skip_special_tokens: bool = True,
        try_squeeze: bool = True,
    ) -> str:
        if len(inp.shape) == 2 and inp.shape[0] == 1 and try_squeeze:
            inp = inp.squeeze()

        string = self.tokenizer.decode(
            inp,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
        return string

    @abstractmethod
    def call_model(
        self,
        inp: torch.Tensor | None = None,
        add_cls: bool = True,
        add_sep: bool = True,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns the logits from calling the model on a tensor of tokens."""
        pass

    @abstractmethod
    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Returns the embeddings for the given token ids."""
        pass

    @abstractmethod
    def get_embedding_weights(self) -> torch.Tensor:
        """Returns the embedding weights for the model."""
        pass

    def _prepend_to_inputs(self, token_id: int, inputs: torch.Tensor) -> torch.Tensor:
        """Prepends the given token id to each sequence of tokens in inputs.

        NOTE: tokens are added to the START of sequences.
        """
        return torch.cat(
            [
                torch.full(
                    size=(inputs.size(0), 1),
                    fill_value=token_id,
                    dtype=torch.int,
                    device=self.device,
                ),
                inputs,
            ],
            dim=1,
        )

    def _append_to_inputs(self, token_id: int, inputs: torch.Tensor) -> torch.Tensor:
        """Appends the given token id to each sequence of tokens in inputs.

        NOTE: tokens are added to the END of sequences.
        """
        return torch.cat(
            [
                inputs,
                torch.full(
                    size=(inputs.size(0), 1),
                    fill_value=token_id,
                    dtype=torch.int,
                    device=self.device,
                ),
            ],
            dim=1,
        )

    def _check_for_padding_tokens(self, token_ids: torch.Tensor) -> None:
        """Checks if padding tokens are present in the token ids.

        When using inputs_embeds, it's important that there are no padding tokens,
        since they are not handled properly."""
        if self.tokenizer.pad_token_id is not None:
            assert (
                self.tokenizer.pad_token_id not in token_ids
            ), "Padding tokens are present in the token ids."

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size  # type: ignore

    @property
    def device(self) -> torch.device:
        return self.model.device


class WrappedBERTModel(SearchBasedAttackWrappedModel):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator,
    ) -> None:
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        assert cls_token_id is not None
        assert sep_token_id is not None
        super().__init__(model, tokenizer, accelerator, cls_token_id, sep_token_id)

    @override
    def call_model(
        self,
        inp: torch.Tensor | None = None,
        add_cls: bool = True,
        add_sep: bool = True,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NOTE: So far the BERT model is the only one with cls and sep tokens to
        # deal with, so when debugging we need to pay special attention to whether
        # they are actually being added.
        if inp is not None:
            # add cls to starts of sequences by default
            if add_cls:
                assert self.cls_token_id is not None
                inp = self._prepend_to_inputs(self.cls_token_id, inp)

            # add sep to ends of sequences by default
            if add_sep:
                assert self.tokenizer.sep_token_id is not None
                inp = self._append_to_inputs(self.tokenizer.sep_token_id, inp)

        return _call_model(self.model, inp, inputs_embeds)

    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        self._check_for_padding_tokens(token_ids)
        return self.model.bert.embeddings.word_embeddings(token_ids)

    def get_embedding_weights(self) -> torch.Tensor:
        # TODO: work out if we should be adding positional embeddings
        return _get_embedding_weights(
            self.accelerator, self.model.bert.embeddings.word_embeddings
        )


class WrappedGPT2Model(SearchBasedAttackWrappedModel):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator,
    ) -> None:
        cls_token_id = None
        sep_token_id = None
        super().__init__(model, tokenizer, accelerator, cls_token_id, sep_token_id)

        # GPT-2 was crashing when using FSDP. In the (unlikely) case we want to run FSDP
        # with GPT-2 in the future, investigate this.
        assert accelerator.distributed_type != DistributedType.FSDP, "not supported!"

    @override
    def call_model(
        self,
        inp: torch.Tensor | None = None,
        add_cls: bool = False,
        add_sep: bool = False,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # add cls and add_sep are ignored but included for consistency with other models
        assert add_cls is False
        assert add_sep is False
        return _call_model(self.model, inp, inputs_embeds)

    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        self._check_for_padding_tokens(token_ids)
        return self.model.transformer.wte(token_ids)

    def get_embedding_weights(self) -> torch.Tensor:
        # TODO: work out if we should be adding positional embeddings
        return _get_embedding_weights(self.accelerator, self.model.transformer.wte)


class WrappedGPTNeoXModel(SearchBasedAttackWrappedModel):
    # NOTE: Pythia models are based on GPTNeoX

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator,
    ) -> None:
        cls_token_id = None
        sep_token_id = None
        super().__init__(model, tokenizer, accelerator, cls_token_id, sep_token_id)
        # special setup needed for pythia
        self.tokenizer.pad_token = tokenizer.eos_token
        self.model.config.pad_token_id = model.config.eos_token_id

    @override
    def call_model(
        self,
        inp: torch.Tensor | None = None,
        add_cls: bool = False,
        add_sep: bool = False,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # add cls and add_sep are ignored but included for consistency with other models
        assert add_cls is False
        assert add_sep is False
        return _call_model(self.model, inp, inputs_embeds)

    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        self._check_for_padding_tokens(token_ids)
        return self.model.gpt_neox.embed_in(token_ids)

    def get_embedding_weights(self) -> torch.Tensor:
        # TODO: work out if we should be adding positional embeddings
        return _get_embedding_weights(self.accelerator, self.model.gpt_neox.embed_in)


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

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.saved_pad_token = model.config.pad_token_id

    def __enter__(self):
        self.model.config.pad_token_id = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.config.pad_token_id = self.saved_pad_token
