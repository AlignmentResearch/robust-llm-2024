from abc import ABC, abstractmethod
from typing import Optional

import torch
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
        return model(inputs_embeds=inputs_embeds).logits

    raise ValueError("exactly one of inp, inputs_embeds must be provided")


class SearchBasedAttackWrappedModel(ABC):
    """Combines a model and a tokenizer and includes model-specific settings for GCG."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cls_token_id: int | None = None,
        sep_token_id: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
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

        # TODO(GH#120): work out whether to keep clean_up_tokenization_spaces
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
    ) -> None:
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        assert cls_token_id is not None
        assert sep_token_id is not None
        super().__init__(model, tokenizer, cls_token_id, sep_token_id)

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
        return self.model.bert.embeddings.word_embeddings(token_ids)

    def get_embedding_weights(self) -> torch.Tensor:
        # TODO: work out if we should be adding positional embeddings
        return self.model.bert.embeddings.word_embeddings.weight


class WrappedGPT2Model(SearchBasedAttackWrappedModel):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        cls_token_id = None
        sep_token_id = None
        super().__init__(model, tokenizer, cls_token_id, sep_token_id)

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
        return self.model.transformer.wte(token_ids)

    def get_embedding_weights(self) -> torch.Tensor:
        # TODO: work out if we should be adding positional embeddings
        return self.model.transformer.wte.weight


class WrappedGPTNeoXModel(SearchBasedAttackWrappedModel):
    # NOTE: Pythia models are based on GPTNeoX

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        cls_token_id = None
        sep_token_id = None
        super().__init__(model, tokenizer, cls_token_id, sep_token_id)
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
        return self.model.gpt_neox.embed_in(token_ids)

    def get_embedding_weights(self) -> torch.Tensor:
        # TODO: work out if we should be adding positional embeddings
        return self.model.gpt_neox.embed_in.weight
