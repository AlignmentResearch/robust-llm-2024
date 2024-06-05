"""Retokenization defense."""

from collections.abc import Sequence
from typing import Any, Union, cast

import torch
from transformers import PreTrainedTokenizerBase

from robust_llm import logger
from robust_llm.config.defense_configs import RetokenizationDefenseConfig
from robust_llm.defenses.defense import MutatingDefendedModel
from robust_llm.models import WrappedModel


def pad_list_of_lists(
    list_of_lists: list[list[int]],
    pad_token_id: int,
    max_length: int,
    padding_side: str = "right",
) -> list[list[int]]:
    """Pad lists to the longest member, up to length `max_length`."""
    # Pad lists to match the longest member
    pad_length = max((len(flat_sublist) for flat_sublist in list_of_lists))
    # Cap at model max length
    pad_length = min(pad_length, max_length)
    # Make sure that the lists are all the same length
    amount_to_pad = [max(0, pad_length - len(sublist)) for sublist in list_of_lists]
    if padding_side == "right":
        return [
            sublist + [pad_token_id] * pad_amount
            for sublist, pad_amount in zip(list_of_lists, amount_to_pad)
        ]
    else:
        return [
            [pad_token_id] * pad_amount + sublist
            for sublist, pad_amount in zip(list_of_lists, amount_to_pad)
        ]


def _broken_token_representations_flat_lists(
    input_ids: Sequence[int],
    attention_mask: Sequence[int],
    broken_tokens: Sequence[tuple[int, Sequence[int]]],
    max_length: int,
    padding_side: str = "right",
) -> tuple[list[int], list[int]]:
    # Convert broken_tokens to dictionary for O(1) lookups
    broken_tokens_dict = {k: v for k, v in broken_tokens}

    new_list = [
        broken_tokens_dict[i] if i in broken_tokens_dict else [i] for i in input_ids
    ]
    new_attention_mask = []
    assert len(input_ids) == len(attention_mask)
    for input_id, mask in zip(input_ids, attention_mask):
        if input_id in broken_tokens_dict:
            new_attention_mask.append([mask] * len(broken_tokens_dict[input_id]))
        else:
            new_attention_mask.append([mask])
    # Flatten the list of lists
    flat_list = [item for sublist in new_list for item in sublist]
    flat_attention_mask = [item for sublist in new_attention_mask for item in sublist]
    clip_length = min(len(flat_list), max_length)
    # Make sure that list hasn't passed model max length
    if padding_side == "right":
        flat_list = flat_list[:clip_length]
        flat_attention_mask = flat_attention_mask[:clip_length]
    else:
        flat_list = flat_list[-clip_length:]
        flat_attention_mask = flat_attention_mask[-clip_length:]

    return flat_list, flat_attention_mask


def _broken_token_representations_nested_lists(
    input_ids: Sequence[Sequence[int]],
    attention_mask: Sequence[Sequence[int]],
    broken_tokens: Sequence[tuple[int, Sequence[int]]],
    max_length: int,
    padding_side: str = "right",
    pad_token_id: int = 0,
) -> tuple[list[list[int]], list[list[int]]]:
    new_list_of_lists = []
    new_attention_mask_list = []
    for input_ids_sublist, attention_mask_sublist in zip(input_ids, attention_mask):
        (
            flat_sublist,
            flat_attention_mask_sublist,
        ) = _broken_token_representations_flat_lists(
            input_ids=input_ids_sublist,
            attention_mask=attention_mask_sublist,
            broken_tokens=broken_tokens,
            max_length=max_length,
            padding_side=padding_side,
        )
        new_list_of_lists.append(flat_sublist)
        new_attention_mask_list.append(flat_attention_mask_sublist)

    new_list_of_lists = pad_list_of_lists(
        new_list_of_lists, pad_token_id, max_length, padding_side
    )
    new_attention_mask_list = pad_list_of_lists(
        new_attention_mask_list, 0, max_length, padding_side
    )

    return new_list_of_lists, new_attention_mask_list


def _broken_token_representations_from_lists(
    input_ids: Union[Sequence[int], Sequence[Sequence[int]]],
    attention_mask: Union[Sequence[int], Sequence[Sequence[int]]],
    broken_tokens: Sequence[tuple[int, Sequence[int]]],
    max_length: int,
    padding_side: str = "right",
    pad_token_id: int = 0,
) -> tuple[Union[list[int], list[list[int]]], ...]:
    if isinstance(input_ids[0], int):
        input_ids = cast(Sequence[int], input_ids)
        attention_mask = cast(Sequence[int], attention_mask)
        return _broken_token_representations_flat_lists(
            input_ids=input_ids,
            attention_mask=attention_mask,
            broken_tokens=broken_tokens,
            max_length=max_length,
            padding_side=padding_side,
        )

    else:
        input_ids = cast(Sequence[Sequence[int]], input_ids)
        attention_mask = cast(Sequence[Sequence[int]], attention_mask)
        return _broken_token_representations_nested_lists(
            input_ids=input_ids,
            attention_mask=attention_mask,
            broken_tokens=broken_tokens,
            max_length=max_length,
            padding_side=padding_side,
            pad_token_id=pad_token_id,
        )


def _broken_token_representations_from_tensors(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    broken_tokens: Sequence[tuple[int, Sequence[int]]],
    max_length: int,
    padding_side: str = "right",
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids, mask = _broken_token_representations_from_lists(
        input_ids=input_ids.tolist(),
        attention_mask=attention_mask.tolist(),
        broken_tokens=broken_tokens,
        max_length=max_length,
        padding_side=padding_side,
        pad_token_id=pad_token_id,
    )
    return (
        torch.tensor(token_ids, device=input_ids.device),
        torch.tensor(mask, device=input_ids.device),
    )


def broken_token_representations(
    input_ids: Union[torch.Tensor, Sequence[int] | Sequence[Sequence[int]]],
    attention_mask: Union[torch.Tensor, Sequence[int] | Sequence[Sequence[int]]],
    broken_tokens: Sequence[tuple[int, Sequence[int]]],
    max_length: int,
    padding_side: str = "right",
    pad_token_id: int = 0,
) -> tuple[Union[torch.Tensor, Sequence[int], Sequence[Sequence[int]]], ...]:
    """Expands tokens in `input_ids` into multiple tokens.

    Transforms a pair of `input_ids` and `attention_mask` by expanding each token
    in `broken_tokens` into multiple tokens. Accepts tensors or lists.

    Args:
        input_ids: the input_ids to expand.
        attention_mask: the attention_mask to expand.
        broken_tokens: a list of (token_id, broken_token_ids) tuples.
        max_length: the max context length allowed by the model.
        padding_side: the side to pad on.
        pad_token_id: the token to use for padding.
    Returns:
        A tuple of (input_ids, attention_mask) with the broken tokens expanded.
    """
    if len(broken_tokens) == 0:
        return input_ids, attention_mask
    assert isinstance(broken_tokens[0], tuple)
    if isinstance(input_ids, torch.Tensor):
        assert isinstance(attention_mask, torch.Tensor)
        return _broken_token_representations_from_tensors(
            input_ids=input_ids,
            attention_mask=attention_mask,
            broken_tokens=broken_tokens,
            max_length=max_length,
            padding_side=padding_side,
            pad_token_id=pad_token_id,
        )
    else:
        assert not isinstance(attention_mask, torch.Tensor)
        return _broken_token_representations_from_lists(
            input_ids=input_ids,
            attention_mask=attention_mask,
            broken_tokens=broken_tokens,
            max_length=max_length,
            padding_side=padding_side,
            pad_token_id=pad_token_id,
        )


class BytePairDecomposer:
    """Decompose words into multiple "broken" tokens.

    See Section 4.3 of https://arxiv.org/pdf/2309.00614.pdf for more details.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        self._create_prefix_set()
        self._create_suffix_set()
        self.compute_byte_pair_merges()

    def _create_prefix_set(self):
        # Create a dictionary for quick prefix lookups
        self.prefixes = {
            token for token in self.tokenizer.get_vocab() if token and "##" not in token
        }

    def _create_suffix_set(self):
        # Create a dictionary for quick suffix lookups
        self.suffixes = {
            token
            for token in self.tokenizer.get_vocab()
            if token and token.startswith("##")
        }

    def compute_byte_pair_merges(self):
        self.merge_strings = []
        self.merge_tokens = []

        # Iterate through the vocab items
        for string, token_id in self.tokenizer.get_vocab().items():
            if "##" in string:
                continue
            # Try to find the prefix in the dictionary
            for i in range(len(string), 0, -1):
                prefix = string[:i]
                suffix = "##" + string[i:]
                broken_rep = self.tokenizer.encode(
                    [prefix, suffix], add_special_tokens=False
                )
                if prefix in self.prefixes and suffix in self.suffixes:
                    self.merge_strings.append((prefix + suffix, prefix, suffix))
                    self.merge_tokens.append((token_id, broken_rep))
                    # we break after finding the longest prefix match
                    break


class RetokenizationDefendedModel(MutatingDefendedModel):
    def __init__(
        self, victim: WrappedModel, defense_config: RetokenizationDefenseConfig
    ) -> None:
        super().__init__(victim)
        self.cfg = defense_config
        self.drop_percentage = self.cfg.drop_percentage
        self.verbose = self.cfg.verbose

        self.bpe_decomposer = BytePairDecomposer(self.tokenizer)
        tokens_to_keep = int(
            len(self.bpe_decomposer.merge_tokens) * self.drop_percentage
        )
        self.broken_tokens = self.bpe_decomposer.merge_tokens[:tokens_to_keep]

    @property
    def defense_config(self) -> RetokenizationDefenseConfig:
        return self.cfg

    def forward(self, **inputs) -> Any:
        assert self.tokenizer.pad_token_id is not None
        input_ids, attention_mask = broken_token_representations(
            inputs["input_ids"],
            inputs["attention_mask"],
            self.broken_tokens,
            padding_side=self.tokenizer.padding_side,
            max_length=self.tokenizer.model_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if self.verbose:
            assert isinstance(input_ids, torch.Tensor)
            assert isinstance(attention_mask, torch.Tensor)
            logger.info("Baseline tokenization:\n")
            logger.info("    seq_len=%s", inputs["input_ids"].shape)
            logger.info("    non_padding=%s", inputs["attention_mask"].sum(dim=1))
            logger.info("retokenization:\n")
            logger.info("    seq_len=%s", input_ids.shape)
            logger.info("    non_padding=%s", attention_mask.sum(dim=1))
        return self._underlying_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
