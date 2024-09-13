import copy
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from typing_extensions import overload

from robust_llm.models.prompt_templates import AttackChunks, PromptTemplate
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)


class TokenizationChangeException(Exception):
    """This exception is raised when the tokenization changes.

    E.g., (tokens(attack) + tokens(target)) is not the same as tokens(attack + target).
    """


class AttackTokenizationChangeException(TokenizationChangeException):
    """Tokenization change because of attack tokens."""


class TargetTokenizationChangeException(TokenizationChangeException):
    """Tokenization change because of target tokens."""


@dataclass(frozen=True)
class ReplacementCandidate:
    attack_position: int
    token_id: int

    @overload
    def compute_tokens_after_replacement(
        self,
        attack_tokens: torch.Tensor | list[list[int]],
        tensors: None,
    ) -> list[list[int]]: ...

    @overload
    def compute_tokens_after_replacement(
        self,
        attack_tokens: torch.Tensor | list[list[int]],
        tensors: str = "pt",
    ) -> torch.Tensor: ...

    def compute_tokens_after_replacement(
        self,
        attack_tokens: torch.Tensor | list[list[int]],
        tensors: str | None = "pt",
    ) -> torch.Tensor | list[list[int]]:
        """Make the replacement in a given sequence of attack tokens.

        N.B. Assumes that the attack tokens are a single example, i.e., batch size 1.

        Args:
            attack_tokens: The tokens to replace, of shape [1, n_attack_tokens]
            tensors: Whether to accept/return PyTorch tensors or lists.

        Returns:
            The attack tokens with replacement, also of shape [1, n_attack_tokens]
        """

        if tensors == "pt":
            if not isinstance(attack_tokens, torch.Tensor):
                raise ValueError("Expected PyTorch tensor with tensors='pt'")
            assert attack_tokens.dim() == 2, "Expected 2D tensor"
            assert attack_tokens.shape[0] == 1, "Expected batch size of 1"

            full_attack_tokens_tensor = attack_tokens.clone()
            full_attack_tokens_tensor[0, self.attack_position] = self.token_id
            return full_attack_tokens_tensor

        elif tensors is None:
            if not isinstance(attack_tokens, list):
                raise ValueError("Expected list with tensors=None")
            assert len(attack_tokens) == 1, "Expected batch size of 1"
            full_attack_tokens_list = copy.deepcopy(attack_tokens)
            full_attack_tokens_list[0][self.attack_position] = self.token_id
            return full_attack_tokens_list
        else:
            raise ValueError(f"Invalid value for tensors: {tensors}")


@dataclass
class AttackIndices:
    """Token indices of the attack and target in the prompt.

    NOTE: It's important that these are in the tokenized prompt, not the string.
    """

    attack_start: int
    attack_end: int
    target_start: int
    target_end: int

    @property
    def attack_length(self) -> int:
        return self.attack_end - self.attack_start

    @property
    def target_length(self) -> int:
        return self.target_end - self.target_start

    @property
    def attack_slice(self) -> slice:
        return slice(self.attack_start, self.attack_end)

    @property
    def target_slice(self) -> slice:
        return slice(self.target_start, self.target_end)

    @property
    def loss_slice(self) -> slice:
        """
        Slice for tokens at which we want to compute the loss.

        The loss slice is offset from the target slice because we want to
        predict the target tokens, which occurs one token before each target
        token appears
        """
        return slice(self.target_start - 1, self.target_end - 1)

    def assert_attack_and_target_tokens_validity(
        self,
        full_tokens: torch.Tensor,
        attack_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> None:
        """Assert that the attack indices actually line up with the expected tokens.

        TODO: work out if there is a better way to do this than brute-force checking
        """
        # check that the attack tokens are correct
        if not torch.equal(full_tokens[:, self.attack_slice], attack_tokens):
            raise AttackTokenizationChangeException

        # check that the target tokens are correct
        if not torch.equal(full_tokens[:, self.target_slice], target_tokens):
            raise TargetTokenizationChangeException


def create_onehot_embedding(
    token_ids: torch.Tensor,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create one-hot embeddings solely for the attack tokens portion"""

    token_ids = token_ids[0]  # drop the batch dimension
    n_tokens = token_ids.shape[0]
    one_hot = torch.zeros(
        n_tokens,
        vocab_size,
        device=device,
        dtype=dtype,
    )
    one_hot.scatter_(
        1,
        token_ids.unsqueeze(1),
        torch.ones(
            n_tokens,
            1,
            device=device,
            dtype=dtype,
        ),
    )
    one_hot.requires_grad_()
    return one_hot


def get_chunking_for_search_based(
    text_chunked: Sequence[str], modifiable_chunk_spec: ModifiableChunkSpec
) -> AttackChunks:
    """Returns the unmodifiable prefix, the modifiable infix, & the unmodifiable suffix.

    GCG needs exactly three chunks, so we guarantee this here as long as
        modifiable_chunk_spec contains exactly one modifiable chunk.
    """

    assert modifiable_chunk_spec.n_modifiable_chunks == 1

    modifiable_index = modifiable_chunk_spec.get_modifiable_chunk_index()
    unmodifiable_prefix = "".join(text_chunked[:modifiable_index])
    modifiable_infix = text_chunked[modifiable_index]
    unmodifiable_suffix = "".join(text_chunked[modifiable_index + 1 :])

    infix_chunk_type = modifiable_chunk_spec.get_modifiable_chunk()
    if infix_chunk_type == ChunkType.OVERWRITABLE:
        modifiable_infix = ""

    return AttackChunks(
        unmodifiable_prefix=unmodifiable_prefix,
        modifiable_infix=modifiable_infix,
        unmodifiable_suffix=unmodifiable_suffix,
    )


@dataclass
class PreppedExample:
    """Example prepared for search-based attacks."""

    prompt_template: PromptTemplate
    clf_label: int
    gen_target: str

    def add_attack_indices(
        self, attack_indices: AttackIndices
    ) -> "ExampleWithAttackIndices":
        return ExampleWithAttackIndices(
            prompt_template=self.prompt_template,
            clf_label=self.clf_label,
            gen_target=self.gen_target,
            attack_indices=attack_indices,
        )


@dataclass
class ExampleWithAttackIndices(PreppedExample):
    """Example prepared for search-based attacks with indices of attackable tokens."""

    attack_indices: AttackIndices
