from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from robust_llm.models.model_utils import PromptTemplate
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.utils import get_randint_with_exclusions


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

    def compute_tokens_after_replacement(
        self, attack_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Make the replacement in a given sequence of attack tokens.

        Args:
            attack_tokens: The tokens to replace, of shape [1, n_attack_tokens]
        Returns:
            The attack tokens with replacement, also of shape [1, n_attack_tokens]
        """
        assert attack_tokens.dim() == 2, "Expected 2D tensor"
        assert attack_tokens.shape[0] == 1, "Expected batch size of 1"

        full_attack_tokens = attack_tokens.clone()
        full_attack_tokens[0, self.attack_position] = self.token_id
        return full_attack_tokens


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
) -> tuple[str, str, str]:
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

    return unmodifiable_prefix, modifiable_infix, unmodifiable_suffix


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


def get_label_and_target_for_attack(
    example: dict[str, Any],
    dataset: RLLMDataset,
) -> tuple[int, str]:
    """Get the clf_label and gen_target ready to be passed into the attack.

    The reason getting this ready is hard is because we don't know
    if the gen_targets are for classification-as-generation or some
    other generative task. In the former case, we want to update the
    gen_target, but in the latter case, we want to leave it alone.

    Args:
        example: The example to attack.
        dataset: The dataset the example is from.

    Returns:
        A tuple of the goal clf_label and gen_target.
    """
    # Generate a goal label that is different from the true label.
    goal_clf_label = get_randint_with_exclusions(
        high=dataset.num_classes, exclusions=[example["clf_label"]]
    )

    # Handle gen_target
    if dataset.classification_as_generation:
        goal_gen_target = dataset.clf_label_to_gen_target(goal_clf_label)
    else:
        goal_gen_target = example["gen_target"]

    return goal_clf_label, goal_gen_target
