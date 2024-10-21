import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from typing_extensions import overload

from robust_llm.models.prompt_templates import AttackChunks, PromptTemplate
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.scoring_callbacks.scoring_callback_utils import (
    CallbackInput,
    TensorCallback,
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


def select_next_candidates(
    candidates: list[tuple[float, str]], n_best_candidates_to_keep: int
) -> tuple[list[str], list[int]]:
    """Selects text candidates for the next round, based on (score, text) pairs.

    Args:
        candidates: A list of (score, text) pairs to select from.
        n_best_candidates_to_keep: the number of candidates to keep.

    Returns:
        A list of the next candidates to consider and a list of their indices.
    """
    indexed_candidates = list(enumerate(candidates))
    sorted_candidates = list(sorted(indexed_candidates, key=lambda x: x[1][0]))
    next_candidates = [
        candidate[1][1] for candidate in sorted_candidates[:n_best_candidates_to_keep]
    ]
    next_candidates_indices = [
        candidate[0] for candidate in sorted_candidates[:n_best_candidates_to_keep]
    ]
    return next_candidates, next_candidates_indices


@torch.no_grad()
def apply_replacements_and_eval_candidates(
    text_replacement_pairs: Sequence[tuple[str, ReplacementCandidate]],
    victim: WrappedModel,
    example: PreppedExample | ExampleWithAttackIndices,
    scores_from_text_callback: TensorCallback,
) -> tuple[list[tuple[float, str]], dict[str, Any]]:
    """Evaluates the candidates using a forward pass through the model.

    Args:
        text_replacement_pairs: A list of (attack_text, replacement) pairs to
            evaluate.
        victim: The model to attack
        example: The example to attack
        scores_from_text_callback: the callback for computing the scores for
            inputs in order to choose the best candidates.

    Returns:
        A tuple containing:
            - a list of (score, attack_text) pairs, where the score is the model's
                output on the attack text.
            - a dictionary containing additional information about the
                evaluation, in particular the logits on the example in the
                classification setting.
    """

    attack_tokens_list = [
        victim.get_tokens(text, return_tensors=None)
        for text, _ in text_replacement_pairs
    ]

    candidate_attack_texts = victim.batch_decode(
        torch.cat(
            [
                candidate.compute_tokens_after_replacement(torch.tensor(attack_tokens))
                for attack_tokens, (_, candidate) in zip(
                    attack_tokens_list, text_replacement_pairs
                )
            ]
        ),
        skip_special_tokens=True,
    )

    # NOTE: We don't add the target here; it'll be added inside the
    # callback.
    # TODO (ian): Check that this doesn't mess up tokenization.
    full_prompts = [
        example.prompt_template.build_prompt(
            attack_text=attack_text,
            target="",
        )
        for attack_text in candidate_attack_texts
    ]
    goal_clf_labels = [example.clf_label] * len(candidate_attack_texts)
    goal_gen_targets = [example.gen_target] * len(candidate_attack_texts)

    callback_input = CallbackInput(
        input_data=full_prompts,
        clf_label_data=goal_clf_labels,
        gen_target_data=goal_gen_targets,
    )
    cb_out = scores_from_text_callback(victim, callback_input)
    losses = cb_out.losses

    evaluated_candidates = []
    for loss, text in zip(losses.to(device="cpu"), candidate_attack_texts):
        evaluated_candidates.append((float(loss), text))

    assert len(evaluated_candidates) == len(text_replacement_pairs)

    return evaluated_candidates, {"logits": cb_out.info["logits"]}
