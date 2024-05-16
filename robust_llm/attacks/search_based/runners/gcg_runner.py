import logging
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.utils.data
from typing_extensions import override

from robust_llm.attacks.search_based.runners.search_based_runner import (
    SearchBasedRunner,
)
from robust_llm.attacks.search_based.utils import ReplacementCandidate

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class GCGRunner(SearchBasedRunner):
    """Runs GCG on a single model and a single prompt/target pair.

    Attributes:
        top_k: the number of token replacements to consider at each
            position in the attack tokens.
    """

    top_k: int = 256

    @override
    def _get_candidate_texts_and_replacements(
        self,
        candidate_texts: Sequence[str],
    ) -> list[Tuple[str, ReplacementCandidate]]:
        # In GCG, the list of candidate texts should contain exactly the single best
        # candidate from the previous iteration
        assert len(candidate_texts) == 1
        attack_text = candidate_texts[0]

        gradients = self._compute_gradients(self.target, attack_text)
        replacements = self._get_replacement_candidates_from_gradients(gradients)
        candidate_texts_and_replacements = [
            (attack_text, replacement) for replacement in replacements
        ]

        return candidate_texts_and_replacements

    @property
    def n_best_candidates_to_keep(self) -> int:
        return 1

    def _compute_gradients(
        self,
        target: str,
        attack_text: str,
    ) -> torch.Tensor:
        full_prompt = self.example.prompt_template.build_prompt(
            attack_text=attack_text,
            target=target,
        )

        # Only need specials if we're doing BERT
        full_prompt_tokens = self._get_tokens(full_prompt, add_special=self.seq_clf).to(
            self.wrapped_model.device
        )
        full_prompt_embeddings = self.wrapped_model.get_embeddings(
            full_prompt_tokens
        ).detach()

        attack_tokens = self._get_tokens(attack_text).to(self.wrapped_model.device)
        attack_onehot = self._get_attack_onehot(attack_tokens)
        attack_embeddings = attack_onehot @ self.wrapped_model.get_embedding_weights()

        combined_embeddings = self._get_combined_embeddings(
            full_prompt_embeddings, attack_embeddings
        )

        # full_prompt_tokens are only used to determine the target in the generation
        # case; they are not used in the classification case.
        loss = self._compute_loss(combined_embeddings, full_prompt_tokens)

        if self.wrapped_model.accelerator is None:
            raise ValueError("An accelerator must be added to the model.")

        self.wrapped_model.accelerator.backward(loss)
        assert attack_onehot.grad is not None
        return attack_onehot.grad.clone()

    def _get_replacement_candidates_from_gradients(
        self,
        gradients: torch.Tensor,
    ) -> list[ReplacementCandidate]:
        """Gets the top k candidates from the gradients.

        `gradients` is a tensor of shape (n_attack_tokens, vocab_size).

        We take the top k from each position, and then randomly sample a
        batch of self.n_candidates_per_it < (top k * n_attack_tokens) replacements
        from the resulting pool.
        """
        # We forbid introducing special tokens in the attack tokens.
        excluded_token_ids = self.wrapped_model.tokenizer.all_special_ids
        for special_token_id in excluded_token_ids:
            gradients[:, special_token_id] = float("inf")

        # For each position, find the 'top_k' tokens with the largest negative gradient;
        # i.e. the tokens which if substituted are estimated to decrease loss the most.
        top_k_by_position = torch.topk(-gradients, self.top_k, dim=1)
        top_k_indices = top_k_by_position.indices
        # TODO(naimenz): do this in pytorch if possible
        pool = []
        for attack_position in range(len(top_k_indices)):
            for token_id in top_k_indices[attack_position]:
                pool.append(
                    ReplacementCandidate(
                        attack_position=attack_position,
                        token_id=token_id.item(),  # type: ignore
                    )
                )
        candidates = self.candidate_sample_rng.sample(pool, self.n_candidates_per_it)
        return candidates
