import logging
from dataclasses import dataclass
from typing import Sequence, Tuple

from typing_extensions import override

from robust_llm.attacks.search_based.runners.search_based_runner import (
    SearchBasedRunner,
)
from robust_llm.attacks.search_based.utils import ReplacementCandidate
from robust_llm.utils import get_randint_with_exclusions

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class BeamSearchRunner(SearchBasedRunner):
    """Runs beam search on a single model and a single prompt/target pair.

    Attributes:
        beam_search_width: how many candidates to keep after each iteration.
    """

    beam_search_width: int = 5

    @override
    def _get_candidate_texts_and_replacements(
        self,
        candidate_texts: Sequence[str],
    ) -> list[Tuple[str, ReplacementCandidate]]:
        # We forbid introducing cls and sep tokens
        cls_token_id = self.wrapped_model.cls_token_id
        sep_token_id = self.wrapped_model.sep_token_id
        excluded_token_ids = [
            tok for tok in [cls_token_id, sep_token_id] if tok is not None
        ]

        text_replacement_pairs: list[Tuple[str, ReplacementCandidate]] = []

        for i in range(self.n_candidates_per_it):
            candidate_text = candidate_texts[i % len(candidate_texts)]
            # Sample a random replacement
            attack_position = self.candidate_sample_rng.randint(
                0, self.n_attack_tokens - 1
            )
            token_id = get_randint_with_exclusions(
                high=self.wrapped_model.vocab_size,
                exclusions=excluded_token_ids,
                rng=self.candidate_sample_rng,
            )
            candidate = ReplacementCandidate(attack_position, token_id)

            text_replacement_pairs.append((candidate_text, candidate))

        return text_replacement_pairs

    @property
    def n_best_candidates_to_keep(self) -> int:
        return self.beam_search_width
