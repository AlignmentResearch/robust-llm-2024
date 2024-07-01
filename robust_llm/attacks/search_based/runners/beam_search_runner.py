from collections.abc import Sequence
from dataclasses import dataclass

from typing_extensions import override

from robust_llm.attacks.search_based.runners.search_based_runner import (
    SearchBasedRunner,
)
from robust_llm.attacks.search_based.utils import PreppedExample, ReplacementCandidate
from robust_llm.config.callback_configs import CallbackConfig
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.utils import get_randint_with_exclusions


@dataclass(kw_only=True)
class BeamSearchRunner(SearchBasedRunner):
    """Runs beam search on a single model and a single prompt/target pair.

    Attributes:
        n_best_candidates_to_keep: how many candidates to keep after each iteration.
    """

    def __init__(
        self,
        victim: WrappedModel,
        n_candidates_per_it: int,
        n_its: int,
        n_attack_tokens: int,
        scores_from_text_callback: CallbackConfig,
        prepped_examples: Sequence[PreppedExample],
        random_seed: int = 0,
        beam_search_width: int = 5,
    ) -> None:
        super().__init__(
            victim=victim,
            n_candidates_per_it=n_candidates_per_it,
            n_its=n_its,
            n_attack_tokens=n_attack_tokens,
            scores_from_text_callback=scores_from_text_callback,
            prepped_examples=prepped_examples,
            random_seed=random_seed,
        )
        self.beam_search_width = beam_search_width

    @override
    def _get_candidate_texts_and_replacements(
        self,
        candidate_texts: Sequence[str],
    ) -> list[tuple[str, ReplacementCandidate]]:

        # We forbid introducing special tokens in the attack tokens.
        excluded_token_ids = self.victim.all_special_ids

        text_replacement_pairs: list[tuple[str, ReplacementCandidate]] = []

        for i in range(self.n_candidates_per_it):
            candidate_text = candidate_texts[i % len(candidate_texts)]
            # Sample a random replacement
            attack_position = self.candidate_sample_rng.randint(
                0, self.n_attack_tokens - 1
            )
            token_id = get_randint_with_exclusions(
                high=self.victim.vocab_size,
                exclusions=excluded_token_ids,
                rng=self.candidate_sample_rng,
            )
            candidate = ReplacementCandidate(attack_position, token_id)

            text_replacement_pairs.append((candidate_text, candidate))

        return text_replacement_pairs

    @property
    def n_best_candidates_to_keep(self) -> int:
        return self.beam_search_width
