from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import torch
import torch.utils.data
from tqdm import tqdm
from typing_extensions import override

from robust_llm import logger
from robust_llm.attacks.search_based.runners.multiprompt_search_based_runner import (
    MultiPromptSearchBasedRunner,
)
from robust_llm.attacks.search_based.utils import (
    ExampleWithAttackIndices,
    PreppedExample,
    ReplacementCandidate,
)
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.scoring_callbacks import CallbackInput, CallbackRegistry


class MultiPromptGCGRunner(MultiPromptSearchBasedRunner):
    """Runs GCG on a single model and a single prompt/target pair.

    Attributes:
        top_k: the number of token replacements to consider at each
            position in the attack tokens.
    """

    def __init__(
        self,
        victim: WrappedModel,
        n_candidates_per_it: int,
        n_its: int,
        n_attack_tokens: int,
        scores_from_text_callback: str,
        prepped_examples: Sequence[PreppedExample],
        differentiable_embeds_callback: str,
        random_seed: int = 0,
        top_k: int = 256,
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

        cb = CallbackRegistry.get_tensor_callback(differentiable_embeds_callback)
        self.differentiable_embeds_callback = cb
        self.top_k = top_k

    top_k: int = 256

    @override
    def run(self) -> tuple[str, dict[str, Any]]:
        """Runs the attack and returns the adversarial text and debug info dict."""
        attack_text = self.initial_attack_text
        candidate_texts = [attack_text]

        # In how many iterations it happened that all candidates were filtered out
        all_filtered_out_count = 0

        for i in (pbar := tqdm(range(self.n_its))):
            # TODO (ian): track whether attack is successful on each prompt
            # and add considered_examples when all are successful. For now we
            # just add one at a time
            considered_examples = self.examples_with_attack_ix[: i + 1]

            candidate_texts_and_replacements = (
                self._get_candidate_texts_and_replacements(
                    candidate_texts, considered_examples
                )
            )
            candidate_texts_and_replacements = self._filter_candidates(
                candidate_texts_and_replacements, considered_examples
            )
            if len(candidate_texts_and_replacements) == 0:
                all_filtered_out_count += 1
                continue
            evaluated_candidates = self._apply_replacements_and_eval_candidates(
                candidate_texts_and_replacements, considered_examples
            )
            candidate_texts = self._select_next_candidates(evaluated_candidates)
            attack_text = candidate_texts[0]

            # TODO(GH#112): track progress more cleanly
            pbar.set_description(f"Attack text: {attack_text}")

        info_dict = {"all_filtered_out_count": all_filtered_out_count}

        return attack_text, info_dict

    def _filter_candidates(
        self,
        text_replacement_pairs: Sequence[tuple[str, ReplacementCandidate]],
        considered_examples: Sequence[ExampleWithAttackIndices],
    ) -> list[tuple[str, ReplacementCandidate]]:
        """Removes candidates where tokenization changes, or the attack is unchanged.

        By default, it's possible for replacements to lead to changes in
        tokenization when the sequence is decoded and re-encoded. This is a
        problem because it means that other tokens in the prompt are changed
        unintentionally, and that the whole sequence length might change
        (meaning that tokens from the attack bleed into the rest of the prompt).

        It's also possible for replacements to replace a token with itself,
        which could lead to being stuck in a loop.

        Args:
            text_replacement_pairs: The attack text and replacement candidates
            considered_examples: The examples used to compute the losses

        Returns:
            A list of the candidates that didn't result in a change in tokenization
                and that changed the attack text
        """

        # The following code is written in a way so that we process whole lists, hence
        # tokenization (which is most costly) can be batched.

        attack_text_list = [attack_text for attack_text, _ in text_replacement_pairs]
        replacement_list = [replacement for _, replacement in text_replacement_pairs]

        attack_tokens_list = self._get_tokens(attack_text_list, return_tensors=None)
        assert isinstance(attack_tokens_list, list)

        # candidates that will be removed because they mess with tokenization of
        # some example
        candidates_to_filter_out = set()

        for example in considered_examples:
            example_candidates_to_filter_out = self._compute_candidates_to_filter_out(
                example,
                attack_text_list,
                attack_tokens_list,
                replacement_list,
            )
            candidates_to_filter_out.update(example_candidates_to_filter_out)
        filtered_candidates = [
            trp for trp in text_replacement_pairs if trp not in candidates_to_filter_out
        ]
        logger.debug(
            f"Filtered from {len(text_replacement_pairs)} to {len(filtered_candidates)}"
        )
        return filtered_candidates

    def _compute_candidates_to_filter_out(
        self,
        example: ExampleWithAttackIndices,
        attack_text_list: Sequence[str],
        attack_tokens_list: Sequence[Sequence[int]],
        replacement_list: Sequence[ReplacementCandidate],
    ) -> set[tuple[str, ReplacementCandidate]]:
        original_full_prompt_list = [
            example.prompt_template.build_prompt(
                attack_text=attack_text,
                target=example.gen_target,
            )
            for attack_text in attack_text_list
        ]

        original_full_prompt_tokens_list = [
            torch.tensor([tokens])
            for tokens in self._get_tokens(
                original_full_prompt_list, return_tensors=None
            )
        ]

        candidate_attack_tokens_list = [
            replacement.compute_tokens_after_replacement(torch.tensor([attack_tokens]))
            for replacement, attack_tokens in zip(replacement_list, attack_tokens_list)
        ]

        # combine the original prompt with the candidate replacement
        candidate_full_prompt_tokens_list = [
            torch.cat(
                [
                    original_full_prompt_tokens[
                        :, : example.attack_indices.attack_start
                    ],
                    candidate_attack_tokens,
                    original_full_prompt_tokens[:, example.attack_indices.attack_end :],
                ],
                dim=1,
            )
            for candidate_attack_tokens, original_full_prompt_tokens in zip(
                candidate_attack_tokens_list, original_full_prompt_tokens_list
            )
        ]

        decoded_list = self.victim.batch_decode(
            torch.cat([tokens for tokens in candidate_full_prompt_tokens_list]),
            skip_special_tokens=True,
        )
        encoded_decoded_list = [
            torch.tensor([tokens])
            for tokens in self._get_tokens(decoded_list, return_tensors=None)
        ]

        # In the checks prepared above, we construct candidate full prompts by modifying
        # tokens in the original full prompt.
        # Below, we instead first decode candidate attacks into strings, and build
        # candidate full prompts from strings. This is the way it is actually done
        # later in the code. It turns out that in some rare cases these two ways do not
        # coincide, and so the checks relying on the first way are not sufficient.
        # TODO(michal): refactor this so that we only have one type of check.
        candidate_attack_texts = self.victim.batch_decode(
            torch.cat(candidate_attack_tokens_list)
        )
        candidate_full_prompt_list_alt = [
            example.prompt_template.build_prompt(
                attack_text=attack_text, target=example.gen_target
            )
            for attack_text in candidate_attack_texts
        ]
        candidate_full_prompt_tokens_list_alt = self._get_tokens(
            candidate_full_prompt_list_alt, return_tensors=None
        )

        candidates_to_filter = set()
        for (
            attack_text,
            candidate,
            candidate_full_prompt_tokens,
            original_full_prompt_tokens,
            encoded_decoded,
            candidate_full_prompt_tokens_alt,
        ) in zip(
            attack_text_list,
            replacement_list,
            candidate_full_prompt_tokens_list,
            original_full_prompt_tokens_list,
            encoded_decoded_list,
            candidate_full_prompt_tokens_list_alt,
        ):
            if torch.equal(candidate_full_prompt_tokens, original_full_prompt_tokens):
                candidates_to_filter.add((attack_text, candidate))

            if self._tokenization_changed(
                candidate_full_prompt_tokens,
                original_full_prompt_tokens,
                encoded_decoded,
                example,
            ):
                candidates_to_filter.add((attack_text, candidate))

            if len(candidate_full_prompt_tokens_alt) != len(
                original_full_prompt_tokens[0]
            ):
                candidates_to_filter.add((attack_text, candidate))

        return candidates_to_filter

    def _tokenization_changed(
        self,
        candidate_full_prompt_tokens: torch.Tensor,
        original_full_prompt_tokens: torch.Tensor,
        encoded_decoded: torch.Tensor,
        example: ExampleWithAttackIndices,
    ) -> bool:
        """Returns True if the tokenization changed after the replacement.

        We only take candidates if the tokenization doesn't change (i.e. False).
        To check that the tokenization doesn't importantly change, we ensure that the
        length of the whole token sequence is the same after decoding and encoding, and
        that each individual section (attack, target) is the same before and after.
        """
        if encoded_decoded.shape[1] != candidate_full_prompt_tokens.shape[1]:
            return True

        # check that the attack hasn't changed after decoding and encoding
        old_attack_tokens = candidate_full_prompt_tokens[
            :, example.attack_indices.attack_slice
        ]
        new_attack_tokens = encoded_decoded[:, example.attack_indices.attack_slice]
        attacks_equal = torch.equal(old_attack_tokens, new_attack_tokens)
        if not attacks_equal:
            return True

        # check specifically that the target is the same in the
        # re-encoded version as in the original, unreplaced prompt
        old_target_tokens = original_full_prompt_tokens[
            :, example.attack_indices.target_slice
        ]
        new_target_tokens = encoded_decoded[:, example.attack_indices.target_slice]
        targets_equal = torch.equal(old_target_tokens, new_target_tokens)
        if not targets_equal:
            return True

        return False

    @torch.no_grad()
    def _apply_replacements_and_eval_candidates(
        self,
        text_replacement_pairs: Sequence[tuple[str, ReplacementCandidate]],
        considered_examples: Sequence[ExampleWithAttackIndices],
    ) -> list[tuple[float, str]]:
        """Evaluates the candidates using a forward pass through the model for
        each prompt."""
        accumulated_scores: dict[str, list[float]] = defaultdict(list)
        for example in considered_examples:
            scores = self._apply_replacements_and_eval_candidates_one_prompt(
                text_replacement_pairs, example
            )
            for score, candidate in scores:
                accumulated_scores[candidate].append(score)
        final_scores = [
            (sum(scores), candidate) for candidate, scores in accumulated_scores.items()
        ]
        return final_scores

    def _apply_replacements_and_eval_candidates_one_prompt(
        self,
        text_replacement_pairs: Sequence[tuple[str, ReplacementCandidate]],
        example: ExampleWithAttackIndices,
    ) -> list[tuple[float, str]]:
        attack_tokens_list = [
            self._get_tokens(text, return_tensors=None)
            for text, _ in text_replacement_pairs
        ]

        candidate_attack_texts = self.victim.batch_decode(
            torch.cat(
                [
                    candidate.compute_tokens_after_replacement(
                        torch.tensor(attack_tokens)
                    )
                    for attack_tokens, (_, candidate) in zip(
                        attack_tokens_list, text_replacement_pairs
                    )
                ]
            ),
            skip_special_tokens=True,
        )

        # NOTE: we don't add the target to the prompt here; that'll be handled
        # in the callback.
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
        callback_out = self.scores_from_text_callback(self.victim, callback_input)
        losses = callback_out.losses

        evaluated_candidates = []
        for loss, text in zip(losses, candidate_attack_texts):
            evaluated_candidates.append((float(loss), text))

        assert len(evaluated_candidates) == len(text_replacement_pairs)

        return evaluated_candidates

    @override
    def _get_candidate_texts_and_replacements(
        self,
        candidate_texts: Sequence[str],
        considered_examples: Sequence[ExampleWithAttackIndices],
    ) -> list[tuple[str, ReplacementCandidate]]:
        # In GCG, the list of candidate texts should contain exactly the single best
        # candidate from the previous iteration
        assert len(candidate_texts) == 1
        attack_text = candidate_texts[0]

        gradients = self._compute_gradients(considered_examples, attack_text)
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
        considered_examples: Sequence[ExampleWithAttackIndices],
        attack_text: str,
    ) -> torch.Tensor:
        """For multi-prompt, we have to sum gradients for all examples we
        consider at this iteration."""
        gradients = None
        # TODO(ian): Batch this (once I make running on embeds support batches).
        for example in considered_examples:
            # NOTE: We don't add the target to the prompt here; that'll be handled
            # in the callback.
            full_prompt = example.prompt_template.build_prompt(
                attack_text=attack_text,
                target="",
            )

            full_prompt_tokens = self._get_tokens(full_prompt, return_tensors="pt")
            full_prompt_embeddings = self.victim.get_embeddings(full_prompt_tokens)

            attack_tokens = self._get_tokens(attack_text, return_tensors="pt").to(
                self.victim.device
            )
            attack_onehot = self._get_attack_onehot(attack_tokens)
            attack_embeddings = attack_onehot @ self.victim.get_embedding_weights()

            combined_embeddings = self._get_combined_embeddings(
                full_prompt_embeddings,
                attack_embeddings,
                example,
            )

            input_data = {
                "input_ids": full_prompt_tokens,
                "embeddings": combined_embeddings,
            }

            cb_in = CallbackInput(
                input_data=input_data,
                clf_label_data=[example.clf_label],
                gen_target_data=[example.gen_target],
            )
            cb_out = self.differentiable_embeds_callback(self.victim, cb_in)
            losses = cb_out.losses

            assert len(losses) == 1, "Batch size should be 1."
            loss = losses[0]

            if self.victim.accelerator is None:
                raise ValueError("An accelerator must be added to the model.")

            self.victim.accelerator.backward(loss)
            assert attack_onehot.grad is not None
            # accumulate gradients, linearly in memory
            if gradients is None:
                gradients = attack_onehot.grad.detach().cpu()
            else:
                gradients += attack_onehot.grad.detach().cpu()
        assert gradients is not None
        return gradients

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
        excluded_token_ids = self.victim.all_special_ids
        for token_id in excluded_token_ids:
            gradients[:, token_id] = float("inf")

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

    def _get_combined_embeddings(
        self,
        full_prompt_embeddings: torch.Tensor,
        attack_embeddings: torch.Tensor,
        example: ExampleWithAttackIndices,
    ) -> torch.Tensor:
        # TODO: check that there are no off-by-one errors
        combined_embeddings = torch.cat(
            [
                full_prompt_embeddings[:, : example.attack_indices.attack_start, :],
                attack_embeddings.unsqueeze(0),  # add back a batch dimension
                full_prompt_embeddings[:, example.attack_indices.attack_end :, :],
            ],
            dim=1,
        )
        assert combined_embeddings.shape == full_prompt_embeddings.shape
        return combined_embeddings
