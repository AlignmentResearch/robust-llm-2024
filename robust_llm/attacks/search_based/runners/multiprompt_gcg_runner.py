from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import torch
import torch.utils.data
from datasets import Dataset
from tqdm import tqdm
from typing_extensions import override

from robust_llm import logger
from robust_llm.attacks.search_based.runners.multiprompt_search_based_runner import (
    MultiPromptSearchBasedRunner,
)
from robust_llm.attacks.search_based.utils import (
    ExampleWithAttackIndices,
    ReplacementCandidate,
)


@dataclass(kw_only=True)
class MultiPromptGCGRunner(MultiPromptSearchBasedRunner):
    """Runs GCG on a single model and a single prompt/target pair.

    Attributes:
        top_k: the number of token replacements to consider at each
            position in the attack tokens.
    """

    top_k: int = 256

    @override
    def run(self) -> Tuple[str, dict[str, Any]]:
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
        text_replacement_pairs: Sequence[Tuple[str, ReplacementCandidate]],
        considered_examples: Sequence[ExampleWithAttackIndices],
    ) -> list[Tuple[str, ReplacementCandidate]]:
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
        attack_tokens_list: Sequence[int],
        replacement_list: Sequence[ReplacementCandidate],
    ) -> set[tuple[str, ReplacementCandidate]]:
        original_full_prompt_list = [
            example.prompt_template.build_prompt(
                attack_text=attack_text,
                target=self.target,
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

        decoded_list = self.wrapped_model.tokenizer.batch_decode(
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
        candidate_attack_texts = self.wrapped_model.tokenizer.batch_decode(
            torch.cat(candidate_attack_tokens_list)
        )
        candidate_full_prompt_list_alt = [
            example.prompt_template.build_prompt(
                attack_text=attack_text, target=self.target
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
        text_replacement_pairs: Sequence[Tuple[str, ReplacementCandidate]],
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
        text_replacement_pairs: Sequence[Tuple[str, ReplacementCandidate]],
        example: ExampleWithAttackIndices,
    ) -> list[tuple[float, str]]:
        attack_tokens_list = [
            self._get_tokens(text, return_tensors=None)
            for text, _ in text_replacement_pairs
        ]

        candidate_attack_texts = self.wrapped_model.tokenizer.batch_decode(
            torch.cat(
                [
                    candidate.compute_tokens_after_replacement(
                        torch.tensor([attack_tokens])
                    )
                    for attack_tokens, (_, candidate) in zip(
                        attack_tokens_list, text_replacement_pairs
                    )
                ]
            ),
            skip_special_tokens=True,
        )

        full_prompts = [
            example.prompt_template.build_prompt(
                attack_text=attack_text,
                target=self.target,
            )
            for attack_text in candidate_attack_texts
        ]
        full_prompts_tokens = self._get_tokens(full_prompts).to(
            self.wrapped_model.device
        )
        if self.seq_clf:
            assert example.clf_target is not None
            targets = torch.full(
                size=(full_prompts_tokens.shape[0],),
                fill_value=example.clf_target,
                device=self.wrapped_model.device,
            )
        else:
            # NOTE: target is the same for each candidate
            targets = full_prompts_tokens[:, example.attack_indices.target_slice]

        accelerator = self.wrapped_model.accelerator

        candidates_dataset = Dataset.from_dict(
            {
                "full_prompt_tokens": full_prompts_tokens,
                "target": targets,
                "candidate_attack_text": candidate_attack_texts,
            }
        ).with_format("torch")

        if accelerator is None:
            raise ValueError("An accelerator must be added to the model.")

        candidates_dataloader = accelerator.prepare(
            torch.utils.data.DataLoader(
                dataset=candidates_dataset,  # type: ignore
                batch_size=self.forward_pass_batch_size,
            )
        )

        evaluated_candidates = []

        for batch in candidates_dataloader:
            full_prompt_tokens = batch["full_prompt_tokens"]
            target = batch["target"]
            candidate_attack_text = batch["candidate_attack_text"]

            logits = self.wrapped_model.call_model(full_prompt_tokens)
            loss = self._compute_loss_from_logits(logits, target, example)

            candidate_attack_text = accelerator.gather_for_metrics(
                candidate_attack_text
            )
            loss = accelerator.gather_for_metrics(loss)

            for text, loss in zip(candidate_attack_text, loss):
                evaluated_candidates.append((float(loss), text))

        assert len(evaluated_candidates) == len(text_replacement_pairs)

        return evaluated_candidates

    @override
    def _get_candidate_texts_and_replacements(
        self,
        candidate_texts: Sequence[str],
        considered_examples: Sequence[ExampleWithAttackIndices],
    ) -> list[Tuple[str, ReplacementCandidate]]:
        # In GCG, the list of candidate texts should contain exactly the single best
        # candidate from the previous iteration
        assert len(candidate_texts) == 1
        attack_text = candidate_texts[0]

        gradients = self._compute_gradients(
            considered_examples, self.target, attack_text
        )
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
        target: str,
        attack_text: str,
    ) -> torch.Tensor:
        """For multi-prompt, we have to sum gradients for all examples we
        consider at this iteration."""
        gradients = None
        for example in considered_examples:
            full_prompt = example.prompt_template.build_prompt(
                attack_text=attack_text,
                target=target,
            )

            # Only need specials if we're doing BERT
            full_prompt_tokens = self._get_tokens(
                full_prompt, add_special=self.seq_clf
            ).to(self.wrapped_model.device)
            full_prompt_embeddings = self.wrapped_model.get_embeddings(
                full_prompt_tokens
            ).detach()

            attack_tokens = self._get_tokens(attack_text).to(self.wrapped_model.device)
            attack_onehot = self._get_attack_onehot(attack_tokens)
            attack_embeddings = (
                attack_onehot @ self.wrapped_model.get_embedding_weights()
            )

            combined_embeddings = self._get_combined_embeddings(
                full_prompt_embeddings,
                attack_embeddings,
                example,
            )

            # full_prompt_tokens are only used to determine the target in the generation
            # case; they are not used in the classification case.
            loss = self._compute_loss(combined_embeddings, full_prompt_tokens, example)
            if self.wrapped_model.accelerator is None:
                raise ValueError("An accelerator must be added to the model.")
            self.wrapped_model.accelerator.backward(loss)
            assert attack_onehot.grad is not None
            # accumulate gradients, linearly in memory
            if gradients is None:
                gradients = attack_onehot.grad.detach().clone()
            else:
                gradients += attack_onehot.grad.detach().clone()
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
        excluded_token_ids = self.wrapped_model.tokenizer.all_special_ids
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

    def _compute_loss(
        self,
        combined_embeddings: torch.Tensor,
        full_prompt_tokens: torch.Tensor,
        example: ExampleWithAttackIndices,
    ) -> torch.Tensor:
        """Compute loss given the combined embeddings.

        NOTE: we assume a batch dimension of size 1
        """
        # preconditions
        assert len(combined_embeddings.shape) == 3
        assert combined_embeddings.shape[0] == 1
        assert len(full_prompt_tokens.shape) == 2
        assert full_prompt_tokens.shape[0] == 1
        assert combined_embeddings.shape[1] == full_prompt_tokens.shape[1]

        logits = self.wrapped_model.call_model(inputs_embeds=combined_embeddings)
        assert logits.shape[0] == 1
        if self.seq_clf:
            # add batch dim
            targets = torch.tensor(
                example.clf_target, device=self.wrapped_model.device
            ).unsqueeze(0)
        else:
            targets = full_prompt_tokens[:, example.attack_indices.target_slice]
        loss = self._compute_loss_from_logits(logits, targets, example)
        assert loss.numel() == 1
        return loss.mean()

    def _compute_loss_from_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        example: ExampleWithAttackIndices,
    ) -> torch.Tensor:
        # only take a slice if it's a causal LM
        if self.seq_clf:
            target_logits = logits
        else:
            target_logits = logits[:, example.attack_indices.loss_slice, :]
            assert len(target_logits.shape) == 3
            # [batch_size, seq_len, vocab_size] -> [batch_size, vocab_size, seq_len]
            target_logits = target_logits.transpose(1, 2)
        ce_loss_func = torch.nn.CrossEntropyLoss(reduction="none")
        loss = ce_loss_func(target_logits, targets)
        if not self.seq_clf:
            # Reduce by seq dimension
            loss = loss.mean(dim=1)
        return loss

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
