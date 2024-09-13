import abc
import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload

import torch
import torch.utils.data

from robust_llm import logger
from robust_llm.attacks.search_based.utils import (
    AttackIndices,
    AttackTokenizationChangeException,
    PreppedExample,
    ReplacementCandidate,
    create_onehot_embedding,
)
from robust_llm.config.callback_configs import CallbackConfig
from robust_llm.dist_utils import DistributedRNG
from robust_llm.models import WrappedModel
from robust_llm.scoring_callbacks import CallbackInput, build_tensor_scoring_callback


@dataclass(frozen=True)
class ReencodedReplacementCandidate:
    """A single candidate replacement with tokenization checks.

    We consider replacing a single token in the attack, decoding and reencoding the
    prompt, and checking if the tokenization changes. If it doesn't, we consider the
    candidate as a valid replacement.

    Attributes:
        text_replacement_pair: The attack text and the replacement candidate.
        attack_tokens: The attack token ids after the replacement.
        attack_text: The attack text after the replacement (from decoding
            `attack_tokens`).
        reencoded_attack_tokens: The attack token ids after decoding and
            re-encoding.
        tokens_by_replacement: The token ids of the full prompt by directly
            replacing the candidate attack token
        tokens_from_text: The token ids of the prompt after the replacement and
            re-tokenization.
    """

    text_replacement_pair: tuple[str, ReplacementCandidate]
    attack_tokens: list[int]
    attack_text: str
    reencoded_attack_tokens: list[int]
    tokens_by_replacement: list[int]
    tokens_from_text: list[int]

    def does_by_placement_match_from_text(self) -> bool:
        """Check that the two ways of building the new prompt match"""
        match = self.tokens_by_replacement == self.tokens_from_text
        if not match:
            logger.debug(
                f"Filtered out {self}"
                "because candidate_by_replacement != candidate_from_text"
            )
        return match

    def does_reencoded_match_candidate(self) -> bool:
        """Check that the new attack tokens are robust to retokenization.

        NOTE: This is *not* a weaker check that then previous one, because
        tokenizers can be affected by the previous tokens even when they
        really shouldn't be.
        (e.g. 'text -> 't | ext but ;'text -> ; | ' | text )
        """
        match = self.reencoded_attack_tokens == self.attack_tokens
        if not match:
            logger.debug(
                f"Filtered out {self}"
                "because reencoded_cand_attack_tokens != candidate_attack_tokens"
            )
        return match

    def is_pre_attack_unchanged(
        self, reference_tokens: list[int], indices: AttackIndices
    ) -> bool:
        """Check that the prompt before the attack is unchanged"""
        match = (
            reference_tokens[: indices.attack_start]
            == self.tokens_by_replacement[: indices.attack_start]
        )
        if not match:
            logger.debug(
                f"Filtered out {self.attack_text=}"
                "because reference_tokens[: indices.attack_start] != "
                "candidate_by_replacement[: indices.attack_start]"
            )
        return match

    def is_post_attack_unchanged(
        self, reference_tokens: list[int], indices: AttackIndices
    ) -> bool:
        """Check that the prompt after the attack is unchanged"""
        match = (
            reference_tokens[indices.attack_end :]
            == self.tokens_by_replacement[indices.attack_end :]
        )
        if not match:
            logger.debug(
                f"Filtered out {self.attack_text=}"
                "because reference_tokens[indices.attack_end :] != "
                "candidate_by_replacement[indices.attack_end :]"
            )
        return match

    def is_replacement_changed(self, reference_tokens: list[int]) -> bool:
        """Check that the new prompt is not identical to the old one"""
        is_novel = reference_tokens != self.tokens_by_replacement
        if not is_novel:
            logger.debug(
                f"Filtered out {self.attack_text=}"
                "because reencoded_cand_attack_tokens != candidate_attack_tokens"
            )
        return is_novel

    def is_valid(self, reference_tokens: list[int], indices: AttackIndices) -> bool:
        return (
            self.does_by_placement_match_from_text()
            and self.does_reencoded_match_candidate()
            and self.is_pre_attack_unchanged(reference_tokens, indices)
            and self.is_post_attack_unchanged(reference_tokens, indices)
            and self.is_replacement_changed(reference_tokens)
        )


class SearchBasedRunner(abc.ABC):
    """Runs search based attack on a single model and a single prompt/target pair.

    Base class for the runners of search-based attacks. All the logic of iteration,
    search, filtering, etc. is implemented mostly here, with the approach-specific
    methods being implemented in subclasses.

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
    ) -> None:
        """Constructor for the SearchBasedRunner class.

        Args:
            victim: The model to attack paired with a tokenizer
                and some model-specific methods
            n_candidates_per_it: the total number of token replacements
                to consider in each iteration (in GCG, this must be less than
                top_k * n_attack_tokens, which is the total number of candidates)
            n_its: Total number of iterations to run
            n_attack_tokens: number of attack tokens to optimize
            scores_from_text_callback: the callback for computing the scores for
                inputs in order to choose the best candidates.
            prepped_examples: list of PreppedExample which includes a
                prompt_template, clf_label, and gen_target
            random_seed: initial seed for a random.Random object used to sample
                replacement candidates
        """
        self.victim = victim
        cb = build_tensor_scoring_callback(scores_from_text_callback)
        self.scores_from_text_callback = cb

        self.n_candidates_per_it = n_candidates_per_it
        self.n_its = n_its
        self.n_attack_tokens = n_attack_tokens
        self.candidate_sample_rng = DistributedRNG(random_seed, victim.accelerator)

        assert len(prepped_examples) == 1, "only one prompt/target pair supported"
        self.example = prepped_examples[0]

        self.initial_attack_text, self.attack_indices = (
            self._get_initial_attack_text_and_indices(self.n_attack_tokens)
        )

    def run(self) -> tuple[str, dict[str, Any]]:
        """Runs the attack and returns the adversarial text and debug info dict."""
        attack_text = self.initial_attack_text
        cand_texts = [attack_text]

        # In how many iterations it happened that all candidates were filtered out
        all_filtered_out_count = 0

        attack_strings = []
        all_logits = []
        for _ in range(self.n_its):
            candidate_texts_and_replacements = (
                self._get_candidate_texts_and_replacements(cand_texts)
            )
            candidate_texts_and_replacements = self._filter_candidates(
                candidate_texts_and_replacements
            )
            if len(candidate_texts_and_replacements) == 0:
                all_filtered_out_count += 1
                continue
            evaluated_candidates, eval_info = (
                self._apply_replacements_and_eval_candidates(
                    candidate_texts_and_replacements
                )
            )
            cand_texts, cand_indices = self._select_next_candidates(
                evaluated_candidates
            )
            attack_text = cand_texts[0]
            attack_index = cand_indices[0]
            attack_strings.append(attack_text)
            eval_logits = eval_info["logits"]
            all_logits.append(
                eval_logits[attack_index] if eval_logits is not None else None
            )

        info_dict = {
            "all_filtered_out_count": all_filtered_out_count,
            "attack_strings": attack_strings,
            "logits": all_logits,
        }

        return attack_text, info_dict

    @abc.abstractmethod
    def _get_candidate_texts_and_replacements(
        self,
        candidate_texts: Sequence[str],
    ) -> list[tuple[str, ReplacementCandidate]]:
        """Proposes a set of (attack_text, replacement) candidate pairs to consider."""

    def _select_next_candidates(
        self, candidates: list[tuple[float, str]]
    ) -> tuple[list[str], list[int]]:
        """Selects text candidates for the next round, based on (score, text) pairs.

        Args:
            candidates: A list of (score, text) pairs to select from.

        Returns:
            A list of the next candidates to consider and a list of their indices.
        """
        indexed_candidates = list(enumerate(candidates))
        sorted_candidates = list(sorted(indexed_candidates, key=lambda x: x[1][0]))
        next_candidates = [
            candidate[1][1]
            for candidate in sorted_candidates[: self.n_best_candidates_to_keep]
        ]
        next_candidates_indices = [
            candidate[0]
            for candidate in sorted_candidates[: self.n_best_candidates_to_keep]
        ]
        return next_candidates, next_candidates_indices

    @property
    @abc.abstractmethod
    def n_best_candidates_to_keep(self) -> int:
        pass

    def _get_initial_attack_text_and_indices(
        self, n_attack_tokens: int
    ) -> tuple[str, AttackIndices]:
        """Initialize attack text with a sequence of "@&@&...@&

        The length of the initial attack text will be `self.n_attack_tokens`.

        In case the initial attack text is not valid (because of retokenization issues),
        randomly perturb it until success.

        The initial attack text used in the reference is "! ! ! !" (and so on).
        I found that when using that with some tokenizers (e.g. gpt2),
        encoding, decoding, and encoding again was removing the spaces and
        then encoding the string into tokens containing multiple `!` together.
        I chose @& because both characters are unlikely to merge with tokens
        before/after them (@ is first because it doesn't merge with ':').

        TODO(GH#117): investigate loss of spaces further.
        """
        optional_character = "@" if n_attack_tokens % 2 == 1 else ""
        attack_text = "@&" * (n_attack_tokens // 2) + optional_character
        attack_tokens = self._get_tokens(attack_text, return_tensors="pt")
        assert len(attack_tokens[0]) == n_attack_tokens

        try:
            attack_indices = self._get_attack_indices(attack_text, self.example)
            return attack_text, attack_indices
        # Note that we only catch the exception in case of retokenization issue caused
        # by the attack tokens. If there is an issue because of target tokens, it means
        # that they got mixed with after_attack tokens. For now we just let it crash.
        # TODO(GH#197): handle the target case properly. This will be needed for
        # generation tasks. In order to do this, we'd need to never tokenize the target
        # together with previous text chunks; always do it separately and then
        # concatenate the tokens. This is fine because the target is not a part of the
        # input passed to the model, so the joint tokenization does not have to be
        # "canonical" one (i.e. robust to retokenization).
        except AttackTokenizationChangeException:
            pass

        MAX_NUM_TRIALS = 10_000
        trials = 0

        while trials < MAX_NUM_TRIALS:
            # Sweep over the indices and try random replacements.
            for i in range(n_attack_tokens):
                trials += 1

                attack_tokens[0, i] = self.candidate_sample_rng.randint(
                    0, self.victim.vocab_size - 1
                )
                attack_text = self.victim.decode_tokens(attack_tokens)
                try:
                    attack_indices = self._get_attack_indices(attack_text, self.example)
                    return attack_text, attack_indices
                except AttackTokenizationChangeException:
                    pass

        # We exceeded the maximum number of trials, so we raise an exception.
        raise AttackTokenizationChangeException

    @overload
    def _get_tokens(
        self,
        inputs: str | list[str],
        return_tensors: Literal[None] = None,
        add_special_and_chat: bool = False,
    ) -> list[list[int]]: ...

    @overload
    def _get_tokens(
        self,
        inputs: str | list[str],
        return_tensors: Literal["pt"],
        add_special_and_chat: bool = False,
    ) -> torch.Tensor: ...

    def _get_tokens(
        self,
        inputs: str | list[str],
        return_tensors: Literal[None, "pt"] = None,
        add_special_and_chat: bool = False,
    ) -> list[list[int]] | torch.Tensor:
        """Tokenize the inputs and return the token ids.

        Use tokenizer which is part of the wrapped model. Handle all the arguments we
        have to add to the tokenizer.

        Args:
            inputs: The input text or list of texts to tokenize.
            return_tensors: Whether to return tensors, and what type of tensors to
                return.
            add_special_and_chat: Whether to add special tokens and use chat template.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        encoded = self.victim.tokenize(
            inputs,
            return_tensors=return_tensors,
            add_special_tokens=add_special_and_chat,
            use_chat_template=add_special_and_chat,
        )
        input_ids = encoded["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(device=self.victim.device)
        return input_ids  # type: ignore  # mypy thinks it's EncodingFast | Any

    def _decode_tokens(
        self,
        inp: torch.Tensor,
        skip_special_tokens: bool = True,
        try_squeeze: bool = True,
    ) -> str:
        strings = self.victim.decode_tokens(
            inp,
            skip_special_tokens=skip_special_tokens,
            try_squeeze=try_squeeze,
        )
        return strings

    def _get_attack_indices(
        self, attack_text: str, example: PreppedExample
    ) -> AttackIndices:
        """Computes the start-end indices of the attack & target.

        Indices are relative to the tokenized string. Returns an object containing
        those indices along with convenience methods.

        NOTE: In general getting the indices this way is not robust, since
        tokenize(s1) + tokenize(s2) != tokenize(s1 + s2) in all cases.
        However, there are several checks to make sure that the tokenization
        does not change after concatenation. This is also the way it
        is implemented in the reference code:
        (https://github.com/llm-attacks/llm-attacks/blob/0f505d/llm_attacks/base/attack_manager.py#L130)  # noqa: E501

        Raises:
            AttackTokenizationChangeException: If the tokenization changes after
                concatenating the strings because of the attack tokens.
            TargetTokenizationChangeException: If the tokenization changes after
                concatenating the strings because of the target tokens.
        """
        before_attack_tokens = self._get_tokens(
            example.prompt_template.before_attack, return_tensors="pt"
        )
        attack_start = before_attack_tokens.shape[1]
        attack_end = self.n_attack_tokens + attack_start

        full_prompt = example.prompt_template.build_prompt(
            attack_text=attack_text,
            target=example.gen_target,
        )
        full_tokens = self._get_tokens(full_prompt, return_tensors="pt")
        target_end = full_tokens.shape[1]
        target_tokens = self._get_tokens(example.gen_target, return_tensors="pt")
        target_start = target_end - target_tokens.shape[1]

        attack_indices = AttackIndices(
            attack_start=attack_start,
            attack_end=attack_end,
            target_start=target_start,
            target_end=target_end,
        )

        # Check that the attack indices actually line up.
        attack_tokens = self._get_tokens(attack_text, return_tensors="pt")
        attack_indices.assert_attack_and_target_tokens_validity(
            full_tokens, attack_tokens, target_tokens
        )

        return attack_indices

    @torch.no_grad()
    def _apply_replacements_and_eval_candidates(
        self,
        text_replacement_pairs: Sequence[tuple[str, ReplacementCandidate]],
    ) -> tuple[list[tuple[float, str]], dict[str, Any]]:
        """Evaluates the candidates using a forward pass through the model.

        Args:
            text_replacement_pairs: A list of (attack_text, replacement) pairs to
                evaluate.

        Returns:
            A tuple containing:
                - a list of (score, attack_text) pairs, where the score is the model's
                    output on the attack text.
                - a dictionary containing additional information about the
                    evaluation, in particular the logits on the example in the
                    classification setting.
        """

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

        # NOTE: We don't add the target here; it'll be added inside the
        # callback.
        # TODO (ian): Check that this doesn't mess up tokenization.
        full_prompts = [
            self.example.prompt_template.build_prompt(
                attack_text=attack_text,
                target="",
            )
            for attack_text in candidate_attack_texts
        ]
        goal_clf_labels = [self.example.clf_label] * len(candidate_attack_texts)
        goal_gen_targets = [self.example.gen_target] * len(candidate_attack_texts)

        callback_input = CallbackInput(
            input_data=full_prompts,
            clf_label_data=goal_clf_labels,
            gen_target_data=goal_gen_targets,
        )
        cb_out = self.scores_from_text_callback(self.victim, callback_input)
        losses = cb_out.losses

        evaluated_candidates = []
        for loss, text in zip(losses.to(device="cpu"), candidate_attack_texts):
            evaluated_candidates.append((float(loss), text))

        assert len(evaluated_candidates) == len(text_replacement_pairs)

        return evaluated_candidates, {"logits": cb_out.info["logits"]}

    def prep_candidates_using_batched_tokenization(
        self,
        text_replacement_pairs: Sequence[tuple[str, ReplacementCandidate]],
        reference_tokens: list[int],
        reference_attack_tokens: list[list[int]],
    ) -> list[ReencodedReplacementCandidate]:
        """Prepares the candidates using batched tokenization.

        Attributes:
            text_replacement_pairs: A list of (attack_text, replacement) pairs to
                prepare.
            reference_tokens: The token ids of a valid full prompt, used to check
                for tokenization changes.
            reference_attack_tokens: The token ids of a valid attack string. This
                is a sub-list of reference_tokens.
        """
        template = self.example.prompt_template
        indices = self.attack_indices

        # Perform the replacements and get the new attack tokens.
        candidate_attack_tokens_list = [
            candidate.compute_tokens_after_replacement(
                reference_attack_tokens, tensors=None
            )[0]
            for _, candidate in text_replacement_pairs
        ]

        # Decode and re-encode to check for tokenization differences.
        candidate_attack_text_list = self.victim.batch_decode(
            candidate_attack_tokens_list
        )
        reencoded_cand_attack_tokens_list = self._get_tokens(
            candidate_attack_text_list, return_tensors=None
        )

        # candidate_by_replacement_list holds the new prompt tokens created
        # *by replacement* of a single attack token in the tokenized prompt with the
        # candidate token.
        candidate_by_replacement_list = [
            copy.deepcopy(reference_tokens) for _ in range(len(text_replacement_pairs))
        ]
        for i, candidate_attack_tokens in enumerate(candidate_attack_tokens_list):
            candidate_by_replacement_list[i][
                indices.attack_start : indices.attack_end
            ] = candidate_attack_tokens

        candidate_text_list = [
            template.build_prompt(attack_text=candidate_attack_text)
            for candidate_attack_text in candidate_attack_text_list
        ]

        # candidate_from_text_list holds the new prompt tokens created by detokenizing
        # the new attack, adding it to the prompt, and tokenizing the whole thing.
        candidate_from_text_list = self._get_tokens(
            candidate_text_list, return_tensors=None
        )
        return [
            ReencodedReplacementCandidate(
                text_replacement_pair=text_replacement_pairs[i],
                attack_tokens=candidate_attack_tokens_list[i],
                attack_text=candidate_attack_text_list[i],
                reencoded_attack_tokens=reencoded_cand_attack_tokens_list[i],
                tokens_by_replacement=candidate_by_replacement_list[i],
                tokens_from_text=candidate_from_text_list[i],
            )
            for i in range(len(text_replacement_pairs))
        ]

    def _filter_candidates(
        self,
        text_replacement_pairs: Sequence[tuple[str, ReplacementCandidate]],
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
            text_replacement_pairs: A list of (attack_text, replacement) pairs
                to filter. The attack text is the current text to replace, and the
                replacement is the token to replace it with.

        Returns:
            A list of the candidates that didn't result in a change in tokenization
                and that changed the attack text
        """
        assert len(text_replacement_pairs) > 0
        indices = self.attack_indices
        template = self.example.prompt_template

        # The reference only needs to be computed once as it's the same for all
        # candidates.
        current_attack_text = text_replacement_pairs[0][0]
        reference_prompt = template.build_prompt(attack_text=current_attack_text)
        reference_tokens = self._get_tokens(reference_prompt, return_tensors=None)[0]
        reference_attack_tokens = self._get_tokens(
            current_attack_text, return_tensors=None
        )

        candidate_replacements = self.prep_candidates_using_batched_tokenization(
            text_replacement_pairs, reference_tokens, reference_attack_tokens
        )
        filtered_candidates = [
            cand.text_replacement_pair
            for cand in candidate_replacements
            if cand.is_valid(reference_tokens, indices)
        ]

        logger.debug(
            f"Filtered from {len(text_replacement_pairs)} to {len(filtered_candidates)}"
        )
        if len(filtered_candidates) == 0:
            logger.warning("All candidates were filtered out!")

        return filtered_candidates

    def _tokenization_changed(
        self,
        candidate_full_prompt_tokens: torch.Tensor,
        original_full_prompt_tokens: torch.Tensor,
        encoded_decoded: torch.Tensor,
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
            :, self.attack_indices.attack_slice
        ]
        new_attack_tokens = encoded_decoded[:, self.attack_indices.attack_slice]
        attacks_equal = torch.equal(old_attack_tokens, new_attack_tokens)
        if not attacks_equal:
            return True

        # check specifically that the target is the same in the
        # re-encoded version as in the original, unreplaced prompt
        old_target_tokens = original_full_prompt_tokens[
            :, self.attack_indices.target_slice
        ]
        new_target_tokens = encoded_decoded[:, self.attack_indices.target_slice]
        targets_equal = torch.equal(old_target_tokens, new_target_tokens)
        if not targets_equal:
            return True

        return False

    def _get_combined_embeddings(
        self,
        full_prompt_embeddings: torch.Tensor,
        attack_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # TODO: check that there are no off-by-one errors
        combined_embeddings = torch.cat(
            [
                full_prompt_embeddings[:, : self.attack_indices.attack_start, :],
                attack_embeddings.unsqueeze(0),  # add back a batch dimension
                full_prompt_embeddings[:, self.attack_indices.attack_end :, :],
            ],
            dim=1,
        )
        if combined_embeddings.shape != full_prompt_embeddings.shape:
            logger.warning(
                "Crashing on example:\n"
                "prompt_template={prompt_template}".format(
                    prompt_template=self.example.prompt_template
                )
            )
            raise ValueError(
                f"Combined embeddings shape {combined_embeddings.shape} "
                f"does not match full prompt embeddings shape "
                f"{full_prompt_embeddings.shape}"
            )
        return combined_embeddings

    def _get_attack_onehot(self, attack_tokens: torch.Tensor) -> torch.Tensor:
        """Creates a one-hot encoding layer before the model embeddings.

        We can then backpropagate through it.
        """
        embedding_weights = self.victim.get_embedding_weights()
        vocab_size = embedding_weights.shape[0]
        dtype = embedding_weights.dtype
        attack_onehot = create_onehot_embedding(
            token_ids=attack_tokens,
            vocab_size=vocab_size,
            dtype=dtype,
            device=self.victim.model.device,
        )
        return attack_onehot
