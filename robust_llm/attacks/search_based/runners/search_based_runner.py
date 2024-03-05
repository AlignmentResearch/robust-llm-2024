import abc
import logging
import random
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.utils.data
from tqdm import tqdm

from robust_llm.attacks.search_based.models import SearchBasedAttackWrappedModel
from robust_llm.attacks.search_based.utils import (
    AttackIndices,
    AttackTokenizationChangeException,
    PromptTemplate,
    ReplacementCandidate,
    create_onehot_embedding,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchBasedRunner(abc.ABC):
    """Runs search based attack on a single model and a single prompt/target pair.

    Base class for the runners of search-based attacks. All the logic of iteration,
    search, filtering, etc. is implemented mostly here, with the approach-specific
    methods being implemented in subclasses.

    Attributes:
        wrapped_model: The model to attack paired with a tokenizer
            and some model-specific methods
        n_candidates_per_it: the total number of token replacements
            to consider in each iteration (in GCG, this must be less than
            top_k * n_attack_tokens, which is the total number of candidates)
        n_its: Total number of iterations to run
        n_attack_tokens: number of attack tokens to optimize
        forward_pass_batch_size: batch size used for forward pass when evaluating
            candidates. If None, defaults to n_candidates_per_it
        target: If using a CausalLM, it's the target string to optimize for.
            If using a SequenceClassification model, it's ignored in favor of
            the target specified by clf_target
        prompt_template: The PromptTemplate defines the format of the prompt,
            and its `build_prompt` method is used for producing full prompts
        seq_clf: Whether we are using a SequenceClassification model
            (default alternative is a CausalLM)
        clf_target: Only used for sequence classification, specifies the class
            to optimize for
        random_seed: initial seed for a random.Random object used to sample
            replacement candidates
    """

    wrapped_model: SearchBasedAttackWrappedModel
    n_candidates_per_it: int
    n_its: int
    n_attack_tokens: int
    forward_pass_batch_size: Optional[int] = None
    target: str = ""
    prompt_template: PromptTemplate = PromptTemplate()
    seq_clf: bool = False
    clf_target: Optional[int] = None
    random_seed: int = 0

    def __post_init__(self):
        self.forward_pass_batch_size = (
            self.forward_pass_batch_size or self.n_candidates_per_it
        )
        self.candidate_sample_rng = random.Random(self.random_seed)

        # TODO(GH#119): clean up if/elses for seq clf
        if self.seq_clf:
            assert self.clf_target is not None, "need clf target for seq clf"
            assert (
                self.clf_target < self.wrapped_model.model.num_labels
            ), "clf target out of range"
            # no string target for sequence classification, just an int (clf_target)
            assert self.target == "", "string target provided for seq task"
        else:
            assert self.target != "", "need non-empty target for causal lm"

        self.initial_attack_text, self.attack_indices = (
            self._get_initial_attack_text_and_indices(self.n_attack_tokens)
        )

    def run(self) -> str:
        """Runs the attack and returns the adversarial text."""
        attack_text = self.initial_attack_text
        candidate_texts = [attack_text]

        for _ in (pbar := tqdm(range(self.n_its))):
            candidate_texts_and_replacements = (
                self._get_candidate_texts_and_replacements(candidate_texts)
            )
            candidate_texts_and_replacements = self._filter_candidates(
                candidate_texts_and_replacements
            )
            evaluated_candidates = self._apply_replacements_and_eval_candidates(
                candidate_texts_and_replacements
            )
            candidate_texts = self._select_next_candidates(evaluated_candidates)
            attack_text = candidate_texts[0]

            # TODO(GH#112): track progress more cleanly
            pbar.set_description(f"Attack text: {attack_text}")

        return attack_text

    @abc.abstractmethod
    def _get_candidate_texts_and_replacements(
        self,
        candidate_texts: Sequence[str],
    ) -> list[Tuple[str, ReplacementCandidate]]:
        """Proposes a set of (attack_text, replacement) candidate pairs to consider."""
        pass

    def _select_next_candidates(self, candidates: list[Tuple[float, str]]) -> list[str]:
        """Selects text candidates for the next round, based on (score, text) pairs."""
        sorted_candidates = list(sorted(candidates, key=lambda x: x[0]))
        next_candidates = [
            text for _, text in sorted_candidates[: self.n_best_candidates_to_keep]
        ]
        return next_candidates

    @property
    @abc.abstractmethod
    def n_best_candidates_to_keep(self) -> int:
        pass

    def _get_initial_attack_text_and_indices(
        self, n_attack_tokens: int
    ) -> Tuple[str, AttackIndices]:
        """Initialize attack text with a sequence of "&@&@...&@".

        The length of the initial attack text will be `self.n_attack_tokens`.

        In case the initial attack text is not valid (because of retokenization issues),
        randomly perturb it until success.

        The initial attack text used in the reference is "! ! ! !" (and so on).
        I found that when using that with some tokenizers (e.g. gpt2),
        encoding, decoding, and encoding again was removing the spaces and
        then encoding the string into tokens containing multiple `!` together.
        I chose &@ because both characters are unlikely to merge with tokens
        before/after them.

        TODO(GH#117): investigate loss of spaces further.
        """
        optional_character = "&" if n_attack_tokens % 2 == 1 else ""
        attack_text = "&@" * (n_attack_tokens // 2) + optional_character
        attack_tokens = self.wrapped_model.get_tokens(attack_text)
        assert len(attack_tokens[0]) == n_attack_tokens

        try:
            attack_indices = self._get_attack_indices(attack_text)
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
                    0, self.wrapped_model.vocab_size - 1
                )
                attack_text = self.wrapped_model.decode_tokens(attack_tokens)
                try:
                    attack_indices = self._get_attack_indices(attack_text)
                    return attack_text, attack_indices
                except AttackTokenizationChangeException:
                    pass

        # We exceeded the maximum number of trials, so we raise an exception.
        raise AttackTokenizationChangeException

    def _get_tokens(
        self, inp: str | list[str], add_special: bool = False
    ) -> torch.Tensor:
        """Tokenize the inputs and return the token ids.

        Use tokenizer which is part of the wrapped model. Handle all the arguments we
        have to add to the tokenizer.
        """
        return self.wrapped_model.get_tokens(inp, add_special=add_special)

    def _decode_tokens(
        self,
        inp: torch.Tensor,
        skip_special_tokens: bool = True,
        try_squeeze: bool = True,
    ) -> str:
        string = self.wrapped_model.decode_tokens(
            inp,
            skip_special_tokens=skip_special_tokens,
            try_squeeze=try_squeeze,
        )
        return string

    def _get_attack_indices(self, attack_text: str) -> AttackIndices:
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
        before_attack_tokens = self._get_tokens(self.prompt_template.before_attack)
        attack_start = before_attack_tokens.shape[1]
        attack_end = self.n_attack_tokens + attack_start

        after_attack_tokens = self._get_tokens(self.prompt_template.after_attack)
        target_start = attack_end + after_attack_tokens.shape[1]
        target_tokens = self._get_tokens(self.target)
        target_end = target_start + target_tokens.shape[1]

        attack_indices = AttackIndices(
            attack_start=attack_start,
            attack_end=attack_end,
            target_start=target_start,
            target_end=target_end,
        )

        # check that the attack indices actually line up
        full_prompt = self.prompt_template.build_prompt(
            attack_text=attack_text,
            target=self.target,
        )
        full_tokens = self._get_tokens(full_prompt)
        attack_tokens = self._get_tokens(attack_text)

        attack_indices.assert_attack_and_target_tokens_validity(
            full_tokens, attack_tokens, target_tokens
        )

        return attack_indices

    @torch.no_grad()
    def _apply_replacements_and_eval_candidates(
        self,
        text_replacement_pairs: Sequence[Tuple[str, ReplacementCandidate]],
    ) -> list[tuple[float, str]]:
        """Evaluate the candidates using a forward pass through the model."""
        candidate_attack_texts = [
            self._decode_tokens(
                candidate.compute_tokens_after_replacement(
                    self._get_tokens(attack_text)
                )
            )
            for attack_text, candidate in text_replacement_pairs
        ]

        full_prompts = [
            self.prompt_template.build_prompt(
                attack_text=attack_text,
                target=self.target,
            )
            for attack_text in candidate_attack_texts
        ]
        full_prompts_tokens = self._get_tokens(full_prompts)
        if self.seq_clf:
            assert self.clf_target is not None
            targets = torch.full(
                size=(full_prompts_tokens.shape[0],),
                fill_value=self.clf_target,
                device=self.wrapped_model.device,
            )
        else:
            # NOTE: target is the same for each candidate
            targets = full_prompts_tokens[:, self.attack_indices.target_slice]

        all_logits_list = []
        for (inp,) in torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(full_prompts_tokens),
            batch_size=self.forward_pass_batch_size,
        ):
            all_logits_list.append(self.wrapped_model.call_model(inp))
        all_logits = torch.cat(all_logits_list, dim=0)

        assert len(all_logits) == len(text_replacement_pairs)

        evaluated_candidates = []
        candidate_losses = self._compute_loss_from_logits(all_logits, targets)
        for text, loss in zip(candidate_attack_texts, candidate_losses):
            evaluated_candidates.append((float(loss), text))

        return evaluated_candidates

    def _filter_candidates(
        self,
        text_replacement_pairs: Sequence[Tuple[str, ReplacementCandidate]],
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
            attack_text: The reference attack text on which we will be making
                replacements
            candidates: The candidates to filter

        Returns:
            A list of the candidates that didn't result in a change in tokenization
                and that changed the attack text
        """
        filtered_candidates = []
        for attack_text, candidate in text_replacement_pairs:
            attack_tokens = self._get_tokens(attack_text)
            full_prompt = self.prompt_template.build_prompt(
                attack_text=attack_text,
                target=self.target,
            )
            original_full_prompt_tokens = self._get_tokens(full_prompt)

            candidate_attack_tokens = candidate.compute_tokens_after_replacement(
                attack_tokens
            )

            candidate_full_prompt_tokens = torch.cat(
                [
                    original_full_prompt_tokens[:, : self.attack_indices.attack_start],
                    candidate_attack_tokens,
                    original_full_prompt_tokens[:, self.attack_indices.attack_end :],
                ],
                dim=1,
            )

            if torch.equal(candidate_full_prompt_tokens, original_full_prompt_tokens):
                continue

            if self._tokenization_changed(
                candidate_full_prompt_tokens, original_full_prompt_tokens
            ):
                continue

            filtered_candidates.append((attack_text, candidate))

        logger.debug(
            f"Filtered from {len(text_replacement_pairs)} to {len(filtered_candidates)}"
        )
        return filtered_candidates

    def _tokenization_changed(
        self,
        candidate_full_prompt_tokens: torch.Tensor,
        original_full_prompt_tokens: torch.Tensor,
    ) -> bool:
        """Returns True if the tokenization changed after the replacement.

        We only take candidates if the tokenization doesn't change (i.e. False).
        To check that the tokenization doesn't importantly change, we ensure that the
        length of the whole token sequence is the same after decoding and encoding, and
        that each individual section (attack, target) is the same before and after.
        """

        decoded = self._decode_tokens(candidate_full_prompt_tokens)
        encoded_decoded = self._get_tokens(decoded)

        # check if length changes when decoding and encoding, since
        # that would imply that two tokens merged
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

    def _compute_loss(
        self, combined_embeddings: torch.Tensor, full_prompt_tokens: torch.Tensor
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
                self.clf_target, device=self.wrapped_model.device
            ).unsqueeze(0)
        else:
            targets = full_prompt_tokens[:, self.attack_indices.target_slice]
        loss = self._compute_loss_from_logits(logits, targets)
        assert loss.numel() == 1
        return loss.mean()

    def _compute_loss_from_logits(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # only take a slice if it's a causal LM
        if self.seq_clf:
            target_logits = logits
        else:
            target_logits = logits[:, self.attack_indices.loss_slice, :]
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
        assert combined_embeddings.shape == full_prompt_embeddings.shape
        return combined_embeddings

    def _get_attack_onehot(self, attack_tokens: torch.Tensor) -> torch.Tensor:
        """Creates a one-hot encoding layer before the model embeddings.

        We can then backpropagate through it.

        TODO: I think we're neglecting position embeddings here (and they do neglect
        them in the reference impl afaict)
        """
        embedding_weights = self.wrapped_model.get_embedding_weights()
        vocab_size = embedding_weights.shape[0]
        dtype = embedding_weights.dtype
        attack_onehot = create_onehot_embedding(
            token_ids=attack_tokens,
            vocab_size=vocab_size,
            dtype=dtype,
            device=self.wrapped_model.model.device,
        )
        return attack_onehot
