import abc
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.utils.data

from robust_llm.attacks.search_based.models import SearchBasedAttackWrappedModel
from robust_llm.attacks.search_based.utils import (
    AttackIndices,
    AttackTokenizationChangeException,
    ExampleWithAttackIndices,
    PreppedExample,
    ReplacementCandidate,
    create_onehot_embedding,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiPromptSearchBasedRunner(abc.ABC):
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
        prepped_examples: A list of prepped examples, each containing a
            PromptTemplate and clf_target
        seq_clf: Whether we are using a SequenceClassification model
            (default alternative is a CausalLM)
        random_seed: initial seed for a random.Random object used to sample
            replacement candidates
    """

    wrapped_model: SearchBasedAttackWrappedModel
    n_candidates_per_it: int
    n_its: int
    n_attack_tokens: int
    prepped_examples: Sequence[PreppedExample]
    forward_pass_batch_size: Optional[int] = None
    target: str = ""
    seq_clf: bool = False
    random_seed: int = 0

    def __post_init__(self):
        self.forward_pass_batch_size = (
            self.forward_pass_batch_size or self.n_candidates_per_it
        )
        self.candidate_sample_rng = random.Random(self.random_seed)

        # TODO(GH#119): clean up if/elses for seq clf
        if self.seq_clf:
            assert all(pe.clf_target is not None for pe in self.prepped_examples)
            assert all(
                pe.clf_target < self.wrapped_model.model.num_labels
                for pe in self.prepped_examples
            )
            # no string target for sequence classification, just an int (clf_target)
            assert self.target == "", "string target provided for seq task"
        else:
            assert self.target != "", "need non-empty target for causal lm"

        self.initial_attack_text, self.examples_with_attack_ix = (
            self._get_initial_attack_text_and_indices(
                self.n_attack_tokens, self.prepped_examples
            )
        )

    def run(self) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError("Not implemented in general for multi-prompt search")

    def _get_candidate_texts_and_replacements(
        self,
        candidate_texts: Sequence[str],
        considered_examples: Sequence[ExampleWithAttackIndices],
    ) -> list[Tuple[str, ReplacementCandidate]]:
        raise NotImplementedError("Not implemented in general for multi-prompt search")

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
        self, n_attack_tokens: int, prepped_examples: Sequence[PreppedExample]
    ) -> Tuple[str, list[ExampleWithAttackIndices]]:
        """Initialize attack text with a sequence of "&@&@...&@".

        NOTE: this (multi-prompt) version has a different arguments than the
        single-prompt version since we need multiple AttackIndices and it's
        easier to keep them with the examples.

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
            attack_indices = [
                self._get_attack_indices(attack_text, pe) for pe in prepped_examples
            ]
            examples_with_attack_ix = [
                pe.add_attack_indices(ai)
                for pe, ai in zip(prepped_examples, attack_indices)
            ]
            return attack_text, examples_with_attack_ix
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
                    attack_indices = [
                        self._get_attack_indices(attack_text, pe)
                        for pe in prepped_examples
                    ]
                    examples_with_attack_ix = [
                        pe.add_attack_indices(ai)
                        for pe, ai in zip(prepped_examples, attack_indices)
                    ]
                    return attack_text, examples_with_attack_ix
                except AttackTokenizationChangeException:
                    pass

        # We exceeded the maximum number of trials, so we raise an exception.
        raise AttackTokenizationChangeException

    def _get_tokens(
        self,
        inputs: str | list[str],
        return_tensors: Optional[str] = "pt",
        add_special: bool = False,
    ) -> torch.Tensor:  # TODO (ian): fix return type when return_tensors is None
        """Tokenize the inputs and return the token ids.

        Use tokenizer which is part of the wrapped model. Handle all the arguments we
        have to add to the tokenizer.
        """
        return self.wrapped_model.get_tokens(
            inputs, return_tensors=return_tensors, add_special=add_special
        )

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
        before_attack_tokens = self._get_tokens(example.prompt_template.before_attack)
        attack_start = before_attack_tokens.shape[1]
        attack_end = self.n_attack_tokens + attack_start

        after_attack_tokens = self._get_tokens(example.prompt_template.after_attack)
        target_start = attack_end + after_attack_tokens.shape[1]
        # TODO (ian): include self.target in PreppedExample (or remove altogether)
        # this will work fine for now because we are doing seq_clf
        target_tokens = self._get_tokens(self.target)
        target_end = target_start + target_tokens.shape[1]

        attack_indices = AttackIndices(
            attack_start=attack_start,
            attack_end=attack_end,
            target_start=target_start,
            target_end=target_end,
        )

        # check that the attack indices actually line up
        full_prompt = example.prompt_template.build_prompt(
            attack_text=attack_text,
            target=self.target,
        )
        full_tokens = self._get_tokens(full_prompt)
        attack_tokens = self._get_tokens(attack_text)

        attack_indices.assert_attack_and_target_tokens_validity(
            full_tokens, attack_tokens, target_tokens
        )

        return attack_indices

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
