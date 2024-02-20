import logging
import random
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.utils.data
import transformers
from datasets import Dataset
from tqdm import tqdm
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.attacks.gcg.models import GCGWrappedModel
from robust_llm.attacks.gcg.utils import (
    AttackIndices,
    AttackTokenizationChangeException,
    PromptTemplate,
    ReplacementCandidate,
    create_onehot_embedding,
    get_gcg_chunking,
    get_wrapped_model,
)
from robust_llm.configs import AttackConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec
from robust_llm.utils import LanguageModel, get_randint_with_exclusions

logger = logging.getLogger(__name__)


class GCGAttack(Attack):
    """Implementation of the Greedy Coordinate Gradient attack.

    For now, we allow only one modifiable chunk, so the inputs are of the form
    <unmodifiable_prefix><modifiable_infix><unmodifiable_suffix>. The attack will either
    completely replace modifiable infix with optimized tokens (if
    `attack_config.gcg_attack_config.wipe_out_modifiable_chunk` is True) or, otherwise,
    will add the tokens after the modifiable infix.
    """

    REQUIRES_INPUT_DATASET = True
    REQUIRES_TRAINING = False

    def __init__(
        self,
        attack_config: AttackConfig,
        modifiable_chunks_spec: ModifiableChunksSpec,
        model: LanguageModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        ground_truth_label_fn: Optional[Callable[[str], int]],
    ) -> None:
        super().__init__(attack_config, modifiable_chunks_spec)

        assert isinstance(
            model, transformers.PreTrainedModel
        ), "DefendedModel is not supported"

        assert sum(modifiable_chunks_spec) == 1

        self.model = model
        self.tokenizer = tokenizer
        self.wrapped_model = get_wrapped_model(self.model, self.tokenizer)
        self.ground_truth_label_fn = ground_truth_label_fn

    @override
    def get_attacked_dataset(
        self,
        dataset: Optional[Dataset],
        max_n_outputs: Optional[int] = None,
    ) -> Tuple[Dataset, Dict[str, Any]]:
        """Run a GCG attack separately on each example in the dataset.

        TODO(GH#113): consider multi-model attacks in the future.
        TODO(GH#114): consider multi-prompt attacks in the future.
        """
        # preconditions
        assert dataset is not None, "GCGAttack requires dataset input"
        assert max_n_outputs is None, "GCGAttack does not support max_n_outputs"

        options = self.attack_config.gcg_attack_config

        attacked_input_texts = []
        for example in dataset:
            assert isinstance(example, dict)

            unmodifiable_prefix, modifiable_infix, unmodifiable_suffix = (
                get_gcg_chunking(example["text_chunked"], self.modifiable_chunks_spec)
            )
            if options.wipe_out_modifiable_chunk:
                modifiable_infix = ""

            prompt_template = PromptTemplate(
                before_attack=unmodifiable_prefix + modifiable_infix,
                after_attack=unmodifiable_suffix,
            )

            if self.ground_truth_label_fn is not None:
                # Update GT label after possible wiping out of the modifiable chunk.
                true_label = self.ground_truth_label_fn(prompt_template.build_prompt())
            else:
                true_label = example["label"]

            # TODO(GH#106): make it work with multi-class classification
            num_possible_classes = 2
            target_label = get_randint_with_exclusions(
                high=num_possible_classes, exclusions=[true_label]
            )

            runner = GCGRunner(
                wrapped_model=self.wrapped_model,
                prompt_template=prompt_template,
                clf_target=target_label,
                top_k=options.top_k,
                n_candidates_per_it=options.n_candidates_per_it,
                n_its=options.n_its,
                n_attack_tokens=options.n_attack_tokens,
                seq_clf=options.seq_clf,
                forward_pass_batch_size=options.forward_pass_batch_size,
                random_seed=options.random_seed,
            )
            attack_text = runner.run_gcg()
            attacked_input_text = prompt_template.build_prompt(
                attack_text=attack_text,
            )
            attacked_input_texts.append(attacked_input_text)

        return (
            Dataset.from_dict(
                {
                    "text": attacked_input_texts,
                    "original_text": dataset["text"],
                    "label": dataset["label"],
                }
            ),
            {},
        )


class GCGRunner:
    """Run GCG on a single model and a single prompt/target pair"""

    def __init__(
        self,
        wrapped_model: GCGWrappedModel,
        top_k: int,
        n_candidates_per_it: int,
        n_its: int,
        n_attack_tokens: int,
        target: str = "",
        prompt_template: PromptTemplate = PromptTemplate(),
        seq_clf: bool = False,
        clf_target: Optional[int] = None,
        forward_pass_batch_size: Optional[int] = None,
        random_seed: int = 42,
    ) -> None:
        """Initializes the GCGRunner on a single model & prompt/target pair.

        Args:
            wrapped_model: The model to attack paired with a tokenizer
                and some model-specific methods.
            top_k: The number of token replacements to consider at each position
            n_candidates_per_it: The total number of token replacements
                to consider in each iteration of GCG (this must be less than
                top_k * n_attack_tokens, which is the total number of candidates).
            n_its: Total number of iterations of GCG to run.
            n_attack_tokens: number of attack tokens to optimize
            target: If using a CausalLM, it's the target string to optimize for.
                If using a SequenceClassification model, it's ignored in favor of
                the target specified by clf_target.
            prompt_template: The PromptTemplate defines the format of the prompt,
                and its `build_prompt` method is used for producing full prompts.
            seq_clf: Whether we are using a SequenceClassification model
                (default alternative is a CausalLM)
            clf_target: Only used for sequence classification, specifies the class
                to optimize for.
            forward_pass_batch_size: batch size used for forward pass when evaluating
                candidates. If None, defaults to n_candidates_per_it.
            random_seed: initial seed for a random.Random object used to sample
                replacement candidates
        """
        self.wrapped_model = wrapped_model
        self.top_k = top_k
        self.n_candidates_per_it = n_candidates_per_it
        self.n_its = n_its
        self.target = target
        self.n_attack_tokens = n_attack_tokens
        self.prompt_template = prompt_template
        self.seq_clf = seq_clf
        self.clf_target = clf_target
        self.forward_pass_batch_size = forward_pass_batch_size or n_candidates_per_it
        self.initial_seed = random_seed
        self.candidate_sample_rng = random.Random(self.initial_seed)

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
            attack_indices = self.get_attack_indices(attack_text)
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
                    attack_indices = self.get_attack_indices(attack_text)
                    return attack_text, attack_indices
                except AttackTokenizationChangeException:
                    pass

        # We exceeded the maximum number of trials, so we raise an exception.
        raise AttackTokenizationChangeException

    def run_gcg(self) -> str:
        """Run the GCG attack and return the adversarial text"""
        attack_text = self.initial_attack_text
        for it in (pbar := tqdm(range(self.n_its))):
            gradients = self.compute_gradients(self.target, attack_text)
            candidates = self._candidates_from_gradients(gradients)
            filtered_candidates = self._filter_candidates(attack_text, candidates)
            evaluated_candidates = self._evaluate_candidates(
                attack_text, filtered_candidates
            )
            attack_text = self.update_attack_text(attack_text, evaluated_candidates)

            # TODO(GH#112): track progress more cleanly
            pbar.set_description(f"Attack text: {attack_text}")
        return attack_text

    def get_tokens(
        self, inp: str | list[str], add_special: bool = False
    ) -> torch.Tensor:
        """Handle all the arguments we have to add to the tokenizer"""
        return self.wrapped_model.get_tokens(inp, add_special=add_special)

    def decode_tokens(
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

    def update_attack_text(
        self,
        attack_text: str,
        evaluated_candidates: Sequence[tuple[float, ReplacementCandidate]],
    ) -> str:
        """Pick the best candidate from the pool"""
        _score, best_candidate = min(evaluated_candidates, key=lambda x: x[0])
        attack_tokens = self.get_tokens(attack_text)
        new_attack_tokens = best_candidate.compute_tokens_after_replacement(
            attack_tokens
        )
        new_attack_text = self.decode_tokens(new_attack_tokens)
        return new_attack_text

    def get_attack_indices(self, attack_text: str) -> AttackIndices:
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
        before_attack_tokens = self.get_tokens(self.prompt_template.before_attack)
        attack_start = before_attack_tokens.shape[1]
        attack_end = self.n_attack_tokens + attack_start

        after_attack_tokens = self.get_tokens(self.prompt_template.after_attack)
        target_start = attack_end + after_attack_tokens.shape[1]
        target_tokens = self.get_tokens(self.target)
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
        full_tokens = self.get_tokens(full_prompt)
        attack_tokens = self.get_tokens(attack_text)

        attack_indices.assert_attack_and_target_tokens_validity(
            full_tokens, attack_tokens, target_tokens
        )

        return attack_indices

    @torch.no_grad()
    def _evaluate_candidates(
        self,
        attack_text: str,
        candidates: Sequence[ReplacementCandidate],
    ) -> list[tuple[float, ReplacementCandidate]]:
        """Evaluate the candidates exactly using a forward pass through the model."""
        attack_tokens = self.get_tokens(attack_text)
        candidate_attack_texts = [
            self.decode_tokens(
                candidate.compute_tokens_after_replacement(attack_tokens)
            )
            for candidate in candidates
        ]
        full_prompts = [
            self.prompt_template.build_prompt(
                attack_text=attack_text,
                target=self.target,
            )
            for attack_text in candidate_attack_texts
        ]
        full_prompts_tokens = self.get_tokens(full_prompts)
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

        evaluated_candidates = []
        candidate_losses = self._compute_loss_from_logits(all_logits, targets)
        for candidate, candidate_loss in zip(candidates, candidate_losses):
            evaluated_candidates.append((float(candidate_loss), candidate))

        return evaluated_candidates

    def _filter_candidates(
        self,
        attack_text: str,
        candidates: Sequence[ReplacementCandidate],
    ) -> list[ReplacementCandidate]:
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
        attack_tokens = self.get_tokens(attack_text)
        full_prompt = self.prompt_template.build_prompt(
            attack_text=attack_text,
            target=self.target,
        )
        original_full_prompt_tokens = self.get_tokens(full_prompt)
        filtered_candidates = []
        for candidate in candidates:
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

            filtered_candidates.append(candidate)

        logger.debug(f"Filtered from {len(candidates)} to {len(filtered_candidates)}")
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

        decoded = self.decode_tokens(candidate_full_prompt_tokens)
        encoded_decoded = self.get_tokens(decoded)

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

    def _candidates_from_gradients(
        self,
        gradients: torch.Tensor,
    ) -> list[ReplacementCandidate]:
        """Gets the top k candidates from the gradients.

        `gradients` is a tensor of shape (n_attack_tokens, vocab_size).

        We take the top k from each position, and then randomly sample a
        batch of self.n_candidates_per_it < (top k * n_attack_tokens) replacements
        from the resulting pool.
        """
        # we disallow sep and cls tokens
        cls_token_id = self.wrapped_model.cls_token_id
        sep_token_id = self.wrapped_model.sep_token_id
        if cls_token_id is not None:
            gradients[:, cls_token_id] = float("inf")
        if sep_token_id is not None:
            gradients[:, sep_token_id] = float("inf")

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

    def compute_gradients(
        self,
        target: str,
        attack_text: str,
    ) -> torch.Tensor:
        full_prompt = self.prompt_template.build_prompt(
            attack_text=attack_text,
            target=target,
        )

        # only need specials if we're doing BERT
        full_prompt_tokens = self.get_tokens(full_prompt, add_special=self.seq_clf)
        full_prompt_embeddings = self.wrapped_model.get_embeddings(
            full_prompt_tokens
        ).detach()

        attack_tokens = self.get_tokens(attack_text)
        attack_onehot = self._get_attack_onehot(attack_tokens)
        attack_embeddings = attack_onehot @ self.wrapped_model.get_embedding_weights()

        combined_embeddings = self._get_combined_embeddings(
            full_prompt_embeddings, attack_embeddings
        )

        # doesn't matter what shape full_prompt_tokens is for seq clf
        loss = self._compute_loss(combined_embeddings, full_prompt_tokens)
        loss.backward()
        assert attack_onehot.grad is not None
        return attack_onehot.grad.clone()

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
