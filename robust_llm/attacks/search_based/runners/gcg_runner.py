from collections.abc import Sequence

import torch
import torch.utils.data
from typing_extensions import override

from robust_llm.attacks.search_based.runners.search_based_runner import (
    SearchBasedRunner,
)
from robust_llm.attacks.search_based.utils import PreppedExample, ReplacementCandidate
from robust_llm.config.callback_configs import CallbackConfig
from robust_llm.models.prompt_templates import PromptTemplate
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.scoring_callbacks import CallbackInput, build_tensor_scoring_callback


class GCGRunner(SearchBasedRunner):
    """Runs GCG on a single model and a single prompt/target pair.

    Attributes:
        n_best_candidates_to_keep: how many candidates to keep after each iteration.
        differentiable_embeds_callback: the callback for computing the loss to
            backprop through.
        top_k: the number of token replacements to consider at each
            position in the attack tokens.
    """

    def __init__(
        self,
        victim: WrappedModel,
        n_candidates_per_it: int,
        n_its: int,
        n_attack_tokens: int,
        scores_from_text_callback: CallbackConfig,
        prepped_examples: Sequence[PreppedExample],
        differentiable_embeds_callback: CallbackConfig,
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

        cb = build_tensor_scoring_callback(differentiable_embeds_callback)
        self.differentiable_embeds_callback = cb
        self.top_k = top_k

    @override
    def _get_candidate_texts_and_replacements(
        self,
        candidate_texts: Sequence[str],
    ) -> list[tuple[str, ReplacementCandidate]]:
        # In GCG, the list of candidate texts should contain exactly the single best
        # candidate from the previous iteration
        assert len(candidate_texts) == 1
        attack_text = candidate_texts[0]

        gradients = self._compute_gradients(attack_text)
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
        attack_text: str,
    ) -> torch.Tensor:

        tokens, combined_embeddings, attack_onehot = self._build_tokens_and_embeddings(
            prompt_template=self.example.prompt_template,
            attack_text=attack_text,
        )

        input_data = {"input_ids": tokens, "embeddings": combined_embeddings}

        cb_in = CallbackInput(
            input_data=input_data,
            clf_label_data=[self.example.clf_label],
            gen_target_data=[self.example.gen_target],
        )
        cb_out = self.differentiable_embeds_callback(self.victim, cb_in)
        losses = cb_out.losses

        assert len(losses) == 1, "Batch size should be 1."
        loss = losses[0]

        if self.victim.accelerator is None:
            raise ValueError("An accelerator must be added to the model.")

        self.victim.accelerator.backward(loss)
        assert attack_onehot.grad is not None
        return attack_onehot.grad.clone()

    def _build_tokens_and_embeddings(
        self, prompt_template: PromptTemplate, attack_text: str
    ):
        # NOTE: We don't add the target to the prompt here; that'll be handled
        # in the callback.
        full_prompt = prompt_template.build_prompt(
            attack_text=attack_text,
            target="",
        )

        full_prompt_tokens = self.victim.get_tokens(full_prompt, return_tensors="pt")
        full_prompt_embeddings = self.victim.get_embeddings(full_prompt_tokens)

        attack_tokens = self.victim.get_tokens(attack_text, return_tensors="pt")
        attack_onehot = self._get_attack_onehot(attack_tokens.to(self.victim.device))
        embed_weights = self.victim.get_embedding_weights()

        attack_onehot.requires_grad_()
        attack_embeddings = attack_onehot @ embed_weights

        try:
            combined_embeddings = self._get_combined_embeddings(
                full_prompt_embeddings, attack_embeddings
            )
        except ValueError as e:
            raise ValueError(
                f"Error combining embeddings for attack text '{attack_text}', "
                f"prompt template '{prompt_template}', and "
                f"full prompt '{full_prompt}'. "
                f"{e}"
            )
        return full_prompt_tokens, combined_embeddings, attack_onehot

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
