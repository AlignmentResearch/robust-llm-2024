import torch
from typing_extensions import override

from robust_llm.attacks.attack import (
    Attack,
    AttackedExample,
    AttackedRawInputOutput,
    AttackOutput,
    AttackState,
)
from robust_llm.attacks.search_based.utils import get_chunking_for_search_based
from robust_llm.config.attack_configs import BeastAttackConfig
from robust_llm.config.configs import ExperimentConfig
from robust_llm.dist_utils import DistributedRNG
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.prompt_templates import PromptTemplate
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.scoring_callbacks import build_tensor_scoring_callback
from robust_llm.scoring_callbacks.scoring_callback_utils import (
    CallbackInput,
    TensorCallbackOutput,
)


class BeastAttack(Attack):
    """Implementation of BEAST (https://arxiv.org/pdf/2402.15570).

    BEAST is a beam-search attack that does not require gradients. It
    generates an adversarial substring as follows:
    - Maintain a beam of beam_search_width candidates (k_1 in the paper). Each
      initial candidate is a token sampled from the victim.
    - For each candidate C, create beam_branch_factor new candidates of the form
      "<C><token sampled from LM>".
    - The new beam is the top beam_search_width candidates among the new
      candidates.
    - Repeat for some number of iterations.
    - Return the best candidate seen throughout the entire process.
    See the paper for a more rigorous algorithm description.

    Since the victim model may be a classifier, we sample from a separate
    "sampling model" rather than the victim model.

    Despite BEAST conceptually being a search-based attack, we don't subclass
    SearchBasedAttack. SearchBasedAttack is designed for attacks that start by
    inserting perturbation_length arbitrary tokens and modifying them, whereas
    BEAST starts with an empty perturbation and grows it. Subclassing
    SearchBasedAttack would require a refactor to SearchBasedAttack.
    """

    CAN_CHECKPOINT = True

    def __init__(
        self,
        exp_config: ExperimentConfig,
        victim: WrappedModel,
        is_training: bool,
    ) -> None:
        super().__init__(exp_config, victim=victim, is_training=is_training)
        assert isinstance(self.attack_config, BeastAttackConfig)

        if self.attack_config.sampling_model is None:
            assert (
                victim.inference_type == InferenceType.GENERATION
            ), "Sampling model must be provided for classification models."
            self.sampling_model = victim
        else:
            assert self.attack_config.sampling_model.family == victim.family, (
                "Sampling model needs to be of the same family as victim model,"
                " otherwise their token spaces may not match."
            )
            self.sampling_model = WrappedModel.from_config(
                self.attack_config.sampling_model, accelerator=victim.accelerator
            )

        self.scores_from_text_callback = build_tensor_scoring_callback(
            self.attack_config.scores_from_text_callback
        )

    def _sample_tokens(
        self,
        prompts: torch.Tensor,
        num_samples: int,
        rng: DistributedRNG,
    ) -> torch.Tensor:
        """Samples next tokens for each prompt in `prompts`.

        Args:
            prompts: A tensor of shape (num_prompts, prompt_len).
            num_samples: Number of next tokens to sample without replacement
                from the next-token distribution induced by each prompt.
            rng: Distributed random number generator.

        Returns:
            A tensor of shape (num_prompts, num_samples) containing sampled
            tokens for each prompt.
        """
        logits_generator = self.sampling_model.generation_logits_from_tokens(
            prompts, attention_mask=torch.ones_like(prompts)
        )
        # shape: (num_prompts, prompt_length, vocab_size)
        logits = torch.cat(list(logits_generator), dim=0)
        # We just want logits for the last token.
        logits = logits[:, -1]
        _, num_token_ids = logits.shape

        return torch.tensor(
            [
                rng.choice(
                    seq=num_token_ids,
                    size=num_samples,
                    replace=False,
                    p=prompt_logits.softmax(dim=-1).cpu().numpy(),
                )
                for prompt_logits in logits
            ],
            dtype=torch.int,
            device=self.victim.device,
        )

    def _get_candidate_scores(
        self,
        candidates: torch.Tensor,
        example: dict,
        prompt_template: PromptTemplate,
    ) -> TensorCallbackOutput:
        """Run the scoring callback relative to the proxy target on each candidate.

        Args:
            candidates: Candidate perturbations to score, as a tensor of shape
                (num_candidates, perturbation_length).
            example: Original example to attack.
            prompt_template: Original example as a prompt template.

        Returns:
            Callback output for each candidate.
        """
        candidate_texts = [
            prompt_template.build_prompt(attack_text=candidate_text)
            for candidate_text in self.victim.batch_decode(candidates)
        ]
        num_candidates = len(candidates)
        callback_input = CallbackInput(
            input_data=candidate_texts,
            original_input_data=[example["text"]] * num_candidates,
            clf_label_data=[example["proxy_clf_label"]] * num_candidates,
            gen_target_data=[example["proxy_gen_target"]] * num_candidates,
        )
        return self.scores_from_text_callback(self.victim, callback_input)

    def _generate_new_candidates(
        self,
        beam: torch.Tensor,
        before_attack_tokens: torch.Tensor,
        rng: DistributedRNG,
    ):
        """From the current beam, generate candidates for next iteration.

        Returns:
            A tensor of shape (beam_search_width * beam_branch_factor,
            beam.shape[-1] + 1) containing candidates for the beam of next
            iteration.
        """
        assert isinstance(self.attack_config, BeastAttackConfig)
        beam_search_width = self.attack_config.beam_search_width
        beam_branch_factor = self.attack_config.beam_branch_factor

        next_tokens = self._sample_tokens(
            prompts=torch.cat(
                [before_attack_tokens.expand(beam_search_width, -1), beam], dim=-1
            ),
            num_samples=beam_branch_factor,
            rng=rng,
        )
        num_new_candidates = beam_search_width * beam_branch_factor
        perturbation_tokens_so_far = beam.shape[-1]
        # Same as beam.repeat_interleave(repeats=beam_branch_factor, dim=0) but
        # avoids copying the data.
        repeated_beam = (
            beam.unsqueeze(1)
            .expand(-1, beam_branch_factor, -1)
            .reshape(num_new_candidates, perturbation_tokens_so_far)
        )
        new_candidates = torch.cat(
            [repeated_beam, next_tokens.reshape(num_new_candidates, 1)], dim=-1
        )
        return new_candidates

    @override
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
        n_its: int = 1,
        resume_from_checkpoint: bool = True,
        epoch: int | None = None,
    ) -> AttackOutput:
        accelerator = self.victim.accelerator
        assert accelerator is not None
        assert (
            dataset.modifiable_chunk_spec.n_modifiable_chunks == 1
        ), "Exactly one modifiable chunk"

        start_attack_state = self.get_first_attack_state(resume_from_checkpoint, epoch)
        checkpoint_path = self.get_attack_checkpoint_path(epoch)
        end_attack_state = self.run_attack_loop(
            dataset=dataset,
            attack_state=start_attack_state,
            checkpoint_path=checkpoint_path if resume_from_checkpoint else None,
            n_its=n_its,
        )

        attacked_input_texts = end_attack_state.get_attacked_input_texts()
        logits_cache = end_attack_state.get_logits_cache()
        all_iteration_texts = end_attack_state.get_all_iteration_texts()
        attack_flops = end_attack_state.get_attack_flops()
        attacked_dataset = dataset.with_attacked_text(attacked_input_texts)
        attack_out = AttackOutput(
            dataset=attacked_dataset,
            attack_data=AttackedRawInputOutput(
                iteration_texts=all_iteration_texts, logits=logits_cache
            ),
            flops=attack_flops,
        )
        return attack_out

    def _run_attack_on_example(
        self, dataset: RLLMDataset, n_its: int, attack_state: AttackState
    ) -> AttackedExample:
        """run_attack_on_example() but returns AttackedExample with an empty flop count.

        This should be wrapped in a flop count context to get the flop count.
        """
        assert isinstance(self.attack_config, BeastAttackConfig)
        example_index = attack_state.example_index
        example = dataset.ds[example_index]
        assert isinstance(example, dict)

        attack_chunks = get_chunking_for_search_based(
            example["chunked_text"], dataset.modifiable_chunk_spec
        )

        prompt_template = self.victim.chunks_to_prompt_template(
            chunks=attack_chunks,
            perturb_min=self.attack_config.perturb_position_min,
            perturb_max=self.attack_config.perturb_position_max,
            rng=attack_state.rng_state.distributed_rng,
        )
        before_attack_tokens = self.victim.get_tokens(
            prompt_template.before_attack,
            return_tensors="pt",
        )

        # Each candidate in the beam is stored as a list of tokens of the
        # perturbation.
        # Shape: (beam_search_width, num_perturbation_tokens_so_far)
        beam = self._sample_tokens(
            prompts=before_attack_tokens,
            num_samples=self.attack_config.beam_branch_factor,
            rng=attack_state.rng_state.distributed_rng,
        ).T
        scores = self._get_candidate_scores(
            candidates=beam,
            example=example,
            prompt_template=prompt_template,
        )
        best_loss, best_candidate_index = torch.min(scores.losses, dim=0)
        best_candidate = beam[best_candidate_index]
        best_logits = scores.info["logits"][best_candidate_index]
        # (Best candidate tokens, best candidate logits) per iteration
        best_candidate_history = [(best_candidate, best_logits)]

        for tokens_so_far in range(1, n_its):
            new_candidates = self._generate_new_candidates(
                beam=beam,
                before_attack_tokens=before_attack_tokens,
                rng=attack_state.rng_state.distributed_rng,
            )

            # Prune new_candidates back down to the best beam_search_width candidates.
            scores = self._get_candidate_scores(
                candidates=new_candidates,
                example=example,
                prompt_template=prompt_template,
            )
            best_losses = torch.topk(
                scores.losses,
                k=self.attack_config.beam_search_width,
                largest=False,
                sorted=True,
            )
            beam = new_candidates[best_losses.indices]
            if best_losses.values[0] < best_loss:
                best_loss = best_losses.values[0]
                best_candidate = beam[0]
                best_logits = scores.info["logits"][best_losses.indices[0]]
            best_candidate_history.append((best_candidate, best_logits))

        final_attack_string = self.victim.decode(best_candidate)
        attacked_input_text = prompt_template.build_prompt(
            attack_text=final_attack_string,
        )

        best_candidate_history_texts = self.victim.batch_decode(
            [tokens for tokens, _ in best_candidate_history]
        )
        iteration_texts = [
            prompt_template.build_prompt(attack_text=text)
            for text in best_candidate_history_texts
        ]
        iteration_logits = [logits for _, logits in best_candidate_history]

        new_attacked_example = AttackedExample(
            example_index=example_index,
            attacked_text=attacked_input_text,
            iteration_texts=iteration_texts,
            logits=iteration_logits,
            flops=0,
        )
        return new_attacked_example

    @override
    def run_attack_on_example(
        self, dataset: RLLMDataset, n_its: int, attack_state: AttackState
    ) -> AttackState:
        with (
            self.victim.flop_count_context() as victim_flop_count,
            self.sampling_model.flop_count_context() as sampling_model_flop_count,
        ):
            attacked_example = self._run_attack_on_example(
                dataset=dataset, n_its=n_its, attack_state=attack_state
            )
        flops = victim_flop_count.flops
        if self.victim != self.sampling_model:
            flops += sampling_model_flop_count.flops

        attacked_example = AttackedExample(
            example_index=attacked_example.example_index,
            attacked_text=attacked_example.attacked_text,
            iteration_texts=attacked_example.iteration_texts,
            logits=attacked_example.logits,
            flops=flops,
        )
        new_previous_examples = attack_state.previously_attacked_examples + (
            attacked_example,
        )
        new_attack_state = AttackState(
            previously_attacked_examples=new_previous_examples,
            rng_state=attack_state.rng_state.update_states(),
        )
        return new_attack_state
