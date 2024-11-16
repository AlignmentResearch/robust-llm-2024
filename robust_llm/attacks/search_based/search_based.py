from typing_extensions import override

from robust_llm.attacks.attack import (
    Attack,
    AttackedExample,
    AttackedRawInputOutput,
    AttackOutput,
    AttackState,
)
from robust_llm.attacks.search_based.runners import make_runner
from robust_llm.attacks.search_based.utils import (
    PreppedExample,
    get_chunking_for_search_based,
)
from robust_llm.config.attack_configs import SearchBasedAttackConfig
from robust_llm.config.configs import ExperimentConfig
from robust_llm.models.caching_wrapped_model import get_caching_model_with_example
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


class SearchBasedAttack(Attack):
    """Implementation of the search-based attacks.

    These are attacks that maintain a list of candidates and iteratively refine them by
    looking at the nearby candidates. Concrete approaches include GCG and beam search.
    Actual implementations of the logic of iteration, search, filtering, etc. is in
    the `runners` submodule, with the `SearchBasedRunner` abstract class as the base.

    For now, we allow only one modifiable chunk, so the inputs are of the form
    <unmodifiable_prefix><modifiable_infix><unmodifiable_suffix>. The attack
    will either completely replace modifiable infix with optimized tokens (if
    the chunk is OVERWRITABLE) or, otherwise if the chunk is PERTURBABLE, will
    add the tokens after the modifiable infix.
    """

    CAN_CHECKPOINT = True

    def __init__(
        self,
        exp_config: ExperimentConfig,
        victim: WrappedModel,
        is_training: bool,
    ) -> None:
        super().__init__(exp_config, victim=victim, is_training=is_training)

    @override
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
        n_its: int,
        resume_from_checkpoint: bool = True,
        epoch: int | None = None,
    ) -> AttackOutput:
        """Run a search-based attack separately on each example in the dataset.

        TODO(GH#113): consider multi-model attacks in the future.
        TODO(GH#114): consider multi-prompt attacks in the future.
        """
        accelerator = self.victim.accelerator
        # preconditions
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

        # We clear the gradients here to avoid using up GPU memory
        # even after the attack has stopped using it.
        # (This is primarily for GCG.)
        self.victim.model.zero_grad()

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

    def run_attack_on_example(
        self,
        dataset: RLLMDataset,
        n_its: int,
        attack_state: AttackState,
    ) -> AttackState:
        """Run the attack on a single example, updating the attack state."""
        example_index = attack_state.example_index
        example = dataset.ds[example_index]
        assert isinstance(example, dict)  # for type checking

        attack_chunks = get_chunking_for_search_based(
            example["chunked_text"], dataset.modifiable_chunk_spec
        )

        prompt_template = self.victim.chunks_to_prompt_template(
            chunks=attack_chunks,
            perturb_min=self.attack_config.perturb_position_min,
            perturb_max=self.attack_config.perturb_position_max,
            rng=attack_state.rng_state.distributed_rng,
        )

        example["text"] = prompt_template.build_prompt()

        prepped_example = PreppedExample(
            prompt_template=prompt_template,
            clf_label=example["proxy_clf_label"],
            gen_target=example["proxy_gen_target"],
        )

        assert isinstance(self.attack_config, SearchBasedAttackConfig)
        empty_prompt = prompt_template.build_prompt()
        with get_caching_model_with_example(self.victim, empty_prompt) as victim:
            runner = make_runner(
                victim=victim,
                prepped_examples=[prepped_example],
                random_seed=self.attack_config.seed,
                n_its=n_its,
                config=self.attack_config,
            )

            with victim.flop_count_context() as flop_count:
                final_attack_string, example_info = runner.run()
            example_flops = flop_count.flops
            final_attack_string, example_info = runner.run()

        iteration_texts = [
            prompt_template.build_prompt(attack_text=a)
            for a in example_info["attack_strings"]
        ]
        logits = example_info["logits"]

        attacked_input_text = prompt_template.build_prompt(
            attack_text=final_attack_string,
        )
        new_attacked_example = AttackedExample(
            example_index=example_index,
            attacked_text=attacked_input_text,
            iteration_texts=iteration_texts,
            logits=logits,
            flops=example_flops,
        )
        # We use tuples for immutability
        new_previous_examples = attack_state.previously_attacked_examples + (
            new_attacked_example,
        )

        # Make a new attack state with the updated examples
        new_attack_state = AttackState(
            previously_attacked_examples=new_previous_examples,
            rng_state=attack_state.rng_state.update_states(),
        )
        return new_attack_state
