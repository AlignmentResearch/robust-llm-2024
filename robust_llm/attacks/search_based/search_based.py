from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm
from typing_extensions import override

from robust_llm.attacks.attack import Attack, AttackData, AttackOutput, AttackState
from robust_llm.attacks.search_based.runners import make_runner
from robust_llm.attacks.search_based.utils import (
    PreppedExample,
    get_chunking_for_search_based,
    get_label_and_target_for_attack,
)
from robust_llm.config.attack_configs import AttackConfig, SearchBasedAttackConfig
from robust_llm.models.caching_wrapped_model import get_caching_model_with_example
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


@dataclass
class SearchBasedAttackState(AttackState):
    all_filtered_out_counts: list[int] = field(default_factory=list)


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
    REQUIRES_TRAINING = False

    def __init__(
        self,
        attack_config: AttackConfig,
        victim: WrappedModel,
        run_name: str,
        logging_name: str | None = None,
    ) -> None:
        super().__init__(
            attack_config, victim=victim, run_name=run_name, logging_name=logging_name
        )
        self.attack_state = SearchBasedAttackState()

    @override
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
        resume_from_checkpoint: bool = True,
    ) -> AttackOutput:
        """Run a search-based attack separately on each example in the dataset.

        TODO(GH#113): consider multi-model attacks in the future.
        TODO(GH#114): consider multi-prompt attacks in the future.
        """
        # preconditions
        assert (
            dataset.modifiable_chunk_spec.n_modifiable_chunks == 1
        ), "Exactly one modifiable chunk"
        if resume_from_checkpoint and self.maybe_load_state():
            assert isinstance(self.attack_state, SearchBasedAttackState)
            all_filtered_out_counts = self.attack_state.all_filtered_out_counts
            attacked_input_texts = self.attack_state.attacked_texts
            starting_index = self.attack_state.example_index + 1
        else:
            # Reset the state if not resuming from checkpoint
            all_filtered_out_counts = []
            attacked_input_texts = []
            self.attack_state = SearchBasedAttackState()
            starting_index = 0

        for example_index in tqdm(
            range(starting_index, len(dataset.ds)), mininterval=5, position=-1
        ):
            example = dataset.ds[example_index]
            assert isinstance(example, dict)  # for type checking

            unmodifiable_prefix, modifiable_infix, unmodifiable_suffix = (
                get_chunking_for_search_based(
                    example["chunked_text"], dataset.modifiable_chunk_spec
                )
            )

            prompt_template = self.victim.chunks_to_prompt_template(
                unmodifiable_prefix=unmodifiable_prefix,
                modifiable_infix=modifiable_infix,
                unmodifiable_suffix=unmodifiable_suffix,
            )

            # Maybe update the example after changing out modifiable chunk
            example["text"] = prompt_template.build_prompt()
            example = dataset.update_example_based_on_text(example)

            goal_label, goal_target = get_label_and_target_for_attack(example, dataset)

            prepped_example = PreppedExample(
                prompt_template=prompt_template,
                clf_label=goal_label,
                gen_target=goal_target,
            )

            assert isinstance(self.attack_config, SearchBasedAttackConfig)
            empty_prompt = prompt_template.build_prompt()
            with get_caching_model_with_example(self.victim, empty_prompt) as victim:
                runner = make_runner(
                    victim=victim,
                    prepped_examples=[prepped_example],
                    random_seed=self.attack_config.seed,
                    config=self.attack_config,
                )

                attack_text, example_debug_info = runner.run()
            attacked_input_text = prompt_template.build_prompt(
                attack_text=attack_text,
            )
            attacked_input_texts.append(attacked_input_text)
            all_filtered_out_counts.append(example_debug_info["all_filtered_out_count"])

            # Save the state
            self.attack_state = SearchBasedAttackState(
                example_index=example_index,
                attacked_texts=attacked_input_texts,
                all_filtered_out_counts=all_filtered_out_counts,
            )

            if resume_from_checkpoint:
                self.maybe_save_state()

        attacked_dataset = dataset.with_attacked_text(attacked_input_texts)
        global_info = _create_info_dict(all_filtered_out_counts)

        # We clear the gradients here to avoid using up GPU memory
        # even after the attack has stopped using it.
        # (This is primarily for GCG.)
        self.victim.model.zero_grad()
        attack_out = AttackOutput(
            dataset=attacked_dataset,
            attack_data=AttackData(),
            global_info=global_info,
        )
        return attack_out


def _create_info_dict(all_filtered_out_counts: list[int]) -> dict[str, Any]:
    return {
        # The number of examples for which there was some iteration where all
        # candidates were filtered out
        "debug/num_examples_all_filtered_out_happened": len(
            [c for c in all_filtered_out_counts if c > 0]
        ),
        # The average number of iterations (across examples) where all candidates
        # were filtered out
        "debug/avg_num_its_all_filtered_out": float(np.mean(all_filtered_out_counts)),
    }
