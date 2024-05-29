from typing import Any

import numpy as np
from tqdm import tqdm
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.attacks.search_based.runners import make_runner
from robust_llm.attacks.search_based.utils import (
    PreppedExample,
    PromptTemplate,
    get_chunking_for_search_based,
    get_label_and_target_for_attack,
)
from robust_llm.config.attack_configs import SearchBasedAttackConfig
from robust_llm.config.configs import AttackConfig
from robust_llm.models import WrappedModel
from robust_llm.models.caching_wrapped_model import get_caching_model_with_example
from robust_llm.rllm_datasets.modifiable_chunk_spec import ChunkType
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

    REQUIRES_TRAINING = False

    def __init__(
        self,
        attack_config: AttackConfig,
        victim: WrappedModel,
    ) -> None:
        super().__init__(attack_config)

        if victim.accelerator is None:
            raise ValueError("Accelerator must be provided")

        self.victim = victim

    @override
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
    ) -> tuple[RLLMDataset, dict[str, Any]]:
        """Run a search-based attack separately on each example in the dataset.

        TODO(GH#113): consider multi-model attacks in the future.
        TODO(GH#114): consider multi-prompt attacks in the future.
        """

        # preconditions
        assert (
            dataset.modifiable_chunk_spec.n_modifiable_chunks == 1
        ), "Exactly one modifiable chunk"

        all_filtered_out_counts: list[int] = []

        attacked_input_texts = []
        for example in tqdm(dataset.ds, mininterval=5, position=-1):
            assert isinstance(example, dict)  # for type checking

            unmodifiable_prefix, modifiable_infix, unmodifiable_suffix = (
                get_chunking_for_search_based(
                    example["chunked_text"], dataset.modifiable_chunk_spec
                )
            )
            infix_chunk_type = dataset.modifiable_chunk_spec.get_modifiable_chunk()
            if infix_chunk_type == ChunkType.OVERWRITABLE:
                modifiable_infix = ""

            prompt_template = PromptTemplate(
                before_attack=unmodifiable_prefix + modifiable_infix,
                after_attack=unmodifiable_suffix,
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

        attacked_dataset = dataset.with_attacked_text(attacked_input_texts)
        info_dict = _create_info_dict(all_filtered_out_counts)

        return attacked_dataset, info_dict


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
