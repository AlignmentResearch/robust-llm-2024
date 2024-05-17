from typing import Any

import numpy as np
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.attacks.search_based.runners import make_runner
from robust_llm.attacks.search_based.utils import (
    PreppedExample,
    PromptTemplate,
    get_chunking_for_search_based,
)
from robust_llm.config.attack_configs import SearchBasedAttackConfig
from robust_llm.config.configs import AttackConfig
from robust_llm.models import WrappedModel
from robust_llm.rllm_datasets.modifiable_chunk_spec import ChunkType
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.utils import get_randint_with_exclusions


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

        num_classes = dataset.num_classes

        all_filtered_out_counts: list[int] = []

        attacked_input_texts = []
        for example in dataset.ds:
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

            # maybe update the label after changing out modifiable chunk
            true_label = dataset.ground_truth_label_fn(
                prompt_template.build_prompt(), example["clf_label"]
            )

            target_label = get_randint_with_exclusions(
                high=num_classes, exclusions=[true_label]
            )

            prepped_example = PreppedExample(
                prompt_template=prompt_template,
                clf_target=target_label,
            )

            assert isinstance(self.attack_config, SearchBasedAttackConfig)
            runner = make_runner(
                wrapped_model=self.victim,
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
