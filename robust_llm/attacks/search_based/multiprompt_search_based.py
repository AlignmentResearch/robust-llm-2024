from typing import Any

import numpy as np
from typing_extensions import override

from robust_llm.attacks.attack import Attack, AttackData, AttackOutput
from robust_llm.attacks.search_based.runners import make_runner
from robust_llm.attacks.search_based.utils import (
    PreppedExample,
    get_chunking_for_search_based,
)
from robust_llm.config.attack_configs import SearchBasedAttackConfig
from robust_llm.models.prompt_templates import PromptTemplate
from robust_llm.rllm_datasets.modifiable_chunk_spec import ChunkType
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


class MultiPromptSearchBasedAttack(Attack):
    """Implementation of the search-based attacks for multiple prompts simultaneously.

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

    CAN_CHECKPOINT = False
    REQUIRES_INPUT_DATASET = True
    REQUIRES_TRAINING = False

    @override
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
        n_its: int,
        resume_from_checkpoint: bool = False,
    ) -> AttackOutput:
        """Run a multi-prompt attack on the dataset.

        TODO(GH#113): consider multi-model attacks in the future.
        """
        all_filtered_out_counts: list[int] = []

        attacked_input_texts = []
        prepped_examples: list[PreppedExample] = []
        for example in dataset.ds:
            assert isinstance(example, dict)

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

            prepped_example = PreppedExample(
                prompt_template=prompt_template,
                clf_label=example["proxy_clf_label"],
                gen_target=example["proxy_gen_target"],
            )

            prepped_examples.append(prepped_example)

        assert isinstance(self.attack_config, SearchBasedAttackConfig)
        runner = make_runner(
            victim=self.victim,
            prepped_examples=prepped_examples,
            random_seed=self.attack_config.seed,
            n_its=n_its,
            config=self.attack_config,
        )
        attack_text, example_debug_info = runner.run()
        attacked_input_texts = [
            example.prompt_template.build_prompt(attack_text=attack_text)
            for example in prepped_examples
        ]
        all_filtered_out_counts.append(example_debug_info["all_filtered_out_count"])

        attacked_dataset = dataset.with_attacked_text(attacked_input_texts)
        info_dict = _create_info_dict(all_filtered_out_counts)
        attack_out = AttackOutput(
            dataset=attacked_dataset,
            attack_data=AttackData(),
            global_info=info_dict,
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
