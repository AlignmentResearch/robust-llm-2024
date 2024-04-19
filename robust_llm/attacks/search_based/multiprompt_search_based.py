import logging
from typing import Any

import numpy as np
import transformers
from accelerate import Accelerator, DistributedType
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.attacks.search_based.runners import make_runner
from robust_llm.attacks.search_based.utils import (
    PreppedExample,
    PromptTemplate,
    get_chunking_for_search_based,
    get_wrapped_model,
)
from robust_llm.configs import AttackConfig
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.utils import LanguageModel, get_randint_with_exclusions

logger = logging.getLogger(__name__)


class MultiPromptSearchBasedAttack(Attack):
    """Implementation of the search-based attacks for multiple prompts simultaneously.

    These are attacks that maintain a list of candidates and iteratively refine them by
    looking at the nearby candidates. Concrete approaches include GCG and beam search.
    Actual implementations of the logic of iteration, search, filtering, etc. is in
    the `runners` submodule, with the `SearchBasedRunner` abstract class as the base.

    For now, we allow only one modifiable chunk, so the inputs are of the form
    <unmodifiable_prefix><modifiable_infix><unmodifiable_suffix>. The attack will either
    completely replace modifiable infix with optimized tokens (if
    `attack_config.append_to_modifiable_chunk` is False) or, otherwise, will add the
    tokens after the modifiable infix.
    """

    REQUIRES_INPUT_DATASET = True
    REQUIRES_TRAINING = False

    def __init__(
        self,
        attack_config: AttackConfig,
        model: LanguageModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        accelerator: Accelerator,
    ) -> None:
        super().__init__(attack_config)

        if accelerator.distributed_type == DistributedType.NO:
            assert isinstance(model, transformers.PreTrainedModel)

        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.wrapped_model = get_wrapped_model(
            self.model, self.tokenizer, self.accelerator  # type: ignore
        )

    @override
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
    ) -> tuple[RLLMDataset, dict[str, Any]]:
        """Run a multi-prompt attack on the dataset.

        TODO(GH#113): consider multi-model attacks in the future.
        """

        config = self.attack_config.search_based_attack_config

        num_classes = dataset.num_classes

        all_filtered_out_counts: list[int] = []

        attacked_input_texts = []
        prepped_examples: list[PreppedExample] = []
        for example in dataset.ds:
            assert isinstance(example, dict)

            unmodifiable_prefix, modifiable_infix, unmodifiable_suffix = (
                get_chunking_for_search_based(
                    example["chunked_text"], dataset.modifiable_chunks_spec
                )
            )
            if not self.attack_config.append_to_modifiable_chunk:
                modifiable_infix = ""

            prompt_template = PromptTemplate(
                before_attack=unmodifiable_prefix + modifiable_infix,
                after_attack=unmodifiable_suffix,
            )

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
            prepped_examples.append(prepped_example)

        runner = make_runner(
            wrapped_model=self.wrapped_model,
            prepped_examples=prepped_examples,
            random_seed=self.attack_config.seed,
            config=config,
        )
        attack_text, example_debug_info = runner.run()
        attacked_input_texts = [
            example.prompt_template.build_prompt(attack_text=attack_text)
            for example in prepped_examples
        ]
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
