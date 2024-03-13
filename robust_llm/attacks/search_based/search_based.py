import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import transformers
from datasets import Dataset
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.attacks.search_based.runners import make_runner
from robust_llm.attacks.search_based.utils import (
    PromptTemplate,
    get_chunking_for_search_based,
    get_wrapped_model,
)
from robust_llm.configs import AttackConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec
from robust_llm.utils import LanguageModel, get_randint_with_exclusions

logger = logging.getLogger(__name__)


class SearchBasedAttack(Attack):
    """Implementation of the search-based attacks.

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

        config = self.attack_config.search_based_attack_config

        all_filtered_out_counts: List[int] = []

        attacked_input_texts = []
        for example in dataset:
            assert isinstance(example, dict)

            unmodifiable_prefix, modifiable_infix, unmodifiable_suffix = (
                get_chunking_for_search_based(
                    example["text_chunked"], self.modifiable_chunks_spec
                )
            )
            if not self.attack_config.append_to_modifiable_chunk:
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

            runner = make_runner(
                wrapped_model=self.wrapped_model,
                prompt_template=prompt_template,
                clf_target=target_label,
                random_seed=self.attack_config.seed,
                config=config,
            )
            attack_text, example_debug_info = runner.run()
            attacked_input_text = prompt_template.build_prompt(
                attack_text=attack_text,
            )
            attacked_input_texts.append(attacked_input_text)
            all_filtered_out_counts.append(example_debug_info["all_filtered_out_count"])

        attacked_dataset = Dataset.from_dict(
            {
                "text": attacked_input_texts,
                "original_text": dataset["text"],
                "label": dataset["label"],
            }
        )
        info_dict = _create_info_dict(all_filtered_out_counts)

        return attacked_dataset, info_dict


def _create_info_dict(all_filtered_out_counts: List[int]) -> Dict[str, Any]:
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
