import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import textattack
import transformers
from datasets import Dataset
from pyparsing import Any
from textattack.attack_recipes import AttackRecipe
from textattack.attack_results import (
    FailedAttackResult,
    MaximizedAttackResult,
    SkippedAttackResult,
    SuccessfulAttackResult,
)
from textattack.constraints.constraint import Constraint
from textattack.constraints.pre_transformation_constraint import (
    PreTransformationConstraint,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.shared import AttackedText
from textattack.transformations import (
    CompositeTransformation,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)
from transformers import PreTrainedTokenizerBase
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.configs import AttackConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec
from robust_llm.defenses.defense import DefendedModel
from robust_llm.utils import LanguageModel

TEXT_ATTACK_ATTACK_TYPES = [
    "textfooler",
    "bae",
    "checklist",
    "pso",
    "random_character_changes",
]

SPECIAL_MODIFIABLE_WORD = "special_modifiable_word"


class ModifyOnlySpecialWordsConstraint(PreTransformationConstraint):
    def _get_modifiable_indices(  # pyright: ignore [reportIncompatibleMethodOverride]
        self, current_text: AttackedText
    ):
        # Compute this once per each `AttackedText` instance and then always return
        # the same indices.
        prev = current_text.attack_attrs.get("previous_attacked_text")
        if prev is not None:
            current_text.attack_attrs["modifiable_indices"] = prev.attack_attrs[
                "modifiable_indices"
            ]
        else:
            current_text.attack_attrs["modifiable_indices"] = set(
                np.where(np.array(current_text.words) == SPECIAL_MODIFIABLE_WORD)[0]
            )
        return current_text.attack_attrs["modifiable_indices"]


class LanguageModelWrapper(textattack.models.wrappers.HuggingFaceModelWrapper):
    """Wrapper for TextAttack's `HuggingFaceModelWrapper` to allow compatibility with
    the `DefendedModel` class."""

    def __init__(
        self,
        model: Union[transformers.PreTrainedModel, DefendedModel],
        tokenizer: PreTrainedTokenizerBase,
    ):
        assert isinstance(model, (transformers.PreTrainedModel, DefendedModel)), (
            "`model` must be of type `transformers.PreTrainedModel` "
            f"or `DefendedModel`, but got type {type(model)}."
        )
        assert isinstance(
            tokenizer,
            transformers.PreTrainedTokenizerBase,
        ), (
            f"`tokenizer` must of type `transformers.PreTrainedTokenizerBase`, "
            f"but got type {type(tokenizer)}."
        )

        self.model = model
        self.tokenizer = tokenizer


class RandomCharacterChanges(AttackRecipe):
    @staticmethod
    def build(model_wrapper: LanguageModelWrapper, **kwargs) -> textattack.Attack:
        transformation = CompositeTransformation(
            [
                WordSwapRandomCharacterDeletion(),
                WordSwapRandomCharacterSubstitution(),
                WordSwapRandomCharacterInsertion(),
            ]
        )
        constraints = [ModifyOnlySpecialWordsConstraint()]
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="random")

        return textattack.Attack(
            goal_function, constraints, transformation, search_method  # type: ignore
        )


def _preprocess_example(
    example: Dict[str, Any],
    modifiable_chunks_spec: ModifiableChunksSpec,
    num_modifiable_words_per_chunk: Optional[int],
    ground_truth_label_fn: Optional[Callable[[str], int]],
) -> Dict[str, Any]:
    """Preprocess text of a single example before the attack.

    In preprocessing, we do things like replacing the modifiable chunk with placeholder
    words if needed, and re-computing the ground truth label if needed.

    Args:
        example: example to preprocess
        modifiable_chunks_spec: Specification for which chunks of the original text can
            be modified
        num_modifiable_words_per_chunk: Number of words to replace each modifiable
            chunk with
        ground_truth_label_fn: function to get the ground truth label from input text

    Returns:
        Preprocessed example.
    """

    example["original_text"] = example["text"]
    example["original_label"] = example["label"]

    # Replace the modifiable chunk with special words if needed.
    if num_modifiable_words_per_chunk is not None:
        text_chunked: Sequence[str] = example["text_chunked"]

        result = []
        for chunk, modifiable in zip(text_chunked, modifiable_chunks_spec):
            if modifiable:
                result.append(
                    f" {SPECIAL_MODIFIABLE_WORD}" * num_modifiable_words_per_chunk
                    + " "  # We want spaces before and after the modifiable chunk
                )
            else:
                result.append(chunk)

        example["text"] = "".join(result)
        if ground_truth_label_fn is not None:
            example["label"] = ground_truth_label_fn(example["text"])

    return example


class TextAttackAttack(Attack):
    """Attack using the TextAttack library."""

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
        """Constructor for TextAttackAttack.

        Args:
            attack_config: config of the attack
            modifiable_chunks_spec: Specification for which chunks of the
                original text can be modified
            model: attacked model
            tokenizer: tokenizer used with the model
            ground_truth_label_fn: function to get the ground truth label from
                input text
        """
        super().__init__(attack_config, modifiable_chunks_spec)

        assert sum(modifiable_chunks_spec) == 1

        assert isinstance(model, (transformers.PreTrainedModel, DefendedModel)), (
            "`model` must be of type `transformers.PreTrainedModel` "
            f"or `DefendedModel`, but got type {type(model)}."
        )
        wrapped_model = LanguageModelWrapper(model, tokenizer)

        self.ground_truth_label_fn = ground_truth_label_fn

        self.num_modifiable_words_per_chunk = (
            self.attack_config.text_attack_attack_config.num_modifiable_words_per_chunk
        )

        if len(modifiable_chunks_spec) > 1:
            assert (
                self.num_modifiable_words_per_chunk is not None
            ), "Special words must be used if only part of the text is modifiable."

        if attack_config.attack_type == "textfooler":
            assert self.num_modifiable_words_per_chunk is None, "Not supported."
            self._attack = textattack.attack_recipes.TextFoolerJin2019.build(
                model_wrapper=wrapped_model
            )
        elif attack_config.attack_type == "bae":
            self._attack = textattack.attack_recipes.BAEGarg2019.build(
                model_wrapper=wrapped_model
            )
        elif attack_config.attack_type == "checklist":
            assert self.num_modifiable_words_per_chunk is None, "Not supported."
            self._attack = textattack.attack_recipes.CheckList2020.build(
                model_wrapper=wrapped_model
            )
        elif attack_config.attack_type == "pso":
            assert self.num_modifiable_words_per_chunk is None, "Not supported."
            self._attack = textattack.attack_recipes.PSOZang2020.build(
                model_wrapper=wrapped_model
            )
        elif attack_config.attack_type == "random_character_changes":
            assert self.num_modifiable_words_per_chunk is not None, "Not supported."

            self._attack = RandomCharacterChanges.build(model_wrapper=wrapped_model)
        else:
            raise ValueError(f"Attack type {attack_config.attack_type} not recognized.")

        if self.num_modifiable_words_per_chunk is not None:
            # In this case, we drop any regular constraints (e.g. on semantics or
            # avoiding repeats) and only allow replacing the special words.
            self._attack = TextAttackAttack._make_attack_with_new_constraints(
                self._attack, [ModifyOnlySpecialWordsConstraint()]
            )

        self.attack_args = textattack.AttackArgs(
            num_examples=self.attack_config.text_attack_attack_config.num_examples,
            query_budget=self.attack_config.text_attack_attack_config.query_budget,
            random_seed=self.attack_config.seed,
            # Despite TextAttack's documentation, we need to set both of these
            # to actually make the attack silent.
            silent=self.attack_config.text_attack_attack_config.silent,
            disable_stdout=self.attack_config.text_attack_attack_config.silent,
        )

    @override
    def get_attacked_dataset(
        self,
        dataset: Optional[Dataset],
        max_n_outputs: Optional[int] = None,
    ) -> Tuple[Dataset, Dict[str, Any]]:
        assert dataset is not None

        dataset = self._preprocess_dataset(dataset)

        text_attack_dataset = textattack.datasets.HuggingFaceDataset(dataset)

        attack_args = copy.deepcopy(self.attack_args)
        if max_n_outputs is not None:
            attack_args.num_examples = max_n_outputs

        attacker = textattack.Attacker(self._attack, text_attack_dataset, attack_args)
        attack_results = attacker.attack_dataset()
        attacked_dataset = self._get_dataset_from_attack_results(
            attack_results=attack_results, original_dataset=dataset
        )

        info_dict = self._get_info_dict(attack_results)

        return attacked_dataset, info_dict

    @staticmethod
    def _get_dataset_from_attack_results(
        attack_results: Sequence[textattack.attack_results.AttackResult],
        original_dataset: Dataset,
    ) -> Dataset:
        texts, labels = [], []
        for attack_result, original_example in zip(attack_results, original_dataset):
            assert isinstance(original_example, dict)

            texts.append(attack_result.perturbed_result.attacked_text.text)
            labels.append(original_example["original_label"])

            assert (
                attack_result.original_result.attacked_text.text
                == original_example["text"]
            )

        return Dataset.from_dict(
            {
                "text": texts,
                "original_text": original_dataset["original_text"],
                "label": labels,
            }
        )

    @staticmethod
    def _make_attack_with_new_constraints(
        attack: textattack.Attack,
        new_constraints: List[Union[Constraint, PreTransformationConstraint]],
    ):
        return textattack.Attack(
            goal_function=attack.goal_function,
            constraints=new_constraints,
            transformation=attack.transformation,
            search_method=attack.search_method,
        )

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess dataset before the attack."""

        # Assignments below are somehow needed so that .map() doesn't complain about
        # caching the context.
        modifiable_chunks_spec = self.modifiable_chunks_spec
        num_modifiable_words_per_chunk = self.num_modifiable_words_per_chunk
        ground_truth_label_fn = self.ground_truth_label_fn
        dataset = dataset.map(
            lambda x: _preprocess_example(
                x,
                modifiable_chunks_spec=modifiable_chunks_spec,
                num_modifiable_words_per_chunk=num_modifiable_words_per_chunk,
                ground_truth_label_fn=ground_truth_label_fn,
            )
        )
        return dataset

    def _get_info_dict(
        self,
        attack_results: Sequence[textattack.attack_results.AttackResult],
    ) -> Dict[str, Any]:
        """Gather some info for debug purposes."""

        # For any attack trial, we define "modified ratio" as the ratio of words that
        # have been modified among the words that possibly could have been modified
        # (i.e., words in modifiable chunks). We gather these ratios, for
        # individual examples, separately for successes, failures and skipped (as
        # defined by TextAttack). Rationale for this is that for debug purposes, we
        # want to look if the ratios can get reasonably high, especially for failures
        # and when the query budget is set high.
        ratios_of_modified_words_among_ta_successes = []
        ratios_of_modified_words_among_ta_failures = []
        ratios_of_modified_words_among_ta_skipped = []

        for attack_result in attack_results:
            num_possible_to_modify = self.num_modifiable_words_per_chunk or len(
                attack_result.perturbed_result.attacked_text.words
            )
            assert num_possible_to_modify > 0

            ratio_of_modified_words = (
                len(
                    attack_result.perturbed_result.attacked_text.attack_attrs[
                        "modified_indices"
                    ]
                )
                / num_possible_to_modify
            )

            if isinstance(
                attack_result, (SuccessfulAttackResult, MaximizedAttackResult)
            ):
                ratios_of_modified_words_among_ta_successes.append(
                    ratio_of_modified_words
                )
            elif isinstance(attack_result, FailedAttackResult):
                ratios_of_modified_words_among_ta_failures.append(
                    ratio_of_modified_words
                )
            elif isinstance(attack_result, SkippedAttackResult):
                ratios_of_modified_words_among_ta_skipped.append(
                    ratio_of_modified_words
                )

        return {
            "debug/ratios_of_modified_words_among_ta_successes": ratios_of_modified_words_among_ta_successes,  # noqa: E501
            "debug/ratios_of_modified_words_among_ta_failures": ratios_of_modified_words_among_ta_failures,  # noqa: E501
            "debug/ratios_of_modified_words_among_ta_skipped": ratios_of_modified_words_among_ta_skipped,  # noqa: E501
        }
