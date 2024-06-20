import copy
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import textattack
import transformers
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
from textattack.search_methods import GreedySearch, SearchMethod
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
from robust_llm.config.attack_configs import TextAttackAttackConfig
from robust_llm.defenses.defense import DefendedModel
from robust_llm.models import WrappedModel
from robust_llm.rllm_datasets.modifiable_chunk_spec import ChunkType
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset

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
        # Use GreedySearch to search until success and do not limit ourselves
        # to one pass over the words.
        search_method = GreedySearch()

        return textattack.Attack(
            goal_function, constraints, transformation, search_method  # type: ignore
        )


def _preprocess_example(
    example: dict[str, Any],
    dataset: RLLMDataset,
    num_modifiable_words_per_chunk: Optional[int],
) -> dict[str, Any]:
    """Preprocess text of a single example before the attack.

    In preprocessing, we do things like replacing the modifiable chunk with placeholder
    words if needed, and re-computing the ground truth label if needed.

    Args:
        example: Example to preprocess.
        dataset: The dataset the example came from.
        num_modifiable_words_per_chunk: Number of words to replace each modifiable
            chunk with.

    Returns:
        Preprocessed example.
    """

    # Replace the modifiable chunk with special words if needed.
    if num_modifiable_words_per_chunk is not None:
        if dataset.modifiable_chunk_spec.n_perturbable_chunks != 0:
            raise ValueError(
                "if `num_modifiable_words_per_chunk` is set, then there should be"
                " no PERTURBABLE chunks in the `modifiable_chunk_spec`"
                " (see GH#353)."
            )

        text_chunked: Sequence[str] = example["chunked_text"]

        result = []
        for chunk, chunk_type in zip(text_chunked, dataset.modifiable_chunk_spec):
            if chunk_type == ChunkType.OVERWRITABLE:
                result.append(
                    f" {SPECIAL_MODIFIABLE_WORD}" * num_modifiable_words_per_chunk
                    + " "  # We want spaces before and after the modifiable chunk
                )
            else:
                result.append(chunk)

        example["text"] = "".join(result)
        example = dataset.update_example_based_on_text(example)

    return example


class TextAttackAttack(Attack):
    """Attack using the TextAttack library."""

    REQUIRES_TRAINING = False

    def __init__(
        self,
        attack_config: TextAttackAttackConfig,
        victim: WrappedModel,
    ) -> None:
        """Constructor for TextAttackAttack.

        Args:
            attack_config: config of the attack
            victim: wrapped victim model
        """
        super().__init__(attack_config)

        assert isinstance(
            victim.model, (transformers.PreTrainedModel, DefendedModel)
        ), (
            "`victim.model` must be of type `transformers.PreTrainedModel` "
            f"or `DefendedModel`, but got type {type(victim.model)}."
        )
        # The reason we have to unwrap and then rewrap the model is that the
        # LanguageModelWrapper is a subclass of TextAttack's HuggingFaceModelWrapper,
        # which is the class that TextAttack attacks expect.
        # TODO (ian): Avoid unwrapping and rewrapping.
        # We use the right_tokenizer because we want to use right-padding for
        # classification.
        wrapped_model = LanguageModelWrapper(victim.model, victim.right_tokenizer)

        self.num_modifiable_words_per_chunk = (
            attack_config.num_modifiable_words_per_chunk
        )

        if attack_config.text_attack_recipe == "textfooler":
            assert self.num_modifiable_words_per_chunk is None, "Not supported."
            self._attack = textattack.attack_recipes.TextFoolerJin2019.build(
                model_wrapper=wrapped_model
            )
        elif attack_config.text_attack_recipe == "bae":
            self._attack = textattack.attack_recipes.BAEGarg2019.build(
                model_wrapper=wrapped_model
            )
            if self.num_modifiable_words_per_chunk is not None:
                # Use GreedySearch to search until success and do not limit ourselves
                # to one pass over the words.
                self._attack = TextAttackAttack._make_modified_attack(
                    self._attack,
                    new_search_method=GreedySearch(),
                )
        elif attack_config.text_attack_recipe == "checklist":
            assert self.num_modifiable_words_per_chunk is None, "Not supported."
            self._attack = textattack.attack_recipes.CheckList2020.build(
                model_wrapper=wrapped_model
            )
        elif attack_config.text_attack_recipe == "pso":
            assert self.num_modifiable_words_per_chunk is None, "Not supported."
            self._attack = textattack.attack_recipes.PSOZang2020.build(
                model_wrapper=wrapped_model
            )
        elif attack_config.text_attack_recipe == "random_character_changes":
            assert self.num_modifiable_words_per_chunk is not None, "Not supported."

            self._attack = RandomCharacterChanges.build(model_wrapper=wrapped_model)
        else:
            raise ValueError(
                f"TextAttack Recipe {attack_config.text_attack_recipe} not recognized."
            )

        if self.num_modifiable_words_per_chunk is not None:
            # In this case, we drop any regular constraints (e.g. on semantics or
            # avoiding repeats) and only allow replacing the special words.
            self._attack = TextAttackAttack._make_modified_attack(
                self._attack, new_constraints=[ModifyOnlySpecialWordsConstraint()]
            )

        self.attack_args = textattack.AttackArgs(
            num_examples=-1,  # Attack all examples
            query_budget=attack_config.query_budget,
            random_seed=attack_config.seed,
            # Despite TextAttack's documentation, we need to set both of these
            # to actually make the attack silent.
            silent=attack_config.silent,
            disable_stdout=attack_config.silent,
        )

    @override
    def get_attacked_dataset(
        self,
        dataset: RLLMDataset,
    ) -> tuple[RLLMDataset, dict[str, Any]]:
        assert dataset.modifiable_chunk_spec.n_modifiable_chunks == 1

        dataset = self._preprocess_dataset(dataset)

        # We need to set the columns schema explicitly because TextAttack
        # doesn't recognise 'clf_label' automatically.
        columns_schema = (["text"], "clf_label")
        text_attack_dataset = textattack.datasets.HuggingFaceDataset(
            name_or_dataset=dataset.ds,
            dataset_columns=columns_schema,
        )

        attack_args = copy.deepcopy(self.attack_args)

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
        original_dataset: RLLMDataset,
    ) -> RLLMDataset:
        assert original_dataset.ds is not None
        texts = []
        for attack_result, original_example in zip(attack_results, original_dataset.ds):
            assert isinstance(original_example, dict)

            texts.append(attack_result.perturbed_result.attacked_text.text)

            assert (
                attack_result.original_result.attacked_text.text
                == original_example["text"]
            )

        attacked_dataset = original_dataset.with_attacked_text(attacked_text=texts)
        return attacked_dataset

    @staticmethod
    def _make_modified_attack(
        attack: textattack.Attack,
        new_constraints: Optional[
            list[Union[Constraint, PreTransformationConstraint]]
        ] = None,
        new_search_method: Optional[SearchMethod] = None,
    ):
        return textattack.Attack(
            goal_function=attack.goal_function,
            constraints=new_constraints
            or (attack.constraints + attack.pre_transformation_constraints),
            transformation=attack.transformation,
            search_method=new_search_method or attack.search_method,
        )

    def _preprocess_dataset(self, dataset: RLLMDataset) -> RLLMDataset:
        """Preprocess dataset before the attack."""
        # Assignments below are somehow needed so that .map() doesn't complain about
        # caching the context.
        num_modifiable_words_per_chunk = self.num_modifiable_words_per_chunk
        new_ds = dataset.ds.map(
            lambda x: _preprocess_example(
                x,
                dataset=dataset,
                num_modifiable_words_per_chunk=num_modifiable_words_per_chunk,
            )
        )
        new_dataset = dataset.with_new_ds(new_ds)
        return new_dataset

    def _get_info_dict(
        self,
        attack_results: Sequence[textattack.attack_results.AttackResult],
    ) -> dict[str, Any]:
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
