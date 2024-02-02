import copy
from typing import Optional, Sequence

import textattack
import torch
import transformers
from datasets import Dataset
from typing_extensions import override

from robust_llm.attacks.attack import Attack
from robust_llm.configs import AttackConfig
from robust_llm.dataset_management.dataset_management import ModifiableChunksSpec

TEXT_ATTACK_ATTACK_TYPES = ["textfooler", "bae", "checklist", "pso"]


class TextAttackAttack(Attack):
    """Attack using the TextAttack library."""

    REQUIRES_INPUT_DATASET = True

    def __init__(
        self,
        attack_config: AttackConfig,
        modifiable_chunks_spec: ModifiableChunksSpec,
        model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizerBase,
    ) -> None:
        """Constructor for TextAttackAttack.

        Args:
            attack_config: config of the attack
            modifiable_chunks_spec: Specification for which chunks of the
                original text can be modified
            model: attacked model
            tokenizer: tokenizer used with the model
        """
        super().__init__(attack_config, modifiable_chunks_spec)

        assert modifiable_chunks_spec == (True,)

        wrapped_model = textattack.models.wrappers.HuggingFaceModelWrapper(
            model, tokenizer
        )

        if attack_config.attack_type == "textfooler":
            self._attack = textattack.attack_recipes.TextFoolerJin2019.build(
                model_wrapper=wrapped_model
            )
        elif attack_config.attack_type == "bae":
            self._attack = textattack.attack_recipes.BAEGarg2019.build(
                model_wrapper=wrapped_model
            )
        elif attack_config.attack_type == "checklist":
            self._attack = textattack.attack_recipes.CheckList2020.build(
                model_wrapper=wrapped_model
            )
        elif attack_config.attack_type == "pso":
            self._attack = textattack.attack_recipes.PSOZang2020.build(
                model_wrapper=wrapped_model
            )
        else:
            raise ValueError(f"Attack type {attack_config.attack_type} not recognized.")

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
    ) -> Dataset:
        assert dataset is not None

        dataset = textattack.datasets.HuggingFaceDataset(dataset)

        attack_args = copy.deepcopy(self.attack_args)
        if max_n_outputs is not None:
            attack_args.num_examples = max_n_outputs

        attacker = textattack.Attacker(self._attack, dataset, attack_args)
        attack_results = attacker.attack_dataset()
        attacked_dataset = self._get_dataset_from_attack_results(attack_results)

        return attacked_dataset

    @staticmethod
    def _get_dataset_from_attack_results(
        attack_results: Sequence[textattack.attack_results.AttackResult],
    ) -> Dataset:
        texts, original_texts, labels = [], [], []
        for attack_result in attack_results:
            texts.append(attack_result.perturbed_result.attacked_text.text)
            original_texts.append(attack_result.original_result.attacked_text.text)
            labels.append(attack_result.perturbed_result.ground_truth_output)

        return Dataset.from_dict(
            {
                "text": texts,
                "original_text": original_texts,
                "label": labels,
            }
        )
