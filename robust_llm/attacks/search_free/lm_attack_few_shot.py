import math
from typing import Optional

import torch
from typing_extensions import override

from robust_llm.attacks.search_free.lm_attack_zero_shot import ZeroShotLMAttack
from robust_llm.config.attack_configs import FewShotLMAttackConfig
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.modifiable_chunk_spec import ChunkType
from robust_llm.scoring_callbacks import CallbackInput
from robust_llm.scoring_callbacks.scoring_callback_utils import (
    CallbackOutput,
    TensorCallbackOutput,
)


class FewShotLMAttack(ZeroShotLMAttack):
    """Stochastic Few-Shot LM red-teaming attack.

    This attack aims to improve on zero-shot red-teaming by giving the adversary
    a few examples of successful attacks to use as a basis for generating new
    attacks. The attack history is sampled with a temperature parameter to
    determine which attacks to use for the current iteration.

    Attributes:
        few_shot_temperature: The temperature parameter for sampling attacks.
        k_shot: The maximum number of attacks to sample from the attack history.
        adversary_shot_template: The template for including a previously successful
            attack in the adversary input.
        with_replacement: Whether to sample with replacement from the attack history.
        attack_history: The history of attacks and their success probability.
            Here 1 is the most successful attack and 0 is the least successful attack.
    """

    def __init__(
        self,
        attack_config: FewShotLMAttackConfig,
        victim: WrappedModel,
        run_name: str,
        logging_name: Optional[str] = None,
    ) -> None:
        super().__init__(attack_config, victim, run_name, logging_name=logging_name)

        self.few_shot_temperature = attack_config.few_shot_temperature
        self.k_shot = attack_config.k_shot
        self.adversary_shot_template = attack_config.adversary_shot_template
        self.with_replacement = attack_config.with_replacement
        self.attack_history: list[tuple[str, float]] = []

    @override
    def use_callback_results(
        self, callback_input: CallbackInput, callback_output: CallbackOutput
    ) -> None:
        assert callback_input.original_input_data is not None
        assert not isinstance(callback_input.original_input_data, torch.Tensor)
        assert not isinstance(callback_input.input_data, torch.Tensor)
        assert isinstance(callback_output, TensorCallbackOutput)
        self.attack_history += [
            # Flip the sign to switch from victim success to attack success
            (
                text.replace(original, ""),
                1 - (score.item() if isinstance(score, torch.Tensor) else score),
            )
            for original, text, score in zip(
                callback_input.original_input_data,
                callback_input.input_data,
                callback_output.losses,
                strict=True,
            )
        ]

    def sample_few_shot_examples(self) -> list[str]:
        """Sample k times from the attack history using the temperature parameter."""
        history = [(p, s) for p, s in self.attack_history if s > 0]
        if len(history) == 0:
            return []
        population = [p for p, _ in history]
        k = self.k_shot if self.with_replacement else min(self.k_shot, len(population))
        if self.few_shot_temperature == 0 and not self.with_replacement:
            # return the top k attacks
            return [x[0] for x in sorted(history, key=lambda x: x[1])[-k:]]
        elif self.few_shot_temperature == 0:
            # return the best attack k times
            return [max(history, key=lambda x: x[1])[0]] * k
        weights = [
            math.exp(math.log(s) / self.few_shot_temperature) for _, s in history
        ]
        if self.with_replacement:
            return self.rng.choices(population=population, weights=weights, k=k)
        else:
            sample = []
            for _ in range(k):
                index = self.rng.choices(
                    population=list(range(len(population))), weights=weights
                )[0]
                sample.append(population[index])
                population = population[:index] + population[index + 1 :]
                weights = weights[:index] + weights[index + 1 :]
            return sample

    @override
    def _get_attack_tokens(
        self,
        chunk_text: str,
        chunk_type: ChunkType,
        current_iteration: int,
        chunk_label: int,
        chunk_seed: Optional[int],
    ) -> list[int]:
        """Returns the few-shot LM red-team attack tokens for the current iteration.

        For classification, we pick a random alternate target label to decide
        the template to use for the attack. For generation, we always use the
        same template.

        We pass the template text to the adversary model, along with the chunk,
        to generate some attack tokens.

        Args:
            chunk_text: The text of the chunk to be attacked.
            chunk_type: The type of the chunk to be attacked (not used).
            current_iteration: Used to determine the seed for adversary generation.
            chunk_label: The label of the chunk to be attacked (for classification).
            chunk_seed: The seed for the chunk to be attacked (for generation).

        Returns:
            The attack tokens for the current iteration.
        """
        assert isinstance(chunk_text, str)
        assert isinstance(chunk_type, ChunkType)
        target_label = self.get_target_label(chunk_label)
        sampled_attacks = self.sample_few_shot_examples()
        formatted_chunk = self.adversary_input_templates[target_label].format(
            chunk_text
        ) + "".join(
            [
                self.adversary_shot_template.format(k=k + 1, shot=attack)
                for k, attack in enumerate(sampled_attacks)
            ]
        )
        return self.generate(formatted_chunk, current_iteration, chunk_seed)
