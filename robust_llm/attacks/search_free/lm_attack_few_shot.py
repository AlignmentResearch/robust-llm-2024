from copy import deepcopy
from typing import Any, Optional

import torch
from typing_extensions import override

from robust_llm.attacks.search_free.lm_attack_zero_shot import ZeroShotLMAttack
from robust_llm.attacks.search_free.search_free import (
    SUCCESS_TYPE,
    SearchFreeAttack,
    get_first_attack_success_index,
)
from robust_llm.config.attack_configs import FewShotLMAttackConfig
from robust_llm.models.prompt_templates import Conversation
from robust_llm.models.wrapped_model import WrappedModel
from robust_llm.rllm_datasets.modifiable_chunk_spec import ChunkType
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


def clean_text(text: str) -> str:
    return text.replace("{", "").replace("}", "")


class FewShotLMAttack(SearchFreeAttack):
    """Stochastic Few-Shot LM red-teaming attack.

    This attack aims to improve on zero-shot red-teaming by giving the adversary
    the history of previous attacks and their success scores. The adversary can
    then use this information to generate more effective attacks.

    Attributes:
        adversary: The adversary model used to generate attack tokens.
        n_turns: The number of turns to use for the adversary chat.
        few_shot_score_template: The template for including a previous attack's
            response and score in the context.
        adversary_input_templates: The templates for the adversary input, one for each
            target label.
        zero_shot_attack: The zero-shot attack called in each turn.
        num_templates: The number of templates for the adversary input.
    """

    def __init__(
        self,
        attack_config: FewShotLMAttackConfig,
        victim: WrappedModel,
        run_name: str,
        logging_name: Optional[str] = None,
    ) -> None:
        super().__init__(attack_config, victim, run_name, logging_name=logging_name)

        self.n_turns = attack_config.n_turns
        self.few_shot_score_template = attack_config.few_shot_score_template
        self.initial_adversary_prefix = (
            attack_config.initial_adversary_prefix
            if attack_config.initial_adversary_prefix is not None
            else attack_config.adversary_prefix
        )
        self.adversary_input_templates = attack_config.adversary_input_templates
        zero_shot_config = deepcopy(attack_config)
        # We only want to run the zero-shot attack once per turn.
        zero_shot_config.initial_n_its = 1
        # We don't want to reinsert the data or reapply the chat template on every turn.
        zero_shot_config.use_raw_adversary_input = True
        self.zero_shot_attack = ZeroShotLMAttack(
            zero_shot_config,
            victim=victim,
            run_name=run_name,
            logging_name=logging_name,
        )

    @property
    def adversary(self) -> WrappedModel:
        return self.zero_shot_attack.adversary

    @property
    def num_templates(self) -> int:
        return len(self.adversary_input_templates)

    def run_single_turn(
        self, turn: int, conv: Conversation, example: dict, dataset: RLLMDataset
    ) -> tuple[str, dict[str, Any], float]:
        """Run a single turn of the few-shot attack.

        This method runs a single zero-shot attack and updates the conversation history
        """
        # Update the adversary input to include the conversation history
        self.zero_shot_attack.adversary_input_templates = [
            conv.get_prompt(skip_last_suffix=True) for _ in range(self.num_templates)
        ]
        attack_text, attack_info, victim_successes = (
            self.zero_shot_attack.attack_example(
                example,
                dataset,
                n_its=1,
            )
        )
        attack_info["turn"] = turn
        assert attack_info.pop("success_index") == 0
        assert attack_info.pop("current_iteration") == 0
        attack_generation = clean_text(attack_info["generation_outputs"])
        adversary_output = self.adversary.clean_chat_artifacts(
            attack_info["adversary_output"]
        )
        attack_generation = self.victim.clean_chat_artifacts(attack_generation)

        # Convert victim to attack success
        assert isinstance(
            victim_successes, torch.Tensor
        ), "Did you forget to use a tensor callback for few-shot?"
        victim_success = victim_successes.item()
        attack_success = 1 - victim_success
        attack_info["attack_score"] = attack_success

        # Update the conversation history with the adversary output
        conv.append_to_last_message(adversary_output)
        conv.append_user_message(
            self.few_shot_score_template.format(
                response=attack_generation,
                score=attack_success,
            )
        )
        conv.append_assistant_message(self.zero_shot_attack.adversary_prefix)
        return attack_text, attack_info, victim_success

    def run_single_iteration(
        self, target_label: int, example: dict, dataset: RLLMDataset
    ) -> tuple[str, dict[str, Any], float]:
        """Run a single iteration of the few-shot attack.

        This method runs the zero-shot attack n_turns times and keeps the best attack.
        We progressively build the adversary prompt by including the success of
        previous attacks.
        """
        conv = self.adversary.init_conversation()
        conv.append_user_message(self.adversary_input_templates[target_label])
        conv.append_assistant_message(self.initial_adversary_prefix)

        best_text = None
        best_info = None
        best_success = None
        for turn in range(self.n_turns):
            attack_text, attack_info, victim_success = self.run_single_turn(
                turn, conv, example, dataset
            )
            if best_success is None or victim_success < best_success:
                best_text = attack_text
                best_info = attack_info
                best_success = victim_success

        assert best_text is not None
        assert best_info is not None
        assert best_success is not None

        return best_text, best_info, best_success

    @override
    def attack_example(
        self, example: dict[str, Any], dataset: RLLMDataset, n_its: int
    ) -> tuple[str, dict[str, Any], SUCCESS_TYPE]:
        """Run few-shot attack on a single example.

        We run the zero-shot attack n_its x n_turns times and keep the best attack.
        In each iteration, we progressively build the adversary prompt by including
        the success of previous attacks in the same iteration.

        Each iteration should be independent. TODO(Oskar): parallelize this
        """
        attack_text_iters = []
        attack_info_iters = []
        victim_successes_iters = []
        example_seed = example["seed"]
        for iteration in range(n_its):
            # Reset the seed for each iteration
            example["seed"] = hash((example_seed, iteration))

            iteration_text, iteration_info, iteration_success = (
                self.run_single_iteration(example["proxy_clf_label"], example, dataset)
            )

            attack_text_iters.append(iteration_text)
            attack_info_iters.append(iteration_info)
            victim_successes_iters.append(iteration_success)
        assert (
            len(attack_text_iters)
            == len(attack_info_iters)
            == len(victim_successes_iters)
            == n_its
        )
        victim_successes_tensor = torch.tensor(
            victim_successes_iters, device=self.victim.device
        )
        success_iteration = get_first_attack_success_index(victim_successes_tensor)
        final_text = attack_text_iters[success_iteration]
        success_info = attack_info_iters[success_iteration]
        info = {
            "success_iteration": success_iteration,
        }
        info.update(success_info)
        return (
            final_text,
            info,
            victim_successes_tensor,
        )

    @override
    def _get_attack_tokens(  # type: ignore[override]
        self,
        chunk_text: str,
        chunk_type: ChunkType,
        current_iteration: int,
        chunk_proxy_label: int,
        chunk_seed: int,
    ) -> list[int]:  # type: ignore
        # We override attack_example instead of this method.
        raise NotImplementedError
