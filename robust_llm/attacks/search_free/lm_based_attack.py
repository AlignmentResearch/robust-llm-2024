from functools import cached_property
from typing import Optional

import torch

from robust_llm.attacks.attack import PromptAttackMode
from robust_llm.attacks.search_free.search_free import SearchFreeAttack
from robust_llm.config.attack_configs import LMBasedAttackConfig
from robust_llm.logging_utils import LoggingCounter
from robust_llm.models import WrappedModel
from robust_llm.rllm_datasets.modifiable_chunk_spec import ChunkType
from robust_llm.scoring_callbacks import CallbackRegistry
from robust_llm.utils import get_randint_with_exclusions


class LMBasedAttack(SearchFreeAttack):

    REQUIRES_TRAINING = False

    def __init__(
        self,
        attack_config: LMBasedAttackConfig,
        victim: WrappedModel,
    ) -> None:
        super().__init__(attack_config)

        if victim.accelerator is None:
            raise ValueError("Accelerator must be provided")

        self.victim = victim
        self.adversary = WrappedModel.from_config(
            attack_config.adversary, accelerator=victim.accelerator
        )
        self.adversary.eval()
        assert self.adversary.can_generate()
        if attack_config.n_its > 1:
            # If we're doing multiple iterations, we need to sample.
            # Otherwise we will always generate the same text.
            assert self.adversary.generation_config is not None
            assert self.adversary.generation_config.do_sample

        self.templates = attack_config.templates
        if self.is_classification_task:
            assert len(self.templates) == self.num_labels
        else:
            assert len(self.templates) == 1
        self.n_its = attack_config.n_its
        self.adversary_batch_size = attack_config.adversary_batch_size
        self.victim_batch_size = attack_config.victim_batch_size
        self.prompt_attack_mode = PromptAttackMode(attack_config.prompt_attack_mode)
        self.victim_success_binary_callback = CallbackRegistry.get_binary_callback(
            name=attack_config.victim_success_binary_callback,
        )

        self.logging_counter = LoggingCounter(_name="lm_based_attack")

    @property
    def num_labels(self) -> int:
        return getattr(self.victim.model, "num_labels", 0)

    @property
    def is_generative_task(self) -> bool:
        return self.num_labels == 0

    @property
    def is_classification_task(self) -> bool:
        return not self.is_generative_task

    @cached_property
    def shared_attack_tokens(self) -> list[list[int]]:
        raise NotImplementedError("multiprompt is not implemented for LMBasedAttack")

    def _get_attack_tokens(
        self,
        chunk_text: str,
        chunk_type: ChunkType,
        current_iteration: int,
        chunk_label: int,
        chunk_seed: Optional[int] = None,
    ) -> list[int]:
        """Returns the LM red-team attack tokens for the current iteration.

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
        assert self.prompt_attack_mode == PromptAttackMode.SINGLEPROMPT
        target_label = (
            get_randint_with_exclusions(high=self.num_labels, exclusions=[chunk_label])
            if self.is_classification_task
            else 0
        )
        chunk_plus_template = chunk_text + self.templates[target_label]
        inputs = self.adversary.tokenizer(
            chunk_plus_template,
            return_tensors="pt",
            add_special_tokens=False,
        )
        inputs = inputs.to(device=self.adversary.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        assert input_ids.shape[0] == 1
        assert attention_mask.shape[0] == 1

        # Ensure that generations are deterministic for a given seed/iteration combo
        self.adversary.set_seed(hash((chunk_seed, current_iteration)))
        all_tokens = self.adversary.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.adversary.tokenizer.pad_token_id,
        )
        assert isinstance(all_tokens, torch.Tensor)
        assert all_tokens.shape[0] == 1
        output_tokens = all_tokens[0, input_ids.shape[1] :]

        return output_tokens.cpu().tolist()
