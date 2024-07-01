from functools import cached_property
from typing import Any, Optional

import torch
from typing_extensions import override

from robust_llm.attacks.attack import PromptAttackMode
from robust_llm.attacks.search_free.search_free import SearchFreeAttack
from robust_llm.config.attack_configs import LMBasedAttackConfig
from robust_llm.logging_utils import LoggingCounter
from robust_llm.models import WrappedModel
from robust_llm.rllm_datasets.modifiable_chunk_spec import (
    ChunkType,
    ModifiableChunkSpec,
)
from robust_llm.scoring_callbacks import build_binary_scoring_callback
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
        if attack_config.n_its > 1:
            # If we're doing multiple iterations, we need to sample.
            # Otherwise we will always generate the same text.
            assert self.adversary.generation_config is not None
            assert self.adversary.generation_config.do_sample

        self.adversary_input_templates = attack_config.adversary_input_templates
        self.adversary_output_templates = attack_config.adversary_output_templates
        if self.is_classification_task:
            assert len(self.adversary_input_templates) == self.num_labels
        else:
            assert len(self.adversary_input_templates) == 1
        self.n_its = attack_config.n_its
        self.prompt_attack_mode = PromptAttackMode(attack_config.prompt_attack_mode)
        cb_config = attack_config.victim_success_binary_callback
        self.victim_success_binary_callback = build_binary_scoring_callback(cb_config)

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

    @override
    def post_process_attack_string(self, attack_tokens: str, chunk_index: int) -> str:
        """Post-process the attack string using the configured template.

        Args:
            attack_tokens: The attack tokens to post-process.
            chunk_index: The index of the chunk in the example.
        """
        return self.adversary_output_templates[chunk_index].format(attack_tokens)

    @override
    def _get_attack_tokens(
        self,
        chunk_text: str,
        chunk_type: ChunkType,
        current_iteration: int,
        chunk_label: int,
        chunk_seed: Optional[int],
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
        formatted_chunk = self.adversary_input_templates[target_label].format(
            chunk_text
        )
        inputs = self.adversary.tokenize(
            formatted_chunk,
            return_tensors="pt",
            # We use left-padding for autoregressive outputs.
            padding_side="left",
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
            pad_token_id=self.adversary.right_tokenizer.pad_token_id,
        )
        assert isinstance(all_tokens, torch.Tensor)
        assert all_tokens.shape[0] == 1
        output_tokens = all_tokens[0, input_ids.shape[1] :]

        return output_tokens.cpu().tolist()

    @override
    def _get_attacked_input(
        self,
        example: dict[str, Any],
        modifiable_chunk_spec: ModifiableChunkSpec,
        current_iteration: int,
    ) -> str:
        assert modifiable_chunk_spec.n_modifiable_chunks == len(
            self.adversary_output_templates
        )
        return super()._get_attacked_input(
            example, modifiable_chunk_spec, current_iteration
        )
