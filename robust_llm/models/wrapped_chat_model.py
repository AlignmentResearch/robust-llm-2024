from __future__ import annotations

from typing import Optional, overload

from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing_extensions import override

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models.model_utils import InferenceType
from robust_llm.models.prompt_templates import PromptTemplateBuilder
from robust_llm.models.wrapped_model import WrappedModel


class WrappedChatModel(WrappedModel):
    """Wrapper for Chat/IFT models.

    Chat and instruction-fine-tuned models require special handling for the
    input. Instead of regular text, they expect the input to be chat-formatted.
    The specific format varies from model to model, but this should abstract that away.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        right_tokenizer: PreTrainedTokenizerBase,
        accelerator: Accelerator | None,
        inference_type: InferenceType,
        train_minibatch_size: int,
        eval_minibatch_size: int,
        family: str,
        generation_config: GenerationConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            model,
            right_tokenizer,
            accelerator,
            inference_type,
            train_minibatch_size,
            eval_minibatch_size,
            family,
            generation_config,
            system_prompt,
        )
        assert self.prompt_builder != PromptTemplateBuilder(
            prompt_prefix="",
            system_prefix="",
            system_suffix="",
            user_prefix="",
            user_suffix="",
        )

    @overload
    def maybe_apply_chat_template(self, text: str) -> str: ...

    @overload
    def maybe_apply_chat_template(self, text: list[str]) -> list[str]: ...

    @override
    def maybe_apply_chat_template(self, text: str | list[str]) -> str | list[str]:
        """If working with a chat model, return text with chat template applied.

        Since this is the base class for chat models, we apply the chat template.

        Args:
            text: The text to apply the chat template to.

        Returns:
            The text with the chat template applied.
        """
        # We want to surround `text` with the user chat delimiters.
        # We could do this in multiple ways, but choose to set the base prompt
        # to be empty and pass the text as an attack.
        template = self.get_prompt_template()
        if isinstance(text, str):
            return template.build_prompt(attack_text=text)

        else:
            return [template.build_prompt(attack_text=t) for t in text]

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        accelerator: Accelerator | None,
        num_classes: Optional[int] = None,
        **kwargs,
    ) -> WrappedChatModel:
        model = super().from_config(
            config, accelerator, num_classes=num_classes, **kwargs
        )
        assert isinstance(model, WrappedChatModel)
        return model
