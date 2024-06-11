from typing import cast, overload

from transformers import BatchEncoding
from typing_extensions import override

from robust_llm.models.wrapped_model import WrappedModel


class WrappedChatModel(WrappedModel):
    """Wrapper for Chat/IFT models.

    Chat and instruction-fine-tuned models require special handling for the
    input. Instead of regular text, they expect the input to be chat-formatted.
    The specific format varies from model to model, but
    tokenizer.apply_chat_template should abstract that away.
    """

    @override
    def tokenize(
        self,
        text: str | list[str],
        return_tensors: str | None = None,
        padding_side: str | None = None,
        add_special_tokens: bool = False,
        apply_chat_template: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        """Tokenize the input text, optionally applying the chat template.

        For now, we assume that the input text is a single 'user' message.

        Args:
            text:
                The input text.
            return_tensors:
                Whether to return tensors, and what type of tensors to return.
            padding_side:
                Whether to pad the input. None means no padding, "right" means
                do right padding, "left" means do left padding.
            add_special_tokens:
                Whether to add special tokens.
            apply_chat_template:
                Whether to apply the chat template.
            add_generation_prompt:
                Whether to add a generation prompt to the end of the prompt.
            **kwargs:
                Additional arguments (for compatibility with the base class,
                since it has **kwargs to accept apply_chat_template and
                add_generation_prompt).

        Returns:
            The tokenized input, as you'd get from calling a tokenizer.

        """
        to_tokenize: str | list[str]
        if apply_chat_template:
            if not add_special_tokens:
                raise ValueError(
                    "add_special_tokens must be True when apply_chat_template is True."
                )
            # This will apply the chat template: the 'maybe' refers to
            # whether this is a WrappedChatModel or not, which it is.
            to_tokenize = self.maybe_apply_chat_template(
                text, add_generation_prompt=add_generation_prompt
            )
        else:
            to_tokenize = text

        return super().tokenize(
            text=to_tokenize,
            return_tensors=return_tensors,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
        )

    @overload
    def maybe_apply_chat_template(
        self, text: str, add_generation_prompt: bool = True
    ) -> str: ...

    @overload
    def maybe_apply_chat_template(
        self, text: list[str], add_generation_prompt: bool = True
    ) -> list[str]: ...

    @override
    def maybe_apply_chat_template(
        self, text: str | list[str], add_generation_prompt: bool = True
    ) -> str | list[str]:
        """If working with a chat model, return text with chat template applied.

        Since this is the base class for chat models, we apply the chat template.

        Args:
            text: The text to apply the chat template to.
            add_generation_prompt: Whether to add a generation prompt. Usually True.

        Returns:
            The text with the chat template applied.
        """
        if isinstance(text, str):
            out_text = self.right_tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            return cast(str, out_text)

        elif isinstance(text, list):
            out_text = [
                self.right_tokenizer.apply_chat_template(
                    conversation=[{"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
                for t in text
            ]

            return cast(list[str], out_text)
        else:
            raise ValueError(f"Unexpected type for text: {type(text)}")
