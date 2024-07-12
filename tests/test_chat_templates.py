from typing import Callable
from unittest.mock import MagicMock

import pytest
from transformers import AutoTokenizer

from robust_llm.models import (
    GemmaChatModel,
    GPTNeoXChatModel,
    Llama2ChatModel,
    QwenChatModel,
    TinyLlamaChatModel,
)
from robust_llm.models.prompt_templates import PromptTemplate
from robust_llm.models.wrapped_chat_model import WrappedChatModel

NAME_AND_TEMPLATE = [
    ("NousResearch/Llama-2-7b-chat-hf", "llama2-chat", Llama2ChatModel),
    ("Qwen/Qwen1.5-1.8B-Chat", "qwen1.5-chat", QwenChatModel),
    ("Qwen/Qwen2-7B-Instruct", "qwen2-chat", QwenChatModel),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama", TinyLlamaChatModel),
    ("Felladrin/Pythia-31M-Chat-v1", "pythia-chat", GPTNeoXChatModel),
]
# Gemma models do not support system prompts.
NAME_AND_TEMPLATE_NO_SYSTEM_PROMPT = NAME_AND_TEMPLATE + [
    ("google/gemma-1.1-2b-it", "gemma-chat", GemmaChatModel),
    ("google/gemma-2-9b-it", "gemma-chat", GemmaChatModel),
]


@pytest.mark.parametrize(
    "model_name, model_family, model_constructor", NAME_AND_TEMPLATE
)
def test_template_with_system_prompt(
    model_name: str, model_family: str, model_constructor: Callable
):
    model = model_constructor(
        model=MagicMock(),
        right_tokenizer=MagicMock(),
        accelerator=None,
        inference_type="generation",
        train_minibatch_size=2,
        eval_minibatch_size=3,
        generation_config=None,
        family=model_family,
        system_prompt="System prompt.",
    )
    assert isinstance(model, WrappedChatModel)
    template = model.get_prompt_template(
        "Unmodifiable prefix.",
        "Modifiable infix.",
        "Unmodifiable suffix.",
    )
    assert isinstance(template, PromptTemplate)
    prompt = template.build_prompt(attack_text="Attack text.")
    assert "Unmodifiable prefix." in prompt
    assert "Modifiable infix." in prompt
    assert "Attack text." in prompt
    assert "Unmodifiable suffix." in prompt
    assert "System prompt." in prompt
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_out = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "System prompt."},
            {
                "role": "user",
                "content": (
                    "Unmodifiable prefix.Modifiable infix.Attack text."
                    "Unmodifiable suffix."
                ),
            },
            {"role": "assistant", "content": ""},
        ],
        tokenize=False,
    )
    assert tokenizer_out[: len(prompt)] == prompt


@pytest.mark.parametrize(
    "model_name, model_family, model_constructor", NAME_AND_TEMPLATE_NO_SYSTEM_PROMPT
)
def test_template_without_system_prompt(
    model_name: str, model_family: str, model_constructor: Callable
):
    model = model_constructor(
        model=MagicMock(),
        right_tokenizer=MagicMock(),
        accelerator=None,
        inference_type="generation",
        train_minibatch_size=2,
        eval_minibatch_size=3,
        generation_config=None,
        family=model_family,
    )
    assert isinstance(model, WrappedChatModel)
    template = model.get_prompt_template(
        "Unmodifiable prefix.",
        "Modifiable infix.",
        "Unmodifiable suffix.",
    )
    assert isinstance(template, PromptTemplate)
    prompt = template.build_prompt(attack_text="Attack text.")
    assert "Unmodifiable prefix." in prompt
    assert "Modifiable infix." in prompt
    assert "Attack text." in prompt
    assert "Unmodifiable suffix." in prompt
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_out = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "Unmodifiable prefix.Modifiable infix.Attack text."
                    "Unmodifiable suffix."
                ),
            },
            {"role": "assistant", "content": ""},
        ],
        tokenize=False,
    )
    assert tokenizer_out[: len(prompt)] == prompt
