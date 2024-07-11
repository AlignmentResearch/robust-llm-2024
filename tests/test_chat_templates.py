from typing import Callable

import pytest
from transformers import AutoTokenizer

from robust_llm.models.prompt_templates import (
    PromptTemplate,
    get_gemma_template,
    get_llama_2_template,
    get_llama_3_template,
    get_qwen_template,
    get_tinyllama_template,
)

NAME_AND_TEMPLATE = [
    ("neuralmagic/Meta-Llama-3-8B-Instruct-FP8", get_llama_3_template),
    ("meta-llama/Llama-2-7b-chat-hf", get_llama_2_template),
    ("Qwen/Qwen1.5-1.8B-Chat", get_qwen_template),
    ("Qwen/Qwen2-7B-Instruct", get_qwen_template),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", get_tinyllama_template),
]
# Gemma models do not support system prompts.
NAME_AND_TEMPLATE_NO_SYSTEM_PROMPT = NAME_AND_TEMPLATE + [
    ("google/gemma-1.1-2b-it", get_gemma_template),
    ("google/gemma-2-9b-it", get_gemma_template),
]


@pytest.mark.parametrize("model_name, template_constructor", NAME_AND_TEMPLATE)
def test_template_with_system_prompt(model_name: str, template_constructor: Callable):
    template = template_constructor(
        "Unmodifiable prefix.",
        "Modifiable infix.",
        "Unmodifiable suffix.",
        "System prompt.",
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
    "model_name, template_constructor", NAME_AND_TEMPLATE_NO_SYSTEM_PROMPT
)
def test_template_without_system_prompt(
    model_name: str, template_constructor: Callable
):
    template = template_constructor(
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
