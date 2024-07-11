import pytest

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.wrapped_chat_model import WrappedChatModel


def model_config_factory():
    return ModelConfig(
        name_or_path="Felladrin/Pythia-31M-Chat-v1",
        family="pythia-chat",
        revision="main",
        inference_type="generation",
        train_minibatch_size=2,
        eval_minibatch_size=3,
        minibatch_multiplier=1,
    )


@pytest.fixture
def wrapped_chat_model() -> WrappedChatModel:
    config = model_config_factory()
    return WrappedChatModel.from_config(config, accelerator=None)


def test_maybe_apply_chat_template(wrapped_chat_model):
    wrapped_chat_model.system_prompt = "You are a chatbot in a unit test environment."
    input_text = "Hello, how are you?"
    chat_formatted = wrapped_chat_model.maybe_apply_chat_template(input_text)
    assert chat_formatted == (
        "<|im_start|>system\nYou are a chatbot in a unit test environment.<|im_end|>\n"
        "<|im_start|>user\nHello, how are you?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_maybe_apply_chat_template_no_system(wrapped_chat_model):
    wrapped_chat_model.system_prompt = None
    input_text = "Hello, how are you?"
    chat_formatted = wrapped_chat_model.maybe_apply_chat_template(input_text)
    assert chat_formatted == (
        "<|im_start|>user\nHello, how are you?<|im_end|>\n" "<|im_start|>assistant\n"
    )
