"""
Possible methods to test:

 '_append_to_inputs',
 '_check_for_padding_tokens',
 '_prepend_to_inputs',
 '_registry',
 'add_accelerator',
 'call_model',
 'can_generate',
 'config',
 'decode_tokens',
 'device',
 'eval',
 'forward',
 'from_config',
 'get_embedding_weights',
 'get_embeddings',
 'get_tokens',
 'load_tokenizer',
 'register_subclass',
 'to',
 'vocab_size'
 """

import dataclasses

import pytest
import torch
from accelerate import Accelerator
from transformers import LlamaForCausalLM, LlamaTokenizer

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models import WrappedModel


def model_config_factory():
    return ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        revision="main",
        inference_type="classification",
    )


@pytest.fixture
def wrapped_model():
    config = model_config_factory()
    return WrappedModel.from_config(config, accelerator=None)


def test_can_generate():
    config = model_config_factory()
    clf_wrapped_model = WrappedModel.from_config(config, accelerator=None)
    assert not clf_wrapped_model.can_generate()

    gen_config = dataclasses.replace(config, inference_type="generation")
    gen_wrapped_model = WrappedModel.from_config(gen_config, accelerator=None)
    assert gen_wrapped_model.can_generate()


def test_add_accelerator(wrapped_model: WrappedModel):
    assert wrapped_model.accelerator is None
    assert wrapped_model.model.device == torch.device("cpu")
    assert not hasattr(wrapped_model.model, "_is_accelerate_prepared")

    accelerator = Accelerator()
    wrapped_model.add_accelerator(accelerator)
    assert wrapped_model.accelerator == accelerator
    assert wrapped_model.model.device.type == accelerator.device.type
    assert hasattr(wrapped_model.model, "_is_accelerate_prepared")


def test_strict_load():
    model_config = model_config_factory()
    strict_config = dataclasses.replace(model_config, strict_load=True)

    # This should raise an error because the base model has no classification
    # head so it'll be randomly initialized.
    with pytest.raises(AssertionError):
        WrappedModel.from_config(strict_config, accelerator=None)


def test_device(wrapped_model: WrappedModel):
    original_device = wrapped_model.device

    wrapped_model.to(torch.device("cpu"))
    assert wrapped_model.device == torch.device("cpu")

    wrapped_model.to(original_device)
    assert wrapped_model.device == original_device


def test_eval_train(wrapped_model: WrappedModel):
    wrapped_model.train()
    assert wrapped_model.model.training

    wrapped_model.eval()
    assert not wrapped_model.model.training


def test_get_embeddings(wrapped_model: WrappedModel):
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    wrapped_embeddings = wrapped_model.get_embeddings(input_ids)
    underlying_embeddings = wrapped_model.model.get_input_embeddings()(input_ids)
    assert torch.allclose(wrapped_embeddings, underlying_embeddings)


def test_forward(wrapped_model: WrappedModel):
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    wrapped_output = wrapped_model(input_ids=input_ids)
    underlying_output = wrapped_model.model(input_ids=input_ids)
    assert torch.allclose(wrapped_output.logits, underlying_output.logits)


def test_get_tokens(wrapped_model: WrappedModel):
    text = ["Hello, my dog is cute"]
    wrapped_input_ids = wrapped_model.get_tokens(text)
    underlying_input_ids = wrapped_model.tokenizer(text, return_tensors="pt").input_ids
    assert torch.allclose(wrapped_input_ids, underlying_input_ids)


def test_llama():
    cfg = ModelConfig(
        name_or_path="HuggingFaceM4/tiny-random-LlamaForCausalLM",
        family="llama2",
        revision="main",
        inference_type="generation",
    )
    wrapped_model = WrappedModel.from_config(cfg, accelerator=None)
    assert isinstance(wrapped_model.model, LlamaForCausalLM)
    assert isinstance(wrapped_model.tokenizer, LlamaTokenizer)
