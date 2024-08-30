import dataclasses

import pytest
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from robust_llm.config.model_configs import GenerationConfig, ModelConfig
from robust_llm.models import WrappedModel


def model_config_factory():
    return ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        revision="main",
        inference_type="classification",
        train_minibatch_size=2,
        eval_minibatch_multiplier=2,
        env_minibatch_multiplier=1,
    )


@pytest.fixture
def wrapped_model():
    config = model_config_factory()
    return WrappedModel.from_config(config, accelerator=None)


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


def test_tokenize(wrapped_model: WrappedModel):
    text = ["Hello, my dog is cute", "Hello, my dog is the cutest dog."]
    wrapped_input_ids = wrapped_model.tokenize(
        text, padding_side="right", return_tensors="pt"
    ).input_ids
    underlying_input_ids = wrapped_model.right_tokenizer(
        text, padding=True, return_tensors="pt"
    ).input_ids
    assert torch.allclose(wrapped_input_ids, underlying_input_ids)


def test_llama():
    cfg = ModelConfig(
        name_or_path="HuggingFaceM4/tiny-random-LlamaForCausalLM",
        family="llama2",
        revision="main",
        inference_type="generation",
        train_minibatch_size=2,
        eval_minibatch_multiplier=2,
        env_minibatch_multiplier=1,
    )
    wrapped_model = WrappedModel.from_config(cfg, accelerator=None)
    assert isinstance(wrapped_model.model, LlamaForCausalLM)
    assert isinstance(wrapped_model.right_tokenizer, LlamaTokenizer)


def test_determinism_single_batch():
    cfg = ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        revision="main",
        inference_type="generation",
        train_minibatch_size=2,
        eval_minibatch_multiplier=2,
        env_minibatch_multiplier=1,
        seed=42,
        generation_config=GenerationConfig(
            max_new_tokens=50, do_sample=True, temperature=10.0
        ),
    )
    assert cfg.generation_config is not None
    model = WrappedModel.from_config(cfg, accelerator=None)
    encoding = model.tokenize("Hello, my dog is cute", return_tensors="pt")
    inputs = {
        "input_ids": encoding.input_ids,
        "attention_mask": encoding.attention_mask,
        "generation_config": model.transformers_generation_config,
    }
    is_equal = model.generate(**inputs) == model.generate(**inputs)
    assert isinstance(is_equal, torch.Tensor)
    assert is_equal.all()

    not_equal = model.model.generate(**inputs) != model.model.generate(**inputs)
    assert isinstance(not_equal, torch.Tensor)
    assert not_equal.any()


def test_determinism_batched():
    cfg = ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        revision="main",
        inference_type="generation",
        train_minibatch_size=2,
        eval_minibatch_multiplier=2,
        env_minibatch_multiplier=1,
        seed=42,
        generation_config=GenerationConfig(
            max_new_tokens=50, do_sample=True, temperature=10.0
        ),
    )
    assert cfg.generation_config is not None
    model = WrappedModel.from_config(cfg, accelerator=None)
    text = ["Hello, my dog is cute", "Greetings from the moon"]
    encoding1 = model.tokenize(text, return_tensors="pt")
    encoding2 = model.tokenize(text[::-1], return_tensors="pt")
    inputs1 = {
        "input_ids": encoding1.input_ids,
        "attention_mask": encoding1.attention_mask,
        "generation_config": model.transformers_generation_config,
    }
    inputs2 = {
        "input_ids": encoding2.input_ids,
        "attention_mask": encoding2.attention_mask,
        "generation_config": model.transformers_generation_config,
    }

    is_equal = model.generate(**inputs1) == model.generate(**inputs2)[[1, 0]]
    assert isinstance(is_equal, torch.Tensor)
    assert is_equal.all()

    model._set_seed()
    simple_out1 = model.model.generate(**inputs1)
    model._set_seed()
    simple_out2 = model.model.generate(**inputs2)
    not_equal = simple_out1 != simple_out2[[1, 0]]
    assert isinstance(not_equal, torch.Tensor)
    assert not_equal.any()


def test_generate_equivalence():
    cfg = ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        revision="main",
        inference_type="generation",
        train_minibatch_size=2,
        eval_minibatch_multiplier=2,
        env_minibatch_multiplier=1,
        seed=42,
        generation_config=GenerationConfig(
            max_new_tokens=50, do_sample=True, temperature=10.0
        ),
    )
    assert cfg.generation_config is not None
    model = WrappedModel.from_config(cfg, accelerator=None)
    text = "Hello, my dog is cute"
    encoding = model.tokenize(text, return_tensors="pt")
    inputs = {
        "input_ids": encoding.input_ids,
        "attention_mask": encoding.attention_mask,
        "generation_config": model.transformers_generation_config,
    }
    model._set_seed()
    simple_out = model.model.generate(**inputs)
    wrapped_out = model.generate(**inputs)
    is_equal = simple_out == wrapped_out
    assert isinstance(is_equal, torch.Tensor)
    assert is_equal.all()

    mb_text = ["Hello, my dog is cute", "Greetings from the moon"]
    mb_encoding = model.tokenize(mb_text, return_tensors="pt")
    mb_inputs = {
        "input_ids": mb_encoding.input_ids,
        "attention_mask": mb_encoding.attention_mask,
        "generation_config": model.transformers_generation_config,
    }
    assert model.generate(**mb_inputs).shape == model.generate(**mb_inputs).shape
