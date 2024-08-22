import pytest
import torch
from accelerate import Accelerator

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models import WrappedModel
from robust_llm.models.model_utils import build_dataloader

FORWARD_FLOP_COUNT = 28135424


@pytest.fixture
def model_config():
    return ModelConfig(
        name_or_path="EleutherAI/pythia-14m",
        family="pythia",
        revision="main",
        inference_type="generation",
        train_minibatch_size=2,
        eval_minibatch_size=3,
        minibatch_multiplier=1,
    )


@pytest.fixture
def wrapped_model(model_config):
    return WrappedModel.from_config(model_config, accelerator=None)


def test_flop_tracked_model_initialization(wrapped_model: WrappedModel):
    assert wrapped_model.flop_count == 0


def test_flop_tracked_model_compute_flops(wrapped_model: WrappedModel):
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}
    flops = wrapped_model.compute_flops(input_dict)
    assert flops == FORWARD_FLOP_COUNT


def test_flop_tracked_model_update_flop_count(wrapped_model: WrappedModel):
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}
    wrapped_model.update_flop_count(input_dict)
    assert wrapped_model.flop_count == FORWARD_FLOP_COUNT


def test_flop_count_context(wrapped_model: WrappedModel):
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}

    with wrapped_model.flop_count_context() as context:
        wrapped_model(**input_dict)

    assert context.flops > 0
    assert context.start_flops == 0
    assert context.end_flops == FORWARD_FLOP_COUNT
    assert context.flops == FORWARD_FLOP_COUNT


def test_nested_flop_count_contexts(wrapped_model: WrappedModel):
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}

    with wrapped_model.flop_count_context() as outer_context:
        wrapped_model(**input_dict)

        with wrapped_model.flop_count_context() as inner_context:
            wrapped_model(**input_dict)

    assert outer_context.flops > inner_context.flops
    assert inner_context.flops > 0


def test_backward_flop_count_increment(wrapped_model: WrappedModel):
    wrapped_model.train()
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}
    out = wrapped_model(**input_dict)
    forward_flop_count = wrapped_model.flop_count
    assert forward_flop_count == FORWARD_FLOP_COUNT
    loss = out.logits.sum()
    loss.backward()
    assert wrapped_model.flop_count > forward_flop_count
    assert wrapped_model.flop_count == 84406272


def test_call_count_increment(wrapped_model: WrappedModel):
    wrapped_model.train()
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}
    assert wrapped_model.n_forward_calls == 0
    out = wrapped_model(**input_dict)
    assert wrapped_model.n_forward_calls == 1
    loss = out.logits.sum()
    assert wrapped_model.n_backward_calls == 0
    loss.backward()
    assert wrapped_model.n_backward_calls == 1


def test_input_shapes(wrapped_model: WrappedModel):
    wrapped_model.train()
    input_dict = {"input_ids": torch.randint(0, 1024, (1, 1))}
    assert wrapped_model.n_forward_calls == 0
    wrapped_model(**input_dict)
    assert wrapped_model.input_shapes == [(1, 1)]


# TODO: run this with multi-GPU
@pytest.mark.parametrize("minibatch_size", [1, 2, 3, 4])
def test_flops_with_dataloader(minibatch_size: int, model_config: ModelConfig):
    accelerator = Accelerator()
    wrapped_model = WrappedModel.from_config(model_config, accelerator=accelerator)
    wrapped_model.train()
    text = ["Hello, how are you?", "What's the craic?"]
    tokenized = wrapped_model.tokenize(
        text,
        return_tensors="pt",
        padding_side="left",
    )
    dataloader = build_dataloader(minibatch_size=minibatch_size, **tokenized)
    dataloader = accelerator.prepare(dataloader)
    for batch in dataloader:
        out = wrapped_model(**batch)
        loss = out.logits.sum()
        loss.backward()
    assert wrapped_model.flop_count == 506437632 * 2
