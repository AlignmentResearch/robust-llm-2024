# %%
import torch

from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.caching_wrapped_model import CachingWrappedModel
from robust_llm.models.wrapped_model import WrappedModel

# %%


def test_unbatched_kv_caching():
    model_config = ModelConfig(
        name_or_path="gpt2",
        family="gpt2",
        inference_type="classification",
    )
    wrapped_model = WrappedModel.from_config(model_config, accelerator=None)
    caching_model = CachingWrappedModel(wrapped_model)

    # Set up cache.
    cache_sequence = "Hello, my dog is cute."
    cache_inp = caching_model.tokenizer(cache_sequence, return_tensors="pt")
    cache_ids = cache_inp["input_ids"]
    assert isinstance(cache_ids, torch.Tensor)
    caching_model.add_to_cache(cache_ids)

    # Evaluate cached and uncached on a sequence that is a superset of the cache.
    super_sequence = cache_sequence + " I like to play with him."
    super_inp = caching_model.tokenizer(super_sequence, return_tensors="pt")

    uncached_super_logits = wrapped_model(**super_inp).logits
    cached_super_logits = caching_model(**super_inp).logits

    assert torch.allclose(uncached_super_logits, cached_super_logits)

    # Evaluate cached and uncached on a sequence that diverges from the cache.
    diverging_sequence = "Hello, my dog is fun. I like to play with him."
    diverging_inp = caching_model.tokenizer(diverging_sequence, return_tensors="pt")

    uncached_diverging_logits = wrapped_model(**diverging_inp).logits
    cached_diverging_logits = caching_model(**diverging_inp).logits

    assert torch.allclose(uncached_diverging_logits, cached_diverging_logits)


def test_batched_kv_caching():
    model_config = ModelConfig(
        name_or_path="gpt2",
        family="gpt2",
        inference_type="classification",
    )
    wrapped_model = WrappedModel.from_config(model_config, accelerator=None)
    caching_model = CachingWrappedModel(wrapped_model)

    # Set up cache.
    cache_sequence = "Hello, my dog is cute."
    cache_inp = caching_model.tokenizer(cache_sequence, return_tensors="pt")
    cache_ids = cache_inp["input_ids"]
    assert isinstance(cache_ids, torch.Tensor)
    caching_model.add_to_cache(cache_ids)

    # Evaluate cached and uncached on two sequences that diverge before the end
    # of the cache and have different lengths.
    super_sequence = cache_sequence + " I like to play with him in the park on Sundays."
    diverging_sequence = "Hello, my dog is fun. I like to play with him."
    batched_sequences = [super_sequence, diverging_sequence]
    batched_inp = caching_model.tokenizer(
        batched_sequences, return_tensors="pt", padding=True
    )

    uncached_batched_logits = wrapped_model(**batched_inp).logits
    cached_batched_logits = caching_model(**batched_inp).logits
    assert torch.allclose(uncached_batched_logits, cached_batched_logits)


def test_wrapping_attributes():
    model_config = ModelConfig(
        name_or_path="gpt2",
        family="gpt2",
        inference_type="classification",
    )
    wrapped_model = WrappedModel.from_config(model_config, accelerator=None)
    caching_model = CachingWrappedModel(wrapped_model)

    # Testing access passes through to underlying model
    assert caching_model.device == wrapped_model.device

    # Testing assignment passes through to underlying model
    caching_model.new_attribute = "new_attribute"
    assert hasattr(wrapped_model, "new_attribute")
    assert wrapped_model.new_attribute == "new_attribute"  # type: ignore[attr-defined]
