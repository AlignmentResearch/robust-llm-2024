from contextlib import contextmanager

import torch

from robust_llm.models.model_utils import PastKeyValues
from robust_llm.models.wrapped_model import WrappedModel


class CachingWrappedModel(WrappedModel):
    """Thin wrapper around a WrappedModel for KV caching.

    We instantiate this class with a WrappedModel and an empty cache.
    The cache maps sequences of token_ids to their KV values.

    The cache is currently populated manually by calling add_to_cache
    TODO(GH#401): Make this work better for multi-prompt attacks.

    Attributes:
    _wrapped_model (WrappedModel): The original model that this class wraps.
    cache (dict[tuple[int], PastKeyValues]): The dictionary keys are tuples of
        token IDs representing input sequences, and the dictionary values are
        PastKeyValues objects that contain the cached key-value pairs for these
        sequences.
    """

    def __init__(self, wrapped_model: WrappedModel):
        self._wrapped_model = wrapped_model
        self.cache: dict[tuple[int], PastKeyValues] = {}

    def __getattr__(self, name):
        """If an attribute is not found in this class, look in the WrappedModel."""
        return getattr(self._wrapped_model, name)

    def __setattr__(self, name, value):
        """Set all attributes except _wrapped_model and cache in the WrappedModel."""
        if name in ["_wrapped_model", "cache"]:
            super().__setattr__(name, value)
        else:
            setattr(self._wrapped_model, name, value)

    def __call__(self, **inputs):
        return self.forward(**inputs)

    def forward(self, **inputs):
        if "input_ids" not in inputs:
            raise ValueError("CachingWrappedModel requires input_ids.")
        input_ids = inputs["input_ids"]
        cache_result = self.return_cache_overlap(input_ids)
        if cache_result is not None:
            prefix, overlapping_cached_kv = cache_result

            if len(prefix) == input_ids.shape[1]:
                raise ValueError(
                    "Cache hit is the entire input. Unfortunately this throws an"
                    " error inside PreTrainedModel. Luckily we shouldn't need this"
                    " use-case since we'll always be appending an attack."
                )

            inputs["input_ids"] = input_ids[:, len(prefix) :]
            # If we have inputs_embeds, we need to truncate them as well.
            if "inputs_embeds" in inputs:
                inputs["inputs_embeds"] = inputs["inputs_embeds"][:, len(prefix) :]

            inputs["past_key_values"] = overlapping_cached_kv
        return self._wrapped_model.forward(**inputs)

    @torch.no_grad()
    def add_to_cache(self, input_ids: torch.Tensor) -> None:
        """Add to the KV cache for the given token_ids.

        The input should contain no padding tokens.

        We use torch.no_grad because having gradients in the cache is
        unnecessary and causes issues with trying to backprop twice.
        """
        # Preconditions.
        assert input_ids.shape[0] == 1, "Only one input sequence at a time (for now)."
        assert self._wrapped_model.tokenizer.pad_token_id not in input_ids

        outputs = self._wrapped_model(input_ids=input_ids, use_cache=True)
        kv_cache = outputs.past_key_values
        # We access the first element of the batch, which is the only one.
        self.cache[tuple(input_ids[0].tolist())] = kv_cache

    def return_cache_overlap(
        self, input_ids: torch.Tensor
    ) -> tuple[tuple[int, ...], PastKeyValues] | None:
        """Return as much of the cache as possible for the given token_ids.

        We look for the longest prefix of the input_ids that is also a prefix
        of something in the cache. We return the corresponding KV values.

        Because ultimately the KV cache consists of tensors, we need to return
        the same number of tokens for each tensor in the cache. This means we
        find a common prefix of *all* the sequences in input_ids, and then
        check for a cache hit with that.

        Note: In the body of this function, we use "query" to refer to the
        input_ids used to search the cache, and "key" to refer to the token_ids
        in the cache; they are not related to attention.
        """

        query = get_common_prefix_for_batch(input_ids)

        # Find the key with the longest common prefix with the query, and
        # its length.
        longest_common_prefix_key = None
        longest_common_prefix_length = -1
        for key in self.cache.keys():
            common_prefix = get_common_prefix(query, key)
            if len(common_prefix) > longest_common_prefix_length:
                longest_common_prefix_key = key
                longest_common_prefix_length = len(common_prefix)

        # If there is no cache hit, return None.
        if longest_common_prefix_length <= 0:
            return None
        assert longest_common_prefix_key is not None

        # Return the KV values corresponding to the longest common prefix.
        cached_kv = self.cache[longest_common_prefix_key]
        overlapping_cached_kv = get_overlapping_kv(
            cached_kv, longest_common_prefix_length
        )
        final_kv = repeat_kv_for_batch(overlapping_cached_kv, input_ids.shape[0])

        longest_common_prefix = longest_common_prefix_key[:longest_common_prefix_length]
        return longest_common_prefix, final_kv

    @classmethod
    def load_tokenizer(cls, model_config):
        raise NotImplementedError("Does not need load_tokenizer.")


def get_common_prefix(input_ids: torch.Tensor, key: tuple[int]) -> torch.Tensor:
    """Return the longest common prefix of the input_ids and the key.

    NOTE: input_ids and key could be different lengths.
    """
    # Preconditions.
    assert len(input_ids.shape) == 1, "Input should be a single sequence."

    prefix = []
    for i, token_id in enumerate(key):
        if i >= len(input_ids) or input_ids[i] != token_id:
            break
        prefix.append(token_id)
    return torch.tensor(prefix, dtype=torch.long)


def get_common_prefix_for_batch(input_ids: torch.Tensor) -> torch.Tensor:
    """Return the longest common prefix of all sequences in the batch of input_ids.

    Args:
        input_ids: Tensor of shape (batch_size, sequence_length) with token ids.

    Returns:
        prefix: Tensor of shape (prefix_length,) containing the common prefix.
    """

    # Preconditions.
    assert len(input_ids.shape) == 2, "Input should be a batch of sequences."
    assert input_ids.shape[0] > 0, "Input should not be empty."

    # Tensor of booleans indicating where each sequence agrees and disagrees
    # with input_ids[0].
    boolean_mask = input_ids != input_ids[0].unsqueeze(0)
    # Look across the sequences and check if any of them diverge from the first.
    first_divergence_points = boolean_mask.any(dim=0)
    # Find the indices of the divergence points.
    divergence_indices = torch.nonzero(first_divergence_points).squeeze(dim=1)
    # If there are no divergence points then all sequences are the same; return
    # the first one arbitrarily.
    if len(divergence_indices) == 0:
        return input_ids[0]
    divergence_ix = divergence_indices[0].item()
    assert isinstance(divergence_ix, int)
    prefix = input_ids[0, :divergence_ix]

    # Postconditions.
    assert len(prefix.shape) == 1, "Prefix should be a 1D tensor."
    assert len(prefix) <= input_ids.shape[1], "Prefix should be shorter than inputs."

    return prefix


def get_overlapping_kv(kv: PastKeyValues, length: int) -> PastKeyValues:
    """Return the KV values corresponding to the first 'length' tokens.

    Because PastKeyValues is a tuple of tuples of tensors, it's a little
    tricky to shorten.
    """

    def shorten(tensor: torch.Tensor) -> torch.Tensor:
        """Shorten a tensor to the first 'length' tokens."""
        # Sequence length is dimension 2.
        return tensor[:, :, :length, :]

    return tuple(tuple(shorten(tensor) for tensor in layer) for layer in kv)


def repeat_kv_for_batch(kv: PastKeyValues, batch_size: int) -> PastKeyValues:
    """Repeat the KV values for a batch of sequences.

    This is useful when we want to use the same KV values for multiple sequences.
    """
    # Preconditions.
    assert batch_size > 0, "Batch size should be positive."
    assert len(kv[0][0].shape) == 4, "KV values should have 4 dimensions."
    assert kv[0][0].shape[0] == 1, "Initial KV values should have batch size 1."

    def repeat(tensor: torch.Tensor) -> torch.Tensor:
        """Repeat a tensor for a batch of sequences."""
        # Dimension 0 is the batch size.
        return tensor.repeat(batch_size, 1, 1, 1)

    return tuple(tuple(repeat(tensor) for tensor in layer) for layer in kv)


@contextmanager
def get_caching_model_with_example(model: WrappedModel, example_text: str):
    """Get a CachingWrappedModel with a single example already cached.

    Note that we can add more examples to the cache later.
    """
    caching_model = CachingWrappedModel(model)
    inp = caching_model.tokenizer(example_text, return_tensors="pt")
    input_ids = inp["input_ids"]
    caching_model.add_to_cache(input_ids)  # type: ignore[arg-type]
    try:
        yield caching_model
    finally:
        del caching_model
    return
