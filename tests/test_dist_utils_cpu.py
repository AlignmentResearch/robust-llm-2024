"""Tests for dist_utils.py, which contains utilities for working with multiple gpus.

Ideally we would test these *on* multiple gpus, but for now I'll test that they
don't mess with the data when not using distributed training.
"""

from copy import deepcopy

import torch
from accelerate import Accelerator
from hypothesis import given
from hypothesis import strategies as st

from robust_llm.dist_utils import (
    broadcast_list_of_bools,
    broadcast_list_of_floats,
    broadcast_tensor,
)


@given(
    tensor_dims=st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=5)
)
def test_broadcast_tensor(tensor_dims: list[int]):
    tensor = torch.randn(*tensor_dims)
    tensor_backup = tensor.clone()
    accelerator = Accelerator()
    tensor = broadcast_tensor(tensor, accelerator=accelerator)
    assert torch.allclose(tensor, tensor_backup)


@given(bools=st.lists(st.booleans(), min_size=0, max_size=100))
def test_broadcast_list_of_bools(bools: list[bool]):
    bools_backup = deepcopy(bools)
    accelerator = Accelerator()
    bools = broadcast_list_of_bools(bools, accelerator=accelerator)
    assert bools == bools_backup


@given(floats=st.lists(st.floats(), min_size=0, max_size=100))
def test_broadcast_list_of_floats(floats: list[float]):
    floats_backup = deepcopy(floats)
    accelerator = Accelerator()
    floats = broadcast_list_of_floats(floats, accelerator=accelerator)
    assert floats == floats_backup
