"""Utility functions for working with torch.distributed/accelerate."""

from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

INT_TO_DTYPE = {
    0: torch.float32,
    1: torch.float64,
    2: torch.float16,
    3: torch.int8,
    4: torch.uint8,
    5: torch.int16,
    6: torch.int32,
    7: torch.int64,
    8: torch.bool,
}
DTYPE_TO_INT = {v: k for k, v in INT_TO_DTYPE.items()}
BIT_GENERATOR = "PCG64"


def is_main_process():
    """Find out if we are the main process without passing in an Accelerator object."""
    accelerator = Accelerator()
    return accelerator.is_main_process


def broadcast_list_of_bools(
    data: list[bool] | None, accelerator: Accelerator
) -> list[bool]:
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, list)
        return data
    if is_main_process():
        assert data is not None
        assert all(isinstance(x, bool) for x in data)
        tensor_input_data = torch.tensor(
            data, dtype=torch.bool, device=accelerator.device
        )
    else:
        tensor_input_data = None
    tensor_data = broadcast_tensor(tensor_input_data, accelerator)
    return tensor_data.tolist()


def broadcast_list_of_floats(
    data: list[float] | None, accelerator: Accelerator
) -> list[float]:
    """Broadcasts floats, rounding to 32-bit precision."""
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, list)
        return data
    if is_main_process():
        assert data is not None
        assert all(isinstance(x, float) for x in data)
        tensor_input_data = torch.tensor(
            data, dtype=torch.float32, device=accelerator.device
        )
    else:
        tensor_input_data = None
    tensor_data = broadcast_tensor(tensor_input_data, accelerator)
    return tensor_data.tolist()


def broadcast_list_of_ints(
    data: list[int] | None,
    accelerator: Accelerator,
) -> list[int]:
    """Broadcasts ints, asserting 32-bit precision."""
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, list)
        return data
    if is_main_process():
        assert data is not None
        assert all(isinstance(x, int) and x.bit_length() <= 32 for x in data)
        tensor_input_data = torch.tensor(
            data, dtype=torch.int32, device=accelerator.device
        )
    else:
        tensor_input_data = None
    tensor_data = broadcast_tensor(tensor_input_data, accelerator)
    return tensor_data.tolist()


def broadcast_list_of_longs(
    data: list[int] | None,
    accelerator: Accelerator,
) -> list[int]:
    """Broadcasts ints, asserting 64-bit precision."""
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, list)
        return data
    if is_main_process():
        assert data is not None
        assert all(isinstance(x, int) and x.bit_length() <= 64 for x in data)
        tensor_input_data = torch.tensor(
            data, dtype=torch.int64, device=accelerator.device
        )
    else:
        tensor_input_data = None
    tensor_data = broadcast_tensor(tensor_input_data, accelerator)
    return tensor_data.tolist()


def broadcast_int(
    data: int | None,
    accelerator: Accelerator,
) -> int:
    """Broadcasts int, rounding to 32-bit precision."""
    return broadcast_list_of_ints([data] if data is not None else None, accelerator)[0]


def broadcast_long(
    data: int | None,
    accelerator: Accelerator,
) -> int:
    """Broadcasts int, rounding to 64-bit precision."""
    return broadcast_list_of_longs([data] if data is not None else None, accelerator)[0]


def split_int128(n: int) -> list[int]:
    """Split a 128-bit integer into a list of four int32 values."""
    mask = (1 << 32) - 1  # Mask for 32 bits
    return [n & mask, (n >> 32) & mask, (n >> 64) & mask, (n >> 96) & mask]


def reconstruct_int128(int_list: list[int]) -> int:
    """Reconstruct a 128-bit integer from a list of four int32 values."""
    assert len(int_list) == 4
    return int_list[0] | (int_list[1] << 32) | (int_list[2] << 64) | (int_list[3] << 96)


def broadcast_int128(
    data: int | None,
    accelerator: Accelerator,
) -> int:
    """Broadcasts a 128-bit integer by splitting into list[int32]."""
    if not dist.is_initialized():
        assert isinstance(data, list)
        return data

    if is_main_process():
        assert data is not None
        assert isinstance(data, int) and data.bit_length() <= 128
        split_data = split_int128(data)
    else:
        split_data = None

    broadcasted_data = broadcast_list_of_longs(split_data, accelerator)
    return reconstruct_int128(broadcasted_data)


def broadcast_float(
    data: float | None,
    accelerator: Accelerator,
) -> float:
    """Broadcasts float, rounding to 32-bit precision."""
    return broadcast_list_of_floats([data] if data is not None else None, accelerator)[
        0
    ]


def broadcast_rng_state(
    data: dict[str, Any] | None,
    accelerator: Accelerator,
) -> dict[str, Any]:
    """Broadcasts the state of the RNG from the main process"""
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, dict)
        return data
    return {
        "bit_generator": BIT_GENERATOR,
        "state": {
            "state": broadcast_int128(
                data["state"]["state"] if data is not None else None, accelerator
            ),
            "inc": broadcast_int128(
                data["state"]["inc"] if data is not None else None, accelerator
            ),
        },
        "has_uint32": broadcast_int(
            data["has_uint32"] if data is not None else None, accelerator
        ),
        "uinteger": broadcast_long(
            data["uinteger"] if data is not None else None, accelerator
        ),
    }


def broadcast_tensor(
    data: torch.Tensor | None, accelerator: Accelerator
) -> torch.Tensor:
    """Broadcast the data in 'data' to all processes in the group.

    We can't just broadcast the data because we need an empty tensor of the
    right shape on the other devices to broadcast into.
    """
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, torch.Tensor)
        return data

    # First we need to broadcast the dimension of the tensor.
    if is_main_process():
        assert isinstance(data, torch.Tensor)
        ndims = torch.tensor(data.ndim, device=accelerator.device)
    else:
        ndims = torch.tensor(-1, device=accelerator.device)
    dist.broadcast(ndims, src=0)

    # Now we need to broadcast the actual shape.
    if is_main_process():
        assert isinstance(data, torch.Tensor)
        shape = torch.tensor(data.shape, device=accelerator.device, dtype=torch.int32)
    else:
        shape = torch.empty(
            int(ndims.item()), device=accelerator.device, dtype=torch.int32
        )
    dist.broadcast(shape, src=0)

    # Then we need to broadcast the datatype.
    data_dtype = broadcast_dtype(data, accelerator)

    # Finally, broadcast the data itself.
    if is_main_process():
        assert data is not None
        data = data.to(accelerator.device)
    else:
        data = torch.empty(*shape.tolist(), dtype=data_dtype, device=accelerator.device)
    dist.broadcast(data, src=0)
    assert isinstance(data, torch.Tensor)
    return data


def broadcast_dtype(data: torch.Tensor | None, accelerator: Accelerator):
    """Broadcast the datatype of a tensor between ranks"""
    if is_main_process():
        assert isinstance(data, torch.Tensor)
        dtype = torch.tensor(DTYPE_TO_INT[data.dtype], device=accelerator.device)
    else:
        dtype = torch.tensor(-1, device=accelerator.device)
    dist.broadcast(dtype, src=0)
    return INT_TO_DTYPE[int(dtype.item())]


class DistributedRNG:

    def __init__(self, seed: int | None, accelerator: Accelerator | None) -> None:
        self._rng = np.random.default_rng(seed) if is_main_process() else None
        self.accelerator = accelerator if accelerator is not None else Accelerator()
        self.skip_broadcast = self.accelerator.device == "cpu"
        assert (
            self._rng is None
            or self._rng.bit_generator.state["bit_generator"] == BIT_GENERATOR
        )

    def randint(self, a: int, b: int) -> int:
        i = (
            int(self._rng.integers(a, b, endpoint=True))
            if self._rng is not None
            else None
        )
        if self.skip_broadcast:
            assert isinstance(i, int)
            return i
        return broadcast_int(i, self.accelerator)

    def random(self) -> float:
        f = float(self._rng.random()) if self._rng is not None else None
        if self.skip_broadcast:
            assert isinstance(f, float)
            return f
        return broadcast_float(f, self.accelerator)

    def choice(
        self,
        seq: int | list[int],
        size: int,
        replace: bool = False,
        p: list[float] | None = None,
    ) -> list[int]:
        if isinstance(seq, int):
            seq = list(range(seq))
        array = (
            self._rng.choice(seq, size=size, replace=replace, p=p).tolist()
            if self._rng is not None
            else None
        )
        if self.skip_broadcast:
            assert isinstance(array, list)
            return array
        return broadcast_list_of_ints(array, self.accelerator)

    def sample(self, seq: list[Any], size: int) -> list[Any]:
        indices = self.choice(list(range(len(seq))), size=size, replace=False)
        return [seq[i] for i in indices]

    def getstate(self) -> dict[str, Any]:
        state = self._rng.bit_generator.state if self._rng is not None else None
        assert state is None or isinstance(state, dict)
        if self.skip_broadcast:
            assert isinstance(state, dict)
            return state
        return broadcast_rng_state(state, self.accelerator)

    def setstate(self, state: dict[str, Any] | None) -> None:
        if self._rng is not None and state is not None:
            self._rng.bit_generator.state = state


def pad_batch_across_processes(
    tokenizer: PreTrainedTokenizerBase, accelerator: Accelerator, batch: dict
) -> dict:
    """Pad a batch across processes.

    We have to do this because otherwise we can end up with different length sequences
    across processes, which will cause a hang.
    """
    pad_token_id = tokenizer.pad_token_id
    assert pad_token_id is not None
    input_ids = accelerator.pad_across_processes(
        batch["input_ids"], pad_index=pad_token_id, dim=1
    )
    attention_mask = accelerator.pad_across_processes(
        batch["attention_mask"], pad_index=0, dim=1
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": batch["labels"],
    }
