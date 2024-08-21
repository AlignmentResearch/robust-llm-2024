"""Utility functions for working with torch.distributed/accelerate."""

import torch
import torch.distributed as dist
from accelerate import Accelerator

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


def broadcast_list_of_bools(
    data: list[bool] | None, accelerator: Accelerator
) -> list[bool]:
    # If we're not using distributed training, we don't need to do anything.
    if not dist.is_initialized():
        assert isinstance(data, list)
        return data
    if accelerator.is_main_process:
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
    if accelerator.is_main_process:
        assert data is not None
        assert all(isinstance(x, float) for x in data)
        tensor_input_data = torch.tensor(
            data, dtype=torch.float32, device=accelerator.device
        )
    else:
        tensor_input_data = None
    tensor_data = broadcast_tensor(tensor_input_data, accelerator)
    return tensor_data.tolist()


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
    if accelerator.is_main_process:
        assert isinstance(data, torch.Tensor)
        ndims = torch.tensor(data.ndim, device=accelerator.device)
    else:
        ndims = torch.tensor(-1, device=accelerator.device)
    dist.broadcast(ndims, src=0)

    # Now we need to broadcast the actual shape.
    if accelerator.is_main_process:
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
    if accelerator.is_main_process:
        assert data is not None
        data = data.to(accelerator.device)
    else:
        data = torch.empty(*shape.tolist(), dtype=data_dtype, device=accelerator.device)
    dist.broadcast(data, src=0)
    assert isinstance(data, torch.Tensor)
    return data


def broadcast_dtype(data: torch.Tensor | None, accelerator: Accelerator):
    """Broadcast the datatype of a tensor between ranks"""
    if accelerator.is_main_process:
        assert isinstance(data, torch.Tensor)
        dtype = torch.tensor(DTYPE_TO_INT[data.dtype], device=accelerator.device)
    else:
        dtype = torch.tensor(-1, device=accelerator.device)
    dist.broadcast(dtype, src=0)
    return INT_TO_DTYPE[int(dtype.item())]
