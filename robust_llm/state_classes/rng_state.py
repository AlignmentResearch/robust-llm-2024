from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate import Accelerator

from robust_llm.dist_utils import DistributedRNG


@dataclass
class RNGState:
    """The state of the random number generators.

    Contains two types:
    - '_rng_state's, which tracks some global state.
    - '_rng's, which can be used to generate random numbers.
    """

    torch_rng_state: torch.Tensor
    distributed_rng: DistributedRNG

    def update_states(self) -> RNGState:
        return RNGState(
            torch_rng_state=torch.random.get_rng_state(),
            distributed_rng=self.distributed_rng,
        )

    def set_random_states(self):
        torch.random.set_rng_state(self.torch_rng_state)

    def save(self, path: Path, process_index: int, accelerator: Accelerator):
        rng_path = path / "rng"
        rng_path.mkdir(exist_ok=True)
        torch.save(
            self.torch_rng_state, rng_path / f"torch_rng_state_{process_index}.pt"
        )
        dist_rng_state = self.distributed_rng.getstate()
        torch.save(
            dist_rng_state, rng_path / f"distributed_rng_state_{process_index}.pt"
        )

    @staticmethod
    def load(path: Path, process_index: int, accelerator: Accelerator) -> RNGState:
        rng_path = path / "rng"
        if accelerator.is_main_process:
            dist_rng_state = torch.load(
                rng_path / f"distributed_rng_state_{process_index}.pt"
            )
        else:
            dist_rng_state = None
        dist_rng = DistributedRNG(seed=0, accelerator=accelerator)
        dist_rng.setstate(dist_rng_state)

        return RNGState(
            torch_rng_state=torch.load(
                rng_path / f"torch_rng_state_{process_index}.pt"
            ),
            distributed_rng=dist_rng,
        )
