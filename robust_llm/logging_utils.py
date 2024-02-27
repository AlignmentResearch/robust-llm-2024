from __future__ import annotations

from dataclasses import dataclass

import wandb


@dataclass
class LoggingCounter:
    """Class for logging precise step and datapoint counts.

    Currently only supports victim training.
    """

    _name: str
    _step_count: int = 0
    _datapoint_count: int = 0
    _is_global: bool = False

    def __post_init__(self) -> None:
        self._parent = None
        if not self._is_global:
            self._parent = GLOBAL_LOGGING_COUNTER

    def increment(self, step_count_to_add: int, datapoint_count_to_add: int) -> None:
        if self._parent is not None:
            self._parent.increment(step_count_to_add, datapoint_count_to_add)

        self._step_count += step_count_to_add
        self._datapoint_count += datapoint_count_to_add
        self._log_to_wandb()

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def datapoint_count(self) -> int:
        return self._datapoint_count

    def _log_to_wandb(self) -> None:
        assert wandb.run is not None
        wandb.log(
            {
                f"{self._name}_step_count": self._step_count,
                f"{self._name}_datapoint_count": self._datapoint_count,
            },
        )


GLOBAL_LOGGING_COUNTER = LoggingCounter(_name="global", _is_global=True)
