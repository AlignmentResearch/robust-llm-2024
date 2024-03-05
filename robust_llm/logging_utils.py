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

    def increment(
        self, step_count_to_add: int, datapoint_count_to_add: int, commit: bool = False
    ) -> None:
        # We never commit the global counter, since we only ever increment
        # it while incrementing a non-global counter. This non-global counter
        # will manage whether or not to commit.
        if self._parent is not None:
            self._parent.increment(
                step_count_to_add, datapoint_count_to_add, commit=False
            )

        self._step_count += step_count_to_add
        self._datapoint_count += datapoint_count_to_add

        self._log_to_wandb(commit=commit)

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def datapoint_count(self) -> int:
        return self._datapoint_count

    def _log_to_wandb(self, commit: bool = False) -> None:
        assert wandb.run is not None
        wandb.log(
            {
                f"{self._name}_step_count": self._step_count,
                f"{self._name}_datapoint_count": self._datapoint_count,
            },
            commit=commit,
        )


GLOBAL_LOGGING_COUNTER = LoggingCounter(_name="global", _is_global=True)


def setup_wandb_metrics():
    # Set the default horizontal axis on wandb
    # NOTE: Older versions of wandb don't have "define_metric",
    # so try updating wandb if you get an error here.
    assert wandb.run is not None
    assert hasattr(wandb, "define_metric")
    wandb.define_metric("global_datapoint_count")
    wandb.define_metric("victim_training_datapoint_count")
    wandb.define_metric("training_attack_datapoint_count")
    wandb.define_metric("validation_attack_datapoint_count")
    wandb.define_metric(
        "*",
        step_metric="global_datapoint_count",
    )
    wandb.define_metric(
        "train/*",
        step_metric="victim_training_datapoint_count",
    )
    wandb.define_metric(
        "eval/*",
        step_metric="victim_training_datapoint_count",
    )
    wandb.define_metric(
        "training_attack/*",
        step_metric="training_attack_datapoint_count",
    )
    wandb.define_metric(
        "validation_attack/*",
        step_metric="validation_attack_datapoint_count",
    )
