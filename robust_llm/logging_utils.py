from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import wandb
import yaml
from omegaconf import OmegaConf

from robust_llm.configs import ExperimentConfig
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


@dataclass
class LoggingCounter:
    """Class for logging precise step and datapoint counts.

    Used to keep track of the number of batches seen during training,
    as well as the precise number of datapoints that corresponds to.
    Individual models can each have their own logging counter, and
    all point to the same global counter, which is used to keep track
    of the total number of batches and datapoints seen across all models
    during the current experiment.
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

    # Make sure step metrics for global counter are logged at least once.
    GLOBAL_LOGGING_COUNTER._log_to_wandb()


def wandb_set_really_finished():
    # Sometimes wandb runs are marked as finished even though in fact they are not.
    # In order to know for sure whether a run finished properly, we manually
    # call this function at the end of the run. This way, we have the guarantee
    # that the run finished properly iff we see `really_finished=1` on wandb.
    assert wandb.run is not None
    wandb.run.log({"really_finished": 1}, commit=True)


def log_dataset_to_wandb(
    dataset: RLLMDataset, dataset_name: str, max_n_examples: Optional[int] = None
) -> None:
    if max_n_examples is not None:
        dataset = dataset.get_subset(range(min(len(dataset), max_n_examples)))

    dataset_table = wandb.Table(columns=["text", "label"])

    for text, label in zip(
        dataset.ds["text"],
        dataset.ds["clf_label"],
    ):
        dataset_table.add_data(text, label)

    wandb.log({dataset_name: dataset_table}, commit=False)


def log_config_to_wandb(config: ExperimentConfig) -> None:
    """Logs the job config to wandb."""
    if not wandb.run:
        raise ValueError("wandb should have been initialized by now, exiting...")
    config_yaml = yaml.load(OmegaConf.to_yaml(config), Loader=yaml.FullLoader)
    wandb.run.summary["experiment_yaml"] = config_yaml  # type: ignore[has-type]


def wandb_initialize(
    config: ExperimentConfig, set_up_step_metrics: bool = True
) -> None:
    """Initializes wandb run and does appropriate setup.

    Args:
        config: config of the experiment
        set_up_step_metrics: whether to set up wandb step metrics which are used to
            define default x-axes for logged values
    """
    wandb.init(
        project="robust-llm",
        group=config.experiment_name,
        job_type=config.job_type,
        name=config.run_name,
    )
    if set_up_step_metrics:
        setup_wandb_metrics()
    log_config_to_wandb(config)


def wandb_cleanup() -> None:
    """Does necessary cleanup for wandb before the experiment ends."""
    wandb_set_really_finished()
    wandb.finish()
