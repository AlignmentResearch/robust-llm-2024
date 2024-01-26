"""Preliminary pipeline to run multiple (scaling) experiments."""

import shlex
import subprocess
from datetime import date

import numpy as np
import wandb.util
from hydra.core.hydra_config import HydraConfig

from robust_llm.configs import OverallConfig

# Model sizes: 14m, 31m, 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b
LIST_OF_PYTHIA_MODELS = [
    "pythia-14m",
    "pythia-31m",
    "pythia-70m-deduped",
    "pythia-160m-deduped",
    "pythia-410m-deduped",
    "pythia-1b-deduped",
    # "pythia-1.4b-deduped",
    # "pythia-2.8b-deduped",
    # "pythia-6.9b-deduped",
    # "pythia-12b-deduped",
]

PREPEND_STRING = "EleutherAI"

# Checkpoints: 0 to 142k inclusive, divide into 8, round to nearest 1000
WIDE_LIST_OF_CHECKPOINTS = np.linspace(0, 142000, 8, dtype=int).round(-3)

LOW_LIST_OF_CHECKPOINTS = [i for i in range(0, 10_000, 1000)]


def run_experiment(
    experiment_yaml: str,
    experiment_name: str,
    checkpoints: list[int] = LOW_LIST_OF_CHECKPOINTS,
    model_names: list[str] = LIST_OF_PYTHIA_MODELS,
) -> None:
    """Run an experiment with the given parameters.

    Args:
        experiment_yaml (str): The path to the yaml file for Hydra to use for
        the experiment.
        experiment_name (str): The name of the experiment. Used to group runs in wandb.
        checkpoints (list[int], optional): The checkpoints to use.
        model_names (list[str], optional): The models to use.
    """
    random_string_for_experiment = wandb.util.generate_id(length=5)
    for model_name in model_names:
        full_model_name = f"{PREPEND_STRING}/{model_name}"
        for checkpoint in checkpoints:
            command_list = [
                "python",
                "-m",
                "robust_llm",
                "+experiment=" + experiment_yaml,
                "experiment.environment.model_name=" + full_model_name,
                f"experiment.training.checkpoint={checkpoint}",
                f"experiment.experiment_name={experiment_name}_{date.today().strftime('%Y-%m-%d')}_{random_string_for_experiment}",  # noqa: E501
                f"experiment.job_type={model_name}_step{checkpoint}",
                f"experiment.run_name=run_{wandb.util.generate_id(length=5)}",
            ]

            command_string = shlex.join(command_list)

            subprocess.run(command_string, shell=True)


# TODO(michal): deprecate this pipeline. Implement general grid searches instead.
def run_scaling_experiments_pipeline(args: OverallConfig):
    print("Running a scaling experiment...")

    hydra_config_name = HydraConfig.get().overrides.task[0].split("=")[1]

    run_experiment(
        experiment_yaml=hydra_config_name,
        experiment_name=args.experiment.experiment_name,
    )
