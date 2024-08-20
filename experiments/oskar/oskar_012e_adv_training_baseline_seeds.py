import os
from typing import Any

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "niki/1_DEBUG_niki_128"


OVERRIDE_ARGS_LIST: list[dict[str, Any]] = [
    {
        "environment.deterministic": False,
    },
]
OVERRIDE_ARGS_LIST = [
    {**{"training.seed": seed}, **override_args}
    for seed in range(5)
    for override_args in OVERRIDE_ARGS_LIST
]
GPU = 1
MEMORY = "80G"
CPU = 4


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory=MEMORY,
        cpu=CPU,
        gpu=GPU,
        priority="low-batch",
    )
