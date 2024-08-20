import os
from typing import Any

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "niki/1_DEBUG_niki_128"


OVERRIDE_ARGS_LIST: list[dict[str, Any]] = [
    {
        "environment.deterministic": True,
    },
    {
        "training.adversarial.only_add_successful_adversarial_examples": True,
    },
    {
        "training.adversarial.num_examples_to_generate_each_round": 250,
    },
    {
        "training.adversarial.num_examples_to_generate_each_round": 100,
    },
    {
        "training.adversarial.max_adv_data_proportion": 0.25,
    },
    {
        "training.adversarial.max_adv_data_proportion": 0.75,
    },
    {
        "training.adversarial.target_adversarial_success_rate": 0.5,
    },
    {
        "training.adversarial.target_adversarial_success_rate": 0.75,
    },
    {
        "training.adversarial.sampling_decay": 0.001,
    },
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
        priority="normal-batch",
    )
