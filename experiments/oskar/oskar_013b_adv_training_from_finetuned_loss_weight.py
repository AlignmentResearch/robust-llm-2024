import os
from typing import Any

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/pythia_14m_rt_spam.yaml"


OVERRIDE_ARGS_LIST: list[dict[str, Any]] = [
    {
        "training.adversarial.adv_sampling_decay": 0.0,
        "environment.deterministic": True,
    },
    {
        "training.adversarial.adv_sampling_decay": 0.001,
        "training.adversarial.only_add_successful_adversarial_examples": True,
        "environment.deterministic": True,
    },
    {
        "training.adversarial.adv_sampling_decay": 0.001,
        "training.adversarial.loss_rank_weight": 0.5,
        "environment.deterministic": True,
    },
    {
        "training.adversarial.adv_sampling_decay": 0.001,
        "training.adversarial.loss_rank_weight": 1.0,
        "environment.deterministic": True,
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
