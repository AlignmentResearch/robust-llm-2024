import os
from typing import Any

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "niki/1_DEBUG_niki_128"


OVERRIDE_ARGS_LIST: list[dict[str, Any]] = [
    {
        "training.adversarial.target_adversarial_success_rate": 0.5,
        "training.adversarial.min_attack_iterations": 1280,
        "training.adversarial.max_attack_iterations": 8192,
        "training.adversarial.training_attack.initial_n_its": 1280,
        "environment.deterministic": True,
    },
    {
        "environment.deterministic": True,
        "training.adversarial.loss_rank_weight": 0.0,
        "training.adversarial.adv_sampling_decay": 0.002,
    },
    {
        "environment.deterministic": True,
        "training.adversarial.loss_rank_weight": 0.0,
        "training.adversarial.adv_sampling_decay": 0.0005,
    },
    {
        "environment.deterministic": True,
        "training.adversarial.loss_rank_weight": 0.5,
        "training.adversarial.adv_sampling_decay": 0.0002,
    },
    {
        "environment.deterministic": True,
        "training.adversarial.loss_rank_weight": 0.5,
        "training.adversarial.adv_sampling_decay": 0.0001,
    },
    {
        "environment.deterministic": True,
        "training.adversarial.loss_rank_weight": 1.0,
        "training.adversarial.adv_sampling_decay": 0.0001,
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
