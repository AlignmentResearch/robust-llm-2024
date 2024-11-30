"""Based on niki_152_adv_tr_gcg_imdb_small"""

import os

from robust_llm.batch_job_utils import run_multiple
from robust_llm.experiment_utils import QWEN_ROUNDS

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "harmless"
ATTACK = "gcg"

HYDRA_CONFIG = f"niki/iclr2024_adv_train_{ATTACK}_{DATASET}"


CLUSTER_NAME = "h100"

MODEL_GPU_MEMORY_CLUSTER = [
    (
        "Qwen2.5-0.5B",
        1,
        "40G",
        "h100",
    ),
    (
        "Qwen2.5-1.5B",
        1,
        "100G",
        "h100",
    ),
    (
        "Qwen2.5-3B",
        2,
        "200G",
        "h100",
    ),
    (
        "Qwen2.5-7B",
        4,
        "300G",
        "h100",
    ),
    (
        "Qwen2.5-14B",
        4,
        "300G",
        "h100",
    ),
]
SEEDS = [0, 1, 2]

OVERRIDE_TUPLES = [
    (
        {
            "+model": f"Default/clf/{DATASET}/{model}-s{seed}",
            "training.save_name": (
                f"{VERSION_NAME}_clf_{DATASET}_{model}_s-{seed}"
                f"_adv_tr_{ATTACK}_t-{seed}"
            ),
            "training.adversarial.num_adversarial_training_rounds": QWEN_ROUNDS[
                model.split("-")[1]
            ],
            "training.adversarial.training_attack.save_total_limit": 100,
            "training.save_to": "DISK",
            "training.save_total_limit": 100,
            "training.seed": seed,
            "environment.minibatch_multiplier": 0.25,
        },
        n_gpus,
        memory,
        cluster,
    )
    for seed in SEEDS
    for model, n_gpus, memory, cluster in MODEL_GPU_MEMORY_CLUSTER
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_TUPLES]
N_GPUS = [x[1] for x in OVERRIDE_TUPLES]
MEMORY = [x[2] for x in OVERRIDE_TUPLES]
CLUSTER = [x[3] for x in OVERRIDE_TUPLES]
PRIORITY = [
    "low-batch" if x["+model"].split("/")[-1].split("-s")[0] == "Qwen2.5-14B" else "high-batch"  # type: ignore # noqa
    for x in OVERRIDE_ARGS_LIST
]
print(PRIORITY)

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        gpu=N_GPUS,
        memory=MEMORY,
        cluster=CLUSTER,
        cpu=8,
        priority=PRIORITY,
        container_tag="2024-11-03-14-15-57-main",
    )
