"""Re-run adversarial training on seed mismatches.

Based on niki_152_adv_tr_gcg_imdb_small
"""

import os

from robust_llm.batch_job_utils import run_multiple
from robust_llm.experiment_utils import get_n_adv_tr_rounds

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "imdb"
ATTACK = "gcg"

HYDRA_CONFIG = f"niki/iclr2024_adv_train_{ATTACK}_{DATASET}"

N_ADV_TR_ROUNDS = get_n_adv_tr_rounds(ATTACK)[4:8]

CLUSTER_NAME = "h100"

MODEL_GPU_MEMORY_CLUSTER_PARALLEL: list[tuple[str, int, str, str, int]] = [
    (
        "pythia-410m",
        1,
        "50G",
        CLUSTER_NAME,
        1,
    ),
    (
        "pythia-1b",
        1,
        "60G",
        CLUSTER_NAME,
        1,
    ),
    (
        "pythia-1.4b",
        1,
        "80G",
        CLUSTER_NAME,
        1,
    ),
    (
        "pythia-2.8b",
        1,
        "100G",
        CLUSTER_NAME,
        1,
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
            "training.adversarial.num_adversarial_training_rounds": n_adv_tr_rounds,
            "training.seed": seed,
            "training.adversarial.attack_schedule.start": 8,
            "training.adversarial.attack_schedule.end": 64,
        },
        n_gpus,
        memory,
        cluster,
        parallel,
    )
    for seed in SEEDS
    for (model, n_gpus, memory, cluster, parallel), n_adv_tr_rounds in zip(
        MODEL_GPU_MEMORY_CLUSTER_PARALLEL, N_ADV_TR_ROUNDS, strict=True
    )
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_TUPLES]
N_GPUS = [x[1] for x in OVERRIDE_TUPLES]
MEMORY = [x[2] for x in OVERRIDE_TUPLES]
CLUSTER = [x[3] for x in OVERRIDE_TUPLES]
PARALLEL = [x[4] for x in OVERRIDE_TUPLES]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        gpu=N_GPUS,
        memory=MEMORY,
        cluster=CLUSTER,
        n_max_parallel=PARALLEL,
        cpu=8,
        priority="normal-batch",
    )
