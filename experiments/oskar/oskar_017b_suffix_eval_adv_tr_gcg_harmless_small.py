"""Evaluate niki_174_adv_tr_gcg_harmless_small"""

import os

from robust_llm.batch_job_utils import run_multiple
from robust_llm.experiment_utils import get_all_n_rounds_to_evaluate
from robust_llm.wandb_utils.constants import MODEL_NAMES

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "harmless"
ATTACK = "gcg"
HYDRA_CONFIG = f"ian/iclr2024_stronger_{ATTACK}_{DATASET}"
CLUSTER_NAME = "a6k"

# Get a range of rounds for each model size
evaluation_rounds = get_all_n_rounds_to_evaluate(ATTACK)
names_to_rounds = {
    f"pythia-{size}": rounds for size, rounds in zip(MODEL_NAMES, evaluation_rounds)
}

MODEL_GPU_MEMORY_CLUSTER_PARALLEL: list[tuple[str, int, str, str, int]] = [
    (
        "pythia-14m",
        1,
        "60G",
        CLUSTER_NAME,
        10,
    ),
    (
        "pythia-31m",
        1,
        "60G",
        CLUSTER_NAME,
        6,
    ),
    (
        "pythia-70m",
        1,
        "60G",
        CLUSTER_NAME,
        4,
    ),
    (
        "pythia-160m",
        1,
        "60G",
        CLUSTER_NAME,
        2,
    ),
    (
        "pythia-410m",
        1,
        "60G",
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
            "+model": f"AdvTrained/clf/{ATTACK}/{DATASET}/{model}-s{seed}",
            "model.revision": f"adv-training-round-{evaluation_round}",
        },
        n_gpus,
        memory,
        cluster,
        parallel,
    )
    for (model, n_gpus, memory, cluster, parallel), evaluation_rounds_list in zip(
        MODEL_GPU_MEMORY_CLUSTER_PARALLEL, evaluation_rounds
    )
    for evaluation_round in names_to_rounds[model]
    for seed in SEEDS
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
