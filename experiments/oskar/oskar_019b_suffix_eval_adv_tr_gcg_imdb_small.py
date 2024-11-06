"""Evaluate oskar_018_adv_tr_gcg_imdb_small"""

import os

from robust_llm.batch_job_utils import run_multiple
from robust_llm.experiment_utils import get_all_n_rounds_to_evaluate
from robust_llm.wandb_utils.constants import MODEL_NAMES

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "imdb"
ATTACK = "gcg"
HYDRA_CONFIG = f"ian/iclr2024_stronger_{ATTACK}_{DATASET}"
CLUSTER_NAME = "a6k"

# Get a range of rounds for each model size
evaluation_rounds = get_all_n_rounds_to_evaluate(ATTACK)
names_to_rounds = dict(zip(MODEL_NAMES, evaluation_rounds, strict=True))

MODEL_GPU_MEMORY_CLUSTER_PARALLEL: list[tuple[str, int, str, str, int]] = [
    (
        "pythia-410m",
        1,
        "20G",
        CLUSTER_NAME,
        1,
    ),
    (
        "pythia-1b",
        1,
        "30G",
        CLUSTER_NAME,
        1,
    ),
    (
        "pythia-1.4b",
        1,
        "40G",
        CLUSTER_NAME,
        1,
    ),
    (
        "pythia-2.8b",
        1,
        "50G",
        CLUSTER_NAME,
        1,
    ),
]
SEEDS = [0, 1, 2]

OVERRIDE_TUPLES = [
    (
        {
            "+model": f"Default/clf/{DATASET}/{model}-s{seed}",
            "model.name_or_path": (
                "AlignmentResearch/robust_llm_"
                f"clf_{DATASET}_{model}_s-{seed}"
                f"_adv_tr_{ATTACK}_t-{seed}"
            ),
            "model.revision": f"adv-training-round-{evaluation_round}",
            "evaluation.num_iterations": 128,
        },
        n_gpus,
        memory,
        cluster,
        parallel,
    )
    for model, n_gpus, memory, cluster, parallel in MODEL_GPU_MEMORY_CLUSTER_PARALLEL
    for evaluation_round in names_to_rounds[model.split("-")[-1]]
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
