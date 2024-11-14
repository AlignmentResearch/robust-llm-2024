import os

from robust_llm.batch_job_utils import run_multiple
from robust_llm.experiment_utils import get_all_n_rounds_to_evaluate

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "pm"
ATTACK = "gcg"
HYDRA_CONFIG = f"ian/iclr2024_stronger_{ATTACK}_{DATASET}"
CLUSTER_NAME = "h100"

# Get a range of rounds for the 2 largest models
evaluation_rounds = get_all_n_rounds_to_evaluate(ATTACK)[-2:]

MODEL_GPU_MEMORY_CLUSTER_PARALLEL: list[tuple[str, int, str, str, int]] = [
    (
        "pythia-6.9b",
        1,
        "80G",
        CLUSTER_NAME,
        1,
    ),
    (
        "pythia-12b",
        2,
        "160G",
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
            "evaluation.num_iterations": 128,
        },
        n_gpus,
        memory,
        cluster,
        parallel,
    )
    for (model, n_gpus, memory, cluster, parallel), evaluation_rounds_list in zip(
        MODEL_GPU_MEMORY_CLUSTER_PARALLEL, evaluation_rounds, strict=True
    )
    for evaluation_round in evaluation_rounds_list
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
