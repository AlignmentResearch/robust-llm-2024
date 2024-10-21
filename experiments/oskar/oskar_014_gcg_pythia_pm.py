import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
DATASET = "pm"
HYDRA_CONFIG = f"ian/gcg_{DATASET}"


MODEL_GPU_MEMORY_CLUSTER: list[tuple[str, int, str, str]] = [
    (
        "Default/clf/{dataset}/pythia-14m-s{seed}",
        2,
        "5G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/pythia-14m-s{seed}",
        2,
        "5G",
        "h100",
    ),
    (
        "Default/clf/{dataset}/pythia-12b-s{seed}",
        2,
        "150G",
        "h100",
    ),
]

OVERRIDE_TUPLES = [
    (
        {
            "+model": model.format(dataset=DATASET, seed=0),
        },
        n_gpus,
        memory,
        cluster,
    )
    for (model, n_gpus, memory, cluster) in MODEL_GPU_MEMORY_CLUSTER
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_TUPLES]
N_GPUS = [x[1] for x in OVERRIDE_TUPLES]
MEMORY = [x[2] for x in OVERRIDE_TUPLES]
CLUSTER = [x[3] for x in OVERRIDE_TUPLES]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        gpu=N_GPUS,
        memory=MEMORY,
        cluster=CLUSTER,
        cpu=8,
        priority="high-batch",
    )
