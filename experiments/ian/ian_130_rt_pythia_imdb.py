import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
DATASET = "imdb"
HYDRA_CONFIG = f"ian/iclr2024_stronger_rt_{DATASET}"


MODEL_GPU_MEMORY_CLUSTER: list[tuple[str, int, str, str]] = [
    (
        "Default/clf/{dataset}/pythia-14m-s{seed}",
        1,
        "5G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/pythia-31m-s{seed}",
        1,
        "10G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/pythia-70m-s{seed}",
        1,
        "10G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/pythia-160m-s{seed}",
        1,
        "10G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/pythia-410m-s{seed}",
        1,
        "15G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/pythia-1b-s{seed}",
        1,
        "15G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/pythia-1.4b-s{seed}",
        1,
        "20G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/pythia-2.8b-s{seed}",
        1,
        "30G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/pythia-6.9b-s{seed}",
        1,
        "100G",
        "h100",
    ),
    (
        "Default/clf/{dataset}/pythia-12b-s{seed}",
        1,
        "150G",
        "h100",
    ),
]

SEEDS = [0, 1, 2, 3, 4]

OVERRIDE_TUPLES = [
    (
        {
            "+model": model.format(dataset=DATASET, seed=seed),
        },
        n_gpus,
        memory,
        cluster,
    )
    for seed in SEEDS
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
        priority="normal-batch",
    )
