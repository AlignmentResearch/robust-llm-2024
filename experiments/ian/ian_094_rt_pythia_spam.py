import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
DATASET = "spam"
HYDRA_CONFIG = f"ian/rt_{DATASET}"


MODEL_GPU_MEMORY: list[tuple[str, int, str]] = [
    (
        "Default/clf/{dataset}/pythia-14m-s{seed}",
        1,
        "5G",
    ),
    (
        "Default/clf/{dataset}/pythia-31m-s{seed}",
        1,
        "10G",
    ),
    (
        "Default/clf/{dataset}/pythia-70m-s{seed}",
        1,
        "10G",
    ),
    (
        "Default/clf/{dataset}/pythia-160m-s{seed}",
        1,
        "10G",
    ),
    (
        "Default/clf/{dataset}/pythia-410m-s{seed}",
        1,
        "15G",
    ),
    (
        "Default/clf/{dataset}/pythia-1b-s{seed}",
        1,
        "15G",
    ),
    (
        "Default/clf/{dataset}/pythia-1.4b-s{seed}",
        1,
        "20G",
    ),
    (
        "Default/clf/{dataset}/pythia-2.8b-s{seed}",
        2,
        "20G",
    ),
    (
        "Default/clf/{dataset}/pythia-6.9b-s{seed}",
        2,
        "60G",
    ),
    (
        "Default/clf/{dataset}/pythia-12b-s{seed}",
        2,
        "80G",
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
    )
    for seed in SEEDS
    for (model, n_gpus, memory) in MODEL_GPU_MEMORY
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_TUPLES]
N_GPUS = [x[1] for x in OVERRIDE_TUPLES]
MEMORY = [x[2] for x in OVERRIDE_TUPLES]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        gpu=N_GPUS,
        memory=MEMORY,
        cpu=8,
        priority="normal-batch",
    )
