import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
DATASET = "imdb"
HYDRA_CONFIG = f"ian/rt_{DATASET}"

MODEL_GPU_MEMORY_CLUSTER: list[tuple[str, int, str, str]] = [
    (
        "Default/clf/{dataset}/pythia-14m-seed{pretrain_seed}-s{finetune_seed}",
        1,
        "5G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/pythia-31m-seed{pretrain_seed}-s{finetune_seed}",
        1,
        "10G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/pythia-70m-seed{pretrain_seed}-s{finetune_seed}",
        1,
        "10G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/pythia-160m-seed{pretrain_seed}-s{finetune_seed}",
        1,
        "10G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/pythia-410m-seed{pretrain_seed}-s{finetune_seed}",
        1,
        "15G",
        "a6k",
    ),
]

PRETRAIN_SEEDS = list(range(1, 10))
FINE_TUNE_SEEDS = [0]


OVERRIDE_TUPLES = [
    (
        {
            "+model": model.format(
                dataset=DATASET,
                pretrain_seed=pretrain_seed,
                finetune_seed=finetune_seed,
            ),
        },
        n_gpus,
        memory,
        cluster,
    )
    for finetune_seed in FINE_TUNE_SEEDS
    for (model, n_gpus, memory, cluster) in MODEL_GPU_MEMORY_CLUSTER
    # Iterates through pretrain_seeds first
    for pretrain_seed in PRETRAIN_SEEDS
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
