"""Evaluate qwen FT models, based on ian_102_gcg_pythia_harmless.py"""

import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
DATASET = "harmless"
HYDRA_CONFIG = f"ian/gcg_{DATASET}"


MODEL_GPU_MEMORY_CLUSTER: list[tuple[str, int, str, str]] = [
    (
        "Default/clf/{dataset}/Qwen2.5-0.5B-s{seed}",
        1,
        "30G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/Qwen2.5-1.5B-s{seed}",
        1,
        "40G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/Qwen2.5-3B-s{seed}",
        1,
        "60G",
        "a6k",
    ),
    (
        "Default/clf/{dataset}/Qwen2.5-7B-s{seed}",
        2,
        "150G",
        "h100",
    ),
    (
        "Default/clf/{dataset}/Qwen2.5-14B-s{seed}",
        2,
        "200G",
        "h100",
    ),
]

SEEDS = [0, 1, 2]

OVERRIDE_TUPLES = [
    (
        {
            "+model": model.format(dataset=DATASET, seed=seed),
            # Scale down all batch sizes by a factor of 4 due to recent memory issues
            "environment.minibatch_multiplier": 0.25,
            "evaluation.upload_artifacts": False,
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
        container_tag="2024-11-03-14-15-57-main",
    )
