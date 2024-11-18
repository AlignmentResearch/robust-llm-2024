"""Based on experiments/ian/ian_135_ft_pythia_harmless.py"""

import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "wl"

HYDRA_CONFIG = (
    f"ian/ft_pythia_{DATASET}"  # Despite the name, this is a generic finetuning config
)

SAVE_TEMPLATE = "{base_model}_clf_{dataset}_v-{version}_s-{seed}"


BASE_GPU_MEMORY_CLUSTER = [
    (
        "Qwen2.5-0.5B",
        1,
        "20G",
        "h100",
        8,
    ),
    (
        "Qwen2.5-1.5B",
        1,
        "50G",
        "h100",
        4,
    ),
    (
        "Qwen2.5-3B",
        2,
        "100G",
        "h100",
        2,
    ),
]

SEEDS = [0, 1, 2]

OVERRIDE_TUPLES = [
    (
        {
            "+model": "Qwen/" + base_model_name,
            "training.save_name": SAVE_TEMPLATE.format(
                dataset=DATASET,
                version=VERSION_NAME,
                seed=seed,
                base_model=base_model_name,
            ),
            "training.seed": seed,
            "training.save_to": "DISK",
            "model.max_minibatch_size": batch_size,
        },
        n_gpus,
        memory,
        cluster,
    )
    for seed in SEEDS  # This will run everything for a seed, then move to next seed
    for (
        base_model_name,
        n_gpus,
        memory,
        cluster,
        batch_size,
    ) in BASE_GPU_MEMORY_CLUSTER
]


OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_TUPLES]
N_GPUS = [x[1] for x in OVERRIDE_TUPLES]
MEMORY = [x[2] for x in OVERRIDE_TUPLES]
CLUSTER = [x[3] for x in OVERRIDE_TUPLES]


if __name__ == "__main__":
    run_multiple(
        experiment_name=EXPERIMENT_NAME,
        hydra_config=HYDRA_CONFIG,
        gpu=N_GPUS,
        cluster=CLUSTER,
        override_args_list=OVERRIDE_ARGS_LIST,
        memory=MEMORY,
        container_tag="2024-11-03-14-15-57-main",
        priority="normal-batch",
    )
