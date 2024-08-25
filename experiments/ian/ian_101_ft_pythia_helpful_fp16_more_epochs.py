import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "helpful"

HYDRA_CONFIG = f"ian/ft_pythia_{DATASET}"


BASE_FT_GPU_MEMORY = [
    (
        "EleutherAI/pythia-14m",
        "pythia-14m_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "10G",
    ),
    (
        "EleutherAI/pythia-31m",
        "pythia-31m_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "10G",
    ),
    (
        "EleutherAI/pythia-70m",
        "pythia-70m_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "10G",
    ),
    (
        "EleutherAI/pythia-160m",
        "pythia-160m_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "20G",
    ),
    (
        "EleutherAI/pythia-410m",
        "pythia-410m_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "20G",
    ),
    (
        "EleutherAI/pythia-1b",
        "pythia-1b_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "50G",
    ),
    (
        "EleutherAI/pythia-1.4b",
        "pythia-1.4b_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "50G",
    ),
    (
        "EleutherAI/pythia-2.8b",
        "pythia-2.8b_clf_{dataset}_v-{version}_s-{seed}",
        2,
        "100G",
    ),
    (
        "EleutherAI/pythia-6.9b",
        "pythia-6.9b_clf_{dataset}_v-{version}_s-{seed}",
        4,
        "80G",
    ),
    (
        "EleutherAI/pythia-12b",
        "pythia-12b_clf_{dataset}_v-{version}_s-{seed}",
        4,
        "100G",
    ),
]

SEEDS = [0, 1, 2, 3, 4]
# SEEDS = [0]

OVERRIDE_TUPLES = [
    (
        {
            "+model": base_model_name,
            "training.force_name_to_save": target_model_name.format(
                dataset=DATASET,
                version=VERSION_NAME,
                seed=seed,
            ),
            "training.seed": seed,
        },
        n_gpus,
        memory,
    )
    for seed in SEEDS  # This will iterate through seeds *second*
    for (
        base_model_name,
        target_model_name,
        n_gpus,
        memory,
    ) in BASE_FT_GPU_MEMORY
]


OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_TUPLES]
N_GPUS = [x[1] for x in OVERRIDE_TUPLES]
MEMORY = [x[2] for x in OVERRIDE_TUPLES]


if __name__ == "__main__":
    run_multiple(
        experiment_name=EXPERIMENT_NAME,
        hydra_config=HYDRA_CONFIG,
        gpu=N_GPUS,
        override_args_list=OVERRIDE_ARGS_LIST,
        memory=MEMORY,
        container_tag="2024-08-07-backoff",
    )
