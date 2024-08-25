import os

from robust_llm.batch_job_utils import run_multiple

HYDRA_CONFIG = "ian/058_ft_pythia_imdb"

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "imdb"

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
]

# SEEDS = [0, 1, 2, 3, 4]
SEEDS = [0]

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
            "training.lr_scheduler_type": "linear",
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
