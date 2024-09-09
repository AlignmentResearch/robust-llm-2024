import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "spam"

HYDRA_CONFIG = f"ian/ft_pythia_{DATASET}"


BASE_FT_GPU_MEMORY_CLUSTER_N_MAX_PARALLEL = [
    (
        "EleutherAI/pythia-14m-seed{pretrain_seed}",
        "pythia-14m-seed{pretrain_seed}_clf_{dataset}_v-{version}_s-{finetune_seed}",
        1,
        "100G",
        "h100",
        10,
    ),
    (
        "EleutherAI/pythia-31m-seed{pretrain_seed}",
        "pythia-31m-seed{pretrain_seed}_clf_{dataset}_v-{version}_s-{finetune_seed}",
        1,
        "100G",
        "h100",
        10,
    ),
    (
        "EleutherAI/pythia-70m-seed{pretrain_seed}",
        "pythia-70m-seed{pretrain_seed}_clf_{dataset}_v-{version}_s-{finetune_seed}",
        1,
        "100G",
        "h100",
        10,
    ),
    (
        "EleutherAI/pythia-160m-seed{pretrain_seed}",
        "pythia-160m-seed{pretrain_seed}_clf_{dataset}_v-{version}_s-{finetune_seed}",
        1,
        "100G",
        "h100",
        5,
    ),
    (
        "EleutherAI/pythia-410m-seed{pretrain_seed}",
        "pythia-410m-seed{pretrain_seed}_clf_{dataset}_v-{version}_s-{finetune_seed}",
        1,
        "30G",
        "h100",
        1,
    ),
]


PRETRAIN_SEEDS = range(1, 10)
FINETUNE_SEEDS = [0]
# SEEDS = [0]

OVERRIDE_TUPLES = [
    (
        {
            "+model": base_model_name.format(pretrain_seed=pretrain_seed),
            "training.force_name_to_save": target_model_name.format(
                dataset=DATASET,
                version=VERSION_NAME,
                pretrain_seed=pretrain_seed,
                finetune_seed=finetune_seed,
            ),
            "training.seed": finetune_seed,
        },
        n_gpus,
        memory,
        cluster,
        n_max_parallel,
    )
    # This will iterate through finetune seeds *second*
    for finetune_seed in FINETUNE_SEEDS
    for (
        base_model_name,
        target_model_name,
        n_gpus,
        memory,
        cluster,
        n_max_parallel,
    ) in BASE_FT_GPU_MEMORY_CLUSTER_N_MAX_PARALLEL
    # This will iterate through pretrain seeds *first*
    for pretrain_seed in PRETRAIN_SEEDS
]


OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_TUPLES]
N_GPUS = [x[1] for x in OVERRIDE_TUPLES]
MEMORY = [x[2] for x in OVERRIDE_TUPLES]
CLUSTER = [x[3] for x in OVERRIDE_TUPLES]
N_MAX_PARALLEL = [x[4] for x in OVERRIDE_TUPLES]

if __name__ == "__main__":
    run_multiple(
        experiment_name=EXPERIMENT_NAME,
        hydra_config=HYDRA_CONFIG,
        gpu=N_GPUS,
        cluster=CLUSTER,
        override_args_list=OVERRIDE_ARGS_LIST,
        memory=MEMORY,
        n_max_parallel=N_MAX_PARALLEL,
        container_tag="2024-08-07-backoff",
    )
