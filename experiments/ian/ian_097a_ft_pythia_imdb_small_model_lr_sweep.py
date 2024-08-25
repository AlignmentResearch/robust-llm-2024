# More eval datapoints, slightly different lrs
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "imdb"

HYDRA_CONFIG = f"ian/ft_pythia_{DATASET}"

EPOCHS = [2, 3, 4, 5]
LR = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6]
BATCH_SIZE = [4, 8, 16, 32]

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
        "15G",
    ),
    (
        "EleutherAI/pythia-70m",
        "pythia-70m_clf_{dataset}_v-{version}_s-{seed}",
        1,
        "20G",
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

total_configs = len(EPOCHS) * len(LR) * len(BATCH_SIZE) * len(BASE_FT_GPU_MEMORY)
print(f"Total configs: {total_configs}")

OVERRIDE_TUPLES = [
    (
        {
            "+model": base_model_name,
            "dataset.n_val": 5_000,
            "training.model_save_path_prefix_or_hf": None,
            "training.num_train_epochs": epoch,
            "training.learning_rate": lr,
            "model.train_minibatch_size": batch_size,
            "model.gradient_accumulation_steps": 1,
        },
        n_gpus,
        memory,
    )
    for epoch in EPOCHS
    for lr in LR
    for batch_size in BATCH_SIZE
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
