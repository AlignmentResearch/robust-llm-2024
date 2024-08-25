import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "harmless"

HYDRA_CONFIG = f"ian/ft_pythia_{DATASET}"

EPOCHS = [3, 6, 10]
LR = [1e-5, 5e-5]
LR_SCHEDULER = ["linear", "cosine"]
BATCH_SIZE = [8, 16, 32, 64]

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
]

total_configs = (
    len(EPOCHS)
    * len(LR)
    * len(LR_SCHEDULER)
    * len(BATCH_SIZE)
    * len(BASE_FT_GPU_MEMORY)
)
print(f"Total configs: {total_configs}")

OVERRIDE_TUPLES = [
    (
        {
            "+model": base_model_name,
            "environment.deterministic": True,
            "training.model_save_path_prefix_or_hf": None,
            "training.optimizer": "adafactor",
            "training.group_by_length": True,
            "training.num_train_epochs": epoch,
            "training.lr_scheduler_type": lr_scheduler,
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
    for lr_scheduler in LR_SCHEDULER
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
print("Made args lists")


if __name__ == "__main__":
    run_multiple(
        experiment_name=EXPERIMENT_NAME,
        hydra_config=HYDRA_CONFIG,
        gpu=N_GPUS,
        override_args_list=OVERRIDE_ARGS_LIST,
        memory=MEMORY,
        container_tag="2024-08-07-backoff",
    )
