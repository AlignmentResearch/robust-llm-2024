import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "wl"
ATTACK = "gcg"

HYDRA_CONFIG = f"niki/iclr2024_adv_train_{ATTACK}_{DATASET}"

N_ADV_TR_ROUNDS = [10, 10]

CLUSTER_NAME = "h100"

MODEL_GPU_MEMORY_CLUSTER_PARALLEL: list[tuple[str, int, str, str, int]] = [
    (
        "pythia-6.9b",
        3,
        "200G",
        CLUSTER_NAME,
        1,
    ),
    (
        "pythia-12b",
        4,
        "200G",
        CLUSTER_NAME,
        1,
    ),
]
FINETUNE_SEEDS = [0, 1, 2]

# NOTE: We match the ADV_TRAIN_SEED and the FINETUNE_SEED
OVERRIDE_TUPLES = [
    (
        {
            "+model": f"Default/clf/{DATASET}/{model}-s{finetune_seed}",
            "training.save_name": (
                f"{VERSION_NAME}_clf_{DATASET}_{model}_s-{finetune_seed}"
                f"_adv_tr_{ATTACK}_t-{finetune_seed}"
            ),
            "training.adversarial.num_adversarial_training_rounds": n_adv_tr_rounds,
            "training.seed": finetune_seed,
            "environment.allow_checkpointing": True,
            "training.save_strategy": "steps",
        },
        n_gpus,
        memory,
        cluster,
        parallel,
    )
    for (model, n_gpus, memory, cluster, parallel), n_adv_tr_rounds in zip(
        MODEL_GPU_MEMORY_CLUSTER_PARALLEL, N_ADV_TR_ROUNDS
    )
    for finetune_seed in FINETUNE_SEEDS
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_TUPLES]
N_GPUS = [x[1] for x in OVERRIDE_TUPLES]
MEMORY = [x[2] for x in OVERRIDE_TUPLES]
CLUSTER = [x[3] for x in OVERRIDE_TUPLES]
PARALLEL = [x[4] for x in OVERRIDE_TUPLES]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        gpu=N_GPUS,
        memory=MEMORY,
        cluster=CLUSTER,
        n_max_parallel=PARALLEL,
        cpu=8,
        priority="normal-batch",
    )
