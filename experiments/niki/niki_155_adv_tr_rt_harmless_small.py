import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "harmless"
ATTACK = "rt"

HYDRA_CONFIG = f"niki/iclr2024_adv_train_{ATTACK}_{DATASET}"

# See https://docs.google.com/spreadsheets/d/1eifxKB_r9IRnVSqm10as-grVEtcOal9zE8B57iD6-Z0  # noqa: E501
N_ADV_TR_ROUNDS = [953, 413, 163, 59, 21, 8, 6, 3, 1, 1]

# Set a floor of 3 and a ceiling of 250 for adv tr rounds
N_ADV_TR_ROUNDS = [max(5, min(250, x)) for x in N_ADV_TR_ROUNDS]

# Increment each of these because we skip the first round to
# avoid training on clean data only.
N_ADV_TR_ROUNDS = [i + 1 for i in N_ADV_TR_ROUNDS]

CLUSTER_NAME = "h100"

MODEL_GPU_MEMORY_CLUSTER_PARALLEL: list[tuple[str, int, str, str, int]] = [
    (
        "pythia-14m",
        1,
        "50G",
        CLUSTER_NAME,
        10,
    ),
    (
        "pythia-31m",
        1,
        "50G",
        CLUSTER_NAME,
        6,
    ),
    (
        "pythia-70m",
        1,
        "50G",
        CLUSTER_NAME,
        4,
    ),
    (
        "pythia-160m",
        1,
        "50G",
        CLUSTER_NAME,
        2,
    ),
    (
        "pythia-410m",
        1,
        "50G",
        CLUSTER_NAME,
        1,
    ),
    (
        "pythia-1b",
        1,
        "60G",
        CLUSTER_NAME,
        1,
    ),
    (
        "pythia-1.4b",
        1,
        "80G",
        CLUSTER_NAME,
        1,
    ),
    (
        "pythia-2.8b",
        1,
        "100G",
        CLUSTER_NAME,
        1,
    ),
]
FINETUNE_SEEDS = [0]
ADV_TRAIN_SEEDS = [0, 1, 2, 3, 4]

OVERRIDE_TUPLES = [
    (
        {
            "+model": f"Default/clf/{DATASET}/{model}-s{finetune_seed}",
            "training.force_name_to_save": (
                f"clf_{DATASET}_{model}_s-{finetune_seed}"
                f"_adv_tr_{ATTACK}_t-{adv_tr_seed}"
            ),
            "training.adversarial.num_adversarial_training_rounds": n_adv_tr_rounds,
            "training.seed": adv_tr_seed,
        },
        n_gpus,
        memory,
        cluster,
        parallel,
    )
    for adv_tr_seed in ADV_TRAIN_SEEDS
    for finetune_seed in FINETUNE_SEEDS
    for (model, n_gpus, memory, cluster, parallel), n_adv_tr_rounds in zip(
        MODEL_GPU_MEMORY_CLUSTER_PARALLEL, N_ADV_TR_ROUNDS
    )
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
