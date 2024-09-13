import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "wl"
ATTACK = "gcg"

HYDRA_CONFIG = f"niki/iclr2024_adv_train_{ATTACK}_{DATASET}"

# See https://docs.google.com/spreadsheets/d/1eifxKB_r9IRnVSqm10as-grVEtcOal9zE8B57iD6-Z0  # noqa: E501
N_ADV_TR_ROUNDS = [1, 1]

# Set a floor of 5 and a ceiling of 60 for adv tr rounds
N_ADV_TR_ROUNDS = [max(5, min(60, x)) for x in N_ADV_TR_ROUNDS]

# Increment each of these because we skip the first round to
# avoid training on clean data only.
N_ADV_TR_ROUNDS = [i + 1 for i in N_ADV_TR_ROUNDS]

CLUSTER_NAME = "h100"

MODEL_GPU_MEMORY_CLUSTER_PARALLEL: list[tuple[str, int, str, str, int]] = [
    (
        "pythia-6.9b",
        4,
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
FINETUNE_SEEDS = [0, 1, 2, 3, 4]
# NOTE: We match adv training seed to finetune seed.
OVERRIDE_TUPLES = [
    (
        {
            "+model": f"Default/clf/{DATASET}/{model}-s{finetune_seed}",
            "training.save_name": (
                f"clf_{DATASET}_{model}_s-{finetune_seed}"
                f"_adv_tr_{ATTACK}_t-{finetune_seed}"
            ),
            "training.adversarial.num_adversarial_training_rounds": n_adv_tr_rounds,
            "training.seed": finetune_seed,
        },
        n_gpus,
        memory,
        cluster,
        parallel,
    )
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
