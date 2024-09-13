import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "imdb"
ATTACK = "rt"

HYDRA_CONFIG = f"niki/iclr2024_adv_train_{ATTACK}_{DATASET}"

MODEL_GPU_MEMORY_CLUSTER: list[tuple[str, int, str, str]] = [
    (
        "Default/clf/{dataset}/pythia-14m-s{seed}",
        1,
        "20G",
        "a6k",
    ),
]
FINETUNE_SEEDS = [0]
ADV_TRAIN_SEEDS = [0]

OVERRIDE_TUPLES = [
    (
        {
            "+model": model.format(dataset=DATASET, seed=finetune_seed),
            "training.force_name_to_save": (
                f"clf_{DATASET}_pythia-14m_s-{finetune_seed}"
                f"_adv_tr_{ATTACK}_t-{adv_tr_seed}"
            ),
        },
        n_gpus,
        memory,
        cluster,
    )
    for adv_tr_seed in ADV_TRAIN_SEEDS
    for finetune_seed in FINETUNE_SEEDS
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
    )
