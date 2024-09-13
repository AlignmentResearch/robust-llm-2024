import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "imdb"
ATTACK = "rt"

HYDRA_CONFIG = f"niki/iclr2024_adv_train_{ATTACK}_{DATASET}"

N_ADV_TR_ROUNDS = [5]

MODEL_GPU_MEMORY_CLUSTER_PARALLEL: list[tuple[str, int, str, str, int]] = [
    (
        "pythia-14m",
        1,
        "10G",
        "a6k",
        1,
    ),
]
FINETUNE_SEEDS = [0]
EVAL_NUM_ITS = [1024, 2048, 4096, 8192]

OVERRIDE_TUPLES = [
    (
        {
            "+model": f"Default/clf/imdb/{model}-s0",
            "training.model_save_path_prefix_or_hf": None,  # Don't save the model
            "training.adversarial.num_adversarial_training_rounds": 3,
            "training.adversarial.attack_schedule.start": 32,
            "training.adversarial.attack_schedule.end": 32,
            "evaluation.num_iterations": num_its,
        },
        n_gpus,
        memory,
        cluster,
        parallel,
    )
    for num_its in EVAL_NUM_ITS
    for (model, n_gpus, memory, cluster, parallel) in MODEL_GPU_MEMORY_CLUSTER_PARALLEL
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
        priority="high-batch",
    )
