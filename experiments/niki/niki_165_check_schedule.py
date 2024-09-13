import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "spam"
ATTACK = "gcg"

HYDRA_CONFIG = f"niki/iclr2024_adv_train_{ATTACK}_{DATASET}"

MODEL_GPU_MEMORY_CLUSTER: list[tuple[str, int, str, str]] = [
    (
        "pythia-14m",
        1,
        "50G",
        "a6k",
    ),
]

OVERRIDE_TUPLES = [
    (
        {
            "+model": f"Default/clf/imdb/{model}-s0",
            "training.save_name": None,
            "training.adversarial.num_adversarial_training_rounds": 4,
            "training.adversarial.attack_schedule.start": 8,
            "training.adversarial.attack_schedule.end": 32,
            "training.seed": 0,
        },
        n_gpus,
        memory,
        cluster,
    )
    for (
        model,
        n_gpus,
        memory,
        cluster,
    ) in MODEL_GPU_MEMORY_CLUSTER
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
