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

MODEL_GPU_MEMORY_CLUSTER: list[tuple[str, int, str, str]] = [
    (
        "pythia-6.9b",
        4,
        "100G",
        "h100",
    ),
]
FINETUNE_SEEDS = [0]

OVERRIDE_TUPLES = [
    (
        {
            "+model": "Default/clf/imdb/pythia-6.9b-s0",
            "training.model_save_path_prefix_or_hf": None,  # Don't save the model
            "training.adversarial.num_adversarial_training_rounds": 5,
            "training.adversarial.num_examples_to_generate_each_round": 200,
        },
        4,
        "100G",
        "h100",
    ),
    (
        {
            "+model": "Default/clf/imdb/pythia-6.9b-s0",
            "training.model_save_path_prefix_or_hf": None,  # Don't save the model
            "training.adversarial.num_adversarial_training_rounds": 5,
            "training.adversarial.num_examples_to_generate_each_round": 400,
        },
        4,
        "100G",
        "h100",
    ),
    (
        {
            "+model": "Default/clf/imdb/pythia-6.9b-s0",
            "training.model_save_path_prefix_or_hf": None,  # Don't save the model
            "training.adversarial.num_adversarial_training_rounds": 10,
            "training.adversarial.max_augmented_data_size": 2000,
        },
        4,
        "100G",
        "h100",
    ),
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
        priority="high-batch",
    )
