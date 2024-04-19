# This looks identical to 019, but I changed 'adv_training_tensor_trust'
# in between to reduce the number of validation examples
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "adv_training_tensor_trust"

PYTHIA_NAMES = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-160m",
]
TRAIN_SET_SIZE = [2000]
SEEDS = [0]
OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": base_model_name,
        "experiment.dataset.n_train": train_set_size,
        # do not save to HF as we eval along the way
        "experiment.training.model_save_path_prefix_or_hf": None,
        "experiment.training.seed": seed,
    }
    for base_model_name in PYTHIA_NAMES
    for train_set_size in TRAIN_SET_SIZE
    for seed in SEEDS
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="60G",
        priority="normal-batch",
    )
