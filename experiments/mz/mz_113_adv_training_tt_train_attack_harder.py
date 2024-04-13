import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "adv_training_tensor_trust"

PYTHIA_NAMES = [
    "EleutherAI/pythia-14m",
]
SEEDS = [1, 2]
TRAIN_N_ITS = [10, 20, 30]
USE_BALANCED_SAMPLING = [True, False]
OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": base_model_name,
        # do not save to HF as we eval along the way
        "experiment.training.model_save_path_prefix_or_hf": None,
        "experiment.training.seed": seed,
        "experiment.training.iterative.training_attack.search_based_attack_config.n_its": train_n_its,  # noqa: E501
        "experiment.training.iterative.use_balanced_sampling": use_balanced_sampling,
    }
    for base_model_name in PYTHIA_NAMES
    for seed in SEEDS
    for train_n_its in TRAIN_N_ITS
    for use_balanced_sampling in USE_BALANCED_SAMPLING
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="60G",
        priority="normal-batch",
    )
