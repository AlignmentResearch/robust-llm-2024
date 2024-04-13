import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "adv_training_tensor_trust"

PYTHIA_NAMES = [
    "EleutherAI/pythia-14m",
]
TRAIN_SET_SIZE = [2000, 20000]
SEEDS = [1, 2]
ADV_EXAMPLES_EACH_ROUND = [500, 2000, 5000]
USE_BALANCED_SAMPLING = [True, False]
OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": base_model_name,
        "experiment.environment.train_set_size": train_set_size,
        # do not save to HF as we eval along the way
        "experiment.training.model_save_path_prefix_or_hf": None,
        "experiment.training.iterative.num_examples_to_generate_each_round": adv_examples_each_round,  # noqa: E501
        "experiment.training.iterative.use_balanced_sampling": use_balanced_sampling,
        "experiment.training.seed": seed,
    }
    for base_model_name in PYTHIA_NAMES
    for train_set_size in TRAIN_SET_SIZE
    for seed in SEEDS
    for adv_examples_each_round in ADV_EXAMPLES_EACH_ROUND
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
