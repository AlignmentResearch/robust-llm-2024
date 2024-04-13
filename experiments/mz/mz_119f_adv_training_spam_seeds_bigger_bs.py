import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "adv_training_imdb"

PYTHIA_NAMES = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
]
SEEDS = list(range(2))
OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.dataset_type": "hf/SetFit/enron_spam",
        "experiment.environment.model_name_or_path": base_model_name,
        # do not save to HF as we eval along the way
        "experiment.training.model_save_path_prefix_or_hf": None,
        "experiment.training.seed": seed,
        "experiment.training.batch_size": 16,
        "experiment.training.iterative.training_attack.search_based_attack_config.forward_pass_batch_size": 128,  # noqa: E501
        "experiment.evaluation.batch_size": 128,
        "experiment.evaluation.evaluation_attack.search_based_attack_config.forward_pass_batch_size": 128,  # noqa: E501
    }
    for base_model_name in PYTHIA_NAMES
    for seed in SEEDS
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="60G",
        priority="high-batch",
    )
