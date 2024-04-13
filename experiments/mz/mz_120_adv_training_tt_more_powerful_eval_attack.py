import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "adv_training_tensor_trust"

PYTHIA_NAMES = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-160m",
]
SEEDS = list(range(2))
OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": base_model_name,
        # do not save to HF as we eval along the way
        "experiment.training.model_save_path_prefix_or_hf": None,
        # Defense uses 3 its, attack uses 10 its
        "experiment.training.iterative.training_attack.search_based_attack_config.n_its": 3,  # noqa: E501
        "experiment.evaluation.evaluation_attack.search_based_attack_config.n_its": 10,  # noqa: E501
        "experiment.training.seed": seed,
        "experiment.training.iterative.num_iterative_training_rounds": 10,
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
        priority="normal-batch",
    )
