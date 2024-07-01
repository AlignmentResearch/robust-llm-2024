"""Another makeup run"""

import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "niki/2024-05-28_niki_084"

PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES = [
    ("EleutherAI/pythia-2.8b", "pythia-pm-2.8b-niki-ada-v4"),
    ("EleutherAI/pythia-6.9b", "pythia-pm-6.9b-niki-ada-v4"),
    ("EleutherAI/pythia-12b", "pythia-pm-12b-niki-ada-v4"),
]
SEEDS = [0, 1, 2]
OVERRIDE_ARGS_LIST = [
    {
        "model.name_or_path": base_model_name,
        "training.force_name_to_save": f"{target_model_name}-s-{seed}",
        "training.seed": seed,
        "training.batch_size": 2,
        "model.train_minibatch_size": 2,
    }
    for (base_model_name, target_model_name) in PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES
    for seed in SEEDS
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        gpu=4,
        use_accelerate=True,
        memory="300G",
        priority="normal-batch",
    )
