"""Model originally trained on random token. Now, evaluate on gcg."""

import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "niki/2024-05-21_niki_066"

BASE_MODEL_NAMES_AND_MAX_PARALLEL = [
    ("EleutherAI/pythia-14m", 4),
    ("EleutherAI/pythia-31m", 3),
    ("EleutherAI/pythia-70m", 2),
    ("EleutherAI/pythia-160m", 2),
    ("EleutherAI/pythia-410m", 1),
    ("EleutherAI/pythia-1b", 1),
]
SEEDS = list(range(3))
ADV_TRAINING_ROUNDS = list(range(10))
project_prefix = "AlignmentResearch/robust_llm"
experiment_prefix = "niki-062"

OVERRIDE_ARGS_LIST_AND_MAX_PARALLEL = [
    (
        {
            # Load from HFHub
            "model.name_or_path": f"{project_prefix}_{base_model_name[11:]}_pretrained_{experiment_prefix}_pm_gcg_seed-{seed}",  # noqa: E501
            "model.revision": f"adv-training-round-{adv_training_round}",
        },
        max_parallel,
    )
    for base_model_name, max_parallel in BASE_MODEL_NAMES_AND_MAX_PARALLEL
    for seed in SEEDS
    for adv_training_round in ADV_TRAINING_ROUNDS
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_MAX_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_MAX_PARALLEL]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="50G",
        priority="normal-batch",
    )
