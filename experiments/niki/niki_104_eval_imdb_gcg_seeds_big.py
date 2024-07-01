"""Model originally trained on random token. Now, evaluate on gcg."""

import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "niki/gcg_imdb_eval"

BASE_MODEL_NAMES_AND_MAX_PARALLEL = [
    ("pythia-imdb-6.9b-niki-ada-v4", 1),
    ("pythia-imdb-12b-niki-ada-v4", 1),
]

SEEDS = [0, 1, 2]
project_prefix = "AlignmentResearch/robust_llm"

OVERRIDE_ARGS_LIST_AND_MAX_PARALLEL = [
    (
        {
            # Load from HFHub
            "model.name_or_path": f"{project_prefix}_{base_model_name}-s-{seed}",  # noqa: E501
        },
        max_parallel,
    )
    for base_model_name, max_parallel in BASE_MODEL_NAMES_AND_MAX_PARALLEL
    for seed in SEEDS
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_MAX_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_MAX_PARALLEL]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="200G",
        cpu=12,
        gpu=2,
        priority="normal-batch",
        dry_run=True,
    )
