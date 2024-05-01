"""Multi-seed experiments for WordLength."""

import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# get just the <name>_<exp_number> part with a hyphen instead of underscore
EXP_PREFIX = "-".join(EXPERIMENT_NAME.split("_")[:2])
HYDRA_CONFIG = "20240429_mz_133"

MODELS_AND_N_MAX_PARALLEL = [
    ("pythia-14m", 4),
    ("pythia-31m", 4),
    ("pythia-70m", 2),
    ("pythia-160m", 2),
    ("pythia-410m", 1),
    ("pythia-1b", 1),
]
DATASETS = ["WordLength"]
N_ITS = [10]
SEEDS = list(range(5))

OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "experiment.environment.model_name_or_path": f"EleutherAI/{model}",
            "experiment.training.force_name_to_save": f"{model}_{EXP_PREFIX}_{dataset}_n-its-{n_its}-seed-{seed}",  # noqa: E501
            "experiment.dataset.dataset_type": f"AlignmentResearch/{dataset}",
            "experiment.training.iterative.training_attack.search_based_attack_config.n_its": n_its,  # noqa: E501
            "experiment.evaluation.evaluation_attack.search_based_attack_config.n_its": n_its,  # noqa: E501
            "experiment.training.seed": seed,
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
    for dataset in DATASETS
    for n_its in N_ITS
    for seed in SEEDS
]
OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="60G",
        priority="normal-batch",
    )
