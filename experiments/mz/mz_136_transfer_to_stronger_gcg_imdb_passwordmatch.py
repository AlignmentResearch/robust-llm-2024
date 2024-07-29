"""IMDB and Passwordmatch experiments with stronger models.

Based on ian_024.
These models were trained on n_its=10; we try n_its=25.
"""

import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
DATASETS = ["IMDB", "PasswordMatch"]
HYDRA_CONFIG = "20240430_mz_136"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-14m_mz-130_{dataset}_n-its-10-seed-0", 2),
    ("AlignmentResearch/robust_llm_pythia-31m_mz-130_{dataset}_n-its-10-seed-0", 2),
    ("AlignmentResearch/robust_llm_pythia-70m_mz-130_{dataset}_n-its-10-seed-0", 2),
    ("AlignmentResearch/robust_llm_pythia-160m_mz-130_{dataset}_n-its-10-seed-0", 2),
    ("AlignmentResearch/robust_llm_pythia-410m_mz-130_{dataset}_n-its-10-seed-0", 1),
    ("AlignmentResearch/robust_llm_pythia-1b_mz-130_{dataset}_n-its-10-seed-0", 1),
]

OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "experiment.environment.model_name_or_path": model.format(dataset=dataset),
            "experiment.dataset.dataset_type": f"AlignmentResearch/{dataset}",
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
    for dataset in DATASETS
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="50G",
        cpu=12,
        priority="normal-batch",
    )
