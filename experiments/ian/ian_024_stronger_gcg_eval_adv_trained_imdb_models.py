"""These models were trained on n_its=10; we try n_its=25"""

import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "gcg_eval_imdb"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-14m_ian-022_IMDB_n-its-10", 4),
    ("AlignmentResearch/robust_llm_pythia-31m_ian-022_IMDB_n-its-10", 4),
    ("AlignmentResearch/robust_llm_pythia-70m_ian-022_IMDB_n-its-10", 4),
    ("AlignmentResearch/robust_llm_pythia-160m_ian-022_IMDB_n-its-10", 3),
    ("AlignmentResearch/robust_llm_pythia-410m_ian-022_IMDB_n-its-10", 2),
    ("AlignmentResearch/robust_llm_pythia-1b_ian-022_IMDB_n-its-10", 1),
]

OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "experiment.environment.model_name_or_path": model,
            "experiment.evaluation.evaluation_attack.search_based_attack_config.n_its": 25,  # noqa: E501
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
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
