# Rerun of 027 with slightly different models/memory usages/logging.
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/027_eval_refactored_random_token_imdb"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-imdb-14m-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-31m-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-70m-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-160m-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-410m-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-1b-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-1.4b-mz-ada-v3", 1),
    # ("AlignmentResearch/robust_llm_pythia-imdb-2.8b-mz-ada-v3", 1),
]


OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model.name_or_path": model,
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
        memory="10G",
        cpu=6,
        priority="high-batch",
    )
