import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "random_token_eval_imdb"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-imdb-14m-mz-ada-v3", 4),
    ("AlignmentResearch/robust_llm_pythia-imdb-31m-mz-ada-v3", 4),
    ("AlignmentResearch/robust_llm_pythia-imdb-70m-mz-ada-v3", 2),
    ("AlignmentResearch/robust_llm_pythia-imdb-160m-mz-ada-v3", 2),
    ("AlignmentResearch/robust_llm_pythia-imdb-410m-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-1b-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-1.4b-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-2.8b-mz-ada-v3", 1),
]

# Corresponds to 10 iterations of 128 candidates per iteration in GCG.
MAX_ITERATIONS = [1280]

OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "experiment.environment.model_name_or_path": model,
            "experiment.evaluation.evaluation_attack.random_token_attack_config.max_iterations": max_iterations,  # noqa: E501
            "experiment.dataset.revision": "0.0.1",
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
    for max_iterations in MAX_ITERATIONS
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
        cpu=12,
        priority="normal-batch",
    )
