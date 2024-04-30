import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "Eval/pm_random-token-len-10"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-tt-14m-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-tt-31m-mz-ada-v3", 1),
]

# Corresponds to 10 iterations of 128 candidates per iteration in GCG.
MAX_ITERATIONS = [128, 1280]

OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model.name_or_path": model,
            "dataset.n_val": 200,
            "evaluation.evaluation_attack.max_iterations": max_iterations,  # noqa: E501
            "evaluation.batch_size": 32,  # noqa: E501
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
        dry_run=True,
    )
