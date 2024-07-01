import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "niki/2024-05-30_niki_096"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-tt-14m-mz-ada-v3", 4),
    ("AlignmentResearch/robust_llm_pythia-tt-31m-mz-ada-v3", 4),
    ("AlignmentResearch/robust_llm_pythia-tt-70m-mz-ada-v3", 2),
    ("AlignmentResearch/robust_llm_pythia-tt-160m-mz-ada-v3", 2),
    ("AlignmentResearch/robust_llm_pythia-tt-410m-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-tt-1b-mz-ada-v3", 1),
]
SEED_SUFFIXES = ["", "-s-1", "-s-2", "-s-3"]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model.name_or_path": f"{model}{seed_suffix}",
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
    for seed_suffix in SEED_SUFFIXES
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="100G",
        cpu=12,
        priority="high-batch",
    )
