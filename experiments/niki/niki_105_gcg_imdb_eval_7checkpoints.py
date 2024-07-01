import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "niki/2024-05-21_niki_050"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-imdb-14m-mz-ada-v3", 4),
    ("AlignmentResearch/robust_llm_pythia-imdb-31m-mz-ada-v3", 4),
    ("AlignmentResearch/robust_llm_pythia-imdb-70m-mz-ada-v3", 2),
    ("AlignmentResearch/robust_llm_pythia-imdb-160m-mz-ada-v3", 2),
    ("AlignmentResearch/robust_llm_pythia-imdb-410m-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-1b-mz-ada-v3", 1),
]

CHECKPOINT_SUFFIXES = [
    "-ch-100000",
    "-ch-107000",
    "-ch-114000",
    "-ch-121000",
    "-ch-129000",
    "-ch-136000",
    "",  # normal model which was trained on latest checkpoint
]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model.name_or_path": f"{model}{checkpoint_suffix}",
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
    for checkpoint_suffix in CHECKPOINT_SUFFIXES
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
