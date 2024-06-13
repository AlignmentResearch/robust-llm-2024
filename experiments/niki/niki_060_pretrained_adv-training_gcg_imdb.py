import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
EXP_PREFIX = "-".join(EXPERIMENT_NAME.split("_")[:2])
HYDRA_CONFIG = "niki/2024-05-17_niki_060"
MAX_N_PARALLEL = [2, 2, 2, 1, 1, 1]
BASE_NAMES = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
]
PRETRAINED_NAMES = [
    "AlignmentResearch/robust_llm_pythia-imdb-14m-mz-ada-v3",
    "AlignmentResearch/robust_llm_pythia-imdb-31m-mz-ada-v3",
    "AlignmentResearch/robust_llm_pythia-imdb-70m-mz-ada-v3",
    "AlignmentResearch/robust_llm_pythia-imdb-160m-mz-ada-v3",
    "AlignmentResearch/robust_llm_pythia-imdb-410m-mz-ada-v3",
    "AlignmentResearch/robust_llm_pythia-imdb-1b-mz-ada-v3",
]
SEEDS = list(range(3))
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model.name_or_path": warmstart_name,
            "training.force_name_to_save": f"{base_model_name[11:]}_pretrained_{EXP_PREFIX}_imdb_gcg-1280_seed-{seed}",  # noqa: E501
            "training.seed": seed,
        },
        max_n_parallel,
    )
    for base_model_name, warmstart_name, max_n_parallel in zip(
        BASE_NAMES, PRETRAINED_NAMES, MAX_N_PARALLEL
    )
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
