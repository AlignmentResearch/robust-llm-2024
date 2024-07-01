import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
EXP_PREFIX = "-".join(EXPERIMENT_NAME.split("_")[:2])
HYDRA_CONFIG = "niki/2024-05-28_niki_081"

PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES_AND_MAX_N_PARALLEL = [
    ("EleutherAI/pythia-14m", "pythia-imdb-14m-niki-ada-v4", 2),
    ("EleutherAI/pythia-31m", "pythia-imdb-31m-niki-ada-v4", 2),
    ("EleutherAI/pythia-70m", "pythia-imdb-70m-niki-ada-v4", 2),
    ("EleutherAI/pythia-160m", "pythia-imdb-160m-niki-ada-v4", 1),
    ("EleutherAI/pythia-410m", "pythia-imdb-410m-niki-ada-v4", 1),
    ("EleutherAI/pythia-1b", "pythia-imdb-1b-niki-ada-v4", 1),
    ("EleutherAI/pythia-1.4b", "pythia-imdb-1.4b-niki-ada-v4", 1),
]
SEEDS = [0, 1, 2]
OVERRIDE_ARGS_LIST_AND_MAX_N_PARALLEL = [
    (
        {
            "model.name_or_path": base_model_name,
            "training.force_name_to_save": f"{target_model_name}-s-{seed}",
            "training.seed": seed,
        },
        max_n_parallel,
    )
    for (
        base_model_name,
        target_model_name,
        max_n_parallel,
    ) in PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES_AND_MAX_N_PARALLEL
    for seed in SEEDS
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_MAX_N_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_MAX_N_PARALLEL]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="100G",
        priority="high-batch",
    )
