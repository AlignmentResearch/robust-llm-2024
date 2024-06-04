import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/031_gen_ft_pythia_imdb"

PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES_AND_PARALLEL = [
    ("EleutherAI/pythia-14m", "pythia-14m-imdb-gen-ian-nd", 2),
    ("EleutherAI/pythia-31m", "pythia-31m-imdb-gen-ian-nd", 2),
    ("EleutherAI/pythia-70m", "pythia-70m-imdb-gen-ian-nd", 2),
    ("EleutherAI/pythia-160m", "pythia-160m-imdb-gen-ian-nd", 2),
    ("EleutherAI/pythia-410m", "pythia-410m-imdb-gen-ian-nd", 1),
    ("EleutherAI/pythia-1b", "pythia-1b-imdb-gen-ian-nd", 1),
    ("EleutherAI/pythia-2.8b", "pythia-2.8b-imdb-gen-ian-nd", 1),
]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model.name_or_path": base_model_name,
            "training.force_name_to_save": target_model_name,
        },
        n_max_parallel,
    )
    for (
        base_model_name,
        target_model_name,
        n_max_parallel,
    ) in PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES_AND_PARALLEL  # noqa: E501
]
OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]

if __name__ == "__main__":
    run_multiple(
        experiment_name=EXPERIMENT_NAME,
        hydra_config=HYDRA_CONFIG,
        n_max_parallel=N_MAX_PARALLEL,
        override_args_list=OVERRIDE_ARGS_LIST,
        memory="30G",
    )
