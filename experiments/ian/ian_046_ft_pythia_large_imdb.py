import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/046_ft_pythia_large_imdb"

PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES = [
    ("EleutherAI/pythia-6.9b", "pythia-6.9b-imdb-ian-nd"),
    ("EleutherAI/pythia-12b", "pythia-12b-imdb-ian-nd"),
]
OVERRIDE_ARGS_LIST = [
    {
        "+model": base_model_name,
        "training.force_name_to_save": target_model_name,
    }
    for (
        base_model_name,
        target_model_name,
    ) in PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES  # noqa: E501
]
if __name__ == "__main__":
    run_multiple(
        experiment_name=EXPERIMENT_NAME,
        hydra_config=HYDRA_CONFIG,
        gpu=4,
        override_args_list=OVERRIDE_ARGS_LIST,
        memory="200G",
    )
