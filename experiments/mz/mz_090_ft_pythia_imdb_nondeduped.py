import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_train_imdb"

PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES = [
    ("EleutherAI/pythia-14m", "pythia-imdb-14m-mz-ada-v3-nd"),
    ("EleutherAI/pythia-31m", "pythia-imdb-31m-mz-ada-v3-nd"),
    ("EleutherAI/pythia-70m", "pythia-imdb-70m-mz-ada-v3-nd"),
    ("EleutherAI/pythia-160m", "pythia-imdb-160m-mz-ada-v3-nd"),
    ("EleutherAI/pythia-410m", "pythia-imdb-410m-mz-ada-v3-nd"),
    ("EleutherAI/pythia-1b", "pythia-imdb-1b-mz-ada-v3-nd"),
]
OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": base_model_name,
        "experiment.training.force_name_to_save": target_model_name,
        "experiment.training.optimizer": "adafactor",
        "experiment.training.gradient_checkpointing": True,
    }
    for (base_model_name, target_model_name) in PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES
]

if __name__ == "__main__":
    run_multiple(EXPERIMENT_NAME, HYDRA_CONFIG, OVERRIDE_ARGS_LIST, memory="60G")
