import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_train_word_length"

PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES = [
    ("EleutherAI/pythia-14m", "pythia-word-length-14m-niki-ada-v0"),
    ("EleutherAI/pythia-31m", "pythia-word-length-31m-niki-ada-v0"),
    ("EleutherAI/pythia-70m-deduped", "pythia-word-length-70m-niki-ada-v0"),
    ("EleutherAI/pythia-160m-deduped", "pythia-word-length-160m-niki-ada-v0"),
    ("EleutherAI/pythia-410m-deduped", "pythia-word-length-410m-niki-ada-v0"),
    ("EleutherAI/pythia-1b-deduped", "pythia-word-length-1b-niki-ada-v0"),
    ("EleutherAI/pythia-1.4b-deduped", "pythia-word-length-1.4b-niki-ada-v0"),
    ("EleutherAI/pythia-2.8b-deduped", "pythia-word-length-2.8b-niki-ada-v0"),
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
