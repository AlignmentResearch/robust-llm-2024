import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_train_word_length"

PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES = [
    ("EleutherAI/pythia-14m", "pythia-wl-14m-mz-ada-v3"),
    ("EleutherAI/pythia-31m", "pythia-wl-31m-mz-ada-v3"),
    ("EleutherAI/pythia-70m-deduped", "pythia-wl-70m-mz-ada-v3"),
    ("EleutherAI/pythia-160m-deduped", "pythia-wl-160m-mz-ada-v3"),
    ("EleutherAI/pythia-410m-deduped", "pythia-wl-410m-mz-ada-v3"),
    ("EleutherAI/pythia-1b-deduped", "pythia-wl-1b-mz-ada-v3"),
]
SEEDS = [0, 1, 2, 3, 4]
OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": base_model_name,
        "experiment.training.force_name_to_save": (
            target_model_name if seed == 0 else f"{target_model_name}-s-{seed}"
        ),
        "experiment.training.optimizer": "adafactor",
        "experiment.training.save_strategy": "no",
        "experiment.training.gradient_checkpointing": True,
        "experiment.training.seed": seed,
    }
    for (base_model_name, target_model_name) in PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES
    for seed in SEEDS
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="60G",
        priority="normal-batch",
    )
