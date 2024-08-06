import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "Training/imdb"

MODEL_CONFIGS_AND_FINETUNED_TARGET_NAMES = [
    ("EleutherAI/pythia-14m", "pythia-tt-14m-ian-ada"),
    ("EleutherAI/pythia-31m", "pythia-tt-31m-ian-ada"),
]
OVERRIDE_ARGS_LIST = [
    {
        "model": model_config,
        "training.force_name_to_save": target_model_name,
        "training.optimizer": "adafactor",
        "training.save_strategy": "no",
        "training.gradient_checkpointing": True,
        "training.log_full_datasets_to_wandb": True,
    }
    for (model_config, target_model_name) in MODEL_CONFIGS_AND_FINETUNED_TARGET_NAMES
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="60G",
        priority="normal-batch",
        # Set to False to actually run this.
        dry_run=True,
    )
