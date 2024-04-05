import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_train_tensor_trust"

INPUT_NAMES_AND_FINETUNED_TARGET_NAMES = [
    ("stanford-crfm/arwen-gpt2-medium-x21", "arwen-ian-tt-v0"),
    ("stanford-crfm/beren-gpt2-medium-x49", "beren-ian-tt-v0"),
]
OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": base_model_name,
        "experiment.training.force_name_to_save": target_model_name,
        "experiment.training.optimizer": "adafactor",
        "experiment.training.save_strategy": "no",
        "experiment.training.gradient_checkpointing": True,
        "experiment.training.log_datasets_to_wandb": False,
        "experiment.environment.model_family": "gpt2",
    }
    for (base_model_name, target_model_name) in INPUT_NAMES_AND_FINETUNED_TARGET_NAMES
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="60G",
        priority="normal-batch",
    )
