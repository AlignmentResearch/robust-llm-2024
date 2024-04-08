import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_train_spam"

PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES = [
    ("EleutherAI/pythia-14m", "pythia-spam-14m-mz-ada-v3"),
    ("EleutherAI/pythia-31m", "pythia-spam-31m-mz-ada-v3"),
    ("EleutherAI/pythia-70m-deduped", "pythia-spam-70m-mz-ada-v3"),
    ("EleutherAI/pythia-160m-deduped", "pythia-spam-160m-mz-ada-v3"),
    ("EleutherAI/pythia-410m-deduped", "pythia-spam-410m-mz-ada-v3"),
    ("EleutherAI/pythia-1b-deduped", "pythia-spam-1b-mz-ada-v3"),
]
CHECKPOINTS = [
    139_000,
    140_000,
    141_000,
    142_000,
    # 143_000,  # 100%
]
OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": base_model_name,
        "experiment.training.force_name_to_save": f"{target_model_name}-ch-{checkpoint}",  # noqa: E501
        "experiment.training.optimizer": "adafactor",
        "experiment.training.save_strategy": "no",
        "experiment.training.gradient_checkpointing": True,
        "experiment.training.checkpoint": checkpoint,
    }
    for (base_model_name, target_model_name) in PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES
    for checkpoint in CHECKPOINTS
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="60G",
        priority="normal-batch",
    )
