import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_train_tensor_trust"

PYTHIA_NAMES_AND_FINETUNED_TARGET_NAMES = [
    ("EleutherAI/pythia-1.4b-deduped", "pythia-tt-1.4b-mz-ada-v3"),
    ("EleutherAI/pythia-2.8b-deduped", "pythia-tt-2.8b-mz-ada-v3"),
]
CHECKPOINTS = [
    100_000,  # 70%
    103_000,
    106_000,
    109_000,
    112_000,
    114_000,  # 80%
    117_000,
    120_000,
    123_000,
    126_000,
    129_000,  # 90%
    132_000,
    135_000,
    138_000,
    141_000,
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
