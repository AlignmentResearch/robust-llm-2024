import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
EXP_PREFIX = "-".join(EXPERIMENT_NAME.split("_")[:2])
HYDRA_CONFIG = "niki/2024-05-03_niki_044"

PYTHIA_NAMES = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
]
SEEDS = list(range(5))
NUM_ROUNDS = 30
OVERRIDE_ARGS_LIST = [
    {
        "model.name_or_path": base_model_name,
        # Save to HF since we want to use these random token hardened models for
        # the beam search attack as well
        "training.force_name_to_save": f"{base_model_name[11:]}_{EXP_PREFIX}_imdb_random-token-1280_{NUM_ROUNDS}-rounds_seed-{seed}",  # noqa: E501
        # Defense uses random token 1280, attack uses gcg 10 iterations
        "training.seed": seed,
        "training.adversarial.num_adversarial_training_rounds": NUM_ROUNDS,
    }
    for base_model_name in PYTHIA_NAMES
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
