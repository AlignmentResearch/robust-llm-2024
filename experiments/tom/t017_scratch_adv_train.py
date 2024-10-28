"""Tiny adv training run to test model saving."""

from pathlib import Path

from robust_llm.batch_job_utils import run_multiple

PATH = Path(__file__)
EXPERIMENT_NAME = f"{PATH.parent.name}_{PATH.stem.split('_')[0].removeprefix('t')}"

DATASET = "pm"
ATTACK = "rt"
HYDRA_CONFIG = f"niki/iclr2024_adv_train_{ATTACK}_{DATASET}"
MODEL = "pythia-14m"
SEED = 0

OVERRIDE_ARGS_LIST = [
    {
        "+model": f"Default/clf/{DATASET}/{MODEL}-s{SEED}",
        "training.save_name": (
            f"tom_scratch_clf_{DATASET}_{MODEL}_s-{SEED}_adv_tr_{ATTACK}_t-{SEED}"
        ),
        "environment.allow_checkpointing": False,
        "training.save_to": "DISK",
        "training.seed": SEED,
        "dataset.n_train": 10,
        "evaluation.num_iterations": 0,
        "training.adversarial.attack_schedule.start": 1,
        "training.adversarial.attack_schedule.end": 1,
        "training.adversarial.num_examples_to_generate_each_round": 1,
        "training.adversarial.num_adversarial_training_rounds": 2,
        "training.adversarial.training_attack.n_attack_tokens": 1,
    },
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        gpu=0,
        memory="30G",
        n_max_parallel=4,
        cpu=1,
        wandb_mode="offline",
    )
