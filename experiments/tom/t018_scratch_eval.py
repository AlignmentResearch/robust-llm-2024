"""Tiny eval against t017 to test model loading."""

from pathlib import Path

from robust_llm.batch_job_utils import run_multiple

PATH = Path(__file__)
EXPERIMENT_NAME = f"{PATH.parent.name}_{PATH.stem.split('_')[0].removeprefix('t')}"

DATASET = "pm"
ATTACK = "rt"  # attack for both train and eval, for simplicity
HYDRA_CONFIG = f"ian/iclr2024_stronger_{ATTACK}_{DATASET}"
MODEL = "pythia-14m"
SEED = 0
ADV_TRAIN_ROUND = 1

OVERRIDE_ARGS_LIST = [
    {
        "+model": f"Default/clf/{DATASET}/{MODEL}-s{SEED}",
        "model.name_or_path": (
            f"AlignmentResearch/robust_llm_tom_scratch_clf_{DATASET}_{MODEL}"
            f"_s-{SEED}_adv_tr_{ATTACK}_t-{SEED}"
        ),
        "model.revision": f"adv-training-round-{ADV_TRAIN_ROUND}",
        "evaluation.num_iterations": 1,
        "dataset.n_val": 20,
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
