"""Model trained on suffix attack. Attack with prefix and with 90% through."""

from pathlib import Path

import numpy as np

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = "tom_" + Path(__file__).stem

ADV_TRAINING_EXPERIMENTS = ["niki_149", "niki_150", "niki_151", "niki_153"]
ATTACK = "rt"
DATASETS = ["imdb", "spam", "pm", "wl"]
HYDRA_CONFIGS = [
    "ian/iclr2024_stronger_{attack}_{dataset}".format(attack=ATTACK, dataset=dataset)
    for dataset in DATASETS
]

BASE_MODELS = [
    "pythia-14m",
    "pythia-31m",
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-2.8b",
]
N_PARALLEL = [4, 3, 2, 2, 1, 1, 1]
FINETUNE_SEEDS = [0]  # this is "wrong" but matches niki_149
ADV_TRAIN_SEEDS = [0, 1, 2, 3, 4]
PROJECT = "AlignmentResearch/robust_llm"

# niki_149 script says 6 rounds for 2.8b but I only see 5 rounds in HF
N_ADV_TR_ROUNDS = [np.clip(x, 5, 250) for x in [953, 413, 163, 59, 21, 8, 5, 3, 1, 1]]
TR_ROUNDS_TO_EVAL = [
    [
        x
        for x in sorted(
            set(
                np.concatenate(
                    [
                        np.arange(10),
                        np.linspace(10, n_rounds - 10, 10).astype(int),
                        np.arange(n_rounds - 10, n_rounds),
                    ]
                )
            )
        )
        if 0 <= x <= n_rounds
    ]
    for n_rounds in N_ADV_TR_ROUNDS
]


def get_overrides_and_parallelism(dataset: str):
    overrides_and_parallelism = [
        (
            {
                "+model": f"Default/clf/{dataset}/{base_model}-s{finetune_seed}",
                "model.name_or_path": (
                    f"{PROJECT}_clf_{dataset}_{base_model}"
                    f"_s-{finetune_seed}_adv_tr_{ATTACK}_t-{adv_tr_seed}"
                ),
                "model.revision": f"adv-training-round-{tr_round}",
            },
            n_parallel,
        )
        for base_model, n_parallel, tr_rounds in zip(
            BASE_MODELS, N_PARALLEL, TR_ROUNDS_TO_EVAL
        )
        for tr_round in tr_rounds
        for finetune_seed in FINETUNE_SEEDS
        for adv_tr_seed in ADV_TRAIN_SEEDS
    ]
    return tuple(list(t) for t in zip(*overrides_and_parallelism))


if __name__ == "__main__":
    for i, (exp, dataset, hydra_config) in enumerate(
        zip(ADV_TRAINING_EXPERIMENTS, DATASETS, HYDRA_CONFIGS)
    ):
        overrides, n_parallel = get_overrides_and_parallelism(dataset)
        # Make one run high priority just to see if the runs are actually working
        priorities = ["high-batch"] * n_parallel[0] + ["normal-batch"] * (
            len(overrides) - n_parallel[0]
        )
        dry_run = False

        # # debugging
        # overrides = overrides[:1]
        # n_parallel = n_parallel[:1]
        # priorities = priorities[:1]
        # dry_run = True

        run_multiple(
            EXPERIMENT_NAME.replace("005a", "005" + chr(ord("a") + i)).replace(
                "niki_149", exp
            ),
            hydra_config,
            overrides,
            n_parallel,
            memory="50G",
            cpu=1,
            priority=priorities,
            dry_run=dry_run,
        )
