"""Evaluate qwen ADV models, based on niki_167_eval_adv_tr_rt_spam_small"""

import os

from robust_llm.batch_job_utils import run_multiple
from robust_llm.experiment_utils import QWEN_EVAL_ROUNDS

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Just take the ian_xxx part as the version name, but use a hyphen to
# fit the convention of _ for separation.
VERSION_NAME = "-".join(EXPERIMENT_NAME.split("_")[:2])
DATASET = "spam"
ATTACK = "gcg"
HYDRA_CONFIG = f"ian/iclr2024_stronger_{ATTACK}_{DATASET}"


MODEL_GPU_MEMORY_CLUSTER_PARALLEL: list[tuple[str, int, str, str, int]] = [
    ("0.5B", 1, "40G", "a6k", 1),
    ("1.5B", 1, "80G", "a6k", 1),
    ("3B", 1, "100G", "a6k", 1),
    ("7B", 2, "150G", "h100", 1),
    ("14B", 2, "200G", "h100", 1),
]
SEEDS = [0, 1, 2]

OVERRIDE_TUPLES = [
    (
        {
            "+model": f"AdvTrained/clf/{ATTACK}/{DATASET}/Qwen2.5-{model}-s{seed}",
            "model.revision": f"adv-training-round-{evaluation_round}",
            "environment.minibatch_multiplier": 0.25,
            "evaluation.evaluation_attack.seed": seed,
            "evaluation.evaluation_attack.save_total_limit": 1024,
        },
        n_gpus,
        memory,
        cluster,
        parallel,
    )
    for model, n_gpus, memory, cluster, parallel in MODEL_GPU_MEMORY_CLUSTER_PARALLEL
    for seed in SEEDS
    for evaluation_round in QWEN_EVAL_ROUNDS[model]
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_TUPLES]
N_GPUS = [x[1] for x in OVERRIDE_TUPLES]
MEMORY = [x[2] for x in OVERRIDE_TUPLES]
CLUSTER = [x[3] for x in OVERRIDE_TUPLES]
PARALLEL = [x[4] for x in OVERRIDE_TUPLES]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        gpu=N_GPUS,
        memory=MEMORY,
        cluster=CLUSTER,
        n_max_parallel=PARALLEL,
        cpu=8,
        priority="normal-batch",
        container_tag="2024-11-03-14-15-57-main",
    )
