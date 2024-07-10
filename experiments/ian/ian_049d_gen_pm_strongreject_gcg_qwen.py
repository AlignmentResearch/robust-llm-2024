# Run a bunch of seeds on 0.5B to reproduce combined_embeddings crash
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# Hack the experiment name to match 049 so it appears in the same wandb.
EXPERIMENT_NAME = EXPERIMENT_NAME.replace("049a", "049")
HYDRA_CONFIG = "ian/049d_gen_pm_strongreject_gcg_qwen"

MODELS_AND_N_MAX_PARALLEL = [
    ("Qwen/Qwen1.5-0.5B-Chat", 1),
]

SEEDS = list(range(1, 11))

OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "+model": model,
            "evaluation.evaluation_attack.seed": seed,
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
    for seed in SEEDS
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="30G",
        cpu=4,
        priority="normal-batch",
        container_tag="strongreject",
    )
