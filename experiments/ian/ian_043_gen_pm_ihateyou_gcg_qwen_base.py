# Same experiment as 040, but with base models rather than IFT models.
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/043_gen_pm_ihateyou_gcg_qwen_base"

MODELS_AND_N_MAX_PARALLEL = [
    ("Qwen/Qwen1.5-0.5B", 2),
    ("Qwen/Qwen1.5-1.8B", 2),
    ("Qwen/Qwen1.5-4B", 1),
    ("Qwen/Qwen1.5-7B", 1),
]


OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "+model": model,
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="60G",
        cpu=8,
        priority="normal-batch",
    )
