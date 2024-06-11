# Similar to 039 but with bug fixes and a different config setup.
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/041_gen_pm_gcg_qwen"

MODELS_AND_N_MAX_PARALLEL = [
    ("Qwen/Qwen1.5-0.5B-Chat", 2),
    ("Qwen/Qwen1.5-1.8B-Chat", 2),
    ("Qwen/Qwen1.5-4B-Chat", 1),
    ("Qwen/Qwen1.5-7B-Chat", 1),
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
