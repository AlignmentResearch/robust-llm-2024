# Rerun 050 to get a sense of variablity for 0.5/1.8B and
# get 4/7/14B to not OOM.
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/050_gen_strongreject_gcg_qwen"

MODELS_AND_N_GPUS = [
    ("Qwen/Qwen1.5-0.5B-Chat", 1),
    ("Qwen/Qwen1.5-1.8B-Chat", 1),
    ("Qwen/Qwen1.5-4B-Chat", 1),
    ("Qwen/Qwen1.5-7B-Chat", 1),
    ("Qwen/Qwen1.5-14B-Chat", 2),
]

OVERRIDE_ARGS_LIST_AND_N_GPUS = [
    (
        {
            "+model": model,
        },
        n_gpus,
    )
    for (model, n_gpus) in MODELS_AND_N_GPUS
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_N_GPUS]
N_GPUS = [x[1] for x in OVERRIDE_ARGS_LIST_AND_N_GPUS]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        gpu=N_GPUS,
        memory="120G",
        cpu=8,
        priority="normal-batch",
    )
