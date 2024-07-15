# Rerun 050 to get a sense of variablity for 0.5/1.8B and
# get 4/7/14B to not OOM.
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/051_gen_strongreject_gcg"

MODELS_N_GPUS_MEMORY = [
    ("google/gemma-1.1-2b-it", 1, "20G"),
    ("google/gemma-1.1-7b-it", 1, "50G"),
    ("google/gemma-2-9b-it", 1, "60G"),
    ("google/gemma-2-27b-it", 2, "150G"),
]

OVERRIDE_TUPLES = [
    (
        {
            "+model": model,
        },
        n_gpus,
        memory,
    )
    for (model, n_gpus, memory) in MODELS_N_GPUS_MEMORY
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_TUPLES]
N_GPUS = [x[1] for x in OVERRIDE_TUPLES]
MEMORY = [x[2] for x in OVERRIDE_TUPLES]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        gpu=N_GPUS,
        memory=MEMORY,
        cpu=8,
        priority="normal-batch",
    )
