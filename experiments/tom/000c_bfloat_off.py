# Test an eval with and without bfloat16.
# Based on ian_036.
from pathlib import Path

from robust_llm.batch_job_utils import run_multiple

FILE = Path(__file__)
EXPERIMENT_NAME = FILE.parent.name + "_" + FILE.stem
HYDRA_CONFIG = "tom/000_gen_pm_ihateyou_gcg"

MODELS_AND_N_MAX_PARALLEL = [
    ("EleutherAI/pythia-2.8b", 1),
]


OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model.name_or_path": model,
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
        memory="30G",
        cpu=1,
        priority="normal-batch",
    )
    EXPERIMENT_NAME = EXPERIMENT_NAME.replace("000c", "000d")
    EXPERIMENT_NAME = EXPERIMENT_NAME.replace("bfloat_off", "bfloat_on")
    for override in OVERRIDE_ARGS_LIST:
        override["model.dtype"] = "bfloat16"
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="30G",
        cpu=1,
        priority="normal-batch",
    )
