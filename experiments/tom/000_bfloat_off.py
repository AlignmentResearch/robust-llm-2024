# Test an eval with and without bfloat16.
# 000_bfloat_off.py and 000b_bfloat_on.py only differ in me hardcoding float32
# vs bfloat16 in the implementation, as I hadn't implemented making dtype a
# config flag yet.
# Based on ian_036.
from pathlib import Path

from robust_llm.batch_job_utils import run_multiple

FILE = Path(__file__)
EXPERIMENT_NAME = FILE.parent.name + "_" + FILE.stem
HYDRA_CONFIG = "tom/000_gen_pm_ihateyou_gcg"

MODELS_AND_N_MAX_PARALLEL = [
    ("EleutherAI/pythia-14m", 2),
    ("EleutherAI/pythia-31m", 2),
    ("EleutherAI/pythia-70m", 2),
    ("EleutherAI/pythia-160m", 2),
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
        memory="16G",
        cpu=1,
        priority="normal-batch",
    )
