import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
EXP_PREFIX = "-".join(EXPERIMENT_NAME.split("_")[:2])
HYDRA_CONFIG = "niki/0_DEBUG_niki_126"


base_model_name = "EleutherAI/pythia-14m"
n_max_parallel = 4

OVERRIDE_ARGS_LIST = [
    {
        "model.name_or_path": base_model_name,
    },
    {
        "model.name_or_path": base_model_name,
    },
]
N_MAX_PARALLEL = [n_max_parallel for _ in OVERRIDE_ARGS_LIST]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="60G",
        priority="high-batch",
        cluster="a6k",
    )
