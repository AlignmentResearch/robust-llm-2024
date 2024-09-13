import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
EXP_PREFIX = "-".join(EXPERIMENT_NAME.split("_")[:2])
HYDRA_CONFIG = "niki/0_DEBUG_niki_126"

OVERRIDE_ARGS_LIST = [
    {
        "training.model_save_path_prefix_or_hf": None,
    },
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        cpu=8,
        # gpu=1,  # NOTE: this works fine
        gpu=2,  # TODO: make this not fail
        memory="60G",
        cluster="a6k",
        priority="normal-batch",
    )
