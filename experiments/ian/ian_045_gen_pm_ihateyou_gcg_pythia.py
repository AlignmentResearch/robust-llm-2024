# Same experiment as 040, but with pythia models rather than qwen models.
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# We use the same Hydra config as in 043 since it actually doesn't mention qwen.
HYDRA_CONFIG = "ian/043_gen_pm_ihateyou_gcg_qwen_base"

MODELS_AND_N_MAX_PARALLEL = [
    ("EleutherAI/pythia-1b", 1),
    ("EleutherAI/pythia-2.8b", 1),
    ("EleutherAI/pythia-6.9b", 1),
    ("EleutherAI/pythia-12b", 1),
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
        experiment_name=EXPERIMENT_NAME,
        hydra_config=HYDRA_CONFIG,
        override_args_list=OVERRIDE_ARGS_LIST,
        n_max_parallel=N_MAX_PARALLEL,
        memory="60G",
        cpu=8,
        gpu=2,
        priority="normal-batch",
    )
