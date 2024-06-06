import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/037_gen_pm_hate_gcg"

MODELS_AND_N_MAX_PARALLEL = [
    ("EleutherAI/pythia-14m", 2),
    ("EleutherAI/pythia-31m", 2),
    ("EleutherAI/pythia-70m", 2),
    ("EleutherAI/pythia-160m", 2),
    ("EleutherAI/pythia-410m", 1),
    ("EleutherAI/pythia-1b", 1),
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
        cpu=6,
        priority="normal-batch",
    )
