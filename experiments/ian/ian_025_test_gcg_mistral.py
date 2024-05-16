import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "Eval/pm_gcg-standard"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_alias-ian-tt-v0", 2),
    ("AlignmentResearch/robust_llm_battlestar-ian-tt-v0", 2),
    ("AlignmentResearch/robust_llm_arwen-ian-tt-v0", 2),
    ("AlignmentResearch/robust_llm_beren-ian-tt-v0", 2),
]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model": "gpt2",
            "model.name_or_path": model,
            "model.strict_load": True,
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
        cpu=12,
        priority="normal-batch",
    )
