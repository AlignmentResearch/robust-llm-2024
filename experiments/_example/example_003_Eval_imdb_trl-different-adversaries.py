import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "Eval/pm_trl"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-tt-14m-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-tt-31m-mz-ada-v3", 1),
]

ADVERSARY_MODEL_CONFIGS = ["pythia-14m", "pythia-31m", "pythia-70m"]

OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model.name_or_path": model,
            "dataset.n_val": 200,
            # We have to use + here because we are setting a whole config rather
            # than a single value, and evaluation.evaluation_attack.adversary
            # does not already appear in the Defaults List of any config. See
            # the README for more discussion.
            "+model@evaluation.evaluation_attack.adversary": adversary,
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
    for adversary in ADVERSARY_MODEL_CONFIGS
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
        dry_run=True,
    )
