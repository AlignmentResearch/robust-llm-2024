import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "Eval/pm_random-token-1280-its"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-tt-14m-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-tt-31m-mz-ada-v3", 1),
]

ATTACK_CONFIGS = ["GCG", "random-token-1280-its"]

OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "model.name_or_path": model,
            "attack@evaluation.evaluation_attack": attack,
            "dataset.n_val": 200,
            "evaluation.batch_size": 32,  # noqa: E501
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
    for attack in ATTACK_CONFIGS
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
