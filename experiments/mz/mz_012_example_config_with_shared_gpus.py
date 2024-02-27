import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "gcg_eval_tensor_trust"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-tt-14m-mz-v0", 16),
    ("AlignmentResearch/robust_llm_pythia-tt-31m-mz-v0", 16),
    ("AlignmentResearch/robust_llm_pythia-tt-70m-mz-v0", 2),
    ("AlignmentResearch/robust_llm_pythia-tt-160m-mz-v0", 2),
    ("AlignmentResearch/robust_llm_pythia-tt-410m-mz-v0", 1),
    ("AlignmentResearch/robust_llm_pythia-tt-1b-mz-v0", 1),
]
N_ITS = [10, 50]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "experiment.environment.train_set_size": 100,
            "experiment.environment.validation_set_size": 100,
            "experiment.environment.model_name_or_path": model,
            "experiment.evaluation.evaluation_attack.gcg_attack_config.n_its": n_its,
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
    for n_its in N_ITS
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="50G",
        cpu=12,
    )
