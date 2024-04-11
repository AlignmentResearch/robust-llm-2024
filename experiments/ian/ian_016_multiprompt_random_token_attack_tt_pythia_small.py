import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "multiprompt_random_token_attack_tensor_trust"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-tt-14m-mz-ada-v3", 4),
    ("AlignmentResearch/robust_llm_pythia-tt-31m-mz-ada-v3", 4),
    ("AlignmentResearch/robust_llm_pythia-tt-70m-mz-ada-v3", 2),
    ("AlignmentResearch/robust_llm_pythia-tt-160m-mz-ada-v3", 2),
    ("AlignmentResearch/robust_llm_pythia-tt-410m-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-tt-1b-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-tt-1.4b-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-tt-2.8b-mz-ada-v3", 1),
]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "experiment.evaluation.num_generated_examples": 200,
            "experiment.environment.model_name_or_path": model,
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
        memory="50G",
        cpu=12,
        priority="normal-batch",
    )
