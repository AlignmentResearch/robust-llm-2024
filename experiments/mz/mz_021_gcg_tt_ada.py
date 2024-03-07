import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "search_based_eval_tensor_trust"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-tt-14m-mz-ada-v2", 4),
    ("AlignmentResearch/robust_llm_pythia-tt-31m-mz-ada-v2", 4),
    ("AlignmentResearch/robust_llm_pythia-tt-70m-mz-ada-v2", 2),
    ("AlignmentResearch/robust_llm_pythia-tt-160m-mz-ada-v2", 2),
    ("AlignmentResearch/robust_llm_pythia-tt-410m-mz-ada-v2", 1),
    ("AlignmentResearch/robust_llm_pythia-tt-1b-mz-ada-v2", 1),
    ("AlignmentResearch/robust_llm_pythia-tt-1.4b-mz-ada-v2", 1),
    ("AlignmentResearch/robust_llm_pythia-tt-2.8b-mz-ada-v2", 1),
]
N_ITS = [1, 3, 10, 30]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            # We limit ourselves to 500 examples per evaluation.
            "experiment.evaluation.num_generated_examples": 500,
            "experiment.environment.model_name_or_path": model,
            "experiment.evaluation.evaluation_attack.search_based_attack_config.n_its": n_its,  # noqa: E501
            "experiment.evaluation.batch_size": 64,
            "experiment.evaluation.evaluation_attack.search_based_attack_config.forward_pass_batch_size": 64,  # noqa: E501
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
