import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "search_based_eval_tensor_trust"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-tt-14m-mz-v0", 16),
    ("AlignmentResearch/robust_llm_pythia-tt-31m-mz-v0", 16),
    ("AlignmentResearch/robust_llm_pythia-tt-70m-mz-v0", 2),
    ("AlignmentResearch/robust_llm_pythia-tt-160m-mz-v0", 2),
    ("AlignmentResearch/robust_llm_pythia-tt-410m-mz-v0", 1),
    ("AlignmentResearch/robust_llm_pythia-tt-1b-mz-v0", 1),
]
N_ITS = 10
N_CANDIDATES_PER_IT = [256, 512, 1024]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            # We limit ourselves to 500 examples per evaluation.
            "experiment.evaluation.num_generated_examples": 500,
            "experiment.environment.model_name_or_path": model,
            "experiment.evaluation.evaluation_attack.search_based_attack_config.n_its": N_ITS,  # noqa: E501
            "experiment.evaluation.evaluation_attack.search_based_attack_config.n_candidates_per_it": n_candidates_per_it,  # noqa: E501
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
    for n_candidates_per_it in N_CANDIDATES_PER_IT
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
