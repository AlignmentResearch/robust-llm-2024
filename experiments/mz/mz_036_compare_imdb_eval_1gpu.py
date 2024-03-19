import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "gcg_eval_imdb"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-imdb-160m-mz-ada-v2", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-1b-mz-ada-v2", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-2.8b-mz-ada-v2", 1),
]
N_ITS = 10
SEARCH_TYPES = ["gcg", "beam_search"]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            # We limit ourselves to 200 examples per evaluation.
            "experiment.evaluation.num_generated_examples": 200,
            "experiment.evaluation.evaluation_attack.search_based_attack_config.search_type": search_type,  # noqa: E501
            "experiment.environment.model_name_or_path": model,
            "experiment.evaluation.evaluation_attack.search_based_attack_config.n_its": N_ITS,  # noqa: E501
            "experiment.evaluation.batch_size": 32,
            "experiment.evaluation.evaluation_attack.search_based_attack_config.forward_pass_batch_size": 32,  # noqa: E501
        },
        n_max_parallel,
    )
    for search_type in SEARCH_TYPES
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
    )
