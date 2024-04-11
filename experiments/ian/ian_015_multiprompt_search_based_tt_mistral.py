import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "multiprompt_search_based_eval_tensor_trust"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_alias-ian-tt-v0", 1),
    ("AlignmentResearch/robust_llm_battlestar-ian-tt-v0", 1),
    ("AlignmentResearch/robust_llm_caprica-ian-tt-v0", 1),
    ("AlignmentResearch/robust_llm_darkmatter-ian-tt-v0", 1),
    ("AlignmentResearch/robust_llm_expanse-ian-tt-v0", 1),
    ("AlignmentResearch/robust_llm_arwen-ian-tt-v0", 1),
    ("AlignmentResearch/robust_llm_beren-ian-tt-v0", 1),
    ("AlignmentResearch/robust_llm_celebrimbor-ian-tt-v0", 1),
    ("AlignmentResearch/robust_llm_durin-ian-tt-v0", 1),
    ("AlignmentResearch/robust_llm_eowyn-ian-tt-v0", 1),
]
N_ITS = [1, 2, 4, 8, 16, 32]
SEARCH_TYPES = ["multiprompt_gcg"]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "experiment.evaluation.num_generated_examples": 200,
            "experiment.evaluation.evaluation_attack.search_based_attack_config.search_type": search_type,  # noqa: E501
            "experiment.environment.model_name_or_path": model,
            "experiment.environment.model_family": "gpt2",
            "experiment.evaluation.evaluation_attack.search_based_attack_config.n_its": n_its,  # noqa: E501
            "experiment.evaluation.evaluation_attack.search_based_attack_config.n_candidates_per_it": 128,  # noqa: E501
            "experiment.evaluation.batch_size": 32,
            "experiment.evaluation.evaluation_attack.search_based_attack_config.forward_pass_batch_size": 32,  # noqa: E501
        },
        n_max_parallel,
    )
    for (model, n_max_parallel) in MODELS_AND_N_MAX_PARALLEL
    for search_type in SEARCH_TYPES
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
        priority="normal-batch",
    )
