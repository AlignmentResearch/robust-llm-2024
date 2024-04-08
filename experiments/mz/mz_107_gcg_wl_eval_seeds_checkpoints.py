import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "search_based_eval_word_length"

MODELS_AND_N_MAX_PARALLEL = [
    ("AlignmentResearch/robust_llm_pythia-wl-14m-mz-ada-v3", 4),
    ("AlignmentResearch/robust_llm_pythia-wl-31m-mz-ada-v3", 4),
    ("AlignmentResearch/robust_llm_pythia-wl-70m-mz-ada-v3", 2),
    ("AlignmentResearch/robust_llm_pythia-wl-160m-mz-ada-v3", 2),
    ("AlignmentResearch/robust_llm_pythia-wl-410m-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-wl-1b-mz-ada-v3", 1),
]
N_ITS = [3, 10]
SEARCH_TYPES = ["gcg"]
SUFFIXES = [
    "",
    "-s-1",
    "-s-2",
    "-s-3",
    "-s-4",
    "-ch-139000",
    "-ch-140000",
    "-ch-141000",
    "-ch-142000",
]
OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "experiment.evaluation.num_generated_examples": 200,
            "experiment.evaluation.evaluation_attack.search_based_attack_config.search_type": search_type,  # noqa: E501
            "experiment.environment.model_name_or_path": f"{model}{seed_suffix}",
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
    for seed_suffix in SUFFIXES
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
