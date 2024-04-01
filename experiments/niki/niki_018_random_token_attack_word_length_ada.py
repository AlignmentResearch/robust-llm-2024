import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "random_token_eval_word_length"

MODELS = [
    "AlignmentResearch/robust_llm_pythia-word-length-14m-niki-ada-v1",
    "AlignmentResearch/robust_llm_pythia-word-length-31m-niki-ada-v1",
    "AlignmentResearch/robust_llm_pythia-word-length-70m-niki-ada-v1",
    "AlignmentResearch/robust_llm_pythia-word-length-160m-niki-ada-v1",
    "AlignmentResearch/robust_llm_pythia-word-length-410m-niki-ada-v1",
    "AlignmentResearch/robust_llm_pythia-word-length-1b-niki-ada-v1",
    "AlignmentResearch/robust_llm_pythia-word-length-1.4b-niki-ada-v1",
    "AlignmentResearch/robust_llm_pythia-word-length-2.8b-niki-ada-v1",
]

# Set the max iterations to match the iterations of search_based
# (eg, as in niki_018_random_token_attack_word_length_ada.py)
MAX_ITERATIONS = [128, 128 * 3, 128 * 10]

OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": model,
        "experiment.evaluation.evaluation_attack.random_token_attack_config.max_iterations": max_iterations,  # noqa: E501
        "experiment.evaluation.evaluation_attack.random_token_attack_config.batch_size": 128,  # noqa: E501
    }
    for model in MODELS
    for max_iterations in MAX_ITERATIONS
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="50G",
    )
