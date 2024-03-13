import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_adv_eval_imdb"

MODELS = [
    "AlignmentResearch/robust_llm_pythia-imdb-14m-mz-v1",
    "AlignmentResearch/robust_llm_pythia-imdb-31m-mz-v1",
    "AlignmentResearch/robust_llm_pythia-imdb-70m-mz-v1",
    "AlignmentResearch/robust_llm_pythia-imdb-160m-mz-v1",
    "AlignmentResearch/robust_llm_pythia-imdb-410m-mz-v1",
    "AlignmentResearch/robust_llm_pythia-imdb-1b-mz-v1",
]

MAX_ITERATIONS = [10_000, 100_000]

OVERRIDE_ARGS_LIST = [
    {
        # We limit ourselves to 500 examples per evaluation.
        "experiment.evaluation.num_generated_examples": 500,
        "experiment.evaluation.evaluation_attack.attack_type": "random_token",
        "experiment.environment.model_name_or_path": model,
        "experiment.evaluation.evaluation_attack.random_token_attack_config.max_iterations": max_iterations,  # noqa: E501
        "experiment.evaluation.evaluation_attack.random_token_attack_config.batch_size": 128,  # noqa: E501
        "experiment.evaluation.evaluation_attack.append_to_modifiable_chunk": True,
        "experiment.evaluation.num_examples_to_log_detailed_info": 10,
    }
    for model in MODELS
    for max_iterations in MAX_ITERATIONS
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="60G",
    )
