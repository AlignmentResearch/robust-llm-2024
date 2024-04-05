import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "search_based_eval_imdb"  # replace dataset below

MODEL = "AlignmentResearch/robust_llm_pythia-tweet-sentiment-14m-mz-test-3classes"
N_ITS = 10
SEARCH_TYPES = ["gcg", "beam_search"]
OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.dataset_type": "hf/mteb/tweet_sentiment_extraction",
        "experiment.evaluation.num_generated_examples": 200,
        "experiment.evaluation.evaluation_attack.search_based_attack_config.search_type": search_type,  # noqa: E501
        "experiment.environment.model_name_or_path": MODEL,
        "experiment.evaluation.evaluation_attack.search_based_attack_config.n_its": N_ITS,  # noqa: E501
        "experiment.evaluation.evaluation_attack.search_based_attack_config.n_candidates_per_it": 128,  # noqa: E501
        "experiment.evaluation.batch_size": 32,
        "experiment.evaluation.evaluation_attack.search_based_attack_config.forward_pass_batch_size": 32,  # noqa: E501
    }
    for search_type in SEARCH_TYPES
]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="50G",
        cpu=12,
        priority="normal-batch",
    )
