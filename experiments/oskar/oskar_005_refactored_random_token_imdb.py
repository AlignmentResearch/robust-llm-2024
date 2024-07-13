# Compare to ian_017
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/027_eval_refactored_random_token_imdb"

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]


OVERRIDE_ARGS_LIST = [
    {
        "model": model,
        "dataset.classification_as_generation": True,
        "dataset.inference_type": "generation",
        "dataset.n_val": "10",
    }
    for model in MODELS
]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="50G",
        cpu=12,
        priority="high-batch",
    )
