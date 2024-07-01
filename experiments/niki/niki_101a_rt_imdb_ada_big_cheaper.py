import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "niki/2024-05-30_niki_098"

MODEL_NAMES = [
    # "AlignmentResearch/robust_llm_pythia-imdb-1.4b-mz-ada-v3",
    # "AlignmentResearch/robust_llm_pythia-imdb-2.8b-mz-ada-v3",
    "AlignmentResearch/robust_llm_pythia-imdb-6.9b-mz-ada-v3",
    "AlignmentResearch/robust_llm_pythia-imdb-12b-mz-ada-v3",
]

OVERRIDE_ARGS_LIST = [
    {
        "model.name_or_path": model_name,
    }
    for model_name in MODEL_NAMES
]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="200G",
        cpu=12,
        gpu=4,
        use_accelerate=True,
        priority="high-batch",
    )
