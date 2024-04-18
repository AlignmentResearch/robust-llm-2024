import os

import numpy as np

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "perplexity_defense_imdb"

MODELS_AND_BATCH_SIZES = [
    ("AlignmentResearch/robust_llm_pythia-imdb-14m-mz-ada-v3", 16),
    ("AlignmentResearch/robust_llm_pythia-imdb-31m-mz-ada-v3", 16),
    ("AlignmentResearch/robust_llm_pythia-imdb-70m-mz-ada-v3", 8),
    ("AlignmentResearch/robust_llm_pythia-imdb-160m-mz-ada-v3", 4),
    ("AlignmentResearch/robust_llm_pythia-imdb-410m-mz-ada-v3", 2),
    ("AlignmentResearch/robust_llm_pythia-imdb-1b-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-1.4b-mz-ada-v3", 1),
    ("AlignmentResearch/robust_llm_pythia-imdb-2.8b-mz-ada-v3", 1),
]

THRESHOLDS = np.linspace(3.8, 6.0, 10).tolist()

OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": model,
        "experiment.evaluation.batch_size": batch_size,
        "experiment.defense.perplexity_defense_config.perplexity_threshold": threshold,
    }
    for threshold in THRESHOLDS
    for model, batch_size in MODELS_AND_BATCH_SIZES
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="50G",
    )
