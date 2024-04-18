import os

import numpy as np

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "perplexity_defense_imdb"

DECODERS = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
]

THRESHOLDS = np.linspace(3.8, 6.0, 10).tolist()

OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.decoder_name": decoder,
        "experiment.environment.decoder_checkpoint": 143_000,
        "experiment.defense.perplexity_defense_config.perplexity_threshold": threshold,
    }
    for threshold in THRESHOLDS
    for decoder in DECODERS
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="50G",
    )
