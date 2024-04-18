import os

import numpy as np

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "perplexity_defense_imdb"

# These values were chosen by inspection and are only good
# when using pythia-14m as the decoder.
THRESHOLDS = np.linspace(3.9, 6.1, 20).tolist()

OVERRIDE_ARGS_LIST = [
    {
        "experiment.defense.perplexity_defense_config.perplexity_threshold": threshold,
    }
    for threshold in THRESHOLDS
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="50G",
    )
