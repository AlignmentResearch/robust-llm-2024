import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "perplexity_defense_example"

MODELS = [
    "AlignmentResearch/robust_llm_pythia-word-length-14m-niki-ada-v1",
]

THRESHOLDS = [
    6.0,
    5.9,
    5.8,
    5.7,
    5.6,
    5.5,
    5.4,
    5.3,
    5.2,
    5.1,
    5.0,
    4.9,
]

OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": model,
        "experiment.defense.perplexity_defense_config.perplexity_threshold": threshold,
    }
    for model in MODELS
    for threshold in THRESHOLDS
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="50G",
    )
