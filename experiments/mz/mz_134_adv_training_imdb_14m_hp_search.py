"""Hyperparameter search for adversarial training on IMDB with 14M model.

In mz_131 we saw that for 14M model, ASR plateaus at around 10%, which is high.
See if we can improve adv training by varying hyperparameters.
"""

import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
# get just the <name>_<exp_number> part with a hyphen instead of underscore
EXP_PREFIX = "-".join(EXPERIMENT_NAME.split("_")[:2])
HYDRA_CONFIG = "20240429_mz_134"

MODEL = "pythia-14m"
DATASET = "IMDB"
NUM_GENERATED_EXAMPLES = [100, 200, 500]
LR = [3e-6, 1e-5, 3e-5]

OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL = [
    (
        {
            "experiment.training.force_name_to_save": f"{MODEL}_{EXP_PREFIX}_{DATASET}",  # noqa: E501
            "experiment.training.iterative.num_examples_to_generate_each_round": num_generated_examples,  # noqa: E501
            "experiment.training.iterative.only_add_successful_adversarial_examples": only_successful,  # noqa: E501
            "experiment.training.iterative.use_balanced_sampling": balanced,
            "experiment.training.learning_rate": lr,
        },
        2,  # n_max_parallel
    )
    for num_generated_examples in NUM_GENERATED_EXAMPLES
    for only_successful in [False, True]
    for balanced in [False, True]
    for lr in LR
]
OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]
N_MAX_PARALLEL = [x[1] for x in OVERRIDE_ARGS_LIST_AND_N_MAX_PARALLEL]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="60G",
        priority="normal-batch",
    )
