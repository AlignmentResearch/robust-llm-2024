import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_train_imdb"

NUM_TRAIN_EPOCHS = [3, 10, 30]
LEARNING_RATES = [1e-5, 3e-5, 5e-5]
OVERRIDE_ARGS_LIST = [
    {
        "experiment.training.num_train_epochs": num_train_epochs,
        "experiment.training.learning_rate": learning_rate,
    }
    for num_train_epochs in NUM_TRAIN_EPOCHS
    for learning_rate in LEARNING_RATES
]

if __name__ == "__main__":
    run_multiple(EXPERIMENT_NAME, HYDRA_CONFIG, OVERRIDE_ARGS_LIST)
