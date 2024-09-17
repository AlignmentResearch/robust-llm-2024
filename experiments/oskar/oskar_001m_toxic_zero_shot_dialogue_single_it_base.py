import itertools
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/toxic-red-team-perez-zero-shot-dialogue"

MODELS = [
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen1.5-14B",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
]


def extract_num_params(model):
    return float(model.split("-")[1][:-1].replace("_", "."))


MODELS_AND_ADVERSARIES = [
    (model, adversary)
    for model, adversary in itertools.product(MODELS, MODELS)
    if model == adversary
]
OVERRIDE_ARGS_LIST = [
    (
        {
            "model": model,
            "+model@evaluation.evaluation_attack.adversary": adversary,
            "environment.minibatch_multiplier": 0.5,
        }
    )
    for model, adversary in MODELS_AND_ADVERSARIES
]
GPUS = [
    1 if extract_num_params(model) + extract_num_params(adversary) < 20 else 2
    for model, adversary in MODELS_AND_ADVERSARIES
]
MEMORY = ["50G" if gpu == 1 else "100G" for gpu in GPUS]
CPU = [4 if gpu == 1 else 8 for gpu in GPUS]

# Run on h100
if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        n_max_parallel=1,
        memory=MEMORY,
        cpu=CPU,
        priority="normal-batch",
        gpu=GPUS,
    )
