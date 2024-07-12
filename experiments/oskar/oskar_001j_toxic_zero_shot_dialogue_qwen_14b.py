import itertools
import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/toxic-red-team-perez-zero-shot-dialogue"

MODELS = [
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen1.5-14B-Chat",
]


def chat_to_base(model):
    return (
        model.replace("-chat", "")
        .replace("-Chat", "")
        .replace("-instruct", "")
        .replace("-Instruct", "")
    )


def extract_num_params(model):
    return float(model.split("-")[1][:-1].replace("_", "."))


MODELS_AND_ADVERSARIES = [
    (model, adversary)
    for model, adversary in itertools.product(MODELS, MODELS)
    if ("-14B-" in model and "-7B-" not in adversary)
    or ("-14B-" in adversary and "-7B-" not in model)
]
OVERRIDE_ARGS_LIST = [
    (
        {
            "model": model,
            "+model@evaluation.evaluation_attack.adversary": chat_to_base(adversary),
            "environment.minibatch_multiplier": 0.5,
        }
    )
    for model, adversary in MODELS_AND_ADVERSARIES
]
GPUS = [
    1 if extract_num_params(model) + extract_num_params(adversary) < 20 else 2
    for model, adversary in MODELS_AND_ADVERSARIES
]

# Run on h100
if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        n_max_parallel=1,
        memory="100G",
        cpu=8,
        priority="normal-batch",
        gpu=GPUS,
        use_accelerate=True,
    )
