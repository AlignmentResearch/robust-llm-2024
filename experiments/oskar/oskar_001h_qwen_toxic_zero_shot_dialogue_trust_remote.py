import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/toxic-red-team-perez-zero-shot-dialogue"

MODELS = [
    "Qwen/Qwen-1_8B-Chat",
    "Qwen/Qwen-7B-Chat",
    "Qwen/Qwen-14B-Chat",
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


OVERRIDE_ARGS_LIST = [
    (
        {
            "model": model,
            "+model@evaluation.evaluation_attack.adversary": chat_to_base(model),
            "environment.minibatch_multiplier": 0.5,
        }
    )
    for model in MODELS
]
GPUS = [1 if extract_num_params(model) < 10 else 2 for model in MODELS]
MEMORY = ["50G" if extract_num_params(model) < 10 else "100G" for model in MODELS]
CPU = [4 if extract_num_params(model) < 10 else 8 for model in MODELS]

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
