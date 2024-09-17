import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/toxic-red-team-perez-zero-shot-dialogue"

MODELS = [
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
]


def chat_to_base(model):
    return (
        model.replace("-chat", "")
        .replace("-Chat", "")
        .replace("-instruct", "")
        .replace("-Instruct", "")
    )


def extract_num_params(model):
    return float(model.split("-")[1][:-1])


OVERRIDE_ARGS_LIST = [
    (
        {
            "model": model,
            "+model@evaluation.evaluation_attack.adversary": chat_to_base(model),
        }
    )
    for model in MODELS
]
GPUS = [1 if extract_num_params(model) < 2 else 2 for model in MODELS]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        n_max_parallel=1,
        memory="80G",
        cpu=8,
        priority="normal-batch",
        gpu=GPUS,
    )
