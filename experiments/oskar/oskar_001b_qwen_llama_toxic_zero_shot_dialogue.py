import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/toxic-red-team-perez-zero-shot-dialogue"

MODELS = [
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
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
GPUS = 2


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        n_max_parallel=1,
        memory="48G",
        cpu=4,
        priority="normal-batch",
        gpu=GPUS,
    )
