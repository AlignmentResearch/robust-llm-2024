import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/toxic-red-team-perez-zero-shot-dialogue"

VICTIMS = [
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
]


def chat_to_base(model):
    return (
        model.replace("-chat", "")
        .replace("-Chat", "")
        .replace("-instruct", "")
        .replace("-Instruct", "")
    )


def extract_num_params(model: str):
    model = chat_to_base(model)
    num_str = model.replace("-hf", "").split("-")[-1].lower()
    if num_str.endswith("b"):
        return int(float(num_str[:-1]) * 1e9)
    elif num_str.endswith("m"):
        return int(float(num_str[:-1]) * 1e6)
    else:
        raise ValueError(f"Unknown model size: {model}")


MODELS_AND_ADVERSARIES = [(model, "meta-llama/Llama-2-7b-chat-hf") for model in VICTIMS]
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
