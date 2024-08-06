import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/toxic-pm-pair"

MODELS = [
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-7B-Chat",
]


def extract_num_params(model: str):
    model = model.lower()
    model = model.replace("-chat", "")
    model = model.replace("-instruct", "")
    model = model.replace("-hf", "")
    num_str = model.split("-")[-1]
    if num_str.endswith("b"):
        return int(float(num_str[:-1]) * 1e9)
    elif num_str.endswith("m"):
        return int(float(num_str[:-1]) * 1e6)
    else:
        raise ValueError(f"Unknown model size: {model}")


OVERRIDE_ARGS_LIST = [
    {
        "+model": model,
    }
    for model in MODELS
]
GPU = 1
MEMORY = [
    "{:.0f}G".format(40 + extract_num_params(model) // 0.25e9) for model in MODELS
]
CPU = [4 + int(extract_num_params(model) // 3e9) for model in MODELS]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory=MEMORY,
        cpu=CPU,
        gpu=GPU,
        priority="normal-batch",
    )
