import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/toxic-pm-gcg"

MODELS_AND_GPUS = [
    ("Qwen/Qwen1.5-14B-Chat", 3),
    ("Qwen/Qwen1.5-32B-Chat", 4),
    ("Qwen/Qwen1.5-72B-Chat", 5),
]
MODELS = [model for model, _ in MODELS_AND_GPUS]


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
        "attack@evaluation.evaluation_attack": "gcg-30-its",
        "evaluation.evaluation_attack.n_its": 128,
    }
    for model in MODELS
]
GPU = [gpu for _, gpu in MODELS_AND_GPUS]
MEMORY = ["{:.0f}G".format(gpu * 160) for gpu in GPU]
CPU = [gpu * 8 for gpu in GPU]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory=MEMORY,
        cpu=CPU,
        gpu=GPU,
        priority="low-batch",
    )
