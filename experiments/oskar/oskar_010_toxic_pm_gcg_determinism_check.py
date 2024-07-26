import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/toxic-pm-gcg"

REPEATS = 5


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
        "+model": "Qwen/Qwen1.5-0.5B-Chat",
        "evaluation.evaluation_attack.n_its": 30,
    }
    for _ in range(REPEATS)
]
GPU = 1
MEMORY = "80G"
CPU = 8


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
