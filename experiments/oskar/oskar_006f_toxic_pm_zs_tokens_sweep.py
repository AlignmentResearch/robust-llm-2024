import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/toxic-pm-zs"

TOKENS_SWEEP = [10, 20, 50, 100, 200, 500]
MAX_NEW_TOKENS = (
    "evaluation.evaluation_attack.adversary.generation_config.max_new_tokens"
)


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
        "evaluation.evaluation_attack.n_its": 20,
        MAX_NEW_TOKENS: tokens,
    }
    for tokens in TOKENS_SWEEP
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
