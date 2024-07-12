import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/toxic-red-team-perez-zero-shot-dialogue"

MODELS = ["Qwen/Qwen1.5-4B-Chat", "Qwen/Qwen1.5-14B-Chat.yaml"]


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
            "environment.minibatch_multiplier": 0.5,
        }
    )
    for model in MODELS
]

# Run on h100
if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        n_max_parallel=1,
        memory="48G",
        cpu=4,
        priority="normal-batch",
        gpu=1,
        use_accelerate=False,
    )
