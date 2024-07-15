import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/toxic-red-team-perez-zero-shot-dialogue"

MODELS = ["Qwen/Qwen1.5-7B-Chat", "meta-llama/Llama-2-7b-chat-hf"]


OVERRIDE_ARGS_LIST = [
    {
        "+model@evaluation.evaluation_attack.adversary": model,
        "model": "pythia-31m-chat",
    }
    for model in MODELS
]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="100G",
        cpu=8,
        priority="normal-batch",
    )
