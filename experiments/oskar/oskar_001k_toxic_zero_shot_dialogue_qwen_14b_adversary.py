import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "oskar/toxic-red-team-perez-zero-shot-dialogue"


OVERRIDE_ARGS_LIST = [
    (
        {
            "model": model,
            "+model@evaluation.evaluation_attack.adversary": "Qwen/Qwen1.5-14B",
            "environment.minibatch_multiplier": 0.5,
        }
    )
    for model in ("Qwen/Qwen1.5-4B-Chat", "Qwen/Qwen1.5-14B-Chat")
]

# Run on h100
if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        n_max_parallel=1,
        memory="128G",
        cpu=8,
        priority="normal-batch",
        gpu=2,
        use_accelerate=True,
    )
