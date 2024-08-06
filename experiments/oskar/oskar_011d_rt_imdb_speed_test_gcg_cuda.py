import os
from typing import Any

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "ian/027_eval_refactored_random_token_imdb"


OVERRIDE_ARGS_LIST: list[dict[str, Any]] = [
    {
        "model.name_or_path": "AlignmentResearch/robust_llm_pythia-imdb-1.4b-mz-ada-v3",
        "model.train_minibatch_size": 8,
        "model.eval_minibatch_size": 8,
    },
    {
        "model.name_or_path": "AlignmentResearch/robust_llm_pythia-imdb-1.4b-mz-ada-v3",
        "model.train_minibatch_size": 8,
        "model.eval_minibatch_size": 8,
        "environment.deterministic": True,
    },
]
GPU = 1
MEMORY = "80G"
CPU = 4


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
