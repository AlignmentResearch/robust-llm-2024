import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "niki/2024-05-29_niki_090"

OVERRIDE_ARGS_LIST = [
    {
        "model.name_or_path": "AlignmentResearch/robust_llm_pythia-tt-6.9b-mz-ada-v3",
        "evaluation.batch_size": 2,
        "evaluation.evaluation_attack.forward_pass_batch_size": 2,
    },
]


if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="100G",
        cpu=12,
        gpu=2,
        priority="high-batch",
    )
