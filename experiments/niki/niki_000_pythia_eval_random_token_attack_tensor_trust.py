import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_adv_eval_tensor_trust"

MODELS = [
    "AlignmentResearch/robust_llm_pythia-tt-14m-mz-v0",
    "AlignmentResearch/robust_llm_pythia-tt-31m-mz-v0",
    "AlignmentResearch/robust_llm_pythia-tt-70m-mz-v0",
    "AlignmentResearch/robust_llm_pythia-tt-160m-mz-v0",
    "AlignmentResearch/robust_llm_pythia-tt-410m-mz-v0",
    "AlignmentResearch/robust_llm_pythia-tt-1b-mz-v0",
]

ATTACK_TYPES = ["random_token"]

OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": model,
        "experiment.evaluation.evaluation_attack.attack_type": attack_type,
    }
    for model in MODELS
    for attack_type in ATTACK_TYPES
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="50G",
        priority="high-batch",
    )
