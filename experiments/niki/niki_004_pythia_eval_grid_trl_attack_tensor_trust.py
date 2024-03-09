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

TRL_MODELS = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
]

OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": model,
        "experiment.evaluation.evaluation_attack.trl_attack_config.adversary_base_model_name": trl_model,  # noqa: E501
        "experiment.evaluation.evaluation_attack.attack_type": "trl",
        "experiment.evaluation.evaluation_attack.trl_attack_config.ppo_epochs": 10,  # noqa: E501
    }
    for model in MODELS
    for trl_model in TRL_MODELS
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="50G",
    )
