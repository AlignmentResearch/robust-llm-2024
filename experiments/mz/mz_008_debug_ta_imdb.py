import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_adv_eval_imdb"

MODELS = [
    "AlignmentResearch/robust_llm_pythia-imdb-14m-mz-v1",
]
ATTACK_TYPES = ["textfooler", "bae", "checklist", "pso"]
QUERY_BUDGETS = [1000, 10_000, 100_000]
OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": model,
        "experiment.evaluation.evaluation_attack.attack_type": attack_type,
        "experiment.evaluation.evaluation_attack.text_attack_attack_config.query_budget": query_budget,  # noqa: E501
        "experiment.environment.validation_set_size": 1000,
        "experiment.environment.shuffle_validation_set": True,
        "experiment.evaluation.num_examples_to_log_detailed_info": 1000,
    }
    for model in MODELS
    for attack_type in ATTACK_TYPES
    for query_budget in QUERY_BUDGETS
]

if __name__ == "__main__":
    run_multiple(EXPERIMENT_NAME, HYDRA_CONFIG, OVERRIDE_ARGS_LIST, memory="50G")
