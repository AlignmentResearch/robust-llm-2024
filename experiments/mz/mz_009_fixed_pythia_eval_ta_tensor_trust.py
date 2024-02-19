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
ATTACK_TYPES = ["bae", "random_character_changes"]
QUERY_BUDGETS = [300, 1000, 3000, 10_000]
NUM_SPECIAL_WORDS = [1, 3]
OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.model_name_or_path": model,
        "experiment.evaluation.evaluation_attack.attack_type": attack_type,
        "experiment.evaluation.evaluation_attack.text_attack_attack_config.query_budget": query_budget,  # noqa: E501
        "experiment.evaluation.evaluation_attack.text_attack_attack_config.num_words_to_replace_modifiable": num_special_words,  # noqa: E501
    }
    for model in MODELS
    for attack_type in ATTACK_TYPES
    for query_budget in QUERY_BUDGETS
    for num_special_words in NUM_SPECIAL_WORDS
]

if __name__ == "__main__":
    run_multiple(EXPERIMENT_NAME, HYDRA_CONFIG, OVERRIDE_ARGS_LIST, memory="50G")
