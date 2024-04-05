import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_adv_eval_imdb"

MODEL = "AlignmentResearch/robust_llm_pythia-tweet-sentiment-14m-mz-test-3classes"
BATCH_SIZE = 128
TRL_REWARD_TYPES = [
    "minus_correct_logit_plus_incorrect_logits",
    "minus_correct_logprob",
    "minus_correct_prob",
]


OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.dataset_type": "hf/mteb/tweet_sentiment_extraction",
        "experiment.evaluation.num_generated_examples": 200,
        "experiment.environment.model_name_or_path": MODEL,
        "experiment.evaluation.evaluation_attack.attack_type": "trl",
        "experiment.evaluation.evaluation_attack.append_to_modifiable_chunk": True,
        "experiment.evaluation.evaluation_attack.trl_attack_config.ppo_epochs": 2,
        "experiment.evaluation.evaluation_attack.trl_attack_config.mini_batch_size": 16,  # noqa: E501
        "experiment.evaluation.evaluation_attack.trl_attack_config.gradient_accumulation_steps": int(  # noqa: E501
            BATCH_SIZE / 16
        ),
        "experiment.evaluation.evaluation_attack.trl_attack_config.reward_type": reward_type,  # noqa: E501
    }
    for reward_type in TRL_REWARD_TYPES
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="60G",
    )
