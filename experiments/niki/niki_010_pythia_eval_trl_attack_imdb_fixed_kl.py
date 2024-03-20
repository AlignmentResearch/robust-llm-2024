import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "simple_adv_eval_imdb"

MODELS_AND_MINIBATCH_SIZES = [
    ("AlignmentResearch/robust_llm_pythia-imdb-14m-mz-v1", 16),
    ("AlignmentResearch/robust_llm_pythia-imdb-31m-mz-v1", 16),
    ("AlignmentResearch/robust_llm_pythia-imdb-70m-mz-v1", 8),
    ("AlignmentResearch/robust_llm_pythia-imdb-160m-mz-v1", 8),
    ("AlignmentResearch/robust_llm_pythia-imdb-410m-mz-v1", 4),
    ("AlignmentResearch/robust_llm_pythia-imdb-1b-mz-v1", 4),
]

BATCH_SIZE = 128

OVERRIDE_ARGS_LIST = [
    {
        # We limit ourselves to 500 examples per evaluation.
        "experiment.evaluation.num_generated_examples": 200,
        "experiment.environment.model_name_or_path": model,
        "experiment.evaluation.evaluation_attack.attack_type": "trl",
        "experiment.evaluation.evaluation_attack.append_to_modifiable_chunk": True,
        "experiment.evaluation.evaluation_attack.trl_attack_config.ppo_epochs": 10,
        "experiment.evaluation.evaluation_attack.trl_attack_config.mini_batch_size": minibatch_size,  # noqa: E501
        "experiment.evaluation.evaluation_attack.trl_attack_config.gradient_accumulation_steps": int(  # noqa: E501
            BATCH_SIZE / minibatch_size
        ),
    }
    for model, minibatch_size in MODELS_AND_MINIBATCH_SIZES
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="60G",
    )
