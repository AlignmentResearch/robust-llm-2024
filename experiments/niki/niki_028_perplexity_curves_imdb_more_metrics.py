import os

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = os.path.basename(__file__).replace(".py", "")
HYDRA_CONFIG = "perplexity_defense_imdb"

DECODERS_AND_BATCH_SIZES = [
    ("EleutherAI/pythia-14m", 16),  # 0
    ("EleutherAI/pythia-31m", 16),  # 1
    ("EleutherAI/pythia-70m", 16),  # 2
    ("EleutherAI/pythia-160m", 4),  # 3
    ("EleutherAI/pythia-410m", 2),  # 4
    ("EleutherAI/pythia-1b", 2),  # 5
    ("EleutherAI/pythia-1.4b", 2),  # 6
    ("EleutherAI/pythia-2.8b", 1),  # 7
]

OVERRIDE_ARGS_LIST = [
    {
        "experiment.environment.decoder_name": decoder,
        "experiment.environment.decoder_checkpoint": 143_000,
        # Setting evalaution batch size and perplexity batch size equal, for simplicity
        "experiment.evaluation.batch_size": batch_size,
        "experiment.evaluation.num_generated_examples": 1000,
        "experiment.evaluation.evaluation_attack.random_token_attack_config.max_iterations": 10,  # noqa: E501
        "experiment.defense.num_preparation_examples": 1000,
        "experiment.defense.perplexity_defense_config.batch_size": batch_size,
        "experiment.defense.perplexity_defense_config.save_perplexity_curves": True,
    }
    for decoder, batch_size in DECODERS_AND_BATCH_SIZES
]

if __name__ == "__main__":
    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        memory="50G",
        container_tag="niki-test",
    )
