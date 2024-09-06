"""Model trained on suffix attack. Attack with prefix and with 90% through."""

from pathlib import Path

from robust_llm.batch_job_utils import run_multiple

EXPERIMENT_NAME = "tom_" + Path(__file__).stem
HYDRA_CONFIG = "tom/004_gcg_prefix"

BASE_MODELS_AND_N_MAX_PARALLEL = [
    ("pythia-14m", 4),
    ("pythia-31m", 3),
    ("pythia-70m", 2),
    ("pythia-160m", 2),
    ("pythia-410m", 1),
    ("pythia-1b", 1),
]
SEEDS = list(range(3))
ADV_TRAINING_ROUNDS = list(range(10))
PROJECT = "AlignmentResearch/robust_llm"

# See "Long from scratch" tab in
# docs.google.com/spreadsheets/d/1dMQkS4Vmuy3UFYgl67Q81BnT7oMso73ecKo7iFiSDlk
ADV_TRAINING_EXPERIMENT = "niki-052"


OVERRIDES_AND_PARALLELISM = [
    (
        {
            "model": f"EleutherAI/{base_model_name}",
            "model.name_or_path": f"{PROJECT}_{base_model_name}_{ADV_TRAINING_EXPERIMENT}_imdb_gcg_seed-{seed}",  # noqa: E501
            "model.revision": f"adv-training-round-{adv_training_round}",
        },
        max_parallel,
    )
    for base_model_name, max_parallel in BASE_MODELS_AND_N_MAX_PARALLEL
    for seed in SEEDS
    for adv_training_round in ADV_TRAINING_ROUNDS
]

OVERRIDE_ARGS_LIST = [x[0] for x in OVERRIDES_AND_PARALLELISM]
N_MAX_PARALLEL = [x[1] for x in OVERRIDES_AND_PARALLELISM]

if __name__ == "__main__":
    overrides_90pct = [
        {
            **args,
            "evaluation.evaluation_attack.perturb_position_min": 0.9,
            "evaluation.evaluation_attack.perturb_position_max": 0.9,
        }
        for args in OVERRIDE_ARGS_LIST
    ]
    run_multiple(
        EXPERIMENT_NAME.replace("004a", "004b").replace("prefix", "90pc"),
        HYDRA_CONFIG,
        overrides_90pct,
        N_MAX_PARALLEL,
        memory="50G",
        cpu=1,
        priority="normal-batch",
    )

    run_multiple(
        EXPERIMENT_NAME,
        HYDRA_CONFIG,
        OVERRIDE_ARGS_LIST,
        N_MAX_PARALLEL,
        memory="50G",
        cpu=1,
        priority="normal-batch",
    )
